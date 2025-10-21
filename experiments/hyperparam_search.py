'''
    To find the best combination of hyperparameters
'''
import os, re
from torch.utils.data import random_split, DataLoader
from torch import nn
import itertools, torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
from training.callbacks import EarlyStopping

from models.DANN import Feature_extractor, Label_classifier, Domain_Classifier
from data.dataset import my_dataset
from utils import DEVICE, load_model, adjust_alpha


class HPSelection():
    def __init__(self, opts):
        super().__init__()

        hp_dict = {
            'batch_size': [48, 64],
            'base_lr': [1e-3, 5e-4],
            'optimizer': ['Adam', 'SGD'],
            'scheduler': ['COS', 'EXP']
        }
        self.all_combinations = list(itertools.product(
            hp_dict['batch_size'],
            hp_dict['base_lr'],
            hp_dict['optimizer'],
            hp_dict['scheduler']
        ))

        self.opts = opts

        # 固定的hyperparamters
        self.source = ['D1']
        self.target = ['D2']

        self.warmup_epochs = 3
        self.min_epochs = 10
        self.max_epochs = 50
        self.mini_train_num = 1000
        self.mini_val_num = 1000
        self.mini_test_num = 1000
        self.patience = 5

        # self.min_epochs = 2
        # self.max_epochs = 5
        # self.mini_train_num = 10
        # self.mini_val_num = 10
        # self.mini_test_num = 10

        self.best_weight_dir = None

        # 加载DANN模型
        self.enc = Feature_extractor().to(DEVICE)
        self.clf = Label_classifier().to(DEVICE)
        self.fd = Domain_Classifier().to(DEVICE)

        # 损失函数
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss()

        if self.opts.isTrain:
            self.train_setup()
        print('共有的超参数组合数：', len(self.all_combinations))

    def train_setup(self):
        # source train
        s_trainset = my_dataset(ds_name_list=self.source, path_key='Stage6_org', txt_name='train.txt')
        self.s_mini_trainset, _ = random_split(s_trainset, [self.mini_train_num, len(s_trainset) - self.mini_train_num])

        # source val
        s_valset = my_dataset(ds_name_list=self.source, path_key='Stage6_org', txt_name='val.txt')
        self.s_mini_valset, _ = random_split(s_valset, [self.mini_val_num, len(s_valset) - self.mini_val_num])

        # target train
        t_trainset = my_dataset(ds_name_list=self.target, path_key='Stage6_org', txt_name='val.txt')
        self.t_mini_trainset, _ = random_split(t_trainset, [self.mini_train_num, len(t_trainset) - self.mini_train_num])

        # target val
        t_valset = my_dataset(ds_name_list=self.target, path_key='Stage6_org', txt_name='train.txt')
        self.t_mini_valset, _ = random_split(t_valset, [self.mini_val_num, len(t_valset) - self.mini_val_num])

        print(f'Mini source train num: {len(self.s_mini_trainset)}\nMini source val num:{len(self.s_mini_valset)}')
        print(f'Mini target train num: {len(self.t_mini_trainset)}\nMini target val num:{len(self.t_mini_valset)}')

        self.total_iters = 0
        self.min_len = 0

        self.txt_dir = os.path.join(self.opts.hp_dir, 'hp_txt')
        os.makedirs(self.txt_dir, exist_ok=True)

    def val_on_epoch_end(self, data_loader, epoch=None):

        self.enc.eval()
        self.clf.eval()

        val_info = {
            'loss': 0.0
        }
        y_true = []
        y_pred = []

        epoch_msg = str(epoch) if epoch is not None else ''

        with torch.no_grad():
            for batch_idx, data_dict in enumerate(tqdm(data_loader, desc=f'Epoch {epoch_msg} val')):
                images, labels = data_dict['image'].to(DEVICE), data_dict['ped_label'].to(DEVICE)

                logits = self.clf(self.enc(images))
                preds = torch.argmax(logits, dim=1)
                loss_value = self.ce(logits, labels)
                val_info['loss'] += loss_value.item()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

            val_bc = balanced_accuracy_score(y_true, y_pred)
            val_info['balanced_accuracy'] = val_bc

            val_cm = confusion_matrix(y_true, y_pred)
            val_info['cm'] = val_cm

        return val_info

    def decomp_cm(self, cm):
        tn, fp, fn, tp = cm.ravel()
        return f'{tn}, {fp}, {fn}, {tp}'

    def test(self):
        '''
            最终在test set上进行检验
        '''

        # source test set
        s_testset = my_dataset(ds_name_list=self.source, path_key='Stage6_org', txt_name='test.txt')
        self.s_mini_testset, _ = random_split(s_testset, [self.mini_test_num, len(s_testset) - self.mini_test_num])
        self.s_mini_testloader = DataLoader(self.s_mini_testset, batch_size=128, shuffle=False)

        # target test set
        t_testset = my_dataset(ds_name_list=self.target, path_key='Stage6_org', txt_name='test.txt')
        self.t_mini_testset, _ = random_split(t_testset, [self.mini_test_num, len(t_testset) - self.mini_test_num])
        self.t_mini_testloader = DataLoader(self.t_mini_testset, batch_size=128, shuffle=False)

        for item in os.listdir(self.opts.weight_dir):
            weight_path = os.path.join(self.opts.weight_dir, item)
            print(f'weight_path: {weight_path}')
            state_dict = torch.load(weight_path)
            if item.split('_')[1] == 'enc':
                self.enc.load_state_dict(state_dict)
            elif item.split('_')[1] == 'clf':
                self.clf.load_state_dict(state_dict)
            elif item.split('_')[1] == 'fd':
                self.fd.load_state_dict(state_dict)

        print(f'Best Model on source test set:')
        s_test_info = self.val_on_epoch_end(data_loader=self.s_mini_testloader)

        print(f'Best Model on target test set:')
        t_test_info = self.val_on_epoch_end(data_loader=self.t_mini_testloader)

        with open(self.opts.test_txt, 'a') as f:
            msg = f'model_weights: {self.opts.weight_dir}\nsource: {self.source}, target: {self.target}\nSource test loss: {s_test_info["loss"]:.4f}, target test loss: {t_test_info["loss"]:.4f}\nSource test ba: {s_test_info["balanced_accuracy"]:.4f}, target test ba: {t_test_info["balanced_accuracy"]:.4f}\nSource tn, fp, fn, tp: {self.decomp_cm(s_test_info["cm"])}\nTarget tn, fp, fn, tp: {self.decomp_cm(t_test_info["cm"])}'
            f.write(msg)

    def train_one_epoch(self, epoch):
        train_info = {
            'loss': 0.0,
        }

        self.clf.train()
        self.enc.train()
        self.fd.train()

        y_true = []
        y_pred = []

        for batch_idx, (source_dict, target_dict) in tqdm(
                enumerate(zip(self.s_mini_trainloader, self.t_mini_trainloader)),
                total=len(self.s_mini_trainloader), desc=f'Epoch {epoch} train'
                ):
            # 调节domain classifier的alpha
            self.total_iters += 1
            alpha = adjust_alpha(batch_idx, epoch, self.min_len, self.max_epochs)

            # 加载数据
            source, s_labels = source_dict['image'].to(DEVICE), source_dict['ped_label'].to(DEVICE)
            target, _ = target_dict['image'].to(DEVICE), target_dict['ped_label'].to(DEVICE)

            # 训练开始
            s_deep = self.enc(source)
            s_out = self.clf(s_deep)
            s_pred = torch.argmax(s_out, 1)

            t_deep = self.enc(target)
            t_out = self.clf(t_deep)

            s_fd_out = self.fd(s_deep, alpha=alpha)
            t_fd_out = self.fd(t_deep, alpha=alpha)

            s_domain_err = self.bce(s_fd_out, self.real_label)
            t_domain_err = self.bce(t_fd_out, self.fake_label)
            disc_loss = s_domain_err + t_domain_err

            s_clf_loss = self.ce(s_out, s_labels)

            loss_value = s_clf_loss + disc_loss

            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()

            train_info['loss'] += loss_value.item()
            y_true.extend(s_labels.cpu().numpy())
            y_pred.extend(s_pred.cpu().numpy())

        balanced_accuracy = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
        train_info.update({'balanced_accuracy': balanced_accuracy})

        return train_info

    def hp_search(self):
        for comb_idx, comb_info in enumerate(self.all_combinations):
            self.batch_size, self.base_lr, optimizer_type, scheduler_type = comb_info

            # data loader
            self.s_mini_trainloader = DataLoader(self.s_mini_trainset, batch_size=self.batch_size, shuffle=True,
                                                 drop_last=True)
            self.s_mini_valloader = DataLoader(self.s_mini_valset, batch_size=128, shuffle=False, drop_last=True)
            self.t_mini_trainloader = DataLoader(self.t_mini_trainset, batch_size=self.batch_size, shuffle=True,
                                                 drop_last=True)
            self.t_mini_valloader = DataLoader(self.t_mini_valset, batch_size=128, shuffle=False, drop_last=True)

            # 用于 domain classifier训练的label
            self.fake_label = torch.FloatTensor(self.batch_size, 1).fill_(0).to(DEVICE)
            self.real_label = torch.FloatTensor(self.batch_size, 1).fill_(1).to(DEVICE)

            # get combination name
            comb = [str(self.batch_size), str(self.base_lr), optimizer_type, scheduler_type]
            comb_name = '_'.join(comb)
            cur_txt_path = os.path.join(self.txt_dir, str(comb_idx + 1) + '.txt')
            with open(cur_txt_path, 'a') as f:
                f.write('Combination: ' + comb_name + '\n')

            print('=' * 30 + f' HP Comb: {comb_name} ({comb_idx + 1}/{len(self.all_combinations)})' + '=' * 30)

            # callbacks
            callback_save_path = os.path.join(self.opts.hp_dir, comb_name)
            if not os.path.exists(callback_save_path):
                os.makedirs(callback_save_path)
            self.early_stopping = EarlyStopping(callback_path=callback_save_path, patience=5)
            print(f'comb_name:{comb_name}, cur_txt_path:{cur_txt_path}, callback_save_dir:{callback_save_path}')

            if optimizer_type == 'Adam':
                self.optimizer = Adam(
                    params=list(self.enc.parameters()) + list(self.clf.parameters()) + list(self.fd.parameters()),
                    lr=self.base_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
            elif optimizer_type == 'SGD':
                self.optimizer = SGD(
                    list(self.enc.parameters()) + list(self.clf.parameters()) + list(self.fd.parameters()),
                    lr=self.base_lr, momentum=0.9, weight_decay=0.0001)

            if scheduler_type == 'COS':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                            T_max=self.max_epochs - self.warmup_epochs)
            elif scheduler_type == 'EXP':
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)

            s_iter_per_epoch = len(self.s_mini_trainloader)
            t_iter_per_epoch = len(self.t_mini_trainloader)
            self.min_len = min(s_iter_per_epoch, t_iter_per_epoch)
            self.total_iters = 0

            for EPOCH in range(self.max_epochs):
                train_info = self.train_one_epoch(EPOCH + 1)
                val_info = self.val_on_epoch_end(self.t_mini_valloader, epoch=(EPOCH + 1))
                self.early_stopping(EPOCH + 1, enc=self.enc, clf=self.clf, fd=self.fd, val_epoch_info=val_info)

                # lr schedule
                if EPOCH <= self.warmup_epochs:
                    self.optimizer.param_groups[0]['lr'] = self.base_lr * EPOCH / self.warmup_epochs
                else:
                    self.scheduler.step()

                self.write_to_txt(EPOCH, txt_path=cur_txt_path, train_info=train_info, val_info=val_info)

                # 在低于min train epoch时，每次重置early stop的参数
                if (EPOCH + 1) <= self.min_epochs:
                    self.early_stopping.counter = 0
                    self.early_stopping.early_stop = False
                else:  # 当训练次数超过最低epoch时，其中early_stop策略
                    self.best_weight_dir = self.early_stopping.best_weight_dir
                    if self.early_stopping.early_stop:
                        print(f'Early Stopping!')
                        break

    def write_to_txt(self, epoch, txt_path, train_info, val_info):
        train_msg = 'Train: ' + self.get_print_msg(info_dict=train_info)
        val_msg = 'Val: ' + self.get_print_msg(info_dict=val_info)
        with open(txt_path, 'a') as f:
            f.write(f'------------------------------ Epoch: {epoch} ------------------------------\n')
            f.write(train_msg)
            f.write(val_msg)

    @staticmethod
    def _get_msg_format(key):
        '''
            tool_function，对 accuracy 和 loss 进行输出时设置不同的打印位数
        '''

        if 'loss' in key:
            return '{:.6f}'
        if 'accuracy' in key or 'bc' in key:
            return '{:.4f}'
        else:
            return '{}'

    def get_print_msg(self, info_dict):
        msg = ', '.join([f"{k}: {self._get_msg_format(k).format(v)}" for k, v in info_dict.items()]) + '\n'
        return msg
