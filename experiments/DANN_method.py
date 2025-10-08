import torch, copy
import torch.nn as nn
import os
from torch import optim
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import adjust_alpha
from models.DANN import Feature_extractor, Label_classifier, Domain_Classifier
from data.dataset import my_dataset
from configs.ds_path import device
from training.callbacks import EarlyStopping
from utils import DotDict, load_model


class DANN_Trainer(object):
    def __init__(self, args):
        self.args = args
        # args.adapt_test_epoch = args.adapt_epochs // 10
        args.adapt_test_epoch = 1

        self.print_args()

        # 加载data
        self.s_train_dataset = my_dataset(ds_name_list=args.source, path_key='Stage6_org', txt_name='train.txt')
        self.s_train_loader = DataLoader(self.s_train_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)

        self.s_val_dataset = my_dataset(ds_name_list=args.source, path_key='Stage6_org', txt_name='val.txt')
        self.s_val_loader = DataLoader(self.s_val_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=True)

        self.t_train_dataset = my_dataset(ds_name_list=args.target, path_key='Stage6_org', txt_name='train.txt')
        self.t_train_loader = DataLoader(self.t_train_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)

        self.t_val_dataset = my_dataset(ds_name_list=args.target, path_key='Stage6_org', txt_name='val.txt')
        self.t_val_loader = DataLoader(self.t_val_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=True)

        # 加载模型
        self.enc = Feature_extractor().to(device)
        self.clf = Label_classifier().to(device)
        self.fd = Domain_Classifier().to(device)

        # 损失函数
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss()

        # callbacks
        self.early_stopping = EarlyStopping(self.callback_path, top_k=self.args.top_k, cur_epoch=0, patience=self.args.patience, monitored_metric=self.args.monitored_metric)

        self.best_acc = 0   # acc
        self.best_ba = 0    # balanced acc
        self.time_taken = None

        # 用于 domain classifier训练的label
        self.fake_label = torch.FloatTensor(self.args.batch_size, 1).fill_(0).to(device)
        self.real_label = torch.FloatTensor(self.args.batch_size, 1).fill_(1).to(device)

        self.best_weight_dir = None


    def print_args(self):
        '''
            Printing args to the console & Saing args to .txt file

        '''
        print('-' * 40 + ' Args ' + '-' * 40)
        self.callback_dir = 'dann' + '_' + self.args.source[0] + self.args.target[0]

        self.callback_path = os.path.join(os.getcwd(), self.callback_dir)
        if not os.path.exists(self.callback_path):
            os.mkdir(self.callback_path)
        print(f'Callback dir: {self.callback_path}')

        with open(os.path.join(self.callback_path, 'Args.txt'), 'a') as f:
            for k, v in vars(self.args).items():
                msg = f'{k}: {v}'
                print(msg)
                f.write(msg + '\n')

    def val_on_epoch_end(self, data_loader):
        self.enc.eval()
        self.clf.eval()

        y_true = []
        y_pred = []
        val_loss = 0

        with torch.no_grad():
            for batch_idx, data_dict in enumerate(tqdm(data_loader)):
                images, labels = data_dict['image'].to(device), data_dict['ped_label'].to(device)

                logits = self.clf(self.enc(images))
                preds = torch.argmax(logits, dim=1)
                loss_value = self.ce(logits, labels)
                val_loss += loss_value.item()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

                break

        val_bc = balanced_accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=range(2))

        print(f'Val loss {val_loss:.6f}, val_bc:{val_bc:.4f}')
        print(f'CM on validation set:\n{cm}')

        val_epoch_info = {
            'val_bc': val_bc,
            'loss': val_loss
        }
        return DotDict(val_epoch_info)

    def test(self):
        self.s_test_dataset = my_dataset(ds_name_list=self.args.source, path_key='Stage6_org', txt_name='test.txt')
        self.s_test_loader = DataLoader(self.s_test_dataset, batch_size=self.args.batch_size, shuffle=False)

        self.t_test_dataset = my_dataset(ds_name_list=self.args.target, path_key='Stage6_org', txt_name='test.txt')
        self.t_test_loader = DataLoader(self.t_test_dataset, batch_size=self.args.batch_size, shuffle=False)

        for item in os.listdir(self.best_weight_dir):
            if item.split('_')[1] == 'enc':
                load_model(self.enc, os.path.join(self.best_weight_dir, item))
            elif item.split('_')[1] == 'clf':
                load_model(self.clf, os.path.join(self.best_weight_dir, item))
            elif item.split('_')[1] == 'fd':
                load_model(self.fd, os.path.join(self.best_weight_dir, item))

        s_train_ba, cm = self.val_on_epoch_end(self.s_train_loader)
        print("Source Train Balanced Acc: %.2f" % (s_train_ba))
        if self.args.cm:
            print(cm)

        s_test_ba, cm = self.val_on_epoch_end(self.s_test_loader)
        print("Source Test Balanced Acc: %.2f" % (s_test_ba))
        if self.args.cm:
            print(cm)

        t_train_ba, cm = self.val_on_epoch_end(self.t_train_loader)
        print("Target Train Balanced Acc: %.2f" % (t_train_ba))
        if self.args.cm:
            print(cm)

        t_test_ba, cm = self.val_on_epoch_end(self.t_test_loader)
        print("Source Test Balanced Acc: %.2f" % (t_test_ba))
        if self.args.cm:
            print(cm)

    def dann(self):
        s_iter_per_epoch = len(self.s_train_loader)
        t_iter_per_epoch = len(self.t_train_loader)
        min_len = min(s_iter_per_epoch, t_iter_per_epoch)
        total_iters = 0

        print("Source iters per epoch: %d" % (s_iter_per_epoch))
        print("Target iters per epoch: %d" % (t_iter_per_epoch))
        print("iters per epoch: %d" % (min(s_iter_per_epoch, t_iter_per_epoch)))

        self.optimizer = optim.Adam(list(self.enc.parameters()) + list(self.clf.parameters()) + list(self.fd.parameters()), self.args.lr, betas=[0.5, 0.999], weight_decay=self.args.weight_decay)

        for EPOCH in range(self.args.adapt_epochs):
            self.clf.train()
            self.enc.train()
            self.fd.train()

            for batch_idx, (source_dict, target_dict) in enumerate(zip(self.s_train_loader, self.t_train_loader)):
                # 调节domain classifier的alpha
                total_iters += 1
                alpha = adjust_alpha(batch_idx, EPOCH, min_len, self.args.adapt_epochs)

                # 加载数据
                source, s_labels = source_dict['image'].cuda(), source_dict['ped_label'].cuda()
                target, t_labels = target_dict['image'].cuda(), target_dict['ped_label'].cuda()

                s_deep = self.enc(source)
                s_out = self.clf(s_deep)

                t_deep = self.enc(target)
                t_out = self.clf(t_deep)

                s_fd_out = self.fd(s_deep, alpha=alpha)
                t_fd_out = self.fd(t_deep, alpha=alpha)

                s_domain_err = self.bce(s_fd_out, self.real_label)
                t_domain_err = self.bce(t_fd_out, self.fake_label)
                disc_loss = s_domain_err + t_domain_err

                s_clf_loss = self.ce(s_out, s_labels)

                loss = s_clf_loss + disc_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch_idx % 50 == 0 or batch_idx == (min_len - 1):
                    print('Ep: %d/%d, iter: %d/%d, total_iters: %d, s_err: %.4f, d_err: %.4f, alpha: %.4f'
                          % (EPOCH + 1, self.args.adapt_epochs, batch_idx + 1, min_len, total_iters, s_clf_loss, disc_loss, alpha))
                break

            if (EPOCH + 1) <= self.args.min_train_epoch:
                if (EPOCH + 1) % self.args.adapt_test_epoch == 0:
                    val_epoch_info = self.val_on_epoch_end(self.t_val_loader)
            else:
                val_epoch_info = self.val_on_epoch_end(self.t_val_loader)
                self.early_stopping(EPOCH+1, enc=self.enc, clf=self.clf, fd=self.fd, val_epoch_info=val_epoch_info)
                self.best_weight_dir = self.early_stopping.best_weight_dir
            if self.early_stopping.early_stop:
                print(f'Early Stopping!')





































