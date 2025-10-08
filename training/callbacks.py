import os, torch, sys, math
import logging
import numpy as np
import torch.optim as optim



class EarlyStopping():
    def __init__(self, callback_save_path,
                 top_k=2,
                 cur_epoch=0,
                 best_monitor_metric=-np.inf,
                 patience=10,
                 delta=0.00001):
        '''
        callback_save_path: 保存模型的文件夹
        :param top_k: 保存几个最好模型
        :param patience: 当监控的 metric 连续 patience 个 epoch 不增加，则触发early stopping
        :param delta: 监控metric增加的最小值，当超过该值的时候表示模型有进步
        '''

        self.model_save_dir = callback_save_path
        self.top_k = top_k

        self.save_prefix = callback_save_path.split(os.sep)[-1]
        self.cur_epoch = cur_epoch
        self.best_monitor_metric = best_monitor_metric
        if self.best_monitor_metric < 0:
            self.monitored_metric = 'acc'
        else:
            self.monitored_metric = 'loss'

        self.patience = patience
        self.counter = 0            # 记录loss不变的epoch数目
        self.early_stop = False     # 是否停止训练
        self.delta = delta

        print('-' * 20 + 'Early Stopping Info' + '-' * 20)
        print(f'Create early stopping, monitoring [validation {self.monitored_metric}] changes')
        print(f'The best {self.top_k} models will be saved to {self.model_save_dir}')
        print(f'File saving format: {self.save_prefix}_epoch_{self.monitored_metric}.pth')
        print(f'Early Stop with patience: {self.patience}')

        msg = f'The best {self.top_k} models will be saved to {self.model_save_dir}\n'
        with open(os.path.join(self.model_save_dir, 'cb_EarlyStop.txt'), 'a') as f:
            f.write(msg)

    def __call__(self, epoch, model, optimizer, val_epoch_info, scheduler=None):

        self.cur_epoch = epoch
        cur_lr = optimizer.param_groups[0]['lr']
        print(f'Current lr: {cur_lr}')

        # 监控指标为准确率的情况
        if self.monitored_metric == 'balanced_accuracy':
            if val_epoch_info.balanced_accuracy < self.best_monitor_metric + self.delta:       # 表现没有提升的情况
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            else:       # 表现提升
                metrics = [self.best_monitor_metric, val_epoch_info.balanced_accuracy]
                self.save_checkpoint(model=model, metrics=metrics, optimizer=optimizer, ckpt_dir=self.model_save_dir, scheduler=scheduler)
                self.counter = 0

        # 监控指标为loss的情况
        elif self.monitored_metric == 'loss':
            if val_epoch_info.loss > self.best_monitor_metric + self.delta:    # 表现没有提升的情况
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            else:
                metrics = [self.best_monitor_metric, val_epoch_info.loss]
                self.save_checkpoint(model=model, metrics=metrics, optimizer=optimizer, ckpt_dir=self.model_save_dir, scheduler=scheduler)
                self.counter = 0
        else:
            raise ValueError('Wrong monitored metrics!')

        # 根据counter判断是否设置停止flag
        if self.counter >= self.patience:
            self.early_stop = True

        # 记录earlystop信息
        msg = f"Epoch:{epoch}, overall counter:{self.counter}/{self.patience}, current lr: {cur_lr}\n"
        with open(os.path.join(self.model_save_dir, 'cb_EarlyStop.txt'), 'a') as f:
            f.write(msg)


    def del_redundant_weights(self, ckpt_dir):
        all_weights_temp = os.listdir(ckpt_dir)
        all_weights = []
        for weights in all_weights_temp:
            if weights.endswith('.pth'):
                all_weights.append(weights)

        # 按存储格式来： save_name = prefix_{epoch}_{balanced_acc/loss}.pth
        if len(all_weights) > self.top_k - 1:
            sorted = []
            for weight in all_weights:
                val_acc = weight.split('-')[-1]
                sorted.append((weight, val_acc))

            if self.monitored_metric == 'balanced_accuracy':
                sorted.sort(key=lambda w: w[1], reverse=False)
            else:
                sorted.sort(key=lambda w: w[1], reverse=True)

            print('After sorting:', sorted)

            del_path = os.path.join(self.model_save_dir, sorted[0][0])
            os.remove(del_path)
            print('Del file:', del_path)


    def save_checkpoint(self, model, metrics, optimizer, ckpt_dir, scheduler=None):
        print(f'Performance [{self.monitored_metric}] better ({metrics[0]} --> {metrics[1]}). Saving Model.')

        self.del_redundant_weights(ckpt_dir)
        save_name = f"{self.save_prefix}-{self.cur_epoch:02d}-{metrics[1]:.5f}.pth"     # 格式：prefix_{epoch}_{balanced_acc}.pth
        self.best_monitor_metric = metrics[1]

        checkpoint = {
            'epoch': self.cur_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_bc': self.best_monitor_metric,
            'lr': scheduler.get_last_lr() if scheduler is not None else 0,
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else '',
        }

        save_path = os.path.join(ckpt_dir, save_name)
        torch.save(checkpoint, save_path)
