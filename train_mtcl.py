import os
import json
import argparse
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import *
from loader import EEGDataLoader
from models.main_model import MainModel
import torch.nn.functional as f

# from test import *


class OneFoldTrainer:
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold

        self.cfg = config
        self.ds_cfg = config['dataset']
        self.fp_cfg = config['feature_pyramid']
        self.tp_cfg = config['training_params']
        self.es_cfg = self.tp_cfg['early_stopping']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('[INFO] Config name: {}'.format(config['name']))

        self.train_iter = 0
        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()

        self.criterion = nn.CrossEntropyLoss()
        self.activate_train_mode()
        self.optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.tp_cfg['lr'],
                                    weight_decay=self.tp_cfg['weight_decay'])

        self.ckpt_path = os.path.join('checkpoints', config['name'])
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)
        self.early_stopping = EarlyStopping(patience=self.es_cfg['patience'], verbose=True, ckpt_path=self.ckpt_path,
                                            ckpt_name=self.ckpt_name, mode=self.es_cfg['mode'])

    def build_model(self):
        model = MainModel(self.cfg)
        print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        '''
        if self.tp_cfg['mode'] != 'scratch':
            print('[INFO] Model loaded for finetune')
            load_name = self.cfg['name'].replace('SL-{:02d}'.format(self.ds_cfg['seq_len']), 'SL-01')
            load_name = load_name.replace('numScales-{}'.format(self.fp_cfg['num_scales']), 'numScales-1')
            load_name = load_name.replace(self.tp_cfg['mode'], 'pretrain')
            load_path = os.path.join('checkpoints', load_name, 'ckpt_fold-{0:02d}.pth'.format(self.fold))
            model.load_state_dict(torch.load(load_path), strict=False)
        '''
        model.to(self.device)
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))

        return model

    def build_dataloader(self):
        train_dataset = EEGDataLoader(self.cfg, self.fold, set='train')
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=True,
                                  num_workers=4 * len(self.args.gpu.split(",")), pin_memory=True, drop_last=True)
        val_dataset = EEGDataLoader(self.cfg, self.fold, set='val')
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=False,
                                num_workers=4 * len(self.args.gpu.split(",")), pin_memory=True, drop_last=True)
        test_dataset = EEGDataLoader(self.cfg, self.fold, set='test')
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=False,
                                 num_workers=4 * len(self.args.gpu.split(",")), pin_memory=True, drop_last=True)
        print('[INFO] Dataloader prepared')

        return {'train': train_loader, 'val': val_loader, 'test': test_loader}

    def activate_train_mode(self):
        self.model.train()
        '''
        if self.tp_cfg['mode'] == 'freezefinetune':
            print('[INFO] Freeze backone')
            self.model.module.feature.train(False)
            for p in self.model.module.feature.parameters():
                p.requires_grad = False

            print('[INFO] Unfreeze conv_c5')
            self.model.module.feature.conv_c5.train(True)
            for p in self.model.module.feature.conv_c5.parameters(): p.requires_grad = True

            if self.fp_cfg['num_scales'] > 1:
                print('[INFO] Unfreeze conv_c4')
                self.model.module.feature.conv_c4.train(True)
                for p in self.model.module.feature.conv_c4.parameters(): p.requires_grad = True

            if self.fp_cfg['num_scales'] > 2:
                print('[INFO] Unfreeze conv_c3')
                self.model.module.feature.conv_c3.train(True)
                for p in self.model.module.feature.conv_c3.parameters(): p.requires_grad = True
        '''

    def train_one_epoch(self, epoch):
        correct, total, train_loss = 0, 0, 0

        for i, (inputs, labels) in enumerate(self.loader_dict['train']):
            loss = 0
            total += labels.size(0)
            inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)

            outputs = self.model(inputs)
            outputs_sum = torch.zeros_like(outputs[0])

            for j in range(len(outputs)):
                loss += self.criterion(outputs[j], labels)
                outputs_sum += outputs[j]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            predicted = torch.argmax(outputs_sum, 1)
            correct += predicted.eq(labels).sum().item()
            self.train_iter += 1

            progress_bar(i, len(self.loader_dict['train']), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (i + 1), 100. * correct / total, correct, total))

            if self.train_iter % self.tp_cfg['val_period'] == 0:
                print('')
                val_acc, val_loss = self.evaluate(mode='val')
                self.early_stopping(val_acc, val_loss, self.model)
                self.activate_train_mode()
                if self.early_stopping.early_stop:
                    break

    @torch.no_grad()
    def evaluate(self, mode):
        self.model.eval()
        correct, total, eval_loss = 0, 0, 0
        y_true = np.zeros(0)
        y_pred = np.zeros((0, self.cfg['classifier']['num_classes']))

        for i, (inputs, labels) in enumerate(self.loader_dict[mode]):
            loss = 0
            total += labels.size(0)
            inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)

            outputs = self.model(inputs)
            outputs_sum = torch.zeros_like(outputs[0])

            cnn_out = outputs[2]
            label_temp = labels

            # ==================== 最终绘图代码 (使用方法2) ====================
            with torch.no_grad():
                flattened_labels = labels.reshape(-1)
                label_map = ['W', 'N1', 'N2', 'N3', 'REM']

                for i in range(cnn_out.size(0)):
                    # --- 1. 获取数据和标签 ---
                    feature_map_to_plot = cnn_out[i]  # 形状: [channels, features]
                    label_index = flattened_labels[i].item()
                    label_str = label_map[label_index]

                    # --- 2. 后处理数据：激活值加权平均 ---
                    if feature_map_to_plot.shape[0] > 1:  # 确保有多个通道可以加权
                        # a. 计算每个通道的激活强度 (使用标准差来捕捉信号变化)
                        channel_activations = torch.std(feature_map_to_plot, dim=1)

                        # b. 使用 Softmax 将激活强度转换为总和为1的权重
                        weights = torch.softmax(channel_activations, dim=0)

                        # c. 进行加权平均
                        #    weights.unsqueeze(1) 将其形状从 [channels] 变为 [channels, 1] 以便广播
                        processed_feature = torch.sum(feature_map_to_plot * weights.unsqueeze(1), dim=0)
                    else:  # 如果只有一个通道，直接使用它
                        processed_feature = feature_map_to_plot.squeeze(0)

                    mean_feature_map_np = f.relu(processed_feature).cpu().numpy()

                    # --- 3. (可选但推荐) 对最终的一维波形进行平滑 ---
                    # 这一步不改变数据本质，只优化视觉效果
                    try:
                        from scipy.signal import savgol_filter
                        if len(mean_feature_map_np) > 5:
                            # window_length 必须是奇数, polyorder 必须小于 window_length
                            mean_feature_map_np =  savgol_filter(mean_feature_map_np, window_length=5, polyorder=2)
                    except ImportError:
                        # 如果没有安装 scipy，就跳过平滑
                        pass

                    # --- 4. 创建并设置图表样式 (与您原来的代码完全相同) ---
                    plt.figure(figsize=(12, 1.5))
                    ax = plt.gca()
                    ax.plot(mean_feature_map_np, label=label_str)
                    ax.set_ylim(-0.05,
                                max(np.max(mean_feature_map_np) * 1.05, 0.8) if np.max(
                                    mean_feature_map_np) > 0 else 0.8)
                    ax.legend(loc='upper right')

                    plt.xticks(rotation=0)  # 如需旋转刻度调整角度
                    plt.tight_layout()  # 自动调整子图参数，避免刻度被截断
                    plt.subplots_adjust(top=0.95, bottom=0.15)  # 根据需要微调上下边距

                    plt.savefig(
                        f'./figures/waveform/feature_map_sample_{self.fold}_{i}_stage_{label_str}.png',
                        bbox_inches='tight',  # 保存时收紧边界，防止截断并去掉多余空白
                        pad_inches=0.02,  # 保存时的额外内边距，按需调小或调大
                        dpi=300
                    )
                    plt.close()
            # ==================== 绘图代码结束 ====================


            for j in range(2):
                loss += self.criterion(outputs[j], labels)
                outputs_sum += outputs[j]

            eval_loss += loss.item()
            predicted = torch.argmax(outputs_sum, 1)
            correct += predicted.eq(labels).sum().item()

            y_true = np.concatenate([y_true, labels.cpu().numpy()])
            y_pred = np.concatenate([y_pred, outputs_sum.cpu().numpy()])

            progress_bar(i, len(self.loader_dict[mode]), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (eval_loss / (i + 1), 100. * correct / total, correct, total))

        self.batch += 1

        if mode == 'val':
            return 100. * correct / total, eval_loss
        elif mode == 'test':
            return y_true, y_pred
        else:
            raise NotImplementedError

    def test(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--seed', type=int, default=42, help='random seed')
        parser.add_argument('--gpu', type=str, default="0", help='gpu id')
        parser.add_argument('--config', type=str,
                            default='./configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2013_freezefinetune.json',
                            help='config file path')
        args = parser.parse_args()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        with open(args.config) as config_file:
            config = json.load(config_file)
        config['name'] = os.path.basename(args.config).replace('.json', '')

        Y_true = np.zeros(0)
        Y_pred = np.zeros((0, config['classifier']['num_classes']))

        for fold in range(1, config['dataset']['num_splits'] + 1):
            evaluator = OneFoldEvaluator(args, fold, config)
            y_true, y_pred = evaluator.run()
            Y_true = np.concatenate([Y_true, y_true])
            Y_pred = np.concatenate([Y_pred, y_pred])

            summarize_result(config, fold, Y_true, Y_pred)

    def run(self):
        for epoch in range(self.tp_cfg['max_epochs']):
            print('\n[INFO] Fold: {}, Epoch: {}'.format(self.fold, epoch))
            self.train_one_epoch(epoch)
            if self.early_stopping.early_stop:
                break

        self.model.load_state_dict(torch.load(os.path.join(self.ckpt_path, self.ckpt_name)))
        y_true, y_pred = self.evaluate(mode='test')
        print('')

        return y_true, y_pred


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument('--config', type=str,
                        default='./configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2013_freezefinetune.json',
                        help='config file path')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # For reproducibility
    set_random_seed(args.seed, use_cuda=True)

    with open(args.config) as config_file:
        config = json.load(config_file)
    config['name'] = os.path.basename(args.config).replace('.json', '')

    Y_true = np.zeros(0)
    Y_pred = np.zeros((0, config['classifier']['num_classes']))

    for fold in range(1, 3):
    # for fold in range(1, config['dataset']['num_splits'] + 1):
        trainer = OneFoldTrainer(args, fold, config)
        y_true, y_pred = trainer.run()
        Y_true = np.concatenate([Y_true, y_true])
        Y_pred = np.concatenate([Y_pred, y_pred])

        summarize_result(config, fold, Y_true, Y_pred)


if __name__ == "__main__":
    main()
