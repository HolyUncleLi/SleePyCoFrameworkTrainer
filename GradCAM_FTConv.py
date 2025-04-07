import os
import json
import argparse
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train_mtcl import features, f1,f2,f3,f4,f5
from utils import *
from loader import EEGDataLoader
from train_mtcl import OneFoldTrainer
from models.main_model import MainModel

from scipy.interpolate import interp1d


warnings.filterwarnings("ignore")


def hook_fn(module, input, output):
    features.append(output)
def hook1(module, input, output):
    f1.append(output)
def hook2(module, input, output):
    f2.append(output)
def hook3(module, input, output):
    f3.append(output)
def hook4(module, input, output):
    f4.append(output)
def hook5(module, input, output):
    f5.append(output)


# Grad-CAM实现
def grad_cam(model, output, channelNum=8):

    # 反向传播
    model.zero_grad()
    target = output[0, torch.argmax(output[0])]
    target.backward()
    print("predict label: ", target)

    # 获取梯度和特征图
    gradients = model.feature.model.ftcnn_1.weight.grad.data.numpy()[0]
    print("grad shape: ", gradients.shape)
    print("features shape: ", len(features), features[-1].shape)

    weights = np.mean(gradients, axis=1)
    weights = weights[:, np.newaxis]
    print("weight shape: ", weights.shape)

    cam = np.zeros([output.shape[-1]])
    for i in range(channelNum):
        cam += weights[i] * features[-1][0, i].detach().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)

    # 原始数组
    original_array = cam
    original_indices = np.linspace(0, 1, len(original_array))
    target_indices = np.linspace(0, 1, 3000)

    interp_function = interp1d(original_indices, original_array, kind='linear')
    cam = interp_function(target_indices)

    return cam


class OneFoldEvaluator(OneFoldTrainer):
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold

        self.cfg = config
        self.ds_cfg = config['dataset']
        self.tp_cfg = config['training_params']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('[INFO] Config name: {}'.format(config['name']))

        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()

        self.criterion = nn.CrossEntropyLoss()
        self.ckpt_path = os.path.join('checkpoints', config['name'])
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)

    def build_model(self):
        model = MainModel(self.cfg)
        # hook = model.feature.model.embed.register_forward_hook(hook_fn)
        hook = model.feature.model.ftcnn_downsample.register_forward_hook(hook_fn)
        h1 = model.feature.model.ftcnn_1.register_forward_hook(hook1)
        h2 = model.feature.model.ftcnn_2.register_forward_hook(hook2)
        h3 = model.feature.model.ftcnn_3.register_forward_hook(hook3)
        h4 = model.feature.model.ftcnn_4.register_forward_hook(hook4)
        h5 = model.feature.model.ftcnn_5.register_forward_hook(hook5)
        print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        model.to(self.device)
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))

        return model

    def build_dataloader(self):
        test_dataset = EEGDataLoader(self.cfg, self.fold, set='test')
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=False,
                                 num_workers=4 * len(self.args.gpu.split(",")), pin_memory=True, drop_last=True)
        print('[INFO] Dataloader prepared')

        return {'test': test_loader}

    def run(self):
        print('\n[INFO] Fold: {}'.format(self.fold))
        self.model.load_state_dict(torch.load(os.path.join(self.ckpt_path, self.ckpt_name)))
        y_true, y_pred = self.evaluate(mode='test')
        print("features: ", len(features), features[0].shape)
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
                        # default='./configs/SleePyCo-Transformer_SL-10_numScales-3_SHHS_freezefinetune.json',
                        help='config file path')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.config) as config_file:
        config = json.load(config_file)
    config['name'] = os.path.basename(args.config).replace('.json', '')

    Y_true = np.zeros(0)
    Y_pred = np.zeros((0, config['classifier']['num_classes']))
    cm = []

    for fold in range(1, 2):
        evaluator = OneFoldEvaluator(args, fold, config)
        y_true, y_pred = evaluator.run()
        Y_true = np.concatenate([Y_true, y_true])
        Y_pred = np.concatenate([Y_pred, y_pred])

        summarize_result(config, fold, Y_true, Y_pred)

        cm.append(confusion_matrix(Y_true.astype(int), Y_pred.argmax(axis=1)))

    # 绘制平均混淆矩阵
    mean_cm = np.mean(cm, axis=0)
    cm_plot(mean_cm, './results/cm.svg')


if __name__ == "__main__":
    main()

