import torch
import matplotlib.pyplot as plt
from models.main_model import MainModel
import numpy as np
import os
import h5py
import torch.nn.functional as f
from scipy.interpolate import interp1d
import argparse
import warnings
import json


features = []


def hook_fn(module, input, output):
    features.append(output)


# Grad-CAM实现
def grad_cam(model, x):
    gradients = []

    def save_gradient(grad):
        gradients.append(grad)

    x = x.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
    x.requires_grad = True
    output = model(torch.rand([1, 30000, 1]).cuda())
    # output = model(x.view(1, 3000, 1))

    # 反向传播
    model.zero_grad()
    target = output[0, torch.argmax(output[0])]
    target.backward()
    print("predict label: ", target)

    # 获取梯度和特征图
    gradients = model.model.embed.weight.grad.data.numpy()[0]
    print("grad shape: ", gradients.shape)
    print("features shape: ", len(features), features[-1].shape)

    weights = np.mean(gradients, axis=1)
    weights = weights[:, np.newaxis]
    print("weight shape: ",weights.shape)
    # cam = np.sum(weights[:, np.newaxis] * x.detach().numpy()[0, 0], axis=0)
    cam = np.zeros([93])
    for i in range(128):
        cam += weights[i] * features[-1][0,i].detach().numpy()
    # print(weights[0].shape, ( weights[0] * features[-1][0,0].detach().numpy()).shape, features[-1][0,0].detach().numpy().shape)
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)

    # 原始数组
    original_array = cam
    original_indices = np.linspace(0, 1, len(original_array))
    target_indices = np.linspace(0, 1, 3000)

    interp_function = interp1d(original_indices, original_array, kind='linear')
    cam = interp_function(target_indices)

    return cam


def getEEGData(h5file, filesname, num, channel = 0):
    data = np.empty(shape=[0, 3000])
    labels = np.empty(shape=[0, 1])
    temp = 0
    for filename in filesname:
        with h5py.File(h5file + filename, 'r') as fileh5:
            data = np.concatenate((data, fileh5[keys[channel]][:]), axis=0)
            labels = np.concatenate((labels, fileh5[keys[2]][:]), axis=0)
        if temp >= num:
            break
    data = (torch.from_numpy(data)).type('torch.FloatTensor')
    labels = (torch.from_numpy(labels)).type('torch.LongTensor')
    labels = labels.squeeze(dim=1)
    return data, labels


keys = ["Fpz-Cz", "Pz-Oz", "label"]
h5file = "F://models//SleepStage//SleepEdfData//SCDataSet//data/"
files = os.listdir(h5file)

x, y = getEEGData(h5file, files, num=3)
x_temp = [[],[],[],[],[]]

for i in range(x.shape[0]):
    x_temp[y[i]] += [x[i]]

a = np.stack(x_temp[0][:])
b = np.stack(x_temp[1][:])
c = np.stack(x_temp[2][:])
d = np.stack(x_temp[3][:])
e = np.stack(x_temp[4][:])

print(a.shape,b.shape,c.shape,d.shape,e.shape)


def grawgradcam(data, model, savePath):
    for i in range(data.shape[0]):
        cam = grad_cam(model, x)
        attention_weights = cam
        # attention_weights = f.softmax(torch.tensor(attention_weights)).detach().numpy()
        attention_weights[attention_weights < 0.1] = 0

        # 创建背景热力图数据
        heatmap = np.tile(attention_weights, (15, 1))

        # 生成颜色映射
        cmap = plt.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=np.min(attention_weights), vmax=np.max(attention_weights))
        colors = cmap(norm(attention_weights))

        plt.figure(figsize=(15, 4))

        # 绘制渐变颜色的热力图作为背景，并使透明度随值变化
        cmap_colors = [(0, '#ffffff'), (0.2, '#ffd7bc'), (0.4, '#ffbe90'), (0.6, '#ffa261'), (0.8, '#ff832e'),
                       (1, '#ff6800')]
        back_cmap = LinearSegmentedColormap.from_list('back_cmap', cmap_colors)

        plt.imshow(heatmap, aspect='auto', cmap=back_cmap, extent=[0, len(signal), np.min(signal), np.max(signal)],
                   alpha=1)
        plt.colorbar(label='Attention Weight')

        # 绘制信号线，并根据注意力权重修改线条颜色
        for i in range(len(signal) - 1):
            plt.plot([i, i + 1], [signal[i], signal[i + 1]], color=colors[i])

        plt.title('Signal with Attention Weights')
        plt.savefig(savePath+str(i)+'.png', bbox_inches='tight', pad_inches=0)


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
with open(args.config) as config_file:
    config = json.load(config_file)
config['name'] = os.path.basename(args.config).replace('.json', '')


for index in range(10):
    # 示例一维信号
    signal = e[index]
    x = torch.tensor(signal, dtype=torch.float32)
    # x = (x - torch.min(x))/(torch.max(x) - torch.min(x))

    signal = x.detach().numpy()
    print("signal shape:", signal.shape)

    # 初始化模型并获取Grad-CAM
    model = MainModel(config)
    hook = model.feature.model.embed.register_forward_hook(hook_fn)
    model = torch.nn.DataParallel(model, device_ids=list(range(len(args.gpu.split(",")))))
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total number of parameters: {total_params}")
    model.load_state_dict(torch.load("F:/models/SleePyCoFramework/results/FTCNN+LKCNN+Transformer_85.8_4.7/checkpoints/ckpt_fold-01.pth"))
    model.eval()
    print("loaded model")
    # model.train()

    cam = grad_cam(model, x)
    print("cam shape:", cam.shape)

    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap

    attention_weights = cam
    # attention_weights = f.softmax(torch.tensor(attention_weights)).detach().numpy()
    attention_weights[attention_weights < 0] = 0

    # 创建背景热力图数据
    heatmap = np.tile(attention_weights, (15, 1))

    # 生成颜色映射
    cmap = plt.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=np.min(attention_weights), vmax=np.max(attention_weights))
    colors = cmap(norm(attention_weights))

    plt.figure(figsize=(15, 4))

    # 绘制渐变颜色的热力图作为背景，并使透明度随值变化
    cmap_colors = [(0, '#ffffff'),(0.2, '#ffd7bc'),(0.4, '#ffbe90'),(0.6, '#ffa261'),(0.8, '#ff832e'),(1, '#ff6800')]
    back_cmap = LinearSegmentedColormap.from_list('back_cmap', cmap_colors)

    plt.imshow(heatmap, aspect='auto', cmap=back_cmap, extent=[0, len(signal), np.min(signal), np.max(signal)], alpha=1)
    plt.colorbar(label='Attention Weight')

    # 绘制信号线，并根据注意力权重修改线条颜色
    for i in range(len(signal) - 1):
        plt.plot([i, i + 1], [signal[i], signal[i + 1]], color=colors[i])

    plt.title('Signal with Attention Weights')
    plt.savefig("./results/cam/wake/"+str(index)+".png", bbox_inches='tight', pad_inches=0)
    # plt.show()

