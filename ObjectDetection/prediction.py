import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
# if torch.cuda.is_available():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ssd import build_ssd

from matplotlib import pyplot as plt

from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import copy

import glob

if __name__ == '__main__':

    CLASSES = ('eyebolt',)
    root = r'Z:\object_detection\2019_03_05_180242'
    args = {'dataset': 'VOC',
            'basenet': 'vgg16_reducedfc.pth',
            'batch_size': 12,
            'resume': '',
            'start_iter': 0,
            'num_workers': 1,
            'cuda': True,
            'lr': 5e-4,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'gamma': 0.1,
            'save_folder': 'weights/'
            }

    cfg = voc

    # (4) モデル定義
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssd_net = build_ssd(device, 'test', cfg['min_dim'], cfg['num_classes'])
    # パラメータのロード
    ssd_net.load_weights('./weights/VOC.pth')
    ssd_net.to(device)

    # 訓練データの読み込み
    testset = DataSetObjectDetection(root=os.path.join(root,'data'),
                                     transform=SSDAugmentation(cfg['min_dim'],
                                                               MEANS), target_transform=AnnotationTransform(
            class_to_index=dict(zip(CLASSES, range(len(CLASSES)))))
                                     )



    # for img_id in range(len(testset)):
    #     image = testset.pull_image(img_id)
    #     # cv2のチャンネルの順番はBGR（青、緑、赤）なので、RGB（赤、緑、青）に入れ替える
    #     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     # View the sampled input image before transform
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(rgb_image)
    #     plt.show()
    #
    #     # 画像のサイズを300×300に変更
    #     x = cv2.resize(image, (300, 300)).astype(np.float32)
    #     x -= (104.0, 117.0, 123.0)
    #     x = x.astype(np.float32)
    #     x = x[:, :, ::-1].copy()
    #     # plt.imshow(x)
    #     # HWCの形状[300, 300, 3]をCHWの形状[3, 300,300]に変更
    #     x = torch.from_numpy(x).permute(2, 0, 1)
    #     xx = x.unsqueeze(0).to(device)
    #
    #     # 順伝播を実行し、推論結果を出力
    #     y = ssd_net(xx)
    #
    #     top_k = 10
    #
    #     plt.figure(figsize=(10, 10))
    #     colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    #     plt.imshow(rgb_image)  # plot the image for matplotlib
    #     currentAxis = plt.gca()
    #     # 推論結果をdetectionsに格納
    #     detections = y.data
    #     # scale each detection back up to the image
    #     scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    #     # クラスiでループ
    #     for i in range(detections.size(1)):
    #         j = 0
    #         # 確信度confが0.6以上のボックスを表示
    #         # jは確信度上位200件のボックスのインデックス
    #         # detections[0,i,j]は[conf,xmin,ymin,xmax,ymax]の形状
    #         while detections[0, i, j, 0] >= 0.6:
    #             score = detections[0, i, j, 0]
    #             label_name = CLASSES[i - 1]
    #             display_txt = '%s: %.2f' % (label_name, score)
    #             pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
    #             coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
    #             color = colors[i]
    #             currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
    #             currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
    #             j += 1
    #
    #     plt.show()

    # クラスVOCDetectionはindexをキーに画像を取得
    img_pathlist = glob.glob(os.path.join(root,'data\color-images', '*.png'))
    for img_path in img_pathlist:
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_image)
        # plt.show()

        # 画像のサイズを300×300に変更
        x = cv2.resize(image, (300, 300)).astype(np.float32)
        x -= (104.0, 117.0, 123.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        # plt.imshow(x)
        # HWCの形状[300, 300, 3]をCHWの形状[3, 300,300]に変更
        x = torch.from_numpy(x).permute(2, 0, 1)
        xx = x.unsqueeze(0).to(device)

        # 順伝播を実行し、推論結果を出力
        y = ssd_net(xx)

        top_k = 10

        plt.figure(figsize=(10, 10))
        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        plt.imshow(rgb_image)  # plot the image for matplotlib
        currentAxis = plt.gca()
        # 推論結果をdetectionsに格納
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
        # クラスiでループ
        for i in range(detections.size(1)):
            j = 0
            # 確信度confが0.6以上のボックスを表示
            # jは確信度上位200件のボックスのインデックス
            # detections[0,i,j]は[conf,xmin,ymin,xmax,ymax]の形状
            while detections[0, i, j, 0] >= 0.3:
                score = detections[0, i, j, 0]
                label_name = CLASSES[i - 1]
                display_txt = '%s: %.2f' % (label_name, score)
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
                color = colors[i]
                currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
                j += 1

        plt.savefig(os.path.join(root,'info', os.path.basename(img_path)))
        # plt.show()

