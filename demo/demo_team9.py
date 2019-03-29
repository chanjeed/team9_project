# !/usr/bin/env python
# -*- coding: utf-8 -*-
##
#   @package demo_team9.py
#   @brief  デモプログラム
#
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from gui.demo import Ui_MainWindow
from PyQt5.QtWidgets import (QWidget, QFileDialog)
import os
import json
import torch
import glob
import cv2
from ssd import build_ssd
import numpy as np
import matplotlib.pyplot as plt
from data import voc
import pandas as pd
import queue
import random
import configparser

# x [xmin,ymin,xmax,ymax]
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] + t_size[1] + 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] + 13), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def rotate(img, angle, scale):
    """
    画像を回転（反転）させる
    [in]  img:   回転させる画像
    [in]  angle: 回転させる角度
    [in]  scale: 拡大率
    [out] 回転させた画像
    """
    #
    # size = img.shape[:2]
    # mat = cv2.getRotationMatrix2D((size[0] // 2, size[1] // 2), angle, scale)
    # return cv2.warpAffine(img, mat, size, flags=cv2.INTER_CUBIC)

    h, w = img.shape[:2]
    size = (w, h)

    angle_rad = angle / 180.0 * np.pi

    # 回転後の画像サイズを計算
    w_rot = int(np.round(h * np.absolute(np.sin(angle_rad)) + w * np.absolute(np.cos(angle_rad))))
    h_rot = int(np.round(h * np.absolute(np.cos(angle_rad)) + w * np.absolute(np.sin(angle_rad))))
    size_rot = (w_rot, h_rot)

    # 元画像の中心を軸に回転する
    center = (w / 2, h / 2)
    scale = 1.0
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # 平行移動を加える (rotation + translation)
    affine_matrix = rotation_matrix.copy()
    affine_matrix[0][2] = affine_matrix[0][2] - w / 2 + w_rot / 2
    affine_matrix[1][2] = affine_matrix[1][2] - h / 2 + h_rot / 2

    img_rot = cv2.warpAffine(img, affine_matrix, size_rot, flags=cv2.INTER_CUBIC)

    return img_rot


def rotateR(img, level=[-10, 10], scale=1.2):
    """
    ランダムに画像を回転させる
    [in]  img:   回転させる画像
    [in]  level: 回転させる角度の範囲
    [out] 回転させた画像
    [out] 回転させた角度
    """

    angle = np.random.randint(level[0], level[1])
    return rotate(img, angle, scale), angle


class OwnImageWidget(QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()


class Application(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        # ---------------- config -----------------------#
        config = configparser.ConfigParser()
        config.read('./config/setting.ini')
        self.mode = config.get('demo','mode')
        self.file_path = dict()
        self.file_path['data'] = config.get('demo','data')
        self.file_path['test_data'] = os.path.join(self.file_path['data'],'test')
        self.file_path['img'] = os.path.join(self.file_path['test_data'], 'data\color-images')
        self.file_path['ssd'] = './model/ssd/ssd.pth'
        with open(os.path.join(r'./class_to_index_en.json'), 'r',
                  encoding='utf-8_sig') as f:
            self.class_to_index = json.load(f)
        with open(os.path.join(r'./class_labels_en.json'), 'r',
                  encoding='utf-8_sig') as f:
            self.class_labels = json.load(f)
        with open(os.path.join(r'./class_labels_en_narrow.json'), 'r',
                  encoding='utf-8_sig') as f:
            self.class_labels_narrow = json.load(f)
        cfg = voc
        dtype = {'id': 'object', 'x01': 'str', 'x02': 'str'}
        self.df = pd.read_csv(os.path.join(self.file_path['data'], 'recipes_.csv'), dtype=dtype)
        ingredients_vec_query = list()
        for path_vec in self.df['vec'].values.tolist():
            ingredients_vec_query.append(np.load(path_vec))
        self.ingredients_vec_query = np.array(ingredients_vec_query)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.class_labels))]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ssd_net = build_ssd(self.device, 'test', cfg['min_dim'], cfg['num_classes'])
        self.ssd_net.load_weights(self.file_path['ssd'])
        self.ssd_net.to(self.device)
        self.img = None
        self.id = 0
        # -----------------------------------------------#
        self.queue = queue.Queue()
        self.img_pathlist = glob.glob(os.path.join(self.file_path['img'], '*.png'))
        # -----------------------------------------------#
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton_load.setEnabled(True)
        self.ui.pushButton_search.setEnabled(False)

        self.ui.textEdit_title.setText("")
        self.ui.textEdit_ingredients.setText("")
        self.ui.textEdit_processes.setText("")
        self.window_width = self.frameSize().width()
        self.window_height = self.frameSize().height()
        self.img_widget = OwnImageWidget(self.ui.img_widget)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

    # 検索
    def click_search(self):
        self.ui.pushButton_load.setEnabled(False)
        self.ui.pushButton_search.setEnabled(False)

        # --- main process --- #
        ingredients_vec_src = np.zeros(len(self.class_labels))
        image = np.copy(self.img)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = cv2.resize(image, (300, 300)).astype(np.float32)
        x -= (104.0, 117.0, 123.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)
        xx = x.unsqueeze(0).to(self.device)
        y = self.ssd_net(xx)
        detections = y.data
        scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.3:
                score = detections[0, i, j, 0]
                label_name = self.class_labels[i - 1]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                color = self.colors[i]
                plot_one_box(pt, rgb_image, color, label_name)
                ingredients_vec_src[i - 1] = 1
                j += 1
        # キューへ格納
        if self.queue.qsize() < 10:
            frame = {}
            frame["img"] = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            self.queue.put(frame)
        else:
            print(self.queue.qsize())

        # -- レシピ検索 --#
        ingredients_vec_src = np.expand_dims(ingredients_vec_src, axis=0)
        result = self.ingredients_vec_query * ingredients_vec_src
        result = np.sum(result, axis=1)
        indices_recommend = np.argsort(result)[::-1]

        self.ui.textEdit_title.setText("")
        for index_recommend in indices_recommend[:1]:
            path = self.df.iloc[index_recommend, 0]
            with open(path, 'r', encoding='utf-8_sig') as f:
                recipe = json.load(f)
            self.ui.textEdit_title.append(recipe['title'])

        index_recommend = indices_recommend[0]
        path = self.df.iloc[index_recommend, 0]
        with open(path, 'r', encoding='utf-8_sig') as f:
            recipe = json.load(f)

        # -- 作り方 -- #
        self.ui.textEdit_processes.setText("")
        for index, process in enumerate(recipe['processes']):
            self.ui.textEdit_processes.append('({}) '.format(index) + process)

        # -- 食材 -- #
        self.ui.textEdit_ingredients.setText("")
        for index, ingredient in enumerate(recipe['ingredients']):
            self.ui.textEdit_ingredients.append('{}:{}'.format(ingredient['name'], ingredient['amount']))

        # -------------- #
        self.ui.pushButton_load.setEnabled(True)
        self.ui.pushButton_search.setEnabled(False)

    # 画像読込
    def click_load(self):
        self.ui.pushButton_load.setEnabled(False)
        self.ui.pushButton_search.setEnabled(False)
        self.ui.textEdit_title.setText("")
        self.ui.textEdit_ingredients.setText("")
        self.ui.textEdit_processes.setText("")

        if self.mode == 'test':
            # 画像読込
            img = cv2.imread(self.img_pathlist[self.id % len(self.img_pathlist)], cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.id += 1
            if self.id > len(self.img_pathlist):
                self.id = 0
            # キューへ格納
            if self.queue.qsize() < 10:
                frame = {}
                frame["img"] = img
                self.queue.put(frame)
            else:
                print(self.queue.qsize())
            self.img = np.copy(img)
        else:
            # 画像生成
            class_labels_selected = random.choices(self.class_labels_narrow, k=random.randint(4, 6))
            bg = 255 * np.ones((480, 640, 4), np.uint8)
            check = np.zeros([10, 10])
            img = bg.copy()
            for class_label_selected in class_labels_selected:
                fg = cv2.imread(os.path.join(self.file_path['data'], 'images', class_label_selected + '.png'), -1)
                scale = random.uniform(0.8, 1.0)
                fg = cv2.resize(fg, (int(scale * fg.shape[1]), int(scale * fg.shape[0])))
                img2, _ = rotateR(fg, [-90, 90], 1.0)
                # I want to put logo on top-left corner, So I create a ROI
                w1, h1 = img.shape[:2]
                w2, h2 = img2.shape[:2]
                while 1:
                    x = np.random.randint(0, w1 - w2 + 1)
                    y = np.random.randint(0, h1 - h2 + 1)
                    cx = x // (160) - 1
                    cy = y // (160) - 1
                    if check[cy, cx] == 0:
                        print('break')
                        check[cy, cx] = 1
                        break

                roi = img[x:x + w2, y:y + h2]
                mask = img2[:, :, 3]
                ret, mask_inv = cv2.threshold(
                    cv2.bitwise_not(mask),
                    200, 255, cv2.THRESH_BINARY
                )

                # Now black-out the area of logo in ROI
                img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                # Take only region of logo from logo image.
                img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

                # Put logo in ROI and modify the main image
                dst = cv2.add(img1_bg, img2_fg)
                img[x:x + w2, y:y + h2] = dst

            img = img[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # キューへ格納
            if self.queue.qsize() < 10:
                frame = {}
                frame["img"] = img
                self.queue.put(frame)
            else:
                print(self.queue.qsize())
            self.img = np.copy(img)

        self.ui.pushButton_load.setEnabled(True)
        self.ui.pushButton_search.setEnabled(True)

    def update_frame(self):
        if not self.queue.empty():
            print('updata frame')
            frame = self.queue.get()
            img = frame["img"]
            img_height, img_width, img_colors = img.shape
            self.window_width = self.ui.img_widget.size().width()
            self.window_height = self.ui.img_widget.size().height()
            scale_w = float(self.window_width) / float(img_width)
            scale_h = float(self.window_height) / float(img_height)
            scale = min([scale_w, scale_h])
            if scale == 0:
                scale = 1
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            height, width, bpc = img.shape
            bpl = bpc * width
            image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
            self.img_widget.setImage(image)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = Application()
    myapp.setWindowTitle('team9')
    myapp.show()
    sys.exit(app.exec_())
