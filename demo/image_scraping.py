# !/usr/bin/env python
# -*- coding: utf-8 -*-
##
#   @package image_scraping.py
#   @brief  Web scrapingを実行し画像データを取得
#
from selenium.webdriver.chrome.options import Options
from time import sleep
from urllib.parse import quote
from bs4 import BeautifulSoup
from selenium import webdriver
import requests
import os
from urllib.parse import quote
import cv2
import numpy as np
import glob
import json


def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


def scroll_bottom(driver):
    """
    ページの最下端までスクロールする

    :param driver: WebDriver
    """
    lastHeight = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        sleep(0.5)  # 読み込まれるのを待つ

        # スクロールされているか判断
        newHeight = driver.execute_script("return document.body.scrollHeight")
        if newHeight == lastHeight:
            break
        lastHeight = newHeight


if __name__ == '__main__':

    # ----------- config --------------#
    # Trueに設定した場合ブラウザは立ち上がらない
    headless_mode = False
    #  ブラウザのオプションを格納する変数を取得
    options = Options()
    #  デフォルトだとHeadlessモードは有効
    #  headless_modeにTrueを設定するとブラウザは立ち上がらなくなる
    options.set_headless(headless_mode)

    chrome_driver_path = './chromedriver_win32/chromedriver.exe'
    timeout = 10

    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.95 Safari/537.36 '

    with open(os.path.join('class_labels.json'), 'r', encoding='utf-8_sig') as f:
        class_labels = json.load(f)

    # -------------------------------#

    # ブラウザを起動
    driver = webdriver.Chrome(
        executable_path=chrome_driver_path,
        chrome_options=options)

    for class_label in class_labels:

        save_dir = os.path.join('./dataset/images/', class_label)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # 検索ワード
        search_term = class_label + '+スーパー'

        # 画像検索のURLを取得
        # hl=ja 国内検索
        # itp:face 顔検索
        # itp: animated
        # itp: photo
        # qdr:y5 5年以内
        # sbd:1 最新の日付順
        search_url = "https://www.google.co.jp/search?hl=ja&q={}&source=lnt&tbm=isch&tbs=itp:photo,ic:trans,qdr:y5,sbd:1".format(
            quote(search_term))

        # 画像検索のURLへアクセス
        driver.get(search_url)
        # スクロールして全ての検索結果を表示する
        # retry_count = 0
        # driver.set_script_timeout(timeout)
        # print('scroll start')
        # scroll_bottom(driver)
        # print('scroll finish')

        # アクセスしたサイトのページソースを返す
        html_source = driver.page_source.encode('utf-8')
        # htmlソースコード解析
        soup = BeautifulSoup(html_source, 'html.parser')
        url_list = []
        for link in soup.find_all("img"):  # imgタグを取得しlinkに入れる
            img_tag = link.get("src")
            if type(img_tag) is str:
                if img_tag[:5] == "https":
                    url_list.append(link.get("src"))

        # -- download --#
        print('download Image')
        img_index = 0
        for url in url_list:
            print('download image url:{}'.format(url))
            # データダウンロード
            response = requests.get(url, allow_redirects=False, timeout=timeout)
            img = None
            img = response.content
            content_type = response.headers["content-type"]
            print("content-type:{}".format(str(content_type)))
            # 拡張子確認
            ext = None
            extensions = ['jpeg', 'jpg', 'png', 'gif']
            for extension in extensions:
                if extension in str(content_type):
                    ext = extension
            if ext is None or img is None:
                sleep(1)
                continue
            # 画像保存
            filename = os.path.join(save_dir, '{0:010d}.{1}'.format(img_index, ext))
            # filename = save_dir + str(img_index) + '.' + ext
            print('file name:{}'.format(filename))
            with open(filename, "wb") as fout:
                fout.write(img)
            img_index = img_index + 1
            sleep(0.3)
            # 描画
            # bin_data = io.BytesIO(img)
            # img_pil = Image.open(bin_data)
            # plt.imshow(img_pil)
            # plt.show()
            #
            # # -- face detection --#
            # if do_face_detection:
            #     # データ番号(ファイル名)取得
            #     face_cascade_path = './haarcascades/haarcascade_frontalface_default.xml'
            #     face_cascade = cv2.CascadeClassifier(face_cascade_path)
            #     files_path = glob.glob('{}/*.jpeg'.format(save_dir))
            #     print(files_path)
            #     for file_path in files_path:
            #         print(file_path)
            #         src = imread(file_path)
            #         src = cv2.resize(src, (300, 300))
            #         src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            #         faces = face_cascade.detectMultiScale(src_gray, scaleFactor=1.1, minNeighbors=10, minSize=(90, 90))
            #         for x, y, w, h in faces:
            #             cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #             face = src[y: y + h, x: x + w]
            #             cv2.imshow('test', src)
            #             cv2.waitKey(1000)
            #             f = cv2.resize(face, (100, 100))
            #             # cv2.imshow('ttest', f)
            #             # cv2.waitKey(0)
