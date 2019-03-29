# !/usr/bin/env python
# -*- coding: utf-8 -*-
##
#   @package image_scraping.py
#   @brief  翻訳
#
import os
import json
from googletrans import Translator
import codecs
import jaconv

if __name__ == '__main__':

    # クラスインデックス
    with open(os.path.join('class_to_index.json'), 'r', encoding='utf-8_sig') as f:
        class_to_index = json.load(f)
    # クラスラベル
    with open(os.path.join('class_labels.json'), 'r', encoding='utf-8_sig') as f:
        class_labels = json.load(f)

    #
    with open(os.path.join('class_labels_transformer.json'), 'r', encoding='utf-8_sig') as f:
        class_labels_transformer = json.load(f)

    # 変換
    class_labels_en = list()
    class_to_index_en = dict()
    for class_label in class_labels:
        try:
            class_label_en = class_labels_transformer[class_label]
            class_to_index_en[class_label_en] = class_to_index[class_label]
            class_labels_en.append(class_label_en)
        except:
            print(class_label)

    # 各クラスラベル
    with codecs.open(os.path.join('class_labels_en.json'), 'w', encoding='utf-8') as fw:
        json.dump(class_labels_en, fw, ensure_ascii=False, indent=4)

    # クラスインデックス
    with codecs.open(os.path.join('class_to_index_en.json'), 'w', encoding='utf-8') as fw:
        json.dump(class_to_index_en, fw, ensure_ascii=False, indent=4)



    #
    # class_labels_transform = dict()
    # for class_label in class_labels:
    #     translator = Translator()
    #     class_label = jaconv.hira2kata(class_label)
    #     class_labels_transform[class_label] = translator.translate(class_label,dest='en').text
    #
    # with codecs.open(os.path.join('class_labels_transformer2.json'), 'w', encoding='utf-8') as fw:
    #     json.dump(class_labels_transform, fw, ensure_ascii=False, indent=4)
