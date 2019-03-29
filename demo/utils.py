# !/usr/bin/env python
# -*- coding: utf-8 -*-
##
#   @package demo_team9.py
#   @brief  ユーティリティ
#
import re
import numpy as np


# 特殊文字削除
def remove_special_characters(text):
    non_CJK_patterns = re.compile("[^"
                                  u"\U00003040-\U0000309F"  # Hiragana
                                  u"\U000030A0-\U000030FF"  # Katakana
                                  u"\U0000FF65-\U0000FF9F"  # Half width Katakana
                                  u"\U0000FF10-\U0000FF19"  # Full width digits
                                  u"\U0000FF21-\U0000FF3A"  # Full width Upper case  English Alphabets
                                  u"\U0000FF41-\U0000FF5A"  # Full width Lower case English Alphabets
                                  u"\U00000030-\U00000039"  # Half width digits
                                  u"\U00000041-\U0000005A"  # Half width  Upper case English Alphabets
                                  u"\U00000061-\U0000007A"  # Half width Lower case English Alphabets
                                  u"\U00003190-\U0000319F"  # Kanbun
                                  u"\U00004E00-\U00009FFF"  # CJK unified ideographs. kanjis
                                  "]+", flags=re.UNICODE)
    return non_CJK_patterns.sub(r"", text)


# 食材のone-hotベクトル作成
def one_hot_vec_encoder(ingredients, class_labels, class_to_index):
    ingredients_one_hot_vector = np.zeros(len(class_labels))
    # 食材名
    for ingredient in ingredients:
        for class_label in class_labels:
            # クラス名が食材名に含まれている場合
            if class_label in ingredient:
                ingredients_one_hot_vector[class_to_index[class_label]] = 1

    return ingredients_one_hot_vector