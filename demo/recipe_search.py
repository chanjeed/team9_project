# !/usr/bin/env python
# -*- coding: utf-8 -*-
##
#   @package recipe_converter.py
#   @brief  レシピの検索
#
import os
import json
from utils import remove_special_characters, one_hot_vec_encoder
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # クラスインデックス
    with open(os.path.join('class_to_index.json'), 'r', encoding='utf-8_sig') as f:
        class_to_index = json.load(f)
    # クラスラベル
    with open(os.path.join('class_labels.json'), 'r', encoding='utf-8_sig') as f:
        class_labels = json.load(f)

    dtype = {'id': 'object', 'x01': 'str', 'x02': 'str'}
    df = pd.read_csv('dataset/recipes.csv', dtype=dtype)
    ingredients_vec_query = list()
    for path_vec in df['vec'].values.tolist():
        ingredients_vec_query.append(np.load(path_vec))

    ingredients_vec_query = np.array(ingredients_vec_query)

    # -- 検索 --#
    ingredients = ['じゃがいも', 'にんじん', 'ぎゅうにく', 'たまねぎ']
    ingredients_vec_src = one_hot_vec_encoder(ingredients=ingredients, class_labels=class_labels,
                                              class_to_index=class_to_index)
    ingredients_vec_src = np.expand_dims(ingredients_vec_src, axis=0)

    result = ingredients_vec_src * ingredients_vec_query
    result = np.sum(result, axis=1)
    indices_recommend = np.argsort(result)[::-1]

    for index_recommend in indices_recommend[:10]:
        path = df.iloc[index_recommend, 0]
        with open(path, 'r', encoding='utf-8_sig') as f:
            recipe = json.load(f)
        print(recipe['title'])
        # print(recipe['processes'])
