# !/usr/bin/env python
# -*- coding: utf-8 -*-
##
#   @package recipe_converter.py
#   @brief  レシピデータ・セットの作成
#
import json
from gensim.models import word2vec
from pykakasi import kakasi
import jaconv
import numpy as np
import os
import pandas as pd
import codecs
from utils import remove_special_characters, one_hot_vec_encoder

if __name__ == '__main__':

    # --------------------- config --------------------- #
    # 漢字->ひらがな変換器
    kakasi = kakasi()
    kakasi.setMode('J', 'H')
    converter = kakasi.getConverter()

    # word2vec
    model_path_w2v = './model/word2vec/word2vec.gensim.model'
    model_w2v = word2vec.Word2Vec.load(model_path_w2v)
    # 調味料リスト
    seasoning = ('BP', '類', 'コショウ', '黄', '白', '湯',
                 'パン', '糖', '塩', '水', '油', '粉', '酒', 'みりん', '酢', '米', '飯', '味噌', '汁', 'しょうゆ',
                 'こしょう', '味醂', 'ミリン', 'サトウ', 'さとう', 'コショウ', '胡椒',
                 'ごはん', 'オイル', 'ソース', 'パウダー', 'コンソメ', 'めんつゆ', 'パスタ', 'ぱすた', 'コチュジャン')
    # model_path_fasttext = './model/fasttext/model.vec'
    # model_fasttext = gensim.models.KeyedVectors.load_word2vec_format(model_path_fasttext, binary=False)

    fp = open('./dataset/recipes.json', 'r', encoding='utf-8_sig')
    # データセット(元データ)
    recipe_dataset_raw = json.load(fp)

    # ---------------------------------------------- #


    # --(1) 食材のクラス名取得 --#

    # {食材名:頻度}
    recipe_dataset_ingredients = dict()

    for recipe_data in recipe_dataset_raw:
        for recipe_data_food in recipe_data['ingredients']:
            food_name = recipe_data_food['name']
            # 特殊文字削除
            food_name = remove_special_characters(food_name)
            # 食材名が重複する場合
            if food_name in recipe_dataset_ingredients:
                recipe_dataset_ingredients[food_name] += 1
            else:
                recipe_dataset_ingredients[food_name] = 1

    # 調味料以外&Wikipediaにない?食材名＆頻度が一定以下の食材名を削除
    recipe_dataset_ingredients_th = dict()
    for k, v in recipe_dataset_ingredients.items():
        if v > 20:
            try:
                vec = model_w2v.wv[str(k)]
                if all(x not in k for x in seasoning):
                    recipe_dataset_ingredients_th[k] = v
            except:
                pass

    print(len(recipe_dataset_ingredients_th))

    # 調味料以外&漢字->ひらがな,カタカナ->ひらがな変換
    recipe_dataset_ingredients_hira = dict()
    for k, v in recipe_dataset_ingredients_th.items():
        k = converter.do(k)
        k = jaconv.kata2hira(k)
        if k in recipe_dataset_ingredients_hira:
            recipe_dataset_ingredients_hira[k] = recipe_dataset_ingredients_hira[k] + v
        else:
            recipe_dataset_ingredients_hira[k] = v

    # 食材名のうち内包しているものを除外
    # 抑制対象 = True
    index_suppressed = [True] * len(recipe_dataset_ingredients_hira)
    ingredients_name = list(recipe_dataset_ingredients_hira.keys())
    for index, ((k, v), suppressed) in enumerate(zip(recipe_dataset_ingredients_hira.items(), index_suppressed)):
        if suppressed == False:
            continue
        # keyが他の食材名に含まれているかどうか
        for iidex, ingredient_name in enumerate(ingredients_name):
            if index != iidex:
                if k in ingredient_name:
                    index_suppressed[iidex] = False

    ingredients_name = np.array(ingredients_name)[index_suppressed]

    # クラスラベリング
    recipe_dataset_ingredients_sort = dict()
    for class_label in ingredients_name.tolist():
        recipe_dataset_ingredients_sort[class_label] = recipe_dataset_ingredients_hira[class_label]

    class_labels = list()
    for k, v in sorted(recipe_dataset_ingredients_sort.items(), key=lambda x: -x[1])[:100]:
        print(str(k) + ": " + str(v))
        class_labels.append(k)
    class_to_index = {class_labels[i]: i for i in range(len(class_labels))}
    print(class_to_index)

    # 各クラスラベル
    with codecs.open(os.path.join('class_labels.json'), 'w', encoding='utf-8') as fw:
        json.dump(class_labels, fw, ensure_ascii=False, indent=4)

    # クラスインデックス
    with codecs.open(os.path.join('class_to_index.json'), 'w', encoding='utf-8') as fw:
        json.dump(class_to_index, fw, ensure_ascii=False, indent=4)

    # データ・セット作成
    recipe_dataset_df = list()
    for index, recipe_data in enumerate(recipe_dataset_raw):
        ingredients = list()
        for recipe_data_food in recipe_data['ingredients']:
            food_name = recipe_data_food['name']
            # 特殊文字削除
            food_name = remove_special_characters(food_name)
            try:
                # wikipediaにあるかどうか
                vec = model_w2v.wv[str(food_name)]
                # 漢字->ひらがな,カタカナ->ひらがな
                food_name = converter.do(food_name)
                food_name = jaconv.kata2hira(food_name)
                ingredients.append(food_name)
            except:
                pass

        # 食材one-hotベクトルの作成
        one_hot_vec = one_hot_vec_encoder(ingredients, class_labels=class_labels, class_to_index=class_to_index)
        path_vec = os.path.join('./dataset/recipes/{0:010d}.npy'.format(index))
        np.save(path_vec, one_hot_vec)
        # 保存
        path_recipe = os.path.join('./dataset/recipes/{0:010d}.json'.format(index))
        with codecs.open(path_recipe, 'w', encoding='utf-8') as fw:
            json.dump(recipe_data, fw, ensure_ascii=False, indent=4)

        recipe_dataset_df.append([path_recipe, path_vec])

    df = pd.DataFrame.from_records(recipe_dataset_df, columns=['filename', 'vec'])
    df.to_csv('./dataset/recipes.csv', index=False)
