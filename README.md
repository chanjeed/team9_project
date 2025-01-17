# Deep learning 基礎講座 自由課題 チーム９

## 冷蔵庫の食材を基にした最適レシピの提案

### 流れ

1. 食材の画像を入力
2. 食材の分散表現とレシピの分散表現の類似度を計算する
3. 最適レシピを提案する

### 使用するレシピのデータセット

- githubにあるレシピデータセット（日本語）10,272件 https://github.com/leetenki/cookpad_data
- Recipe1M（英語）1,000,000件

データセットのうち調理方法だけを抽出し、
日本語については、
MeCabおよびCOTOHA APIを用いて
形態素解析を実施、
英語についてはnltkを用いてTokenizeし、
学習させやすい形への前処理を行った。

### プログラムについて

- 単語の分散表現
  - 各単語に対してWord2Vecの学習モデル（CBOW、Skipgram、Skipgram with Negative Sampling (SGNS)）を用いて
単語のベクトルを学習し、単語の分散表現を作った。類似単語の評価が最も高い SGNS を採用した。

- クラスタリング
  - cos類似度を使ったクラスタリング
(sd-CRP clustering algorithm)と、
ユークリッド距離を使ったk-means法での
クラスタリングの二種類を試した

- レシピ文の分散表現
  - レシピ文に含まれるすべての単語のベクトルを足すと、料理に関係のない単語まで足されてしまうので、
料理に関係するような特定のクラスタに属する単語のベクトルのみの総和を
レシピ文の分散表現とした
