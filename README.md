# atmaCup #5 solution (public: 2nd -> Private: 6th)

atmaCup #5 で実装した解法です。

GitHub 公開にあたりコード整理と追試を行い、Private1 位相当のスコアを出すことができました。
本リポジトリはコンペ時点での最終提出のコードではなく、追試のコードであることを予めご理解ください。

## 解法概要

全体像

-   LightGBM と NN(Conv1d)のアンサンブルです
-   スペクトルには Savitzky-Golay Filtering をかけました
-   スペクトルに微分をかけることで情報量を増やしました

LightGBM

-   生のスペクトル及び微分したスペクトルに対し、集約特徴量を作成しました
-   最大ピークに対して n%の高さで切ったときのピークの個数、ピークが存在する領域の幅などの特徴量を作成しました

NN

-   スペクトルは Conv1D に通しました
-   LightGBM で使った特徴量のうち、聞いていたものを MLP に通しました
-   上 2 つを concat し、MLP に通して出力を得ました
-   Conv1D は複数の kernel_size に並列に通し、GlobalMaxPooling をかけたあとに concat しました
-   Conv1D の kernel_size を大きめに取ることでスコアが改善しました

バリデーション

-   StratifiedKFold を利用しました。CV と LB でスコアの相関が取れていたためです

アンサンブル

-   コンペ時点では `LightGBM:NN = 0.25:0.75` でアンサンブルしていました
-   結果として、LightGBM を混ぜない方がスコアが良かったです

## 実行方法

```bash
$ poetry install
$ poetry run python -m src.features  # 特徴量作成
$ poetry run python run_lgbm.py  # LightGBMを訓練&予測
$ poetry run python run_nn.py  # NNを訓練&予測
```
