# atmaCup #5 solution (public: 2nd -> Private: 6th)

![leaderboard](https://user-images.githubusercontent.com/7401498/83965339-f2a77e00-a8ed-11ea-9ba9-763605bddd3d.png)

atmaCup #5で実装した解法です。

公開にあたりコード整理と追試を行い、Private1位相当のスコアを出すことができました。
本リポジトリはコンペ時点での最終提出のコードではなく、追試のコードであることを予めご理解ください。

## 解法概要

全体像

- LightGBMとNN(Conv1d)のアンサンブルです
- スペクトルにはSavitzky-Golay Filteringをかけました
- スペクトルに微分をかけることで情報量を増やしました

LightGBM

- 生のスペクトル及び微分したスペクトルに対し、集約特徴量を作成しました
- 最大ピークに対して n%の高さで切ったときのピークの個数、ピークが存在する領域の幅などの特徴量を作成しました

NN

- スペクトルはConv1Dに通しました
- LightGBMで使った特徴量のうち、効いていたものをMLPに通しました
- 上2つをconcatし、MLPに通して出力を得ました
- Conv1Dは複数のkernel_sizeに並列に通し、GlobalMaxPoolingをかけたあとにconcatしました
- Conv1Dのkernel_sizeを大きめに取ることでスコアが改善しました

バリデーション

- GroupKFoldとStratifiedKFoldを試しました
- 最終的にはStratifiedKFoldを利用しました。CVとLBでスコアの相関が取れていたためです

アンサンブル

- コンペ時点では `LightGBM:NN = 0.25:0.75` でアンサンブルしていました
- 結果として、LightGBM を混ぜない方がスコアが良かったです

## 実行方法

```bash
$ poetry install
$ poetry run python -m src.features  # 特徴量作成
$ poetry run python run_lgbm.py  # LightGBMを訓練&予測
$ poetry run python run_nn.py  # NNを訓練&予測
```

## リンク

- [atmaCup #5に参加しました（Public2位→Private6位）](https://amalog.hateblo.jp/entry/atmacup-5)
