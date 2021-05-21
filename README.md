# comp_breast
## 今回のやったこと
### データの処理
* Group Kfold
* 50×50の画像をモデルごとにresize。
モデルごとに異なるが、画像サイズをある程度大きくしたほうが結果は良かった。(128×128など)

### 学習
* 最適化手法:adam
* スケジューラー:CosineAnnealingWarmRestarts
* 損失関数:CrossEntropyLoss

### 後処理
* 平均、logistic、LightGBMの3手法でアンサンブル
使用したモデルは以下に記載。この時点でとりあえず陽性陰性の分類をする。

* x,y座標のユークリッド距離を求めて、knnで近傍点をK個抽出
ある画像の近傍点に陽性の画像が閾値以上あれば陽性(1)、そうでなけば陰性(0)とした。
Kや閾値は、validationデータで最もaccが高くなるように求めた（K=13, th=0.4）
![knn_images](./images/knn_sample.png)  
(左:正解 中:knnの後処理前 右:knnの後処理後)

モデルの求めた0~1の値の情報を残すようなknnも試してみたが、あまりうまくいかず、、

### 学習させてみたモデル
* Vision Transformer (resnetのハイブリッド版も使用)
* Efficientnet_b0~b2
* nfnet_f0~1(推論結果は提出に間に合わず)
* efficientnetv2(推論結果は提出に間に合わず)

### 最終提出に使用したモデル
* Vision Transformer (vit_base_patch16_224) 
efficientnetとのアンサンブルは単体より精度が低くなったので、最終提出では出していません。
