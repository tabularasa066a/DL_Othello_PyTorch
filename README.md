# Deep-Othello
>## 概要
AIと対戦するオセロゲーム。CNNを使って[譜面データ](http://www.ffothello.org/informatique/la-base-wthor)を学習させ対戦AIを作成。<br/>
`.wtb`形式のデータの解析については[こちら](http://hp.vector.co.jp/authors/VA015468/platina/algo/append_a.html)を参考。<br/>
`PyTorch` 1.6.0を使用。<br/><br/>

>## 実行方法
1. クローンしたディレクトリに`model.pth`があるか確認
2. 存在する場合は`python3 game.py`でAIと対戦できる
3. 存在しない場合は`python3 train.py`を実行し学習が必要
（譜面データがない場合ダウンロードして`./data`ディレクトリを作成し自動格納します）
<br/>
※1 `unicode decode error`のような旧世紀の遺産（失礼！）を引き継ぐことを嫌いプログラムの表示には英語を用いています（こちらもレビューしてほしい。。。）。

※2 DL実装のライブラリとして`Pytorch 1.6.0`を使用しています（他は標準的なライブラリのはずです）。<br/>
```
pip3 install torch==1.6.0
```
などでインストールしてください。

※3 すでに`model.pth`ファイルが存在している状態で学習すると、学習パフォーマンスが下がるので`trains.py`実行したい場合は`model.pth`を削除するかファイル名を変えてください。
（推論時はsoftmax出力としているが学習時はラベルデータとの整合性の点から確率出力とせず譜面出力としておく必要がある為、順方向出力をフェーズに応じて切り替えている。）

>## メモ: 使用した損失関数について
`nn.CrossEntropyLoss`を使用した
```python
import torch.nn as nn
loss_func = nn.CrossEntropyLoss()
for epoch in range(EPOCHS):
  for i in range(ITERATION_NUM):
    ## 訓練を行う
    outputs = model(x, t)
    loss = loss_func(outputs, t)
```


