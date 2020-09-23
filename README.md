# An Othello AI Player Implemented with Deep Learning
>## 概要
AIと対戦するオセロゲーム。CNNを使って[譜面データ](http://www.ffothello.org/informatique/la-base-wthor)を学習させ対戦AIを作成。<br/>
`.wtb`形式のデータの解析については[こちら](http://hp.vector.co.jp/authors/VA015468/platina/algo/append_a.html)を参考。<br/>
`PyTorch` 1.6.0を使用。<br/><br/>

>## 実行方法
1. クローンしたディレクトリに`model.pth`があるか確認
2. 存在する場合は`python3 game.py`でAIと対戦できる
3. 存在しない場合は`python3 train.py`を実行し学習が必要
（譜面データがない場合ダウンロードして`./data`ディレクトリを作成し自動格納）
<br/>

※ DL実装のライブラリとして`Pytorch 1.6.0`を使用（他は標準的なライブラリのはず）。<br/>

```
pip3 install torch==1.6.0
```
などでインストール。

>## メモ: 使用した損失関数について
`nn.CrossEntropyLoss`を使用
```python
import torch.nn as nn
loss_func = nn.CrossEntropyLoss()
for epoch in range(EPOCHS):｀
  for i in range(ITERATION_NUM):
    ## 訓練を行う
    outputs = model(x, t)
    loss = loss_func(outputs, t)
```
式は以下の通り<br/>

$$
loss(x,t) = -\log(\frac{\exp(x[t])}{\sum_{j=0}^{n} \exp(x[j])})
$$

ここで$x[t]$を変数とするsoftmax関数は<br/>
$$softmax(x[t]) = \frac{\exp(x[t])}{\sum_{j=0}^{n} \exp(x[j])}$$
となる。これは、$x[t]$が起こりうる確率を表すので、損失関数の式は$x[t]$が起こりうる確率に対しての情報量と見なすことできる。<br/>
すなわち$x[t]$の確実性が増すほど情報としてインパクトが下がり、０に漸近するため、今回のようなケースの場合に用いるのに適していると言える。
