# W2VBiLSTM
GoogleColaboratoryとVSCODE(Windows)で動作を確認

.ipynbファイルを以下の場所に作成する

ディレクトリ構成は
```
lab/
　├ code/
　│　├ ().ipynb
　│　└ save_model/
　│ 　 └ ().pth
　└ data/
　 　└ 短文音声/
    　 └ test/
         └ thiswas/
```
モデルに入力してネイティブ率を判定したい音声データは次のパスにおく。

`lab/data/短文音声/test/{任意の名前}/`

音声のファイル名は`jp_1.wav`または`us_1.wav`から始めて`jp_15.wav`のように連続する整数の番号を振る。

音声はサンプリングレート16000Hz、モノラルの.wavファイルのみ使用できる。

テストに使用する音声データはjp,usそれぞれ同じ数用意する必要がある(jp,usは本来ノンネイティブ,ネイティブの音声をそれぞれ入れるために名付けられたが,テストにおいては単なるラベルに過ぎない)。

出力されるモデルでの正解率は、このラベルに基づいて算出される(jpの音声が ネイティブ率>ノンネイティブ率 であれば不正解と判定される)。


> ver 0.0.1 2023/04/09 仮

# 使い方
まず必要なライブラリのインストール
```py
!pip install transformers
!pip install scikit-learn
```
インポート
```py
import numpy as np
import torch
from statistics import mean
from torch import nn, optim
from torch.utils.data import (Dataset,
                              DataLoader,
                              TensorDataset)
import tqdm
import pandas as pd
#import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset
#from torchvision.models import resnet50
import sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold
import random
from tqdm import tqdm
import time
import glob
import scipy.io.wavfile
from transformers import Wav2Vec2Processor, Wav2Vec2Model
```
このライブラリをインストールしてインポート
```py
!pip install git+https://github.com/treeb23/W2VBiLSTM.git
import W2VBiLSTM as wb
```
最初に作業ディレクトリをfilepathに設定等
```py
wb.setup()
```
学習済みモデルで用意した音声をテストしネイティブ率を出力する

`wb.test()`の引数は(学習済みモデルの名前,用意した音声(JPの数),音声のあるディレクトリ名)
```py
wb.test("LSTM_01",15,"thiswas")
```
