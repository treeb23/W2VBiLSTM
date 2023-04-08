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


> ver 0.0.1 2023/04/09 仮

# 使い方
まず必要なライブラリのインストール
```
!pip install transformers
```
インポート
```
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
from openpyxl import Workbook, load_workbook
```
このライブラリをインストールしてインポート
```
!pip install git+https://github.com/treeb23/W2VBiLSTM.git
import W2VBiLSTM as wb
```
最初に作業ディレクトリをfilepathに設定等
```
wb.setup()
```
学習済みモデルでテストしネイティブ率の判定
```
wb.test("LSTM_01",15,"thiswas")
```
