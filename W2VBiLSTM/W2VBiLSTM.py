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
import glob
import scipy.io.wavfile
from transformers import Wav2Vec2Processor, Wav2Vec2Model


def setup(): # ファイルパスの設定
    np.random.seed(1)
    torch.cuda.manual_seed_all(1)
    torch.cuda.manual_seed(1)
    torch.manual_seed(1)
    random.seed(1)
    global user,lr,v_size,epoch,bs,hiddensize,dropout,layer,filenames,file_nums,processor,model,f_path,t_path
    user = 2 #2クラスにわける
    lr = 0.0001 #learning rate 学習の重みの変化
    v_size = 768 #wave2vecから出力されるベクトルのサイズ
    epoch =30 #学習回数
    bs = 190000 #データの最大サイズ
    hiddensize =128 #ネットワークの中間層のノード数
    dropout=0.3 #出力層に出力しない割合　過学習を防ぐため
    layer=1 #中間層の層の数
    filenames=['jpn','us'] #ファイルの名前の共通部分
    file_nums=[130,55] #ファイルの数
    #import pdb; pdb.set_trace()

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    f_path=".."
    try:
        # Google Driveをcolabにマウント
        from google.colab import drive
        drive.mount('/content/drive')
        f_path="/content/drive/MyDrive/lab"
    except ModuleNotFoundError as e:
        print(e)
    print(f"[ファイルパスf_pathを'{f_path}'に設定]")
    t_path=f_path


class SequenceTaggingNet(nn.Module):
    def __init__(self,
                 input_dim=v_size, #入力層のノード数
                 hidden_size=hiddensize, #中間層のノード数
                 num_layers=layer, #中間層の層の数
                 dropout=dropout): #どろっぷあうと率
        super().__init__()
        self.lstm = nn.LSTM(input_dim,hidden_size, num_layers,
                            dropout=dropout,batch_first=True,bidirectional=True) #LSTMの設定
        self.linear = nn.Linear(hidden_size*2,user) #全結合層の設定
        #self.softmax = nn.Softmax()
    def forward(self, x, h0=None,l=None ): #実際のネットワークの流れ
        x = processor(x,sampling_rate=16000,return_tensors="pt") #wavファイルをwav2vecに入れる前の処理 pt:pytorchのテンソル
        x = model(**x).last_hidden_state #wav2vecにwavファイルを入力
        x=x.view(1,-1,768) #LSTMに入力するためにデータの形を変更
        x = x.to(device) #データをGPUに飛ばす
        x,h = self.lstm(x, h0) #LSTMにデータを入力
       # x = x[:,-1,:]
        x = torch.cat([h[0][0], h[0][1]], dim=1) #前方向と後ろ方向の最後の出力を結合
        x = self.linear(x) #全結合層へ入力
        return x


def eval_net(net, train_loader, device): #評価用関数
    net.eval() #ネットワークを評価モードに
    ys = []
    ypreds = []
    tmp = []
    correct = 0 #正解数格納用変数
    uncorrect = 0
    uncleSum = 0 #全体の問題数
    q=0
    for (a,b) in train_loader:
        b = b[0]
        b = b.view(1,user)
        x = a.to(device)
        y = b.to(device)
       # l = l.to(device)
        with torch.no_grad(): #重みの更新がされない（検証用のデータで学習させない）
            out,outlabel = [],[]
            y_pred = net(x) #ネットワークにデータを入れる
            tmp.append(y_pred)
            out = torch.argmax(y_pred,dim=1) #ネットワークから返ってきた値の最大値の場世を探す
            outlabel = torch.argmax(y,dim=1) #正解ラベルの1の場所
            loss = loss_f(y_pred, outlabel.long()) #誤差関数の値を調べる
            for a in range (len(b)):
                if(out[a] == outlabel[a]): #答え合わせ
                    correct +=1
                    uncleSum +=1
                    #else:uncleSum += 1
                else:uncleSum += 1
    acc = correct / uncleSum #正解率
    return acc,loss,tmp


def load_model(model_name):
    global device,loss_f,ALL_lost,net_test,opt,model_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_f = nn.CrossEntropyLoss()
    ALL_lost=0
    net_test = SequenceTaggingNet()
    net_test.to(device)
    opt = optim.Adam(net_test.parameters(),lr)
    model_path = f"{f_path}/code/save_model/{model_name}.pth" # モデル保存先パス
    net_test.load_state_dict(torch.load(model_path, map_location=device))


class ACDataset_test(Dataset):
    def __init__(self):
        self.data_num = 0 #全体のデータ数を保存する変数
        self.userindex = torch.Tensor(user)
        for USER in range(user): #クラス分類の数回るfor文
            flag=0 #一番最初のデータを処理する用のフラッグ
            w=0
            wav_paths=[]
            for i in range(maxc): #実際に入力データのパスを格納する　file_nums[USER] 10
                wav_path=glob.glob(f_path+f'/data/短文音声/test/{t_path}/{filenames[USER]}_{i+1}.wav')
                wav_paths.append(wav_path)
            for x in wav_paths: #実際にデータを読み込んで正解ラベルを生成する
                _,data_x = scipy.io.wavfile.read(x[0]) #wavファイルを読み込む
                data_X=np.array(data_x,dtype=float) #wavファイルをnumpy型に変換
                data_X=torch.from_numpy(data_X) #wavファイルをテンソル型に変換
                i = data_X.shape[0] #データの長さを取得
                if USER == 0 and flag == 0: #最初のデータを処理する
                    if i%bs != 0: #現在のデータの長さがbsで割りきれないなら処理
                        amari = i%bs
                        amari = bs-amari #bsと現在のデータの長さの差を取得
                        zeros = torch.zeros(amari,dtype=torch.float64) #amariの数だけ0が入った配列を作る
                        self.label = torch.Tensor() #正解ラベル格納用
                        data_X = torch.cat([zeros,data_X],axis=0) #zerosと結合してbsに長さを合わせる
                        self.data = data_X
                        self.label = torch.Tensor()
                        i = amari+i
                        flag=1
                    else:
                        self.data = data_X;
                        self.label = torch.Tensor()
                        i = data_X.shape[0]
                        flag=1
                else:
                    if i%bs != 0:
                        amari = i%bs
                        amari = bs-amari
                        zeros = torch.zeros(amari,dtype=torch.float64)
                        data_X = torch.cat([zeros,data_X],axis=0)
                        self.data = torch.cat([self.data,data_X],axis=0)
                        i = amari+i
                        #print(i)
                    else:
                        self.data = torch.cat([self.data,data_X],axis=0)
                        i = data_X.shape[0]
                w+=1
                s = torch.zeros(i,user) #正解ラベルを生成
                s[0:i,USER]=1 #指定した行のラベルを1にする
                self.label=torch.cat([self.label,s],dim=0) #全体のラベルと結合
                self.data_num = self.data_num + i #データの個数を数える

                self.userindex[USER] = len(self.data)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label =  self.label[idx]
        out_seqlen = self.seqlen[idx]

        return out_data, out_label,out_seqlen
    def getindex(self,USER):

        return self.userindex[USER]

def test(model_name,maxcount,testdatapath): #maxcountはJPのファイル数
    load_model(model_name)
    global t_path,maxc
    maxc=maxcount
    t_path=testdatapath
    alldata= ACDataset_test()
    train_data = ACDataset_test()
    #print(alldata.getindex(0).item())
    #flag_net=1
    for i in range(user):
        print(i)
        if(i==0):
            train_data.data = alldata.data[0:int((alldata.getindex(0).item()))]
            train_data.label = alldata.label[0:int((alldata.getindex(0).item()))]

        else:
            train_data.data = torch.cat([train_data.data,alldata.data[int(alldata.getindex(i-1).item()):(int(alldata.getindex(i).item()))]])
            train_data.label = torch.cat([train_data.label,alldata.label[int(alldata.getindex(i-1).item()):(int(alldata.getindex(i).item()))]])

    #import pdb; pdb.set_trace()

    train_dataset_test = torch.utils.data.TensorDataset(train_data.data,train_data.label)
    train_loader_test = DataLoader(train_dataset_test, batch_size=bs,
                            shuffle=False, num_workers=0)
    #val_acc,_ ,softmax_result= eval_net(net, train_loader_test, device)


    val_acc,_ ,softmax_result= eval_net(net_test, train_loader_test, device)
    print(model_name,"での正解率",val_acc*100,"%")


    #x=[-0.1761,0.1189]
    def softmax(x):
        c = np.max(x)
        exp_x = np.exp(x - c)
        sum_exp_x = np.sum(exp_x)
        y = exp_x / sum_exp_x
        return y
    #softmax(x)

    count=0

    print("ここからJP")
    for i in softmax_result:
        print(softmax([i[0][0],i[0][1]])*100)
        count=count+1
        if count==maxcount:
            print("ここからUS")
