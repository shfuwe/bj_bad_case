import torch
device0 = torch.device('cuda:6' if torch.cuda.is_available() else "cpu")#训练集gpu
device1 = torch.device('cuda:6' if torch.cuda.is_available() else "cpu")#测试集gpu

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader,TensorDataset,Dataset
import numpy as np
import torch

import config
from data_process import data2token,SentimentDataset

from SupConLoss import SupConLoss
criterion_ct= SupConLoss()
criterion_ct.to(device0)


# 定义日志（data文件夹下，同级目录新建一个data文件夹）
import time
import datetime
import pytz
tz = pytz.timezone('Asia/Shanghai')
def write_log(w):
    file_name = config.sys_path+'/data/' + datetime.date.today().strftime('%m%d') + "_{}.log".format("bert_base_st")
    t0 = datetime.datetime.now(tz).strftime('%H:%M:%S')
    info = "{} : {}".format(t0, w)
    print(info)
    with open(file_name, 'a') as f:
        f.write(info + '\n')

write_log('_________ data process _________')
data_train=pd.read_csv(config.train_path,header=None)
data_val=pd.read_csv(config.test_path,header=None)

data_train.columns=['label','title','00']
data_val.columns=['label','title','00']
del data_train['00']
del data_val['00']

data_train_label=data_train['label'].tolist()
data_val_label=data_val['label'].tolist()

data_train_label=[i-1 for i in data_train_label]
data_val_label=[i-1 for i in data_val_label]

data_train['label']=data_train_label
data_val['label']=data_val_label


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

data_train=data2token(data_train,tokenizer)
data_val=data2token(data_val,tokenizer)

#按batch_size分
batch_size=32
train_loader = DataLoader(
    SentimentDataset(data_train), 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=0
)
test_loader = DataLoader(
    SentimentDataset(data_val), 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=0
)


write_log('_________ train _________')
from train import train_one_epoch_ct
def train(epoch_num):
    min_test_epoch_loss=999999
    for i in range(epoch_num):
        write_log("Current lr : {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        min_test_epoch_loss=train_one_epoch_ct(config.sys_path+'/data/bert_ct/',min_test_epoch_loss,device0,device1,
                                i,train_loader,test_loader,
                                write_log,cls,optimizer,
                                test_epoch_loss_l,test_acc_l,train_epoch_loss_l,train_acc_l,criterion_ct,lmd=150)
        scheduler.step()

from model import fn_cls
cls = fn_cls(device0)
# cls=torch.load(config.sys_path+'/data/cls_0.86353_183.34991_1642153938.7560434.model",map_location=device0)
from torch import optim
optimizer = optim.Adam(cls.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.gamma)
# test(device1)
train_acc_l=[]
train_epoch_loss_l=[]
test_acc_l=[]
test_epoch_loss_l=[]
train(10)

write_log('_________ plt _________')
import matplotlib.pyplot as plt
def _plt():
    plt.plot([i for i in range(len(train_acc_l))], train_acc_l)
    plt.title('train_acc')
    plt.show()
    plt.plot([i for i in range(len(train_epoch_loss_l))], train_epoch_loss_l)
    plt.title('train_epoch_loss')
    plt.show()
    plt.plot([i for i in range(len(test_acc_l))], test_acc_l)
    plt.title('test_acc')
    plt.show()
    plt.plot([i for i in range(len(test_epoch_loss_l))], test_epoch_loss_l)
    plt.title('test_epoch_loss')
    plt.show()
_plt()