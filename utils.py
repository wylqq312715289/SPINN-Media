#-*- coding:utf-8 -*-
import re,os,copy
import xml.etree.cElementTree as ET
import numpy as np
np.random.seed(2017) # 固定随机种子
import pandas as pd
import cv2
import h5py
from sklearn import preprocessing
from sklearn.utils import shuffle

# 一般矩阵归一化
def my_normalization( data_ary, axis=0 ):
    # axis = 0 按列归一化; 1时按行归一化
    if axis == 1:
        data_ary = np.matrix(data_ary).T
        ans = preprocessing.scale(data_ary)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1.0,1.0))
        ans = min_max_scaler.fit_transform(ans)
        ans = np.matrix(ans).T
        ans = np.array(ans)
    else:
        ans = preprocessing.scale(data_ary)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1.0,1.0))
        ans = min_max_scaler.fit_transform(ans)
    return ans

def one_hot( data_ary, one_hot_len):
    # data_ary = array([1,2,3,5,6,7,9])
    # one_hot_len: one_hot最长列
    max_num = np.max(data_ary);
    ans = np.zeros((len(data_ary),one_hot_len),dtype=np.float64)
    for i in range(len(data_ary)):
        ans[ i, int(data_ary[i]) ] = 1.0
    return ans

def re_onehot( data_ary ):
    # 一定要注意输入格式是 否则用函数 
    # pred = np.argmax(pred,axis=1)
    # data_ary = array([[0,0,0,1.0],[1.0,0,0,0],...])
    ans = np.zeros((len(data_ary),),dtype=np.float64)
    for i in range(len(data_ary)):
        for j in range(len(data_ary[i,:])):
            if data_ary[i,j] == 1.0:
                ans[i] = 1.0*j;
                break;
    return ans

# 画图时列分桶
def data_buckets(x,xlabel):
    for bucket_id in range(len(xlabel)):
       if x<=xlabel[bucket_id]: return bucket_id
       else: continue;
    return len(xlabel)

# 数据归一化
def sub_normalize(img_mtx):
    # z-score
    std = np.std(img_mtx)
    mean = np.mean(img_mtx)
    img_mtx = (1.0*img_mtx-mean)/std
    # min-max 归一化
    _max = np.max(img_mtx)
    _min = np.min(img_mtx)
    img_mtx = (img_mtx-_min)/(_max - _min)
    return (img_mtx-0.5)*2.0  #规范到区间[-1,1] 方便relu使用

# 将数据写入h5文件
def write2H5(h5DumpFile,data):
    # if not os.path.exists(h5DumpFile): os.makedir(h5DumpFile)
    with h5py.File(h5DumpFile, "w") as f:
        f.create_dataset("train_feat", data=data, dtype=np.float32)

# 将数据从h5文件导出
def readH5(h5DumpFile):
    feat = [];
    with h5py.File(h5DumpFile, "r") as f:
        feat.append(f['train_feat'][:])
    feat = np.concatenate(feat, 1)
    print('readH5 Feature.shape=', feat.shape)
    return feat.astype(np.float32)

# 二分类问题过采样方案 没有使用smote过采样方案
def over_sampling(train_x,train_y,pos_nag_rato):
    print "begin to over_sampling ......"
    label_df = pd.DataFrame(train_y,columns=["label"])
    label_df["idx"] = range(0,len(label_df.index))
    pos_num = len( label_df[label_df["label"]==1].index )
    nag_num = len( label_df[label_df["label"]==0].index )
    # 正样本过采样
    if (1.0*pos_num / nag_num) < pos_nag_rato:
        need_pos_epochs = int(pos_nag_rato*nag_num - pos_num) // pos_num
        extra_need_pos_num = int(pos_nag_rato*nag_num - pos_num) % pos_num
        for i in range(need_pos_epochs):
            idx = list(label_df["idx"][label_df["label"]==1].values)
            train_x = np.concatenate((train_x, train_x[idx]), axis=0)
            train_y = np.concatenate((train_y, train_y[idx]), axis=0)
        idx = list(label_df["idx"][label_df["label"]==1].values)
        idx = shuffle(idx)[:extra_need_pos_num]
        train_x = np.concatenate((train_x, train_x[idx]), axis=0)
        train_y = np.concatenate((train_y, train_y[idx]), axis=0)
    # 负样本过采样
    if (1.0*pos_num / nag_num) > pos_nag_rato:
        need_nag_epochs = int(1.0*pos_num/pos_nag_rato - nag_num) // nag_num
        extra_need_nag_num = int(1.0*pos_num/pos_nag_rato - nag_num) % nag_num
        for i in range(need_nag_epochs):
            idx = list(label_df["idx"][label_df["label"]==0].values)
            train_x = np.concatenate((train_x, train_x[idx]), axis=0)
            train_y = np.concatenate((train_y, train_y[idx]), axis=0)
        idx = list(label_df["idx"][label_df["label"]==0].values)
        idx = shuffle(idx)[:extra_need_nag_num]
        train_x = np.concatenate((train_x, train_x[idx]), axis=0)
        train_y = np.concatenate((train_y, train_y[idx]), axis=0)
    idx = range(len(train_y))
    idx = shuffle(idx)
    print "end over_sampling, pos/nag=%lf,pos_nag_rato=%lf"%(1.0*np.sum(train_y)/(len(train_y)-np.sum(train_y)),pos_nag_rato)
    return train_x[idx], train_y[idx]
