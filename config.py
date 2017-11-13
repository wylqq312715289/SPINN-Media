#-*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from easydict import EasyDict as edict


config = edict()
config.pos_path = "./data/MedlatTrainingData/pulsars/"
config.nag_path = "./data/MedlatTrainingData/RFI/"
config.pos_h5_file = "./cache/pos_data_feat.h5" # 特征处理完后持久化到h5py文件
config.nag_h5_file = "./cache/nag_data_feat.h5" # 特征处理完后持久化到h5py文件
config.samples_info_file = "./cache/samples_info.csv"

config.ary_feats_file = "./cache/ary_feats.csv"   # 持久化ary特征文件保存地址
config.sn_mtx_feats_file = "./cache/h5/sn_mtx_feats.h5"
config.int_mtx_feats_file = "./cache/h5/int_mtx_feats.h5"
config.evaluate_file = "./cache/evaluate.csv"
# config.roc_file = "./logs/roc.pdf" # main执行完毕后 roc曲线在该目录下生成图像

if not os.path.exists("./cache/models/"): os.makedirs("./cache/models/")
if not os.path.exists("./cache/h5/"): os.makedirs("./cache/h5/")

config.nag_num = int(1e10)  # 这里控制负样本个数，1e10是正无穷(表示使用所有负样本)
if not os.path.exists(config.samples_info_file):
	pos_files = os.listdir(config.pos_path)
	pos_files = [ config.pos_path + x for x in pos_files ]
	nag_files = os.listdir(config.nag_path)
	nag_files = [ config.nag_path + x for x in nag_files ]
	nag_files = shuffle(nag_files)
	nag_files = list(nag_files[:config.nag_num])
	labels = [ 1.0 for i in range(len(pos_files)) ] + [ 0.0 for i in range(len(nag_files)) ]
	info_df = pd.DataFrame(pos_files+nag_files,columns=["file_path"])
	info_df["label"] = labels
	info_df.to_csv(config.samples_info_file,index=False,index_label=False)

config.kfold = 5 # 总样本集5折交叉验证,这里设置交叉折数
# 单信噪比矩阵，即数据集中的 cand.pdm_plane[2], 规范后矩阵size，按图像处理做CNN
config.img_x, config.img_y, config.channel = 64,64,1
config.batch_size = 512     # 深度模型 分批训练的批量大小
config.epochs = 50           # 总共训练的轮数（实际不会超过该轮次，因为有early_stop限制）
config.cnn_early_stop = 5  # 最优epoch的置信epochs
config.class_num = 2        # 二分类问题
config.pos_nag_rato = 0.25  # 设置正负样本比为1:4