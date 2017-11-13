#-*- coding:utf-8 -*-
import os
import numpy as np

CUDA_VISIBLE_DEVICES = "1" # 使用第 CUDA_VISIBLE_DEVICES 块GPU显卡
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

from utils import over_sampling
from config import config
from modules import MedlatModel,DataLoader

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

from utils import *

def main():
    loader = DataLoader(config.pos_path,config.nag_path)   # 正负样本生成器
    train_data, labels = loader.load_data()
    skf = StratifiedKFold( n_splits = config.kfold, shuffle=True, random_state=2017 )
    
    for i,(train_idx,vali_idx) in enumerate(skf.split(train_data, np.zeros((len(labels),))),1):
        
        train_x, train_y = train_data[train_idx], labels[train_idx] # 交叉验证训练集部分
        train_x, train_y = over_sampling( train_x, train_y, config.pos_nag_rato ) # 数据过采样到要求比例
        
        vali_x, vali_y = train_data[vali_idx], labels[vali_idx]     # 交叉验证验证集部分
        # vali_x, vali_y = over_sampling(vali_x,vali_y,config.pos_nag_rato) # 验证集不用过采样时请将此行注释

        model = MedlatModel("./cache/models/cnn_model_%d.h5"%(i))

        train_x = loader.get_inputs_set(train_x)
        vali_x = loader.get_inputs_set(vali_x)
        train_y = np.array([ [0.,1.] if train_y[x] == 1.0 else [1.,0.] for x in range(len(train_y)) ])
        vali_y = np.array([ [0.,1.] if vali_y[x] == 1.0 else [1.,0.] for x in range(len(vali_y)) ])
        
        model.train_model(train_x, train_y, vali_x, vali_y)
        data_y, pred = model.evaluate(vali_x, vali_y)
        if i == 1:
            all_pred_y = pred;
            all_real_y = data_y;
        else:
            all_pred_y = np.concatenate((all_pred_y,pred),axis=0)
            all_real_y = np.concatenate((all_real_y,data_y),axis=0)
        print "*"*20 + "  %d fold end  "%(i) + "*"*20
        # if i >= 1: break #交叉验证时请注释此行

    ######################### 总评估方法 #########################
    roc_df = pd.DataFrame(all_pred_y,columns=["prob_0","prob_1"])
    roc_df["real"] = all_real_y
    roc_df.to_csv(config.evaluate_file,index=False,index_label=False)
    print( "%s had saved. Please download it."%(config.evaluate_file) )
    all_pred_y = np.argmax(all_pred_y, axis=1)
    confuse_matrix = metrics.confusion_matrix(all_real_y, all_pred_y)
    TN, FP, FN, TP = confuse_matrix.ravel()
    print "confusion matrix: \n",confuse_matrix
    print "accuracy score: ",metrics.accuracy_score(all_real_y, all_pred_y)
    print "classification report:\n",metrics.classification_report(all_real_y, all_pred_y)
    print "F1 score: ",metrics.f1_score(all_real_y, all_pred_y)
    print "recall: ", 1.* TP / (TP + FN)
    print "False positive rate: ", 1.*FP / (FP + TN)

if __name__ == '__main__':
    main()