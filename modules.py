#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import os,math
from sklearn.utils import shuffle
import cv2
from tqdm import tqdm
import xml.etree.cElementTree as ET

from sklearn import metrics

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras import optimizers
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Convolution2D, MaxPooling2D, Conv2D, GlobalAveragePooling2D, concatenate
from keras.layers import Activation, Dropout, Flatten, Dense, Dropout, Input 
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras import applications
from keras.models import Model
from keras.utils.data_utils import get_file

from utils import *
from config import config
from feats_and_plot import get_all_samples_feats

# vgg-net
class MedlatModel(object):
	def __init__(self, model_file):
		self.model_file = model_file
		self.early_stop = config.cnn_early_stop
		if K.image_data_format() == 'channels_first': # 通道
			# print "="*20 + "channels_first" + "="*20 
			self.input_shape  = (config.channel, config.img_x, config.img_y)
		else:
			# print "="*20 + "channels_last" + "="*20 
			self.input_shape  = (config.img_x, config.img_y, config.channel)
		
	def build_model(self, input_dim):
		input1 = Input( shape=(input_dim,) )
		input2 = Input( shape=(config.img_x,config.img_y,1) )
		input3 = Input( shape=(config.img_x,config.img_y,1) )

		# 一维特征深度网络 
		x1 = Dense(32, activation='relu')(input1)
		x1 = Dropout(0.15)(x1)
		x1 = Dense(8, activation='relu')(x1)

		# s/n 矩阵特征卷积网络 
		x2 = Conv2D(2, (3, 3), padding='same')(input2) # 62*62*8
		x2 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(x2) # 31*31*8
		x2 = Conv2D(4, (3, 3), padding='same')(x2) # 29*29*16
		x2 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(x2) # 15*15*16
		x2 = Conv2D(4, (3, 3), padding='same')(x2) # 13*13*32
		x2 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(x2) # 7*7*32
		x2 = Conv2D(16, (3, 3), padding='same')(x2) # 5*5*32
		x2 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(x2) # 3*3*32
		x2 = Flatten()(x2)
		x2 = Dense(32, activation='relu')(x2)
		x2 = Dropout(0.15)(x2)
		x2 = Dense(8, activation='relu')(x2)

		# s/n 矩阵特征卷积网络 
		x3 = Conv2D(2, (3, 3), padding='same')(input3) # 62*62*8
		x3 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(x3) # 31*31*8
		x3 = Conv2D(4, (3, 3), padding='same')(x3) # 29*29*16
		x3 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(x3) # 15*15*16
		x3 = Conv2D(4, (3, 3), padding='same')(x3) # 13*13*32
		x3 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(x3) # 7*7*32
		x3 = Conv2D(16, (3, 3), padding='same')(x3) # 5*5*32
		x3 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(x3) # 3*3*16
		x3 = Flatten()(x3)
		x3 = Dense(32, activation='relu')(x3)
		x3 = Dropout(0.15)(x3)
		x3 = Dense(8, activation='relu')(x3)

		top_tensor = concatenate([x1,x2], axis=1)
		top_tensor = Dense(8, activation='relu')(top_tensor)
		top_tensor = Dropout(0.15)(top_tensor)
		top_tensor = Dense(2, activation='softmax')(top_tensor)

		self.model = Model(inputs=[input1,input2,input3], outputs=top_tensor)
		# self.model = Model(inputs=[ input1], outputs=top_tensor)
		sgd = keras.optimizers.SGD(lr=0.001, momentum=0.85, decay=1e-4, nesterov=False)
		adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
		self.model.compile( loss="categorical_crossentropy", optimizer = adam, metrics=["acc"])

	# 模型设置
	def fit_model(self, train_x, train_y, vali_x, vali_y):
		self.model.fit(
			train_x, 
			train_y, 
			batch_size = config.batch_size, 
			epochs = config.epochs,
			validation_data = (vali_x, vali_y),
			verbose = 2,
			callbacks = [
				# monitor=acc时 mode=max, patience是多少轮性能不好就停止
				EarlyStopping(monitor='val_acc', patience=self.early_stop, verbose=1 ),
				# ModelCheckpoint(filepath=self.model_file,monitor='val_acc',save_best_only=True),
			],
		)
		# self.model = load_model(self.model_file)
	
	# 训练整个模型
	def train_model( self, train_x, train_y, vali_x, vali_y ):
		print "train_model in MedlatModel ......"
		print "train pos num = %d, train nag num = %d"%(np.sum(train_y[:,1]),np.sum(train_y[:,0]))
		print "vali pos num = %d, vali nag num = %d"%(np.sum(vali_y[:,1]),np.sum(vali_y[:,0]))
		if os.path.exists(self.model_file): os.remove(self.model_file); print "Remove file %s."%(self.model_file)
		if os.path.exists(self.model_file): 
			self.model = load_model(self.model_file)
		else:
			self.build_model( input_dim = len(train_x[0][0]) )
			self.fit_model( train_x, train_y, vali_x, vali_y )
			self.model.save(self.model_file)

	# 评估模型
	def evaluate(self, data_x, data_y):
		pred_y = self.model.predict(data_x)
		pred = np.argmax(pred_y, axis=1) # 求每行最大值索引
		data_y = np.argmax(data_y, axis=1)
		confuse_matrix = metrics.confusion_matrix(data_y, pred)
		TN, FP, FN, TP = confuse_matrix.ravel()
		print "confusion matrix: \n",confuse_matrix
		print "accuracy score: ",metrics.accuracy_score(data_y, pred)
		print "classification report:\n",metrics.classification_report(data_y, pred)
		print "F1 score: ",metrics.f1_score(data_y, pred)
		print "recall: ", 1.* TP / (TP + FN)
		print "False positive rate: ", 1.*FP / (FP + TN)
		return data_y, pred_y
 
# 加载数据模块
class DataLoader(object):
	def __init__(self, pos_path, nag_path):
		# pos_path: 正样本phcx文件群所在的目录
		# nag_path: 负样本phcx文件群所在的目录
		self.pos_path = pos_path
		self.nag_path = nag_path
		# 将信噪比矩阵转化为单通道图像矩阵处理
		self.img_x = config.img_x
		self.img_y = config.img_y
		self.split_point = [] # 特征分割点

	def load_data(self,need_shuffle=True):
		feats_df, sn_mtx_feats, int_mtx_feats = get_all_samples_feats()
		sn_mtx_feats = np.array(sn_mtx_feats)
		int_mtx_feats = np.array(int_mtx_feats)
		sn_mtx_feats  =  sn_mtx_feats.reshape((-1,self.img_x*self.img_y))
		int_mtx_feats = int_mtx_feats.reshape((-1,self.img_x*self.img_y))
		label = feats_df["label"].values
		feats_df = feats_df.replace(np.nan,0)
		ary_feats = feats_df.ix[:,1:].values
		ary_feats = my_normalization(ary_feats)
		data = np.concatenate([ary_feats, sn_mtx_feats, int_mtx_feats],axis=1)
		self.split_point = [len(ary_feats[0]), self.img_x*self.img_y+len(ary_feats[0])]
		if need_shuffle: idx = shuffle( range(len(label)) )
		return data[idx], label[idx]

	def get_inputs_set(self, data):
		input1 = data[:,:self.split_point[0]]
		input2 = data[:,self.split_point[0]:self.split_point[1]].reshape((-1,self.img_x,self.img_y,1))
		input3 = data[:,self.split_point[1]:].reshape((-1,self.img_x,self.img_y,1))
		# return [ input2, input3 ]
		return [ input1, input2, input3 ]

# 爬取phcx数据接口
class Candidate(object):
    def __init__(self, fname):
        """ Build a new Candidate object from a PHCX file path.
        """
        xmlroot = ET.parse(fname).getroot()
        # Read Coordinates
        coordNode = xmlroot.find('head').find('Coordinate')
        self.rajd = float(coordNode.find('RA').text)
        self.decjd = float(coordNode.find('Dec').text)
        # Separate PDMP & FFT sections
        for section in xmlroot.findall('Section'):
            if 'pdmp' in section.get('name').lower():
                opt_section = section
            else:
                fft_section = section
        # Best values as returned by PDMP
        opt_values = {
            node.tag : float(node.text)
            for node in opt_section.find('BestValues').getchildren()
            }
        
        self.bary_period = opt_values['BaryPeriod']
        self.topo_period = opt_values['TopoPeriod']
        self.dm = opt_values['Dm']
        self.snr = opt_values['Snr']
        self.width = opt_values['Width']
        
        ##### P-DM plane #####
        pdmNode = opt_section.find('SnrBlock')
        # DmIndex
        string = pdmNode.find('DmIndex').text
        dm_index = np.asarray(map(float, string.split()))
        
        # PeriodIndex
        string = pdmNode.find('PeriodIndex').text
        period_index = np.asarray(map(float, string.split()))
        period_index /= 1.0e12 # Picoseconds to seconds

        # S/N data
        pdmDataNode = pdmNode.find('DataBlock')
        pdm_plane = self.readDataBlock(pdmDataNode).reshape(
            dm_index.size, 
            period_index.size
            )
        
        # Pack all P-DM plane arrays into a tuple
        self.pdm_plane = (period_index, dm_index, pdm_plane)
        
        ### Sub-Integrations
        subintsNode = opt_section.find('SubIntegrations')
        nsubs = int(subintsNode.get('nSub'))
        nbins = int(subintsNode.get('nBins'))
        self.subints = self.readDataBlock(subintsNode).reshape(nsubs, nbins)
        
        ### Sub-Bands
        subbandsNode = opt_section.find('SubBands')
        nsubs = int(subbandsNode.get('nSub'))
        nbins = int(subbandsNode.get('nBins'))
        self.subbands = self.readDataBlock(subbandsNode).reshape(nsubs, nbins)
        
        ### Profile
        profileNode = opt_section.find('Profile')
        self.profile = self.readDataBlock(profileNode)
        
        ##### Parse FFT Section (PEASOUP Data) #####
        fft_values = {
            node.tag : float(node.text)
            for node in fft_section.find('BestValues').getchildren()
            }
        self.accn = fft_values['Accn']
        self.hits = fft_values['Hits']
        self.rank = fft_values['Rank']
        self.fftsnr = fft_values['SpectralSnr']
        
        ### DmCurve: FFT S/N vs. PEASOUP Trial DM, at best candidate acceleration
        dmcurve_node = fft_section.find('DmCurve')
        
        text = dmcurve_node.find('DmValues').text
        dm_values = np.asarray(map(float, text.split()))
        text = dmcurve_node.find('SnrValues').text
        snr_values = np.asarray(map(float, text.split()))
        
        # Pack the DM curve into a tuple of arrays
        self.dm_curve = (dm_values, snr_values)
        
        ### AccnCurve: FFT S/N vs. PEASOUP Trial Acc, at best candidate DM
        accncurve_node = fft_section.find('AccnCurve')
        
        text = accncurve_node.find('AccnValues').text
        accn_values = np.asarray( map(float,text.split()) )
        text = accncurve_node.find('SnrValues').text
        snr_values = np.asarray( map(float,text.split()) )
        
        # Pack the Accn curve into a tuple of arrays
        self.accn_curve = (accn_values, snr_values)

    # 读取phcx区块
    def readDataBlock( self, xmlnode ):
        """ Turn any 'DataBlock' XML node into a np array of floats """
        vmin = float(xmlnode.get('min'))
        vmax = float(xmlnode.get('max'))
        string = xmlnode.text
        string = re.sub("[\t\s\n]", "", string)
        data = np.asarray( bytearray.fromhex(string), dtype = float, )
        return data * (vmax - vmin) / 255. + vmin

if __name__ == '__main__':
	loader = DataLoader(config.pos_path,config.nag_path)
	loader.load_data()

