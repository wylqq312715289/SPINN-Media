#-*- coding:utf-8 -*-
import re, os, copy, math, cv2
import xml.etree.cElementTree as ET
import numpy as np
import pandas as pd
np.random.seed(2017) # 固定随机种子
import cv2,h5py
from sklearn.utils import shuffle
from sklearn import preprocessing
import pylab as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

from config import config
from utils import *
# from modules import DataLoader, get_feat, Candidate # 本地测试机器win+2.7无法安装tensor 需要升级3.5

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

# 获取单个样本特征
def get_feats( cand ):
    """ 注：
        非天文专业，研究了很久物理概念,但还是不太清楚. 
        SPINN 论文中的dm理解为 best_dm 若非best_dm，则在此修改best_dm值.
        对于特征6中的 pulse window 我们无法获得或计算，如果你们会计算，请将特征6计算好添加到特征尾部
    """
    p, dm, snr = cand.pdm_plane #p为period映射向量，dm为映射向量向量，snr对应信噪比矩阵。
    best_dm = cand.dm
    best_snr = cand.snr
    profile = cand.profile
    sub_integrations_mtx = cand.subints
    int_num, phase_bin_num = sub_integrations_mtx.shape
    bands_num, phase_bin_num = cand.subbands.shape
    ary_feats = [] # 保存该样本特征向量 ([特征1,特征2,...])
    ary_feats_name = [] # 保存该样本特征向量名称，为后期画图提供便利
    mtx_feats = [] # 保存该样本特征矩阵 ([特征矩阵1,特征矩阵2,...])
    ######################## 是s/n特征矩阵  归一化办法是自己设计的 跑CNN ########################
    feat1_mtx = snr # 获取snr特征 p, dm, snr = cand.pdm_plane
    img_x, img_y = config.img_x, config.img_y# 矩阵规范化
    feat1_mtx = cv2.resize(feat1_mtx, (img_x,img_y), interpolation=cv2.INTER_LINEAR)
    mtx_feats.append( my_normalization(feat1_mtx.reshape(-1,1),0).reshape(img_x,img_y) )
    # 后根据vincent添加的实数特征  加入到1D向量中跑DNN
    ary_feats.append( math.log(best_snr) )  # s/n 特征后期需要归一化 
    ary_feats_name.append( "snr" )
    ary_feats.append( cand.width ); 
    ary_feats_name.append( "W_eq" )
    ######################## log(10,p/dm) p是best_dm对应的period #################################
    idx_tuple = np.where(snr==np.max(snr)) 
    max_z_p_idx =  idx_tuple[1][0]
    max_z_p = p[max_z_p_idx]
    ary_feats.append( math.log(max(1.0*max_z_p/best_dm*1e6,1e-3))  )  # s变ns
    ary_feats_name.append( "log(p/dm)" )
    ######################## V_dm  tanh(dm-dm_min) #############################
    ary_feats.append( math.tanh(best_dm - 2.0) )
    ary_feats_name.append( "V_dm" )
    ######################## lambda(s) ################################################################ 
    b = 2.0*8.0/math.sqrt(1.0*bands_num)
    lambda_s = copy.deepcopy(sub_integrations_mtx)
    lambda_s[ lambda_s >= 0 ] = 1.0 - np.exp(-1.0*lambda_s[ lambda_s >= 0 ]/b)
    lambda_s[ lambda_s < 0 ] = 1.0*lambda_s[ lambda_s < 0 ] / b
    lambda_s = cv2.resize(lambda_s, (img_x,img_y), interpolation=cv2.INTER_LINEAR)
    mtx_feats.append( my_normalization(lambda_s.reshape(-1,1),0).reshape(img_x,img_y) )
    ######################## D_rms 需要限制i属于W，请自行修改 ########################################
    w = int_num # pulse window 宽度买家一直未提供, 所以使用integration_num作为宽度
    profile_ = preprocessing.scale(profile) # 标准化
    D_rms = np.sum((profile_ - sub_integrations_mtx)**2 )*1.0 / (w*bands_num)
    ary_feats.append( math.sqrt(D_rms) )
    ary_feats_name.append( "D_rms" )
    ######################## 最后外加的8维特征 ############################
    p_ = np.mean(profile)
    p2_ = np.mean((profile-p_)**2)
    p3_ = np.mean((profile-p_)**3)
    p4_ = np.mean((profile-p_)**4)

    ary_feats.append( p_ )
    ary_feats.append( math.sqrt( np.sum( (profile-p_)**2 ) / (1.0*len(profile)-1.0) ) )
    ary_feats.append( p4_/(p2_**2) - 3.0 )
    ary_feats.append( p3_/(math.sqrt(p2_)**3) )
    ary_feats_name.extend( ["p_0","p_1","p_2","p_3"] )

    d,_ = cand.dm_curve
    d_ = np.mean(d)
    d2_ = np.mean((d-d_)**2)
    d3_ = np.mean((d-d_)**3)
    d4_ = np.mean((d-d_)**4)

    ary_feats.append( d_ )
    ary_feats.append( math.sqrt( np.sum( (d-d_)**2 ) / (1.0*len(d)-1.0) ) )
    ary_feats.append( d4_/(d2_**2) - 3.0 )
    ary_feats.append( d3_/(math.sqrt(d2_)**3) )
    ary_feats_name.extend( ["d_0","d_1","d_2","d_3"] )

    return np.array(ary_feats), ary_feats_name, mtx_feats

# phase相位相关图
def phase_plots(cand):
    import pylab as plt
    plt.figure(1, figsize=(9, 7), dpi=70, facecolor="#FFFFFF")
    plt.subplot(311)
    print "cand.subbands",cand.subbands.shape, np.min(cand.subbands),np.max(cand.subbands) # ( 16L, 64L )
    plt.imshow(cand.subbands, origin='lower', interpolation='nearest', aspect='auto', cmap=plt.cm.Greys)
    plt.title('Sub-Bands')
    plt.ylabel('Band Index')
    
    plt.subplot(312)
    print "cand.subints", cand.subints.shape,np.min(cand.subints),np.max(cand.subints) # ( 19L, 64L )
    plt.imshow(cand.subints, origin='lower', interpolation='nearest', aspect='auto', cmap=plt.cm.Greys)
    plt.title('Sub-Integrations')
    plt.ylabel('Integration Index')
    
    plt.subplot(313)
    print "cand.profile",cand.profile.shape,np.min(cand.profile),np.max(cand.profile)# ( 64L,)
    plt.bar(xrange(cand.profile.size), cand.profile, width=1)
    plt.xlim(0, cand.profile.size)
    plt.xlabel('Phase Bin Index')
    
    plt.tight_layout()
    plt.show()

# 显示s/n信息 信噪比矩阵
def bullseye_plot(cand):
    p, dm, snr = cand.pdm_plane
    # p是 period
    # snr = cv2.resize(snr, (64,64), interpolation=cv2.INTER_LINEAR)
    # snr = sub_normalize(snr)
    print p.shape,dm.shape,snr.shape
    print np.min(p),np.max(p)
    print np.min(dm),np.max(dm)
    print np.min(snr),np.max(snr)
    plt.figure(2, figsize=(7, 5), dpi=80, facecolor="#FFFFFF")
    # IMPORTANT NOTE: imshow() must be called with origin='lower' here, otherwise 
    # the DM values on the Y axis are reversed (and therefore wrong).
    plt.imshow(
        snr, 
        extent=[p.min(), p.max(), dm.min(), dm.max()], # x和y坐标值是(idx 还是[min,max])
        aspect='auto', 
        origin='lower',
        interpolation='nearest'
        )

    plt.xlabel('Period Correction (s)') # 与p对应
    plt.ylabel('Trial DM') # 与dm对应
    
    cb = plt.colorbar()
    cb.set_label(u'Folded S/N(color-bar)')
    
    plt.tight_layout()
    plt.show()

# 测试单个样本
def test_single_samples( file_name ):
	# Load example.phcx file (must be in the same directory as this python script)
    directory, fname = os.path.split( os.path.abspath(__file__) )
    cand = Candidate( os.path.join(directory, file_name ))
    print cand.snr
    print cand.topo_period
    print cand.bary_period
    print cand.width
    print cand.dm
    print cand.accn
    print cand.rajd
    print cand.decjd
    print cand.fftsnr
    print cand.profile.shape
    print cand.subints.shape
    print cand.subbands.shape
    print cand.dm_curve[0].shape, cand.dm_curve[1].shape
    print cand.accn_curve[0].shape, cand.accn_curve[1].shape
    print cand.rank
    print cand.hits
    print cand.pdm_plane[0].shape,cand.pdm_plane[1].shape,cand.pdm_plane[2].shape
    print cand.pdm_plane[0]
    # Make some cool plots
    phase_plots(cand)
    bullseye_plot(cand)

# 获取所有候选集特征( 数值特征和矩阵特征 )
def get_all_samples_feats():
    # if os.path.exists(config.ary_feats_file):  os.remove(config.ary_feats_file)
    if os.path.exists(config.ary_feats_file):
        feats_df = pd.read_csv(config.ary_feats_file)
        sn_mtx_feats = readH5( config.sn_mtx_feats_file )
        int_mtx_feats = readH5( config.int_mtx_feats_file )
        return feats_df, sn_mtx_feats, int_mtx_feats

    info_df = pd.read_csv(config.samples_info_file)
    info_df = shuffle(info_df).reset_index(drop=True)
    samples_feats = []; sn_mtx_feats = []; int_mtx_feats = []
    for i in range(len(info_df.index)):
        # if i >= 100: break;
        file_path = info_df.ix[i,0]
        label = info_df.ix[i,'label']
        cand = Candidate( file_path )
        feat, feats_name, mtx_list = get_feats(cand)
        samples_feats.append( [label] + list(feat) )
        sn_mtx_feats.append( mtx_list[0] )
        int_mtx_feats.append( mtx_list[1] )
        if i % 5000 == 0: print "finished [%d/%d]."%(i,len(info_df.index))

    feats_df = pd.DataFrame (samples_feats, columns=["label"]+feats_name )
    feats_df.to_csv(config.ary_feats_file,index=False,index_label=False)
    write2H5(config.sn_mtx_feats_file,sn_mtx_feats)
    write2H5(config.int_mtx_feats_file, int_mtx_feats)
    return feats_df, sn_mtx_feats, int_mtx_feats

# 画图Log(Period/MD) # D_eq 没有 用W_eq来代替，我们也是用心良苦
def plot1():
    feats_df = pd.read_csv(config.ary_feats_file)
    feats_df = feats_df[["label","log(p/dm)","W_eq"]]

    plt.figure(1, figsize=(14,11), dpi=70, facecolor="#FFFFFF")
    plt.subplot(311)
    # print len(feats_df.index)
    # print feats_df[["W_eq"]].describe()
    # print feats_df.ix[:10,"W_eq"]
    # tmp_df = feats_df[["W_eq"]].drop_duplicates(["W_eq"])
    # print len(tmp_df.index)

    pos_samples = feats_df[["log(p/dm)","W_eq"]][feats_df["label"]==1.0].values
    nag_samples = feats_df[["log(p/dm)","W_eq"]][feats_df["label"]==0.0].values
    
    p_x = pos_samples[:,0]
    p_y = pos_samples[:,1]
    n_x = nag_samples[:,0]
    n_y = nag_samples[:,1]

    plt.plot(p_x, p_y, 'r.', label='pulsers')
    plt.plot(n_x, n_y, 'b+', label='RFI' )
    # plt.ylim(0.2968,0.2969)
    plt.xlabel('Log(Period/DM)')
    plt.ylabel('W_eq')
    plt.legend()

    plt.subplot(312)
    plt.plot(p_x, p_y, 'r.', label='pulsers')
    # plt.ylim(0.2968,0.2969)
    plt.xlabel('Log(Period/DM)')
    plt.ylabel('W_eq')

    plt.subplot(313)
    plt.plot(n_x, n_y, 'b+', label='RFI' )
    # plt.ylim(0.2968,0.2969)
    plt.xlabel('Log(Period/DM)')
    plt.ylabel('W_eq')

    plt.show()

# 画图需求方后添加的8个公式中的前4个    普通点图
def plot2():
    feats_df = pd.read_csv(config.ary_feats_file)
    feats_df = feats_df[["label","p_0","p_1","p_2","p_3"]]

    plt.figure(1, figsize=(14,11), dpi=70, facecolor="#FFFFFF")
    plt.subplot(331)
    pos_samples = feats_df[["p_0","p_1"]][feats_df["label"]==1.0].values
    nag_samples = feats_df[["p_0","p_1"]][feats_df["label"]==0.0].values
    p_x1 = pos_samples[:,0]
    p_y1 = pos_samples[:,1]
    n_x1 = nag_samples[:,0]
    n_y1 = nag_samples[:,1]
    plt.plot(p_x1, p_y1, 'r.', label='pulsers')
    plt.plot(n_x1, n_y1, 'b+', label='RFI' )
    # plt.ylim(0.2968,0.2969)
    plt.xlabel('mean(p)')
    plt.ylabel('p_1')
    plt.legend()

    plt.subplot(332)
    plt.plot(p_x1, p_y1, 'r.',)
    # plt.ylim(0.2968,0.2969)
    plt.xlabel('mean(p)')
    plt.ylabel('p_1')

    plt.subplot(333)
    plt.plot(n_x1, n_y1, 'b+',)
    # plt.ylim(0.2968,0.2969)
    plt.xlabel('mean(p)')
    plt.ylabel('p_1')


    plt.subplot(334)
    pos_samples = feats_df[["p_0","p_2"]][feats_df["label"]==1.0].values
    nag_samples = feats_df[["p_0","p_2"]][feats_df["label"]==0.0].values
    p_x2 = pos_samples[:,0]
    p_y2 = pos_samples[:,1]
    n_x2 = nag_samples[:,0]
    n_y2 = nag_samples[:,1]
    plt.plot(p_x2, p_y2, 'r.', n_x2, n_y2, 'bx',)
    # plt.ylim(0.2968,0.2969)
    plt.xlabel('mean(p)')
    plt.ylabel('p_2')

    plt.subplot(335)
    plt.plot(p_x2, p_y2, 'r.',)
    # plt.ylim(0.2968,0.2969)
    plt.xlabel('mean(p)')
    plt.ylabel('p_2')

    plt.subplot(336)
    plt.plot(n_x2, n_y2, 'b+',)
    # plt.ylim(0.2968,0.2969)
    plt.xlabel('mean(p)')
    plt.ylabel('p_2')


    plt.subplot(337)
    pos_samples = feats_df[["p_0","p_3"]][feats_df["label"]==1.0].values
    nag_samples = feats_df[["p_0","p_3"]][feats_df["label"]==0.0].values
    p_x3 = pos_samples[:,0]
    p_y3 = pos_samples[:,1]
    n_x3 = nag_samples[:,0]
    n_y3 = nag_samples[:,1]
    plt.plot(p_x3, p_y3, 'r.', n_x3, n_y3, 'bx',)
    # plt.ylim(0.2968,0.2969)
    plt.xlabel('mean(p)')
    plt.ylabel('p_3')

    plt.subplot(338)
    plt.plot(p_x3, p_y3, 'r.',)
    # plt.ylim(0.2968,0.2969)
    plt.xlabel('mean(p)')
    plt.ylabel('p_3')

    plt.subplot(339)
    plt.plot(n_x3, n_y3, 'b+',)
    # plt.ylim(0.2968,0.2969)
    plt.xlabel('mean(p)')
    plt.ylabel('p_3')

    plt.show()

# 画图需求方后添加的8个公式中的后4个    半透明点图
def plot3():
    feats_df = pd.read_csv(config.ary_feats_file)
    feats_df = feats_df[["label","d_0","d_1","d_2","d_3"]]

    plt.figure(1, figsize=(14,10), dpi=70, facecolor="#FFFFFF")
    plt.subplot(331)
    pos_samples = feats_df[["d_0","d_1"]][feats_df["label"]==1.0].values
    nag_samples = feats_df[["d_0","d_1"]][feats_df["label"]==0.0].values
    p_x1 = pos_samples[:,0]
    p_y1 = pos_samples[:,1]
    n_x1 = nag_samples[:,0]
    n_y1 = nag_samples[:,1]
    plt.scatter(p_x1,p_y1, s=75, alpha=.1, c="red")
    plt.scatter(n_x1,n_y1, s=75, alpha=.1, c="blue")
    # plt.ylim(0.2968,0.2969)
    plt.xlabel('mean(d)')
    plt.ylabel('d_1')
    plt.legend()

    plt.subplot(332)
    # plt.plot(p_x1, p_y1, 'r.',)
    plt.scatter(p_x1,p_y1, s=75, alpha=.1, c="red")
    # plt.ylim(0.2968,0.2969)
    plt.xlabel('mean(d)')
    plt.ylabel('d_1')

    plt.subplot(333)
    plt.scatter(n_x1,n_y1, s=75, alpha=.1, c="blue")
    # plt.ylim(0.2968,0.2969)
    plt.xlabel('mean(d)')
    plt.ylabel('d_1')


    plt.subplot(334)
    pos_samples = feats_df[["d_0","d_2"]][feats_df["label"]==1.0].values
    nag_samples = feats_df[["d_0","d_2"]][feats_df["label"]==0.0].values
    p_x2 = pos_samples[:,0]
    p_y2 = pos_samples[:,1]
    n_x2 = nag_samples[:,0]
    n_y2 = nag_samples[:,1]
    plt.scatter(p_x2,p_y2, s=75, alpha=.1, c="red")
    plt.scatter(n_x2,n_y2, s=75, alpha=.1, c="blue")
    # plt.ylim(0.2968,0.2969)
    plt.xlabel('mean(d)')
    plt.ylabel('d_2')

    plt.subplot(335)
    plt.scatter(p_x2,p_y2, s=75, alpha=.1, c="red")
    # plt.ylim(0.2968,0.2969)
    plt.xlabel('mean(d)')
    plt.ylabel('d_2')

    plt.subplot(336)
    plt.scatter(n_x2,n_y2, s=75, alpha=.1, c="blue")
    # plt.ylim(0.2968,0.2969)
    plt.xlabel('mean(d)')
    plt.ylabel('d_2')


    plt.subplot(337)
    pos_samples = feats_df[["d_0","d_3"]][feats_df["label"]==1.0].values
    nag_samples = feats_df[["d_0","d_3"]][feats_df["label"]==0.0].values
    p_x3 = pos_samples[:,0]
    p_y3 = pos_samples[:,1]
    n_x3 = nag_samples[:,0]
    n_y3 = nag_samples[:,1]
    plt.scatter(p_x3,p_y3, s=75, alpha=.2, c="red")
    plt.scatter(n_x3,n_y3, s=75, alpha=.2, c="blue")
    # plt.ylim(0.2968,0.2969)
    plt.xlabel('mean(d)')
    plt.ylabel('d_3')

    plt.subplot(338)
    plt.scatter(p_x3,p_y3, s=75, alpha=.2, c="red")
    # plt.ylim(0.2968,0.2969)
    plt.xlabel('mean(d)')
    plt.ylabel('d_3')

    plt.subplot(339)
    plt.scatter(n_x3,n_y3, s=75, alpha=.2, c="blue")
    # plt.ylim(0.2968,0.2969)
    plt.xlabel('mean(d)')
    plt.ylabel('d_3')

    plt.show()

# 画图各基础特征与label之间的柱状图
def plot4_1():
    main_columns = ["label","snr","W_eq","log(p/dm)","V_dm","D_rms"]
    feats_df = pd.read_csv(config.ary_feats_file)
    feats_df = feats_df[main_columns]
    plt.figure(1, figsize=(14,10), dpi=70, facecolor="#FFFFFF")
    plt.title("basic features")
    for subplot_id in range(1,len(main_columns)):
        plt.subplot(320+subplot_id)
        # 数据分桶
        column = [main_columns[subplot_id]]
        max_ = np.max(feats_df[column].values)
        min_ = np.min(feats_df[column].values)
        xlabel = np.arange(min_,max_,1.0*(max_-min_)/20) #分成20个桶
        fun = lambda x: data_buckets(x,xlabel)
        # p_y = [ 0.0 for i in range(len(xlabel)) ]
        # n_y = [ 0.0 for i in range(len(xlabel)) ]

        # fun = lambda x: x
        tmp_df = feats_df[column+["label"]]
        tmp_df[column[0]] = tmp_df[column[0]].map(fun)
        p_df = tmp_df[tmp_df["label"]==1.0].groupby(column,as_index=False).count().reset_index(drop=True)
        n_df = tmp_df[tmp_df["label"]==0.0].groupby(column,as_index=False).count().reset_index(drop=True)
        p_x = p_df[column[0]].values
        p_y = p_df["label"].values
        n_x = n_df[column[0]].values
        n_y = n_df["label"].values
        n_y /= np.sum(n_y)/np.sum(p_y) # 负样本太多，num比例规范到一个量纲下

        # plt.bar( p_x, p_y, label='pulsers')
        # plt.bar( n_x, n_y, label='RFI' )
        plt.bar(p_x, +p_y, alpha=.7, facecolor='lightskyblue', edgecolor='white', label='pulsers')
        plt.bar(n_x, +n_y, alpha=.7, facecolor='#ff9999', edgecolor='white', label='RFI')
        # plt.ylim(0.2968,0.2969)
        plt.xlabel('%s'%column[0])
        plt.ylabel('counts')
        plt.legend()
    plt.show()

# 画图各基础特征与label之间的半透明点图
# x轴为特征值，y轴为特征值所对应的label=0或1的数量
# 因为分成100个桶所以该函数运行的时间较慢大约10s生成特征图像。
def plot4_2():
    main_columns = ["label","snr","W_eq","log(p/dm)","V_dm","D_rms"]
    feats_df = pd.read_csv(config.ary_feats_file)
    feats_df = feats_df[main_columns]
    plt.figure(1, figsize=(14,12), dpi=70, facecolor="#FFFFFF")
    plt.title("basic features")
    for subplot_id in range(1,len(main_columns)):
        plt.subplot(320+subplot_id)
        column = [main_columns[subplot_id]]
        # 数据分桶
        max_ = np.max(feats_df[column].values)
        min_ = np.min(feats_df[column].values)
        xlabel = np.arange(min_,max_,1.0*(max_-min_)/50) #分成50个桶 
        fun = lambda x: data_buckets(x,xlabel)
        # p_y = [ 0.0 for i in range(len(xlabel)) ]
        # n_y = [ 0.0 for i in range(len(xlabel)) ]

        # fun = lambda x: x
        tmp_df = feats_df[column+["label"]]
        tmp_df[column[0]] = tmp_df[column[0]].map(fun)
        p_df = tmp_df[tmp_df["label"]==1.0].groupby(column,as_index=False).count().reset_index(drop=True)
        n_df = tmp_df[tmp_df["label"]==0.0].groupby(column,as_index=False).count().reset_index(drop=True)
        p_x = p_df[column[0]].values
        p_y = p_df["label"].values
        n_x = n_df[column[0]].values
        n_y = n_df["label"].values
        n_y /= np.sum(n_y)/np.sum(p_y) # 负样本太多，num比例规范到一个量纲下

        # plt.bar( p_x, p_y, label='pulsers')
        # plt.bar( n_x, n_y, label='RFI' )
        # plt.bar(p_x, +p_y, alpha=.7, facecolor='lightskyblue', edgecolor='white', label='pulsers')
        # plt.bar(n_x, +n_y, alpha=.7, facecolor='#ff9999', edgecolor='white', label='RFI')
        if subplot_id==1:
            plt.scatter(p_x,p_y, s=275, alpha=.5, c="red", label='pulsers')
            plt.scatter(n_x,n_y, s=275, alpha=.5, c="blue", label="RFI")
        else:
            plt.scatter(p_x,p_y, s=275, alpha=.5, c="red",)
            plt.scatter(n_x,n_y, s=275, alpha=.5, c="blue",)
        # plt.ylim(0.2968,0.2969)
        plt.xlabel('%s'%column[0])
        plt.ylabel('counts')
        plt.legend()
    plt.show()

# 画图各后加的8个特征与label之间的柱状图
def plot5_1():
    main_columns = ["label","p_0","p_1","p_2","p_3"]
    feats_df = pd.read_csv(config.ary_feats_file)
    feats_df = feats_df[main_columns]
    plt.figure(1, figsize=(14,10), dpi=70, facecolor="#FFFFFF")
    plt.title("basic features")
    for subplot_id in range(1,len(main_columns)):
        plt.subplot(220+subplot_id)
        # 数据分桶
        column = [main_columns[subplot_id]]
        feats_df[column[0]] = feats_df[column[0]].replace(np.nan,0.0)
        max_ = np.max(feats_df[column].values)
        min_ = np.min(feats_df[column].values)
        print min_, max_
        xlabel = np.arange(min_,max_,1.0*(max_-min_)/20) #分成20个桶
        fun = lambda x: data_buckets(x,xlabel)
        # p_y = [ 0.0 for i in range(len(xlabel)) ]
        # n_y = [ 0.0 for i in range(len(xlabel)) ]

        # fun = lambda x: x
        tmp_df = feats_df[column+["label"]]
        tmp_df[column[0]] = tmp_df[column[0]].map(fun)
        p_df = tmp_df[tmp_df["label"]==1.0].groupby(column,as_index=False).count().reset_index(drop=True)
        n_df = tmp_df[tmp_df["label"]==0.0].groupby(column,as_index=False).count().reset_index(drop=True)
        p_x = p_df[column[0]].values
        p_y = p_df["label"].values
        n_x = n_df[column[0]].values
        n_y = n_df["label"].values
        n_y /= np.sum(n_y)/np.sum(p_y) # 负样本太多，num比例规范到一个量纲下

        # plt.bar( p_x, p_y, label='pulsers')
        # plt.bar( n_x, n_y, label='RFI' )
        if subplot_id==1:
            plt.bar(p_x, +p_y, alpha=.7, facecolor='lightskyblue', edgecolor='white', label='pulsers count')
            plt.bar(n_x, +n_y, alpha=.7, facecolor='#ff9999', edgecolor='white', label='RFI count')
        else:
            plt.bar(p_x, +p_y, alpha=.7, facecolor='lightskyblue', edgecolor='white')
            plt.bar(n_x, +n_y, alpha=.7, facecolor='#ff9999', edgecolor='white')
        # plt.ylim(0.2968,0.2969)
        plt.xlabel('%s'%column[0])
        plt.ylabel('counts')
        plt.legend()
    plt.show()

# 画图各后加的8个特征与label之间的半透明点图
def plot5_2():
    main_columns = ["label","p_0","p_1","p_2","p_3"]
    feats_df = pd.read_csv(config.ary_feats_file)
    feats_df = feats_df[main_columns]
    plt.figure(1, figsize=(14,10), dpi=70, facecolor="#FFFFFF")
    plt.title("basic features")
    for subplot_id in range(1,len(main_columns)):
        plt.subplot(220+subplot_id)
        # 数据分桶
        column = [main_columns[subplot_id]]
        feats_df[column[0]] = feats_df[column[0]].replace(np.nan,0.0)
        max_ = np.max(feats_df[column].values)
        min_ = np.min(feats_df[column].values)
        print min_, max_
        xlabel = np.arange(min_,max_,1.0*(max_-min_)/50) #分成20个桶
        fun = lambda x: data_buckets(x,xlabel)
        # p_y = [ 0.0 for i in range(len(xlabel)) ]
        # n_y = [ 0.0 for i in range(len(xlabel)) ]

        # fun = lambda x: x
        tmp_df = feats_df[column+["label"]]
        tmp_df[column[0]] = tmp_df[column[0]].map(fun)
        p_df = tmp_df[tmp_df["label"]==1.0].groupby(column,as_index=False).count().reset_index(drop=True)
        n_df = tmp_df[tmp_df["label"]==0.0].groupby(column,as_index=False).count().reset_index(drop=True)
        p_x = p_df[column[0]].values
        p_y = p_df["label"].values
        n_x = n_df[column[0]].values
        n_y = n_df["label"].values
        n_y /= np.sum(n_y)/np.sum(p_y) # 负样本太多，num比例规范到一个量纲下

        # plt.bar( p_x, p_y, label='pulsers')
        # plt.bar( n_x, n_y, label='RFI' )
        if subplot_id==1:
            plt.scatter(p_x,p_y, s=75, alpha=.5, c="red", label='pulsers')
            plt.scatter(n_x,n_y, s=75, alpha=.5, c="blue", label="RFI")
        else:
            plt.scatter(p_x,p_y, s=75, alpha=.5, c="red",)
            plt.scatter(n_x,n_y, s=75, alpha=.5, c="blue",)
        # plt.ylim(0.2968,0.2969)
        plt.xlabel('%s'%column[0])
        plt.ylabel('counts')
        plt.legend()
    plt.show()

# 画图各后加的8个特征与label之间的柱状图
def plot6_1():
    main_columns = ["label","d_0","d_1","d_2","d_3"]
    feats_df = pd.read_csv(config.ary_feats_file)
    feats_df = feats_df[main_columns]
    plt.figure(1, figsize=(14,10), dpi=70, facecolor="#FFFFFF")
    plt.title("basic features")
    for subplot_id in range(1,len(main_columns)):
        plt.subplot(220+subplot_id)
        # 数据分桶
        column = [main_columns[subplot_id]]
        feats_df[column[0]] = feats_df[column[0]].replace(np.nan,0.0)
        max_ = np.max(feats_df[column].values)
        min_ = np.min(feats_df[column].values)
        print min_, max_
        xlabel = np.arange(min_,max_,1.0*(max_-min_)/20) #分成20个桶
        fun = lambda x: data_buckets(x,xlabel)
        # p_y = [ 0.0 for i in range(len(xlabel)) ]
        # n_y = [ 0.0 for i in range(len(xlabel)) ]

        # fun = lambda x: x
        tmp_df = feats_df[column+["label"]]
        # tmp_df[column[0]] = pd.Series( preprocessing.scale(tmp_df[column[0]].values) )
        tmp_df[column[0]] = tmp_df[column[0]].map(fun)
        p_df = tmp_df[tmp_df["label"]==1.0].groupby(column,as_index=False).count().reset_index(drop=True)
        n_df = tmp_df[tmp_df["label"]==0.0].groupby(column,as_index=False).count().reset_index(drop=True)
        p_x = p_df[column[0]].values
        p_y = p_df["label"].values
        n_x = n_df[column[0]].values
        n_y = n_df["label"].values
        n_y /= np.sum(n_y)/np.sum(p_y) # 负样本太多，num比例规范到一个量纲下

        # plt.bar( p_x, p_y, label='pulsers')
        # plt.bar( n_x, n_y, label='RFI' )
        if subplot_id==1:
            plt.bar(p_x, +p_y, alpha=.7, facecolor='lightskyblue', edgecolor='white', label='pulsers count')
            plt.bar(n_x, +n_y, alpha=.7, facecolor='#ff9999', edgecolor='white', label='RFI count')
        else:
            plt.bar(p_x, +p_y, alpha=.7, facecolor='lightskyblue', edgecolor='white')
            plt.bar(n_x, +n_y, alpha=.7, facecolor='#ff9999', edgecolor='white')
        # plt.ylim(0.2968,0.2969)
        xlabel = [ "%.1f"%k for k in xlabel ]
        plt.xticks(n_x, xlabel, rotation=45)
        plt.xlabel('%s'%column[0])
        plt.ylabel('counts')
        plt.legend()
    plt.show()

# 画图各后加的8个特征与label之间的半透明点图
def plot6_2():
    main_columns = ["label","d_0","d_1","d_2","d_3"]
    feats_df = pd.read_csv(config.ary_feats_file)
    feats_df = feats_df[main_columns]
    plt.figure(1, figsize=(14,10), dpi=70, facecolor="#FFFFFF")
    plt.title("basic features")
    for subplot_id in range(1,len(main_columns)):
        plt.subplot(220+subplot_id)
        # 数据分桶
        column = [main_columns[subplot_id]]
        feats_df[column[0]] = feats_df[column[0]].replace(np.nan,0.0)
        max_ = np.max(feats_df[column].values)
        min_ = np.min(feats_df[column].values)
        print min_, max_
        xlabel = np.arange(min_,max_,1.0*(max_-min_)/20) #分成20个桶
        fun = lambda x: data_buckets(x,xlabel)
        # p_y = [ 0.0 for i in range(len(xlabel)) ]
        # n_y = [ 0.0 for i in range(len(xlabel)) ]

        # fun = lambda x: x
        tmp_df = feats_df[column+["label"]]
        # tmp_df[column[0]] = pd.Series( preprocessing.scale(tmp_df[column[0]].values) )
        tmp_df[column[0]] = tmp_df[column[0]].map(fun)
        p_df = tmp_df[tmp_df["label"]==1.0].groupby(column,as_index=False).count().reset_index(drop=True)
        n_df = tmp_df[tmp_df["label"]==0.0].groupby(column,as_index=False).count().reset_index(drop=True)
        p_x = p_df[column[0]].values
        p_y = p_df["label"].values
        n_x = n_df[column[0]].values
        n_y = n_df["label"].values
        n_y /= np.sum(n_y)/np.sum(p_y) # 负样本太多，num比例规范到一个量纲下

        # plt.bar( p_x, p_y, label='pulsers')
        # plt.bar( n_x, n_y, label='RFI' )
        if subplot_id==1:
            plt.scatter(p_x,p_y, s=175, alpha=.2, c="red", label='pulsers')
            plt.scatter(n_x,n_y, s=175, alpha=.2, c="blue", label="RFI")
        else:
            plt.scatter(p_x,p_y, s=175, alpha=.2, c="red",)
            plt.scatter(n_x,n_y, s=175, alpha=.2, c="blue",)
        # plt.ylim(0.2968,0.2969)
        xlabel = [ "%.1f"%k for k in xlabel ]
        # plt.xticks(n_x, xlabel, rotation=45)
        plt.xlabel('%s'%column[0])
        plt.ylabel('counts')
        plt.legend()
    plt.show()

# main执行完毕后 才能运行该方法
# 执行前需要将./cache/下的evaluate.csv下载下来 或查看./cache/目录下是否存在evaluate.csv文件
def plot_evaluate():
    feats_df = pd.read_csv(config.evaluate_file)
    xlabel = np.arange(0.0,1.0,1.0/20)
    False_nagtive_rate = [ 0.0 for i in range(len(xlabel))]
    False_positive_rate = [ 0.0 for i in range(len(xlabel))]
    precision = [ 0.0 for i in range(len(xlabel))]
    recall = [ 0.0 for i in range(len(xlabel))]
    for i,my_threshold in enumerate(xlabel,0): # 分成1000份做评估
        fun = lambda x: 1.0 if x >= my_threshold else 0.0
        feats_df["pred"] = feats_df["prob_1"].map(fun)
        pred = feats_df["pred"].values
        real = feats_df["real"].values
        tn, fp, fn, tp = metrics.confusion_matrix( real, pred ).ravel()
        recall[i] = 1.0*tp / ( tp + fn )
        False_nagtive_rate[i] = 1.0 - recall[i]
        False_positive_rate[i] = 1.0*fp / ( tn + fp )
        precision[i] = 1.0*tp/(tp+fp)
        if i %20==0: print "[%d/%d]"%(i,len(xlabel))

    False_nagtive_rate = np.log(False_nagtive_rate)/math.log(10)
    False_positive_rate = np.log(False_positive_rate)/math.log(10)
    precision = np.log(precision)/math.log(10)
    recall = np.log(recall)/math.log(10)

    plt.figure(1, figsize=(14,10), dpi=70, facecolor="#FFFFFF")
    plt.subplot(121)
    plt.plot(xlabel,False_nagtive_rate,c="green",label="False_nagtive_rate")
    plt.plot(xlabel,False_positive_rate,c="red",label="False_positive_rate")
    # plt.yticks(False_nagtive_rate, [math.pow(10,x) for x in False_nagtive_rate] , rotation=45)
    plt.legend()
    plt.xlabel("MY model score threshold")
    plt.ylabel('rate (10^-y)')
    
    plt.subplot(122)
    plt.plot(xlabel,precision,c="black",label="precision")
    plt.plot(xlabel,recall,c="orange",label="recall")
    plt.ylim(-0.1,0.0)
    plt.legend()
    plt.xlabel("MY model score threshold")
    plt.ylabel('rate (10^-y)')
    plt.show()

# 得出SPINN-1658页 Table.1  在main执行完毕后才能执行该方法
# 执行前需要将./cache/下的evaluate.csv下载下来 或查看./cache/目录下是否存在evaluate.csv文件
def plot_scorethreshold():
    feats_df = pd.read_csv(config.evaluate_file)
    # 输出prob在0到1之间 不能跟论文中的[-0.65,0.20,0.52,0.86]作对比，我们进行了相应的变换
    threshold_list = np.array([-0.65,0.20,0.52,0.86])
    threshold_list = threshold_list*0.5 + 0.5;
    for threshold in threshold_list:
        tmp = feats_df.copy()
        fun = lambda x: 1 if x >= threshold else 0
        tmp["prob_1"] = tmp["prob_1"].map(fun)
        confuse_matrix = metrics.confusion_matrix(tmp["real"].values, tmp["prob_1"].values)
        TN, FP, FN, TP = confuse_matrix.ravel()
        print "score threshold: ", threshold,
        print " recall: ", metrics.recall_score(tmp["real"].values,tmp["prob_1"].values),
        print " False positive rate: ", 1.*FP / (FP + TN)


# main执行完毕后 才能运行该方法
def plotRoc():
    roc_df = pd.read_csv(config.evaluate_file)
    plt.figure(1, figsize=(9, 7), dpi=70, facecolor="#FFFFFF")
    a = np.argmax(roc_df[["prob_0","prob_1"]].values, axis=1)
    print a[:10]
    roc_df["pred"] = np.argmax(roc_df[["prob_0","prob_1"]].values, axis=1)
    fpr, tpr, thresholds = roc_curve(roc_df["real"].values, roc_df["pred"].values)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='%s (auc=%0.4f)'%("Our Model",roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6) )
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    # get_all_samples_feats() 
    # test_single_samples('./data/MedlatTrainingData/pulsars/pulsar_0229.phcx')
    # test_single_samples('./data/MedlatTrainingData/RFI/cand_000001.phcx')
    # plot1()
    # plot2()
    # plot3()
    # plot4_1()
    # plot4_2()
    # plot5_1()
    # plot5_2()
    # plot6_1()
    # plot6_2()
    # plot_evaluate()
    # plot_scorethreshold()
    plotRoc()



