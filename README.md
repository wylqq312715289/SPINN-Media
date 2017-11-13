负样本个数RFI：     89996
正样本个数pulsars： 1196

运行环境要求：
1、包含有GPU环境(显存6G以上) 显存不够可以调节config.py文件中的config.batch_size值调小即可
  没有GPU的情况下 跑keras或者tensorflow速度会相当慢（慎重）
2、内存：16G以上 若爆内存，可以去掉一些负样本,在第三步之前调节config.py文件中的config.nag_num,否则重新执行第三步。
3、硬盘要求1G以上即可


依赖包
tensorflow ( pip install --upgrade --ignore-installed tensorflow ) GPU版本请参考网络
keras ( pip install keras ) 
h5py
cv2 ( pip install opencv-python )


第一步: 将 HTRU Medlat Training Data 数据集解压到文件夹data下 构造成如下的目录结构
   	--data|--MedlatTrainingData|--pulsars|--pulsar_0000.phcx
                                         |--pulsar_0001.phcx
                                         |    ...(1196)...
                               |--RFI    |--cand_000001.phcx
                               	         |--cand_000002.phcx
                               	         |    ...(89996)...
    剩下的目录结构系统自动创建。


第二步: 修改配置文件
	配置文件为./config.py，在这里可以修改模型参数以及各中间文件路径
  该文件属于配置文件，在任何程序调用该配置文件时会在目录./cache/下自动生成samples_info.csv文件，samples_info.csv文件内保存phcx文件路径和该phcx文件对应的label值(1.0就是pulser,0.0就是noise)


第三步: 将所有特征规范化并写入文件
  这个需要运行一段时间 大约10分钟不到
  执行feats_and_plot.py下的get_all_samples_feats方法将各特征持久化到文件内
  并在目录./cache/下生成ary_feats.csv文件，内保存数值特征（未规范化,但是模型运行时会规范化）
  在目录./cache/h5/目录下生成矩阵特征(sn,int)的文件,方便后期模型快速调用
  如果需要修改特征，请在feats_and_plot.py文件下的get_feats函数内修改


第四步：分析特征-并画图展示
  画图文件是./feats_and_plot.py,展示相应的输出图像，在该文件中最后的"if __name__ == '__main__': "内去掉相应的注释 plot1,plot2,plot3,... 即可查看相应的plot图
  数值特征分为基础特征5个("snr","W_eq","log(p/dm)","V_dm","D_rms") + 后来添加的特征8个("p_0","p_1","p_2","p_3","d_0","d_1","d_2","d_3")


第五步：运行主程序
	python main.py
	一个样本的特征为: 矩阵 + 向量 的形式，矩阵跑CNN网络，向量跑DNN网络，最后这两个网络并联为一个dense,最后一层为softmax输出是pulser或RFI的概率值(二维,和为1.0)。针对模型的输出给定相应的评估办法。
  在目录./cache/下生成评估文件evaluate.csv

	
第六步：模型评估
  执行feats_and_plot.py下的plot_evaluate生成模型评估曲线图


总结
该程序中，实际数值特征效果并不好。论文和数据相差甚远。经过人为观察，我们小组确定使用sn矩阵做cnn。果不其然cnn效果远大于数值特征的效果。需求方若对这几个数值特征有研究，并与数据集参数研究出了特定的关系与规范化方法，在feats_and_plot.py文件下的get_feats函数内修改即可。


我们模型最后一层使用的是softmax区间映射为[0,1] 并非论文中的tanh[-1,1] 因为softmax更先进，你可以查阅相关资料

修改modules.py 中的100行左右的verbose参数=2 可以不显示训练模型时的进度条（注释后显示进度条）

分五折查看最终效果如下
样本集 = [折1,折2,折3,折4,折5]

[折1,折2,折3,折4] = 训练集   [折5] = 验证集
[折1,折2,折3,折5] = 训练集   [折4] = 验证集
[折1,折2,折4,折5] = 训练集   [折3] = 验证集
[折1,折3,折4,折5] = 训练集   [折2] = 验证集
[折2,折3,折4,折5] = 训练集   [折1] = 验证集

5次模型的验证集整合到一起就是下面的混淆矩阵的效果。我们只过采样训练集，验证集保持原来的正负样本比例
confusion matrix: 
[[89879   117]
 [   77  1119]]
accuracy score:  0.997872620405
classification report:
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     89996
          1       0.91      0.94      0.92      1196

avg / total       1.00      1.00      1.00     91192

F1 score:  0.920230263158
recall:  0.935618729097
False positive rate:  0.00130005778035


# 得出SPINN-1658页 Table.1 
我们模型最后一层使用的是softmax区间映射为[0,1] 并非论文中的tanh[-1,1] 因为softmax更先进，你可以查阅相关资料
score threshold:  0.175  recall:  0.945652173913  False positive rate:  0.00193341926308
score threshold:  0.6    recall:  0.933110367893  False positive rate:  0.00115560691586
score threshold:  0.76   recall:  0.92474916388  False positive rate:  0.000944486421619
score threshold:  0.93   recall:  0.909698996656  False positive rate:  0.000644473087693












