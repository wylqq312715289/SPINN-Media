#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

a = pd.DataFrame(np.zeros((100,3)))
# print a

# import pylab as plt
# plt.imshow(np.zeros((5,10)),origin='lower', interpolation='nearest', aspect='auto', cmap=plt.cm.Greys)
# plt.show()

a = np.array([1,2,3,5])
b = np.array([[1,1,1,1],[2,2,2,9],[3,3,3,3]])
print b
print np.where(b==9)[0][0],np.where(b==9)[1][0]
# c = a - b 
# print c
# print c**2
# print np.sum((a - b)**2)

a = np.array([[1,2,3,5]])
b = np.array([[1,1,1,1],[2,2,2,9],[3,3,3,3]])
print np.concatenate((a,b),axis=0)


# main执行完毕后 才能运行该方法
def saveRoc():
    roc_df = pd.read_csv(config.evaluate_file)
    plt.figure(1, figsize=(9, 7), dpi=70, facecolor="#FFFFFF")
    a = np.argmax(roc_df[["prob_1"]].values, axis=1)
    print a[:10]
    roc_df["pred"] = np.argmax(roc_df[["prob_1"]].values, axis=1)
    fpr, tpr, thresholds = roc_curve(roc_df["real"].values, roc_df["pred"].values)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='%s (auc=%0.4f)'%("our model",roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6) )

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()

    # plt.savefig(config.roc_file)
a = shuffle(range(10))
print type(a)






