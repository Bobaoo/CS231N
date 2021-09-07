#! /usr/bin/env python
#encoding='UTF-8'

from cifar import Cifar
from knn_train import KNN

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    c = Cifar()
    root = "/home/baobao/cs231n-bao/assignment1/cifar-10-batches-py"
    Xtr , Ytr , Xte , Yte = c.all_batch(root)
    #print('训练数据：',Xtr.shape)
    #print('训练标签：',Ytr.shape)
    #print('测试数据：',Xte.shape)
    #print('测试标签：',Yte.shape)

    #查看部分数据图片
    # clas = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # num_class = len(clas)
    # num_samples = 7
    # for idx , cla in enumerate(clas):
    #     idxx = np.flatnonzero(Ytr == idx)
    #     idxx = np.random.choice(idxx , num_samples,replace=False)
    #     for inx , cl in enumerate(idxx):
    #         plt.subplot(num_samples, num_class,inx*num_class+idx+1)
    #         plt.imshow(Xtr[cl].astype('uint8'))
    #         plt.axis('off')
    #         if inx == 0:
    #             plt.title(cla)
            
    # plt.show()


    num_traing = 5000
    Xtr = Xtr[list(range(num_traing))]
    Ytr = Ytr[list(range(num_traing))]

    num_testing = 500
    Xte = Xte[list(range(num_testing))]
    Yte = Yte[list(range(num_testing))]

    Xtr = np.reshape(Xtr,(Xtr.shape[0],32*32*3))
    Xte = np.reshape(Xte,(Xte.shape[0],32*32*3))
    #print(Xte.shape,Xtr.shape)

    kn = KNN()
    kn.train(Xtr,Ytr)
    for k in range(1,5):
        Yte_e = kn.predict(Xte,k)
        num_cor = np.sum(Ytr == Yte_e)
        acc = float(num_cor / num_testing)
        print("%d/%d,精确率：%f"%num_cor,num_testing,acc)




