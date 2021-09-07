#! /usr/bin/env python
#ecoding="UTF-8"
import os
import pickle
import numpy as np


class Cifar(object):
    def __init__(self):
        pass

    def single_batch(self, filename):
        with open(filename,'rb') as f:
            f = pickle.load(f,encoding='latin1')
            X = f["data"]
            Y = f["labels"]
            X = X.reshape(10000,3,32,32).transpose(0, 2, 3,1).astype("float")
            Y = np.array(Y)
            return X,Y

    def all_batch(self,root):
        xs = []
        ys = []
        for i in range(1,6):
            filename = os.path.join(root,"data_batch_%d"%i)
            X , Y = self.single_batch(filename)
            xs.append(X)
            ys.append(Y)

        Xtr = np.concatenate(xs)
        Ytr = np.concatenate(ys)

        Xte,Yte = self.single_batch(os.path.join(root,"test_batch"))

        return Xtr,Ytr,Xte,Yte

    