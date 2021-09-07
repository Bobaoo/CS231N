#! /usr/bin/env python

import numpy as np

class KNN(object):
    def __init__(self):
        pass

    def train(self,Xtr,Ytr):
        self.Xtr = Xtr
        self.Ytr = Ytr

    def predict(self,Xte,k):
        num_test = Xte.shape[0]
        num_train = self.Xtr.shape[0]
        Yte_e = np.zeros(num_test, dtype=self.Ytr.dtype)
        distances = np.zeros((num_test,num_train))
        for i in range(num_test):
            distances = np.sqrt(np.sum(np.square(self.Xtr-Xte[i])))
            inx = np.argmin(distances)
            
        for i in range(num_test):
            near = self.Ytr[np.argsort(distances[i])[:k]]
            Yte_e[i] = np.argmax(np.bincount(near))
        
        return Yte_e