# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 17:53:18 2022

@author: satya
"""

import torch
import torch.nn as nn
import torch.optim as optimiz

class FeedforwardNeuralNetModelC(nn.Module):
    def __init__(self):
        super().__init__()
        self.a1 = nn.Linear(20,80)
        self.actfn = torch.nn.Tanh()
        self.a2 = nn.Linear(80,1)
        #self.a3 = nn.Linear(1,1)
        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.25)

    def forward(self,x):
        o1 = self.a1(x)
        actfn = self.actfn(o1)
        output = self.dropout(x)
        output = self.a2(actfn)
        #output = self.sigmoid(output)
        return output

    def predict(self, X):
       Y_pred = self.forward(X)
       return Y_pred


    def accuracy(y_hat, y):
        pred = torch.argmax(y_hat, dim=1)
        return (pred == y).float().mean()
    

