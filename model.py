# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 07:05:33 2021

@author: Hui Hu
"""
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from deeprobust.graph.utils import accuracy
from copy  import deepcopy
from Original_GCN import GCN

class DPGCN:
    def __init__(self,model,args,device):
        self.device = device
        self.args = args
        self.best_val_acc = 0
        self.best_val_loss = np.inf
        self.best_s_acc = 0
        self.best_s_loss = np.inf
        self.best_S1 = None
        self.best_S2 = None
        self.weights = None
        self.model = model.to(device)
        self.estimator1 = None
        self.estimator2 = None
        self.features = []         
        
    def fit(self,features,adj,S1,S2,labels,sens,idx_train,idx_val,idx_test,idx_sensitive,idx_nosensitive,sen_attr_index,**kwargs):
        args = self.args
        self.optimizer = optim.Adam(self.model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
        estimator1 = EstimateS1(S1,device=self.device).to(self.device)
        estimator2 = EstimateS2(S2,device=self.device).to(self.device)
        self.estimator1 = estimator1
        self.estimator2 = estimator2
        self.optimizer_S1 = optim.SGD(estimator1.parameters(),momentum=0.9, lr=args.lr_w1)
        self.optimizer_S2 = optim.SGD(estimator2.parameters(),momentum=0.9, lr=args.lr_w2)
        for epoch in range(args.epoch1):
             for i in range(int(args.S1)):
                  self.train_S1(epoch,features,adj,labels,sens,S1,idx_sensitive,idx_train,idx_val,sen_attr_index)  
             for i in range(int(args.S2)):
                  self.train_S2(epoch,features,adj,labels,sens,S2,idx_sensitive,idx_train,idx_val,sen_attr_index)
             for i in range(int(args.inner_steps)):
                 self.train_gcn(epoch,features,adj,labels,S1,S2,idx_train,idx_val,sen_attr_index)
        self.model.load_state_dict(self.weights)
    
    def train_gcn(self,epoch,features,adj,labels,S1,S2,idx_train,idx_val,sen_attr_index):
        estimator1 = self.estimator1
        estimator2 = self.estimator2
        S1= estimator1.estimated_S1
        S2= estimator2.estimated_S2
        self.model.train()
        self.optimizer.zero_grad()   
        feature_temp = features.clone()
        feature_temp = torch.cat((feature_temp[:,0:sen_attr_index],feature_temp[:,sen_attr_index+1:]),1)
        #feature_temp = torch.mm(feature_temp,W)
        latent = torch.mm(feature_temp,S2)
        output = self.model(latent,adj)
        loss_train = F.nll_loss(output[idx_train],labels[idx_train])
        acc_train = accuracy(output[idx_train],labels[idx_train])
        loss_train.backward()  
        self.optimizer.step() 
        self.model.eval()
        output = self.model(latent,adj)
        loss_val = F.nll_loss(output[idx_val],labels[idx_val])
        acc_val = accuracy(output[idx_val],labels[idx_val])
        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_S1 = S1.detach()
            self.best_S2 = S2.detach()
            self.weights = deepcopy(self.model.state_dict())
        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_S1 = S1.detach()
            self.best_S2 = S2.detach()
            self.weights = deepcopy(self.model.state_dict())
        if epoch % 1 == 0:
            print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()))
  
    def train_S1(self,epoch,features,adj,labels,sens,S1,idx_sensitive,idx_train,idx_val,sen_attr_index):
        args = self.args
        feature_copy = features[idx_sensitive,:].clone()
        estimator1 = self.estimator1
        estimator2 = self.estimator2
        estimator1.train()
        self.optimizer_S1.zero_grad()
        S1= estimator1.estimated_S1
        S2= estimator2.estimated_S2
        feature_copy = torch.cat((feature_copy[:,0:sen_attr_index],feature_copy[:,sen_attr_index+1:]),1)
        #feature_copy = torch.mm(feature_copy,W)
        non_sensitive_feature = feature_copy 
        S = torch.zeros(feature_copy.shape[0], features.shape[1])
        S[:,sen_attr_index] = sens[idx_sensitive]
        sensitive_feature = S
        loss_re = torch.norm(torch.mm(non_sensitive_feature,S1)-sensitive_feature,"fro")
        loss_orth = torch.norm(torch.mm(S1.T,S2),"fro")
        loss_total = args.alpha*loss_re + args.beta*loss_orth
        loss_total.backward()
        self.optimizer_S1.step()
        self.model.eval()
        self.best_S1 = S1.detach()         
        if epoch % 1 == 0:
            print('S1 Epoch: {:04d}'.format(epoch+1),
              'loss_re: {:.4f}'.format(loss_re.item()),
              'loss_orth: {:.4f}'.format(loss_orth.item()))
            
    def train_S2(self,epoch,features,adj,labels,sens,S1,idx_sensitive,idx_train,idx_val,sen_attr_index):
        args = self.args
        S1 = self.best_S1
        estimator2 = self.estimator2
        estimator2.train()
        self.optimizer_S2.zero_grad()
        S2= estimator2.estimated_S2
        feature_temp = features.clone()
        feature_temp = torch.cat((feature_temp[:,0:sen_attr_index],feature_temp[:,sen_attr_index+1:]),1)
        #feature_temp = torch.mm(feature_temp,W)
        latent = torch.mm(feature_temp,S2)
        output = self.model(latent,adj)
        loss_train = F.nll_loss(output[idx_train],labels[idx_train]) 
        acc_train = accuracy(output[idx_train],labels[idx_train])
        loss_orth = torch.norm(torch.mm(S1.T,S2),"fro")
        loss_total = args.gamma*loss_train + args.beta*loss_orth  
        loss_total.backward()
        self.optimizer_S2.step()
        self.model.eval()
        latent = torch.mm(feature_temp,S2)
        output = self.model(latent,adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_S2 = S2.detach()
            self.weights = deepcopy(self.model.state_dict())
        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_S2 = S2.detach()
            self.weights = deepcopy(self.model.state_dict())    
        if epoch % 1 == 0:
            print('S2 Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'loss_orth: {:.4f}'.format(loss_orth.item()))
        
    def test(self,features,adj,labels,idx_test,sen_attr_index):
        self.model.eval()
        S2 = self.best_S2
        feature_temp = features.clone()
        feature_temp = torch.cat((feature_temp[:,0:sen_attr_index],feature_temp[:,sen_attr_index+1:]),1)
        #feature_temp = torch.mm(feature_temp,W)
        latent = torch.mm(feature_temp,S2)
        output = self.model(latent,adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("\tTest set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
   
    def infer(self,features,adj,sen_attr_index,idx_sensitive,idx_nosensitive,sens):
        self.model.eval()
        args = self.args
        S2= self.best_S2
        feature_temp = features.clone()
        feature_temp = torch.cat((feature_temp[:,0:sen_attr_index],feature_temp[:,sen_attr_index+1:]),1)
        #feature_temp = torch.mm(feature_temp,W)
        latent = torch.mm(feature_temp,S2)
        labels = sens
        model_s = GCN(nfeat=latent.shape[1],nhid=args.hidden,nclass=int(labels.max().item()) + 1,dropout=args.dropout, device=self.device)
        model_s.fit(latent,adj,labels,idx_sensitive,None,args.epoch2,verbose=True)    
        model_s.eval()
        acc,preds = model_s.inference(idx_nosensitive)
        print("The inference accuracy is: %f",acc)
        
class EstimateS1(nn.Module):
    def __init__(self, S1,device):
        super(EstimateS1, self).__init__()
        n = S1.shape[0]    
        m = S1.shape[1]
        self.estimated_S1 = nn.Parameter(torch.FloatTensor(n,m))
        self._init_estimation(S1)
        self.device = device
    def _init_estimation(self, S1):
        with torch.no_grad():
            self.estimated_S1.data.copy_(S1)
    def forward(self):
        return self.estimated_S1
    
class EstimateS2(nn.Module):
    def __init__(self, S2,device):
        super(EstimateS2, self).__init__()
        n = S2.shape[0]    
        m = S2.shape[1]
        self.estimated_S2 = nn.Parameter(torch.FloatTensor(n,m))
        self._init_estimation(S2)
        self.device = device
    def _init_estimation(self, S2):
        with torch.no_grad():
            self.estimated_S2.data.copy_(S2)
    def forward(self):
        return self.estimated_S2