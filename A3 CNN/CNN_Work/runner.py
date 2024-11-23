import torch
import numpy as np
from cnn_module import AGGNet
from torch import nn
from dataset_prepare import dataset_make, minist_dataset_make
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from sklearn.model_selection import KFold
import copy
from torch.utils.data import DataLoader

class Runner:
    def __init__(self, module : nn.Module, batch_size=256, num_workers=8, 
                 epochs=15, lr=1e-3, resize=None, device='cuda', set_model_name=None,
                 datasets='CIFAR-10', weight_decay=0, kf=True,
                 optimizer:torch.optim.Adam = None) -> None:
        self.module = module.to(device)
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.Kf=kf
        self.folders=10
        
        if datasets == 'CIFAR-10':
            self.train_iter, self.test_iter, self.train_set = dataset_make(batch_size, num_workers, resize=resize)
        else:
            self.train_iter, self.test_iter, self.train_set = minist_dataset_make(batch_size, num_workers, resize=resize)
        
        if self.Kf == False:
            self.val_iter = self.test_iter
        else:
            self.val_iter = None

        # if isinstance(self.module, AGGNet):
        #     self.optimizers = []
        #     for net in self.module.nets:
        #         optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        #         self.optimizers.append(optimizer)
        # else:
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)
            
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()
        
        self.train_acc = []
        self.test_acc = []
        self.val_acc = []
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.set_name = None
        if set_model_name:
            self.set_name = set_model_name
        
    def _get_train_result(self):
        return self.train_losses, self.train_acc
    
    def _get_valid_result(self):
        return self.val_losses, self.val_acc
    
    def _get_test_result(self):
        return self.test_losses, self.test_acc
    
    @torch.no_grad()
    def outputs_metric(self, Y_hat, Y):
        len = Y_hat.shape[0]
        score = 0.0
        
        for y_hat, y in zip(Y_hat, Y):
            if torch.argmax(y_hat) == y:
                score += 1
                
        return score/float(len)
    
    def train_steps(self, X:torch.Tensor, Y:torch.Tensor):
        self.optimizer.zero_grad()
        X, Y = X.to(self.device), Y.to(self.device)
        X.requires_grad_()
        Y_c = []
        yy = [0,0,0,0,0,0,0,0,0,0]
        for y in Y:
            yy = [0,0,0,0,0,0,0,0,0,0]
            yy[y] = 1.0
            Y_c.append(yy)
        Y_c = torch.tensor(Y_c, device=self.device,requires_grad=True)
        y_hat = self.module(X)
        loss = self.loss(y_hat, Y_c)
        loss.backward()
        self.optimizer.step()

        train_score = self.outputs_metric(y_hat, Y)
        return train_score, loss.item()
    
    @torch.no_grad()
    def test_steps(self, X, Y):
        X, Y = X.to(self.device), Y.to(self.device)
        Y_c = []
        yy = [0,0,0,0,0,0,0,0,0,0]
        for y in Y:
            yy = [0,0,0,0,0,0,0,0,0,0]
            yy[y] = 1.0
            Y_c.append(yy)
        Y_c = torch.tensor(Y_c, device=self.device)
        y_hat = self.module(X)
        loss = self.loss(y_hat, Y_c)
        test_score = self.outputs_metric(y_hat, Y)
    
        return test_score, loss.item()
    
    
    def train(self):
        
        # 使用K折交叉验证 - epoch=1
        if self.Kf:
            kf = KFold(n_splits=10, shuffle=True, random_state=0)
            folders = 0
            torch.save(self.module.state_dict(), './lenet_origin.pth')
            for train_index, val_index in kf.split(self.train_set):
                self.module.load_state_dict(torch.load('./lenet_origin.pth'))
                self.optimizer = torch.optim.Adam(self.module.parameters(), lr=self.lr)
                
                train_fold = torch.utils.data.dataset.Subset(self.train_set, train_index)
                val_fold = torch.utils.data.dataset.Subset(self.train_set, val_index)    
                self.train_iter = DataLoader(dataset=train_fold, batch_size=self.batch_size, shuffle=True)
                self.val_iter = DataLoader(dataset=val_fold, batch_size=128, shuffle=True, drop_last=False)
                
                self.module.train()
                train_loss = 0.0
                train_acc = 0.0
                for i in range(self.epochs):
                    with tqdm(total=len(self.train_iter)) as t:
                        for idx, (X, Y) in enumerate(self.train_iter):
                            t.set_description("Fold: %i"%folders+" Epoch: %i"%i)
                            a, b = self.train_steps(X, Y)
                            t.set_postfix(train_loss='%.4f'%b,train_acc='%.4f'%a)
                            train_loss += b
                            train_acc += a
                            t.update(1)
                    self.train_acc.append(train_acc/len(self.train_iter))
                    self.train_losses.append(train_loss/len(self.train_iter))
                    
                self.module.eval()
                self.val()
                
                folders += 1
        else:
            for i in range(self.epochs):
                self.module.train()
                train_loss = 0.0
                train_acc = 0.0
                with tqdm(total=len(self.train_iter)) as t:
                    for idx, (X, Y) in enumerate(self.train_iter):
                        t.set_description("Epoch: %i" %i)
                        a, b = self.train_steps(X, Y)
                        t.set_postfix(train_loss='%.4f'%b,train_acc='%.4f'%a)
                        train_loss += b
                        train_acc += a
                        t.update(1)
                self.train_acc.append(train_acc/len(self.train_iter))
                self.train_losses.append(train_loss/len(self.train_iter))
                self.module.eval()
                self.val()
        self.test()
                        
    @torch.no_grad()
    def val(self):
        test_acc = 0.0
        test_loss = 0.0
        with tqdm(total=len(self.val_iter)) as t:
            for idx, (X, Y)  in enumerate(self.val_iter):
                t.set_description("Valid")
                a, b = self.test_steps(X, Y)
                test_acc += a 
                test_loss += b
                t.set_postfix(val_loss='%.4f'%b,val_acc='%.4f'%a)
                t.update(1)
    
        test_acc /= float(len(self.val_iter))
        test_loss /= float(len(self.val_iter))
        self.val_acc.append(test_acc)
        self.val_losses.append(test_loss)
        
        # print("Test Finish!")
        print("Val_acc:%.3f"%test_acc+" Val_loss:%.4f"%test_loss) 
        
        return self.val_losses, self.val_acc
                
    @torch.no_grad()
    def test(self):
        test_acc = 0.0
        test_loss = 0.0
        with tqdm(total=len(self.test_iter)) as t:
            for idx, (X, Y)  in enumerate(self.test_iter):
                t.set_description("Testing")
                a, b = self.test_steps(X, Y)
                test_acc += a 
                test_loss += b
                t.set_postfix(test_loss='%.4f'%b,test_acc='%.4f'%a)
                t.update(1)
    
        test_acc /= float(len(self.test_iter))
        test_loss /= float(len(self.test_iter))
        self.test_acc.append(test_acc)
        self.test_losses.append(test_loss)
        
        print("Test Finish!")
        print("test_acc:%.3f"%test_acc+" test_loss:%.4f"%test_loss)


    def get_model_name(self):
        if self.set_name:
            return self.set_name
        
        model_type = self.module.__class__.__name__
        return model_type