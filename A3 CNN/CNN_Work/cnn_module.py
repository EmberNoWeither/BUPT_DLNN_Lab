import torch
from typing import List
from torch import nn

class LeNet(nn.Module):
    def __init__(self, in_dims=3, droup_rate=0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
        nn.Conv2d(in_dims, 6, kernel_size=5, padding=2), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 6 * 6, 120), nn.Sigmoid(),
        nn.Dropout(p=droup_rate),
        nn.Linear(120, 84), nn.Sigmoid(),
        nn.Linear(84, 10),
        nn.Softmax())
        
        for layer in self.net:
            if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
                nn.init.xavier_uniform_(layer.weight)
        
    def __call__(self, X):
        return self.forward(X)
        
        
    def forward(self, X):    
        return self.net(X)
    

class AlexNet(nn.Module):
    def __init__(self, in_dims=3,droup_rate=0.5, BN=False) -> None:
        super().__init__()
        if BN:
            self.net = nn.Sequential(
                        nn.Conv2d(in_dims, 96, kernel_size=11, stride=4, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2),
                        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.BatchNorm2d(256), nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2),
                        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.BatchNorm2d(384), nn.ReLU(),
                        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.BatchNorm2d(384), nn.ReLU(),
                        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2),
                        nn.Flatten(),
                        #使用dropout层来减轻过拟合
                        nn.Linear(6400, 4096), nn.BatchNorm1d(4096), nn.ReLU(),
                        nn.Dropout(p=droup_rate),
                        nn.Linear(4096, 4096), nn.BatchNorm1d(4096), nn.ReLU(),
                        nn.Dropout(p=droup_rate),
                        nn.Linear(4096, 10))
        else:
            self.net = nn.Sequential(
                        nn.Conv2d(in_dims, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2),
                        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2),
                        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
                        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
                        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2),
                        nn.Flatten(),
                        #使用dropout层来减轻过拟合
                        nn.Linear(6400, 4096), nn.ReLU(),
                        nn.Dropout(p=droup_rate),
                        nn.Linear(4096, 4096), nn.ReLU(),
                        nn.Dropout(p=droup_rate),
                        nn.Linear(4096, 10))
        
        for layer in self.net:
            if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
                nn.init.xavier_uniform_(layer.weight)
        
        
    def __call__(self, X):
        return self.forward(X)
        
    def forward(self, X):
        return self.net(X)
    
    
    
    
class AGGNet(nn.Module):
    def __init__(self, nets:List[nn.Module]) -> None:
        self.nets = nets
        self.params = [{"params": net.parameters()} for net in nets] # 各模型参数
        
        
    def parameters(self, recurse: bool = True):
        return self.params
    
    def train(self, mode: bool = True):
        for net in self.nets:
            net.train()
    
    def eval(self):
        for net in self.nets:
            net.eval()

    def to(self, params):
        for net in self.nets:
            net.to(params)
        return self
        
    def __call__(self, X):
        return self.forward(X)
        
    def forward(self, X):   # 求各网络结果平均值进行聚合
        outputs = []
        for net in self.nets:
            output = net(X)
            outputs.append(output)
            
        out = torch.zeros_like(outputs[0], device=X.device, requires_grad=True)
        for output in outputs:
            out = out + output
            
        out /= len(outputs)
            
        return out