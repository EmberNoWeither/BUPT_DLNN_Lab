
import cupy as np
from fuction_utils import math_fuction


class Optimizer:
    def __init__(self, model, type:str = 'mini-batch', lr:float = 5e-3, decay:float = 0.0) -> None:
        self.type = type
        self.lr = lr
        self.decay = decay
        self.epoch = 0
        
    def decay_process(self):
        self.lr = self.lr / (1 + self.decay)
        
    def optimize_params(self, model, dw:list[float], db:list[float], epoch):
        for i in range(0, model.num_layers):
            model.w[i] = model.w[i] - self.lr * dw[model.num_layers - i - 1].T
            model.b[i] = model.b[i] - self.lr * db[model.num_layers - i - 1]
            
        if self.epoch != epoch:
            self.epoch = epoch
            self.decay_process()
            
        return model.w, model.b
    

class Optimizer_Adam(Optimizer):
    def __init__(self, model, beta1:float = 0.9, beta2=0.999,  type: str = 'Adam', lr: float = 0.005, decay: bool = False) -> None:
        super().__init__(model, type, lr, decay)
        self.Vdw = []
        self.Vdb = []
        self.Sdw = []
        self.Sdb = []
        self.beta1 = beta1
        self.beta2 = beta2
        # print(self.type)
        
        for i in range(model.num_layers):
            self.Vdw.append(0)
            self.Vdb.append(0)
            self.Sdb.append(0)
            self.Sdw.append(0)
            
        
    def optimize_params(self, model, dw: list[float], db: list[float], epoch):
        for i in range(0, model.num_layers):
            self.Vdw[i] = self.beta * self.Vdw[i] + (1 - self.beta1) * dw[model.num_layers - i - 1].T
            self.Vdb[i] = self.beta * self.Vdb[i] + (1 - self.beta1) * db[model.num_layers - i - 1]
            self.Sdw[i] = self.beta * self.Sdw[i] + (1 - self.beta2) * np.power(dw[model.num_layers - i - 1].T, 2)
            self.Sdb[i] = self.beta * self.Sdb[i] + (1 - self.beta2) * np.power(db[model.num_layers - i - 1], 2)

            model.w[i] = model.w[i] - self.lr * self.Vdw[i] / (np.power(self.Sdw[i], 1 / 2) + 0.00000001)
            model.b[i] = model.b[i] - self.lr * self.Vdb[i] / (np.power(self.Sdb[i], 1 / 2) + 0.00000001)
            
        if self.epoch != epoch:
            self.epoch = epoch
            self.decay_process()
            
        return model.w, model.b
            
            
class Optimizer_Momentum(Optimizer):
    def __init__(self, model, beta: float = 0.9, type: str = 'momentum', lr: float = 0.005, decay: bool = False) -> None:
        super().__init__(model, type, lr, decay)
        self.Vdw = []
        self.Vdb = []
        self.beta = beta
        
        for i in range(model.num_layers):
            self.Vdw.append(0)
            self.Vdb.append(0)

    def optimize_params(self, model, dw: list[float], db: list[float], epoch):
        for i in range(0, model.num_layers):
            self.Vdw[i] = self.beta * self.Vdw[i] + (1 - self.beta) * dw[model.num_layers - i - 1].T
            self.Vdb[i] = self.beta * self.Vdb[i] + (1 - self.beta) * db[model.num_layers - i - 1]
            model.w[i] = model.w[i] - self.lr * self.Vdw[i]
            model.b[i] = model.b[i] - self.lr * self.Vdb[i]
            
        if self.epoch != epoch:
            self.epoch = epoch
            self.decay_process()
            
        return model.w, model.b
        