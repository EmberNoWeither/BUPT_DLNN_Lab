import cupy as np
from fuction_utils import math_fuction

f = math_fuction()
class SoftMaxClassifier:
    def __init__(self, lr=0.005, lossf='cross-entropy', optim='adam', beta=0.9) -> None:
        
        # 参数初始化
        np.random.seed(2023)
        self.w = np.random.randn(28*28,10)
        self.b = np.zeros((10, 1))
        self.lr = lr
        self.lossf = lossf
        self.optim = optim
        self.num_layers = 1
        self.is_test = False
        
        self.Vdw = 0
        self.Vdb = 0
        self.Sdw = 0
        self.Sdb = 0
        
        self.beta = beta
        
    def forward(self, x, bsz):
        x = np.array([item.reshape(-1, 1) for item in x])
        result = self.w.T @ x + self.b
        
        result = f.softmax(result)
        return result
    
    
    def backward(self, X, Y, bsz):
        
        result = self.forward(X, bsz)
        
        X = np.array([item.reshape(-1, 1) for item in X])
        Y = np.array([item.reshape(10, 1) for item in Y])
        dw, db = [], []
        if self.lossf == 'cross-entropy':
            loss = f.binary_cross_entropy_loss(Y, result)
            loss = loss.mean() 
        elif self.lossf == 'L2':
            loss = f.L2_loss(Y, result)
            loss = loss / bsz
        
        if self.lossf == 'cross-entropy':
            dz = result - Y
        elif self.lossf == 'L2':
            dz = []
            for k in range(10):
                add = 0
                for j in range(10):
                    if j != k:
                        add += result[:,j] * (Y[:,j] - result[:,j])
                res = 2 * np.power(result[:,k], 2) * (Y[:,k] - result[:,k]) - 2 * result[:,k] * (Y[:,k] - result[:,k]) + 2 * result[:,k] * add
                dz.append(res)
            dz = np.array(dz).transpose(1,0,2)
            
        Dw = dz @ X.transpose((0, 2, 1))
        Dw = np.sum(Dw, axis=0)
        Db = np.sum(dz, axis=0)
        Db = np.sum(Db, axis=1, keepdims=True)
            
        return Dw, Db, loss
        
    def optimize_back(self, dw, db):
        if self.optim == 'adam':
            self.Vdw = self.beta * self.Vdw + (1 - self.beta) * dw.T
            self.Vdb = self.beta * self.Vdb + (1 - self.beta) * db
            self.Sdw = self.beta * self.Sdw + (1 - self.beta) * np.power(dw.T, 2)
            self.Sdb = self.beta * self.Sdb + (1 - self.beta) * np.power(db, 2)

            self.w = self.w - self.lr * self.Vdw / (np.power(self.Sdw, 1 / 2) + 0.00000001)
            self.b = self.b - self.lr * self.Vdb / (np.power(self.Sdb, 1 / 2) + 0.00000001)
            
        elif self.optim == 'mini-batch':
            self.w -= self.lr * dw.T
            self.b -= self.lr * db
            
        elif self.optim == 'momentum':  
            self.Vdw = self.beta * self.Vdw + (1 - self.beta) * dw.T
            self.Vdb = self.beta * self.Vdb + (1 - self.beta) * db
            self.w = self.w - self.lr * self.Vdw
            self.b = self.b - self.lr * self.Vdb
            
            
    def predict(self, x, bsz):
        outputs = self.forward(x, bsz)
        cls = []
        for output in outputs:
            number = np.argmax(output)
            cls.append(number)

        return cls
            
    def train_steps(self, X, Y, bsz, optim=None, epoch=None):
        Dw, Db, loss = self.backward(X, Y, bsz)
        self.optimize_back(Dw, Db)
        return loss
        
    
    
    