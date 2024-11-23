import cupy as np
from fuction_utils import math_fuction
import json
import os
from optimizer import Optimizer
import math

f = math_fuction()


class MLP:
    def __init__(self, num_layers, num_neus, lossf='cross-entropy', isBatchNorm=False, isL2=False, lambd=0.01, 
                 is_test=False, lr=0.005, beta=0.9, isdropout=False, dropout=[]):

        if not isinstance(num_neus, list):
            assert "num_neus应为列表类型对应每层神经元数量"
            
        if not isinstance(dropout, list):
            assert "dropout(list)为每层dropout的概率值"

        if len(num_neus) != num_layers:
            assert "层数与每层神经元层数不一致"
            
        if len(dropout) < num_layers:
            for i in range(len(dropout) - num_layers):
                dropout.append(0.0)

        self.num_layers = num_layers
        self.num_neus = num_neus
        self.isBatchNorm = isBatchNorm
        self.is_test = is_test
        self.lossf = lossf
        
        # 用于 dropout 正则化
        self.droupout = dropout.copy()
        self.droup = []
        self.isdropout = isdropout
        
        self.lr = lr
        self.beta = beta

        # L2正则化
        self.isL2 = isL2
        self.lambd = lambd

        # 根据信息初始化权重
        self.w = []
        self.b = []

        # 用于前向传播时归一化输入
        self.sigma2 = []
        self.mu = []

        # 保证参数随机初始化一致
        np.random.seed(2023)

        for i in range(num_layers):
            if i != num_layers - 1:
                self.w.append(np.random.randn(num_neus[i], num_neus[i+1]))
                self.b.append(0.0)
            else:
                self.w.append(np.random.randn(num_neus[i], 10))
                self.b.append(0.0)

    def forward(self, x, bsz):
        if self.isBatchNorm and not self.is_test:
            result, z, L2_loss = self.forward_back_batchnorm(x, bsz)
        elif self.isBatchNorm and self.is_test:
            result, z, L2_loss = self.forward_pre_batchnorm(x, bsz)
        else:
            result, z, L2_loss = self.forward_normal(x, bsz)

        return result, z, L2_loss

    def forward_normal(self, x, bsz):
        # 可定义网络结构为每层间嵌套relu函数作为激活函数，最后输出层后使用softmax函数计算概率
        Z = []
        L2_loss = 0
        self.droup.clear()
        x = np.array([item.reshape(-1, 1) for item in x])
        for i in range(self.num_layers):
            result = self.w[i].T @ x + self.b[i]
            Z.append(result)
            x = f.relu(result)
            
            if self.isdropout and not self.is_test:
                mask = np.random.rand(x.shape[0], x.shape[1], x.shape[2])
                mask = np.where(mask > self.droupout[i], 1.0, 0.0)
                self.droup.append(mask)
                if not math.fabs(self.droupout[i]) < 1e-5:
                    x = mask * x / (1 - self.droupout[i])

            if self.isL2:
                L2_loss += np.sum((self.lambd/(2*bsz)) *
                                  (self.w[i] @ self.w[i].T))

        result = f.softmax(result)

        return result, Z, L2_loss

    def forward_back_batchnorm(self, x, bsz):   # 用于训练反向传播的含有归一化输入
        Z = []
        L2_loss = 0
        x = np.array([item.reshape(-1, 1) for item in x])
        self.mu.clear()
        self.sigma2.clear()
        self.droup.clear()

        for i in range(self.num_layers):
            result = self.w[i].T @ x + self.b[i]
            # 归一化输入
            muL = (1 / bsz) * np.sum(result, axis=(0, 2), keepdims=True)
            sigmaL = (1 / bsz) * np.sum(np.power(result - muL, 2),
                                        axis=(0, 2), keepdims=True)
            z_norm = (result - muL) / (np.sqrt(sigmaL + 0.00000001))
            gamma, beta_1 = 1 * \
                np.sqrt(sigmaL + 0.00000001), muL + 0  # 此时的方差为1，均值为0
            result = np.multiply(z_norm, gamma) + beta_1
            self.mu.append(muL)
            self.sigma2.append(sigmaL)

            Z.append(result)
            x = f.relu(result)
            
            if self.isdropout and not self.is_test:
                mask = np.random.rand(x.shape[0], x.shape[1], x.shape[2])
                mask = np.where(mask > self.droupout[i], 1.0, 0.0)
                self.droup.append(mask)
                
                
                
                if not math.fabs(self.droupout[i]) < 1e-5:
                    x = mask * x /  (1 - self.droupout[i])


            if self.isL2:
                L2_loss += np.sum((self.lambd/(2*bsz)) *
                                  (self.w[i] @ self.w[i].T))

        result = f.softmax(result)

        return result, Z, L2_loss

    def forward_pre_batchnorm(self, x, bsz):    # 用于测试的forward（含归一化）
        x = np.array([item.reshape(-1, 1) for item in x])
        result = 0
        for i in range(0, self.num_layers):
            result = self.w[i].T@x + self.b[i]
            # 归一化输入
            z_norm = (result - self.mu[i]) / \
                (np.sqrt(self.sigma2[i] + 0.00000001))
            gamma, beta_1 = 1 * \
                np.sqrt(self.sigma2[i] + 0.00000001), self.mu[i] + 0
            result = np.multiply(z_norm, gamma) + beta_1
            x = f.relu(result)

        result = f.softmax(result)
        return result, [], 0.0       # 保持接口统一
    
    def backward(self, X, Y, bsz):
        if self.lossf == 'cross-entropy':
            dw, db, loss = self.backward_crossentropy(X, Y, bsz)
        elif self.lossf == 'L2':
            dw, db, loss = self.backward_L2(X, Y, bsz)
        
        return dw, db, loss

    def backward_crossentropy(self, X, Y, bsz):
        if len(X) != len(Y):
            assert "数据与其标签长度不一致"
        result, z, L2_loss = self.forward(X, bsz)
        X = np.array([item.reshape(-1, 1) for item in X])
        Y = np.array([item.reshape(10, 1) for item in Y])
        dw, db = [], []
        loss = f.binary_cross_entropy_loss(Y, result)
        loss = loss.mean()  # 为方便损失度量，不添加L2正则化损失，但L2正则化若被启用会加入到梯度下降的计算中

        for i in range(self.num_layers - 1, 0, -1):
            if i == self.num_layers - 1:
                dz = result - Y
            else:
                dz = self.w[i + 1] @ dz * f.relu_derivative(z[i])

            if self.isL2:
                Dw = dz @ f.relu(z[i - 1].transpose((0, 2, 1))) + \
                    (self.lambd) * self.w[i].T
            else:
                Dw = dz @ f.relu(z[i - 1].transpose((0, 2, 1)))
                
            if self.isdropout and not math.fabs(self.droupout[i]) < 1e-5:
                self.droup[i] = np.tile(self.droup[i], Dw.shape[2])
                Dw = Dw * self.droup[i] /  (1 - self.droupout[i])
                
                
            Dw = np.sum(Dw, axis=0)
            Db = np.sum(dz, axis=0)
            Db = np.sum(Db, axis=1, keepdims=True)
            dw.append(Dw)
            db.append(Db)

        dz = self.w[1] @ dz * f.relu_derivative(z[0])

        if self.isL2:
            Dw = dz @ X.transpose((0, 2, 1)) + (self.lambd) * self.w[0].T
        else:
            Dw = dz @ X.transpose((0, 2, 1))

        Dw = np.sum(Dw, axis=0)
        Db = np.sum(dz, axis=0)
        Db = np.sum(Db, axis=1, keepdims=True)
        dw.append(Dw)
        db.append(Db)

        return dw, db, loss
    
    def backward_L2(self, X, Y, bsz):       # 使用L2损失的反向传播
        if len(X) != len(Y):
            assert "数据与其标签长度不一致"
        result, z, L2_loss = self.forward(X, bsz)
        X = np.array([item.reshape(-1, 1) for item in X])
        Y = np.array([item.reshape(10, 1) for item in Y])
        dw, db = [], []
        loss = f.L2_loss(Y, result)
        loss = loss/bsz # 为方便损失度量，不添加L2正则化损失，但L2正则化若被启用会加入到梯度下降的计算中

        for i in range(self.num_layers - 1, 0, -1):
            if i == self.num_layers - 1:
                dz = []
                for k in range(10):
                    add = 0
                    for j in range(10):
                        if j != k:
                            add += result[:,j] * (Y[:,j] - result[:,j])
                    res = 2 * np.power(result[:,k], 2) * (Y[:,k] - result[:,k]) - 2 * result[:,k] * (Y[:,k] - result[:,k]) + 2 * result[:,k] * add
                    dz.append(res)
                dz = np.array(dz).transpose(1,0,2)
            else:
                dz = self.w[i + 1] @ dz * f.relu_derivative(z[i])

            if self.isL2:
                Dw = dz @ f.relu(z[i - 1].transpose((0, 2, 1))) + \
                    (self.lambd / bsz) * self.w[i].T
            else:
                Dw = dz @ f.relu(z[i - 1].transpose((0, 2, 1)))
                
            if self.isdropout and not math.fabs(self.droupout[i]) < 1e-5:
                self.droup[i] = np.tile(self.droup[i], Dw.shape[2])
                Dw = Dw * self.droup[i] /  (1 - self.droupout[i])
                
            Dw = np.sum(Dw, axis=0)
            Db = np.sum(dz, axis=0)
            Db = np.sum(Db, axis=1, keepdims=True)
            dw.append(Dw)
            db.append(Db)

        dz = self.w[1] @ dz * f.relu_derivative(z[0])

        if self.isL2:
            Dw = dz @ X.transpose((0, 2, 1)) + (self.lambd / bsz) * self.w[0].T
        else:
            Dw = dz @ X.transpose((0, 2, 1))

        Dw = np.sum(Dw, axis=0)
        Db = np.sum(dz, axis=0)
        Db = np.sum(Db, axis=1, keepdims=True)
        dw.append(Dw)
        db.append(Db)

        return dw, db, loss

    def predict(self, x, bsz):
        self.is_test = True
        outputs, Z, l2_loss = self.forward(x, bsz)
        cls = []
        for output in outputs:
            number = np.argmax(output)
            cls.append(number)

        return cls

    def train_steps(self, X, Y, bsz, optimizer: Optimizer, epoch):
        self.is_test = False
        dw, db, loss = self.backward(X, Y, bsz)
        optimizer.optimize_params(self, dw, db, epoch)
        return loss

    def save_weights(self, file_name='weights.json'):
        print("Saving....")
        if self.isBatchNorm:
            weights = {
                'w': [weight.tolist() for weight in self.w],
                'b': [weight.tolist() for weight in self.b],
                'mu': [weight.tolist() for weight in self.mu],
                'sigma2': [weight.tolist() for weight in self.sigma2],
            }
        else:
            weights = {
                'w': [weight.tolist() for weight in self.w],
                'b': [weight.tolist() for weight in self.b],
            }
        json_weights = json.dumps(weights)
        try:
            with open(file_name, 'w') as weight:
                weight.write(json_weights)
            print("Save Success!")
        except FileNotFoundError:
            os.mknod(file_name)
            self.save_weights(file_name)

    def load_weights(self, file_name='weights.json'):
        print("Loading Checkpoint File " + file_name + ".....")
        weights = {}
        with open(file_name, 'r') as weight:
            weights = json.load(weight)

        self.w = [np.array(weight) for weight in weights['w']]
        self.b = [np.array(weight) for weight in weights['b']]

        try:
            self.mu = [np.array(weight) for weight in weights['mu']]
            self.sigma2 = [np.array(weight) for weight in weights['sigma2']]
        except:
            print('并未启用归一化输入')
            self.isBatchNorm = False

        print("Loading Success!")
