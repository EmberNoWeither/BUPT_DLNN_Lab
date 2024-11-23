import  torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib
import torch.nn as nn
import torch.utils.data as Data
from tqdm import tqdm
from sklearn.decomposition import PCA

from data_prepare import minist_dataset_make

transform = transforms.Compose(
    [transforms.ToTensor()])

train_dataset=torchvision.datasets.MNIST(root="./dataset",train=True,transform=transform,download=True)
test_dataset=torchvision.datasets.MNIST(root="./dataset", transform=transform,train=False, download=True)

pca_train_datasets = []
train_label = []
pca_test_datasets = []
test_label = []

for i in range(len(train_dataset)):
    # print(train_dataset[i][0].cpu().numpy().reshape(-1).shape)
    pca_train_datasets.append(train_dataset[i][0].cpu().numpy().reshape(-1))
    y_onehot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y_onehot[train_dataset[i][1]] = 1
    train_label.append(y_onehot)


for i in range(len(test_dataset)):
    pca_test_datasets.append(test_dataset[i][0].cpu().numpy().reshape(-1))
    y_onehot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y_onehot[test_dataset[i][1]] = 1
    test_label.append(y_onehot)
    
    
# 调用 SKlearn的 PCA 方法
n_components = 200
pca = PCA(n_components=n_components).fit(pca_train_datasets)

eigenvalues = pca.components_.reshape(n_components, 28, 28)

#提取PCA主成分（特征值），仔细想想应该是特征向量
eigenvalues = pca.components_

#画图
n_row = 1
n_col = 2

# # Plot the first 8 eignenvalues
# plt.figure(figsize=(13,12))
# for i in list(range(n_row * n_col)):
#     offset =0
#     plt.subplot(n_row, n_col, i + 1)
#     plt.imshow(eigenvalues[i].reshape(28,28), cmap='jet')
#     title_text = 'Eigenvalue ' + str(i + 1)
#     plt.title(title_text, size=6.5)
#     plt.xticks(())
#     plt.yticks(())
# plt.show()
# plt.savefig('./pca_minst.png')

pca_train_datasets = pca.transform(pca_train_datasets)
pca_test_datasets = pca.transform(pca_test_datasets)

train_dataloader = []
test_dataloader = []
batch_size = 128

train_dataset = Data.TensorDataset(torch.tensor(pca_train_datasets, dtype=torch.float), torch.tensor(train_label, dtype=torch.float))
test_dataset = Data.TensorDataset(torch.tensor(pca_test_datasets, dtype=torch.float), torch.tensor(test_label, dtype=torch.float))


train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True,
                                  num_workers=8)

test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=False,
                                  num_workers=8)


class SimpleClassifier(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(200, 256),
            nn.ReLU(True),
            nn.Linear(256, 10),
            nn.ReLU(True),
            nn.Softmax(dim=1)
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.fc(x)
    
    def train_step(self, X, y):
        outputs = self.forward(X)
        loss = self.loss_fn(outputs, y)
        return loss
    
    def predict(self, X, y):
        outputs = self.forward(X)
        i = torch.argmax(outputs)
        if i == torch.argmax(y):
            return 1
        else:
            return 0
    

model = SimpleClassifier().to('cuda')
Epochs = 15
LR = 5e-4
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


train_losses = []
acc = []
for i in range(Epochs):
    model.train()
    with tqdm(total=len(train_dataloader)) as t:
        for idx, (X, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = model.train_step(X.to('cuda'), y.to('cuda'))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            t.set_description("Epoch: %i" %i)
            t.set_postfix(train_loss='%.4f'%loss.item())
            t.update(1)
        
    model.eval()
    score = 0
    with tqdm(total=len(test_dataloader)) as t:
        for idx, (X, y) in enumerate(test_dataloader):
            score += model.predict(X.to('cuda'), y.to('cuda'))
            t.set_description("Test: %i" %i)
            t.set_postfix(test_acc='%.4f'%(score/float(idx+1)))
            t.update(1)
    
    acc.append(score/float(len(test_dataloader)))
    
plt.figure()
plt.plot(train_losses)
plt.ylabel("Loss:")
plt.xlabel("Steps:")
plt.show()
plt.savefig("./pca_classifier_loss.png")  
        
plt.figure()
plt.plot(acc)
plt.ylabel("Acc:")
plt.xlabel("Epoch:")
plt.show()
plt.savefig("./pca_classifier_acc.png")  
        
        
        