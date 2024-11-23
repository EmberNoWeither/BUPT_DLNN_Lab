import torch
import numpy as np
import PIL.Image as Image
import torchvision
import pandas as pd
import cv2
from os import walk
np.set_printoptions(threshold=np.inf)

#  为在某些准备环节上方便实验，torch等框架仅用于数据集读取操作，主体框架用numpy实现
def load_dataset(batch_size = 2, test_batch_size = 50):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    path = './data/'
    batch_size = batch_size
    x_train, y_train, x_test, y_test = [], [], [], []
    trainData = torchvision.datasets.MNIST(path,train = True,transform=transform,download = True)
    testData = torchvision.datasets.MNIST(path,transform=transform,train = False)
    trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=batch_size,shuffle=True)
    testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=test_batch_size)

    for idx, data_batch in enumerate(trainDataLoader):
        x = data_batch[0].numpy()/255.0
        y = data_batch[1].reshape(1, -1)
        y = np.squeeze(y)
        Y = []
        for yl in y:
            yy = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            yy[yl] = 1.0
            Y.append(yy)
        Y = np.array(Y)
        x_train.append(x)
        y_train.append(Y)

    for idx, data_batch in enumerate(testDataLoader):
        x = data_batch[0].numpy()/255.0
        y = data_batch[1].reshape(1, -1)
        y = np.squeeze(y)
        Y = []
        if test_batch_size == 1:
            for yl in [y]:
                yy = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                yy[yl] = 1
                Y.append(yy)
        else:
            for yl in y:
                yy = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                yy[yl] = 1
                Y.append(yy)
        Y = np.array(Y)
        x_test.append(x)
        y_test.append(Y)
        
    print(len(x_test))

    return x_train, y_train, x_test[:100], y_test[:100], x_test[100:], y_test[100:]
