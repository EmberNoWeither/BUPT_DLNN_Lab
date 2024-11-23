import  torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold


def dataset_make(train_batch_size=256, num_workers=8, resize=None, device='cuda'):
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    if resize:
        transform = transforms.Compose(
    [transforms.Resize(resize),
     transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set=torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=transform,download=True)
    test_set=torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=transform,download=True)
    
    train_dataloader = DataLoader(dataset=train_set,
                                  batch_size=train_batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    
    test_dataloader = DataLoader(dataset=test_set,
                                batch_size=128,
                                shuffle=False,
                                num_workers=2,
                                drop_last=False)

    return train_dataloader, test_dataloader, train_set


def minist_dataset_make(train_batch_size=256, num_workers=8, resize=None, device='cuda'):
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))])
    
    if resize:
        transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(resize),
    transforms.Normalize((0.5), (0.5))])

    train_set=torchvision.datasets.MNIST(root="./dataset",train=True,transform=transform,download=True)
    test_set=torchvision.datasets.MNIST(root="./dataset",train=False,transform=transform,download=True)
    
    train_dataloader = DataLoader(dataset=train_set,
                                  batch_size=train_batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    
    test_dataloader = DataLoader(dataset=test_set,
                                batch_size=128,
                                shuffle=False,
                                num_workers=2)

    return train_dataloader, test_dataloader, train_set


# if __name__ == '__main__':

#     train_dataloader, test_dataloader = dataset_make()
    
#     print(len(test_dataloader))

#     for idx, (X, Y) in enumerate(test_dataloader):
#         print(Y.shape)
#         print(Y)
#         break