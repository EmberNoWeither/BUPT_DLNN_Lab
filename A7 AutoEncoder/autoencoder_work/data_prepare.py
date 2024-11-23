import  torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader


def minist_dataset_make(train_batch_size=256, num_workers=8, resize=None, device='cuda'):
    transform = transforms.Compose(
    [transforms.ToTensor()])
    
    if resize:
        transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(resize)])

    train_set=torchvision.datasets.MNIST(root="./dataset",train=True,transform=transform,download=True)
    test_set=torchvision.datasets.MNIST(root="./dataset",train=False,transform=transform,download=True)
    
    train_dataloader = DataLoader(dataset=train_set,
                                  batch_size=train_batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    
    test_dataloader = DataLoader(dataset=test_set,
                                batch_size=32,
                                shuffle=False,
                                num_workers=8)

    return train_dataloader, test_dataloader