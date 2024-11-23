import torch
import torch.nn as nn
import  torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from autoencoder import AE, VAE, add_noise
from data_prepare import minist_dataset_make
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

device = 'cuda'
model = AE().to('cuda')
train_dataloader, test_dataloader = minist_dataset_make(4)


if sys.argv[1]:
    if sys.argv[1] == 'VAE':
        model = VAE().to('cuda')


train_mode = 'normal'
if sys.argv[2]:
    if sys.argv[2] == 'Denoise':
        train_mode = 'Denoise'
        
model_path = sys.argv[3]
model.load_state_dict(torch.load(model_path))
        
        
transform = transforms.Compose(
    [transforms.ToTensor()])
test_dataset=torchvision.datasets.MNIST(root="./dataset",train=False,transform=transform,download=True)



@torch.no_grad()
def plot_ae_outputs(n=5):
  plt.figure(figsize=(10,4.5))
  for i in range(n):
    ax = plt.subplot(3,n,i+1)
    img = test_dataset[i][0].unsqueeze(0)
    
    if isinstance(model, AE):
        rec_img  = model(img.to(device))
    elif isinstance(model, VAE):
        rec_img, kld  = model(img.to(device))
    plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    if i == n//2:
        ax.set_title('Original images')
        
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    if i == n//2:
        ax.set_title('Reconstructed images')
          
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.7,
                        top=0.9,
                        wspace=0.3,
                        hspace=0.3)
    plt.show() 
    plt.savefig("./"+model.__class__.__name__+"_"+train_mode+"_visual.png")


@torch.no_grad()
def plot_ae_outputs_den(n=5,noise_factor=0.3):
  plt.figure(figsize=(10,4.5))
  for i in range(n):
    ax = plt.subplot(3,n,i+1)
    img = test_dataset[i][0].unsqueeze(0)
    image_noisy = add_noise(img,noise_factor)
    image_noisy = image_noisy.to(device)

    if isinstance(model, AE):
        rec_img  = model(img.to(device))
    elif isinstance(model, VAE):
        rec_img, kld  = model(img.to(device))
    plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == n//2:
        ax.set_title('Original images')
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(image_noisy.cpu().squeeze().numpy(), cmap='gist_gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == n//2:
        ax.set_title('Corrupted images')
    ax = plt.subplot(3, n, i + 1 + n + n)
    plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == n//2:
        ax.set_title('Reconstructed images')
        
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.7,
                        top=0.9,
                        wspace=0.3,
                        hspace=0.3)
    plt.show() 
    
    plt.savefig("./"+model.__class__.__name__+"_"+train_mode+"_visual.png")
    
            
            
if train_mode == 'normal':
    plot_ae_outputs()
elif train_mode == 'Denoise':
    plot_ae_outputs_den()