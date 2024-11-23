import torch
import torch.nn as nn

from autoencoder import AE, VAE, add_noise
from data_prepare import minist_dataset_make
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

model = AE().to('cuda')

train_dataloader, test_dataloader = minist_dataset_make(32)

if sys.argv[1]:
    if sys.argv[1] == 'VAE':
        model = VAE().to('cuda')

train_mode = 'normal'
if sys.argv[2]:
    if sys.argv[2] == 'Denoise':
        train_mode = 'Denoise'


LR = 5e-4
epochs = 20
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_func = nn.MSELoss()


def Train(img:torch.Tensor):
    optimizer.zero_grad() 
    y = torch.clone(img)
    
    if train_mode == 'Denoise':
        img = add_noise(img).to(img.device)
    
    if isinstance(model, AE):
        re_img = model(img)
        loss = loss_func(re_img, y)
    elif isinstance(model, VAE):
        re_img, kld = model(img)
        loss = loss_func(re_img, y) + kld * 1.0
        
    # loss = loss_func(re_img, y)
    loss.backward()
    optimizer.step()
    return loss_func(re_img, y)
    

@torch.no_grad()
def Val(img):
    y = torch.clone(img)
    
    # if train_mode == 'Denoise':
    #     img = add_noise(img).to(img.device)
    
    if isinstance(model, AE):
        re_img = model(img)
    elif isinstance(model, VAE):
        re_img, kld = model(img)
        
    loss = loss_func(re_img, y)
    return loss


if __name__ == '__main__':
    
    train_losses_cur = []
    test_losses_cur = []
    
    for i in range(epochs):
        model.train()
        train_los = 0.0
        test_los = 0.0
        with tqdm(total=len(train_dataloader)) as t:
            for idx, (img, _) in enumerate(train_dataloader):
                loss = Train(img.to('cuda'))
                train_los += loss.item()
                
                t.set_description("Epoch: %i" %i)
                t.set_postfix(train_loss='%.4f'%loss.item())
                t.update(1)
        
        train_los /= len(train_dataloader)
        train_losses_cur.append(train_los)
        
        
        with tqdm(total=len(test_dataloader)) as t:
            for idx, (img, _) in enumerate(test_dataloader):
                loss = Val(img.to('cuda'))
                test_los += loss.item()
                
                t.set_description("Test: %i" %i)
                t.set_postfix(test_loss='%.4f'%loss.item())
                t.update(1)
        
        test_los /= len(test_dataloader)
        test_losses_cur.append(train_los)
        
    plt.figure()
    plt.plot(train_losses_cur, label='train_loss')
    plt.plot(test_losses_cur, label='test_loss')
    plt.legend(loc='upper right')
    plt.ylabel("Loss:")
    plt.xlabel("Epoch:")
    
    plt.savefig("./"+model.__class__.__name__+"_"+train_mode+"_loss.png")
    
    torch.save(model.state_dict(), "./"+model.__class__.__name__+"_"+train_mode+".pth")