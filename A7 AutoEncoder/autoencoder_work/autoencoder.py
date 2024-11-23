import torch.nn as nn
import torch


class AE(nn.Module):
    def __init__(self, encoded_space_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
        nn.Conv2d(1, 8, 3, stride=2, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(True),
        
        nn.Conv2d(8, 16, 3, stride=2, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(True),
        
        nn.Conv2d(16, 32, 3, stride=2, padding=0),
        nn.BatchNorm2d(32),
        nn.ReLU(True),
        
        nn.Flatten(start_dim=1),
        nn.Linear(3 * 3 * 32, 128),
        nn.ReLU(True),
        nn.Linear(128, encoded_space_dim)
        )
        
        
        self.decoder = nn.Sequential(
          nn.Linear(encoded_space_dim, 128),
          nn.ReLU(True),
          nn.Linear(128, 3 * 3 * 32),
          nn.ReLU(True),
          nn.Unflatten(dim=1,unflattened_size=(32, 3, 3)),
          nn.ConvTranspose2d(32, 16, 3,
                             stride=2, output_padding=0),
          nn.BatchNorm2d(16),
          nn.ReLU(True),
          nn.ConvTranspose2d(16, 8, 3, stride=2,
                             padding=1, output_padding=1),
          nn.BatchNorm2d(8), 
          nn.ReLU(True),
          nn.ConvTranspose2d(8, 1, 3, stride=2,
                             padding=1, output_padding=1)
        )


    def forward(self, x):
        """
        :param [b, 1, 28, 28]:
        :return [b, 1, 28, 28]:
        """
        batchsz = x.size(0)
        # encode
        x = self.encoder(x)
        # decode
        x = self.decoder(x)
        # reshape
        x = x.view(batchsz, 1, 28, 28)
        return x



class VAE(nn.Module):
    def __init__(self, encoded_space_dim=20):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
        nn.Conv2d(1, 8, 3, stride=2, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(True),
        
        nn.Conv2d(8, 16, 3, stride=2, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(True),
        
        nn.Conv2d(16, 32, 3, stride=2, padding=0),
        nn.BatchNorm2d(32),
        nn.ReLU(True),
        
        nn.Flatten(start_dim=1),### Linear section
        nn.Linear(3 * 3 * 32, 128),
        nn.ReLU(True),
        nn.Linear(128, encoded_space_dim)
        )
        
        
        self.decoder = nn.Sequential(
          nn.Linear(encoded_space_dim // 2, 128),
          nn.ReLU(True),
          nn.Linear(128, 3 * 3 * 32),
          nn.ReLU(True),
          nn.Unflatten(dim=1,unflattened_size=(32, 3, 3)),
          nn.ConvTranspose2d(32, 16, 3,
                             stride=2, output_padding=0),
          nn.BatchNorm2d(16),
          nn.ReLU(True),
          nn.ConvTranspose2d(16, 8, 3, stride=2,
                             padding=1, output_padding=1),
          nn.BatchNorm2d(8), 
          nn.ReLU(True),
          nn.ConvTranspose2d(8, 1, 3, stride=2,
                             padding=1, output_padding=1)
        )

    def forward(self, x):
        """
        :param [b, 1, 28, 28]:
        :return [b, 1, 28, 28]:
        """
        batchsz = x.size(0)
        q = self.encoder(x)

        mu, sigma = q.chunk(2, dim=1)
        q = mu + sigma * torch.randn_like(sigma)

        x_hat = self.decoder(q)
        x_hat = x_hat.view(batchsz, 1, 28, 28)

        # KL
        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) +
            torch.pow(sigma, 2) -
            torch.log(1e-8 + torch.pow(sigma, 2)) - 1
        ) / (batchsz*28*28)

        return x_hat, kld



def add_noise(inputs,noise_factor=0.3):
  noisy = inputs+torch.randn_like(inputs) * noise_factor
  noisy = torch.clip(noisy,0.,1.)
  return noisy