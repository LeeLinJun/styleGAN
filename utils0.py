import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torchvision.transforms as trans
import torch.optim as optim

def preprocess_img(x):
    return 2.0 * (x - 0.5)

def deprocess_img(x):
    return (x + 1.0) / 2.0

class Flatten(nn.Module):
    def forward(self, x):
        N, _, _, _ = x.size()
        return x.view(N, -1)
    
class Unflatten(nn.Module):
    def __init__(self, N, C, H, W):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)

def sample_noise(batch_size, noise_dim):
    return 2.0 * (torch.rand(batch_size, noise_dim) - 0.5)

def dc_generator(noise_dim=100):
    return nn.Sequential(
        nn.Linear(noise_dim, 1024 * 4 * 4),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(1024 * 4 * 4),
        Unflatten(-1, 1024, 4, 4),
        nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(512),
        nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(256),
        nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=5, stride=2, padding=2, output_padding=1),
        nn.Tanh()
    )

def dc_discriminator():
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.BatchNorm2d(64),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.BatchNorm2d(128),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.BatchNorm2d(256),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.BatchNorm2d(512),
        Flatten(),
        nn.Linear(512 * 4 * 4, 1),
        nn.Sigmoid()
    )

# loss with soft label
def ls_discriminator_loss(scores_real, scores_fake):
    soft_real = torch.zeros_like(scores_real).uniform_(0.78, 1.12)
    soft_fake = torch.zeros_like(scores_fake).uniform_(0.0, 0.24)
    loss = 0.5 * ((soft_real - scores_real)**2 + (soft_fake - scores_fake)**2)
    return loss.mean()

def ls_generator_loss(scores_fake):
    soft_real = torch.zeros_like(scores_fake).uniform_(0.78, 1.12)
    loss = 0.5 * (soft_real - scores_fake)**2
    return loss.mean()

def get_optimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    return optimizer

to32 = trans.Compose([
    trans.Resize(32),
    trans.CenterCrop(32),
    trans.ToTensor()
])

to64 = trans.Compose([
    trans.Resize(64),
    trans.CenterCrop(64),
    trans.ToTensor()
])

to112 = trans.Compose([
    trans.Resize(112),
    trans.CenterCrop(112),
    trans.ToTensor()
])
