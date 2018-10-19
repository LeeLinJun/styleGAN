import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torchvision.transforms as trans
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils0 import *

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

batch_size = 128

def show_images(images, iter_count):
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.transpose(1, 2, 0))
    plt.savefig('./savefigs0/' + str(iter_count - 1) + '.png')
    plt.close(fig)
    return

def show_loss(d_loss, g_loss, iter_count):
    x_axis = np.arange(iter_count)
    fig = plt.figure()
    plt.plot(x_axis, d_loss)
    plt.xlabel('Iteration')
    plt.ylabel('D Loss')
    plt.title('Training Loss')
    plt.savefig('./savefigs0/dloss.png')
    plt.close(fig)
    fig = plt.figure()
    plt.plot(x_axis, g_loss)
    plt.xlabel('Iteration')
    plt.ylabel('G Loss')
    plt.title('Training Loss')
    plt.savefig('./savefigs0/gloss.png')
    plt.close(fig)
    return

def run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, show_every=250,
              batch_size=128, noise_size=100, num_epochs=10):
    iter_count = 1
    d_loss = []
    g_loss = []
    for epoch in range(num_epochs):
        for x, _ in loader_train:
            if len(x) != batch_size:
                continue
            if iter_count % 1000 == 0:
                # learning rate decay
                for param_group in D_solver.param_groups:
                    param_group['lr'] *= 0.85
                for param_group in G_solver.param_groups:
                    param_group['lr'] *= 0.9
                # save model     
                torch.save(D.state_dict(), './savemodels/D_' + str(iter_count) + '.pkl')
                torch.save(G.state_dict(), './savemodels/G_' + str(iter_count) + '.pkl')
                
            D_solver.zero_grad()
            real_data = preprocess_img(x.type(dtype))
            logits_real = D(real_data).type(dtype)

            g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
            fake_images = G(g_fake_seed).detach()
            logits_fake = D(fake_images)

            d_total_error = discriminator_loss(logits_real, logits_fake)
            
            if d_total_error.item() > 0.1:
                d_total_error.backward()        
                D_solver.step()

            G_solver.zero_grad()
            g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
            fake_images = G(g_fake_seed)

            gen_logits_fake = D(fake_images)
            g_error = generator_loss(gen_logits_fake)
            g_error.backward()
            G_solver.step()
            
            d_loss.append(d_total_error.item())
            g_loss.append(g_error.item())
            
            if (iter_count % show_every == 1):
                imgs_numpy = deprocess_img(fake_images.cpu().detach().numpy())
                show_images(imgs_numpy[0:64], iter_count)
                show_loss(d_loss, g_loss, iter_count)

            iter_count += 1

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

lsun_train = dset.LSUN('./datasets/LSUN0', classes=['church_outdoor_train'], transform=to64)
loader_train = DataLoader(lsun_train, batch_size=batch_size, shuffle=True)

# Make the discriminator
D = dc_discriminator().type(dtype)

# Make the generator
G = dc_generator().type(dtype)

# Use the function you wrote earlier to get optimizers for the Discriminator and the Generator
D_solver = get_optimizer(D)
G_solver = get_optimizer(G)

# Run it!
run_a_gan(D, G, D_solver, G_solver, ls_discriminator_loss, ls_generator_loss)