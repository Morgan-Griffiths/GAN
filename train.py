from tqdm import tqdm
import torch.nn as nn
import torch
import os
from PIL import Image
from torch import optim

def imsave(path,tensor, i):
    grid = tensor[0]
    grid.clamp_(-1, 1).add_(1).div_(2)
    # Add 0.5 after normalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    img = Image.fromarray(ndarr)
    img.save(f'{path}/generated_frog_{i}.png')

def load_weights(path,policy):
    policy.load_state_dict(torch.load(path))
    policy.eval()
    return policy

def save_weights(path,policy):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.mkdir(directory)
    torch.save(policy.state_dict(), path)
    
def set_grad_flag(module, flag):
    for p in module.parameters():
        p.requires_grad = flag

def train(params, datasets, generator, discriminator, d_losses =[],g_losses=[]):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Training on {device}')
    generator.to(device)
    discriminator.to(device)
    d_optim        = optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.0, 0.99))
    g_optim        = optim.Adam(generator.parameters(),lr=0.001, betas=(0.0, 0.99))
    base_dir,image_dir,weight_dir,iterations,batch_size,latent_dim,alpha,resolution = params.values()
    start=0
    progress_bar = tqdm(total=iterations, initial=start)
    for step in range(0,iterations+1):
        progress_bar.update(1)
        stop=start + batch_size
        real_images = torch.tensor(datasets['x_train'][start:stop]).float().to(device)
        # Train discriminator first
        discriminator.zero_grad()
        set_grad_flag(discriminator, True)
        set_grad_flag(generator, False)
        real_images.requires_grad = True
        # print('real_images',real_images.size())
        real_predict = discriminator(real_images, 0, alpha)
        real_predict = nn.functional.softplus(-real_predict).mean()
        real_predict.backward(retain_graph=True)
        
        grad_real = torch.autograd.grad(outputs=real_predict.sum(), inputs=real_images, create_graph=True)[0]
        grad_penalty_real = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
        grad_penalty_real = 10 / 2 * grad_penalty_real
        grad_penalty_real.backward()
        
        latent_w1 = [torch.randn(1, latent_dim, device=device)]
        latent_w2 = [torch.randn(1, latent_dim, device=device)]
        # print('latent_w1',latent_w1[0].size())
        noise_1 = []
        noise_2 = []
        for m in range(0 + 1):
            size = 4 * 2 ** m # Due to the upsampling, size of noise will grow
            noise_1.append(torch.randn((resolution, 1, size, size), device=device))
            noise_2.append(torch.randn((resolution, 1, size, size), device=device))
        
        fake_image = generator(latent_w1[0])#, step, alpha, noise_1)
        # print('fake_image',fake_image.size())
        fake_predict = discriminator(fake_image, 0, alpha)
        fake_predict = nn.functional.softplus(fake_predict).mean()
        fake_predict.backward()
        
        if step % 20 == 0:
            d_losses.append((real_predict + fake_predict).item())
            
        d_optim.step()
        # Avoid possible memory leak
        del grad_penalty_real, grad_real, fake_predict, real_predict, fake_image, real_images, latent_w1
        generator.zero_grad()
        set_grad_flag(discriminator, False)
        set_grad_flag(generator, True)
        fake_image = generator(latent_w2[0])#, step, alpha, noise_2)
        # print('fake_image 2',fake_image.size())
        fake_predict = discriminator(fake_image, 0, alpha)
        fake_predict = nn.functional.softplus(-fake_predict).mean()
        fake_predict.backward()
        
        if step % 20 == 0:
            g_losses.append(fake_predict.item())
        
        g_optim.step()
        
        if step % 100 == 0:
            save_weights(f'{weight_dir}/generator_{step}.ckpt',generator)
            save_weights(f'{weight_dir}/discriminator_{step}.ckpt',discriminator)
            imsave(image_dir,fake_image.data.cpu(), step)
        del fake_predict, fake_image, latent_w2
        progress_bar.set_description((f'Resolution: {resolution}*{resolution}  D_Loss: {d_losses[-1]:.4f}  G_Loss: {g_losses[-1]:.4f}  Alpha: {alpha:.4f}'))
        start += 1