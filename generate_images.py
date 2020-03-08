import torch
from matplotlib import pyplot
import torchvision
from models.style_verbose import G_synthesis,G_mapping

g_all = nn.Sequential(OrderedDict([
    ('g_mapping', G_mapping()),
    #('truncation', Truncation(avg_latent)),
    ('g_synthesis', G_synthesis())    
]))

# Optionally load stylegan weights
# g_all.load_state_dict(torch.load('/Users/morgan/Downloads/karras2019stylegan-ffhq-1024x1024.for_g_all.pt'))

torch.manual_seed(6)
nb_rows = 2
nb_cols = 2
nb_samples = nb_rows * nb_cols
latents = torch.randn(nb_samples, 512, device=device)
print(latents)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
g_all.eval()
g_all.to(device)

torch.manual_seed(7)
nb_rows = 2
nb_cols = 2
nb_samples = nb_rows * nb_cols
latents = torch.randn(nb_samples, 512, device=device)
with torch.no_grad():
    imgs = g_all(latents)
    imgs = (imgs.clamp(-1, 1) + 1) / 2.0 # normalization to 0..1 range
imgs = imgs.cpu()

print(imgs.shape)
imgs = torchvision.utils.make_grid(imgs, nrow=nb_cols)
pyplot.figure(figsize=(15, 6))
pyplot.imshow(imgs.permute(1, 2, 0).detach().numpy())