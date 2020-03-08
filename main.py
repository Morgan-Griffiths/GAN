
from collections import OrderedDict
from train import train
import numpy as np
import torch.nn as nn
from models.style_verbose import G_mapping,G_synthesis
from models.styleGan import Discriminator,StyleBased_Generator
from dataset import return_cifar10

def main():
    # Load data
    train_data,test_data = return_cifar10()
    train_x,train_labels = train_data
    test_x,test_labels = test_data

    params = {
        'save_dir' : '/Users/morgan/Code/GAN/generated_images',
        'iterations':200,
        'batch_size':1,
        'latent_dim':512,
        'alpha': 0,
        'resolution':32
    }
    datasets = {
        'x_train':train_x.transpose(0,3,2,1),
        'x_test':test_x.transpose(0,3,2,1),
        'train_labels':train_labels,
        'test_labels':test_labels
    }
    generator = nn.Sequential(OrderedDict([
        ('g_mapping', G_mapping()),
        #('truncation', Truncation(avg_latent)),
        ('g_synthesis', G_synthesis())    
    ]))
    discriminator = Discriminator()
    train(params,datasets,generator,discriminator)

if __name__ == "__main__":
    main()