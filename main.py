
from collections import OrderedDict
from train import train
import numpy as np
import torch.nn as nn
from models.style_verbose import G_mapping,G_synthesis
from models.styleGan import Discriminator,StyleBased_Generator
from dataset import return_cifar10
import os

def check_paths():
    base_dir = os.getcwd()
    paths = ['generated_images','model_weights']
    for path in paths:
        dir_path = os.path.join(base_dir,path)
        if not os.path.isdir(path): 
            os.mkdir(dir_path) 

def main():
    check_paths()
    # Load data
    cifar10_path = '/home/shuza/Documents/Data/cifar-10-batches-py'
    train_data,test_data = return_cifar10(cifar10_path)
    train_x,train_labels = train_data
    test_x,test_labels = test_data

    params = {
        'base_dir': os.getcwd(),
        'image_dir' : os.path.join(os.getcwd(),'generated_images'),
        'weight_dir' : os.path.join(os.getcwd(),'model_weights'),
        'iterations':1000,
        'batch_size':1,
        'latent_dim':512,
        'alpha': 0,
        'resolution':32,
        'step' : 3
    }
    datasets = {
        'x_train':train_x.transpose(0,3,2,1),
        'x_test':test_x.transpose(0,3,2,1),
        'train_labels':train_labels,
        'test_labels':test_labels
    }
    # generator = nn.Sequential(OrderedDict([
    #     ('g_mapping', G_mapping()),
    #     #('truncation', Truncation(avg_latent)),
    #     ('g_synthesis', G_synthesis())    
    # ]))
    # Model params
    n_fc = 8
    dim_latent = 512
    dim_input = 4

    generator = StyleBased_Generator(n_fc,dim_latent,dim_input)
    discriminator = Discriminator()
    train(params,datasets,generator,discriminator)

if __name__ == "__main__":
    main()