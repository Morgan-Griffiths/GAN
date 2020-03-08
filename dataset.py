import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
#         transform = transforms.Compose([
#             transforms.Resize(image_size),          # Resize to the same size
#             transforms.CenterCrop(image_size),      # Crop to get square area
#             transforms.RandomHorizontalFlip(),      # Increase number of samples
#             transforms.ToTensor(),            
#             transforms.Normalize((0.5, 0.5, 0.5),
#                                  (0.5, 0.5, 0.5))])

#         dataset.transform = transform
        self.root_dir = root_dir
        self.image_files = [name for name in os.listdir(root_dir) if name != '.DS_Store']
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.image_files[idx])
        image = torch.tensor(io.imread(img_name))
        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample

"""
CIFAR50
"""

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def stack_cifar10(path):
    data = np.array([])
    labels = np.array([])
    for digit in range(1,6):
        batch = unpickle(path+str(digit))
        # print(batch.keys())
        # print(batch[b'data'].shape)
        if data.size > 0:
            data = np.vstack((data,batch[b'data']))
            labels = np.hstack((labels,batch[b'labels']))
        else:
            data = batch[b'data']
            labels = batch[b'labels']
    return data,labels

def return_selected_class(train_x,test_x,train_labels,test_labels,select=6):
    # 6 == frog
    train = train_x[train_labels == 6]
    test = test_x[test_labels == 6]
    train_l = train_labels[train_labels == 6]
    test_l = test_labels[test_labels == 6]
    return (train,train_l),(test,test_l)
    
def return_cifar10(path):
    print(f'Loading cifar10 path {path}')
    train_path = os.path.join(path,'data_batch_')
    test_path = os.path.join(path,'test_batch')
    train_inputs,train_labels = stack_cifar10(train_path)
    test_data = unpickle(test_path)
    test_inputs = test_data[b'data']
    test_labels = np.array(test_data[b'labels'])
    train_x = train_inputs.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    test_x = test_inputs.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    train,test = return_selected_class(train_x,test_x,train_labels,test_labels)
    print(f'train_x {train_x.shape},train_labels {train_labels.shape},test_x {test_x.shape},test_labels {test_labels.shape}')
    return train,test

def show_img(dataset,index):
    img = dataset[index]
    print(img.shape)
    plt.imshow(img, interpolation='nearest')