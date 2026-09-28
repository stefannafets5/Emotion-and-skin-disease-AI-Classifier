import os
import pandas as pd
import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from PIL import Image
from albumentations.pytorch import ToTensorV2

class SkinDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        csv_file: path to train.csv or test.csv
        img_dir: folder with immmages (train/ sau test/)
        transform: aplied augumentations
        """
        self.data_info = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # map diagnostic to numers
        self.label_map = { 'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6 }

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_name = self.data_info.iloc[idx, 0] 
        # check for .jpg extension
        if not img_name.endswith('.jpg'):
            img_name += '.jpg'
            
        img_path = os.path.join(self.img_dir, img_name)
        image = np.array(Image.open(img_path).convert("RGB"))
        
        label_name = self.data_info.iloc[idx, 1]
        label = self.label_map[label_name]

        # apply transformations
        if self.transform:
            if "albumentations" in str(type(self.transform)):
                augmented = self.transform(image=image)
                image = augmented['image']
            else:
                # Standard transform
                img_pil = Image.fromarray(image)
                image = self.transform(img_pil)

        return image, label

class EmotionDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        csv_file: path to train.csv or local_test.csv from 'splits'
        root_dir: the base directory where 'DATASETS' is located
        transform: applied augmentations (albumentations)
        """
        self.data_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data_info.iloc[idx, 0])
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Label is 1-7 in CSV, standardizing to 0-6 for PyTorch
        label = int(self.data_info.iloc[idx, 1]) - 1

        if self.transform:
            if "albumentations" in str(type(self.transform)):
                augmented = self.transform(image=image)
                image = augmented['image']
            else:
                # Standard transform
                img_pil = Image.fromarray(image)
                image = self.transform(img_pil)

        return image, label

def get_dynamic_transforms(dataset_type='skin', img_size=128, aug_type='all'):
    if dataset_type == 'skin':
        mean = (0.7646, 0.5464, 0.5712)
        std = (0.1410, 0.1532, 0.1706)
    else:  # emotion
        mean = (0.5752, 0.4497, 0.4013)
        std = (0.2653, 0.2423, 0.2407)

    transforms_list = [A.Resize(img_size, img_size)]

    if aug_type == 'geometric' or aug_type == 'all':
        transforms_list.append(A.HorizontalFlip(p=0.5))
        if dataset_type == 'skin':
            transforms_list.append(A.VerticalFlip(p=0.5))
            transforms_list.append(A.Rotate(limit=180, p=0.5))
        else:
            transforms_list.append(A.Rotate(limit=15, p=0.5))

    if aug_type == 'color' or aug_type == 'all':
        if dataset_type == 'skin':
            transforms_list.append(A.ColorJitter(brightness=0.1, contrast=0.1, p=0.3))
        else:
            transforms_list.append(A.GaussNoise(p=0.2))
            transforms_list.append(A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3))

    # normalize
    transforms_list.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return A.Compose(transforms_list)

def visualize_augmentations(dataset, num_samples=5):
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        image, label = dataset[i]
        # invert normalization
        image = image.permute(1, 2, 0).numpy()
        image = (image * [0.1410, 0.1532, 0.1706] + [0.7646, 0.5464, 0.5712])
        image = np.clip(image, 0, 1)
        
        plt.subplot(1, num_samples, i+1)
        plt.imshow(image)
        plt.title(f"Class: {label}")
        plt.axis('off')
    plt.show()
