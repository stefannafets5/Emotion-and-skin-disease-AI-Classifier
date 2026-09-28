import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dataset_loader import SkinDataset, EmotionDataset, get_dynamic_transforms

def save_augmentation_montage(dataset, label_names, output_filename, title):
    """
    Finds the first image belonging to each class, applies augmentations,
    and plots the original vs augmented image side-by-side.
    """
    num_classes = len(label_names)
    fig, axes = plt.subplots(num_classes, 2, figsize=(8, 2 * num_classes))
    plt.suptitle(title, fontsize=14, y=0.98)
    
    found_indices = {}
    
    # Scan dataset to find the first image for each class index
    for idx in range(len(dataset)):
        # To get the original image, we temporarily bypass the transform
        orig_transform = dataset.transform
        dataset.transform = None
        orig_img, label = dataset[idx]
        dataset.transform = orig_transform
        
        if label not in found_indices:
            found_indices[label] = idx
        if len(found_indices) == num_classes:
            break
            
    # plot each class
    for class_idx in range(num_classes):
        if class_idx in found_indices:
            dataset_idx = found_indices[class_idx]
            
            # 1. Original Image (temporary read without transforms)
            orig_transform = dataset.transform
            dataset.transform = None
            orig_img_tensor, _ = dataset[dataset_idx]
            dataset.transform = orig_transform
            
            orig_img = Image.fromarray(orig_img_tensor)
            aug_img_tensor, _ = dataset[dataset_idx]
            
            # Inverse normalization of the augmented image tensor to plot it properly
            if "Skin" in str(type(dataset)):
                # mean=(0.7646, 0.5464, 0.5712), std=(0.1410, 0.1532, 0.1706)
                img_np = aug_img_tensor.permute(1, 2, 0).numpy()
                img_np = (img_np * [0.1410, 0.1532, 0.1706] + [0.7646, 0.5464, 0.5712])
            else:
                # mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
                img_np = aug_img_tensor.permute(1, 2, 0).numpy()
                img_np = (img_np * 0.5 + 0.5)
                
            img_np = np.clip(img_np, 0, 1)
            
            # Plot Original
            axes[class_idx, 0].imshow(orig_img)
            axes[class_idx, 0].set_ylabel(label_names[class_idx], fontsize=10, rotation=0, labelpad=40, ha='right')
            if class_idx == 0:
                axes[class_idx, 0].set_title("Original", fontsize=12)
            axes[class_idx, 0].set_xticks([])
            axes[class_idx, 0].set_yticks([])
            
            # Plot Augmented
            axes[class_idx, 1].imshow(img_np)
            if class_idx == 0:
                axes[class_idx, 1].set_title("Augmented", fontsize=12)
            axes[class_idx, 1].set_xticks([])
            axes[class_idx, 1].set_yticks([])
        else:
            # Placeholder if a class has no samples in the selection
            axes[class_idx, 0].text(0.5, 0.5, f"No sample found", ha='center', va='center')
            axes[class_idx, 1].text(0.5, 0.5, f"No sample found", ha='center', va='center')
            axes[class_idx, 0].set_axis_off()
            axes[class_idx, 1].set_axis_off()
            
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization montage to {output_filename}")

def generate_augmentations():
    # 1. Skin Dataset Config
    skin_csv = 'vai-de-pielea-mea/train.csv'
    skin_dir = 'vai-de-pielea-mea/train'
    skin_classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    
    # 2. Emotion Dataset Config
    emotion_csv = 'you-re-on-candid-camera/splits/train.csv'
    emotion_dir = 'you-re-on-candid-camera'
    emotion_classes = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
    
    # Process Skin Augmentations
    print("Processing skin lesion dataset augmentations...")
    skin_transform = get_dynamic_transforms(dataset_type='skin', img_size=128, aug_type='all')
    skin_dataset = SkinDataset(skin_csv, skin_dir, transform=skin_transform)
    save_augmentation_montage(
        dataset=skin_dataset,
        label_names=skin_classes,
        output_filename='skin_augmentations.png',
        title='Skin Mole Dataset Augmentations (3.1)'
    )
        
    # Process Emotion Augmentations
    print("Processing emotion facial expression dataset augmentations...")
    emotion_transform = get_dynamic_transforms(dataset_type='emotion', img_size=100, aug_type='all')
    emotion_dataset = EmotionDataset(emotion_csv, emotion_dir, transform=emotion_transform)
    save_augmentation_montage(
        dataset=emotion_dataset,
        label_names=emotion_classes,
        output_filename='emotion_augmentations.png',
        title='Emotion Dataset Augmentations (3.1)'
    )

