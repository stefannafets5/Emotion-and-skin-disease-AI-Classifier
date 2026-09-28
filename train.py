import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset_loader import SkinDataset, EmotionDataset, get_dynamic_transforms
from models import SimpleMLP, SimpleCNN
from finetune2 import run_resnet_finetuning
from vizualize_augumentations import generate_augmentations
from common import evaluate_model, generate_final_eval_metrics, train_one_epoch 

def run_experiment(model_type='cnn', dataset_type='skin', epochs=10, batch_size=32, lr=0.001, use_dropout=True, use_batch_norm=True, aug_type='all', use_imbalance_fix=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_name = f"{model_type}_dr{int(use_dropout)}_bn{int(use_batch_norm)}_aug_{aug_type}_bal_{int(use_imbalance_fix)}"
    print(f"RUNNING: {experiment_name.upper()} on {dataset_type.upper()}")
    
    if dataset_type == 'skin':
        # Default local paths (update these if running on Kaggle or custom directories)
        train_csv = 'vai-de-pielea-mea/train.csv'
        train_dir = 'vai-de-pielea-mea/train'
        test_csv = 'vai-de-pielea-mea/test.csv'
        test_dir = 'vai-de-pielea-mea/test'
        
        img_size, num_classes = 128, 7
        class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
            
        train_transform = get_dynamic_transforms(dataset_type='skin', img_size=img_size, aug_type=aug_type)
        val_transform = get_dynamic_transforms(dataset_type='skin', img_size=img_size, aug_type='none')
        
        train_set = SkinDataset(train_csv, train_dir, transform=train_transform)
        val_set = SkinDataset(test_csv, test_dir, transform=val_transform)
        
    elif dataset_type == 'emotion':
        train_csv = 'you-re-on-candid-camera/splits/train.csv'
        test_csv = 'you-re-on-candid-camera/splits/local_test.csv'
        base_dir = 'you-re-on-candid-camera'
        
        img_size, num_classes = 100, 7
        class_names = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
            
        train_transform = get_dynamic_transforms(dataset_type='emotion', img_size=img_size, aug_type=aug_type)
        val_transform = get_dynamic_transforms(dataset_type='emotion', img_size=img_size, aug_type='none')
        
        train_set = EmotionDataset(train_csv, base_dir, transform=train_transform)
        val_set = EmotionDataset(test_csv, base_dir, transform=val_transform)

    sampler = None
    shuffle_mode = True
    if use_imbalance_fix:
        print("Applying WeightedRandomSampler to address class imbalance...")
        if dataset_type == 'skin':
            labels = np.array([train_set.label_map[item] for item in train_set.data_info.iloc[:, 1]])
        else:
            labels = np.array([int(item) - 1 for item in train_set.data_info.iloc[:, 1]])
            
        class_counts = np.bincount(labels)
        class_weights = 1.0 / np.maximum(1, class_counts)
        sample_weights = class_weights[labels]
        
        sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights).double(), num_samples=len(sample_weights), replacement=True)
        shuffle_mode = False
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_mode, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize Model
    if model_type == 'mlp':
        input_size = 3 * img_size * img_size
        model = SimpleMLP(input_size=input_size, num_classes=num_classes, use_dropout=use_dropout).to(device)
    else:
        model = SimpleCNN(num_classes=num_classes, use_batch_norm=use_batch_norm).to(device)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1:02d}/{epochs} ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    # Final Evaluation
    generate_final_eval_metrics(model, val_loader, device, class_names, history, dataset_type, experiment_name)

    return history

def run_all_combinations(dataset_type):
    # MLP with / without Dropout
    # run_experiment(model_type='mlp', dataset_type=dataset_type, epochs=epochs_to_run, use_dropout=True)
    # run_experiment(model_type='mlp', dataset_type=dataset_type, epochs=epochs_to_run, use_dropout=False)
    
    # # CNN with / without BatchNorm
    # run_experiment(model_type='cnn', dataset_type=dataset_type, epochs=epochs_to_run, use_batch_norm=True)
    # run_experiment(model_type='cnn', dataset_type=dataset_type, epochs=epochs_to_run, use_batch_norm=False)
    
    # # CNN with different augumentations
    # run_experiment(model_type='cnn', dataset_type=dataset_type, epochs=epochs_to_run, aug_type='none')
    # run_experiment(model_type='cnn', dataset_type=dataset_type, epochs=epochs_to_run, aug_type='geometric')
    # run_experiment(model_type='cnn', dataset_type=dataset_type, epochs=epochs_to_run, aug_type='color')
    # run_experiment(model_type='cnn', dataset_type=dataset_type, epochs=epochs_to_run, aug_type='all')
    
    # # with / without WeightedRandomSampler
    # run_experiment(model_type='cnn', dataset_type=dataset_type, epochs=epochs_to_run, use_imbalance_fix=False)
    # run_experiment(model_type='cnn', dataset_type=dataset_type, epochs=epochs_to_run, use_imbalance_fix=True)

    if dataset_type == 'skin':
        train_csv, test_csv = 'vai-de-pielea-mea/train.csv', 'vai-de-pielea-mea/test.csv'
        train_dir, test_dir = 'vai-de-pielea-mea/train', 'vai-de-pielea-mea/test'
        classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        train_dataset = SkinDataset(train_csv, train_dir, transform=None)
        val_dataset = SkinDataset(test_csv, test_dir, transform=None)
    elif dataset_type == 'emotion':
        train_csv = 'you-re-on-candid-camera/splits/train.csv'
        test_csv = 'you-re-on-candid-camera/splits/local_test.csv'
        base_dir = 'you-re-on-candid-camera'
        classes = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
        train_dataset = EmotionDataset(train_csv, base_dir, transform=None)
        val_dataset = EmotionDataset(test_csv, base_dir, transform=None)
        
    # Finetuning ResNet with / without warmup and with / without freezing
    run_resnet_finetuning(train_dataset=train_dataset, val_dataset=val_dataset, class_names=classes, dataset_name=dataset_type, use_warmup=True, freeze=True, epochs=epochs_to_run, batch_size=32)
    run_resnet_finetuning(train_dataset=train_dataset, val_dataset=val_dataset, class_names=classes, dataset_name=dataset_type, use_warmup=False, freeze=True, epochs=epochs_to_run, batch_size=32)
    run_resnet_finetuning(train_dataset=train_dataset, val_dataset=val_dataset, class_names=classes, dataset_name=dataset_type, use_warmup=True, freeze=False, epochs=epochs_to_run, batch_size=32)
    run_resnet_finetuning(train_dataset=train_dataset, val_dataset=val_dataset, class_names=classes, dataset_name=dataset_type, use_warmup=False, freeze=False, epochs=epochs_to_run, batch_size=32)

if __name__ == "__main__":

    epochs_to_run = 30
    # generate_augmentations()
    run_all_combinations('skin')
    run_all_combinations('emotion')
    