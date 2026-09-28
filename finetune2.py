import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

from common import train_one_epoch, evaluate_model, generate_final_eval_metrics

def get_finetune_transforms():
    # ResNet-18 expects 3-channel 224x224 images normalized using ImageNet-1K statistics.
    train_transform = transforms.Compose([
        # Resize to a larger area to allow random cropping variations
        transforms.Resize((256, 256)),
        # Extract a random crop of 224x224
        transforms.RandomCrop((224, 224)),
        # Standard augmentations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        # Conversion to normalized Tensor representation
        transforms.ToTensor(),
        # ImageNet-1K normalization for pre-trained weights
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        # Deterministic resizing for evaluation without random crops or flips
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform

def build_finetuning_model(num_classes=7, freeze=True):
    # Load the ResNet-18 architecture with pre-trained ImageNet-1K weights
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)


    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if freeze:
        optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    else:
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if name.startswith("fc."):
                head_params.append(param)
            else:
                backbone_params.append(param)
                
        # small LR on backbone (1e-4) and 1e-3 on new fc
        optimizer = optim.Adam([{"params": backbone_params, "lr": 1e-4}, {"params": head_params, "lr": 1e-3}], weight_decay=1e-4)

    return model, optimizer

def setup_schedulers(optimizer, warmup_epochs=3, total_epochs=30, use_warmup=True):
    if use_warmup:
        warmup_fn = lambda epoch: (epoch + 1) / warmup_epochs
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_fn)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_epochs - warmup_epochs))
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    else: # no warmup
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)
    
    return scheduler

def run_resnet_finetuning(train_dataset, val_dataset, class_names, dataset_name='skin', use_warmup=True, freeze = True ,batch_size=64, epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warmup_status = "WITH_WARMUP" if use_warmup else "NO_WARMUP"
    freeze_status = "FROZEN" if freeze else "UNFROZEN"
    print(f"\n--- RUNNING RESNET FINE-TUNING ({freeze_status.upper()} Backbone | {warmup_status}) ON {dataset_name.upper()} ---")
    
    train_transform, val_transform = get_finetune_transforms()
    train_dataset.transform = train_transform
    val_dataset.transform = val_transform
    
    # Bind loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Setup computational layers
    model, optimizer = build_finetuning_model(num_classes=len(class_names), freeze=freeze)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    scheduler = setup_schedulers(optimizer, warmup_epochs=3, total_epochs=epochs, use_warmup=use_warmup)
    
    # Info for analytical curve
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr_log": []}

    for epoch in range(epochs):
        start_time = time.time()
        # Capture current Backbone Learning Rate for TensorBoard or logging
        current_lr = optimizer.param_groups[0]["lr"]
        history["lr_log"].append(current_lr)
        
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        # Advance Sequential Scheduler progression step
        scheduler.step()
        
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1:02d}/{epochs} ({epoch_time:.1f}s) | "
              f"LR: {current_lr:.5f} | "
              f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Final Evaluation
    generate_final_eval_metrics(model, val_loader, device, class_names, history, dataset_name, f"resnet_{warmup_status.lower()}_{freeze_status}")

    return model, history