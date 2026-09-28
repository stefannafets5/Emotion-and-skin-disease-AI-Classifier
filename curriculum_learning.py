import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from dataset_loader import EmotionDataset
from finetune2 import build_finetuning_model, get_finetune_transforms, setup_schedulers
from common import train_one_epoch, evaluate_model  

@torch.no_grad()
def get_per_sample_losses(model, dataset, criterion_no_reduction, device, batch_size=64):
    model.eval()
    losses = []
    # Use a clean standard loader without shuffling or dropping last
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion_no_reduction(logits, labels)
        losses.append(loss.cpu())
    return torch.cat(losses)

def run_curriculum_experiment(epochs=15, batch_size=32):    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    train_csv = 'you-re-on-candid-camera/splits/train.csv'
    val_csv = 'you-re-on-candid-camera/splits/local_test.csv'
    base_dir = 'you-re-on-candid-camera'

    num_classes = 7
    class_names = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
    
    # BASELINE OVERVIEW
    print("\n[1/3] BASELINE TRAINING (WITHOUT CURRICULUM)")
    train_transform, val_transform = get_finetune_transforms()
    train_dataset = EmotionDataset(train_csv, base_dir, transform=train_transform)
    val_dataset = EmotionDataset(val_csv, base_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    baseline_model, baseline_opt = build_finetuning_model(num_classes=num_classes, freeze=False)
    baseline_model = baseline_model.to(device)
    baseline_sched = setup_schedulers(baseline_opt, warmup_epochs=2, total_epochs=epochs, use_warmup=True)
    criterion = nn.CrossEntropyLoss()
    
    baseline_accs = []
    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(baseline_model, train_loader, criterion, baseline_opt, device)
        val_loss, val_acc = evaluate_model(baseline_model, val_loader, criterion, device)
        baseline_sched.step()
        baseline_accs.append(val_acc)
        print(f"  Epoch {epoch+1:02d}/{epochs} | Val Acc Baseline: {val_acc:.4f} (Loss: {val_loss:.4f})")
        
    print("\n[2/3] SELF-PACED LEARNING TRAINING (LOSS-BASED)")
    sp_model, sp_opt = build_finetuning_model(num_classes=num_classes, freeze=False)
    sp_model = sp_model.to(device)
    sp_sched = setup_schedulers(sp_opt, warmup_epochs=2, total_epochs=epochs, use_warmup=True)
    
    criterion_no_reduction = nn.CrossEntropyLoss(reduction="none")
    sp_accs = []
    
    for epoch in range(epochs):
        per_sample_losses = get_per_sample_losses(sp_model, train_dataset, criterion_no_reduction, device, batch_size)
        
        # 2. Define the percentile of active easy samples: grows progressively
        q_limit = min(1.0, 0.30 + 0.05 * epoch)
        threshold = torch.quantile(per_sample_losses, q=q_limit)
        
        # 3. Filter only the indices that have a difficulty (loss) below the threshold
        easy_indices = (per_sample_losses <= threshold).nonzero(as_tuple=True)[0].tolist()
        
        # 4. Create a Subset and the corresponding loader for this epoch
        epoch_subset = Subset(train_dataset, easy_indices)
        epoch_loader = DataLoader(epoch_subset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True if len(epoch_subset) > batch_size else False)
        
        # 5. Model training step on selected subset
        tr_loss, tr_acc = train_one_epoch(sp_model, epoch_loader, criterion, sp_opt, device)
        val_loss, val_acc = evaluate_model(sp_model, val_loader, criterion, device)
        sp_sched.step()
        sp_accs.append(val_acc)
        print(f"  Epoch {epoch+1:02d}/{epochs} (Fraction {q_limit:.2f}) | Val Acc SPL: {val_acc:.4f} (Active Samples: {len(easy_indices)})")

    print(f"Epoch  | Baseline Acc | Self-Paced Acc | Curriculum Gain")
    for ep in range(epochs):
        diff = sp_accs[ep] - baseline_accs[ep]
        print(f" {ep+1:02d}   |    {baseline_accs[ep]:.4f}    |     {sp_accs[ep]:.4f}     |     {diff:+.4f}")
    
    avg_gain = np.mean([sp_accs[ep] - baseline_accs[ep] for ep in range(epochs)])
    print(f"Average general gain with Curriculum during training: {avg_gain:+.4f}")

if __name__ == "__main__":
    run_curriculum_experiment(epochs=10, batch_size=32)
