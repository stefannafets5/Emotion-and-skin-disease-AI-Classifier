from matplotlib import pyplot as plt
import numpy as np
import torch


def generate_final_eval_metrics(model, val_loader, device, class_names, history, dataset_name, model_name):
    print("\nGenerating final evaluation metrics...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    accuracy, f1_macro, f1_micro, cm = calculate_metrics(all_preds, all_labels, len(class_names))

    print("\n" + "-"*40)
    print(f"{model_name} on {dataset_name} FINAL Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Micro: {f1_micro:.4f}")
    print("-"*40)
    print(f"Confusion Matrix:\n{cm}")
    print("-"*40)

    # Plot & Save Confusion Matrix visualization
    cm_filename = f"{dataset_name}_{model_name}_confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_filename, f"Confusion Matrix: {model_name.upper()} on {dataset_name.upper()}")

    # Render train history curves
    plot_history(history, f"{dataset_name}_{model_name}_curves.png")

def plot_history(history, filename):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved training history curves to {filename}")

def calculate_metrics(all_preds, all_labels, num_classes):
    # Computes Accuracy, Macro F1, Micro F1, and Confusion Matrix
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 1. Confusion Matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, l in zip(all_preds, all_labels):
        if 0 <= p < num_classes and 0 <= l < num_classes:
            cm[l, p] += 1

    # 2. Accuracy
    accuracy = np.sum(all_preds == all_labels) / max(1, len(all_labels))

    # 3. Macro and Micro F1-Scores
    # F1 Micro = F1 computed globally across all classes
    # F1 Macro = Average of F1 scores computed individually for each class
    
    # Globally count True Positives (TP), False Positives (FP), False Negatives (FN)
    global_tp = 0
    global_fp = 0
    global_fn = 0
    
    class_f1_scores = []
    
    for c in range(num_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        
        global_tp += tp
        global_fp += fp
        global_fn += fn
        
        # Class Precision & Recall
        precision = tp / max(1, (tp + fp))
        recall = tp / max(1, (tp + fn))
        
        # Class F1
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        class_f1_scores.append(f1)

    f1_macro = np.mean(class_f1_scores)
    
    # Global Precision & Recall for micro
    global_precision = global_tp / max(1, (global_tp + global_fp))
    global_recall = global_tp / max(1, (global_tp + global_fn))
    
    if (global_precision + global_recall) > 0:
        f1_micro = 2 * (global_precision * global_recall) / (global_precision + global_recall)
    else:
        f1_micro = 0.0

    return accuracy, f1_macro, f1_micro, cm

def plot_confusion_matrix(cm, class_names, filename, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
            
    fig.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Confusion Matrix plot to {filename}")

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Stats
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / total, correct / total

def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / total, correct / total