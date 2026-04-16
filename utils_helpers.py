import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, List


def load_image(image_path, target_size=(224, 224)):
    """Load and preprocess image"""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read {image_path}")
        
        # Resize
        img = cv2.resize(img, target_size)
        
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def normalize_image(img, method='standard'):
    """Normalize image intensities"""
    img = img.astype(np.float32)
    
    if method == 'standard':
        # Standard normalization
        mean = img.mean()
        std = img.std()
        img = (img - mean) / (std + 1e-8)
    
    elif method == 'minmax':
        # Min-max normalization to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    elif method == 'z_score':
        # Z-score normalization
        img = (img - 128) / 256
    
    return img

def enhance_contrast(img, clip_limit=2.0, tile_size=8):
    """Enhance image contrast using CLAHE"""
    img_uint8 = (np.clip(img, 0, 255)).astype(np.uint8)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(img_uint8)
    
    return enhanced.astype(np.float32)

def apply_gaussian_blur(img, kernel_size=5, sigma=1.0):
    """Apply Gaussian blur for noise reduction"""
    img_uint8 = (np.clip(img, 0, 255)).astype(np.uint8)
    blurred = cv2.GaussianBlur(img_uint8, (kernel_size, kernel_size), sigma)
    return blurred.astype(np.float32)

def compute_image_statistics(img):
    """Compute image statistics"""
    return {
        'mean': np.mean(img),
        'std': np.std(img),
        'min': np.min(img),
        'max': np.max(img),
        'median': np.median(img),
        'entropy': compute_entropy(img)
    }

def compute_entropy(img):
    """Compute image entropy (measure of information content)"""
    img_uint8 = (np.clip(img, 0, 255)).astype(np.uint8)
    hist, _ = np.histogram(img_uint8, bins=256, range=(0, 256))
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return entropy


def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_backbone(model, freeze=True):
    """Freeze or unfreeze backbone parameters"""
    if hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = not freeze
    return model

def get_model_size(model, input_size=(1, 3, 224, 224)):
    """Estimate model size in MB"""
    num_params = count_parameters(model)
    size_mb = (num_params * 4) / (1024 ** 2)  # Assuming float32
    return size_mb


def compute_class_weights(labels):
    """Compute class weights for imbalanced dataset"""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(unique) * counts)
    return weights / weights.sum()  # Normalize

def compute_iou(pred_mask, true_mask):
    """Compute Intersection over Union (IoU)"""
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    iou = intersection / (union + 1e-8)
    return iou

def compute_dice_coefficient(pred_mask, true_mask):
    """Compute Dice Similarity Coefficient"""
    intersection = np.logical_and(pred_mask, true_mask).sum()
    dice = (2 * intersection) / (pred_mask.sum() + true_mask.sum() + 1e-8)
    return dice

def specificity_score(y_true, y_pred, pos_label=1):
    """Compute specificity (true negative rate)"""
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    specificity = tn / (tn + fp + 1e-8)
    return specificity

def sensitivity_score(y_true, y_pred, pos_label=1):
    """Compute sensitivity (true positive rate / recall)"""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    sensitivity = tp / (tp + fn + 1e-8)
    return sensitivity


def create_image_grid(images, labels, predictions, class_names, grid_size=4):
    """Create grid of images with predictions"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, (img, label, pred) in enumerate(zip(images[:grid_size**2], 
                                                   labels[:grid_size**2], 
                                                   predictions[:grid_size**2])):
        ax = axes[idx]
        
        # Display image
        if img.ndim == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap='gray')
        
        # Title with prediction
        true_class = class_names[label]
        pred_class = class_names[pred]
        
        color = 'green' if label == pred else 'red'
        ax.set_title(f'True: {true_class}\nPred: {pred_class}', color=color)
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def create_stratified_split(X, y, train_size=0.8, random_state=42):
    """Create stratified train-test split"""
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        train_size=train_size,
        random_state=random_state,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def get_class_distribution(labels, class_names):
    """Get class distribution statistics"""
    unique, counts = np.unique(labels, return_counts=True)
    
    dist = {}
    for class_idx, count in zip(unique, counts):
        dist[class_names[class_idx]] = {
            'count': count,
            'percentage': (count / len(labels)) * 100
        }
    
    return dist


def get_device():
    """Get best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("GPU not available, using CPU")
    
    return device

def print_torch_info():
    """Print PyTorch and CUDA information"""
    print("\n" + "=" * 60)
    print("PyTorch Environment Information")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print("=" * 60 + "\n")



def get_augmentation_strength(dataset_size, class_imbalance_ratio):
    """Determine augmentation strength based on dataset size"""
    if dataset_size < 1000:
        return 'strong'  # Strong augmentation for small datasets
    elif dataset_size < 5000:
        return 'moderate'  # Moderate for medium datasets
    else:
        return 'weak'  # Weak for large datasets
    
    # Also consider imbalance
    if class_imbalance_ratio > 3:
        return 'strong'  # Increase augmentation for imbalanced data

def create_augmentation_pipeline(strength='moderate'):
    """Create augmentation pipeline"""
    import torchvision.transforms as transforms
    
    if strength == 'strong':
        pipeline = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.RandomPerspective(distortion_scale=0.3),
        ])
    
    elif strength == 'moderate':
        pipeline = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])
    
    else:  # weak
        pipeline = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.1),
        ])
    
    return pipeline


def save_checkpoint(model, optimizer, epoch, path):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(model, optimizer, path, device):
    """Load training checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from epoch {epoch}: {path}")
    return model, optimizer, epoch


def print_training_status(epoch, train_loss, val_loss, train_acc, val_acc, lr):
    """Print formatted training status"""
    print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
          f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {lr:.2e}")

def save_metadata(metadata, save_path):
    """Save experiment metadata"""
    import json
    with open(save_path, 'w') as f:
        json.dump(metadata, f, indent=4, default=str)
    print(f"Metadata saved: {save_path}")

if __name__ == '__main__':
    print_torch_info()
    print(f"Device: {get_device()}")
