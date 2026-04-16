import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASET_PATH = Path(__file__).parent
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

class BrainTumorDataset(Dataset):
    """Custom dataset for brain tumor images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Read image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        # Convert to 3 channels
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        
        return img, label, img_path.stem


def get_augmentation_transforms():
    """Create augmentation pipelines"""
    
    # Training augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Validation/Test - no augmentation
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return train_transform, val_transform


def get_class_weights(labels):
    """Calculate class weights for imbalanced dataset"""
    unique, counts = np.unique(labels, return_counts=True)
    weights = len(labels) / (len(unique) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class ResNetBrainTumor(nn.Module):
    """ResNet50 with custom head"""
    
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        
        # Replace final FC layer
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class EfficientNetBrainTumor(nn.Module):
    """EfficientNet-B3 with custom head"""
    
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = models.efficientnet_b3(pretrained=True)
        
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class AttentionBrainTumor(nn.Module):
    """ResNet with Spatial Attention Mechanism"""
    
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        
        # Spatial attention
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Get features before final FC layer
        features = self.backbone.conv1(x)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)
        
        # Apply attention
        attention_map = self.attention(features)
        features = features * attention_map
        
        # Average pooling and classification
        features_avg = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features_avg = features_avg.view(features_avg.size(0), -1)
        
        return self.classifier(features_avg)



def load_dataset():
    """Load all images and labels"""
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_path = DATASET_PATH / class_name
        images = sorted(list(class_path.glob('*.jpg')))
        
        for img_path in images:
            image_paths.append(img_path)
            labels.append(class_idx)
    
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    print(f"\nTotal images loaded: {len(image_paths)}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    return image_paths, labels


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for batch_idx, (images, labels, _) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating', leave=False)
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    """Full training loop"""
    best_val_acc = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model, history



def main():
    print("\n" + "=" * 70)
    print("BRAIN TUMOR CLASSIFICATION - MODEL TRAINING")
    print("=" * 70 + "\n")
    
    # Load data
    image_paths, labels = load_dataset()
    
    # Train-val split with stratification
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets and dataloaders
    train_transform, val_transform = get_augmentation_transforms()
    
    train_dataset = BrainTumorDataset(X_train, y_train, transform=train_transform)
    val_dataset = BrainTumorDataset(X_val, y_val, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Get class weights
    class_weights = get_class_weights(y_train)
    
    # Train multiple models
    models_config = [
        ('ResNet50', ResNetBrainTumor),
        ('EfficientNet-B3', EfficientNetBrainTumor),
        ('ResNet50+Attention', AttentionBrainTumor)
    ]
    
    all_results = {}
    
    for model_name, ModelClass in models_config:
        print(f"\n{'='*70}")
        print(f"Training {model_name}")
        print(f"{'='*70}\n")
        
        # Initialize model
        model = ModelClass(num_classes=len(CLASS_NAMES)).to(DEVICE)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                         patience=5)
        
        # Train
        model, history = train_model(model, train_loader, val_loader, criterion, 
                                     optimizer, scheduler, EPOCHS, DEVICE)
        
        # Validate
        _, final_val_acc, val_preds, val_labels = validate(model, val_loader, criterion, DEVICE)
        
        # Save model
        model_save_path = DATASET_PATH / f'model_{model_name.replace(" ", "_").replace("+", "_")}.pth'
        torch.save(model.state_dict(), model_save_path)
        print(f"\n✓ Model saved: {model_save_path}")
        
        # Store results
        all_results[model_name] = {
            'model': model,
            'history': history,
            'val_acc': final_val_acc,
            'predictions': val_preds,
            'ground_truth': val_labels
        }
        
        print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for model_name, results in all_results.items():
        history = results['history']
        axes[0].plot(history['train_loss'], label=f'{model_name} Train')
        axes[0].plot(history['val_loss'], label=f'{model_name} Val')
        axes[1].plot(history['train_acc'], label=f'{model_name} Train')
        axes[1].plot(history['val_acc'], label=f'{model_name} Val')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History - Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training History - Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(DATASET_PATH / 'training_history.png', dpi=300, bbox_inches='tight')
    print("\n✓ Training history saved: training_history.png")
    plt.close()
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    
    for model_name, results in all_results.items():
        print(f"{model_name:25}: Final Val Accuracy = {results['val_acc']:.2f}%")

if __name__ == '__main__':
    main()
