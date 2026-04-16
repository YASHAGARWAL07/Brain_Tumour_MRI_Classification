import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, auc, precision_recall_curve, f1_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASET_PATH = Path(__file__).parent
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GradCAM:
    """GradCAM for model interpretability"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, class_idx):
        """Generate Class Activation Map"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)
        
        # Generate CAM
        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam


class UncertaintyEstimator:
    """Estimate prediction uncertainty"""
    
    @staticmethod
    def entropy(probabilities):
        """Calculate entropy of predictions"""
        probs = probabilities / (probabilities.sum(axis=1, keepdims=True) + 1e-10)
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        return entropy
    
    @staticmethod
    def margin(probabilities):
        """Calculate margin between top two predictions"""
        sorted_probs = np.sort(probabilities, axis=1)[:, -2:]
        margin = sorted_probs[:, 1] - sorted_probs[:, 0]
        return margin
    
    @staticmethod
    def confidence_interval(probabilities, confidence=0.95):
        """Calculate confidence intervals"""
        max_probs = np.max(probabilities, axis=1)
        
        # Normalize to 0-1 if not already
        max_probs = np.clip(max_probs, 0, 1)
        
        # Simple interval based on confidence
        interval_width = (1 - confidence) / 2
        return max_probs, max_probs - interval_width, max_probs + interval_width



def ensemble_predict(models, dataloader, device):
    """Ensemble predictions from multiple models"""
    all_ensemble_preds = []
    all_confidences = []
    all_uncertainties = []
    all_labels = []
    
    for images, labels, names in dataloader:
        images = images.to(device)
        batch_preds = []
        
        for model in models:
            model.eval()
            with torch.no_grad():
                outputs = model(images)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                batch_preds.append(probs)
        
        # Ensemble prediction (mean of probabilities)
        ensemble_pred = np.mean(batch_preds, axis=0)
        ensemble_output = np.argmax(ensemble_pred, axis=1)
        
        # Uncertainty estimation
        uncertainties = UncertaintyEstimator.entropy(ensemble_pred)
        confidences = np.max(ensemble_pred, axis=1)
        
        all_ensemble_preds.append(ensemble_output)
        all_confidences.extend(confidences)
        all_uncertainties.extend(uncertainties)
        all_labels.extend(labels.numpy())
    
    return (
        np.concatenate(all_ensemble_preds),
        np.array(all_confidences),
        np.array(all_uncertainties),
        np.array(all_labels)
    )


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt.gcf()

def plot_roc_auc(y_true, y_scores):
    """Plot ROC curves"""
    y_true_bin = label_binarize(y_true, classes=list(range(len(CLASS_NAMES))))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for i, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2, 
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curves - Multi-class', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_precision_recall(y_true, y_scores):
    """Plot Precision-Recall curves"""
    y_true_bin = label_binarize(y_true, classes=list(range(len(CLASS_NAMES))))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for i, class_name in enumerate(CLASS_NAMES):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        ap = np.mean(precision)
        ax.plot(recall, precision, color=colors[i], lw=2,
                label=f'{class_name} (AP = {ap:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall Curves - Multi-class', fontsize=14, fontweight='bold')
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_uncertainty_calibration(confidences, uncertainties, correct):
    """Plot uncertainty calibration"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confidence vs Accuracy
    ax = axes[0]
    ax.scatter(confidences[correct], correct[correct], alpha=0.6, label='Correct', color='green', s=30)
    ax.scatter(confidences[~correct], correct[~correct], alpha=0.6, label='Incorrect', color='red', s=30)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.set_xlabel('Confidence', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Confidence Calibration', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Uncertainty distribution
    ax = axes[1]
    ax.hist(uncertainties[correct], bins=30, alpha=0.6, label='Correct', color='green')
    ax.hist(uncertainties[~correct], bins=30, alpha=0.6, label='Incorrect', color='red')
    ax.set_xlabel('Entropy (Uncertainty)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Uncertainty Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def plot_per_class_metrics(y_true, y_pred):
    """Plot per-class metrics"""
    report = classification_report(y_true, y_pred, 
                                   target_names=CLASS_NAMES, 
                                   output_dict=True)
    
    metrics_data = []
    for class_name in CLASS_NAMES:
        metrics_data.append({
            'Class': class_name,
            'Precision': report[class_name]['precision'],
            'Recall': report[class_name]['recall'],
            'F1-Score': report[class_name]['f1-score'],
            'Support': report[class_name]['support']
        })
    
    df = pd.DataFrame(metrics_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    
    ax.bar(x - width, df['Precision'], width, label='Precision', alpha=0.8)
    ax.bar(x, df['Recall'], width, label='Recall', alpha=0.8)
    ax.bar(x + width, df['F1-Score'], width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Per-Class Performance Metrics', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    return fig, df



def generate_clinical_report(y_true, y_pred, uncertainties, confidences):
    """Generate comprehensive clinical report"""
    
    report = "=" * 80 + "\n"
    report += "BRAIN TUMOR MRI CLASSIFICATION - COMPREHENSIVE EVALUATION REPORT\n"
    report += "=" * 80 + "\n\n"
    
    # Overall metrics
    report += "1. OVERALL PERFORMANCE METRICS\n"
    report += "-" * 80 + "\n"
    
    accuracy = np.mean(y_true == y_pred)
    report += f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n"
    
    # Per-class metrics
    report += "\n2. PER-CLASS PERFORMANCE\n"
    report += "-" * 80 + "\n"
    
    class_report = classification_report(y_true, y_pred, 
                                        target_names=CLASS_NAMES, 
                                        output_dict=True)
    
    for class_name in CLASS_NAMES:
        metrics = class_report[class_name]
        report += f"\n{class_name.upper()}:\n"
        report += f"  Precision:  {metrics['precision']:.4f}\n"
        report += f"  Recall:     {metrics['recall']:.4f}\n"
        report += f"  F1-Score:   {metrics['f1-score']:.4f}\n"
        report += f"  Support:    {int(metrics['support'])} samples\n"
    
    # Uncertainty analysis
    report += "\n3. UNCERTAINTY ESTIMATION\n"
    report += "-" * 80 + "\n"
    
    correct = (y_true == y_pred)
    report += f"Mean confidence (correct): {np.mean(confidences[correct]):.4f}\n"
    report += f"Mean confidence (incorrect): {np.mean(confidences[~correct]):.4f}\n"
    report += f"Mean uncertainty (correct): {np.mean(uncertainties[correct]):.4f}\n"
    report += f"Mean uncertainty (incorrect): {np.mean(uncertainties[~correct]):.4f}\n"
    
    # Confusion analysis
    report += "\n4. COMMON MISCLASSIFICATIONS\n"
    report += "-" * 80 + "\n"
    
    cm = confusion_matrix(y_true, y_pred)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i != j and cm[i, j] > 0:
                report += f"{CLASS_NAMES[i]} misclassified as {CLASS_NAMES[j]}: {cm[i, j]} times\n"
    
    # Clinical recommendations
    report += "\n5. CLINICAL RECOMMENDATIONS\n"
    report += "-" * 80 + "\n"    
    
    if accuracy > 0.95:
        report += "✓ Model shows excellent performance (>95% accuracy)\n"
    elif accuracy > 0.90:
        report += "✓ Model shows good performance (>90% accuracy)\n"
    else:
        report += "⚠ Model needs further improvement\n"    
    
    # Find most confused classes
    sorted_misclass = []
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i != j:
                sorted_misclass.append((cm[i, j], CLASS_NAMES[i], CLASS_NAMES[j]))
    sorted_misclass.sort(reverse=True)
    
    if sorted_misclass[0][0] > 0:
        report += f"⚠ Most common confusion: {sorted_misclass[0][1]} → {sorted_misclass[0][2]}\n"
    report += "✓ Use uncertainty estimation for low-confidence predictions\n"
    report += "✓ Combine with radiologist review for critical cases\n"
    report += "✓ Regular model retraining with new data\n"
    
    report += "\n" + "=" * 80 + "\n"
    
    return report



def main():
    print("\n" + "=" * 80)
    print("BRAIN TUMOR CLASSIFICATION - COMPREHENSIVE EVALUATION")
    print("=" * 80 + "\n")