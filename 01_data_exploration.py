import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASET_PATH = Path(__file__).parent
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def count_images_per_class():
    """Count images in each class"""
    print("=" * 60)
    print("DATASET CLASS DISTRIBUTION")
    print("=" * 60)
    
    class_counts = {}
    for class_name in CLASS_NAMES:
        class_path = DATASET_PATH / class_name
        count = len(list(class_path.glob('*.jpg')))
        class_counts[class_name] = count
        print(f"{class_name:15} : {count:5} images")
    
    total = sum(class_counts.values())
    print(f"{'Total':15} : {total:5} images")
    print()
    
    return class_counts

def analyze_image_properties():
    """Analyze image dimensions and properties"""
    print("=" * 60)
    print("IMAGE PROPERTIES ANALYSIS")
    print("=" * 60)
    
    properties = defaultdict(list)
    
    for class_name in CLASS_NAMES:
        class_path = DATASET_PATH / class_name
        images = list(class_path.glob('*.jpg'))[:50]  # Sample 50 per class
        
        for img_path in images:
            try:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    h, w = img.shape
                    properties[class_name].append({
                        'height': h,
                        'width': w,
                        'mean': img.mean(),
                        'std': img.std(),
                        'min': img.min(),
                        'max': img.max()
                    })
            except Exception as e:
                print(f"Error reading {img_path}: {e}")
    
    # Report statistics
    for class_name in CLASS_NAMES:
        props = properties[class_name]
        if props:
            props_df = pd.DataFrame(props)
            print(f"\n{class_name.upper()}:")
            print(f"  Dimensions: {props_df['height'].mean():.0f}x{props_df['width'].mean():.0f}")
            print(f"  Mean intensity: {props_df['mean'].mean():.1f} ± {props_df['mean'].std():.1f}")
            print(f"  Std deviation: {props_df['std'].mean():.1f} ± {props_df['std'].std():.1f}")
            print(f"  Min value: {props_df['min'].mean():.1f}")
            print(f"  Max value: {props_df['max'].mean():.1f}")
    
    print()
    return properties

def detect_quality_issues():
    """Detect potential quality issues in images"""
    print("=" * 60)
    print("IMAGE QUALITY ASSESSMENT")
    print("=" * 60)
    
    quality_issues = defaultdict(int)
    
    for class_name in CLASS_NAMES:
        class_path = DATASET_PATH / class_name
        images = list(class_path.glob('*.jpg'))[:100]  # Sample
        
        low_contrast_count = 0
        low_detail_count = 0
        
        for img_path in images:
            try:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Check contrast (std of pixel values)
                    if img.std() < 15:
                        low_contrast_count += 1
                    
                    # Check detail using Laplacian
                    laplacian = cv2.Laplacian(img, cv2.CV_64F)
                    if laplacian.var() < 100:
                        low_detail_count += 1
            except:
                pass
        
        quality_issues[class_name] = {
            'low_contrast': low_contrast_count,
            'low_detail': low_detail_count,
            'total_checked': len(images)
        }
        
        print(f"{class_name:15}: Low contrast: {low_contrast_count:3}/{len(images)}, " \
              f"Low detail: {low_detail_count:3}/{len(images)}")
    
    print()
    return quality_issues

def analyze_class_imbalance(class_counts):
    """Analyze class imbalance"""
    print("=" * 60)
    print("CLASS IMBALANCE ANALYSIS")
    print("=" * 60)
    
    total = sum(class_counts.values())
    
    imbalance_ratio = {}
    for class_name, count in class_counts.items():
        percentage = (count / total) * 100
        imbalance_ratio[class_name] = percentage
        print(f"{class_name:15}: {percentage:6.2f}% ({count} images)")
    
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_factor = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\nImbalance Factor (max/min): {imbalance_factor:.2f}x")
    print("This means we need class balancing techniques!")
    print()
    
    return imbalance_ratio

def create_visualizations(class_counts, imbalance_ratio):
    """Create visualization plots"""
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Class distribution
    ax = axes[0, 0]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    ax.bar(class_counts.keys(), class_counts.values(), color=colors, edgecolor='black', linewidth=1.5)
    ax.set_title('Class Distribution', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Percentage distribution
    ax = axes[0, 1]
    ax.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%',
           colors=colors, startangle=90, textprops={'fontsize': 10})
    ax.set_title('Class Distribution (%)', fontsize=12, fontweight='bold')
    
    # Plot 3: Imbalance visualization
    ax = axes[1, 0]
    imb_df = pd.DataFrame(list(imbalance_ratio.items()), columns=['Class', 'Percentage'])
    ax.barh(imb_df['Class'], imb_df['Percentage'], color=colors, edgecolor='black', linewidth=1.5)
    ax.set_title('Class Percentage Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Percentage (%)')
    ax.grid(axis='x', alpha=0.3)
    
    # Plot 4: Sample images
    ax = axes[1, 1]
    ax.axis('off')
    
    # Display sample images
    sample_text = "Dataset Summary:\n"
    total = sum(class_counts.values())
    sample_text += f"Total Images: {total}\n\n"
    for class_name, count in class_counts.items():
        sample_text += f"• {class_name.upper()}: {count} images\n"
    
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    sample_text += f"\nImbalance Ratio: {max_count/min_count:.2f}x\n"
    sample_text += "\nRecommendations:\n"
    sample_text += "✓ Use weighted loss\n"
    sample_text += "✓ Under/over-sampling\n"
    sample_text += "✓ Data augmentation\n"
    sample_text += "✓ Stratified k-fold\n"
    
    ax.text(0.1, 0.5, sample_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(DATASET_PATH / 'data_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: data_analysis.png")
    plt.close()

def sample_and_visualize():
    """Display sample images from each class"""
    print("=" * 60)
    print("SAMPLE IMAGE VISUALIZATION")
    print("=" * 60)
    
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    axes = axes.flatten()
    
    idx = 0
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_path = DATASET_PATH / class_name
        images = sorted(list(class_path.glob('*.jpg')))[:4]
        
        for img_idx, img_path in enumerate(images):
            ax = axes[class_idx * 4 + img_idx]
            try:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                ax.imshow(img, cmap='gray')
                ax.set_title(f'{class_name}\n{img_path.stem}', fontsize=9)
                ax.axis('off')
            except:
                pass
    
    plt.tight_layout()
    plt.savefig(DATASET_PATH / 'sample_images.png', dpi=200, bbox_inches='tight')
    print("✓ Saved: sample_images.png")
    plt.close()

def generate_report(class_counts, imbalance_ratio):
    """Generate comprehensive report"""
    print("=" * 60)
    print("GENERATING COMPREHENSIVE REPORT")
    print("=" * 60)
    
    report = "=" * 70 + "\n"
    report += "BRAIN TUMOR MRI DATASET - COMPREHENSIVE ANALYSIS REPORT\n"
    report += "=" * 70 + "\n\n"
    
    report += "1. DATASET OVERVIEW\n"
    report += "-" * 70 + "\n"
    report += f"Total Images: {sum(class_counts.values())}\n"
    report += f"Number of Classes: {len(class_counts)}\n"
    report += f"Classes: {', '.join(CLASS_NAMES)}\n\n"
    
    report += "2. CLASS DISTRIBUTION\n"
    report += "-" * 70 + "\n"
    for class_name, count in class_counts.items():
        pct = imbalance_ratio[class_name]
        bar = "█" * int(pct / 2)
        report += f"{class_name:12} : {count:5} images ({pct:5.2f}%) {bar}\n"
    
    report += "\n3. CLASS IMBALANCE ANALYSIS\n"
    report += "-" * 70 + "\n"
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    report += f"Maximum class size: {max_count}\n"
    report += f"Minimum class size: {min_count}\n"
    report += f"Imbalance ratio: {max_count/min_count:.2f}x\n"
    report += f"Severity: {'HIGH' if (max_count/min_count) > 2 else 'MODERATE' if (max_count/min_count) > 1.5 else 'LOW'}\n\n"
    
    report += "4. RECOMMENDED TECHNIQUES\n"
    report += "-" * 70 + "\n"
    report += "Data Handling:\n"
    report += "  • Weighted CrossEntropyLoss with inverse class frequencies\n"
    report += "  • Focal Loss for hard example mining\n"
    report += "  • Stratified K-Fold cross-validation\n"
    report += "  • Data augmentation (rotation, flip, elastic distortion)\n\n"
    
    report += "Model-level:\n"
    report += "  • Class-balanced batch sampling\n"
    report += "  • Ensemble methods combining multiple architectures\n"
    report += "  • Threshold optimization per class\n\n"
    
    report += "5. CLINICAL CONSIDERATIONS\n"
    report += "-" * 70 + "\n"
    report += "  • Class imbalance reflects real-world prevalence\n"
    report += "  • Model should be evaluated on per-class metrics\n"
    report += "  • Sensitivity vs Specificity trade-offs important\n"
    report += "  • Confidence calibration crucial for clinical deployment\n\n"
    
    report += "6. NEXT STEPS\n"
    report += "-" * 70 + "\n"
    report += "  1. Implement data preprocessing and augmentation\n"
    report += "  2. Train multiple architectures (ResNet, EfficientNet, ViT)\n"
    report += "  3. Implement uncertainty estimation methods\n"
    report += "  4. Add interpretability techniques (GradCAM, attention maps)\n"
    report += "  5. Perform comprehensive evaluation and validation\n"
    report += "  6. Generate clinical insights and recommendations\n\n"
    
    report += "=" * 70 + "\n"
    
    print(report)
    
    # Save report
    with open(DATASET_PATH / 'dataset_analysis_report.txt', 'w', encoding='UTF-8') as f:
        f.write(report)
    
    print("✓ Report saved: dataset_analysis_report.txt")

def main():
   
    print("\n" + "=" * 60)
    print("BRAIN TUMOR DATASET - COMPREHENSIVE ANALYSIS")
    print("=" * 60 + "\n")
    
    # Run analysis
    class_counts = count_images_per_class()
    properties = analyze_image_properties()
    quality_issues = detect_quality_issues()
    imbalance_ratio = analyze_class_imbalance(class_counts)
    
    # Create visualizations
    create_visualizations(class_counts, imbalance_ratio)
    sample_and_visualize()
    
    # Generate report
    generate_report(class_counts, imbalance_ratio)
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("Check the generated PNG files and text report for details")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    main()
