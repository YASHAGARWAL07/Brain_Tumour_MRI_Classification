import subprocess
import sys
import time
from pathlib import Path

# Configuration
DATASET_PATH = Path(__file__).parent
SCRIPTS = [
    ('01_data_exploration.py', 'Data Exploration & Analysis'),
    ('02_train_models.py', 'Model Training'),
    ('03_evaluate_models.py', 'Model Evaluation & Interpretation'),
    ('04_clinical_report.py', 'Clinical Insights & Recommendations'),
]

def print_header(title):
    """Print formatted header"""
    width = 80
    print("\n" + "=" * width)
    print(f" {title.center(width-2)} ")
    print("=" * width + "\n")

def print_section(title):
    """Print section header"""
    print("\n" + "-" * 80)
    print(f" {title}")
    print("-" * 80 + "\n")

def run_script(script_name, description):
    """Run a single script"""
    script_path = DATASET_PATH / script_name
    
    if not script_path.exists():
        print(f"✗ Error: {script_path} not found!")
        return False
    
    print_section(f"Running: {description}")
    print(f"Script: {script_path}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=False, timeout=None)
        
        if result.returncode == 0:
            print(f"\n✓ {description} completed successfully!")
            return True
        else:
            print(f"\n✗ {description} failed with return code {result.returncode}")
            return False
    
    except subprocess.TimeoutExpired:
        print(f"\n✗ {description} timed out!")
        return False
    except Exception as e:
        print(f"\n✗ Error running {description}: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    print_section("Checking Dependencies")
    
    required_packages = [
        'torch',
        'torchvision',
        'numpy',
        'pandas',
        'scipy',
        'sklearn',
        'cv2',  # opencv-python
        'matplotlib',
        'seaborn',
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n✗ Missing packages: {', '.join(missing_packages)}")
        print("\nInstall them using:")
        print("  pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed!")
    return True

def check_dataset():
    """Verify dataset structure"""
    print_section("Verifying Dataset Structure")
    
    CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    all_exist = True
    for class_name in CLASS_NAMES:
        class_path = DATASET_PATH / class_name
        if class_path.exists() and class_path.is_dir():
            image_count = len(list(class_path.glob('*.jpg')))
            print(f"✓ {class_name}: {image_count} images")
        else:
            print(f"✗ {class_name}: MISSING or not a directory")
            all_exist = False
    
    if not all_exist:
        print("\n✗ Dataset structure incomplete!")
        print("Please verify that all tumor class folders exist in the dataset directory.")
        return False
    
    print("\n✓ Dataset structure verified!")
    return True

def print_summary(results):
    """Print execution summary"""
    print_header("EXECUTION SUMMARY")
    
    print("Status of Each Script:\n")
    for (script, description), success in zip(SCRIPTS, results):
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {status:10} | {description}")
    
    passed = sum(results)
    total = len(results)
    success_rate = (passed / total) * 100
    
    print(f"\nOverall: {passed}/{total} scripts passed ({success_rate:.0f}%)\n")
    
    if all(results):
        print("✓ Pipeline completed successfully!")
        print("\nGenerated Files:")
        print("  • data_analysis.png - Dataset visualization")
        print("  • sample_images.png - Sample tumor images")
        print("  • model_*.pth - Trained model weights")
        print("  • training_history.png - Training curves")
        print("  • eval_*.png - Evaluation visualizations")
        print("  • *_report.txt - Detailed reports")
        print("\nNext Steps:")
        print("  1. Review the generated PNG visualizations")
        print("  2. Read the report files for detailed analysis")
        print("  3. Check clinical_insights_and_recommendations.txt")
        print("  4. Integrate models into your workflow")
        return True
    else:
        print("✗ Pipeline encountered errors. Please check the output above.")
        return False

def print_welcome():
    """Print welcome message"""
    print_header("BRAIN TUMOR CLASSIFICATION SYSTEM")
    print("AI-Driven Medical Imaging Analysis Pipeline\n")
    print("This script will execute the complete analysis pipeline:")
    print("  1. Explore and analyze the dataset")
    print("  2. Train multiple deep learning models")
    print("  3. Evaluate models with comprehensive metrics")
    print("  4. Generate clinical insights and recommendations\n")
    print("Estimated Time: 1-5 hours (depending on GPU)")
    print("GPU Required: Yes (CPU fallback available but slow)\n")

def main():
    """Main execution function"""
    
    # Welcome
    print_welcome()
    
    # Pre-flight checks
    print_header("PRE-FLIGHT CHECKS")
    
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n✗ Dependency check failed!")
        sys.exit(1)
    
    print()
    dataset_ok = check_dataset()
    if not dataset_ok:
        print("\n✗ Dataset check failed!")
        sys.exit(1)
    
    # Ask for confirmation
    print_header("READY TO PROCEED")
    print("Everything looks good! Ready to start the pipeline?")
    print("\nThis will:")
    print("  • Analyze your dataset (5-10 min)")
    print("  • Train 3 neural network models (1-4 hours)")
    print("  • Evaluate and interpret results (20-30 min)")
    print("  • Generate clinical report (5 min)")
    print("  • Total: approximately 2-5 hours")
    
    response = input("\nProceed? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\nPipeline cancelled.")
        sys.exit(0)
    
    # Execute pipeline
    print_header("EXECUTING PIPELINE")
    
    start_time = time.time()
    results = []
    
    for script_name, description in SCRIPTS:
        success = run_script(script_name, description)
        results.append(success)
        
        if not success:
            print(f"\n⚠ Warning: {description} failed. Continuing anyway...\n")
            response = input("Continue with remaining scripts? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                break
    
    # Final summary
    elapsed_time = time.time() - start_time
    print_summary(results)
    
    # Print timing
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"\nTotal Execution Time: {hours}h {minutes}m {seconds}s")
    print(f"Dataset Path: {DATASET_PATH}")
    print("\n" + "=" * 80 + "\n")
    
    # Determine exit code
    if all(results):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)
