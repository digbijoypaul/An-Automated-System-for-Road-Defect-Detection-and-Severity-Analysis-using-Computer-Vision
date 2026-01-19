"""
Automated Road Defect Detection and Severity Analysis Using YOLOv8
==================================================================
This script trains a YOLOv8 model for pothole detection and provides
severity analysis based on detected defect characteristics.

Author: AI Assistant
Date: December 2024
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Check and install required packages
def install_requirements():
    """Install required packages if not present."""
    required_packages = [
        'ultralytics',
        'opencv-python',
        'numpy',
        'matplotlib',
        'pandas',
        'seaborn',
        'Pillow'
    ]
    
    import subprocess
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').split('==')[0])
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])

# Install requirements
print("Checking and installing required packages...")
install_requirements()

# Now import all required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pandas as pd

# Configuration
class Config:
    """Configuration class for training parameters."""
    # Paths
    BASE_DIR = Path(r"c:/Users/knida/Downloads/POTHOLE_DETECTION")
    DATA_YAML = BASE_DIR / "data.yaml"
    OUTPUT_DIR = BASE_DIR / "runs"
    
    # Training parameters - REDUCED FOR CPU TRAINING
    MODEL_SIZE = "yolov8n.pt"  # Using nano model (smallest/fastest)
    EPOCHS = 10                # Reduced from 50
    BATCH_SIZE = 4             # Reduced from 8
    IMAGE_SIZE = 320           # Reduced from 640
    PATIENCE = 5               # Early stopping patience
    
    # Device - automatically detect GPU or use CPU
    DEVICE = "cpu"  # Using CPU since no CUDA GPU available
    
    # Severity thresholds (based on pothole area percentage)
    SEVERITY_THRESHOLDS = {
        'low': 0.05,      # < 5% of image area
        'medium': 0.15,   # 5-15% of image area
        'high': 0.30,     # 15-30% of image area
        'critical': 1.0   # > 30% of image area
    }


def verify_dataset():
    """Verify dataset structure and files."""
    print("\n" + "="*60)
    print("DATASET VERIFICATION")
    print("="*60)
    
    base_dir = Config.BASE_DIR
    splits = ['train', 'valid', 'test']
    
    dataset_info = {}
    for split in splits:
        images_dir = base_dir / split / "images"
        labels_dir = base_dir / split / "labels"
        
        if images_dir.exists():
            images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            labels = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []
            
            dataset_info[split] = {
                'images': len(images),
                'labels': len(labels)
            }
            print(f"\n{split.upper()} set:")
            print(f"  - Images: {len(images)}")
            print(f"  - Labels: {len(labels)}")
        else:
            print(f"\n{split.upper()} set: NOT FOUND")
            dataset_info[split] = {'images': 0, 'labels': 0}
    
    total_images = sum(d['images'] for d in dataset_info.values())
    print(f"\nTotal images: {total_images}")
    
    return dataset_info


def train_model():
    """Train YOLOv8 model for pothole detection."""
    print("\n" + "="*60)
    print("TRAINING YOLOv8 MODEL FOR POTHOLE DETECTION")
    print("="*60)
    
    # Initialize model
    print(f"\nLoading pretrained model: {Config.MODEL_SIZE}")
    model = YOLO(Config.MODEL_SIZE)
    
    # Training timestamp for unique run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"pothole_detector_{timestamp}"
    
    print(f"\nStarting training with the following configuration:")
    print(f"  - Model: {Config.MODEL_SIZE}")
    print(f"  - Epochs: {Config.EPOCHS}")
    print(f"  - Batch Size: {Config.BATCH_SIZE}")
    print(f"  - Image Size: {Config.IMAGE_SIZE}")
    print(f"  - Device: {Config.DEVICE}")
    print(f"  - Output: {Config.OUTPUT_DIR / run_name}")
    
    # Train the model
    results = model.train(
        data=str(Config.DATA_YAML),
        epochs=Config.EPOCHS,
        batch=Config.BATCH_SIZE,
        imgsz=Config.IMAGE_SIZE,
        device=Config.DEVICE,
        project=str(Config.OUTPUT_DIR),
        name=run_name,
        patience=Config.PATIENCE,
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        plots=True,
        verbose=True,
        pretrained=True,
        optimizer='auto',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    
    return model, results, run_name


def evaluate_model(model, run_name):
    """Evaluate the trained model on validation set."""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Validate on the validation set
    metrics = model.val(
        data=str(Config.DATA_YAML),
        split='val',
        plots=True,
        save_json=True
    )
    
    print("\nValidation Metrics:")
    print(f"  - mAP50: {metrics.box.map50:.4f}")
    print(f"  - mAP50-95: {metrics.box.map:.4f}")
    print(f"  - Precision: {metrics.box.mp:.4f}")
    print(f"  - Recall: {metrics.box.mr:.4f}")
    
    return metrics


def calculate_severity(detection_results, image_shape):
    """
    Calculate severity of detected potholes based on their characteristics.
    
    Severity is determined by:
    1. Area coverage (percentage of image)
    2. Number of detections
    3. Confidence scores
    
    Returns severity classification: Low, Medium, High, Critical
    """
    if detection_results is None or len(detection_results) == 0:
        return "None", 0, []
    
    img_area = image_shape[0] * image_shape[1]
    severity_details = []
    total_pothole_area = 0
    
    for result in detection_results:
        boxes = result.boxes
        if boxes is None:
            continue
            
        for i, box in enumerate(boxes):
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            
            # Calculate area
            width = x2 - x1
            height = y2 - y1
            area = width * height
            area_percentage = (area / img_area) * 100
            total_pothole_area += area
            
            # Determine individual pothole severity
            if area_percentage < 5:
                individual_severity = "Low"
            elif area_percentage < 15:
                individual_severity = "Medium"
            elif area_percentage < 30:
                individual_severity = "High"
            else:
                individual_severity = "Critical"
            
            severity_details.append({
                'id': i + 1,
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': float(conf),
                'area_percentage': area_percentage,
                'severity': individual_severity
            })
    
    # Overall severity based on total coverage and count
    total_coverage = (total_pothole_area / img_area) * 100
    num_potholes = len(severity_details)
    
    # Calculate overall severity score
    severity_score = total_coverage + (num_potholes * 2)  # Weight count
    
    if severity_score < 5 or num_potholes == 0:
        overall_severity = "None"
    elif severity_score < 15:
        overall_severity = "Low"
    elif severity_score < 30:
        overall_severity = "Medium"
    elif severity_score < 50:
        overall_severity = "High"
    else:
        overall_severity = "Critical"
    
    return overall_severity, total_coverage, severity_details


def predict_and_analyze(model, image_path, save_path=None):
    """
    Run prediction on an image and perform severity analysis.
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run prediction
    results = model.predict(
        source=image_path,
        conf=0.25,
        iou=0.45,
        save=False,
        verbose=False
    )
    
    # Calculate severity
    overall_severity, coverage, details = calculate_severity(results, image.shape[:2])
    
    # Annotate image
    annotated_image = image.copy()
    
    # Define colors for severity levels
    severity_colors = {
        'Low': (0, 255, 0),       # Green
        'Medium': (0, 255, 255),   # Yellow
        'High': (0, 165, 255),     # Orange
        'Critical': (0, 0, 255)    # Red
    }
    
    for detail in details:
        x1, y1, x2, y2 = detail['bbox']
        severity = detail['severity']
        conf = detail['confidence']
        
        color = severity_colors.get(severity, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = f"Pothole #{detail['id']}: {severity} ({conf:.2f})"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(annotated_image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add overall severity indicator
    overall_color = severity_colors.get(overall_severity, (255, 255, 255))
    cv2.rectangle(annotated_image, (10, 10), (300, 80), (0, 0, 0), -1)
    cv2.putText(annotated_image, f"Overall Severity: {overall_severity}", 
               (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, overall_color, 2)
    cv2.putText(annotated_image, f"Potholes: {len(details)} | Coverage: {coverage:.1f}%",
               (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if save_path:
        cv2.imwrite(str(save_path), annotated_image)
    
    return {
        'image_path': str(image_path),
        'overall_severity': overall_severity,
        'total_coverage': coverage,
        'num_potholes': len(details),
        'details': details,
        'annotated_image': annotated_image
    }


def batch_analysis(model, image_dir, output_dir):
    """
    Perform batch analysis on multiple images.
    """
    print("\n" + "="*60)
    print("BATCH ANALYSIS")
    print("="*60)
    
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    results_summary = []
    
    for i, img_path in enumerate(image_files[:10]):  # Process first 10 images
        print(f"\nProcessing {i+1}/{min(len(image_files), 10)}: {img_path.name}")
        
        save_path = output_dir / f"analyzed_{img_path.name}"
        result = predict_and_analyze(model, img_path, save_path)
        
        if result:
            results_summary.append({
                'image': img_path.name,
                'severity': result['overall_severity'],
                'num_potholes': result['num_potholes'],
                'coverage': result['total_coverage']
            })
            print(f"  Severity: {result['overall_severity']}, "
                  f"Potholes: {result['num_potholes']}, "
                  f"Coverage: {result['total_coverage']:.2f}%")
    
    # Create summary DataFrame
    df = pd.DataFrame(results_summary)
    summary_path = output_dir / "analysis_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    
    # Print severity distribution
    if not df.empty:
        print("\nSeverity Distribution:")
        print(df['severity'].value_counts())
    
    return df


def export_model(model, run_name):
    """Export model to various formats."""
    print("\n" + "="*60)
    print("MODEL EXPORT")
    print("="*60)
    
    export_dir = Config.OUTPUT_DIR / run_name / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Export to ONNX format
    print("\nExporting to ONNX format...")
    model.export(format='onnx', simplify=True)
    
    print(f"\nModel exports saved to: {export_dir}")


def generate_report(dataset_info, metrics, run_name):
    """Generate a comprehensive training report."""
    print("\n" + "="*60)
    print("GENERATING TRAINING REPORT")
    print("="*60)
    
    report_path = Config.OUTPUT_DIR / run_name / "training_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("POTHOLE DETECTION MODEL - TRAINING REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("TRAINING CONFIGURATION\n")
        f.write("-"*40 + "\n")
        f.write(f"Model: {Config.MODEL_SIZE}\n")
        f.write(f"Epochs: {Config.EPOCHS}\n")
        f.write(f"Batch Size: {Config.BATCH_SIZE}\n")
        f.write(f"Image Size: {Config.IMAGE_SIZE}\n")
        f.write(f"Device: {Config.DEVICE}\n\n")
        
        f.write("DATASET INFORMATION\n")
        f.write("-"*40 + "\n")
        for split, info in dataset_info.items():
            f.write(f"{split.capitalize()}: {info['images']} images, {info['labels']} labels\n")
        f.write("\n")
        
        f.write("EVALUATION METRICS\n")
        f.write("-"*40 + "\n")
        f.write(f"mAP@0.5: {metrics.box.map50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {metrics.box.map:.4f}\n")
        f.write(f"Precision: {metrics.box.mp:.4f}\n")
        f.write(f"Recall: {metrics.box.mr:.4f}\n\n")
        
        f.write("SEVERITY CLASSIFICATION THRESHOLDS\n")
        f.write("-"*40 + "\n")
        f.write("Low: < 5% image coverage\n")
        f.write("Medium: 5-15% image coverage\n")
        f.write("High: 15-30% image coverage\n")
        f.write("Critical: > 30% image coverage\n\n")
        
        f.write("="*60 + "\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n")
    
    print(f"Report saved to: {report_path}")
    return report_path


def main():
    """Main function to run the complete pipeline."""
    print("\n" + "="*60)
    print("AUTOMATED ROAD DEFECT DETECTION AND SEVERITY ANALYSIS")
    print("Using YOLOv8 and Computer Vision")
    print("="*60)
    
    # Step 1: Verify dataset
    dataset_info = verify_dataset()
    
    # Step 2: Train model
    model, train_results, run_name = train_model()
    
    # Step 3: Evaluate model
    metrics = evaluate_model(model, run_name)
    
    # Step 4: Generate report
    report_path = generate_report(dataset_info, metrics, run_name)
    
    # Step 5: Run batch analysis on test images
    test_images_dir = Config.BASE_DIR / "test" / "images"
    analysis_output_dir = Config.OUTPUT_DIR / run_name / "severity_analysis"
    
    if test_images_dir.exists():
        analysis_df = batch_analysis(model, test_images_dir, analysis_output_dir)
    
    # Step 6: Export model (optional)
    try:
        export_model(model, run_name)
    except Exception as e:
        print(f"Model export skipped: {e}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nResults saved to: {Config.OUTPUT_DIR / run_name}")
    print(f"Best model weights: {Config.OUTPUT_DIR / run_name / 'weights' / 'best.pt'}")
    print(f"Training report: {report_path}")
    
    return model, run_name


if __name__ == "__main__":
    model, run_name = main()
