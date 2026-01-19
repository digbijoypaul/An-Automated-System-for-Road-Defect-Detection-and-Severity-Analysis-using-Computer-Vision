"""
Pothole Detection Inference Script
===================================
Use this script to run predictions on new images using the trained model.
"""

import os
import sys
import argparse
from pathlib import Path

# Install required packages
def install_requirements():
    import subprocess
    packages = ['ultralytics', 'opencv-python', 'numpy']
    for pkg in packages:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

install_requirements()

import cv2
import numpy as np
from ultralytics import YOLO


class PotholeDetector:
    """Pothole detection and severity analysis class."""
    
    def __init__(self, model_path):
        """
        Initialize the detector with a trained model.
        
        Args:
            model_path: Path to the trained YOLOv8 model weights (.pt file)
        """
        self.model = YOLO(model_path)
        self.severity_colors = {
            'None': (128, 128, 128),    # Gray
            'Low': (0, 255, 0),          # Green
            'Medium': (0, 255, 255),     # Yellow
            'High': (0, 165, 255),       # Orange
            'Critical': (0, 0, 255)      # Red
        }
    
    def calculate_severity(self, boxes, image_shape):
        """Calculate severity based on detection results."""
        if boxes is None or len(boxes) == 0:
            return "None", 0, []
        
        img_area = image_shape[0] * image_shape[1]
        severity_details = []
        total_area = 0
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            
            area = (x2 - x1) * (y2 - y1)
            area_pct = (area / img_area) * 100
            total_area += area
            
            if area_pct < 5:
                sev = "Low"
            elif area_pct < 15:
                sev = "Medium"
            elif area_pct < 30:
                sev = "High"
            else:
                sev = "Critical"
            
            severity_details.append({
                'id': i + 1,
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': conf,
                'area_pct': area_pct,
                'severity': sev
            })
        
        total_pct = (total_area / img_area) * 100
        score = total_pct + len(severity_details) * 2
        
        if score < 5:
            overall = "None"
        elif score < 15:
            overall = "Low"
        elif score < 30:
            overall = "Medium"
        elif score < 50:
            overall = "High"
        else:
            overall = "Critical"
        
        return overall, total_pct, severity_details
    
    def predict(self, image_path, conf_threshold=0.25, save_result=True, output_dir=None):
        """
        Run prediction on an image.
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections
            save_result: Whether to save annotated image
            output_dir: Directory to save results
            
        Returns:
            Dictionary with detection and severity results
        """
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Run detection
        results = self.model.predict(
            source=str(image_path),
            conf=conf_threshold,
            save=False,
            verbose=False
        )
        
        # Get boxes from first result
        boxes = results[0].boxes if len(results) > 0 else None
        
        # Calculate severity
        overall_sev, coverage, details = self.calculate_severity(boxes, image.shape[:2])
        
        # Annotate image
        annotated = self._annotate_image(image, overall_sev, coverage, details)
        
        # Save if requested
        if save_result:
            if output_dir is None:
                output_dir = image_path.parent / "predictions"
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"pred_{image_path.name}"
            cv2.imwrite(str(output_path), annotated)
            print(f"Saved prediction to: {output_path}")
        
        return {
            'image_path': str(image_path),
            'overall_severity': overall_sev,
            'coverage_percent': coverage,
            'num_detections': len(details),
            'detections': details,
            'annotated_image': annotated
        }
    
    def _annotate_image(self, image, overall_sev, coverage, details):
        """Add annotations to image."""
        annotated = image.copy()
        
        for det in details:
            x1, y1, x2, y2 = det['bbox']
            color = self.severity_colors[det['severity']]
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            label = f"#{det['id']} {det['severity']} ({det['confidence']:.2f})"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1-h-10), (x1+w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Overall severity box
        color = self.severity_colors[overall_sev]
        cv2.rectangle(annotated, (10, 10), (350, 80), (0, 0, 0), -1)
        cv2.putText(annotated, f"Road Condition: {overall_sev}",
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(annotated, f"Defects: {len(details)} | Coverage: {coverage:.1f}%",
                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    

    def predict_video(self, video_path, output_path=None, conf_threshold=0.25, 
                     frame_skip=1, device='cpu', output_format='mp4'):
        """
        Run prediction on a video file with optimization options.
        
        Args:
            video_path: Path to input video
            output_path: Path for output video
            conf_threshold: Confidence threshold
            frame_skip: Process every nth frame (e.g., 2 = process every 2nd frame)
            device: 'cpu' or 'gpu' or 'cuda' for acceleration
            output_format: 'mp4' or 'avi' for output codec
        
        Returns:
            Dictionary with video processing statistics
        """
        import threading
        import queue
        
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Switch device if GPU available
        if device in ['gpu', 'cuda']:
            try:
                self.model.to('cuda')
            except:
                print("GPU not available, falling back to CPU")
                device = 'cpu'
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if output_path is None:
            output_path = video_path.parent / f"pred_{video_path.name}"
        
        # Select codec based on format
        codec_map = {'mp4': 'mp4v', 'avi': 'XVID'}
        codec = codec_map.get(output_format, 'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        processed_count = 0
        skipped_count = 0
        detections_total = 0
        
        print(f"Processing video: {video_path.name}")
        print(f"Total frames: {total_frames}, Processing every {frame_skip} frame(s)")
        print(f"Device: {device}, Output: {output_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for faster processing
            if (frame_count - 1) % frame_skip != 0:
                out.write(frame)
                skipped_count += 1
                continue
            
            # Run inference with reduced image size for speed
            results = self.model.predict(
                source=frame,
                conf=conf_threshold,
                save=False,
                verbose=False,
                device=0 if device in ['gpu', 'cuda'] else 'cpu',
                imgsz=416  # Smaller image size for faster inference
            )
            
            boxes = results[0].boxes if len(results) > 0 else None
            overall_sev, coverage, details = self.calculate_severity(boxes, frame.shape[:2])
            
            # Annotate frame
            annotated = self._annotate_image(frame, overall_sev, coverage, details)
            out.write(annotated)
            
            detections_total += len(details)
            processed_count += 1
            
            # Progress update
            progress = (frame_count / total_frames) * 100
            if processed_count % max(1, (total_frames // frame_skip) // 10) == 0:
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        cap.release()
        out.release()
        
        print(f"Video processing complete!")
        print(f"  Processed: {processed_count} frames")
        print(f"  Skipped: {skipped_count} frames")
        print(f"  Total detections: {detections_total}")
        print(f"  Output: {output_path}")
        
        return {
            'output_path': str(output_path),
            'total_frames': total_frames,
            'processed_frames': processed_count,
            'detections': detections_total,
            'fps': fps,
            'size': f"{width}x{height}"
        }


def main():
    parser = argparse.ArgumentParser(description='Pothole Detection Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--source', type=str, required=True, help='Path to image or video')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = PotholeDetector(args.model)
    
    source = Path(args.source)
    
    if source.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        detector.predict_video(source, args.output, args.conf)
    else:
        result = detector.predict(source, args.conf, output_dir=args.output)
        print(f"\nResults for: {source.name}")
        print(f"  Overall Severity: {result['overall_severity']}")
        print(f"  Number of Potholes: {result['num_detections']}")
        print(f"  Road Coverage: {result['coverage_percent']:.2f}%")


if __name__ == "__main__":
    main()
