"""
Advanced Video Processing Module for Pothole Detection
========================================================
Includes enhancements like detection smoothing, merging, and statistics.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
from collections import deque
import time


class DetectionSmoother:
    """Smooths detections across frames to reduce jitter."""
    
    def __init__(self, window_size: int = 3):
        """
        Args:
            window_size: Number of frames to use for smoothing
        """
        self.window_size = window_size
        self.detection_history = deque(maxlen=window_size)
    
    def smooth_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Apply temporal smoothing to detections.
        Reduces jitter by averaging bounding boxes over time.
        """
        self.detection_history.append(detections)
        
        if len(self.detection_history) < self.window_size:
            return detections
        
        # Average bounding boxes across history
        smoothed = []
        for det_idx in range(len(detections)):
            if det_idx < len(detections):
                avg_bbox = self._average_bbox(det_idx)
                if avg_bbox:
                    smoothed_det = detections[det_idx].copy()
                    smoothed_det['bbox'] = avg_bbox
                    smoothed.append(smoothed_det)
        
        return smoothed if smoothed else detections
    
    def _average_bbox(self, det_idx: int) -> Dict:
        """Average bounding boxes across frames."""
        try:
            x1s, y1s, x2s, y2s = [], [], [], []
            
            for frame_dets in self.detection_history:
                if det_idx < len(frame_dets):
                    bbox = frame_dets[det_idx]['bbox']
                    x1s.append(bbox['x1'])
                    y1s.append(bbox['y1'])
                    x2s.append(bbox['x2'])
                    y2s.append(bbox['y2'])
            
            if x1s:
                return {
                    'x1': int(np.median(x1s)),
                    'y1': int(np.median(y1s)),
                    'x2': int(np.median(x2s)),
                    'y2': int(np.median(y2s))
                }
        except:
            pass
        
        return None


class DetectionMerger:
    """Merges nearby detections to reduce duplicates."""
    
    def __init__(self, merge_threshold: float = 0.3):
        """
        Args:
            merge_threshold: IoU threshold for merging (0.0-1.0)
        """
        self.merge_threshold = merge_threshold
    
    def merge_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Merge overlapping detections.
        """
        if len(detections) <= 1:
            return detections
        
        merged = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            cluster = [det1]
            used.add(i)
            
            # Find overlapping detections
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                
                iou = self._calculate_iou(det1['bbox'], det2['bbox'])
                if iou > self.merge_threshold:
                    cluster.append(det2)
                    used.add(j)
            
            # Merge cluster
            merged_det = self._merge_cluster(cluster)
            merged.append(merged_det)
        
        return merged
    
    def _calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """Calculate Intersection over Union."""
        x1_min, y1_min = box1['x1'], box1['y1']
        x1_max, y1_max = box1['x2'], box1['y2']
        
        x2_min, y2_min = box2['x1'], box2['y1']
        x2_max, y2_max = box2['x2'], box2['y2']
        
        # Calculate intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _merge_cluster(self, cluster: List[Dict]) -> Dict:
        """Merge a cluster of detections."""
        x1s = [d['bbox']['x1'] for d in cluster]
        y1s = [d['bbox']['y1'] for d in cluster]
        x2s = [d['bbox']['x2'] for d in cluster]
        y2s = [d['bbox']['y2'] for d in cluster]
        
        merged = cluster[0].copy()
        merged['bbox'] = {
            'x1': int(min(x1s)),
            'y1': int(min(y1s)),
            'x2': int(max(x2s)),
            'y2': int(max(y2s))
        }
        
        # Average confidence
        avg_conf = np.mean([d['confidence'] for d in cluster])
        merged['confidence'] = float(avg_conf)
        
        return merged


class VideoStatistics:
    """Tracks statistics during video processing."""
    
    def __init__(self):
        self.frame_times = deque(maxlen=30)  # Last 30 frame times
        self.confidences = []
        self.detection_counts = []
        self.start_time = time.time()
    
    def add_frame_time(self, frame_time: float):
        """Record frame processing time."""
        self.frame_times.append(frame_time)
    
    def add_detections(self, detections: List[Dict]):
        """Record detections for this frame."""
        self.detection_counts.append(len(detections))
        if detections:
            self.confidences.extend([d['confidence'] for d in detections])
    
    def get_fps(self) -> float:
        """Get current processing FPS."""
        if not self.frame_times:
            return 0
        return 1.0 / np.mean(self.frame_times)
    
    def get_stats(self) -> Dict:
        """Get all statistics."""
        return {
            'avg_fps': float(self.get_fps()),
            'total_detections': sum(self.detection_counts),
            'detections_per_frame': float(np.mean(self.detection_counts)) if self.detection_counts else 0,
            'avg_confidence': float(np.mean(self.confidences)) if self.confidences else 0,
            'max_confidence': float(max(self.confidences)) if self.confidences else 0,
            'min_confidence': float(min(self.confidences)) if self.confidences else 0,
            'processing_time': time.time() - self.start_time
        }


class AdaptiveFrameSkipper:
    """Adaptively adjusts frame skip based on motion detection."""
    
    def __init__(self, base_skip: int = 1):
        """
        Args:
            base_skip: Base frame skip value
        """
        self.base_skip = base_skip
        self.last_frame = None
    
    def should_process_frame(self, frame: np.ndarray) -> bool:
        """
        Determine if frame should be processed based on motion.
        """
        if self.last_frame is None:
            self.last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return True
        
        # Calculate optical flow or frame difference
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(self.last_frame, gray)
        motion_level = np.mean(frame_diff)
        
        self.last_frame = gray
        
        # Process frame if significant motion detected
        return motion_level > 5  # Threshold can be tuned


def draw_detections_enhanced(frame: np.ndarray, detections: List[Dict], 
                             include_stats: bool = False) -> np.ndarray:
    """
    Draw detections on frame with enhanced visualization.
    
    Args:
        frame: Input frame
        detections: List of detections
        include_stats: Whether to include confidence in labels
    
    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    
    for det in detections:
        bbox = det['bbox']
        color_hex = det['severity']['color']
        
        # Convert hex color to BGR
        color = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
        
        # Draw filled rectangle background for label
        cv2.rectangle(annotated, (bbox['x1'], bbox['y1']), 
                    (bbox['x2'], bbox['y2']), color, 2)
        
        # Create label text
        label = f"Pothole {det['confidence']:.1f}%"
        if include_stats:
            label += f" ({det['severity']['level']})"
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw background rectangle for text
        cv2.rectangle(annotated, 
                    (bbox['x1'], bbox['y1'] - text_height - 8),
                    (bbox['x1'] + text_width + 5, bbox['y1']),
                    color, -1)
        
        # Draw text
        cv2.putText(annotated, label, 
                   (bbox['x1'] + 2, bbox['y1'] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return annotated
