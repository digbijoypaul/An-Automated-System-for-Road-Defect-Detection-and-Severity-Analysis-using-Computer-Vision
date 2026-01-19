"""
FastAPI Backend for Pothole Detection Model
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
import numpy as np
import cv2
import base64
from ultralytics import YOLO
from pathlib import Path
import threading
import tempfile
import os
import sys
from pathlib import Path as PathlibPath

# Import video processor from same directory
sys.path.insert(0, str(PathlibPath(__file__).parent))
from video_processor import DetectionSmoother, DetectionMerger, VideoStatistics, draw_detections_enhanced
import time

app = FastAPI(
    title="Pothole Detection API",
    description="API for detecting potholes in road images using YOLOv8",
    version="1.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
MODEL_PATH = Path(__file__).parent.parent / "runs" / "pothole_cpu_round10" / "weights" / "best.pt"
model = None

def load_model():
    global model
    if model is None:
        model = YOLO(str(MODEL_PATH))
    return model

def get_severity(area_ratio):
    """Determine severity based on pothole area ratio."""
    if area_ratio < 0.05:
        return {"level": "Low", "color": "#22c55e", "description": "Minor surface damage"}
    elif area_ratio < 0.15:
        return {"level": "Medium", "color": "#f59e0b", "description": "Moderate road damage"}
    elif area_ratio < 0.30:
        return {"level": "High", "color": "#ef4444", "description": "Significant road hazard"}
    else:
        return {"level": "Critical", "color": "#dc2626", "description": "Severe road damage - Immediate attention required"}

# Path to frontend
FRONTEND_PATH = Path(__file__).parent.parent / "frontend" / "index.html"

# Global processing state
processing_tasks = {}
processing_lock = threading.Lock()

@app.on_event("startup")
async def startup_event():
    load_model()
    print(f"Model loaded from {MODEL_PATH}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML page with cache-busting headers."""
    if FRONTEND_PATH.exists():
        response = FileResponse(FRONTEND_PATH)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    return HTMLResponse("<h1>Pothole Detection API</h1><p>Frontend not found.</p>")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Detect potholes in uploaded image.
    Returns bounding boxes, confidence scores, and severity analysis.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        img_height, img_width = image.shape[:2]
        img_area = img_height * img_width
        
        # Run inference
        model = load_model()
        results = model(image, conf=0.25)[0]
        
        # Process detections
        detections = []
        total_pothole_area = 0
        
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            
            # Calculate area ratio - convert to Python native float
            box_area = float((x2 - x1) * (y2 - y1))
            area_ratio = float(box_area / img_area)
            total_pothole_area += box_area
            
            severity = get_severity(area_ratio)
            
            detections.append({
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                },
                "confidence": float(round(confidence * 100, 1)),
                "area_ratio": float(round(area_ratio * 100, 2)),
                "severity": severity
            })
        
        # Draw bounding boxes on image
        annotated_image = image.copy()
        for det in detections:
            bbox = det["bbox"]
            color_hex = det["severity"]["color"]
            # Convert hex to BGR
            color = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
            
            cv2.rectangle(
                annotated_image,
                (bbox["x1"], bbox["y1"]),
                (bbox["x2"], bbox["y2"]),
                color, 3
            )
            
            # Add label
            label = f"Pothole {det['confidence']}%"
            cv2.putText(
                annotated_image, label,
                (bbox["x1"], bbox["y1"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
        
        # Save annotated image to outputs folder
        backend_dir = Path(__file__).parent
        outputs_dir = backend_dir / "outputs"
        outputs_dir.mkdir(exist_ok=True)
        
        # Generate output filename
        output_filename = f"detected_{file.filename}"
        output_path = outputs_dir / output_filename
        cv2.imwrite(str(output_path), annotated_image)
        
        # Encode annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Overall analysis
        overall_severity = get_severity(total_pothole_area / img_area) if detections else None
        
        # Convert numpy types to Python native types for JSON serialization
        response_data = {
            "success": True,
            "image_size": {"width": int(img_width), "height": int(img_height)},
            "detections_count": len(detections),
            "detections": detections,
            "overall_severity": overall_severity,
            "annotated_image": f"data:image/jpeg;base64,{img_base64}",
            "total_area_affected": float(round((total_pothole_area / img_area) * 100, 2)),
            "saved_image_path": str(output_path),
            "download_url": f"/download-image/{output_filename}"
        }
        
        return JSONResponse(response_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...), conf_threshold: float = 0.25, enhance: bool = True):
    """
    Detect potholes in uploaded video file with advanced enhancements.
    
    Args:
        file: Video file (MP4, AVI, MOV, etc.)
        conf_threshold: Confidence threshold (0.0-1.0, default 0.25)
        enhance: Enable smoothing, merging, and advanced statistics (default True)
    
    Returns:
        task_id for polling status
    """
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        # Generate unique task ID
        import uuid
        task_id = str(uuid.uuid4())
        
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            contents = await file.read()
            tmp_video.write(contents)
            tmp_video_path = tmp_video.name
        
        # Initialize task status
        with processing_lock:
            processing_tasks[task_id] = {
                'status': 'processing',
                'progress': 0,
                'message': 'Starting video processing...',
                'total_frames': 0,
                'processed_frames': 0,
                'current_frame': 0
            }
        
        # Start processing in background thread
        def process_video_task():
            try:
                model = load_model()
                
                # Get video properties
                cap = cv2.VideoCapture(tmp_video_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                # Update task with total frames
                with processing_lock:
                    processing_tasks[task_id]['total_frames'] = total_frames
                    processing_tasks[task_id]['message'] = f'Processing {total_frames} frames...'
            
                # Process video with frame skipping for performance
                output_video_path = Path(tmp_video_path).parent / "output_video.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
                
                cap = cv2.VideoCapture(tmp_video_path)
                frame_count = 0
                processed_frames = 0
                all_detections = []
                last_detections = []
                start_time = time.time()
                
                # Initialize enhancement modules
                smoother = DetectionSmoother(window_size=3) if enhance else None
                merger = DetectionMerger(merge_threshold=0.3) if enhance else None
                stats = VideoStatistics() if enhance else None
                
                # Process every frame (or every nth frame for speed)
                frame_skip = max(1, total_frames // 100)  # Target ~100 processed frames max
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    frame_start = time.time()
                    
                    # Update progress every 10 frames
                    if frame_count % 10 == 0:
                        progress = int((frame_count / total_frames) * 100)
                        elapsed = time.time() - start_time
                        
                        with processing_lock:
                            processing_tasks[task_id]['progress'] = progress
                            processing_tasks[task_id]['current_frame'] = frame_count
                            processing_tasks[task_id]['message'] = f'Processing frame {frame_count}/{total_frames} ({progress}%)'
                    
                    annotated_frame = frame.copy()
                    detections = []
                    
                    # Run inference on selected frames
                    if frame_count % frame_skip == 0:
                        results = model(frame, conf=conf_threshold)[0]
                        
                        img_height, img_width = frame.shape[:2]
                        img_area = img_height * img_width
                        
                        # Process detections
                        for box in results.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0].cpu().numpy())
                            
                            box_area = float((x2 - x1) * (y2 - y1))
                            area_ratio = float(box_area / img_area)
                            severity = get_severity(area_ratio)
                            
                            detections.append({
                                "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                                "confidence": float(round(confidence * 100, 1)),
                                "area_ratio": float(round(area_ratio * 100, 2)),
                                "severity": severity
                            })
                        
                        # Apply enhancements
                        if enhance and detections:
                            detections = merger.merge_detections(detections)  # Merge overlapping
                            detections = smoother.smooth_detections(detections)  # Smooth jitter
                        
                        last_detections = detections
                        all_detections.extend(detections)
                        processed_frames += 1
                        
                        if enhance and stats:
                            stats.add_detections(detections)
                    else:
                        # Use last frame's detections for skipped frames
                        detections = last_detections
                    
                    # Draw detections on current frame with enhanced visualization
                    if enhance:
                        annotated_frame = draw_detections_enhanced(annotated_frame, detections, include_stats=True)
                    else:
                        # Basic drawing
                        for det in detections:
                            bbox = det["bbox"]
                            color_hex = det["severity"]["color"]
                            color = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
                            
                            cv2.rectangle(annotated_frame, (bbox["x1"], bbox["y1"]), 
                                        (bbox["x2"], bbox["y2"]), color, 2)
                            label = f"Pothole {det['confidence']:.1f}%"
                            cv2.putText(annotated_frame, label, (bbox["x1"], bbox["y1"] - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    out.write(annotated_frame)
                    
                    # Track frame time
                    if enhance and stats:
                        frame_time = time.time() - frame_start
                        stats.add_frame_time(frame_time)
                
                cap.release()
                out.release()
                
                # Get final statistics
                final_stats = {}
                if enhance and stats:
                    final_stats = stats.get_stats()
                
                # Instead of returning the video directly, save it and provide a download link
                # This avoids memory issues with large videos
                output_filename = f"processed_{file.filename}"
                # Use absolute path to ensure it works regardless of where the app is run from
                import shutil
                backend_dir = Path(__file__).parent
                outputs_dir = backend_dir / "outputs"
                outputs_dir.mkdir(exist_ok=True)
                final_output_path = outputs_dir / output_filename
                
                # Move the output video
                shutil.move(str(output_video_path), str(final_output_path))
                
                # Cleanup temp files
                os.unlink(tmp_video_path)
                
                # Count total detections
                total_detections = len(all_detections)
                
                # Update task status to completed
                with processing_lock:
                    result = {
                        'status': 'completed',
                        'progress': 100,
                        'message': 'Video processing complete!',
                        'video_size': {"width": width, "height": height},
                        'fps': fps,
                        'total_frames': total_frames,
                        'processed_frames': processed_frames,
                        'total_detections': total_detections,
                        'detections_per_frame': float(total_detections / max(1, processed_frames)),
                        'sample_detections': all_detections[:10],
                        'download_url': f"/download/{output_filename}"
                    }
                    
                    # Add statistics if enhancement was enabled
                    if enhance and final_stats:
                        result['statistics'] = final_stats
                    
                    processing_tasks[task_id].update(result)
                
            except Exception as e:
                # Update task status to failed
                with processing_lock:
                    processing_tasks[task_id].update({
                        'status': 'failed',
                        'progress': 0,
                        'message': f'Error: {str(e)}',
                        'error': str(e)
                    })
                try:
                    os.unlink(tmp_video_path)
                except:
                    pass
        
        # Start background thread
        thread = threading.Thread(target=process_video_task, daemon=True)
        thread.start()
        
        # Return task ID immediately
        return JSONResponse({
            "success": True,
            "task_id": task_id,
            "message": "Video processing started. Use /video-status/{task_id} to check progress."
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video-status/{task_id}")
async def get_video_status(task_id: str):
    """
    Get the processing status of a video task.
    
    Returns:
        - status: 'processing', 'completed', or 'failed'
        - progress: percentage (0-100)
        - current_frame: current frame being processed
        - total_frames: total frames in video
        - message: status message
        - download_url: URL to download processed video (only when completed)
    """
    with processing_lock:
        if task_id not in processing_tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task_info = processing_tasks[task_id].copy()
    
    return JSONResponse(task_info)

@app.get("/download-image/{filename}")
async def download_image(filename: str):
    """Download processed image file"""
    backend_dir = Path(__file__).parent
    file_path = backend_dir / "outputs" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="image/jpeg", filename=filename)

@app.get("/download/{filename}")
async def download_video(filename: str):
    """Download processed video file"""
    file_path = Path("backend") / "outputs" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
