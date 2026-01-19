## Custom Training Plots

- Purpose: Generate custom loss and validation metric plots from any YOLOv8 run.
- Script: plot_curves.py

### Quick Use

```
python plot_curves.py --run runs/pothole_detector_20251229_125847
```

- Outputs are saved to runs/pothole_detector_20251229_125847/custom_plots:
   - losses.png: Train vs Val losses (box/cls/dfl if present)
   - metrics.png: Validation metrics (precision, recall, mAP50, mAP50-95)

### Options

- `--run <folder>`: Path to a run directory containing results.csv
- `--csv <file>`: Path directly to a results.csv file
- `--out <folder>`: Custom output directory (defaults to `<run>/custom_plots`)

If `pandas` or `matplotlib` are missing, install them:

```
python -m pip install pandas matplotlib
```

# Automated Road Defect Detection and Severity Analysis

## Overview

This project implements an automated pothole detection system using **YOLOv8** deep learning model. The system can detect potholes in road images, videos, and real-time webcam streams with severity analysis.

## ✨ Features

### Core Detection
- **Real-time Pothole Detection**: Uses YOLOv8 for accurate object detection
- **Severity Analysis**: Classifies road conditions into severity levels (None, Low, Medium, High, Critical)
- **Batch Processing**: Process multiple images at once

### 🎬 Video & Streaming (NEW!)
- **Video File Processing**: Analyze video footage for road inspections
- **Real-time Webcam Streaming**: Live detection from connected camera
- **Optimized Inference**: GPU acceleration and frame skipping for fast processing
- **Progress Tracking**: Real-time progress indicators for video processing

### Web Interface
- **Modern Web UI**: Three-tab interface for Image/Video/Webcam detection
- **Drag & Drop Upload**: Easy file uploading
- **Live Results**: Real-time detection visualization
- **Downloadable Output**: Save annotated videos and images

### Deployment
- **Exportable Models**: Export trained models to ONNX format
- **REST API**: FastAPI backend with comprehensive endpoints
- **WebSocket Support**: Real-time streaming capabilities

## 🚀 Quick Start

### Option 1: Web Interface (Recommended)
```bash
# Start the server
python start_server.py

# Open browser
http://localhost:8000
```

### Option 2: Command Line
```bash
# Image detection
python inference.py --model runs/pothole_cpu_round10/weights/best.pt --source image.jpg

# Video detection
python inference.py --model runs/pothole_cpu_round10/weights/best.pt --source video.mp4
```

### Option 3: Python API
```python
from inference import PotholeDetector

detector = PotholeDetector('model.pt')

# Image
result = detector.predict('road.jpg')

# Video
result = detector.predict_video('road.mp4', device='cuda', frame_skip=2)
```

## Project Structure

```
POTHOLE_DETECTION/
├── backend/
│   ├── app.py              # FastAPI server (NEW: video & webcam endpoints)
│   └── requirements.txt    # Backend dependencies
├── frontend/
│   └── index.html          # Web interface (NEW: 3-tab UI)
├── train/                  # Training data
├── valid/                  # Validation data
├── test/                   # Test data
├── runs/                   # Training outputs & models
├── data.yaml               # Dataset configuration
├── train_pothole_detector.py   # Training script
├── inference.py            # Inference script (NEW: optimized video)
├── start_server.py         # Quick start script (NEW)
├── demo.py                 # Demo scripts (NEW)
├── requirements.txt        # Python dependencies
├── VIDEO_DETECTION_GUIDE.md    # Detailed docs (NEW)
├── QUICK_REFERENCE.md      # Quick reference (NEW)
└── README.md              # This file
```

## Installation

1. **Clone/Download the repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # For backend
   pip install -r backend/requirements.txt
   ```

3. **Verify GPU support** (optional but recommended):
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

4. **For GPU acceleration** (5-10x faster):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Usage

### 🌐 Web Interface (Easiest)

1. **Start the server**:
   ```bash
   python start_server.py
   ```

2. **Open browser**: `http://localhost:8000`

3. **Choose your mode**:
   - **📷 Image Tab**: Upload and analyze single images
   - **🎬 Video Tab**: Upload and process video files
   - **📹 Webcam Tab**: Real-time detection from webcam

### 💻 Command Line

#### Image Detection
```bash
python inference.py --model runs/pothole_cpu_round10/weights/best.pt --source image.jpg
```

#### Video Detection (NEW!)
```bash
# Basic video processing
python inference.py --model runs/pothole_cpu_round10/weights/best.pt --source video.mp4

# With GPU acceleration (5-10x faster)
python inference.py --model runs/pothole_cpu_round10/weights/best.pt --source video.mp4 --device cuda

# With frame skipping (2-3x faster)
python inference.py --model runs/pothole_cpu_round10/weights/best.pt --source video.mp4 --frame-skip 2
```

#### Batch Processing
```bash
# Process all images in a directory
python demo.py  # Interactive demo with batch processing
```

### 🐍 Python API

```python
from inference import PotholeDetector

# Initialize detector
detector = PotholeDetector('runs/pothole_cpu_round10/weights/best.pt')

# Image detection
result = detector.predict('road.jpg', conf_threshold=0.25)
print(f"Found {result['num_detections']} potholes")
print(f"Severity: {result['overall_severity']}")

# Video detection with optimizations
result = detector.predict_video(
    video_path='road.mp4',
    output_path='output.mp4',
    conf_threshold=0.25,
    frame_skip=2,      # Process every 2nd frame
    device='cuda',     # Use GPU
    output_format='mp4'
)
print(f"Processed {result['processed_frames']} frames")
print(f"Found {result['detections']} potholes")
```

### 🔌 REST API

```bash
# Image detection
curl -X POST http://localhost:8000/predict -F "file=@image.jpg"

# Video detection (NEW!)
curl -X POST http://localhost:8000/predict-video -F "file=@video.mp4"

# Health check
curl http://localhost:8000/health

# API documentation
http://localhost:8000/docs
```

### Training

Run the training script to train the YOLOv8 model:

```bash
python train_pothole_detector.py
```

This will:
- Verify the dataset structure
- Train YOLOv8 for 50 epochs
- Evaluate on validation set
- Generate training report
- Perform severity analysis on test images
- Export model to ONNX format

## 🎬 Video Detection Features

### Performance Optimizations
- **GPU Acceleration**: 5-10x faster with CUDA support
- **Frame Skipping**: Process every nth frame (2-3x faster)
- **Reduced Resolution**: Smaller inference size (416x416)
- **Batch Processing**: Efficient memory usage

### Expected Performance
| Video Length | Resolution | Device | Time |
|--------------|------------|--------|------|
| 30 seconds   | 720p       | CPU    | ~10-15 sec |
| 30 seconds   | 720p       | GPU    | ~2-3 sec |
| 1 minute     | 1080p      | GPU    | ~5-8 sec |

### Supported Formats
- **Input**: MP4, AVI, MOV, MKV, WebM
- **Output**: MP4 (recommended), AVI

## 📹 Webcam Streaming

Real-time detection from connected webcam:
1. Open web interface at `http://localhost:8000`
2. Switch to "📹 Webcam Stream" tab
3. Click "▶️ Start Webcam"
4. View live detections with bounding boxes
5. Monitor FPS and detection count

**Requirements**:
- Connected webcam
- Browser with WebSocket support (Chrome recommended)
- Camera permissions granted

## Severity Classification

The system classifies road conditions based on:

| Severity | Coverage | Description |
|----------|----------|-------------|
| None | 0% | No potholes detected |
| Low | < 5% | Minor surface damage |
| Medium | 5-15% | Moderate road damage |
| High | 15-30% | Significant damage requiring attention |
| Critical | > 30% | Severe damage requiring immediate repair |

## Training Configuration

Default training parameters (can be modified in `train_pothole_detector.py`):

- **Model**: YOLOv8n (nano) - fastest, suitable for real-time
- **Epochs**: 50
- **Batch Size**: 16
- **Image Size**: 640x640
- **Optimizer**: Auto (SGD/AdamW)
- **Early Stopping**: 10 epochs patience

### Model Variants

You can change `MODEL_SIZE` in the config to:
- `yolov8n.pt` - Nano (fastest, least accurate)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (slowest, most accurate)

## Output

After training, results are saved in:

```
runs/pothole_detector_YYYYMMDD_HHMMSS/
├── weights/
│   ├── best.pt         # Best model weights
│   └── last.pt         # Last epoch weights
├── results.csv         # Training metrics
├── confusion_matrix.png
├── F1_curve.png
├── PR_curve.png
├── P_curve.png
├── R_curve.png
├── severity_analysis/  # Test image analysis
│   ├── analyzed_*.jpg
│   └── analysis_summary.csv
├── exports/
│   └── best.onnx       # ONNX export
└── training_report.txt # Summary report
```

## API Usage

```python
from inference import PotholeDetector

# Initialize detector
detector = PotholeDetector("path/to/best.pt")

# Run prediction
result = detector.predict("road_image.jpg")

print(f"Severity: {result['overall_severity']}")
print(f"Potholes found: {result['num_detections']}")
print(f"Road coverage: {result['coverage_percent']:.2f}%")

# Access annotated image
annotated_image = result['annotated_image']
```

## Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+ (for GPU training/inference - optional but recommended)
- 8GB+ RAM (16GB+ recommended for video processing)
- NVIDIA GPU with 4GB+ VRAM (recommended for real-time performance)
- Webcam (for live streaming feature)

## 📚 Documentation

- **[VIDEO_DETECTION_GUIDE.md](VIDEO_DETECTION_GUIDE.md)** - Comprehensive guide for video features
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick reference card for all features
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical implementation details
- **API Docs** - `http://localhost:8000/docs` (when server is running)

## 🎮 Demo Scripts

Run interactive demos to explore features:
```bash
python demo.py
```

Choose from:
1. Image Detection Demo
2. Video Detection Demo
3. Batch Processing Demo
4. API Usage Examples

## 🐛 Troubleshooting

### "API Offline" message
```bash
# Check server status
curl http://localhost:8000/health

# Restart server
python start_server.py
```

### Slow video processing
- Install GPU support: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Use frame skipping: `--frame-skip 2`
- Reduce video resolution before uploading

### Webcam not working
- Grant browser camera permissions
- Ensure no other app is using the camera
- Use Chrome browser (best WebSocket support)

## 🔄 Recent Updates (v2.0)

- ✅ Added video file detection with optimized processing
- ✅ Added real-time webcam streaming with WebSocket
- ✅ Implemented GPU acceleration for 5-10x speedup
- ✅ Added frame skipping for faster video processing
- ✅ Created modern 3-tab web interface
- ✅ Added REST API endpoints for video detection
- ✅ Comprehensive documentation and demos
- ✅ Quick start script for easy setup

## License

This project is for educational and research purposes.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Dataset structured in YOLO format
- FastAPI for backend framework
- WebSocket support for real-time streaming

---

**Version**: 2.0 | **Last Updated**: January 8, 2026 | **Status**: ✅ Production Ready
