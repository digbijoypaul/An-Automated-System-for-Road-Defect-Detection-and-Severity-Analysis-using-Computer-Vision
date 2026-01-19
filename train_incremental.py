"""
Incremental Training Script for Pothole Detection
Trains in batches of 5 epochs up to 50 total epochs
"""
from ultralytics import YOLO
import os

# Configuration
DATA_YAML = "c:/Users/knida/Downloads/POTHOLE_DETECTION/data.yaml"
PROJECT_DIR = "c:/Users/knida/Downloads/POTHOLE_DETECTION/runs"
EPOCHS_PER_ROUND = 5
TOTAL_EPOCHS = 50
BATCH_SIZE = 2
IMAGE_SIZE = 320

def get_latest_model():
    """Find the latest trained model."""
    runs_dir = PROJECT_DIR
    
    # Check for round models first
    for i in range(20, 0, -1):
        model_path = f"{runs_dir}/pothole_cpu_round{i}/weights/best.pt"
        if os.path.exists(model_path):
            return model_path, i * 5 + 5  # round number * 5 + initial 5
    
    # Check initial training
    initial_model = f"{runs_dir}/pothole_cpu_round2/weights/best.pt"
    if os.path.exists(initial_model):
        return initial_model, 10
    
    initial_model = f"{runs_dir}/pothole_cpu_light/weights/best.pt"
    if os.path.exists(initial_model):
        return initial_model, 5
    
    return "yolov8n.pt", 0

def train_round(model_path, round_num):
    """Train one round of 5 epochs."""
    print(f"\n{'='*60}")
    print(f"TRAINING ROUND {round_num} (Epochs {(round_num-1)*5 + 1}-{round_num*5})")
    print(f"Starting from: {model_path}")
    print(f"{'='*60}\n")
    
    model = YOLO(model_path)
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS_PER_ROUND,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        device='cpu',
        patience=10,  # Increased patience since we're doing rounds
        workers=0,
        project=PROJECT_DIR,
        name=f'pothole_cpu_round{round_num}'
    )
    
    return f"{PROJECT_DIR}/pothole_cpu_round{round_num}/weights/best.pt"

def main():
    # Find where we left off
    current_model, completed_epochs = get_latest_model()
    
    print(f"\n{'='*60}")
    print(f"INCREMENTAL TRAINING TO {TOTAL_EPOCHS} EPOCHS")
    print(f"{'='*60}")
    print(f"Completed epochs: {completed_epochs}")
    print(f"Starting model: {current_model}")
    print(f"Remaining epochs: {TOTAL_EPOCHS - completed_epochs}")
    print(f"{'='*60}\n")
    
    if completed_epochs >= TOTAL_EPOCHS:
        print("Training already complete!")
        return
    
    # Calculate starting round
    start_round = (completed_epochs // 5) + 1
    total_rounds = TOTAL_EPOCHS // EPOCHS_PER_ROUND
    
    # Train remaining rounds
    for round_num in range(start_round, total_rounds + 1):
        current_model = train_round(current_model, round_num)
        print(f"\n✓ Round {round_num} complete! Model saved to: {current_model}")
        print(f"  Progress: {round_num * 5}/{TOTAL_EPOCHS} epochs")
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"Final model: {current_model}")
    print(f"Total epochs: {TOTAL_EPOCHS}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
