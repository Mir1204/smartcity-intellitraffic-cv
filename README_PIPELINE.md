# Traffic Signal Control System - Implementation Guide

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   TRAFFIC VIDEO INPUT                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  LAYER 1: VIDEO DETECTION (layer1_yolo/detector.py)             │
│  ─────────────────────────────────────────────────────────────  │
│  • YOLOv8n vehicle detection from video frames                   │
│  • SimpleTracker: centroid-based vehicle identity tracking      │
│  • Prevents double-counting across frames                        │
│  • Output: frame_logs (per-frame counts)                        │
│           aggregated_counts (windowed averages)                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    Vehicle Count Summary
                    (cars, buses, bikes)
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  LAYER 2: ML PREDICTION (layer2_ml/)                             │
│  ─────────────────────────────────────────────────────────────  │
│  • XGBoost classifier with 4 discrete outputs                    │
│  • Input: vehicle counts + rain condition                        │
│  • Output: green light timing (30s, 60s, 90s, 120s)             │
│  • Confidence score & class probabilities                        │
│  Files:                                                          │
│    - train_model.py: Model training (91.5% accuracy)            │
│    - predict.py: Inference on aggregated counts                 │
│    - data/synthetic_traffic.csv: Training data (2000 rows)      │
│    - models/xgb_green_time.joblib: Trained model               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    Signal Timing Recommendation
                        (30/60/90/120s)
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  LAYER 3: SUMO SIMULATION (layer3_sumo/)                        │
│  ─────────────────────────────────────────────────────────────  │
│  • SUMO traffic simulator integration                            │
│  • Adaptive vs fixed timing comparison                           │
│  • Performance metrics: throughput, wait times                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  LAYER 4: DASHBOARD (layer4_dashboard/)                         │
│  ─────────────────────────────────────────────────────────────  │
│  • Streamlit web interface                                        │
│  • Live visualization of:                                        │
│    - Video feed with detection overlays                         │
│    - Vehicle count trends                                       │
│    - Signal timing recommendations                              │
│    - Performance metrics                                        │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Test the Pipeline (No Video Required)

```bash
python test_pipeline.py
```

This runs three tests:
- **TEST 1**: ML model predictions with various traffic scenarios
- **TEST 2**: Temporal aggregation (simulated video)
- **TEST 3**: Integration status check

Output: `test_results.json`

### 2. Process Actual Traffic Video

```bash
python traffic_pipeline.py <video_path> [output_video_path] [rain]
```

**Parameters:**
- `video_path`: Path to input traffic video (MP4, MOV, AVI, etc.)
- `output_video_path` (optional): Save annotated video with detection overlays
- `rain` (optional): 0=no rain, 1=raining (affects predictions)

**Examples:**
```bash
# Basic video processing
python traffic_pipeline.py traffic.mp4

# With annotated output
python traffic_pipeline.py traffic.mp4 output_annotated.mp4

# With rain condition flag
python traffic_pipeline.py traffic.mp4 output_annotated.mp4 1
```

**Output:**
- Console report with per-window predictions
- JSON file: `layer3_sumo/results/<video_name>_predictions.json`

## Component Details

### Layer 1: Video Detection (detector.py)

**Key Classes:**
- `SimpleTracker`: Tracks vehicles across frames
  - `update(detections)`: Match detections to existing tracks
  - Prevents duplicate counting with Euclidean distance matching
  - Configurable: `max_dist=50px`, `max_frames_missing=30`

**Key Functions:**
- `detect_vehicles(image_source)`: Legacy image detection
- `process_video(video_path, output_path, aggregation_frames=30)`: 
  - Returns dict with `frame_logs` (per-frame) and `aggregated_counts` (windowed)
  - Annotates detected vehicles with bounding boxes and tracking IDs

**Vehicle Classification:**
```python
VEHICLE_GROUPS = {
    'CAR': [2],           # COCO class 2
    'BUS_TRUCK': [5, 7],  # COCO classes 5, 7
    'BIKE': [0, 1, 3]     # COCO classes 0, 1, 3
}
```

### Layer 2: ML Prediction (predict.py)

**Model Architecture:**
```
Input Features (4):
├── car_count: number of cars detected
├── bus_truck_count: number of buses/trucks
├── bike_count: number of motorcycles/bicycles
└── rain: weather condition (0 or 1)

↓ XGBClassifier (4 classes, softmax objective)

Output:
├── predicted_green_time: 30 | 60 | 90 | 120 seconds
├── confidence: probability of predicted class
└── probabilities: dict of all class probabilities
```

**Training Performance:**
```
Accuracy:  91.5%
Precision: 91.45%
Recall:    91.50%
F1-Score:  91.42%
```

**Class Distribution (Training Data):**
```
Class 0 (30s):   798 samples  (0-44.9s original)
Class 1 (60s):   994 samples  (45-74.9s original)
Class 2 (90s):   208 samples  (75-104.9s original)
Class 3 (120s):    0 samples  (105+s original)
```

⚠️ **Note**: Class 3 has no training data, limiting predictions to 90s max.

### Data Format

**Video Processing Output:**
```json
{
  "frame_logs": [
    {
      "frame": 0,
      "timestamp": 0.0,
      "car_count": 5,
      "bus_truck_count": 1,
      "bike_count": 2,
      "tracked_ids": ["CAR_001", "TRUCK_002", "BIKE_001"]
    },
    ...
  ],
  "aggregated_counts": [
    {
      "window": 0,
      "start_frame": 0,
      "end_frame": 30,
      "timestamp": 0.0,
      "car_count": 5,
      "bus_truck_count": 1,
      "bike_count": 2
    },
    ...
  ],
  "metadata": {
    "total_frames": 300,
    "fps": 30.0,
    "duration": 10.0,
    "unique_vehicles_tracked": 45
  }
}
```

**ML Prediction Output:**
```json
{
  "predicted_green_time": 60,
  "confidence": 0.9823,
  "probabilities": {
    "class_30s": 0.0105,
    "class_60s": 0.9823,
    "class_90s": 0.0072,
    "class_120s": 0.0000
  }
}
```

## Configuration Parameters

### Video Processing (detector.py)

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `aggregation_frames` | 30 | 10-120 | Window size for averaging (frames) |
| `confidence_threshold` | 0.5 | 0.1-0.9 | YOLO detection confidence |
| `max_dist` | 50 | 20-100 | Max distance for track matching (pixels) |
| `max_frames_missing` | 30 | 5-100 | Frames before track is dropped |

### Model Retraining

To improve predictions, especially for higher traffic volumes:

```bash
cd layer2_ml
python train_model.py
```

**To enhance training data:**
1. Collect more real traffic videos
2. Generate videos with high-traffic scenarios
3. Manually balance class distribution (especially class 3)
4. Adjust bucketing thresholds in `constants.py`

```python
# In utils/constants.py
CLASS_BOUNDARIES = {
    30: (0, 45),      # Adjust ranges as needed
    60: (45, 75),
    90: (75, 105),
    120: (105, float('inf'))
}
```

## Troubleshooting

### Issue: Model always predicts 30s

**Cause**: Training data imbalance (class 0 has 798 samples, class 3 has 0)

**Solution**: 
```bash
# Generate more balanced synthetic data
cd layer2_ml
python generate_dataset.py  # Modify to create balanced distribution
python train_model.py       # Retrain
```

### Issue: Vehicle counts seem too high/low

**Cause**: SimpleTracker distance threshold mismatch

**Solution**: Adjust in detector.py:
```python
# Increase if vehicles are getting double-counted
max_dist = 70  # was 50

# Decrease if vehicles are skipped
aggregation_frames = 20  # was 30
```

### Issue: Video processing is slow

**Cause**: High resolution video or complex scenes

**Solution**:
```bash
# Reduce frame processing with lower confidence
python traffic_pipeline.py video.mp4 output.mp4 0 --confidence 0.7

# Or skip annotation for speed
python -c "from layer1_yolo.detector import process_video; \
    process_video('video.mp4', output_path=None)"
```

## Integration Flow

### Step-by-Step Execution

```python
# 1. Process video → get vehicle counts
from layer1_yolo.detector import process_video
result = process_video('traffic.mp4')
aggregated = result['aggregated_counts']

# 2. For each aggregation window, predict signal timing
from layer2_ml.predict import predict_green_time_class
for window in aggregated:
    pred = predict_green_time_class(
        window['car_count'],
        window['bus_truck_count'],
        window['bike_count'],
        rain=0
    )
    print(f"Recommended green time: {pred['predicted_green_time']}s")
    print(f"Confidence: {pred['confidence']:.1%}")

# 3. Send to SUMO simulation or actual signal controller
# (Integration in layer3_sumo pending)
```

## Next Steps

1. ✅ **Video Processing**: Implemented with SimpleTracker
2. ✅ **ML Classification**: 4-class model with 91.5% accuracy
3. ✅ **Pipeline Integration**: Complete video→prediction pipeline
4. ⏳ **SUMO Integration**: Connect predictions to traffic simulator
5. ⏳ **Dashboard**: Real-time visualization interface
6. ⏳ **Real-world Testing**: Validate on actual traffic camera feeds
7. ⏳ **Model Improvement**: Collect balanced training data for class 3

## File Structure Reference

```
.
├── traffic_pipeline.py          # Main integration script
├── test_pipeline.py             # Testing suite
├── layer1_yolo/
│   ├── detector.py             # Video processing + tracking
│   └── sample_images/
├── layer2_ml/
│   ├── train_model.py          # Model training
│   ├── predict.py              # Inference
│   ├── generate_dataset.py      # Synthetic data generation
│   ├── data/
│   │   └── synthetic_traffic.csv
│   └── models/
│       └── xgb_green_time.joblib
├── layer3_sumo/
│   ├── run_adaptive.py         # SUMO adaptive simulation
│   ├── run_fixed.py            # SUMO fixed timing
│   └── results/
└── layer4_dashboard/
    └── app.py                  # Streamlit dashboard
```

## References

- **YOLO Detection**: ultralytics YOLOv8n
- **ML Classification**: XGBoost with softmax objective
- **Tracking**: Centroid-based multi-object tracking
- **Simulation**: SUMO (Simulation of Urban Mobility)
- **Dashboard**: Streamlit
- **Video Processing**: OpenCV (cv2)
