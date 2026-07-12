---
title: Bearing Fault Monitor
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Bearing Fault Monitor Intelligent Industrial Fault Detection

A lightweight 1D-CNN system for detecting and classifying rolling bearing faults from vibration signals, with a controlled comparison of three SGD optimizer variants (Standard, Momentum, Nesterov). Built as a final-year ECE PBL project targeting semiconductor/hardware-adjacent placement roles.

## Live Demo
FastAPI backend + interactive dashboard for uploading vibration signals and viewing real-time fault predictions with optimizer comparison.

## Problem
Rolling bearings are among the most common failure points in rotating industrial machinery. Vibration signatures carry early warning signs of degradation (inner race, outer race, and ball faults) well before catastrophic failure — enabling predictive maintenance instead of reactive repair.

## Dataset
[CWRU Bearing Dataset](https://engineering.case.edu/bearingdatacenter) — 12kHz Drive End accelerometer data, 1hp motor load condition.

**10 classes:** Normal + 3 fault types (Inner Race, Ball, Outer Race) x 3 severities each (0.007", 0.014", 0.021").

## Methodology

1. **Windowing:** raw signals sliced into 1024-sample windows with 512-sample stride (50% overlap).
2. **Leakage prevention:** train/test split performed on the raw continuous signal *before* windowing, ensuring no overlapping window appears in both sets. Validated via a shuffled-label sanity check (model collapses to random-chance accuracy on nonsense labels, confirming no leakage).
3. **Normalization:** z-score normalization applied per window.
4. **Model:** lightweight 1D-CNN — 3 convolutional blocks (16/32/64 filters) with BatchNorm + ReLU + MaxPooling, followed by global average pooling and a fully-connected classifier. ~15,000 trainable parameters.
5. **Optimizer study:** identical architecture, weight initialization (seed=42), and data across three SGD configurations:
   - Standard SGD (momentum=0)
   - Momentum SGD (momentum=0.9)
   - Nesterov SGD (momentum=0.9, nesterov=True)

## Results

| Optimizer | Final Test Accuracy | Final Test Loss | Epochs to 95% Train Acc |
|---|---|---|---|
| Standard SGD | 61.37% | 2.2331 | 32 |
| Momentum SGD | 97.24% | 0.0964 | 7 |
| Nesterov SGD | 98.25% | 0.0568 | 7 |

**Key finding:** Standard SGD exhibited severe late-training instability and failed to reliably distinguish visually/statistically similar classes (Normal vs. mild inner-race fault), resulting in degraded test accuracy. Momentum-based variants converged both faster (7 vs. 32 epochs) and to substantially more stable, accurate solutions — demonstrating the practical necessity of momentum for this classification task.

See `early_epoch_convergence.png`, `full_curves.png`, and `confusion_matrices.png` for detailed training curves and per-class performance.

## Architecture
Vibration Sensor -> Raw Signal -> Window Segmentation -> Normalization
-> 1D-CNN (3 conv blocks + global avg pool + FC classifier)
-> Fault Classification (10 classes)

## Tech Stack

- **Model:** PyTorch (1D-CNN)
- **Backend:** FastAPI, Uvicorn
- **Frontend:** Vanilla JS, Chart.js
- **Data processing:** NumPy, SciPy, scikit-learn

## Running Locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision numpy scipy matplotlib pandas scikit-learn seaborn fastapi uvicorn python-multipart

python3 preprocess_data.py
python3 train.py
python3 generate_results.py

python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in a browser, upload a `.mat`/`.npy`/`.csv` vibration signal, and select an optimizer to see live predictions.

## Project Structure
```
├── data/                      
├── static/index.html          
├── app.py                     
├── model.py                   
├── preprocess_data.py        
├── train.py                   
├── generate_results.py        
├── sanity_check.py            
├── model_*.pth                
└── training_history.json
```

## Future Work

- Extend to multiple load conditions (cross-load generalization)
- Deploy on embedded/edge hardware for real-time monitoring
- Compare against Adam/AdamW optimizers

## Author

Seku Mohamed Hanifa A 