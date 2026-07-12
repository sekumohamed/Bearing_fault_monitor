import io
import csv
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List
from pydantic import BaseModel
from model import BearingFaultCNN

# ---- Config ----
WINDOW_SIZE = 1024
STRIDE = 512
CLASS_NAMES = ['Normal', 'IR_007', 'IR_014', 'IR_021', 'Ball_007',
               'Ball_014', 'Ball_021', 'OR_007', 'OR_014', 'OR_021']

MODEL_FILES = {
    'Standard_SGD': 'model_Standard_SGD.pth',
    'Momentum_SGD': 'model_Momentum_SGD.pth',
    'Nesterov_SGD': 'model_Nesterov_SGD.pth',
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Bearing Fault Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load all 3 trained models at startup ----
loaded_models = {}
for name, filepath in MODEL_FILES.items():
    model = BearingFaultCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()
    loaded_models[name] = model

print(f"Loaded models: {list(loaded_models.keys())}")

def normalize_window(window):
    mean = np.mean(window)
    std = np.std(window)
    return (window - mean) / (std + 1e-8)


def create_windows(signal, window_size, stride):
    windows = []
    for start in range(0, len(signal) - window_size + 1, stride):
        windows.append(signal[start:start + window_size])
    return np.array(windows)


def run_prediction(model, windows_array):
    """windows_array: shape (N, WINDOW_SIZE), already normalized."""
    X = torch.tensor(windows_array, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        outputs = model(X)
        probs = F.softmax(outputs, dim=1)
        confidences, preds = torch.max(probs, dim=1)
    return preds.cpu().numpy(), confidences.cpu().numpy(), probs.cpu().numpy()


# =========================================================
# ENDPOINTS
# =========================================================

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.get("/models")
def get_models():
    return {"models": list(MODEL_FILES.keys())}


@app.get("/classes")
def get_classes():
    return {"classes": CLASS_NAMES}



@app.post("/predict-file")
async def predict_file(model_name: str = Query(...), file: UploadFile = File(...)):
    if model_name not in loaded_models:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    contents = await file.read()
    filename = file.filename.lower()

    try:
        if filename.endswith('.mat'):
            data = scipy.io.loadmat(io.BytesIO(contents))
            de_key = [key for key in data.keys() if 'DE_time' in key]
            if not de_key:
                raise HTTPException(status_code=400, detail="No DE_time signal found in .mat file")
            signal = data[de_key[0]].flatten()
        elif filename.endswith('.npy'):
            signal = np.load(io.BytesIO(contents)).flatten()
        elif filename.endswith('.csv'):
            signal = np.loadtxt(io.BytesIO(contents), delimiter=',').flatten()
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use .mat, .npy, or .csv")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

    if len(signal) < WINDOW_SIZE:
        raise HTTPException(status_code=400, detail=f"Signal too short. Need at least {WINDOW_SIZE} samples, got {len(signal)}")

    windows = create_windows(signal, WINDOW_SIZE, STRIDE)
    windows_norm = np.array([normalize_window(w) for w in windows])

    model = loaded_models[model_name]
    preds, confidences, probs = run_prediction(model, windows_norm)

    # Majority vote across all windows
    pred_counts = np.bincount(preds, minlength=len(CLASS_NAMES))
    majority_pred_idx = int(np.argmax(pred_counts))
    majority_confidence = float(np.mean(confidences[preds == majority_pred_idx]))

    per_window_results = [
        {
            "window_index": i,
            "predicted_label": CLASS_NAMES[int(preds[i])],
            "confidence": float(confidences[i])
        }
        for i in range(len(preds))
    ]

    return {
        "num_windows": len(windows),
        "majority_prediction": CLASS_NAMES[majority_pred_idx],
        "majority_confidence": majority_confidence,
        "class_vote_counts": {CLASS_NAMES[i]: int(pred_counts[i]) for i in range(len(CLASS_NAMES))},
        "per_window_results": per_window_results,
        "waveform_preview": signal[:2000].tolist()  # first 2000 samples for plotting
    }


class WindowsRequest(BaseModel):
    model_name: str = None
    windows: list 

# Map filename substrings to known class labels (for validation when uploading
# original CWRU-style files that already carry ground-truth in their names)
FILENAME_LABEL_HINTS = {
    'normal': 'Normal',
    'ir007': 'IR_007', 'ir014': 'IR_014', 'ir021': 'IR_021',
    'b007': 'Ball_007', 'b014': 'Ball_014', 'b021': 'Ball_021',
    'or007': 'OR_007', 'or014': 'OR_014', 'or021': 'OR_021',
}


def infer_true_label_from_filename(filename: str):
    fname = filename.lower().replace('_', '').replace('-', '').replace('.', '')
    for key, label in FILENAME_LABEL_HINTS.items():
        if key in fname:
            return label
    return None


def parse_signal_from_upload(filename: str, contents: bytes):
    fname = filename.lower()
    if fname.endswith('.mat'):
        data = scipy.io.loadmat(io.BytesIO(contents))
        de_key = [key for key in data.keys() if 'DE_time' in key]
        if not de_key:
            raise HTTPException(status_code=400, detail=f"No DE_time signal found in {filename}")
        return data[de_key[0]].flatten()
    elif fname.endswith('.npy'):
        return np.load(io.BytesIO(contents)).flatten()
    elif fname.endswith('.csv'):
        return np.loadtxt(io.BytesIO(contents), delimiter=',').flatten()
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")


@app.post("/predict-files")
async def predict_files(model_name: str = Query(...), files: List[UploadFile] = File(...)):
    if model_name not in loaded_models:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    model = loaded_models[model_name]
    all_windows = []
    all_sources = []
    all_true_labels = []
    per_file_summary = {}

    for f in files:
        contents = await f.read()
        try:
            signal = parse_signal_from_upload(f.filename, contents)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse {f.filename}: {str(e)}")

        if len(signal) < WINDOW_SIZE:
            raise HTTPException(status_code=400, detail=f"{f.filename} too short: needs {WINDOW_SIZE}+ samples")

        windows = create_windows(signal, WINDOW_SIZE, STRIDE)
        windows_norm = [normalize_window(w) for w in windows]

        true_label = infer_true_label_from_filename(f.filename)

        all_windows.extend(windows_norm)
        all_sources.extend([f.filename] * len(windows_norm))
        all_true_labels.extend([true_label] * len(windows_norm))

        per_file_summary[f.filename] = {
            "num_windows": len(windows_norm),
            "inferred_true_label": true_label
        }

    windows_arr = np.array(all_windows)
    preds, confidences, probs = run_prediction(model, windows_arr)

    results = []
    for i in range(len(preds)):
        pred_label = CLASS_NAMES[int(preds[i])]
        true_label = all_true_labels[i]
        results.append({
            "source_file": all_sources[i],
            "window": windows_arr[i].tolist(),
            "predicted_label": pred_label,
            "confidence": float(confidences[i]),
            "probabilities": {CLASS_NAMES[j]: float(probs[i][j]) for j in range(len(CLASS_NAMES))},
            "true_label": true_label,
            "correct": (true_label == pred_label) if true_label else None
        })

    return {
        "total_windows": len(results),
        "per_file_summary": per_file_summary,
        "results": results
    }


@app.post("/predict-windows")
def predict_windows(req: WindowsRequest):
    """Re-run prediction on already-uploaded windows with a different model (no re-upload needed)."""
    if req.model_name not in loaded_models:
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model_name}")

    model = loaded_models[req.model_name]
    windows_arr = np.array(req.windows)
    preds, confidences, probs = run_prediction(model, windows_arr)

    results = []
    for i in range(len(preds)):
        results.append({
            "predicted_label": CLASS_NAMES[int(preds[i])],
            "confidence": float(confidences[i]),
            "probabilities": {CLASS_NAMES[j]: float(probs[i][j]) for j in range(len(CLASS_NAMES))}
        })
    return {"results": results}

@app.post("/predict-all-models")
def predict_all_models(req: WindowsRequest):
    windows_arr = np.array(req.windows)
    output = {}
    for name, model in loaded_models.items():
        preds, confidences, probs = run_prediction(model, windows_arr)
        output[name] = {
            "predicted_labels": [CLASS_NAMES[int(p)] for p in preds],
            "confidences": [float(c) for c in confidences]
        }
    return {"models": output}


@app.get("/comparison-summary")
def comparison_summary():
    rows = []
    try:
        with open('comparison_summary.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="comparison_summary.csv not found")
    return {"rows": rows}

# ---- Serve the frontend dashboard ----
@app.get("/")
def serve_dashboard():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
