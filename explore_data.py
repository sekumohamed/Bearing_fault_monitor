import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Files to inspect
files = {
    'Normal': 'data/Normal_1.mat',
    'Inner Race Fault': 'data/IR007_1.mat',
    'Ball Fault': 'data/B007_1.mat',
    'Outer Race Fault': 'data/OR007_1.mat'
}

signals = {}

for label, filepath in files.items():
    data = scipy.io.loadmat(filepath)

    # Find the DE_time key automatically
    de_key = [key for key in data.keys() if 'DE_time' in key][0]
    signal = data[de_key].flatten()

    signals[label] = signal

    print(f"{label}:")
    print(f"  File: {filepath}")
    print(f"  Signal key: {de_key}")
    print(f"  Signal length: {len(signal)} samples")
    print()

# Plot all 4 signals for comparison (first 2000 samples each)
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

for ax, (label, signal) in zip(axes, signals.items()):
    ax.plot(signal[:2000])
    ax.set_title(label)
    ax.set_ylabel("Acceleration")
    ax.grid(True)

axes[-1].set_xlabel("Sample index")
plt.tight_layout()
plt.savefig('signal_comparison.png', dpi=150)
print("Saved plot to signal_comparison.png")