import scipy.io
import numpy as np

WINDOW_SIZE = 1024
STRIDE = 512
TRAIN_FRACTION = 0.8

# 10 classes: Normal + 3 fault types x 3 severities each
files = {
    'Normal':          ('data/Normal_1.mat', 0),
    'IR_007':          ('data/IR007_1.mat', 1),
    'IR_014':          ('data/IR014_1.mat', 2),
    'IR_021':          ('data/IR021_1.mat', 3),
    'Ball_007':        ('data/B007_1.mat', 4),
    'Ball_014':        ('data/B014_1.mat', 5),
    'Ball_021':        ('data/B021_1.mat', 6),
    'OuterRace_007':   ('data/OR007_1.mat', 7),
    'OuterRace_014':   ('data/OR014_1.mat', 8),
    'OuterRace_021':   ('data/OR021_1.mat', 9),
}

def create_windows(signal, window_size, stride):
    windows = []
    for start in range(0, len(signal) - window_size + 1, stride):
        windows.append(signal[start:start + window_size])
    return np.array(windows)

def normalize_window(window):
    mean = np.mean(window)
    std = np.std(window)
    return (window - mean) / (std + 1e-8)

X_train_list, y_train_list = [], []
X_test_list, y_test_list = [], []

for label_name, (filepath, label_idx) in files.items():
    data = scipy.io.loadmat(filepath)
    de_key = [key for key in data.keys() if 'DE_time' in key][0]
    signal = data[de_key].flatten()

    split_point = int(len(signal) * TRAIN_FRACTION)
    train_signal = signal[:split_point]
    test_signal = signal[split_point:]

    train_windows = create_windows(train_signal, WINDOW_SIZE, STRIDE)
    test_windows = create_windows(test_signal, WINDOW_SIZE, STRIDE)

    train_windows = np.array([normalize_window(w) for w in train_windows])
    test_windows = np.array([normalize_window(w) for w in test_windows])

    train_labels = np.full(len(train_windows), label_idx)
    test_labels = np.full(len(test_windows), label_idx)

    X_train_list.append(train_windows)
    y_train_list.append(train_labels)
    X_test_list.append(test_windows)
    y_test_list.append(test_labels)

    print(f"{label_name}: {len(train_windows)} train windows, {len(test_windows)} test windows")

X_train = np.concatenate(X_train_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)
X_test = np.concatenate(X_test_list, axis=0)
y_test = np.concatenate(y_test_list, axis=0)

rng = np.random.RandomState(42)
train_shuffle_idx = rng.permutation(len(X_train))
X_train = X_train[train_shuffle_idx]
y_train = y_train[train_shuffle_idx]

print(f"\nTotal train: X={X_train.shape}, y={y_train.shape}")
print(f"Total test: X={X_test.shape}, y={y_test.shape}")
print(f"Train class distribution: {np.bincount(y_train)}")
print(f"Test class distribution: {np.bincount(y_test)}")

np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
print("\nSaved: X_train.npy, X_test.npy, y_train.npy, y_test.npy")