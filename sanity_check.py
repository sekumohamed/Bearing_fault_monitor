import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import BearingFaultCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# ---- Shuffle the TRAINING labels randomly (breaks any real signal-label relationship) ----
rng = np.random.RandomState(0)
y_train_shuffled = y_train.copy()
rng.shuffle(y_train_shuffled)

X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_t = torch.tensor(y_train_shuffled, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

torch.manual_seed(42)
model = BearingFaultCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

print("Training on SHUFFLED (nonsense) labels — expect accuracy to stay near random (25%)...")
for epoch in range(1, 21):
    model.train()
    correct, total = 0, 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Train Acc on shuffled labels = {correct/total:.4f}")

# Evaluate on REAL test labels (should be near-random too, since model learned nonsense)
model.eval()
with torch.no_grad():
    outputs = model(X_test_t)
    _, preds = torch.max(outputs, 1)
    test_acc = (preds == y_test_t).sum().item() / len(y_test_t)

print(f"\nTest accuracy (trained on shuffled labels, evaluated on real test labels): {test_acc:.4f}")
print("If this is near 0.25 (25%), your pipeline is clean — no leakage.")
print("If this is anywhere near 0.90+, there IS a leak/bug we need to find.")