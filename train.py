import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import json

from model import BearingFaultCNN

# ---- Config ----
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MOMENTUM = 0.9
EPOCHS = 50
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- Load preprocessed data ----
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Convert to PyTorch tensors, add channel dimension: (N, 1024) -> (N, 1, 1024)
X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


def train_one_optimizer(optimizer_name, momentum, nesterov):
    """Train a fresh model with a specific optimizer config, return history dict."""

    # Fix seed so all 3 runs start from identical weight initialization
    torch.manual_seed(SEED)
    model = BearingFaultCNN(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=momentum,
        nesterov=nesterov
    )

    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }

    print(f"\n{'='*50}")
    print(f"Training with optimizer: {optimizer_name}")
    print(f"{'='*50}")

    for epoch in range(1, EPOCHS + 1):
        # ---- Training ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ---- Evaluation on test set ----
        model.eval()
        test_running_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                test_running_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == batch_y).sum().item()
                test_total += batch_y.size(0)

        test_loss = test_running_loss / test_total
        test_acc = test_correct / test_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

    return history, model


# ---- Run all 3 optimizer experiments ----
results = {}

configs = [
    ("Standard_SGD", 0.0, False),
    ("Momentum_SGD", 0.9, False),
    ("Nesterov_SGD", 0.9, True),
]

trained_models = {}

for name, momentum, nesterov in configs:
    history, model = train_one_optimizer(name, momentum, nesterov)
    results[name] = history
    trained_models[name] = model
    # Save each trained model's weights
    torch.save(model.state_dict(), f'model_{name}.pth')

# ---- Save all training histories for Phase 5 plotting ----
with open('training_history.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n\nAll 3 optimizers trained. Results saved to training_history.json")
print("Model weights saved as model_Standard_SGD.pth, model_Momentum_SGD.pth, model_Nesterov_SGD.pth")