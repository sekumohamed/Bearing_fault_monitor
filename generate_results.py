import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from model import BearingFaultCNN

# ---- Load training history ----
with open('training_history.json', 'r') as f:
    results = json.load(f)

optimizer_names = list(results.keys())
class_names = ['Normal', 'IR_007', 'IR_014', 'IR_021', 'Ball_007', 'Ball_014', 'Ball_021', 'OR_007', 'OR_014', 'OR_021']

# =========================================================
# 1. FULL 50-EPOCH LOSS & ACCURACY CURVES
# =========================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for name in optimizer_names:
    epochs = range(1, len(results[name]['train_loss']) + 1)
    axes[0, 0].plot(epochs, results[name]['train_loss'], label=name)
    axes[0, 1].plot(epochs, results[name]['test_loss'], label=name)
    axes[1, 0].plot(epochs, results[name]['train_acc'], label=name)
    axes[1, 1].plot(epochs, results[name]['test_acc'], label=name)

axes[0, 0].set_title('Training Loss (Full 50 Epochs)')
axes[0, 1].set_title('Test Loss (Full 50 Epochs)')
axes[1, 0].set_title('Training Accuracy (Full 50 Epochs)')
axes[1, 1].set_title('Test Accuracy (Full 50 Epochs)')

for ax in axes.flat:
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

axes[0, 0].set_ylabel('Loss')
axes[0, 1].set_ylabel('Loss')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 1].set_ylabel('Accuracy')

plt.tight_layout()
plt.savefig('full_curves.png', dpi=150)
plt.close()
print("Saved: full_curves.png")

# =========================================================
# 2. ZOOMED EARLY-EPOCH CURVES (the real optimizer story)
# =========================================================
EARLY_EPOCHS = 15

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name in optimizer_names:
    epochs = range(1, EARLY_EPOCHS + 1)
    axes[0].plot(epochs, results[name]['train_loss'][:EARLY_EPOCHS], marker='o', label=name)
    axes[1].plot(epochs, results[name]['train_acc'][:EARLY_EPOCHS], marker='o', label=name)

axes[0].set_title(f'Training Loss (First {EARLY_EPOCHS} Epochs) — Convergence Speed Comparison')
axes[1].set_title(f'Training Accuracy (First {EARLY_EPOCHS} Epochs) — Convergence Speed Comparison')

for ax in axes:
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel('Loss')
axes[1].set_ylabel('Accuracy')

plt.tight_layout()
plt.savefig('early_epoch_convergence.png', dpi=150)
plt.close()
print("Saved: early_epoch_convergence.png")

# =========================================================
# 3. CONFUSION MATRICES (one per optimizer)
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

summary_rows = []

for idx, name in enumerate(optimizer_names):
    model = BearingFaultCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(f'model_{name}.pth'))
    model.eval()

    with torch.no_grad():
        outputs = model(X_test_t)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()

    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[idx])
    axes[idx].set_title(f'{name}')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

    # ---- Compute epochs-to-95%-train-accuracy ----
    train_acc_list = results[name]['train_acc']
    epoch_to_95 = next((i + 1 for i, acc in enumerate(train_acc_list) if acc >= 0.95), None)

    final_test_acc = results[name]['test_acc'][-1]
    final_test_loss = results[name]['test_loss'][-1]

    summary_rows.append({
        'Optimizer': name,
        'Final Test Accuracy': f"{final_test_acc*100:.2f}%",
        'Final Test Loss': f"{final_test_loss:.4f}",
        'Epochs to reach 95% Train Acc': epoch_to_95 if epoch_to_95 else "Not reached"
    })

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
plt.close()
print("Saved: confusion_matrices.png")

# =========================================================
# 4. SUMMARY COMPARISON TABLE
# =========================================================
print("\n" + "="*70)
print("OPTIMIZER COMPARISON SUMMARY")
print("="*70)
print(f"{'Optimizer':<15} {'Final Test Acc':<18} {'Final Test Loss':<18} {'Epochs to 95% Train Acc'}")
print("-"*70)
for row in summary_rows:
    print(f"{row['Optimizer']:<15} {row['Final Test Accuracy']:<18} {row['Final Test Loss']:<18} {row['Epochs to reach 95% Train Acc']}")

# Save as a simple text/csv file too
with open('comparison_summary.csv', 'w') as f:
    f.write("Optimizer,Final Test Accuracy,Final Test Loss,Epochs to 95% Train Acc\n")
    for row in summary_rows:
        f.write(f"{row['Optimizer']},{row['Final Test Accuracy']},{row['Final Test Loss']},{row['Epochs to reach 95% Train Acc']}\n")

print("\nSaved: comparison_summary.csv")

# =========================================================
# 5. CLASSIFICATION REPORTS (precision/recall/F1 per class)
# =========================================================
print("\n" + "="*70)
print("PER-CLASS CLASSIFICATION REPORTS")
print("="*70)

for name in optimizer_names:
    model = BearingFaultCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(f'model_{name}.pth'))
    model.eval()

    with torch.no_grad():
        outputs = model(X_test_t)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()

    print(f"\n--- {name} ---")
    print(classification_report(y_test, preds, target_names=class_names, digits=4))