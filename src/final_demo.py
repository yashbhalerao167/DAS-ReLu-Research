import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from src.dataset import MURADataset
from src.model import SimpleCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3


# -------------------------
# Beta Schedules
# -------------------------

def constant_01(epoch):
    return 0.1

def cosine_schedule(epoch):
    return 0.1 + 0.05 * np.cos(epoch / 5)

def linear_schedule(epoch):
    return 0.02 + (0.18 * (epoch / EPOCHS))


# -------------------------
# Metrics
# -------------------------

def compute_conv3_metrics(model):

    activations = model.get_last_conv3_activation().detach().cpu()

    inactivity = (activations == 0).float().mean().item()

    grads = model.conv3.weight.grad
    if grads is None:
        starvation = 0.0
    else:
        grads = grads.detach().cpu()
        grad_norms = grads.view(grads.size(0), -1).norm(dim=1)
        starvation = (grad_norms < 1e-6).float().mean().item()

    dead = (activations.view(activations.size(1), -1).sum(dim=1) == 0).sum().item()

    return inactivity, starvation, dead


# -------------------------
# Single Experiment
# -------------------------

def run_experiment(name, beta_schedule):

    print("\n============================================")
    print(f"Running Configuration: {name}")
    print("============================================")

    project_root = Path(__file__).resolve().parent.parent
    image_root = project_root / "MURA-v1.1"
    split_root = project_root / "data" / "splits"

    train_dataset = MURADataset(image_root, split_root / "train.txt")
    val_dataset = MURADataset(image_root, split_root / "val.txt")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = SimpleCNN(init_scale=1.0, beta=0.1).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):

        model.train()
        model.beta = beta_schedule(epoch)

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Final Validation Accuracy
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds.squeeze() == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total

    # Get metrics using one batch
    model.train()
    images, labels = next(iter(train_loader))
    images = images.to(DEVICE)
    labels = labels.float().unsqueeze(1).to(DEVICE)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()

    inactivity, starvation, dead = compute_conv3_metrics(model)

    print(f"\nFinal Validation Accuracy: {val_acc:.4f}")
    print(f"Conv3 Inactivity Ratio: {inactivity:.4f}")
    print(f"Conv3 Starvation Ratio: {starvation:.4f}")
    print(f"Conv3 Dead Neurons: {dead}")

    return val_acc


# -------------------------
# Main
# -------------------------

def main():

    print("Device:", DEVICE)

    results = {}

    results["constant_01"] = run_experiment("constant_01 (β=0.1)", constant_01)
    results["cosine"] = run_experiment("cosine schedule", cosine_schedule)
    results["linear"] = run_experiment("linear schedule", linear_schedule)

    print("\n============================================")
    print("FINAL SUMMARY")
    print("============================================")

    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()