import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import math
from torch.utils.data import DataLoader
from pathlib import Path
import random

from src.dataset import MURADataset
from src.model import SimpleCNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
SCALE = 1.0
STARVATION_THRESHOLD = 1e-6
DEAD_EPOCH_THRESHOLD = 3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------
# Beta Schedules
# -----------------------

def constant_0(epoch):
    return 0.0


def constant_01(epoch):
    return 0.1


def sinusoidal(epoch):
    return 0.1 + 0.05 * math.sin(epoch / 10)


def linear(epoch):
    return min(0.02 * epoch, 0.2)


def cosine(epoch):
    return 0.1 + 0.05 * math.cos(epoch / 10)


SCHEDULES = {
    "constant_0": constant_0,
    "constant_01": constant_01,
    "sinusoidal": sinusoidal,
    "linear": linear,
    "cosine": cosine
}


def run_phase3(schedule_name, schedule_fn, seed):

    print("\n==============================================")
    print(f"PHASE 3 | Schedule: {schedule_name} | Seed: {seed}")
    print("==============================================")

    set_seed(seed)

    project_root = Path(__file__).resolve().parent.parent
    image_root = project_root / "MURA-v1.1"
    split_root = project_root / "data" / "splits"

    train_dataset = MURADataset(image_root, split_root / "train.txt")
    val_dataset = MURADataset(image_root, split_root / "val.txt")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = SimpleCNN(init_scale=SCALE, beta=0.0).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    dead_counters = {
        "conv1": torch.zeros(16),
        "conv2": torch.zeros(32),
        "conv3": torch.zeros(64)
    }

    logs = []

    for epoch in range(1, EPOCHS + 1):

        beta_value = schedule_fn(epoch)
        model.set_beta(beta_value)

        # ------------------ TRAIN ------------------
        model.train()
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # ------------------ VALIDATION ANALYSIS ------------------
        model.eval()

        layer_stats = {}

        with torch.no_grad():
            for images, labels in val_loader:

                images = images.to(DEVICE)
                _ = model(images)

                for layer_name, activation in model.activations.items():

                    act = activation.cpu()
                    B, C, H, W = act.shape

                    if layer_name not in layer_stats:
                        layer_stats[layer_name] = {
                            "inactive": torch.zeros(C),
                            "sparsity_sum": torch.zeros(C),
                            "activation_sum": torch.zeros(C),
                            "batches": 0
                        }

                    for c in range(C):

                        neuron_map = act[:, c, :, :]

                        if torch.max(neuron_map) <= 0:
                            layer_stats[layer_name]["inactive"][c] += 1

                        zero_count = (neuron_map == 0).sum().item()
                        total_count = neuron_map.numel()

                        layer_stats[layer_name]["sparsity_sum"][c] += zero_count / total_count
                        layer_stats[layer_name]["activation_sum"][c] += torch.mean(torch.abs(neuron_map)).item()

                    layer_stats[layer_name]["batches"] += 1

        # ------------------ GRADIENT STARVATION ------------------

        grad_stats = {}

        for name, param in model.named_parameters():

            if "conv" in name and "weight" in name and param.grad is not None:

                layer_name = name.split(".")[0]

                grad = param.grad.detach().cpu()
                grad = grad.view(grad.size(0), -1)
                grad_norm = torch.norm(grad, dim=1)

                grad_stats[layer_name] = grad_norm

        # ------------------ AGGREGATE ------------------

        for layer_name in layer_stats:

            C = len(layer_stats[layer_name]["inactive"])
            batches = layer_stats[layer_name]["batches"]

            inactivity_ratio = layer_stats[layer_name]["inactive"].sum().item() / (C * batches)

            sparsity = layer_stats[layer_name]["sparsity_sum"].mean().item() / batches

            mean_activation = layer_stats[layer_name]["activation_sum"].mean().item() / batches

            grad_norms = grad_stats[layer_name]
            starvation_ratio = (grad_norms < STARVATION_THRESHOLD).sum().item() / len(grad_norms)

            # Dead neuron tracking
            dead_mask = layer_stats[layer_name]["inactive"] == batches
            dead_counters[layer_name] += dead_mask

            dead_neurons = (dead_counters[layer_name] >= DEAD_EPOCH_THRESHOLD).sum().item()

            logs.append({
                "epoch": epoch,
                "schedule": schedule_name,
                "seed": seed,
                "layer": layer_name,
                "beta": beta_value,
                "inactivity_ratio": inactivity_ratio,
                "sparsity": sparsity,
                "mean_activation": mean_activation,
                "starvation_ratio": starvation_ratio,
                "dead_neurons": dead_neurons
            })

        print(f"Epoch {epoch} completed.")

    df = pd.DataFrame(logs)
    save_path = project_root / f"phase3_{schedule_name}_seed_{seed}.csv"
    df.to_csv(save_path, index=False)

    print(f"Saved: {save_path}")


def main():

    print("Running Phase 3 Neuron Utilization Study")

    seeds = [42, 123, 999]

    for schedule_name, schedule_fn in SCHEDULES.items():
        for seed in seeds:
            run_phase3(schedule_name, schedule_fn, seed)

    print("\nPhase 3 completed successfully.")


if __name__ == "__main__":
    main()