# src/metrics.py

import numpy as np


def compute_depth_ratio(model):
    grads = []
    for name, param in model.named_parameters():
        if param.grad is not None and "weight" in name:
            grads.append(param.grad.norm().item())

    if len(grads) < 2:
        return 0.0

    return grads[0] / (grads[-1] + 1e-8)


def compute_gsi(model):
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.norm().item())

    if len(grads) == 0:
        return 0.0

    grads = np.array(grads)
    return grads.mean() / (grads.std() + 1e-8)