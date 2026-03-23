# Activation Dynamics in Convolutional Neural Networks for Medical X-ray Classification

## 📌 Overview

This project investigates how activation scaling (β) affects training
dynamics, neuron utilization, and performance in CNNs for medical X-ray
classification using the MURA dataset.

The goal is not just improving accuracy, but understanding internal
behavior such as gradient flow, neuron inactivity, and training
stability.

------------------------------------------------------------------------

## 🚀 Key Contributions

-   β-controlled activation scaling (generalized ReLU)
-   Systematic ablation study (activation + initialization)
-   β scheduling (constant, linear, cosine, sinusoidal)
-   Neuron utilization analysis:
    -   Inactivity ratio
    -   Sparsity
    -   Gradient starvation
    -   Dead neurons
-   Gradient flow metrics:
    -   Depth Ratio
    -   Gradient Stability Index (GSI)

------------------------------------------------------------------------

## 🧠 Project Structure

Research Project/ │ ├── src/ │ ├── train.py │ ├── final_demo.py │ ├──
model.py │ ├── dataset.py │ ├── data/splits/ │ ├── train.txt │ ├──
val.txt │ ├── MURA-v1.1/ ├── phase_3_aggregated_allfiles.csv ├──
results/ └── README.md

------------------------------------------------------------------------

## 📥 Dataset Setup

Download MURA dataset:
https://cs.stanford.edu/group/mlgroup/MURA-v1.1.zip

Extract into project root:

MURA-v1.1/

------------------------------------------------------------------------

## ⚙️ Environment Setup

### Create Virtual Environment

python -m venv venv

### Activate

Windows: venv`\Scripts`{=tex}`\activate`{=tex}

Linux/Mac: source venv/bin/activate

### Install Dependencies

pip install torch torchvision numpy pandas matplotlib pillow tqdm

------------------------------------------------------------------------

## ▶️ Running the Project

### Phase 1 & 2

python -m src.train

### Phase 3

Uses aggregated CSV for neuron analysis.

### Phase 4

Also executed via src.train (β scheduling experiments)

------------------------------------------------------------------------

## 🏆 Final Demo

Run best configurations: python -m src.final_demo

Outputs: - Validation accuracy - Conv3 inactivity - Gradient
starvation - Dead neurons

------------------------------------------------------------------------

## 📊 Metrics

Depth Ratio = \|\|∇W_conv1\|\| / \|\|∇W_conv3\|\|

GSI = mean(grad norms) / std(grad norms)

Neuron metrics: - Inactivity ratio - Sparsity - Starvation ratio - Dead
neurons

------------------------------------------------------------------------

## 🧪 Experimental Phases

Phase 1: Baseline CNN (ReLU + He)

Phase 2: Ablation Study (Activation & Initialization)

Phase 3: Neuron Utilization Analysis

Phase 4: β Scheduling Study

------------------------------------------------------------------------

## 🏁 Key Findings

-   β = 0.1 (constant_01) gives best performance
-   Cosine schedule provides stable training
-   Linear improves neuron survival
-   ReLU baseline shows dead neurons and instability

------------------------------------------------------------------------

## 🖥️ Requirements

-   GPU recommended
-   8GB+ RAM
-   Dataset \~6GB

------------------------------------------------------------------------

## ❗ Troubleshooting

CUDA issues: pip install torch --index-url
https://download.pytorch.org/whl/cpu

Dataset issues: Ensure MURA-v1.1 is in root

------------------------------------------------------------------------

## 👨‍💻 Author

Yash Bhalerao\
VIT Pune

------------------------------------------------------------------------

## 📌 Future Work

-   Extend to ResNet
-   Adaptive β learning
-   Other medical datasets
