import os
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from codecarbon import EmissionsTracker

from data.dataset_loader import get_cifar10_loaders
from models.build_model import get_resnet18_cifar10
from models.utils import evaluate_accuracy, model_size_mb, latency_ms_per_image

# ✅ CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = ""
DEVICE = "cpu"

# Experiment config
EPOCHS = 2
BATCH_SIZE = 8          # ✅ smaller batch to avoid swap / slowdowns
NUM_WORKERS = 0         # ✅ stable on CPU
LR = 1e-4

def train():
    os.makedirs("./outputs", exist_ok=True)

    # Data
    train_loader, test_loader = get_cifar10_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Model (ResNet18 pretrained on ImageNet, head replaced for CIFAR-10)
    model = get_resnet18_cifar10(device=DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ✅ CodeCarbon tracker (CPU-only)
    tracker = EmissionsTracker(
        project_name="ResNet18_CPU_CIFAR10",
        gpu_ids=[],                 # Disable GPU tracking
        measure_power_secs=5,
        log_level="error"
    )

    tracker.start()

    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        start_time = time.time()
        running_loss = 0.0

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # ✅ progress print so it never looks "stuck"
            if step % 50 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Step {step}/{len(train_loader)} | Loss {loss.item():.4f}")

        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{EPOCHS}] Avg Loss: {running_loss/max(len(train_loader),1):.4f} | Time: {epoch_time:.2f}s")

    # Stop tracker and get emissions (kg CO2)
    training_emissions = tracker.stop()
    print("Training emissions (kg CO₂):", training_emissions)

    # Evaluate accuracy
    baseline_acc = evaluate_accuracy(model, test_loader, DEVICE)
    print(f"Baseline Top-1 Accuracy (%): {baseline_acc:.2f}")

    # Model size + latency
    size_mb = model_size_mb(model)
    latency = latency_ms_per_image(model, test_loader, DEVICE)

    print(f"Model size (MB): {size_mb:.2f}")
    print(f"Latency (ms/image): {latency:.2f}")

    # Save model
    model_path = "./outputs/resnet18_cifar10_cpu.pt"
    torch.save(model.state_dict(), model_path)
    print("Saved model:", model_path)

    # Save summary JSON
    summary = {
        "model": "resnet18",
        "dataset": "CIFAR-10 (resized to 224, ImageNet normalized)",
        "device": DEVICE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "lr": LR,
        "accuracy_top1_percent": baseline_acc,
        "model_size_mb": size_mb,
        "latency_ms_per_image": latency,
        "codecarbon_training_emissions_kgco2": training_emissions
    }

    summary_path = "./outputs/summary_resnet18_cpu.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved summary:", summary_path)

    return model

if __name__ == "__main__":
    train()
