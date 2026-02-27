
import os
import time
import json
from pathlib import Path
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from codecarbon import EmissionsTracker
except Exception:
    EmissionsTracker = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def save_state_dict_mb(state_dict: Dict, path: str) -> float:
    torch.save(state_dict, path)
    return os.path.getsize(path) / (1024 * 1024)

@torch.no_grad()
def evaluate_top1(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(total, 1)

@torch.no_grad()
def benchmark_latency_ms_per_image(model: nn.Module, loader: DataLoader, device: torch.device,
                                   warmup_batches: int = 10, max_batches: int = 50) -> float:
    model.eval()
    it = iter(loader)
    # warmup
    for _ in range(warmup_batches):
        try:
            x, _ = next(it)
        except StopIteration:
            break
        x = x.to(device, non_blocking=True)
        _ = model(x)
    total_images = 0
    t0 = time.time()
    for _ in range(max_batches):
        try:
            x, _ = next(it)
        except StopIteration:
            break
        x = x.to(device, non_blocking=True)
        _ = model(x)
        total_images += x.size(0)
    t1 = time.time()
    return (t1 - t0) * 1000.0 / max(total_images, 1)

def build_cifar10_loaders(img_size: int, batch_size: int, num_workers: int,
                          data_root: str = "data",
                          train_aug: bool = False) -> Tuple[DataLoader, DataLoader]:
    if train_aug:
        # Light CPU-friendly augmentation (RandAugment is expensive on CPU)
        train_tf = T.Compose([
            T.Resize(img_size),
            T.RandomHorizontalFlip(),
            T.RandomCrop(img_size, padding=4),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        train_tf = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    test_tf = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_ds = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return train_loader, test_loader

@torch.no_grad()
def kd_kl_div_loss(teacher_logits: torch.Tensor, student_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    # KL(soft(T)||soft(S)) with temperature scaling
    t = temperature
    p_t = torch.softmax(teacher_logits / t, dim=1)
    log_p_s = torch.log_softmax(student_logits / t, dim=1)
    return torch.nn.functional.kl_div(log_p_s, p_t, reduction="batchmean") * (t * t)

@torch.no_grad()
def kd_fidelity_score(model: nn.Module, teacher: nn.Module, loader: DataLoader,
                      device: torch.device, temperature: float = 4.0,
                      max_batches: int = 50) -> float:
    """Average KL divergence from teacher to model on a subset of batches (lower is better)."""
    model.eval(); teacher.eval()
    total = 0.0
    n = 0
    it = iter(loader)
    for _ in range(max_batches):
        try:
            x, _ = next(it)
        except StopIteration:
            break
        x = x.to(device, non_blocking=True)
        tlog = teacher(x)
        slog = model(x)
        loss = kd_kl_div_loss(tlog, slog, temperature)
        total += float(loss.item())
        n += 1
    return total / max(n, 1)

def atomic_write_json(obj: Dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
