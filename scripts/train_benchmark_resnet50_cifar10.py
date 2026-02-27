import os
import time
import json
import argparse
from pathlib import Path

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


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_model_and_get_size_mb(model: nn.Module, path: str) -> float:
    torch.save(model.state_dict(), path)
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
                                   warmup_batches=10, max_batches=50) -> float:
    model.eval()
    it = iter(loader)

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


def set_finetune_mode(model: nn.Module, mode: str):
    """Set which parameters require gradients."""
    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        return

    # Always train fc for non-full modes
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    if mode == "layer4":
        for p in model.layer4.parameters():
            p.requires_grad = True
    elif mode == "head":
        pass
    else:
        raise ValueError(f"Unknown finetune mode: {mode}")


def build_optimizer(model: nn.Module, optimizer_name: str, lr: float, backbone_lr: float,
                    weight_decay: float):
    """Build optimizer with param groups for better fine-tuning stability."""
    head_params = list(model.fc.parameters())
    backbone_params = [p for n, p in model.named_parameters()
                       if (not n.startswith("fc.")) and p.requires_grad]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    param_groups.append({"params": head_params, "lr": lr})

    if optimizer_name.lower() == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    if optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    raise ValueError("optimizer must be 'sgd' or 'adamw'")


def build_scheduler(optimizer, scheduler_name: str, epochs: int, steps_per_epoch: int, warmup_epochs: int):
    if scheduler_name == "none":
        return None

    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        # cosine decay
        return float(0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(model, loader, device, criterion, optimizer, scheduler, epoch: int, epochs: int,
                    log_interval: int):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    if tqdm is not None:
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        iterator = pbar
    else:
        pbar = None
        iterator = loader

    for step, (x, y) in enumerate(iterator, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

        if pbar is not None:
            pbar.set_postfix(loss=float(loss.item()), acc=float(100.0 * correct / max(total, 1)))
        elif log_interval > 0 and step % log_interval == 0:
            print(f"  step {step}/{len(loader)} | loss={loss.item():.4f} | acc={100.0 * correct / max(total, 1):.2f}%")

    return running_loss / max(total, 1), 100.0 * correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()

    # Benchmarks + training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--pretrained", action="store_true")

    # Accuracy knobs (CPU deployable at inference time)
    parser.add_argument("--img_size", type=int, default=96,
                        help="96/128 improves accuracy vs 64 while still CPU-deployable; 224 is slowest.")
    parser.add_argument("--finetune", type=str, default="staged", choices=["head", "layer4", "full", "staged"])
    parser.add_argument("--head_warmup_epochs", type=int, default=2)
    parser.add_argument("--layer4_epochs", type=int, default=6)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--lr", type=float, default=0.1, help="Head LR")
    parser.add_argument("--backbone_lr", type=float, default=0.01, help="Backbone LR")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"])
    parser.add_argument("--warmup_epochs", type=int, default=1)

    # Augmentation
    parser.add_argument("--randaugment", action="store_true",
                        help="Enable RandAugment for accuracy (small extra CPU cost).")

    # Logging / tracking
    parser.add_argument("--log_interval", type=int, default=0)
    parser.add_argument("--track_energy", action="store_true",
                        help="Track estimated energy/CO2 using CodeCarbon (if installed).")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resize_side = int(args.img_size * 256 / 224)
    aug_list = [T.RandAugment()] if args.randaugment else []

    train_tf = T.Compose([
        T.Resize(resize_side),
        T.RandomResizedCrop(args.img_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        *aug_list,
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_tf = T.Compose([
        T.Resize(resize_side),
        T.CenterCrop(args.img_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = torchvision.datasets.CIFAR10(root=str(Path("data")), train=True, download=True, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR10(root=str(Path("data")), train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    if args.pretrained:
        model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
    else:
        model = torchvision.models.resnet50(weights=None)

    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    total_params = count_params(model)
    tmp_path = output_dir / "tmp.pt"
    size_mb_before = save_model_and_get_size_mb(model, str(tmp_path))
    tmp_path.unlink(missing_ok=True)

    tracker = None
    if args.track_energy:
        if EmissionsTracker is None:
            print("Warning: codecarbon not installed; skipping energy tracking.")
        else:
            tracker = EmissionsTracker(
                project_name="resnet50_cifar10_accuracy",
                gpu_ids=[] if device.type == "cpu" else None,
                measure_power_secs=5,
                log_level="error",
                output_dir=str(output_dir),
            )
            tracker.start()

    history = []
    best_acc = -1.0
    best_path = output_dir / "resnet50_cifar10_best.pt"

    train_wall_start = time.time()
    epoch_times = []
    steps_per_epoch = len(train_loader)

    def current_stage(ep):
        if args.finetune != "staged":
            return args.finetune
        if ep <= args.head_warmup_epochs:
            return "head"
        if ep <= args.head_warmup_epochs + args.layer4_epochs:
            return "layer4"
        return "full"

    optimizer = None
    scheduler = None

    for epoch in range(1, args.epochs + 1):
        stage = current_stage(epoch)
        set_finetune_mode(model, stage)

        # Rebuild optimizer when stage changes
        if optimizer is None or (len(history) > 0 and history[-1]["stage"] != stage):
            optimizer = build_optimizer(
                model,
                optimizer_name=args.optimizer,
                lr=args.lr,
                backbone_lr=args.backbone_lr,
                weight_decay=args.weight_decay,
            )
            if args.scheduler == "cosine":
                scheduler = build_scheduler(
                    optimizer,
                    scheduler_name="cosine",
                    epochs=args.epochs,
                    steps_per_epoch=steps_per_epoch,
                    warmup_epochs=args.warmup_epochs,
                )
            else:
                scheduler = None

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, device, criterion, optimizer, scheduler,
            epoch=epoch, epochs=args.epochs, log_interval=args.log_interval
        )
        test_acc = evaluate_top1(model, test_loader, device)

        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)
        avg_epoch = sum(epoch_times) / len(epoch_times)
        eta_sec = avg_epoch * (args.epochs - epoch)

        row = {
            "epoch": epoch,
            "stage": stage,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "test_acc": float(test_acc),
            "lr_head": float(optimizer.param_groups[-1]["lr"]),
            "lr_backbone": float(optimizer.param_groups[0]["lr"]) if len(optimizer.param_groups) > 1 else 0.0,
            "epoch_time_sec": float(epoch_time),
            "eta_sec_after_epoch": float(eta_sec),
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{args.epochs} [{stage}] | loss={train_loss:.4f} | "
            f"train_acc={train_acc:.2f}% | test_acc={test_acc:.2f}% | "
            f"epoch_time={epoch_time/60:.1f}m | ETA={eta_sec/60:.1f}m"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)

    train_wall_time_sec = time.time() - train_wall_start

    emissions_kg = None
    if tracker is not None:
        emissions_kg = tracker.stop()

    model.load_state_dict(torch.load(best_path, map_location=device))
    final_test_acc = evaluate_top1(model, test_loader, device)

    final_path = output_dir / "resnet50_cifar10_final.pt"
    final_size_mb = save_model_and_get_size_mb(model, str(final_path))

    ms_per_image = benchmark_latency_ms_per_image(model, test_loader, device, warmup_batches=10, max_batches=50)

    summary = {
        "model": "resnet50",
        "dataset": f"CIFAR-10 (resized to {args.img_size})",
        "pretrained": bool(args.pretrained),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "finetune_plan": str(args.finetune),
        "head_warmup_epochs": int(args.head_warmup_epochs),
        "layer4_epochs": int(args.layer4_epochs),
        "optimizer": str(args.optimizer),
        "lr_head": float(args.lr),
        "lr_backbone": float(args.backbone_lr),
        "weight_decay": float(args.weight_decay),
        "label_smoothing": float(args.label_smoothing),
        "scheduler": str(args.scheduler),
        "warmup_epochs": int(args.warmup_epochs),
        "randaugment": bool(args.randaugment),
        "num_workers": int(args.num_workers),
        "params_total": int(total_params),
        "model_state_dict_mb_before": float(size_mb_before),
        "model_state_dict_mb_final": float(final_size_mb),
        "test_top1_acc": float(final_test_acc),
        "latency_ms_per_image": float(ms_per_image),
        "training_wall_time_sec": float(train_wall_time_sec),
        "epoch_times_sec": [float(x) for x in epoch_times],
        "history": history,
    }
    if emissions_kg is not None:
        summary["codecarbon_training_emissions_kgco2"] = float(emissions_kg)

    out_json = output_dir / "summary_resnet50_cifar10.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()