
#!/usr/bin/env python3
"""Train a student model with Knowledge Distillation on CIFAR-10 (CPU-friendly).

- Teacher: ResNet50 checkpoint (state_dict) trained on CIFAR-10.
- Student: ResNet18 (default) or ResNet34.

Outputs (in --output_dir):
- student_best.pt (state_dict)
- summary_student_kd.json (updated after every epoch, crash-safe)
- emissions.csv (if --track_energy and codecarbon installed)

Example:
  python scripts/train_student_kd_cifar10.py \
    --teacher_ckpt outputs/resnet50_cifar10_best.pt \
    --student resnet18 --img_size 128 --epochs 20 \
    --batch_size 64 --num_workers 2 \
    --alpha 0.7 --temperature 4.0 \
    --lr 0.05 --weight_decay 1e-4 \
    --track_energy --output_dir outputs_kd_student
"""

import argparse
import math
import os
import time
import torch
import torch.nn as nn
import torchvision

from kd_common import (
    EmissionsTracker, tqdm,
    build_cifar10_loaders, evaluate_top1, benchmark_latency_ms_per_image,
    kd_kl_div_loss, count_params, save_state_dict_mb, atomic_write_json
)

def build_teacher(ckpt_path: str) -> nn.Module:
    teacher = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    teacher.fc = nn.Linear(teacher.fc.in_features, 10)
    sd = torch.load(ckpt_path, map_location="cpu")
    teacher.load_state_dict(sd, strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher

def build_student(name: str) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    elif name == "resnet34":
        m = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
    else:
        raise ValueError("Unsupported student: %s (use resnet18 or resnet34)" % name)
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m

def cosine_lr(base_lr: float, epoch: int, total_epochs: int) -> float:
    # cosine to 0
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / max(total_epochs, 1)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_ckpt", required=True)
    ap.add_argument("--student", default="resnet18", choices=["resnet18", "resnet34"])
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.7, help="weight of distillation term")
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--train_aug", action="store_true", help="enable light augmentation")
    ap.add_argument("--track_energy", action="store_true")
    ap.add_argument("--output_dir", type=str, default="outputs_kd_student")
    args = ap.parse_args()

    device = torch.device("cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, test_loader = build_cifar10_loaders(
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_aug=args.train_aug
    )

    teacher = build_teacher(args.teacher_ckpt).to(device)
    student = build_student(args.student).to(device)

    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    opt = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # tracker
    tracker = None
    if args.track_energy and EmissionsTracker is not None:
        tracker = EmissionsTracker(project_name="kd_student_train", output_dir=args.output_dir, log_level="error")
        tracker.start()

    best_acc = -1.0
    t_start = time.time()
    epoch_times = []

    summary_path = os.path.join(args.output_dir, "summary_student_kd.json")
    best_path = os.path.join(args.output_dir, "student_best.pt")
    final_path = os.path.join(args.output_dir, "student_final.pt")

    base = {
        "teacher_ckpt": args.teacher_ckpt,
        "student": args.student,
        "dataset": f"CIFAR-10 (resized to {args.img_size})",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": "cpu",
        "alpha": args.alpha,
        "temperature": args.temperature,
        "lr": args.lr,
        "label_smoothing": args.label_smoothing,
        "params_total": count_params(student),
        "history": [],
    }
    atomic_write_json(base, summary_path)

    try:
        for epoch in range(1, args.epochs + 1):
            ep0 = time.time()
            student.train()
            correct = total = 0
            running_loss = 0.0

            # cosine lr
            lr_now = cosine_lr(args.lr, epoch - 1, args.epochs)
            for pg in opt.param_groups:
                pg["lr"] = lr_now

            iterator = train_loader
            if tqdm is not None:
                iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

            for x, y in iterator:
                x = x.to(device)
                y = y.to(device)

                with torch.no_grad():
                    tlog = teacher(x)

                slog = student(x)

                loss_ce = ce(slog, y)
                loss_kd = kd_kl_div_loss(tlog, slog, args.temperature)
                loss = (1.0 - args.alpha) * loss_ce + args.alpha * loss_kd

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                running_loss += float(loss.item()) * x.size(0)
                preds = slog.argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            train_loss = running_loss / max(total, 1)
            train_acc = 100.0 * correct / max(total, 1)

            test_acc = evaluate_top1(student, test_loader, device)
            ep_time = time.time() - ep0
            epoch_times.append(ep_time)

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(student.state_dict(), best_path)

            # update summary each epoch (crash-safe)
            base["history"].append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "lr": lr_now,
                "epoch_time_sec": ep_time,
            })
            base["best_test_acc_so_far"] = best_acc
            base["epoch_times_sec"] = epoch_times
            base["training_wall_time_sec_so_far"] = time.time() - t_start
            atomic_write_json(base, summary_path)

            print(f"Epoch {epoch}/{args.epochs} | loss={train_loss:.4f} | train_acc={train_acc:.2f}% | test_acc={test_acc:.2f}% | epoch_time={ep_time/60:.1f}m")

    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl+C). Keeping best checkpoint saved so far.")

    # Save final
    torch.save(student.state_dict(), final_path)

    # Final benchmark
    # Use test_loader for latency (consistent)
    latency = benchmark_latency_ms_per_image(student, test_loader, device)
    final_mb = save_state_dict_mb(student.state_dict(), os.path.join(args.output_dir, "student_state_dict.pt"))

    base["test_top1_acc_final"] = evaluate_top1(student, test_loader, device)
    base["test_top1_acc_best"] = best_acc
    base["latency_ms_per_image"] = latency
    base["model_state_dict_mb_final"] = final_mb
    base["training_wall_time_sec"] = time.time() - t_start

    if tracker is not None:
        try:
            emissions = tracker.stop()
            base["codecarbon_training_emissions_kgco2"] = float(emissions) if emissions is not None else None
        except Exception:
            base["codecarbon_training_emissions_kgco2"] = None

    atomic_write_json(base, summary_path)
    print(f"Saved best: {best_path}")
    print(f"Saved summary: {summary_path}")

if __name__ == "__main__":
    main()
