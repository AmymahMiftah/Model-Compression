
#!/usr/bin/env python3
"""Structured pruning for a student model (ResNet18/34) + optional KD fine-tune.

This script:
- loads student checkpoint (state_dict)
- applies structured filter pruning to Conv2d layers in selected blocks
- optionally fine-tunes for a few epochs with KD (teacher) to recover accuracy
- writes summary JSON and pruned checkpoint

NOTE: This uses masking (dense tensors with zeros). It improves *effective compute* only if you
later export with compilers that exploit structure. It is still valid for accuracy/energy trade-off
analysis and as a precursor to quantization.

Example:
  python scripts/prune_student_structured.py \
    --student_ckpt outputs_kd_student/student_best.pt \
    --teacher_ckpt outputs/resnet50_cifar10_best.pt \
    --student resnet18 --img_size 128 \
    --prune_scope layer4 --prune_amount 0.3 \
    --finetune_epochs 3 --alpha 0.7 --temperature 4.0 \
    --output_dir outputs_pruned
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torchvision
import torch.nn.utils.prune as prune

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
        raise ValueError("Unsupported student: %s" % name)
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m

def iter_conv_modules_by_scope(model: nn.Module, scope: str):
    scope = scope.lower()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        # ResNet blocks are named layer1..layer4
        if scope == "layer4" and "layer4" not in name:
            continue
        if scope == "layer3_layer4" and not (("layer3" in name) or ("layer4" in name)):
            continue
        if scope == "all":
            pass
        yield name, module

def apply_structured_pruning(model: nn.Module, scope: str, amount: float):
    pruned = 0
    for name, conv in iter_conv_modules_by_scope(model, scope):
        # prune entire output channels by L_n norm (n=2)
        prune.ln_structured(conv, name="weight", amount=amount, n=2, dim=0)
        prune.remove(conv, "weight")  # make pruning permanent (still dense, but zeros)
        pruned += 1
    return pruned

def global_sparsity(model: nn.Module) -> float:
    total = 0
    zeros = 0
    for p in model.parameters():
        t = p.detach()
        total += t.numel()
        zeros += int((t == 0).sum().item())
    return zeros / max(total, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_ckpt", required=True)
    ap.add_argument("--teacher_ckpt", required=True)
    ap.add_argument("--student", default="resnet18", choices=["resnet18", "resnet34"])
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--prune_scope", default="layer4", choices=["layer4", "layer3_layer4", "all"])
    ap.add_argument("--prune_amount", type=float, default=0.3)
    ap.add_argument("--finetune_epochs", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--track_energy", action="store_true")
    ap.add_argument("--output_dir", default="outputs_pruned")
    args = ap.parse_args()

    device = torch.device("cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, test_loader = build_cifar10_loaders(args.img_size, args.batch_size, args.num_workers, train_aug=False)

    teacher = build_teacher(args.teacher_ckpt).to(device)
    student = build_student(args.student).to(device)
    sd = torch.load(args.student_ckpt, map_location="cpu")
    student.load_state_dict(sd, strict=True)

    # Pre-prune metrics
    base_acc = evaluate_top1(student, test_loader, device)

    pruned_layers = apply_structured_pruning(student, args.prune_scope, args.prune_amount)
    sparsity = global_sparsity(student)

    tracker = None
    if args.track_energy and EmissionsTracker is not None:
        tracker = EmissionsTracker(project_name="prune_student", output_dir=args.output_dir, log_level="error")
        tracker.start()

    t0 = time.time()
    history = []
    best_acc = -1.0
    best_path = os.path.join(args.output_dir, "student_pruned_best.pt")

    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    opt = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    summary = {
        "student_ckpt": args.student_ckpt,
        "teacher_ckpt": args.teacher_ckpt,
        "student": args.student,
        "img_size": args.img_size,
        "prune_scope": args.prune_scope,
        "prune_amount": args.prune_amount,
        "pruned_conv_layers": pruned_layers,
        "effective_global_sparsity": sparsity,
        "base_test_acc_before_prune": base_acc,
        "finetune_epochs": args.finetune_epochs,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "history": [],
    }
    summary_path = os.path.join(args.output_dir, "summary_pruned.json")
    atomic_write_json(summary, summary_path)

    # optional KD fine-tune to recover accuracy
    try:
        for epoch in range(1, args.finetune_epochs + 1):
            student.train()
            correct = total = 0
            loss_sum = 0.0
            iterator = train_loader
            if tqdm is not None:
                iterator = tqdm(train_loader, desc=f"FT {epoch}/{args.finetune_epochs}", leave=False)

            for x, y in iterator:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    tlog = teacher(x)
                slog = student(x)

                loss_ce = ce(slog, y)
                loss_kd = kd_kl_div_loss(tlog, slog, args.temperature)
                loss = (1.0 - args.alpha) * loss_ce + args.alpha * loss_kd

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                loss_sum += float(loss.item()) * x.size(0)
                preds = slog.argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            train_loss = loss_sum / max(total, 1)
            train_acc = 100.0 * correct / max(total, 1)
            test_acc = evaluate_top1(student, test_loader, device)
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(student.state_dict(), best_path)

            summary["history"].append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
            })
            summary["best_test_acc_so_far"] = best_acc
            summary["wall_time_sec_so_far"] = time.time() - t0
            atomic_write_json(summary, summary_path)
            print(f"FT Epoch {epoch}/{args.finetune_epochs} | loss={train_loss:.4f} | train_acc={train_acc:.2f}% | test_acc={test_acc:.2f}%")

    except KeyboardInterrupt:
        print("\nInterrupted. Keeping best checkpoint saved so far.")

    # Final metrics
    final_acc = evaluate_top1(student, test_loader, device)
    latency = benchmark_latency_ms_per_image(student, test_loader, device)
    state_mb = save_state_dict_mb(student.state_dict(), os.path.join(args.output_dir, "student_pruned_state_dict.pt"))
    final_path = os.path.join(args.output_dir, "student_pruned_final.pt")
    torch.save(student.state_dict(), final_path)

    summary.update({
        "test_top1_acc_final": final_acc,
        "latency_ms_per_image": latency,
        "model_state_dict_mb_final": state_mb,
        "params_total": count_params(student),
        "prune_wall_time_sec": time.time() - t0,
    })

    if tracker is not None:
        try:
            emissions = tracker.stop()
            summary["codecarbon_prune_finetune_emissions_kgco2"] = float(emissions) if emissions is not None else None
        except Exception:
            summary["codecarbon_prune_finetune_emissions_kgco2"] = None

    atomic_write_json(summary, summary_path)
    print(f"Saved summary: {summary_path}")
    print(f"Saved final: {final_path}")
    if args.finetune_epochs > 0:
        print(f"Saved best (during finetune): {best_path}")

if __name__ == "__main__":
    main()
