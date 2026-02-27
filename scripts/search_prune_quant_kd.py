
#!/usr/bin/env python3
"""Search pruning structures + quantization settings using KD fidelity as the main evaluator.

Workflow:
1) Start with a KD-trained student checkpoint (ResNet18/34).
2) For each candidate (scope x prune_amount x quant_mode):
   - Apply structured pruning mask
   - (Optional) fine-tune briefly with KD (0-2 epochs recommended for CPU)
   - Quantize (none or int8 PTQ)
   - Score:
       KD_loss vs teacher (lower better)
       test_acc (higher better)
       latency (lower better)
       size_mb (lower better)
3) Write a leaderboard CSV + JSON.

Example:
  python scripts/search_prune_quant_kd.py \
    --student_ckpt outputs_kd_student/student_best.pt \
    --teacher_ckpt outputs/resnet50_cifar10_best.pt \
    --student resnet18 --img_size 128 \
    --prune_scopes layer4 layer3_layer4 \
    --prune_amounts 0.1 0.2 0.3 0.4 \
    --quant_modes fp32 int8 \
    --kd_eval_batches 50 \
    --calib_batches 200 \
    --output_dir outputs_search
"""

import argparse
import copy
import csv
import os
import time
import torch
import torch.nn as nn
import torchvision
import torch.nn.utils.prune as prune

from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

from kd_common import (
    build_cifar10_loaders, evaluate_top1, benchmark_latency_ms_per_image,
    kd_fidelity_score, count_params, save_state_dict_mb, atomic_write_json
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
        if scope == "layer4" and "layer4" not in name:
            continue
        if scope == "layer3_layer4" and not (("layer3" in name) or ("layer4" in name)):
            continue
        if scope == "all":
            pass
        yield name, module

def apply_structured_pruning(model: nn.Module, scope: str, amount: float):
    for _, conv in iter_conv_modules_by_scope(model, scope):
        prune.ln_structured(conv, name="weight", amount=amount, n=2, dim=0)
        prune.remove(conv, "weight")

def global_sparsity(model: nn.Module) -> float:
    total = 0
    zeros = 0
    for p in model.parameters():
        t = p.detach()
        total += t.numel()
        zeros += int((t == 0).sum().item())
    return zeros / max(total, 1)

def quantize_int8_fx(model: nn.Module, img_size: int, calib_loader, calib_batches: int):
    torch.backends.quantized.engine = "fbgemm"
    qconfig_mapping = get_default_qconfig_mapping("fbgemm")
    example_inputs = (torch.randn(1, 3, img_size, img_size),)
    prepared = prepare_fx(model, qconfig_mapping, example_inputs)
    prepared.eval()
    with torch.no_grad():
        it = iter(calib_loader)
        for i in range(calib_batches):
            try:
                x, _ = next(it)
            except StopIteration:
                break
            prepared(x)
    int8_model = convert_fx(prepared).eval()
    return int8_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_ckpt", required=True)
    ap.add_argument("--teacher_ckpt", required=True)
    ap.add_argument("--student", default="resnet18", choices=["resnet18", "resnet34"])
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--prune_scopes", nargs="+", default=["layer4", "layer3_layer4"])
    ap.add_argument("--prune_amounts", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4])
    ap.add_argument("--quant_modes", nargs="+", default=["fp32", "int8"], choices=["fp32", "int8"])
    ap.add_argument("--kd_temperature", type=float, default=4.0)
    ap.add_argument("--kd_eval_batches", type=int, default=50)
    ap.add_argument("--calib_batches", type=int, default=200)
    ap.add_argument("--output_dir", default="outputs_search")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cpu")

    train_loader, test_loader = build_cifar10_loaders(args.img_size, args.batch_size, args.num_workers, train_aug=False)

    teacher = build_teacher(args.teacher_ckpt).to(device)

    base_student = build_student(args.student).to(device)
    sd = torch.load(args.student_ckpt, map_location="cpu")
    base_student.load_state_dict(sd, strict=True)
    base_student.eval()

    rows = []
    start = time.time()

    for scope in args.prune_scopes:
        for amt in args.prune_amounts:
            # build candidate from scratch to avoid cumulative pruning
            cand = copy.deepcopy(base_student).to(device).eval()
            apply_structured_pruning(cand, scope, amt)
            sparsity = global_sparsity(cand)

            for qmode in args.quant_modes:
                model_for_eval = cand
                quant_tag = "fp32"
                if qmode == "int8":
                    model_for_eval = quantize_int8_fx(cand, args.img_size, train_loader, args.calib_batches)
                    quant_tag = "int8_fx"

                kd_loss = kd_fidelity_score(model_for_eval, teacher, test_loader, device,
                                            temperature=args.kd_temperature, max_batches=args.kd_eval_batches)
                test_acc = evaluate_top1(model_for_eval, test_loader, device)
                latency = benchmark_latency_ms_per_image(model_for_eval, test_loader, device)

                # Save size of state_dict (quantized or fp32)
                ckpt_name = f"cand_{scope}_p{int(amt*100)}_{quant_tag}.pt"
                ckpt_path = os.path.join(args.output_dir, ckpt_name)
                size_mb = save_state_dict_mb(model_for_eval.state_dict(), ckpt_path)

                rows.append({
                    "scope": scope,
                    "prune_amount": amt,
                    "quant": quant_tag,
                    "kd_loss": kd_loss,
                    "test_acc": test_acc,
                    "latency_ms_per_image": latency,
                    "state_dict_mb": size_mb,
                    "effective_sparsity": sparsity,
                    "ckpt_path": ckpt_path,
                })
                print(f"[{scope} p={amt:.2f} {quant_tag}] kd_loss={kd_loss:.4f} acc={test_acc:.2f}% lat={latency:.2f}ms size={size_mb:.2f}MB sparsity={sparsity:.2%}")

    # sort by kd_loss then latency (you can change)
    rows_sorted = sorted(rows, key=lambda r: (r["kd_loss"], r["latency_ms_per_image"]))
    leaderboard_csv = os.path.join(args.output_dir, "leaderboard.csv")
    with open(leaderboard_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()) if rows_sorted else [])
        w.writeheader()
        w.writerows(rows_sorted)

    leaderboard_json = os.path.join(args.output_dir, "leaderboard.json")
    atomic_write_json({
        "student_ckpt": args.student_ckpt,
        "teacher_ckpt": args.teacher_ckpt,
        "student": args.student,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "kd_temperature": args.kd_temperature,
        "kd_eval_batches": args.kd_eval_batches,
        "calib_batches": args.calib_batches,
        "candidates": rows_sorted,
        "wall_time_sec": time.time() - start,
    }, leaderboard_json)

    print("Saved:", leaderboard_csv)
    print("Saved:", leaderboard_json)

if __name__ == "__main__":
    main()
