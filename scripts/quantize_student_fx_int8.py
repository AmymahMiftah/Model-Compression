
#!/usr/bin/env python3
"""Quantize a (possibly pruned) student model to INT8 using FX static PTQ and benchmark it.

Example:
  python scripts/quantize_student_fx_int8.py \
    --student_ckpt outputs_pruned/student_pruned_best.pt \
    --student resnet18 --img_size 128 \
    --calib_batches 200 --batch_size 64 --num_workers 2 \
    --output_dir outputs_int8
"""

import argparse
import os
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

from kd_common import (
    build_cifar10_loaders, evaluate_top1, benchmark_latency_ms_per_image,
    count_params, atomic_write_json, save_state_dict_mb
)

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_ckpt", required=True)
    ap.add_argument("--student", default="resnet18", choices=["resnet18", "resnet34"])
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--calib_batches", type=int, default=200)
    ap.add_argument("--output_dir", default="outputs_int8")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cpu")

    train_loader, test_loader = build_cifar10_loaders(args.img_size, args.batch_size, args.num_workers, train_aug=False)

    student = build_student(args.student).to(device)
    sd = torch.load(args.student_ckpt, map_location="cpu")
    student.load_state_dict(sd, strict=True)
    student.eval()

    fp32_acc = evaluate_top1(student, test_loader, device)
    fp32_latency = benchmark_latency_ms_per_image(student, test_loader, device)

    torch.backends.quantized.engine = "fbgemm"
    qconfig_mapping = get_default_qconfig_mapping("fbgemm")

    example_inputs = (torch.randn(1, 3, args.img_size, args.img_size),)
    prepared = prepare_fx(student, qconfig_mapping, example_inputs)
    prepared.eval()

    # calibration
    with torch.no_grad():
        it = iter(train_loader)
        for i in range(args.calib_batches):
            try:
                x, _ = next(it)
            except StopIteration:
                break
            prepared(x.to(device))
    int8_model = convert_fx(prepared).eval()

    int8_acc = evaluate_top1(int8_model, test_loader, device)
    int8_latency = benchmark_latency_ms_per_image(int8_model, test_loader, device)

    # Save int8 state_dict
    int8_path = Path(args.output_dir) / "student_int8_fx_state_dict.pt"
    int8_mb = save_state_dict_mb(int8_model.state_dict(), str(int8_path))

    summary = {
        "student_ckpt": args.student_ckpt,
        "student": args.student,
        "img_size": args.img_size,
        "quantization": "fx_static_int8_fbgemm",
        "calib_batches": args.calib_batches,
        "fp32_test_acc": fp32_acc,
        "fp32_latency_ms_per_image": fp32_latency,
        "int8_test_acc": int8_acc,
        "int8_latency_ms_per_image": int8_latency,
        "int8_state_dict_mb": int8_mb,
        "params_total": count_params(student),
    }
    summary_path = Path(args.output_dir) / "summary_int8.json"
    atomic_write_json(summary, str(summary_path))
    print("Saved:", int8_path)
    print("Saved:", summary_path)
    print(summary)

if __name__ == "__main__":
    main()
