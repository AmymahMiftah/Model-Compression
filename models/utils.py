import os
import time
import torch

def evaluate_accuracy(model, dataloader, device="cpu"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / max(total, 1)  # ✅ percentage


def model_size_mb(model, path="./outputs/tmp_model.pt"):
    os.makedirs("./outputs", exist_ok=True)
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    os.remove(path)
    return size_mb


@torch.no_grad()
def latency_ms_per_image(model, dataloader, device="cpu", warmup_batches=5, timed_batches=20):
    """
    Measures average inference latency in ms/image on the provided dataloader.
    CPU-only friendly.
    """
    model.eval()
    it = iter(dataloader)

    # Warmup
    for _ in range(warmup_batches):
        try:
            x, _ = next(it)
        except StopIteration:
            return float("nan")
        x = x.to(device)
        _ = model(x)

    # Timed
    total_images = 0
    t0 = time.time()
    for _ in range(timed_batches):
        try:
            x, _ = next(it)
        except StopIteration:
            break
        x = x.to(device)
        _ = model(x)
        total_images += x.size(0)
    t1 = time.time()

    return (t1 - t0) * 1000.0 / max(total_images, 1)
