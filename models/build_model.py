import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet18_cifar10(device="cpu"):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(device)

