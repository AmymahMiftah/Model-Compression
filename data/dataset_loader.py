import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

BATCH_SIZE = 64

def get_cifar10_loaders(batch_size: int = BATCH_SIZE,
                        num_workers: int = 2,
                        img_size: int = 96,
                        randaugment: bool = False,
                        data_root: str = "./data"):
    """CIFAR-10 DataLoaders with ImageNet-style normalization for pretrained backbones."""
    resize_side = int(img_size * 256 / 224)

    aug = []
    if randaugment:
        aug.append(transforms.RandAugment())

    transform_train = transforms.Compose([
        transforms.Resize(resize_side),
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        *aug,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(resize_side),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, test_loader