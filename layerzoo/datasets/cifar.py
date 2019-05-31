from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms as tf


def cifar10():
    train_transform = tf.Compose([
        tf.RandomCrop(32, padding=4),
        tf.RandomHorizontalFlip(),
        tf.ToTensor(),
        tf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CIFAR10(root='./data', train=True, download=True,
                            transform=train_transform)

    val_transform = tf.Compose([
        tf.ToTensor(),
        tf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    val_dataset = CIFAR10(root='./data', train=False, download=True,
                          transform=val_transform)

    return train_dataset, val_dataset
