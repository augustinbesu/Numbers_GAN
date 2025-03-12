import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from config import BATCH_SIZE, DATA_DIR

def get_data_transforms():
    """Devuelve las transformaciones para el dataset MNIST"""
    return transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def get_mnist_dataset(train=True):
    """Carga el dataset MNIST"""
    transform = get_data_transforms()
    return datasets.MNIST(
        root=DATA_DIR, 
        train=train, 
        download=True, 
        transform=transform
    )

def get_dataloader(batch_size=BATCH_SIZE, train=True, num_workers=0):
    """Devuelve un DataLoader para el dataset MNIST"""
    dataset = get_mnist_dataset(train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True
    )

def create_directories():
    """Crea los directorios necesarios para el proyecto"""
    from config import IMAGES_DIR, MODELS_DIR, EVALUATION_DIR, DATA_DIR
    
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EVALUATION_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True) 