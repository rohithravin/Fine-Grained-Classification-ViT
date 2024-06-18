import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from transformers import AutoImageProcessor
from torch.utils.data import DataLoader, random_split
import torch

import config

def create_dataset(model_checkpoint, batch_size_ = 32,  train_pct = 0.7, val_pct = 0.15, resize_img = 224):
    
    image_processor = None
    transform = None
    try:
        # Load the image processor
        image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
        # Define data transformations using the image processor's configuration
        transform = transforms.Compose([
            transforms.Resize((resize_img, resize_img)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        ])
    except:
        transform = transforms.Compose([
            transforms.Resize((resize_img, resize_img)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

   

    # Load the dataset
    dataset = datasets.ImageFolder(root=config.STANFORD_DOG_DATASET_LOCAL_PATH, transform=transform)

    # Define the train, validation, and test splits
    train_size = int(train_pct * len(dataset))
    val_size = int(val_pct * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Define batch size
    batch_size = batch_size_

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Print dataset sizes for verification
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader