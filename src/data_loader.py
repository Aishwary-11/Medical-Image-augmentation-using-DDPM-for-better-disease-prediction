import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import os
import random

class NIHChestXrayDataset(Dataset):
    def __init__(self, dataset_path, image_list, transform=None, max_images=None):
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.image_paths = []
        image_path_mapping = {}
        sub_directories_found = 0
        for category_dir in self.dataset_path.iterdir():
            if category_dir.is_dir():
                sub_directories_found += 1
                for ext in ["*.png", "*.jpeg"]:
                    for img_file in category_dir.glob(ext):
                        if img_file.is_file():
                            image_path_mapping[img_file.name] = img_file
        if sub_directories_found == 0:
            print(f"Dataset.__init__ Warning: No subdirectories found in '{self.dataset_path}'.")
        else:
            print(f"Dataset.__init__: Found {len(image_path_mapping)} total image files across {sub_directories_found} category directories.")
        num_images_in_list = len(image_list)
        for img_name in image_list:
            if img_name in image_path_mapping:
                self.image_paths.append(image_path_mapping[img_name])
        print(f"Dataset.__init__: {len(self.image_paths)} images found that are present in the provided list ({num_images_in_list} names total).")
        if max_images and len(self.image_paths) > max_images:
            random.shuffle(self.image_paths)
            self.image_paths = self.image_paths[:max_images]
            print(f"Dataset.__init__: Dataset size limited to {max_images} images.")
        if not self.image_paths:
            print("Dataset.__init__ Warning: No images found after filtering. Check dataset_path and image_list contents.")
        else:
            print(f"Dataset initialized with {len(self.image_paths)} images for use.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('L')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Returning fallback image.")
            img_size = (256, 256)
            if self.transform:
                for t in self.transform.transforms:
                    if isinstance(t, transforms.Resize):
                        size = t.size
                        if isinstance(size, int):
                            img_size = (size, size)
                        else:
                            img_size = size
                        break
            fallback_image = Image.new('L', img_size, 0)
            if self.transform:
                return self.transform(fallback_image)
            return fallback_image

class ChestXrayDataset(Dataset):
    def __init__(self, data_dir, transform=None, include_synthetic=False, synthetic_path=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        if not self.data_dir.exists():
            print(f"Directory not found: {self.data_dir}")
            return
        for class_idx, class_name in enumerate(['NORMAL', 'PNEUMONIA']):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                image_files = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg'))
                for img_path in image_files:
                    self.image_paths.append(str(img_path))
                    self.labels.append(class_idx)
                print(f"Found {len(image_files)} {class_name} images")
            else:
                print(f"⚠️ Class directory not found: {class_dir}")
        if include_synthetic and synthetic_path and os.path.exists(synthetic_path):
            self.image_paths.append(synthetic_path)
            self.labels.append(0)
            print(f"Added synthetic image: {synthetic_path}")
        print(f"Total loaded: {len(self.image_paths)} images")
        if len(self.image_paths) > 0:
            print(f"Normal: {self.labels.count(0)}, Pneumonia: {self.labels.count(1)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def create_ddpm_data_loaders(dataset_path, batch_size=4, image_size=256, max_train_images=None, max_val_images=1000):
    train_val_list_path = Path(dataset_path) / "train_val_list.txt"
    if not train_val_list_path.exists():
        raise FileNotFoundError(f"'{train_val_list_path}' not found.")
    with open(train_val_list_path, 'r') as f:
        train_val_list = [line.strip() for line in f.readlines()]
    random.shuffle(train_val_list)
    split_idx = int(0.8 * len(train_val_list))
    train_list = train_val_list[:split_idx]
    val_list = train_val_list[split_idx:]
    print(f"Total image names in train_val_list.txt: {len(train_val_list)}")
    print(f"Train image names (from list): {len(train_list)}")
    print(f"Validation image names (from list): {len(val_list)}")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_dataset = NIHChestXrayDataset(
        dataset_path, train_list, transform=transform, max_images=max_train_images
    )
    val_dataset = NIHChestXrayDataset(
        dataset_path, val_list, transform=transform, max_images=max_val_images
    )
    dataloader_persistent_workers = True if os.name != 'nt' else False
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=os.cpu_count() // 2 if os.cpu_count() > 1 else 0,
        pin_memory=True,
        persistent_workers=dataloader_persistent_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=os.cpu_count() // 2 if os.cpu_count() > 1 else 0,
        pin_memory=True,
        persistent_workers=dataloader_persistent_workers
    )
    if len(train_dataset) == 0:
        print("\nWARNING: Train dataset is empty.")
    if len(val_dataset) == 0:
        print("\nWARNING: Validation dataset is empty.")
    return train_loader, val_loader
