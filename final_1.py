import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import os
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ChestXrayDataset(Dataset):
    def __init__(self, data_dir, transform=None, include_synthetic=False, synthetic_path=None):
        """
        Dataset for chest X-ray classification with better error handling
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Check if directory exists
        if not self.data_dir.exists():
            print(f"Directory not found: {self.data_dir}")
            return

        # Load real images
        for class_idx, class_name in enumerate(['NORMAL', 'PNEUMONIA']):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                image_files = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg')) + list(
                    class_dir.glob('*.jpeg'))
                for img_path in image_files:
                    self.image_paths.append(str(img_path))
                    self.labels.append(class_idx)
                print(f"Found {len(image_files)} {class_name} images")
            else:
                print(f"⚠️ Class directory not found: {class_dir}")

        # Optionally include synthetic images
        if include_synthetic and synthetic_path and os.path.exists(synthetic_path):
            # Add synthetic image as normal (you can modify this)
            self.image_paths.append(synthetic_path)
            self.labels.append(0)  # Treat as normal
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
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class PneumoniaClassifier(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=2, use_weights=True):
        super(PneumoniaClassifier, self).__init__()

        if model_name == 'resnet50':
            if use_weights:
                self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet50(weights=None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)

        elif model_name == 'densenet121':
            if use_weights:
                self.backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.densenet121(weights=None)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(0.1),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def train_model(model, train_loader, val_loader, num_epochs=25, device='cuda'):
    # Calculate class weights for imbalanced dataset
    train_dataset = train_loader.dataset
    class_counts = [train_dataset.labels.count(i) for i in range(2)]
    total_samples = sum(class_counts)
    class_weights = [total_samples / (2 * count) for count in class_counts]
    class_weights = torch.FloatTensor(class_weights).to(device)

    print(f"Class distribution - Normal: {class_counts[0]}, Pneumonia: {class_counts[1]}")
    print(f"Class weights - Normal: {class_weights[0]:.3f}, Pneumonia: {class_weights[1]:.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model.to(device)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_acc = 0.0
    best_model_path = 'best_pneumonia_model.pth'

    # Check if we have validation data
    has_validation = len(val_loader) > 0
    if not has_validation:
        print(" No validation data found. Using training data for validation.")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_acc = 100 * correct_train / total_train if total_train > 0 else 0

        # Validation phase
        if has_validation:
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_loss = val_loss / len(val_loader)
            val_acc = 100 * correct_val / total_val
        else:
            # Use training metrics as validation if no val data
            val_loss = train_loss
            val_acc = train_acc

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc
            }, best_model_path)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        scheduler.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        print('-' * 50)

    return train_losses, val_losses, train_accuracies, val_accuracies, best_model_path


def evaluate_model(model, test_loader, device='cuda'):
    if len(test_loader) == 0:
        print("No test data available for evaluation")
        return 0, 0, 0, 0

    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary',
                                                               zero_division=0)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    return accuracy, precision, recall, f1


def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(train_accs, label='Training Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


# Main training script
if __name__ == "__main__":
    # Configuration - UPDATE THESE PATHS!
    TRAIN_DIR =  "C:\\Users\\Aishwary\\PycharmProjects\\PythonProject\\dataset\\chest_xray\\chest_xray\\train" # Update this path
    VAL_DIR = "C:\\Users\\Aishwary\\PycharmProjects\\PythonProject\\dataset\\chest_xray\\chest_xray\\val"  # Update this path
    TEST_DIR =  "C:\\Users\\Aishwary\\PycharmProjects\\PythonProject\\dataset\\chest_xray\\chest_xray\\test" # Update this path
    SYNTHETIC_PATH = "generated_chest_xrays.png"  # Your DDPM generated images

    BATCH_SIZE = 8  # Reduced batch size for small datasets
    NUM_EPOCHS = 10  # Reduced epochs for testing
    MODEL_NAME = 'resnet50'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {DEVICE}")

    # Get transforms
    train_transform, val_transform = get_transforms()

    # Create datasets with error handling
    print("Loading datasets...")
    train_dataset = ChestXrayDataset(TRAIN_DIR, transform=train_transform,
                                     include_synthetic=True, synthetic_path=SYNTHETIC_PATH)
    val_dataset = ChestXrayDataset(VAL_DIR, transform=val_transform)
    test_dataset = ChestXrayDataset(TEST_DIR, transform=val_transform)

    # Check if we have enough data
    if len(train_dataset) == 0:
        print(" No training data found!")
        print("Please check your dataset paths or run the dataset setup helper first.")
        exit(1)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True) if len(val_dataset) > 0 else []
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4, pin_memory=True) if len(test_dataset) > 0 else []

    # Initialize model
    model = PneumoniaClassifier(model_name=MODEL_NAME, num_classes=2, use_weights=True)

    print(f"\n Training {MODEL_NAME} for pneumonia classification...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    if len(train_dataset) < 10:
        print("⚠️ Very small dataset - this is just for testing!")

    # Train the model
    train_losses, val_losses, train_accs, val_accs, best_model_path = train_model(
        model, train_loader, val_loader, NUM_EPOCHS, DEVICE
    )

    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)

    # Load best model and evaluate on test set
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        print("\n" + "=" * 50)
        print("FINAL TEST EVALUATION")
        print("=" * 50)

        if len(test_loader) > 0:
            accuracy, precision, recall, f1 = evaluate_model(model, test_loader, DEVICE)
            print(f"\nBest model saved as: {best_model_path}")
            print(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")
        else:
            print("No test data available for final evaluation")
    else:
        print(" No model checkpoint found")