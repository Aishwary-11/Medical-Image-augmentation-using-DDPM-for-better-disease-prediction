import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from models.classifier import PneumoniaClassifier
from data_loader import ChestXrayDataset
from utils import plot_training_history, evaluate_model
import config

def train_classifier():
    device = config.DEVICE
    print(f"Using device: {device}")

    train_transform, val_transform = get_transforms()

    print("Loading datasets...")
    train_dataset = ChestXrayDataset(config.TRAIN_DIR, transform=train_transform,
                                     include_synthetic=True, synthetic_path=config.SYNTHETIC_PATH)
    val_dataset = ChestXrayDataset(config.VAL_DIR, transform=val_transform)
    test_dataset = ChestXrayDataset(config.TEST_DIR, transform=val_transform)

    if len(train_dataset) == 0:
        print(" No training data found!")
        exit(1)

    train_loader = DataLoader(train_dataset, batch_size=config.CLASSIFIER_BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.CLASSIFIER_BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True) if len(val_dataset) > 0 else []
    test_loader = DataLoader(test_dataset, batch_size=config.CLASSIFIER_BATCH_SIZE,
                             shuffle=False, num_workers=4, pin_memory=True) if len(test_dataset) > 0 else []

    model = PneumoniaClassifier(model_name=config.CLASSIFIER_MODEL_NAME, num_classes=2, use_weights=True)

    print(f"\n Training {config.CLASSIFIER_MODEL_NAME} for pneumonia classification...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_losses, val_losses, train_accs, val_accs, best_model_path = train_model(
        model, train_loader, val_loader, config.CLASSIFIER_NUM_EPOCHS, device
    )

    plot_training_history(train_losses, val_losses, train_accs, val_accs)

    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        print("\n" + "=" * 50)
        print("FINAL TEST EVALUATION")
        print("=" * 50)

        if len(test_loader) > 0:
            accuracy, precision, recall, f1 = evaluate_model(model, test_loader, device)
            print(f"\nBest model saved as: {best_model_path}")
            print(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")
        else:
            print("No test data available for final evaluation")
    else:
        print(" No model checkpoint found")

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

def train_model(model, train_loader, val_loader, num_epochs, device):
    train_dataset = train_loader.dataset
    class_counts = [train_dataset.labels.count(i) for i in range(2)]
    total_samples = sum(class_counts)
    class_weights = [total_samples / (2 * count) for count in class_counts]
    class_weights = torch.FloatTensor(class_weights).to(device)

    print(f"Class distribution - Normal: {class_counts[0]}, Pneumonia: {class_counts[1]}")
    print(f"Class weights - Normal: {class_weights[0]:.3f}, Pneumonia: {class_weights[1]:.3f}")

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.CLASSIFIER_LEARNING_RATE, weight_decay=config.CLASSIFIER_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.CLASSIFIER_STEP_SIZE, gamma=config.CLASSIFIER_GAMMA)

    model.to(device)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_acc = 0.0
    best_model_path = 'best_pneumonia_model.pth'

    has_validation = len(val_loader) > 0
    if not has_validation:
        print(" No validation data found. Using training data for validation.")

    for epoch in range(num_epochs):
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
            val_loss = train_loss
            val_acc = train_acc

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc
            }, best_model_path)

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

if __name__ == "__main__":
    import torchvision.transforms as transforms
    train_classifier()
