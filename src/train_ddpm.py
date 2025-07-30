import torch
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from models.ddpm import UNet, DDPM
from data_loader import create_ddpm_data_loaders
import config

def train_ddpm():
    device = config.DEVICE
    print(f"Using device: {device}")

    os.makedirs(config.SAVE_DIR, exist_ok=True)

    print("Creating data loaders...")
    train_loader, val_loader = create_ddpm_data_loaders(
        config.DATASET_PATH,
        config.DDPM_BATCH_SIZE,
        config.DDPM_IMAGE_SIZE,
        config.DDPM_MAX_TRAIN_IMAGES,
        config.DDPM_MAX_VAL_IMAGES
    )

    print("Initializing model...")
    model = UNet(
        in_channels=1,
        out_channels=1,
        time_dim=config.DDPM_TIME_EMBEDDING_DIM,
        base_channels=config.DDPM_BASE_UNET_CHANNELS
    ).to(device)
    ddpm = DDPM(model, device)

    optimizer = optim.AdamW(model.parameters(), lr=config.DDPM_LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.DDPM_NUM_EPOCHS)

    print("Starting training...")
    train_losses = []
    val_losses = []

    for epoch in range(config.DDPM_NUM_EPOCHS):
        model.train()
        train_loss_epoch = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.DDPM_NUM_EPOCHS} [Training]")
        for batch_idx, images in enumerate(pbar):
            if images is None or images.nelement() == 0:
                print(f"Skipping empty batch at index {batch_idx} in training.")
                continue
            images = images.to(device)
            optimizer.zero_grad()
            loss = ddpm.train_step(images)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_epoch += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            if device == 'cuda' and batch_idx % 200 == 0:
                torch.cuda.empty_cache()
        avg_train_loss = train_loss_epoch / len(train_loader) if len(train_loader) > 0 else 0.0
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config.DDPM_NUM_EPOCHS} [Validation]")
            for images in vbar:
                if images is None or images.nelement() == 0:
                    print(f"Skipping empty batch in validation.")
                    continue
                images = images.to(device)
                loss = ddpm.train_step(images)
                val_loss_epoch += loss.item()
                vbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
        avg_val_loss = val_loss_epoch / len(val_loader) if len(val_loader) > 0 else 0.0
        val_losses.append(avg_val_loss)

        scheduler.step()

        print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % config.DDPM_CHECKPOINT_INTERVAL == 0 or epoch == config.DDPM_NUM_EPOCHS - 1:
            checkpoint_path = os.path.join(config.SAVE_DIR, f"ddpm_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': {
                    'image_size': config.DDPM_IMAGE_SIZE,
                    'base_unet_channels': config.DDPM_BASE_UNET_CHANNELS,
                    'time_embedding_dim': config.DDPM_TIME_EMBEDDING_DIM,
                    'num_timesteps': ddpm.num_timesteps
                }
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        if (epoch + 1) % config.DDPM_SAMPLE_INTERVAL == 0 or epoch == config.DDPM_NUM_EPOCHS - 1:
            print("Generating sample images...")
            model.eval()
            samples = ddpm.sample(4, (config.DDPM_IMAGE_SIZE, config.DDPM_IMAGE_SIZE))
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            for i, ax in enumerate(axes.flat):
                img = samples[i].cpu().squeeze().numpy()
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                ax.axis('off')
            plt.tight_layout()
            sample_image_path = os.path.join(config.SAVE_DIR, f"samples_epoch_{epoch + 1}.png")
            plt.savefig(sample_image_path)
            print(f"Sample images saved to {sample_image_path}")
            plt.close(fig)
            if device == 'cuda':
                torch.cuda.empty_cache()

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('DDPM Training Loss')
    plt.savefig(os.path.join(config.SAVE_DIR, 'training_curves.png'))
    plt.close()

    print("Training completed!")

if __name__ == "__main__":
    train_ddpm()
