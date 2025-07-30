import torch
import matplotlib.pyplot as plt
import os
from models.ddpm import UNet, DDPM
import config

def generate_images():
    device = config.DEVICE
    print(f"Using device: {device}")

    print("Initializing model...")
    model = UNet(
        in_channels=1,
        out_channels=1,
        time_dim=config.DDPM_TIME_EMBEDDING_DIM,
        base_channels=config.DDPM_BASE_UNET_CHANNELS
    ).to(device)
    ddpm = DDPM(model, device)

    checkpoint_dir = config.SAVE_DIR
    print(f"Looking for checkpoints in: {os.path.abspath(checkpoint_dir)}")

    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('ddpm_epoch_') and f.endswith('.pth')]

    if not checkpoint_files:
        print("No checkpoint files found!")
        return

    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['train_loss']:.4f}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")

    print("Generating new chest X-ray samples...")
    model.eval()

    with torch.no_grad():
        samples = ddpm.sample(num_samples=8, image_size_tuple=(config.DDPM_IMAGE_SIZE, config.DDPM_IMAGE_SIZE))
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Generated Synthetic Chest X-rays', fontsize=16)

        for i, ax in enumerate(axes.flat):
            img = samples[i].cpu().squeeze().numpy()
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Generated X-ray {i + 1}')
            ax.axis('off')

        plt.tight_layout()
        output_path = config.SYNTHETIC_PATH
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f" Saved generated images to: {output_path}")
        plt.show()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    generate_images()
