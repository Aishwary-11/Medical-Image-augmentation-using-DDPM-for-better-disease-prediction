import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import os


# First, recreate the model classes (copy from your training script)
class UNet(nn.Module):
    """Simplified U-Net for DDPM - Memory optimized for RTX 3050"""

    def __init__(self, in_channels=1, out_channels=1, time_dim=256):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Encoder (downsampling)
        self.enc1 = self.conv_block(in_channels, 64, time_dim)
        self.enc2 = self.conv_block(64, 128, time_dim)
        self.enc3 = self.conv_block(128, 256, time_dim)
        self.enc4 = self.conv_block(256, 256, time_dim)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512, time_dim)

        # Decoder (upsampling)
        self.dec4 = self.conv_block(512 + 256, 256, time_dim)
        self.dec3 = self.conv_block(256 + 256, 256, time_dim)
        self.dec2 = self.conv_block(256 + 128, 128, time_dim)
        self.dec1 = self.conv_block(128 + 64, 64, time_dim)

        # Output
        self.final = nn.Conv2d(64, out_channels, 1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def conv_block(self, in_ch, out_ch, time_dim):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, t):
        # Time embedding
        t_emb = self.get_time_embedding(t, x.device)
        t_emb = self.time_mlp(t_emb)

        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder with skip connections
        dec4 = self.dec4(torch.cat([self.upsample(bottleneck), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.upsample(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.upsample(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.upsample(dec2), enc1], dim=1))

        return self.final(dec1)

    def get_time_embedding(self, timesteps, device):
        """Sinusoidal time embeddings"""
        half_dim = 128
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class DDPM:
    """Denoising Diffusion Probabilistic Model"""

    def __init__(self, model, device, num_timesteps=1000):
        self.model = model
        self.device = device
        self.num_timesteps = num_timesteps

        # Define beta schedule (linear)
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # Precompute values for sampling
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    @torch.no_grad()
    def sample(self, num_samples, image_size):
        """Generate samples"""
        # Start with pure noise
        x = torch.randn(num_samples, 1, image_size, image_size, device=self.device)

        # Reverse diffusion process
        for i in tqdm(reversed(range(self.num_timesteps)), desc="Sampling"):
            t = torch.full((num_samples,), i, device=self.device, dtype=torch.long)

            # Predict noise
            predicted_noise = self.model(x, t)

            # Remove predicted noise
            alpha_t = self.alphas[i]
            alpha_bar_t = self.alpha_bars[i]
            beta_t = self.betas[i]

            # Compute denoised image
            x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)

            # Add noise (except for last step)
            if i > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta_t) * noise

        return x


def load_and_generate():
    """Load trained model and generate new samples"""

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model architecture
    print("Initializing model...")
    model = UNet(in_channels=1, out_channels=1).to(device)
    ddpm = DDPM(model, device)

    # Find checkpoint files in the ddpm_chest_xray directory
    checkpoint_dir = './ddpm_chest_xray'

    print(f"Looking for checkpoints in: {os.path.abspath(checkpoint_dir)}")

    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        # Try to find any directory with ddpm in the name
        for item in os.listdir('.'):
            if os.path.isdir(item) and 'ddpm' in item.lower():
                print(f"Found similar directory: {item}")
                checkpoint_dir = item
                break
        else:
            print("No checkpoint directory found!")
            return

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('ddpm_epoch_') and f.endswith('.pth')]

    if not checkpoint_files:
        print("No checkpoint files found!")
        return

    # Get the latest epoch checkpoint
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['train_loss']:.4f}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")

    # Generate new samples
    print("Generating new chest X-ray samples...")
    model.eval()

    with torch.no_grad():
        # Generate 8 samples
        samples = ddpm.sample(num_samples=8, image_size=256)

        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Generated Synthetic Chest X-rays', fontsize=16)

        for i, ax in enumerate(axes.flat):
            img = samples[i].cpu().squeeze().numpy()
            img = (img + 1) / 2  # Denormalize from [-1,1] to [0,1]
            img = torch.clamp(torch.tensor(img), 0, 1).numpy()  # Ensure valid range

            ax.imshow(img, cmap='gray')
            ax.set_title(f'Generated X-ray {i + 1}')
            ax.axis('off')

        plt.tight_layout()

        # Save results
        output_path = 'generated_chest_xrays.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f" Saved generated images to: {output_path}")

        plt.show()

        # Clear GPU memory
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("=== DDPM Sample Generation ===")
    load_and_generate()