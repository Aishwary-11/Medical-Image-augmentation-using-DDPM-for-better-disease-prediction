import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random
import math

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class NIHChestXrayDataset(Dataset):
    """NIH Chest X-ray Dataset for DDPM Training (adapted for user's folder structure)"""

    def __init__(self, dataset_path, image_list, transform=None, max_images=None):
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.image_paths = []

        # --- MODIFIED LOGIC FOR IMAGE COLLECTION ---
        # Create a mapping of image names to full paths from your actual dataset structure
        image_path_mapping = {}
        print(f"Dataset.__init__: Scanning subdirectories in {self.dataset_path} for image files...")

        sub_directories_found = 0
        # Iterate through top-level directories within dataset_path (e.g., NORMAL, PNEUMONIA)
        for category_dir in self.dataset_path.iterdir():
            if category_dir.is_dir():
                sub_directories_found += 1
                # print(f"  Scanning category: {category_dir.name}") # Uncomment for very detailed debug

                # Collect .png and .jpeg files directly within this category directory
                for ext in ["*.png", "*.jpeg"]: # Both types are present in your dataset
                    for img_file in category_dir.glob(ext):
                        if img_file.is_file(): # Ensure it's a file
                            image_path_mapping[img_file.name] = img_file

        if sub_directories_found == 0:
            print(f"Dataset.__init__ Warning: No subdirectories found in '{self.dataset_path}'. "
                  f"Expected 'NORMAL', 'PNEUMONIA' etc. to be direct children. "
                  f"Please ensure dataset_path points to the parent of these category folders.")
        else:
            print(f"Dataset.__init__: Found {len(image_path_mapping)} total image files across {sub_directories_found} category directories.")
        # --- END MODIFIED LOGIC ---

        # Filter based on provided image list (e.g., from train_val_list.txt)
        num_images_in_list = len(image_list)
        for img_name in image_list:
            if img_name in image_path_mapping:
                self.image_paths.append(image_path_mapping[img_name])

        print(f"Dataset.__init__: {len(self.image_paths)} images found that are present in the provided list ({num_images_in_list} names total).")


        # Limit dataset size if specified (for testing/memory constraints)
        if max_images and len(self.image_paths) > max_images:
            random.shuffle(self.image_paths) # Shuffle before truncating to ensure a random subset
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
            # Load image
            # .convert('L') ensures it's grayscale, necessary for 1-channel model
            image = Image.open(img_path).convert('L')

            if self.transform:
                image = self.transform(image)

            return image

        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Returning fallback image.")
            # Determine image size from transform if possible, otherwise default
            img_size = (256, 256) # Default size, matching CONFIG['image_size']
            if self.transform:
                for t in self.transform.transforms:
                    if isinstance(t, transforms.Resize):
                        size = t.size
                        if isinstance(size, int):
                            img_size = (size, size)
                        else:
                            img_size = size
                        break

            # Create and return a black image as fallback
            fallback_image = Image.new('L', img_size, 0)
            if self.transform:
                return self.transform(fallback_image)
            return fallback_image


# Helper module for sinusoidal position embeddings
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1:  # Zero pad if dim is odd
            embeddings = torch.nn.functional.pad(embeddings, (0, 1))
        return embeddings


# Modified UNet ConvBlock to include time embedding and residual connection
class TimeEmbeddedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, num_groups=8):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)  # Project to get scale and shift
        )

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act2 = nn.SiLU()

        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        # Initial convolution and normalization
        hidden_state = self.norm1(self.conv1(x))

        # Project time embedding to get scale and shift parameters
        time_params = self.time_mlp(t_emb)
        time_params = time_params.unsqueeze(-1).unsqueeze(-1)  # Reshape to (B, C*2, 1, 1)
        scale, shift = torch.chunk(time_params, 2, dim=1)  # Split into scale and shift

        # Apply scale and shift (AdaGN-like, or FiLM-like)
        h = hidden_state * (1 + scale) + shift  # Modulate features before activation
        h = self.act2(self.norm2(self.conv2(h)))  # Second convolution part

        return h + self.res_conv(x)  # Add residual connection


class UNet(nn.Module):
    """Simplified U-Net for DDPM - Memory optimized for RTX 3050, with proper time embedding"""

    def __init__(self, in_channels=1, out_channels=1, time_dim=256, base_channels=64):
        super().__init__()

        # Time embedding
        self.time_embed = SinusoidalPositionEmbeddings(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        ch_mults = (1, 2, 4, 4)  # Channel multipliers for encoder stages

        # Encoder
        self.enc1 = TimeEmbeddedConvBlock(in_channels, base_channels * ch_mults[0], time_dim)
        self.enc2 = TimeEmbeddedConvBlock(base_channels * ch_mults[0], base_channels * ch_mults[1], time_dim)
        self.enc3 = TimeEmbeddedConvBlock(base_channels * ch_mults[1], base_channels * ch_mults[2], time_dim)
        self.enc4 = TimeEmbeddedConvBlock(base_channels * ch_mults[2], base_channels * ch_mults[3], time_dim)

        # Bottleneck
        self.bottleneck = TimeEmbeddedConvBlock(base_channels * ch_mults[3], base_channels * 8, time_dim)

        # Decoder
        # Input channels for decoder blocks are (bottleneck_channels + corresponding_encoder_channels)
        self.dec4 = TimeEmbeddedConvBlock(base_channels * 8 + base_channels * ch_mults[3], base_channels * ch_mults[3],
                                          time_dim)
        self.dec3 = TimeEmbeddedConvBlock(base_channels * ch_mults[3] + base_channels * ch_mults[2],
                                          base_channels * ch_mults[2], time_dim)
        self.dec2 = TimeEmbeddedConvBlock(base_channels * ch_mults[2] + base_channels * ch_mults[1],
                                          base_channels * ch_mults[1], time_dim)
        self.dec1 = TimeEmbeddedConvBlock(base_channels * ch_mults[1] + base_channels * ch_mults[0],
                                          base_channels * ch_mults[0], time_dim)

        # Output
        self.final_conv = nn.Conv2d(base_channels * ch_mults[0], out_channels, 1)

        self.pool = nn.MaxPool2d(2)
        # Use F.interpolate for Upsample for better control (align_corners=False for images)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) # Changed to False

    def forward(self, x, t):
        # Process time
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)  # Shape: (batch_size, time_dim)

        # Encoder
        s1 = self.enc1(x, t_emb)  # (B, C1, H, W)
        s2 = self.enc2(self.pool(s1), t_emb)  # (B, C2, H/2, W/2)
        s3 = self.enc3(self.pool(s2), t_emb)  # (B, C3, H/4, W/4)
        s4 = self.enc4(self.pool(s3), t_emb)  # (B, C4, H/8, W/8)

        # Bottleneck
        b = self.bottleneck(self.pool(s4), t_emb)  # (B, C_bottle, H/16, W/16)

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.upsample(b), s4], dim=1), t_emb)
        d3 = self.dec3(torch.cat([self.upsample(d4), s3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.upsample(d3), s2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.upsample(d2), s1], dim=1), t_emb)

        return self.final_conv(d1)


class DDPM:
    """Denoising Diffusion Probabilistic Model"""

    def __init__(self, model, device, num_timesteps=1000):
        self.model = model
        self.device = device
        self.num_timesteps = num_timesteps

        # Define beta schedule (linear)
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # Precompute values for sampling and noising
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)  # For sampling
        # DDPM Eq. (7) variance: beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        # alpha_bar_{t-1} is alpha_bars[:-1] for t > 0, and 1 for t=0 (no previous alpha_bar)
        # For simplicity, many implementations use beta_t for sigma_t^2 directly, or a clipped version.
        # The Ho et al. (2020) paper uses `beta_t` for `sigma_t` in sampling (eq. 11), often `sqrt(beta_t)` for std.
        # Let's stick closer to DDPM Eq. 7 for posterior_variance calculation, ensuring no negative values.
        # posterior_variance should be (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t
        alpha_bar_prev = torch.cat([torch.tensor([1.0], device=device), self.alpha_bars[:-1]])
        self.posterior_variance = self.betas * (1. - alpha_bar_prev) / (1. - self.alpha_bars)
        # Clip variance to prevent numerical issues with very small values
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)) # Use clamp for stability

    def add_noise(self, x_start, t):
        """Add noise to images according to timestep t. q(x_t | x_0)"""
        noise = torch.randn_like(x_start)
        # Select appropriate values for each item in the batch based on its 't'
        sqrt_alpha_bar_t = self.sqrt_alpha_bars.gather(0, t).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars.gather(0, t).view(-1, 1, 1, 1)

        noisy_x = sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise
        return noisy_x, noise

    def train_step(self, x_start):
        """Single training step"""
        batch_size = x_start.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device, dtype=torch.long)

        # Add noise
        noisy_x, noise_gt = self.add_noise(x_start, t)

        # Predict noise
        predicted_noise = self.model(noisy_x, t)

        # Compute loss
        loss = nn.MSELoss()(predicted_noise, noise_gt)
        return loss

    @torch.no_grad()
    def p_sample(self, x_t, t_idx):
        """Sample x_{t-1} from x_t."""
        t_tensor = torch.full((x_t.shape[0],), t_idx, device=self.device, dtype=torch.long)

        betas_t = self.betas.gather(0, t_tensor).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bars_t = self.sqrt_one_minus_alpha_bars.gather(0, t_tensor).view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas.gather(0, t_tensor).view(-1, 1, 1, 1)

        # Equation 11 in DDPM paper: x_t-1 = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1-alpha_bar_t) * epsilon_theta) + sigma_t * z
        predicted_noise = self.model(x_t, t_tensor)
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alpha_bars_t)

        if t_idx == 0: # If at the last step (t=0), no noise is added
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance.gather(0, t_tensor).view(-1, 1, 1, 1)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, num_samples, image_size_tuple):
        """Generate samples (full reverse diffusion process)"""
        H, W = image_size_tuple
        # Start with pure noise
        x_t = torch.randn(num_samples, 1, H, W, device=self.device)

        # Reverse diffusion process
        for i in tqdm(reversed(range(self.num_timesteps)), desc="Sampling", total=self.num_timesteps):
            x_t = self.p_sample(x_t, i)

        # Denormalize images from [-1, 1] to [0, 1] for saving/display
        x_t = (x_t + 1.0) / 2.0
        return torch.clamp(x_t, 0.0, 1.0) # Ensure values are within [0, 1]


def create_data_loaders(dataset_path, batch_size=4, image_size=256, max_train_images=None, max_val_images=1000):
    """Create train and validation data loaders"""

    # Load train/val split from the provided list
    train_val_list_path = Path(dataset_path) / "train_val_list.txt"
    if not train_val_list_path.exists():
        raise FileNotFoundError(f"'{train_val_list_path}' not found. Please ensure it exists in your dataset_path "
                                f"and contains a list of image filenames.")

    with open(train_val_list_path, 'r') as f:
        train_val_list = [line.strip() for line in f.readlines()]

    # Split train/val (80/20) - shuffle *before* splitting
    random.shuffle(train_val_list)
    split_idx = int(0.8 * len(train_val_list))
    train_list = train_val_list[:split_idx]
    val_list = train_val_list[split_idx:]

    print(f"Total image names in train_val_list.txt: {len(train_val_list)}")
    print(f"Train image names (from list): {len(train_list)}")
    print(f"Validation image names (from list): {len(val_list)}")

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # Scales to [0, 1]
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])

    # Create datasets
    train_dataset = NIHChestXrayDataset(
        dataset_path, train_list, transform=transform, max_images=max_train_images
    )
    val_dataset = NIHChestXrayDataset(
        dataset_path, val_list, transform=transform, max_images=max_val_images
    )

    # Persistent workers can cause issues on Windows with certain multiprocessing setups.
    # Set to False explicitly on Windows to avoid potential freezes.
    dataloader_persistent_workers = True if os.name != 'nt' else False

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=os.cpu_count() // 2 if os.cpu_count() > 1 else 0, # Use half CPU cores, or 0 if only 1 core
        pin_memory=True,
        persistent_workers=dataloader_persistent_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=os.cpu_count() // 2 if os.cpu_count() > 1 else 0,
        pin_memory=True,
        persistent_workers=dataloader_persistent_workers
    )

    # Check if loaders are empty
    if len(train_dataset) == 0:
        print("\nWARNING: Train dataset is empty. Check your dataset_path and train_val_list.txt contents.")
    if len(val_dataset) == 0:
        print("\nWARNING: Validation dataset is empty. Check your dataset_path and train_val_list.txt contents.")


    return train_loader, val_loader


def train_ddpm(dataset_path, num_epochs=50, batch_size=4, image_size=256,
               learning_rate=1e-4, save_dir="./ddpm_checkpoints",
               max_train_images=14000, max_val_images=5000, # These limits apply to the number of images actually loaded, not just in the list
               base_unet_channels=64, time_embedding_dim=256,
               checkpoint_interval=10, sample_interval=20):
    """Train DDPM model"""

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        dataset_path, batch_size, image_size, max_train_images, max_val_images
    )

    # Initialize model
    print("Initializing model...")
    model = UNet(in_channels=1, out_channels=1, time_dim=time_embedding_dim, base_channels=base_unet_channels).to(
        device)
    ddpm = DDPM(model, device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss_epoch = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]")
        for batch_idx, images in enumerate(pbar):
            # Check for empty batches or problematic data (due to fallback)
            if images is None or images.nelement() == 0:
                print(f"Skipping empty batch at index {batch_idx} in training.")
                continue
            images = images.to(device)

            optimizer.zero_grad()
            loss = ddpm.train_step(images)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            train_loss_epoch += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

            # Memory management (optional, can slow down if too frequent)
            if device == 'cuda' and batch_idx % 200 == 0: # Adjust frequency if needed
                torch.cuda.empty_cache()

        avg_train_loss = train_loss_epoch / len(train_loader) if len(train_loader) > 0 else 0.0
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]")
            for images in vbar:
                if images is None or images.nelement() == 0:
                    print(f"Skipping empty batch in validation.")
                    continue
                images = images.to(device)
                loss = ddpm.train_step(images)  # Use train_step to compute loss on val data
                val_loss_epoch += loss.item()
                vbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})

        avg_val_loss = val_loss_epoch / len(val_loader) if len(val_loader) > 0 else 0.0
        val_losses.append(avg_val_loss)

        # Update learning rate
        scheduler.step()

        print(
            f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(save_dir, f"ddpm_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': {  # Save config for reproducibility
                    'image_size': image_size,
                    'base_unet_channels': base_unet_channels,
                    'time_embedding_dim': time_embedding_dim,
                    'num_timesteps': ddpm.num_timesteps # Add DDPM specific config
                }
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        # Generate sample images
        if (epoch + 1) % sample_interval == 0 or epoch == num_epochs - 1:
            print("Generating sample images...")
            model.eval()  # Ensure model is in eval mode
            samples = ddpm.sample(4, (image_size, image_size))  # Pass tuple for H, W

            # Save samples
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            for i, ax in enumerate(axes.flat):
                img = samples[i].cpu().squeeze().numpy()
                ax.imshow(img, cmap='gray', vmin=0, vmax=1) # Images are already [0,1]
                ax.axis('off')

            plt.tight_layout()
            sample_image_path = os.path.join(save_dir, f"samples_epoch_{epoch + 1}.png")
            plt.savefig(sample_image_path)
            print(f"Sample images saved to {sample_image_path}")
            plt.close(fig) # Close the figure to free up memory

            if device == 'cuda':
                torch.cuda.empty_cache()

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('DDPM Training Loss')
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close() # Close the figure

    print("Training completed!")
    return model, ddpm


if __name__ == "__main__":
    # Configuration for RTX 3050 (4GB VRAM)
    CONFIG = {
        # IMPORTANT: Set this to your dataset path, which is the parent of NORMAL/ and PNEUMONIA/
        'dataset_path': r"C:\Users\Aishwary\PycharmProjects\PythonProject\dataset\chest_xray\diesease",
        'num_epochs': 50,
        'batch_size': 4,  # Conservative for 4GB VRAM with 256x256
        'image_size': 256,  # Common resolution for DDPMs on limited VRAM
        'learning_rate': 1e-4,  # Common starting LR for AdamW
        'max_train_images': 14000,  # Limit training set to prevent loading all images at once
        'max_val_images': 5000,  # Limit validation set
        'save_dir': "./ddpm_chest_xray_output",
        'base_unet_channels': 64,  # Base channels for U-Net
        'time_embedding_dim': 256,  # Dimension for time embeddings
        'checkpoint_interval': 10,  # How often to save checkpoints
        'sample_interval': 10  # How often to generate and save sample images
    }

    # Verify dataset_path and required files
    dataset_root = Path(CONFIG['dataset_path'])
    train_val_list_file = dataset_root / "train_val_list.txt"

    if not dataset_root.exists():
        print(f"ERROR: Dataset path '{CONFIG['dataset_path']}' does not exist.")
        print("Please update the 'dataset_path' in the CONFIG dictionary to point to the directory that contains 'NORMAL' and 'PNEUMONIA' folders.")
    elif not train_val_list_file.exists():
        print(f"ERROR: '{train_val_list_file}' not found.")
        print("This file is required for splitting data. Please ensure it's in your 'dataset_path' and contains a list of image filenames (e.g., IM-0001-0001.jpeg or 00013774_026.png, one per line).")
    else:
        print("=== DDPM Training for NIH Chest X-ray Dataset ===")
        print(f"Configuration: {CONFIG}")

        # Start training
        model, ddpm_sampler = train_ddpm(**CONFIG)

        print("\nTraining completed! Check the 'save_dir' for checkpoints and samples.")
        print(f"To generate more samples later, load a checkpoint and use `ddpm_sampler.sample()`.")