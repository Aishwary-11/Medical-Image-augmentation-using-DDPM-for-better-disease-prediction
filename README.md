# Medical-Image-augmentation-using-DDPM-for-better-disease-prediction
This project uses Denoising Diffusion Probabilistic Models (DDPM) to generate synthetic medical images for augmenting limited datasets. DDPM learns to create realistic images by reversing a noise process, helping overcome data scarcity in medical imaging.


# Optimized Conditional DDPM for High-Performance Medical Image Generation

This project implements an advanced Conditional Denoising Diffusion Probabilistic Model (DDPM) specifically engineered for high performance on modern GPU hardware (e.g., NVIDIA A5000). It focuses on generating synthetic medical-style images (using MNIST as a placeholder dataset) and incorporates numerous state-of-the-art optimization techniques for faster training and efficient inference.

The model leverages an optimized UNet architecture with efficient attention mechanisms (including Flash Attention if available), sinusoidal time embeddings, and class conditioning for guided image generation using Classifier-Free Guidance (CFG).

## Table of Contents

1.  [Overview](#overview)
2.  [Key Features](#key-features)
3.  [Architectural Highlights](#architectural-highlights)
    *   [Optimized UNet](#optimized-unet)
    *   [Diffusion Process & Classifier-Free Guidance](#diffusion-process--classifier-free-guidance)
    *   [Performance Optimizations](#performance-optimizations)
4.  [Dataset](#dataset)
5.  [Prerequisites](#prerequisites)
6.  [Installation](#installation)
7.  [Usage](#usage)
    *   [Training](#training)
    *   [Resuming Training](#resuming-training)
    *   [Generating Samples](#generating-samples)
    *   [Benchmarking](#benchmarking)
8.  [Configuration Parameters](#configuration-parameters)
9.  [Output Structure](#output-structure)
10. [Logging](#logging)
11. [Future Work & Contributions](#future-work--contributions)

## Overview

Denoising Diffusion Probabilistic Models (DDPMs) have shown remarkable success in generating high-fidelity images. This project provides a robust and highly optimized implementation tailored for conditional image generation, particularly aimed at scenarios requiring efficient use of high-end GPU resources, such as medical image synthesis.

The core idea is to train a model (typically a UNet) to denoise an image gradually from pure noise to a clean image over a series of timesteps. By conditioning this process on class labels, we can guide the generation towards specific types of images. This implementation incorporates numerous optimizations to accelerate training, reduce memory footprint, and improve overall efficiency.

## Key Features

*   **Conditional DDPM:** Generates images conditioned on class labels.
*   **Optimized UNet Architecture:**
    *   Residual Blocks with Group Normalization and SiLU activations.
    *   Efficient Attention Blocks leveraging `F.scaled_dot_product_attention` (enabling Flash Attention if PyTorch and hardware support it).
    *   Sinusoidal Position Embeddings for time steps.
    *   Learnable class embeddings for conditioning.
    *   Increased base channel capacity (`base_channels=128`) for potentially higher quality.
*   **Classifier-Free Guidance (CFG):** Allows for adjustable guidance strength during sampling to trade off sample diversity for quality. Includes CFG dropout during training.
*   **Performance Optimizations:**
    *   **Automatic Mixed Precision (AMP):** Utilizes `torch.cuda.amp` for faster training with reduced memory usage (enabled by default).
    *   **`torch.compile`:** Leverages PyTorch 2.0+ model compilation (`mode='max-autotune'`) for significant speedups.
    *   **Optimized CUDA & cuDNN Settings:** Includes `torch.backends.cuda.enable_flash_sdp(True)`, `torch.backends.cudnn.benchmark = True`, and TF32 support for faster computations.
    *   **Efficient Memory Management:** Sets `PYTORCH_CUDA_ALLOC_CONF` to optimize GPU memory allocation.
    *   **Gradient Accumulation:** Simulates larger batch sizes to improve model performance without increasing memory proportionally.
    *   **Fast Exponential Moving Average (EMA):** Custom, efficient EMA implementation for model weights, often leading to better sample quality.
    *   **Optimized DataLoader:**
        *   Adjustable number of workers (defaults to `min(16, mp.cpu_count())`).
        *   `pin_memory=True` for faster CPU-to-GPU data transfers.
        *   `persistent_workers=True` and `prefetch_factor` to reduce data loading overhead.
        *   Optional in-memory dataset caching (`--cache_dataset`).
    *   **JIT Compilation:** Key functions like `extract` are JIT-compiled for speed.
    *   **Efficient Batch Inference for CFG:** Processes conditional and unconditional predictions in a single batch during sampling.
*   **Flexible Beta Schedules:** Supports both `linear` and `cosine` beta schedules (cosine often preferred).
*   **Advanced Learning Rate Scheduling:** Uses `OneCycleLR` with cosine annealing for robust training.
*   **Optimized AdamW Optimizer:** Tuned `betas` and `eps` for potentially better stability and performance.
*   **Hardware-Aware Settings:** Automatically checks GPU/CPU specs and optimizes PyTorch CPU threads.
*   **Comprehensive Training Loop:** Includes progress bars, loss tracking, checkpointing, and sample generation during training.
*   **Sample Generation Mode:** Allows loading a trained model to generate new samples.
*   **Built-in Benchmarking:** Includes a function to benchmark the forward pass throughput of the model.
*   **Robust Logging:** Logs training progress and system information to console and `training.log`.

## Architectural Highlights

### Optimized UNet

The backbone of the denoising process is an `OptimizedUNet` model. It features:
*   **Initial Convolution:** A 7x7 convolution to process the input image.
*   **Time and Class Embeddings:**
    *   `SinusoidalPositionEmbeddings` for timesteps, processed through an MLP.
    *   Learnable `nn.Embedding` for class labels, with a dedicated `null_class_emb` for CFG.
    *   These embeddings are concatenated and used to condition the `ResidualBlock`s.
*   **Downsampling Path:**
    *   Consists of `ResidualBlock`s (GroupNorm, SiLU, Conv2D) and `EfficientAttentionBlock`s (for deeper layers).
    *   Downsampling is performed by strided convolutions.
    *   Skip connections are maintained for the upsampling path.
*   **Middle Block:** Contains `ResidualBlock`s and an `EfficientAttentionBlock` at the bottleneck.
*   **Upsampling Path:**
    *   Uses `ConvTranspose2d` for upsampling.
    *   Concatenates skip connections from the downsampling path.
    *   Consists of `ResidualBlock`s and `EfficientAttentionBlock`s.
*   **Output:** A final GroupNorm, SiLU, and 3x3 convolution to predict the noise.
*   **Weight Initialization:** Kaiming Normal for convolutional layers and Normal distribution for linear layers.

The `EfficientAttentionBlock` utilizes PyTorch's `scaled_dot_product_attention`, which can automatically use Flash Attention if available, providing significant speedups and memory savings for the attention mechanism.

### Diffusion Process & Classifier-Free Guidance

The `OptimizedDiffusion` class manages the diffusion mathematics:
*   **Beta Schedules:** Precomputes constants (`alphas`, `alphas_cumprod`, etc.) based on the chosen beta schedule (`linear` or `cosine`).
*   **Forward Process (`q_sample`):** Adds noise to an image `x_start` at a given timestep `t`. Optimized with precomputed values.
*   **Training Loss (`p_losses`):**
    1.  Takes a clean image `x_start`.
    2.  Randomly samples a timestep `t`.
    3.  Adds noise to `x_start` to get `x_noisy` using `q_sample`.
    4.  **CFG Dropout:** With a probability `cfg_dropout`, class labels are masked (set to `None`) during training. This forces the model to learn both conditional and unconditional noise prediction.
    5.  Predicts the noise from `x_noisy` at timestep `t` using the UNet model (conditioned on `t` and potentially class labels).
    6.  Computes the Mean Squared Error (MSE) between the true added noise and the predicted noise.
*   **Reverse Process (`p_sample` - DDPM Sampler):**
    1.  Given a noisy image `x` at timestep `t`, predicts the noise using the UNet.
    2.  **Classifier-Free Guidance:** If `guidance_scale > 1.0` and class labels are provided:
        *   The model performs two forward passes: one with the provided `class_labels` (conditional prediction) and one with null/unconditional labels.
        *   The final noise prediction is extrapolated: `noise_uncond + guidance_scale * (noise_cond - noise_uncond)`. This is done efficiently by batching the two inferences.
    3.  Calculates the mean of the denoised image.
    4.  Adds stochastic noise (scaled by posterior variance) unless it's the final timestep.
*   **Sampling Loop (`sample`):** Iteratively applies `p_sample` from `T-1` down to `0` to generate an image from pure noise.

### Performance Optimizations

The project is heavily focused on speed and efficiency:
*   **`torch.compile(model, mode='max-autotune')`:** Compiles the UNet model for significant speedups on compatible PyTorch versions and hardware.
*   **Automatic Mixed Precision (AMP):** `torch.cuda.amp.autocast` and `GradScaler` reduce memory and accelerate training by using FP16 for eligible operations.
*   **Flash Attention:** `torch.backends.cuda.enable_flash_sdp(True)` and `F.scaled_dot_product_attention` in `EfficientAttentionBlock` enable memory-efficient and fast attention.
*   **Optimized PyTorch Backend Settings:** `cudnn.benchmark = True`, TF32 allowed for matmuls and cuDNN operations.
*   **Fast EMA:** `FastEMAModel` updates shadow weights efficiently without redundant model iteration.
*   **DataLoader Optimizations:** `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`, and optional `cache_dataset` minimize data loading bottlenecks.
*   **Gradient Accumulation:** Allows training with effectively larger batch sizes than GPU memory might permit directly.
*   **Memory Allocation:** `PYTORCH_CUDA_ALLOC_CONF = 'max_split_size_mb:128'` can help manage memory fragmentation.
*   **CPU Optimization:** `torch.set_num_threads` optimizes CPU usage.

## Dataset

The current implementation uses an `OptimizedMedicalMNIST` dataset class, which is a wrapper around `torchvision.datasets.MNIST`.
*   It resizes MNIST digits to the specified `img_size`.
*   It normalizes images to `[-1, 1]`.
*   It supports an option (`--cache_dataset`) to load and preprocess the entire dataset into RAM for extremely fast access during training, suitable if system RAM is ample.

While MNIST is used here, the framework is designed to be adaptable to other 2D medical image datasets by modifying the `OptimizedMedicalMNIST` class or creating a new `Dataset` implementation.

## Prerequisites

*   Python 3.8+
*   PyTorch 2.0+ (recommended for `torch.compile` and Flash Attention support)
*   torchvision
*   tqdm (for progress bars)
*   psutil (for hardware monitoring)
*   A CUDA-enabled GPU is highly recommended for reasonable performance.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Install PyTorch according to your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/). For example:
    ```bash
    # For CUDA 11.8
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    # Or for CUDA 12.1
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
    Then install other packages:
    ```bash
    pip install tqdm psutil
    ```
    (Consider creating a `requirements.txt` file for easier installation).

4.  **Data:**
    The MNIST dataset will be automatically downloaded to the directory specified by `--data_dir` (default: `./data`) if it's not already present.

## Usage

The main script is `main.py` (assuming your file is named `main.py`).

### Training

To start training with default parameters (optimized for an NVIDIA A5000-like GPU):
```bash
python main.py


To customize training, use the available command-line arguments. For example:

python main.py \
    --data_dir ./datasets/my_medical_data \
    --batch_size 32 \
    --img_size 256 \
    --epochs 200 \
    --lr 5e-5 \
    --guidance_scale 5.0 \
    --num_classes 5 \
    --save_dir ./results_experiment1 \
    --beta_schedule cosine \
    --compile_model \
    --use_amp \
    --gradient_accumulation_steps 2 \
    --cache_dataset
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Note on torch.compile: If you are using PyTorch 2.0+, the --compile_model flag will significantly speed up training after an initial compilation phase. This is highly recommended.

Note on AMP: --use_amp is enabled by default. To disable it (e.g., for debugging or if issues arise on specific hardware), you would need to modify the parse_args default or add a --no-use_amp argument. (The current code has default=True for use_amp, so it's always on unless explicitly made configurable to be off).

The script will first run a quick benchmark of the model's forward pass.

Resuming Training

To resume training from a saved checkpoint:

python main.py --resume ./output/checkpoints/checkpoint_epoch_X.pth
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Replace X with the epoch number of the checkpoint. Other training parameters should ideally match the original training run, or be set as desired for continued training.

Generating Samples

To generate samples using a trained model (e.g., best_model.pth or final_model.pth):

python main.py \
    --sample_only \
    --resume ./output/best_model.pth \
    --save_dir ./output \
    --img_size 128 \
    --num_classes 10 \
    --guidance_scale 7.5
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

This will generate a grid of samples for each class and unconditional samples, saving them to ./output/final_samples/.
Ensure --img_size, --num_classes, and other model-defining parameters match the loaded checkpoint.

Benchmarking

The model benchmark (forward pass throughput) is automatically run at the beginning of a training session (if not --sample_only). There isn't a separate CLI flag to only run the benchmark, but the benchmark_model function could be called independently if needed.

Configuration Parameters

Key command-line arguments (see python main.py --help for all options):

--data_dir (str, default: ./data): Path to the dataset.

--batch_size (int, default: 64): Batch size. Optimized for A5000 with default settings.

--img_size (int, default: 128): Image size for training and generation.

--timesteps (int, default: 1000): Number of diffusion timesteps.

--epochs (int, default: 100): Number of training epochs.

--lr (float, default: 1e-4): Learning rate.

--device (str, default: cuda if available, else cpu): Device to use.

--guidance_scale (float, default: 7.5): Classifier-Free Guidance scale for sampling.

--cfg_dropout (float, default: 0.1): Probability of dropping class labels during CFG training.

--num_classes (int, default: 10): Number of classes for conditional generation.

--save_dir (str, default: ./output): Directory to save outputs (checkpoints, samples).

--beta_schedule (str, default: cosine, choices: ['linear', 'cosine']): Beta schedule type.

--resume (str, help: Path to checkpoint to resume from):

--sample_only (action, default: False): If set, only generate samples (requires --resume).

--num_workers (int, default: min(16, mp.cpu_count())): Number of DataLoader workers.

--save_freq (int, default: 5): Save checkpoint and samples every N epochs.

--compile_model (action, default: False): Use torch.compile for optimization (PyTorch 2.0+).

--use_amp (action, default: True): Use Automatic Mixed Precision.

--gradient_accumulation_steps (int, default: 1): Number of steps to accumulate gradients over.

--cache_dataset (action, default: False): Cache the entire dataset in RAM.

--prefetch_factor (int, default: 4): DataLoader prefetch factor.

--pin_memory (action, default: True): Pin memory for DataLoader.

Output Structure

All outputs are saved in the directory specified by --save_dir (default: ./output):

./output/training.log: Log file containing training progress and system information.

./output/checkpoints/: Saved model checkpoints during training (e.g., checkpoint_epoch_X.pth).

./output/samples/: Grid of generated image samples saved periodically during training (e.g., epoch_X.png).

./output/best_model.pth: Model state dictionary of the epoch with the best validation loss (using EMA weights).

./output/final_model.pth: Model state dictionary of the final trained model (using EMA weights).

./output/final_samples/ (if --sample_only is used):

class_X.png: Grid of samples generated for class X.

unconditional.png: Grid of unconditionally generated samples.

progression.png: Grid showing the denoising progression for a batch of samples.

Logging

Comprehensive logging is set up using the logging module:

Messages are printed to the console.

All messages are also saved to training.log within the save_dir.

Logs include timestamps, log level, and messages covering arguments, hardware specs, optimization settings, epoch progress, loss, learning rate, and memory usage.

Future Work & Contributions

Distributed Data Parallel (DDP) Training: While DDP is imported, full multi-GPU training setup using torch.distributed is not explicitly implemented in the main training function. This would be a valuable addition for scaling to larger models/datasets.

More Sophisticated Schedulers/Samplers: Explore advanced DDPM samplers like DDIM, DPM-Solver, etc., for faster/higher-quality sampling.

Support for Custom Medical Datasets: Provide clearer examples or template for integrating diverse medical imaging datasets (e.g., DICOM, NIfTI).

Evaluation Metrics: Integrate quantitative evaluation metrics for image quality and diversity (e.g., FID, IS, Precision/Recall).

Hyperparameter Tuning Framework: Integrate tools like Optuna or Ray Tune for systematic hyperparameter optimization.

Contributions are welcome! Please feel free to open an issue or submit a pull request.

IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
