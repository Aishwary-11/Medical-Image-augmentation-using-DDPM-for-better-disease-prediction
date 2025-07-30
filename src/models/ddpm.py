import torch
import torch.nn as nn
from tqdm import tqdm
import math

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


class TimeEmbeddedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, num_groups=8):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)
        )

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act2 = nn.SiLU()

        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        hidden_state = self.norm1(self.conv1(x))
        time_params = self.time_mlp(t_emb)
        time_params = time_params.unsqueeze(-1).unsqueeze(-1)
        scale, shift = torch.chunk(time_params, 2, dim=1)
        h = hidden_state * (1 + scale) + shift
        h = self.act2(self.norm2(self.conv2(h)))
        return h + self.res_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_dim=256, base_channels=64):
        super().__init__()

        self.time_embed = SinusoidalPositionEmbeddings(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        ch_mults = (1, 2, 4, 4)
        self.enc1 = TimeEmbeddedConvBlock(in_channels, base_channels * ch_mults[0], time_dim)
        self.enc2 = TimeEmbeddedConvBlock(base_channels * ch_mults[0], base_channels * ch_mults[1], time_dim)
        self.enc3 = TimeEmbeddedConvBlock(base_channels * ch_mults[1], base_channels * ch_mults[2], time_dim)
        self.enc4 = TimeEmbeddedConvBlock(base_channels * ch_mults[2], base_channels * ch_mults[3], time_dim)

        self.bottleneck = TimeEmbeddedConvBlock(base_channels * ch_mults[3], base_channels * 8, time_dim)

        self.dec4 = TimeEmbeddedConvBlock(base_channels * 8 + base_channels * ch_mults[3], base_channels * ch_mults[3], time_dim)
        self.dec3 = TimeEmbeddedConvBlock(base_channels * ch_mults[3] + base_channels * ch_mults[2], base_channels * ch_mults[2], time_dim)
        self.dec2 = TimeEmbeddedConvBlock(base_channels * ch_mults[2] + base_channels * ch_mults[1], base_channels * ch_mults[1], time_dim)
        self.dec1 = TimeEmbeddedConvBlock(base_channels * ch_mults[1] + base_channels * ch_mults[0], base_channels * ch_mults[0], time_dim)

        self.final_conv = nn.Conv2d(base_channels * ch_mults[0], out_channels, 1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)
        s1 = self.enc1(x, t_emb)
        s2 = self.enc2(self.pool(s1), t_emb)
        s3 = self.enc3(self.pool(s2), t_emb)
        s4 = self.enc4(self.pool(s3), t_emb)
        b = self.bottleneck(self.pool(s4), t_emb)
        d4 = self.dec4(torch.cat([self.upsample(b), s4], dim=1), t_emb)
        d3 = self.dec3(torch.cat([self.upsample(d4), s3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.upsample(d3), s2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.upsample(d2), s1], dim=1), t_emb)
        return self.final_conv(d1)


class DDPM:
    def __init__(self, model, device, num_timesteps=1000):
        self.model = model
        self.device = device
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        alpha_bar_prev = torch.cat([torch.tensor([1.0], device=device), self.alpha_bars[:-1]])
        self.posterior_variance = self.betas * (1. - alpha_bar_prev) / (1. - self.alpha_bars)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))

    def add_noise(self, x_start, t):
        noise = torch.randn_like(x_start)
        sqrt_alpha_bar_t = self.sqrt_alpha_bars.gather(0, t).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars.gather(0, t).view(-1, 1, 1, 1)
        noisy_x = sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise
        return noisy_x, noise

    def train_step(self, x_start):
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device, dtype=torch.long)
        noisy_x, noise_gt = self.add_noise(x_start, t)
        predicted_noise = self.model(noisy_x, t)
        loss = nn.MSELoss()(predicted_noise, noise_gt)
        return loss

    @torch.no_grad()
    def p_sample(self, x_t, t_idx):
        t_tensor = torch.full((x_t.shape[0],), t_idx, device=self.device, dtype=torch.long)
        betas_t = self.betas.gather(0, t_tensor).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bars_t = self.sqrt_one_minus_alpha_bars.gather(0, t_tensor).view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas.gather(0, t_tensor).view(-1, 1, 1, 1)
        predicted_noise = self.model(x_t, t_tensor)
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alpha_bars_t)
        if t_idx == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance.gather(0, t_tensor).view(-1, 1, 1, 1)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, num_samples, image_size_tuple):
        H, W = image_size_tuple
        x_t = torch.randn(num_samples, 1, H, W, device=self.device)
        for i in tqdm(reversed(range(self.num_timesteps)), desc="Sampling", total=self.num_timesteps):
            x_t = self.p_sample(x_t, i)
        x_t = (x_t + 1.0) / 2.0
        return torch.clamp(x_t, 0.0, 1.0)
