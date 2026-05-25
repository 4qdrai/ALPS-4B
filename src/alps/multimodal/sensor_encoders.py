import torch
import torch.nn as nn
import torch.nn.functional as F

class IMUEncoder(nn.Module):
    """
    Lightweight 1D CNN Encoder for Inertial Measurement Unit (IMU) streams.
    Transforms raw [B, channels, length] signals (e.g. 6 axis accel/gyro)
    into the shared latent space [B, N, D_model].
    """
    def __init__(self, in_channels: int = 6, d_model: int = 384, seq_len: int = 100):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LayerNorm(128),
            nn.GELU()
        )
        
        # Calculate compressed length after two stride-2 convolutions
        compressed_len = ((seq_len + 1) // 2 + 1) // 2
        self.fc = nn.Linear(128 * compressed_len, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw IMU signals, Shape: [B, C_imu, Seq_len]
            
        Returns:
            latent: Latent embeddings, Shape: [B, 1, D_model]
        """
        h = self.conv(x) # [B, 128, compressed_len]
        h = h.flatten(1) # [B, 128 * compressed_len]
        out = self.fc(h).unsqueeze(1) # [B, 1, D_model]
        return out

class LidarEncoder(nn.Module):
    """
    1D CNN Encoder for LiDAR range scans (e.g., 2D radial scans).
    Transforms raw scans [B, channels, scan_beams] into the shared latent space [B, N, D_model].
    """
    def __init__(self, in_channels: int = 1, d_model: int = 384, num_beams: int = 360):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=3, padding=3),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LayerNorm(128),
            nn.GELU()
        )
        
        # Calculate size
        h_size = num_beams
        h_size = (h_size + 2) // 3
        h_size = (h_size + 1) // 2
        
        self.fc = nn.Linear(128 * h_size, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw LiDAR beam distances, Shape: [B, C_lidar, Num_beams]
            
        Returns:
            latent: Latent embeddings, Shape: [B, 1, D_model]
        """
        h = self.conv(x)
        h = h.flatten(1)
        out = self.fc(h).unsqueeze(1) # [B, 1, D_model]
        return out
