import torch
import torch.nn as nn

class SpatiotemporalMasker(nn.Module):
    """
    Spatiotemporal Tube Masker.
    
    Implements a spatiotemporal masking policy for training video JEPA architectures.
    Instead of masking random patches, it masks the same spatial coordinates across
    the entire temporal axis (a "tube" mask). This forces the model to actually learn
    predictive dynamics rather than trivial temporal interpolation.
    
    Attributes:
        mask_ratio: Fraction of patches to mask (e.g. 0.90 / 90% masking).
        t_size: Temporal dimension of patches (e.g., T/patch_t = 8).
        h_size: Height dimension of patches (e.g., H/patch_h = 14).
        w_size: Width dimension of patches (e.g., W/patch_w = 14).
    """
    def __init__(self, mask_ratio: float = 0.9, t_size: int = 8, h_size: int = 14, w_size: int = 14):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.t_size = t_size
        self.h_size = h_size
        self.w_size = w_size
        self.num_spatial_patches = h_size * w_size
        self.total_patches = t_size * h_size * w_size
        
    def generate_tube_mask(self, batch_size: int, device: torch.device) -> tuple:
        """
        Generates a spatiotemporal tube mask for a batch of videos.
        
        Returns:
            keep_indices: Boolean mask of shape [B, total_patches] where True indicates patches
                          to KEEP (unmasked), and False indicates patches to MASK.
            mask_indices: Inverse boolean mask.
        """
        # Determine how many spatial positions to keep
        num_spatial_keep = int(self.num_spatial_patches * (1.0 - self.mask_ratio))
        num_spatial_keep = max(1, num_spatial_keep)
        
        keep_indices = torch.zeros(batch_size, self.total_patches, dtype=torch.bool, device=device)
        
        for b in range(batch_size):
            # 1. Randomly sample spatial positions to keep
            spatial_perm = torch.randperm(self.num_spatial_patches, device=device)
            keep_spatial_slots = spatial_perm[:num_spatial_keep] # [num_spatial_keep]
            
            # 2. Replicate this spatial keep mask across all temporal slices
            # We map spatial index 's' to patch index 't * num_spatial_patches + s'
            batch_keep = torch.zeros(self.total_patches, dtype=torch.bool, device=device)
            
            for t in range(self.t_size):
                offset = t * self.num_spatial_patches
                batch_keep[offset + keep_spatial_slots] = True
                
            keep_indices[b] = batch_keep
            
        mask_indices = ~keep_indices
        return keep_indices, mask_indices
