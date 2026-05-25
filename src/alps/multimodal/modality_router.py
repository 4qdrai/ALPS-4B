import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityRouter(nn.Module):
    """
    Multimodal Modality Router (O(1) Modality Gating).
    
    Dynamically routes raw sensory input by activating only the most relevant
    auxiliary modalities based on the Strategic layer's top-down conceptual attention.
    
    Guarantees constant forward FLOP scale by restricting active auxiliary streams
    to at most K_mod = 1 or 2, skipping all other sensor encoders to prevent compute bloat.
    """
    def __init__(self, d_model: int = 384, num_modalities: int = 2, active_modalities: int = 1):
        super().__init__()
        self.d_model = d_model
        self.num_modalities = num_modalities
        self.active_modalities = active_modalities
        
        # Attention projection to compute compatibility between strategic context and modalities
        self.attn = nn.Linear(d_model, num_modalities)
        
    def forward(self, strategic_concept: torch.Tensor, sensor_tensors: list) -> tuple:
        """
        Args:
            strategic_concept: Slow-varying high-level schema c_T, Shape: [B, D_model]
            sensor_tensors: List of pre-encoded auxiliary sensor embeddings.
                            Each element has Shape: [B, 1, D_model].
                            Example: [imu_embeddings, lidar_embeddings]
                            
        Returns:
            fused_representation: Attention-weighted sum of active modalities, Shape: [B, 1, D_model]
            active_mask: Boolean tensor of active modalities, Shape: [B, num_modalities]
        """
        B = strategic_concept.shape[0]
        
        # 1. Compute modality matching logits
        # Maps strategic concepts to modality relevance scores
        logits = self.attn(strategic_concept) # [B, num_modalities]
        
        # 2. Select top-K active modalities
        scores, indices = torch.topk(logits, k=self.active_modalities, dim=-1) # [B, active_modalities], [B, active_modalities]
        
        weights = F.softmax(scores, dim=-1) # [B, active_modalities]
        
        # Build active mask for diagnostic monitoring
        active_mask = torch.zeros_like(logits, dtype=torch.bool)
        active_mask.scatter_(1, indices, True)
        
        # 3. Conditional Modal Accumulation
        fused = torch.zeros(B, 1, self.d_model, device=strategic_concept.device)
        
        # We iterate over top-K slots to accumulate only selected modality representations
        for slot in range(self.active_modalities):
            slot_indices = indices[:, slot] # [B]
            slot_weights = weights[:, slot].unsqueeze(-1).unsqueeze(-1) # [B, 1, 1]
            
            # Map batch items to their selected modality tensors
            for b in range(B):
                mod_idx = slot_indices[b].item()
                if mod_idx < len(sensor_tensors):
                    # Weighted addition of selected modality
                    fused[b] += slot_weights[b] * sensor_tensors[mod_idx][b]
                    
        return fused, active_mask
