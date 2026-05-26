import torch
import torch.nn as nn
import torch.nn.functional as F

class InverseMonitor(nn.Module):
    """
    Inverse Monitoring Loop (Efference Copy Verification).
    
    Continually compares the predictor's simulated latent state (efference copy)
    against the actual encoded latent state. A significant divergence indicates
    an unexpected environmental surprise or prediction failure, which triggers
    an interrupt to pause lower execution and escalate to higher layers for replanning.
    """
    def __init__(self, threshold: float = 0.5, history_len: int = 10):
        super().__init__()
        self.threshold = threshold
        self.history_len = history_len
        
        # Track historical divergence scores for running average baseline
        self.register_buffer("divergence_history", torch.zeros(history_len))
        self.register_buffer("history_pointer", torch.tensor(0, dtype=torch.long))
        self.register_buffer("is_warm", torch.tensor(False, dtype=torch.bool))
        
    def forward(self, predicted_z: torch.Tensor, actual_z: torch.Tensor) -> tuple:
        """
        Args:
            predicted_z: Simulated target latents, Shape: [B, N, D]
            actual_z: True encoder latents, Shape: [B, N, D]
            
        Returns:
            divergence: Scalar divergence value.
            triggered: Boolean indicating whether surprise threshold was breached.
        """
        # Calculate Mean Squared Error as divergence score
        # Note: We compute distance normalized by latent dimension D
        B, N, D = actual_z.shape
        diff = predicted_z - actual_z
        divergence = torch.sum(diff ** 2) / (B * N * D)
        
        # Adaptive Decision Trigger (check BEFORE updating history):
        # System 2 is invoked if the current divergence is significantly higher than 
        # the running average (a sudden spike in prediction error).
        running_mean = self.get_running_average()
        # If history is empty, rely on static threshold, otherwise use adaptive multiplier
        if running_mean == 0.0:
            triggered = (divergence.item() > self.threshold)
        else:
            # Trigger if current error spikes above (1 + threshold) * running mean
            # e.g., if threshold is 0.5, triggers when error is 1.5x higher than average
            triggered = (divergence.item() > ((1.0 + self.threshold) * running_mean))
        
        # Update history buffer AFTER the check
        ptr = self.history_pointer.item()
        self.divergence_history[ptr] = divergence.detach()
        
        next_ptr = (ptr + 1) % self.history_len
        self.history_pointer.copy_(torch.tensor(next_ptr, device=self.history_pointer.device))
        
        if next_ptr == 0:
            self.is_warm.copy_(torch.tensor(True, device=self.is_warm.device))
        
        return divergence, triggered
        
    def get_running_average(self) -> float:
        """Returns the average divergence score of recent history."""
        if not self.is_warm.item():
            ptr = self.history_pointer.item()
            if ptr == 0:
                return 0.0
            return self.divergence_history[:ptr].mean().item()
        return self.divergence_history.mean().item()
        
    def set_threshold(self, new_threshold: float):
        """Allows dynamically adjusting sensitivity based on task difficulty."""
        self.threshold = new_threshold
