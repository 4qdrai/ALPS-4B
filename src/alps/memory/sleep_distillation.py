import torch
import torch.nn as nn
import torch.optim as optim

class SleepConsolidation(nn.Module):
    """
    Sleep Consolidation & Distillation Scheduler.
    
    Overnight (or periodic offline cycles), the system consolidates short-term episodic
    corrections into long-term parametric neural network weights.
    
    Process:
    1. Audits the Latent-RAG KV cache to extract the most frequently accessed corrections.
    2. Runs a training loop to distill these corrections into the core Predictor's weights
       by minimizing: MSE(Predictor(key) - (key + delta_z)).
    3. Purges consolidated items from the RAG cache to maintain VRAM efficiency.
    """
    def __init__(self, lr: float = 1e-4, epochs: int = 5, use_frequency_threshold: int = 2):
        super().__init__()
        self.lr = lr
        self.epochs = epochs
        self.use_frequency_threshold = use_frequency_threshold
        
    def consolidate(self, predictor: nn.Module, rag: nn.Module) -> dict:
        """
        Consolidates frequent RAG corrections into the predictor weights.
        
        Args:
            predictor: Predictor network (e.g. MultiScalePredictor)
            rag: Latent-RAG memory module
            
        Returns:
            metrics: Dictionary of consolidation statistics.
        """
        size = rag.current_size.item()
        if size == 0:
            return {"consolidated_count": 0, "initial_loss": 0.0, "final_loss": 0.0}
            
        device = rag.keys.device
        
        # 1. Identify frequently used records
        # Filter keys that exceed the utilization threshold
        frequent_mask = rag.usage_counts[:size] >= self.use_frequency_threshold
        num_to_consolidate = frequent_mask.sum().item()
        
        if num_to_consolidate == 0:
            return {"consolidated_count": 0, "initial_loss": 0.0, "final_loss": 0.0}
            
        # Extract keys, values (delta z), and indices
        keys_to_train = rag.keys[:size][frequent_mask].detach() # [M, D]
        deltas_to_train = rag.values[:size][frequent_mask].detach() # [M, D]
        
        # In RAG, target is key + delta_z
        targets = keys_to_train + deltas_to_train # [M, D]
        
        # Set up a small optimizer
        optimizer = optim.AdamW(predictor.parameters(), lr=self.lr)
        
        # Convert to batch [M, 1, D] so each memory is an independent batch item
        # with sequence length 1. This prevents transformer self-attention from mixing
        # different memories during distillation (cross-contamination prevention).
        batch_keys = keys_to_train.unsqueeze(1)    # [M, 1, D]
        batch_targets = targets.unsqueeze(1)        # [M, 1, D]
        
        # Standard JEPA prediction setup:
        # Predictor takes state and a conditioning signal (we query d_cond dynamically to support all layers)
        d_cond = predictor.cond_proj[0].in_features
        cond = torch.zeros(num_to_consolidate, d_cond, device=device)  # [M, d_cond]
        
        initial_loss = 0.0
        final_loss = 0.0
        
        # 2. Rehearsal/Distillation Training Loop
        predictor.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Forward prediction
            predictions = predictor(batch_keys, cond) # [1, M, D]
            
            # Loss: MSE between prediction and consolidated target
            loss = F_mse(predictions, batch_targets)
            
            if epoch == 0:
                initial_loss = loss.item()
                
            loss.backward()
            optimizer.step()
            
            final_loss = loss.item()
            
        # 3. Purge/Audit memory rows
        # Remove items from RAG since their patterns are now memorized parametrically.
        # We iterate in reverse to avoid index shifting issues during deletion
        indices = torch.nonzero(frequent_mask).flatten()
        for idx in reversed(indices.tolist()):
            rag.delete_memory(idx)
            
        return {
            "consolidated_count": num_to_consolidate,
            "initial_loss": initial_loss,
            "final_loss": final_loss
        }

def F_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a - b) ** 2)
