import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentRAG(nn.Module):
    """
    Latent-RAG: Non-Parametric Key-Value Cache.
    Retrieves episodic correction vectors (delta z) from an in-memory KV cache 
    based on semantic similarity during the forward pass, facilitating
    "No-Retraining" instantaneous learning.
    
    Includes methods for reading from, writing to, and auditing (deleting) memory records.
    """
    def __init__(self, d_model: int = 384, max_size: int = 5000, sim_threshold: float = 0.7):
        super().__init__()
        self.d_model = d_model
        self.max_size = max_size
        self.sim_threshold = sim_threshold
        
        # In-memory stores for keys (contexts) and values (corrections)
        # registered as buffers or simply standard tensors that we manage manually
        self.register_buffer("keys", torch.zeros(max_size, d_model))
        self.register_buffer("values", torch.zeros(max_size, d_model))
        self.register_buffer("usage_counts", torch.zeros(max_size, dtype=torch.long))
        
        # We track how many slots are filled
        self.register_buffer("current_size", torch.tensor(0, dtype=torch.long))
        
    def write_memory(self, key: torch.Tensor, value: torch.Tensor):
        """
        Writes a new experience key-value pair to the cache.
        
        Args:
            key: Semantic context vector, Shape: [D] or [1, D]
            value: Latent correction vector (delta z), Shape: [D] or [1, D]
        """
        key = key.squeeze().detach()
        value = value.squeeze().detach()
        
        size = self.current_size.item()
        
        # If cache is full, evict the least used item (LFU eviction policy)
        if size >= self.max_size:
            evict_idx = torch.argmin(self.usage_counts)
            self.keys[evict_idx] = key
            self.values[evict_idx] = value
            self.usage_counts[evict_idx] = 0
        else:
            self.keys[size] = key
            self.values[size] = value
            self.usage_counts[size] = 0
            self.current_size += 1
            
    def retrieve_correction(self, q: torch.Tensor) -> torch.Tensor:
        """
        Queries the RAG cache using current latent q to fetch matching correction vectors.
        
        Args:
            q: Query latents, Shape: [B, N, D]
            
        Returns:
            correction: Accumulated correction vectors, Shape: [B, N, D]
        """
        size = self.current_size.item()
        if size == 0:
            return torch.zeros_like(q)
            
        B, N, D = q.shape
        flat_q = q.reshape(-1, D) # [B*N, D]
        
        # Normalize queries and keys for cosine similarity
        norm_q = F.normalize(flat_q, p=2, dim=-1) # [B*N, D]
        norm_keys = F.normalize(self.keys[:size], p=2, dim=-1) # [size, D]
        
        # Compute cosine similarity matrix: [B*N, size]
        sim = torch.matmul(norm_q, norm_keys.t())
        
        # Identify matches exceeding the retrieval threshold
        # For simplicity, we can do a weighted combination of matching corrections
        mask = (sim >= self.sim_threshold).float() # [B*N, size]
        
        if not mask.any():
            return torch.zeros_like(q)
            
        # Weighted similarity (scores)
        scores = sim * mask # [B*N, size]
        sum_scores = scores.sum(dim=-1, keepdim=True) + 1e-8 # [B*N, 1]
        weights = scores / sum_scores # [B*N, size]
        
        # Gather matching correction values
        # weights: [B*N, size], values: [size, D] -> [B*N, D]
        retrieved_flat = torch.matmul(weights, self.values[:size])
        
        # Update usage statistics for matched keys (only in training/active execution)
        if self.training:
            # Add hits to keys that had similarity >= sim_threshold
            matched_indices = torch.nonzero(mask) # [Num_matches, 2] -> cols are database row indices
            if matched_indices.shape[0] > 0:
                unique_rows, counts = torch.unique(matched_indices[:, 1], return_counts=True)
                self.usage_counts[unique_rows] += counts
                
        correction = retrieved_flat.reshape(B, N, D)
        return correction
        
    def delete_memory(self, index: int):
        """Audits and purges a bad habit/faulty entry from episodic memory by index."""
        size = self.current_size.item()
        if index < 0 or index >= size:
            raise IndexError("Index out of bounds for RAG cache.")
            
        # Shift everything after the deleted index to the left
        if index < size - 1:
            self.keys[index:size-1] = self.keys[index+1:size]
            self.values[index:size-1] = self.values[index+1:size]
            self.usage_counts[index:size-1] = self.usage_counts[index+1:size]
            
        # Zero out the last active slot
        self.keys[size-1] = 0.0
        self.values[size-1] = 0.0
        self.usage_counts[size-1] = 0
        self.current_size -= 1
        
    def clear(self):
        """Completely purges the episodic memory database."""
        self.keys.zero_()
        self.values.zero_()
        self.usage_counts.zero_()
        self.current_size.zero_()
