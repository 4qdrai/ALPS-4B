import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticVectorDatabase(nn.Module):
    """
    Strategic Permanent Semantic Vector Database (System 2 Long-term Memory).
    
    Stores permanent conceptual knowledge and schemas.
    Allows searching and retrieving semantic representations using high-dimensional
    concept signatures via cosine similarity.
    """
    def __init__(self, d_model: int = 384, max_concepts: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.max_concepts = max_concepts
        
        # Buffers for storing semantic signatures and attributes
        self.register_buffer("signatures", torch.zeros(max_concepts, d_model))
        self.register_buffer("concepts", torch.zeros(max_concepts, d_model))
        self.register_buffer("current_count", torch.tensor(0, dtype=torch.long))
        
    def add_concept(self, signature: torch.Tensor, concept: torch.Tensor):
        """
        Adds a new conceptual record to long-term memory.
        
        Args:
            signature: Semantic query signature, Shape: [D_model]
            concept: Conceptual schema vector, Shape: [D_model]
        """
        device = self.signatures.device
        signature = signature.squeeze().detach().to(device)
        concept = concept.squeeze().detach().to(device)
        
        count = self.current_count.item()
        
        if count < self.max_concepts:
            self.signatures[count] = signature
            self.concepts[count] = concept
            self.current_count += 1
        else:
            # Random eviction if full (or replace oldest)
            evict_idx = int(torch.randint(0, self.max_concepts, (1,)).item())
            self.signatures[evict_idx] = signature
            self.concepts[evict_idx] = concept
            
    def query_concepts(self, q: torch.Tensor, top_k: int = 1) -> tuple:
        """
        Queries the database for closest semantic concepts.
        
        Args:
            q: Query signature, Shape: [B, D_model]
            
        Returns:
            retrieved_concepts: Closest concept representations, Shape: [B, top_k, D_model]
            scores: Cosine similarities, Shape: [B, top_k]
        """
        count = self.current_count.item()
        if count == 0:
            return torch.zeros(q.shape[0], top_k, self.d_model, device=q.device), \
                   torch.zeros(q.shape[0], top_k, device=q.device)
                   
        B = q.shape[0]
        
        # Normalize queries and database keys
        norm_q = F.normalize(q, p=2, dim=-1) # [B, D]
        norm_keys = F.normalize(self.signatures[:count], p=2, dim=-1) # [count, D]
        
        # Similarity matrix: [B, count]
        sim = torch.matmul(norm_q, norm_keys.t())
        
        # Clamp top_k to database size
        k = min(top_k, count)
        scores, indices = torch.topk(sim, k=k, dim=-1) # [B, k], [B, k]
        
        # Gather matching concepts
        retrieved_flat = self.concepts[indices.flatten()] # [B * k, D]
        retrieved = retrieved_flat.reshape(B, k, self.d_model)
        
        return retrieved, scores
