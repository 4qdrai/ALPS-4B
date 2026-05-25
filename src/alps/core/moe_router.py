import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """A single feed-forward expert module inside the MoE block."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

class SparseMoERouter(nn.Module):
    """
    Sparse Mixture of Experts (MoE) routing block.
    Dynamically routes latent tokens to a subset of Expert networks based on semantic similarity,
    enabling conditional computation (FLOPs are constant O(1) relative to total expert count).
    """
    def __init__(self, d_model: int, num_experts: int = 8, active_experts: int = 2, d_ff: int = 1024):
        super().__init__()
        self.num_experts = num_experts
        self.active_experts = active_experts
        
        # Expert modules list
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        
        # Router gate: maps inputs to raw routing logit scores
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Input tokens, Shape: [B, N, D]
            
        Returns:
            out: Expert-combined token output, Shape: [B, N, D]
            balance_loss: Load balancing auxiliary loss term (scalar)
        """
        B, N, D = x.shape
        flat_x = x.reshape(-1, D) # [B*N, D]
        num_tokens = flat_x.shape[0]
        
        # 1. Compute routing logits
        logits = self.gate(flat_x) # [B*N, num_experts]
        
        # Add noise to logits during training to aid exploration (standard MoE practice)
        if self.training:
            noise = torch.randn_like(logits) * (1.0 / self.num_experts)
            logits = logits + noise
            
        # 2. Select top-K experts
        scores, indices = torch.topk(logits, k=self.active_experts, dim=-1) # [B*N, active_experts]
        
        # Apply softmax over the selected top-K expert scores
        routing_weights = F.softmax(scores, dim=-1) # [B*N, active_experts]
        
        # 3. Load Balancing Loss Calculation
        # Load balancing is calculated to ensure experts are used equally.
        # We compute: (num_experts) * sum_{e=1}^{E} f_e * P_e
        # f_e: fraction of tokens routed to expert e
        # P_e: average probability allocated to expert e
        probs = F.softmax(logits, dim=-1) # [B*N, num_experts]
        
        # Fraction of tokens routed to each expert:
        # Create one-hot vectors for indices
        first_expert_indices = indices[:, 0] # [B*N]
        f_e = torch.zeros(self.num_experts, device=x.device)
        f_e.scatter_add_(0, first_expert_indices, torch.ones_like(first_expert_indices, dtype=torch.float))
        f_e = f_e / num_tokens
        
        # Average probability allocated to each expert:
        P_e = probs.mean(dim=0) # [num_experts]
        
        # Balancing loss (we want f_e and P_e to be uniform -> dot product is minimized when uniform)
        balance_loss = self.num_experts * torch.sum(f_e * P_e)
        
        # 4. Expert Computation (conditional processing)
        out_flat = torch.zeros_like(flat_x) # [B*N, D]
        
        # Vectorized routing implementation
        # To avoid running every expert on sparse indexing individually which can be slow in Python,
        # we iterate over experts, find tokens routed to them, and process them in bulk.
        for e_idx in range(self.num_experts):
            # Find which elements of the batch are routed to expert e_idx
            # mask has Shape: [B*N, active_experts]
            mask = (indices == e_idx)
            if not mask.any():
                continue
                
            # Extract which token indices and which of their top-K slots match this expert
            token_ids, slot_ids = torch.nonzero(mask, as_tuple=True)
            
            # Extract inputs for expert
            expert_inputs = flat_x[token_ids] # [Num_routed, D]
            
            # Execute expert
            expert_outputs = self.experts[e_idx](expert_inputs) # [Num_routed, D]
            
            # Weight outputs by their routing weights
            weights = routing_weights[token_ids, slot_ids].unsqueeze(-1) # [Num_routed, 1]
            weighted_outputs = expert_outputs * weights
            
            # Accumulate back into output buffer (using scatter_add or direct index addition)
            out_flat.index_add_(0, token_ids, weighted_outputs)
            
        out = out_flat.reshape(B, N, D)
        return out, balance_loss
