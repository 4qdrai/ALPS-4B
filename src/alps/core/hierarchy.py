import torch
import torch.nn as nn
import torch.nn.functional as F

from alps.core.sigreg import SIGReg
from alps.core.predictor import MultiScalePredictor
from alps.core.vq_bottleneck import VectorQuantizer
from alps.core.moe_router import SparseMoERouter
from alps.core.latent_rag import LatentRAG

class StrategicLayer(nn.Module):
    """
    Strategic Layer (System 2 - Slow Conceptual Reasoning).
    
    Operates at long temporal horizons (step T+k^2).
    Forces discrete conceptual abstractions via a Vector Quantization (VQ) bottleneck.
    Collapse is prevented locally via SIGReg.
    """
    def __init__(self, d_model: int = 384, num_embeddings: int = 512, 
                 predictor_depth: int = 4, lambda_sigreg: float = 0.1):
        super().__init__()
        self.vq = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=d_model)
        self.predictor = MultiScalePredictor(d_model=d_model, d_cond=d_model, depth=predictor_depth)
        self.sigreg = SIGReg(d_model=d_model)
        self.lambda_sigreg = lambda_sigreg
        
    def forward(self, continuous_latents: torch.Tensor) -> tuple:
        """
        Args:
            continuous_latents: Inputs from higher abstraction layers, Shape: [B, N, D]
            
        Returns:
            quantized_concept: Discrete conceptual guidance, Shape: [B, N, D]
            vq_loss: Commitment codebook loss (scalar)
            sigreg_loss: Collapse prevention loss (scalar)
        """
        # 1. Quantize representations to enforce discrete conceptual bottleneck
        quantized_concept, vq_loss, _ = self.vq(continuous_latents)
        
        # 2. Local collapse prevention check (SIGReg on the continuous representation before quantization)
        sigreg_loss = self.sigreg(continuous_latents) * self.lambda_sigreg
        
        return quantized_concept, vq_loss, sigreg_loss
        
    def predict_next_concept(self, c_T: torch.Tensor, strategic_guidance: torch.Tensor) -> torch.Tensor:
        """Predicts future concept c_{T+1} conditioned on high-level goals."""
        return self.predictor(c_T, strategic_guidance)

class TacticalLayer(nn.Module):
    """
    Tactical Layer (System 2 - Mid-range sub-goal planning).
    
    Operates at medium horizons (step T+k).
    Conditions on top-down Strategic guidance c_T via stop-gradients to prevent gradient contamination.
    Employs:
    1. Sparse Mixture of Experts (MoE) for semantic action routing.
    2. Latent-RAG memory KV cache for one-shot failure correction.
    """
    def __init__(self, d_model: int = 384, num_experts: int = 8, active_experts: int = 2,
                 predictor_depth: int = 4, lambda_sigreg: float = 0.1, sim_threshold: float = 0.7):
        super().__init__()
        self.moe = SparseMoERouter(d_model=d_model, num_experts=num_experts, active_experts=active_experts)
        self.rag = LatentRAG(d_model=d_model, sim_threshold=sim_threshold)
        self.predictor = MultiScalePredictor(d_model=d_model, d_cond=d_model, depth=predictor_depth)
        self.sigreg = SIGReg(d_model=d_model)
        self.lambda_sigreg = lambda_sigreg
        
    def forward(self, x: torch.Tensor, strategic_guidance: torch.Tensor) -> tuple:
        """
        Args:
            x: Input tactical state sequence, Shape: [B, N, D]
            strategic_guidance: Concept c_T from Strategic layer, Shape: [B, N, D]
            
        Returns:
            tactical_out: Refined sub-goal representation, Shape: [B, N, D]
            moe_loss: Expert balancing loss (scalar)
            sigreg_loss: Collapse prevention loss (scalar)
        """
        # Stop-gradient top-down guidance to enforce clean structural isolation
        c_T = strategic_guidance.detach()
        
        # 1. Condition the state using top-down conceptual constraints
        conditioned_x = x + c_T
        
        # 2. MoE expert routing
        moe_out, moe_loss = self.moe(conditioned_x)
        
        # 3. Retrieve Latent-RAG episodic corrections (delta-z) for failure correction
        corrections = self.rag.retrieve_correction(moe_out)
        tactical_out = moe_out + corrections
        
        # 4. Local collapse prevention (SIGReg)
        sigreg_loss = self.sigreg(tactical_out) * self.lambda_sigreg
        
        return tactical_out, moe_loss, sigreg_loss
        
    def predict_next_subgoal(self, h_T: torch.Tensor, strategic_concept: torch.Tensor) -> torch.Tensor:
        """Predicts future sub-goal trajectory h_{T+1} conditioned on conceptual plan."""
        return self.predictor(h_T, strategic_concept.mean(dim=1))

class OperativeLayer(nn.Module):
    """
    Operative Layer (System 1 - High-frequency sensorimotor prediction).
    
    Operates at millisecond intervals (step t+1).
    Receives top-down sub-goals h_T from Tactical layer via stop-gradients.
    Executes raw state updates and drives actions.
    """
    def __init__(self, d_model: int = 384, predictor_depth: int = 6, 
                 d_action: int = 64, lambda_sigreg: float = 0.1):
        super().__init__()
        self.predictor = MultiScalePredictor(d_model=d_model, d_cond=d_action, depth=predictor_depth)
        self.sigreg = SIGReg(d_model=d_model)
        self.lambda_sigreg = lambda_sigreg
        
    def forward(self, z_t: torch.Tensor, tactical_subgoal: torch.Tensor) -> tuple:
        """
        Args:
            z_t: Encoded physical state embeddings, Shape: [B, N, D]
            tactical_subgoal: Sub-goal h_T from Tactical layer, Shape: [B, N, D]
            
        Returns:
            conditioned_z: Decoupled latent representation, Shape: [B, N, D]
            sigreg_loss: Collapse prevention loss (scalar)
        """
        # Stop-gradient top-down subgoal guidance
        h_T = tactical_subgoal.detach()
        
        # Merge physical state and sub-goal expectations
        conditioned_z = z_t + h_T
        
        # Local collapse prevention (SIGReg)
        sigreg_loss = self.sigreg(conditioned_z) * self.lambda_sigreg
        
        return conditioned_z, sigreg_loss
        
    def predict_next_state(self, z_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predicts future latent state z_{t+1} conditioned on immediate action."""
        return self.predictor(z_t, action)
