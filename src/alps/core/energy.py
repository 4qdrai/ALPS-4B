import torch
import torch.nn as nn
import torch.nn.functional as F

class EBMBinder(nn.Module):
    """
    Energy-Based Model (EBM) Binding Layer.
    
    In a JEPA, prediction errors correspond to energy levels.
    This class aggregates per-layer prediction errors (energies) from the
    Strategic, Tactical, and Operative layers into a unified multi-scale energy
    landscape E_total.
    
    E_total = alpha * E_strategic + beta * E_tactical + gamma * E_operative
    
    A low total energy represents a coherent prediction and plan, whereas
    high energy signals cross-layer incompatibility or planning failure.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def forward(self, e_strategic: torch.Tensor, e_tactical: torch.Tensor, e_operative: torch.Tensor) -> torch.Tensor:
        """
        Computes the weighted total energy of the hierarchical JEPA system.
        
        Args:
            e_strategic: Energy (prediction MSE) from Strategic Layer.
            e_tactical: Energy (prediction MSE) from Tactical Layer.
            e_operative: Energy (prediction MSE) from Operative Layer.
            
        Returns:
            e_total: Unified scalar energy score representing total system coherence.
        """
        # Ensure all energy tensors are reduced to scalars or matched shapes
        if e_strategic.dim() > 0:
            e_strategic = e_strategic.mean()
        if e_tactical.dim() > 0:
            e_tactical = e_tactical.mean()
        if e_operative.dim() > 0:
            e_operative = e_operative.mean()
            
        e_total = self.alpha * e_strategic + self.beta * e_tactical + self.gamma * e_operative
        return e_total
        
    def check_coherence(self, e_strategic: torch.Tensor, e_tactical: torch.Tensor, 
                        e_operative: torch.Tensor, thresholds: tuple = (1.0, 1.0, 1.0)) -> dict:
        """
        Performs a diagnostic check across the energy landscape to pinpoint
        which layer is experiencing planning conflicts.
        
        Args:
            thresholds: Tuple of (tau_strategic, tau_tactical, tau_operative)
        """
        es = e_strategic.mean().item()
        et = e_tactical.mean().item()
        eo = e_operative.mean().item()
        
        ts, tt, to = thresholds
        
        return {
            "strategic_energy": es,
            "tactical_energy": et,
            "operative_energy": eo,
            "strategic_coherent": es < ts,
            "tactical_coherent": et < tt,
            "operative_coherent": eo < to,
            "system_coherent": (es < ts) and (et < tt) and (eo < to)
        }
