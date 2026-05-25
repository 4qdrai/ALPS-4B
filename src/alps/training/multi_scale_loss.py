import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleLoss(nn.Module):
    """
    ALPS-4B Hierarchical Multi-Scale Loss Module.
    
    Aggregates per-layer prediction MSE errors and local SIGReg collapse prevention
    regularization terms.
    
    Loss = alpha * Loss_Strategic + beta * Loss_Tactical + gamma * Loss_Operative
    
    Where:
    - Within each layer: LeWM's single-scale (no EMA, no target encoder) formulation.
    - Between layers: Stop-gradients enforce clean temporal and semantic separation.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def forward(self, 
                z_op: torch.Tensor, z_op_pred: torch.Tensor, sigreg_op: torch.Tensor,
                z_tac: torch.Tensor, z_tac_pred: torch.Tensor, sigreg_tac: torch.Tensor, moe_loss: torch.Tensor,
                z_str: torch.Tensor, z_str_pred: torch.Tensor, sigreg_str: torch.Tensor, vq_loss: torch.Tensor) -> dict:
        """
        Computes the multi-scale loss.
        """
        # 1. Operative Layer Loss
        # Predicts z_{t+1} given z_t. Gradients do not flow from operative predictor up to tactical subgoal
        loss_op_pred = F.mse_loss(z_op_pred, z_op.detach())
        loss_operative = loss_op_pred + sigreg_op
        
        # 2. Tactical Layer Loss
        # Predicts h_{T+k}. Gradients do not flow up to strategic concept
        loss_tac_pred = F.mse_loss(z_tac_pred, z_tac.detach())
        loss_tactical = loss_tac_pred + sigreg_tac + moe_loss
        
        # 3. Strategic Layer Loss
        # Predicts concept c_{T+k^2}
        loss_str_pred = F.mse_loss(z_str_pred, z_str.detach())
        loss_strategic = loss_str_pred + sigreg_str + vq_loss
        
        # Total aggregated loss
        total_loss = (
            self.alpha * loss_strategic +
            self.beta * loss_tactical +
            self.gamma * loss_operative
        )
        
        return {
            "loss": total_loss,
            "loss_strategic": loss_strategic,
            "loss_tactical": loss_tactical,
            "loss_operative": loss_operative,
            "op_pred_mse": loss_op_pred,
            "tac_pred_mse": loss_tac_pred,
            "str_pred_mse": loss_str_pred
        }
