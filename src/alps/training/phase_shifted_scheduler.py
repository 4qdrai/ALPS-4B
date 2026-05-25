import torch

class PhaseShiftedScheduler:
    """
    Phase-Shifted Update Scheduler.
    
    Coordinates the update frequencies of the hierarchical layers:
    - Operative: Every step (high frequency physical control).
    - Tactical: Every k steps (medium frequency sub-goals).
    - Strategic: Every k^2 steps (slow frequency concepts).
    
    This ensures that high-level conceptual layers remain stable and do not
    overfit or churn on high-frequency noise.
    """
    def __init__(self, k_tactical: int = 4, k_strategic: int = 16):
        self.k_tactical = k_tactical
        self.k_strategic = k_strategic
        self.step_count = 0
        
    def step(self) -> dict:
        """
        Increments the step counter and returns active update flags for each layer.
        
        Returns:
            update_flags: Dictionary of booleans:
                          {
                             "update_operative": bool,
                             "update_tactical": bool,
                             "update_strategic": bool
                          }
        """
        self.step_count += 1
        
        update_op = True
        update_tac = (self.step_count % self.k_tactical == 0)
        update_str = (self.step_count % self.k_strategic == 0)
        
        return {
            "update_operative": update_op,
            "update_tactical": update_tac,
            "update_strategic": update_str,
            "step": self.step_count
        }
        
    def reset(self):
        self.step_count = 0
