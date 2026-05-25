import torch
import torch.nn as nn

class WorkingStateBuffer(nn.Module):
    """
    Operative Working State Buffer (System 1 Working Memory).
    
    A fast, in-memory FIFO queue that stores recent latent states z_t and actions a_t
    to provide the immediate spatiotemporal context for high-frequency sensorimotor prediction.
    """
    def __init__(self, d_model: int = 384, d_action: int = 64, capacity: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_action = d_action
        self.capacity = capacity
        
        # Buffer containers registered for device management
        self.register_buffer("state_queue", torch.zeros(capacity, d_model))
        self.register_buffer("action_queue", torch.zeros(capacity, d_action))
        self.register_buffer("pointer", torch.tensor(0, dtype=torch.long))
        self.register_buffer("filled", torch.tensor(False, dtype=torch.bool))
        
    def push(self, state: torch.Tensor, action: torch.Tensor):
        """
        Pushes a new state-action pair into the working memory FIFO.
        
        Args:
            state: Latent state z_t, Shape: [D_model] or [1, D_model]
            action: Action a_t, Shape: [D_action] or [1, D_action]
        """
        state = state.squeeze().detach()
        action = action.squeeze().detach()
        
        ptr = self.pointer.item()
        
        self.state_queue[ptr] = state
        self.action_queue[ptr] = action
        
        next_ptr = (ptr + 1) % self.capacity
        self.pointer.copy_(torch.tensor(next_ptr, device=self.pointer.device))
        
        if next_ptr == 0:
            self.filled.copy_(torch.tensor(True, device=self.filled.device))
            
    def get_history(self) -> tuple:
        """
        Returns the queued state-action history sorted chronologically (oldest to newest).
        """
        ptr = self.pointer.item()
        is_filled = self.filled.item()
        
        if not is_filled:
            if ptr == 0:
                return torch.zeros(0, self.d_model, device=self.state_queue.device), \
                       torch.zeros(0, self.d_action, device=self.action_queue.device)
            return self.state_queue[:ptr], self.action_queue[:ptr]
            
        # Chronological reconstruction
        indices = torch.roll(torch.arange(self.capacity, device=self.state_queue.device), shifts=-ptr, dims=0)
        return self.state_queue[indices], self.action_queue[indices]
        
    def clear(self):
        """Purges working memory."""
        self.state_queue.zero_()
        self.action_queue.zero_()
        self.pointer.zero_()
        self.filled.copy_(torch.tensor(False, device=self.filled.device))
