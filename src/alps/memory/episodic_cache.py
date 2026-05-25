import torch
import torch.nn as nn

class EpisodicRolloutCache(nn.Module):
    """
    Tactical Episodic Rollout Cache (System 2 Episodic Memory).
    
    Stores complete multi-step trajectory paths (rollouts) of latent states and actions.
    Each episodic record is associated with an importance score that decays
    exponentially over time unless the experience is actively rehearsed or retrieved.
    
    This acts as a buffer for consolidation into long-term semantic memory.
    """
    def __init__(self, d_model: int = 384, d_action: int = 64, max_episodes: int = 100, 
                 sequence_len: int = 16, decay_rate: float = 0.95):
        super().__init__()
        self.d_model = d_model
        self.d_action = d_action
        self.max_episodes = max_episodes
        self.sequence_len = sequence_len
        self.decay_rate = decay_rate
        
        # Buffer containers
        self.register_buffer("states", torch.zeros(max_episodes, sequence_len, d_model))
        self.register_buffer("actions", torch.zeros(max_episodes, sequence_len, d_action))
        self.register_buffer("importance", torch.zeros(max_episodes))
        self.register_buffer("current_episodes", torch.tensor(0, dtype=torch.long))
        
    def add_episode(self, state_sequence: torch.Tensor, action_sequence: torch.Tensor, initial_importance: float = 1.0):
        """
        Pushes a new episodic rollout sequence into the cache.
        
        Args:
            state_sequence: Trajectory states, Shape: [Seq_len, D_model]
            action_sequence: Trajectory actions, Shape: [Seq_len, D_action]
        """
        device = self.states.device
        state_sequence = state_sequence.detach().to(device)
        action_sequence = action_sequence.detach().to(device)
        
        # Apply temporal decay to all existing episodic records
        self.decay_importance()
        
        idx = self.current_episodes.item()
        
        if idx >= self.max_episodes:
            # Evict the episode with the lowest importance score
            evict_idx = torch.argmin(self.importance)
            self.states[evict_idx] = state_sequence[:self.sequence_len]
            self.actions[evict_idx] = action_sequence[:self.sequence_len]
            self.importance[evict_idx] = initial_importance
        else:
            self.states[idx] = state_sequence[:self.sequence_len]
            self.actions[idx] = action_sequence[:self.sequence_len]
            self.importance[idx] = initial_importance
            self.current_episodes += 1
            
    def decay_importance(self):
        """Applies exponential temporal decay to all cached episodes."""
        size = self.current_episodes.item()
        if size > 0:
            self.importance[:size] *= self.decay_rate
            
    def reinforce_episode(self, index: int, value: float = 0.5):
        """Boosts the importance score of a specific episode upon retrieval/rehearsal."""
        size = self.current_episodes.item()
        if 0 <= index < size:
            self.importance[index] = min(1.0, self.importance[index].item() + value)
            
    def get_important_episodes(self, threshold: float = 0.1) -> tuple:
        """
        Returns all episodic sequences whose importance score exceeds the threshold.
        """
        size = self.current_episodes.item()
        if size == 0:
            return torch.zeros(0, self.sequence_len, self.d_model), \
                   torch.zeros(0, self.sequence_len, self.d_action)
                   
        mask = self.importance[:size] >= threshold
        return self.states[:size][mask], self.actions[:size][mask]
        
    def clear(self):
        """Purges episodic cache."""
        self.states.zero_()
        self.actions.zero_()
        self.importance.zero_()
        self.current_episodes.zero_()
