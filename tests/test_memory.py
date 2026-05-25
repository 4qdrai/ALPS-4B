import pytest
import torch
from alps.memory.working_buffer import WorkingStateBuffer
from alps.memory.episodic_cache import EpisodicRolloutCache
from alps.memory.semantic_memory import SemanticVectorDatabase
from alps.memory.abstraction_scorer import AbstractionScorer
from alps.memory.sleep_distillation import SleepConsolidation

def test_memory_system_components():
    """Validates the 3-tier memory structures, abstraction scoring, and sleep consolidation."""
    d_model = 48
    
    # 1. FIFO Working Buffer
    buf = WorkingStateBuffer(d_model=d_model, d_action=8, capacity=3)
    buf.push(torch.ones(1, d_model), torch.ones(1, 8))
    buf.push(torch.ones(1, d_model) * 2.0, torch.ones(1, 8))
    buf.push(torch.ones(1, d_model) * 3.0, torch.ones(1, 8))
    
    # Push 4th should pop first (1.0)
    buf.push(torch.ones(1, d_model) * 4.0, torch.ones(1, 8))
    states, actions = buf.get_history()
    assert len(states) == 3
    assert states[0].mean().item() == 2.0
    assert states[-1].mean().item() == 4.0
    
    # 2. Episodic Rollout Cache
    cache = EpisodicRolloutCache(d_model=d_model, d_action=8, max_episodes=5, sequence_len=10, decay_rate=0.9)
    cache.add_episode(torch.randn(10, d_model), torch.randn(10, 8))
    cache.decay_importance()
    assert cache.decay_rate == 0.9
    
    # 3. Semantic Vector Database
    db = SemanticVectorDatabase(d_model=d_model)
    key = torch.randn(d_model)
    val = torch.randn(d_model)
    db.add_concept(key, val)
    assert db.current_count.item() == 1
    
    retrieved, scores = db.query_concepts(key.unsqueeze(0), top_k=1)
    assert retrieved.shape == (1, 1, d_model)
    
    # 4. Abstraction Scorer
    scorer = AbstractionScorer(d_model=d_model)
    z_curr = torch.randn(2, 5, d_model)
    z_prev = torch.randn(2, 5, d_model)
    
    invariance = scorer.compute_temporal_invariance(z_curr, z_prev)
    compression = scorer.compute_dimensional_compression(z_curr)
    tier = scorer.classify_abstraction_tier(z_curr, z_prev)
    assert invariance >= 0.0
    assert compression >= 0.0
    assert tier in ["STRATEGIC", "TACTICAL", "OPERATIVE"]
    
    # 5. Sleep Consolidation
    consolidation = SleepConsolidation(lr=1e-4, epochs=2)
    from alps.core.predictor import MultiScalePredictor
    from alps.core.latent_rag import LatentRAG
    predictor = MultiScalePredictor(d_model=d_model, d_cond=d_model)
    rag = LatentRAG(d_model=d_model)
    
    metrics = consolidation.consolidate(predictor, rag)
    assert metrics["consolidated_count"] == 0
