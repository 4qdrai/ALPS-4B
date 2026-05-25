import pytest
import torch
from alps.core.latent_rag import LatentRAG

def test_latent_rag():
    """Validates Latent-RAG Key-Value cache mechanics."""
    d_model = 64
    rag = LatentRAG(d_model=d_model, max_size=5, sim_threshold=0.5)
    
    # Verify initial empty cache yields zero correction
    q = torch.randn(2, 5, d_model)
    correction = rag.retrieve_correction(q)
    assert (correction == 0).all()
    
    # Add items to memory cache
    key = torch.randn(d_model)
    value = torch.randn(d_model)
    rag.write_memory(key, value)
    assert rag.current_size.item() == 1
    
    # Query with identical vector (similarity 1.0 > 0.5) -> should retrieve value
    q_identical = key.unsqueeze(0).unsqueeze(0) # [1, 1, D]
    retrieved = rag.retrieve_correction(q_identical)
    assert retrieved.shape == (1, 1, d_model)
    assert not (retrieved == 0).all()
    
    # Delete memory
    rag.delete_memory(0)
    assert rag.current_size.item() == 0
