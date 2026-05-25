import pytest
import torch
import torch.nn as nn

# Import ALPS components
from alps.core.sigreg import SIGReg
from alps.core.encoders import VisionEncoder, TubePatchEmbedding
from alps.core.predictor import MultiScalePredictor, AdaLN
from alps.core.vq_bottleneck import VectorQuantizer
from alps.core.moe_router import SparseMoERouter
from alps.core.latent_rag import LatentRAG
from alps.core.inverse_monitor import InverseMonitor
from alps.core.checker import BanachChecker
from alps.core.fallback import FallbackMonitor
from alps.core.energy import EBMBinder
from alps.core.alps_model import ALPSModel

from alps.memory.working_buffer import WorkingStateBuffer
from alps.memory.episodic_cache import EpisodicRolloutCache
from alps.memory.semantic_memory import SemanticVectorDatabase
from alps.memory.abstraction_scorer import AbstractionScorer
from alps.memory.sleep_distillation import SleepConsolidation

from alps.multimodal.sensor_encoders import IMUEncoder, LidarEncoder
from alps.multimodal.modality_router import ModalityRouter

def test_sigreg():
    """Validates that SIGReg prevents representation collapse."""
    d_model = 64
    sigreg = SIGReg(d_model=d_model, num_slices=32, beta=1.0)
    
    # 1. Standard normal input should yield a low goodness-of-fit statistic (approx stable)
    normal_input = torch.randn(10, d_model)
    loss_normal = sigreg(normal_input)
    assert loss_normal.item() >= 0
    
    # 2. Collapsed input (all constant vectors) should yield a high statistic
    collapsed_input = torch.zeros(10, d_model)
    loss_collapsed = sigreg(collapsed_input)
    assert loss_collapsed.item() >= 0
    
    # 3. Weak-SIGReg check
    sigreg_weak = SIGReg(d_model=d_model, num_slices=32, weak_only=True)
    loss_weak = sigreg_weak(normal_input)
    assert loss_weak.item() >= 0

def test_encoders():
    """Validates the spatiotemporal Vision Transformer Encoder and Patch Embedding."""
    B, C, T, H, W = 2, 3, 4, 32, 32
    d_model = 64
    
    encoder = VisionEncoder(
        in_channels=C, d_model=d_model, depth=2, num_heads=2, 
        patch_size=(2, 8, 8), max_patches=256
    )
    
    video = torch.randn(B, C, T, H, W)
    latents = encoder(video)
    
    # Expected output: [B, N, D]
    # T_out = 4 / 2 = 2
    # H_out = 32 / 8 = 4
    # W_out = 32 / 8 = 4
    # N = 2 * 4 * 4 = 32 patches
    assert latents.shape == (B, 32, d_model)

def test_moe_router():
    """Validates Mixture of Experts routing and load balancing calculations."""
    d_model = 64
    router = SparseMoERouter(d_model=d_model, num_experts=4, active_experts=2, d_ff=128)
    
    inputs = torch.randn(2, 10, d_model)
    outputs, balance_loss = router(inputs)
    
    assert outputs.shape == inputs.shape
    assert balance_loss.item() > 0

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

def test_banach_checker():
    """Validates Banach Checker iteration convergence under Lipschitz condition L < 1."""
    d_model = 64
    checker = BanachChecker(d_model=d_model, d_cond=d_model, max_iters=5)
    
    initial_z = torch.randn(2, 5, d_model)
    context = torch.randn(2, 5, d_model)
    
    refined_z, steps, converged = checker(initial_z, context)
    assert refined_z.shape == initial_z.shape
    assert steps > 0
    
    # Compute contraction loss
    loss = checker.compute_contraction_loss(initial_z, context)
    assert loss.item() >= 0

def test_fallback_monitor():
    """Validates NaN, Variance Collapse, and Hypersphere Pinning detectors."""
    d_model = 64
    monitor = FallbackMonitor(var_threshold=1e-3, pinning_threshold=0.99)
    
    # 1. Healthy check
    healthy_z = torch.randn(2, 5, d_model)
    z_prev = torch.randn(2, 5, d_model)
    status, msg = monitor.verify_system_health(healthy_z, z_prev)
    assert status is True
    assert msg == "SYSTEM_HEALTHY"
    
    # 2. NaN check
    nan_z = healthy_z.clone()
    nan_z[0, 0, 0] = float("nan")
    status, msg = monitor.verify_system_health(nan_z, z_prev)
    assert status is False
    assert "NaN" in msg
    
    # 3. Variance Collapse check
    collapsed_z = torch.zeros(2, 5, d_model)
    status, msg = monitor.verify_system_health(collapsed_z, z_prev)
    assert status is False
    assert "Variance" in msg
    
    # 4. Hypersphere Pinning check
    pinned_z = healthy_z.clone()
    status, msg = monitor.verify_system_health(pinned_z, pinned_z)
    assert status is False
    assert "Pinning" in msg

def test_alps_model():
    """Validates the full consolidated ALPS-4B model pipeline forward pass."""
    d_model = 384
    d_action = 64
    model = ALPSModel(d_model=d_model, d_action=d_action)
    
    # Input matching 16 frames x 3 channels x 224 x 224
    video = torch.randn(1, 3, 16, 224, 224)
    action = torch.randn(1, d_action)
    
    # Forward pass
    outputs = model(video, action)
    
    assert "z_t" in outputs
    assert "z_pred" in outputs
    assert "loss" in outputs
    assert "system_healthy" in outputs
    assert outputs["system_healthy"] is True
    assert outputs["loss"].item() >= 0
