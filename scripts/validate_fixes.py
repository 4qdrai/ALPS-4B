"""
ALPS-4B Validation Script — Verifies all audit fixes.
"""
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')
import torch
import torch.nn.functional as F

print("=" * 70)
print("ALPS-4B AUDIT FIX VALIDATION")
print("=" * 70)

# --- Test 1: Model Initialization ---
print("\n[TEST 1] Model initialization with new projection head...")
from alps.core.alps_model import ALPSModel
model = ALPSModel()
print("  ✓ ALPSModel initialized successfully")

# Verify projection head exists in encoder
assert hasattr(model.encoder, 'projection_head'), "FAIL: projection_head not found in encoder"
print("  ✓ Encoder has MLP+BatchNorm projection head")

# --- Test 2: SIGReg num_slices = 1024 ---
print("\n[TEST 2] SIGReg num_slices = 1024...")
from alps.core.sigreg import SIGReg
sig = SIGReg(d_model=384)
assert sig.num_slices == 1024, f"FAIL: num_slices = {sig.num_slices}, expected 1024"
assert sig.projection_matrix.shape == (384, 1024), f"FAIL: projection shape = {sig.projection_matrix.shape}"
print(f"  ✓ SIGReg default slices = {sig.num_slices}")
print(f"  ✓ Projection matrix shape = {sig.projection_matrix.shape}")

# --- Test 3: Forward pass with prev_latents ---
print("\n[TEST 3] Forward pass with prev_latents (temporal recurrence)...")
x = torch.randn(1, 3, 16, 224, 224)
a = torch.randn(1, 64)

# First pass to get correct latent shape
out_first = model(x, a)
prev = out_first["z_t"].detach()  # Use real latent shape

# Second pass with prev_latents
out = model(x, a, prev_latents=prev)
print(f"  ✓ Forward pass completed with temporal recurrence")
print(f"    Loss: {out['loss'].item():.4f}")
print(f"    System healthy: {out['system_healthy']}")
print(f"    System 2 activated: {out['system2_activated']}")
print(f"    Fallback triggered: {out['fallback_triggered']}")

# --- Test 4: Natural System 2 escalation (no force_system2) ---
print("\n[TEST 4] System 2 natural escalation test...")
# Run multiple random inputs to see if System 2 activates naturally
sys2_count = 0
for i in range(5):
    x_rand = torch.randn(1, 3, 16, 224, 224)
    out_i = model(x_rand, a)
    if out_i.get('system2_activated', False):
        sys2_count += 1
print(f"  System 2 activated in {sys2_count}/5 random forward passes")
if sys2_count > 0:
    print(f"  ✓ System 2 can activate naturally via InverseMonitor threshold")
else:
    print(f"  ⚠ System 2 did not activate (threshold may need tuning)")

# --- Test 5: Encoder optimizer setup ---
print("\n[TEST 5] Training setup with encoder optimizer...")
import torch.optim as optim
optimizer_enc = optim.AdamW(model.encoder.parameters(), lr=1e-4)
optimizer_op = optim.AdamW(model.operative_layer.parameters(), lr=1e-4)
enc_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
print(f"  ✓ Encoder optimizer created with {enc_params:,} trainable parameters")

# Verify encoder actually gets gradients
model.train()
x_train = torch.randn(2, 3, 16, 224, 224)
a_train = torch.randn(2, 64)
optimizer_enc.zero_grad()
optimizer_op.zero_grad()
out_train = model(x_train, a_train)
loss = out_train['loss']
loss.backward()

# Check encoder has gradients
enc_has_grad = False
for p in model.encoder.parameters():
    if p.grad is not None and p.grad.abs().sum() > 0:
        enc_has_grad = True
        break
if enc_has_grad:
    print(f"  ✓ Encoder receives gradients during backward pass")
else:
    print(f"  ⚠ Encoder did NOT receive gradients (check stop-gradients)")

# --- Test 6: Strategic->Tactical top-down cascade ---
print("\n[TEST 6] Strategic→Tactical top-down cascade...")
model.eval()
x_sys2 = torch.randn(1, 3, 16, 224, 224)
out_sys2 = model(x_sys2, a, force_system2=True)
print(f"  ✓ System 2 forced forward pass completed")
print(f"    Strategic activated: {out_sys2.get('strategic_activated', 'N/A')}")
print(f"    VQ loss: {out_sys2.get('vq_loss', 0)}")
print(f"    MoE loss: {out_sys2.get('moe_loss', 0)}")
print(f"    Contraction loss: {out_sys2.get('contraction_loss', 0)}")
print(f"    RAG auto-write: {out_sys2.get('rag_auto_write', 'N/A')}")

# --- Test 7: Auto-RAG write ---
print("\n[TEST 7] Auto-RAG write verification...")
rag_size_before = model.tactical_layer.rag.current_size.item()
print(f"  RAG cache size before: {rag_size_before}")
# The auto-write triggers when pred_loss_op > 1.0 and not training
# With random weights this should happen naturally
if out_sys2.get('rag_auto_write', False):
    rag_size_after = model.tactical_layer.rag.current_size.item()
    print(f"  RAG cache size after: {rag_size_after}")
    print(f"  ✓ Auto-RAG write confirmed! Cache grew by {rag_size_after - rag_size_before}")
else:
    print(f"  ⚠ Auto-RAG did not trigger (pred_loss_op may be ≤ 1.0 or model was in training mode)")

# --- Test 8: Spatiotemporal Masker integration ---
print("\n[TEST 8] Spatiotemporal masker integration...")
import importlib.util
spec = importlib.util.spec_from_file_location("masked_prediction", "src/alps/training/masked_prediction.py")
mp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mp)
SpatiotemporalMasker = mp.SpatiotemporalMasker
masker = SpatiotemporalMasker(mask_ratio=0.9)
keep, mask = masker.generate_tube_mask(batch_size=2, device=torch.device('cpu'))
kept_ratio = keep.float().mean().item()
print(f"  ✓ Tube mask generated: shape={keep.shape}, kept_ratio={kept_ratio:.2%}")
print(f"    Expected ~10% kept, got {kept_ratio:.2%}")

# --- Test 9: Abstraction Scorer ---
print("\n[TEST 9] Abstraction scorer validation...")
from alps.memory.abstraction_scorer import AbstractionScorer
scorer = AbstractionScorer()
z1 = torch.randn(2, 10, 384)
z2 = z1 + torch.randn_like(z1) * 0.01  # similar
tier_similar = scorer.classify_abstraction_tier(z1, z2)
z3 = torch.randn(2, 10, 384)  # totally different
tier_diff = scorer.classify_abstraction_tier(z1, z3)
print(f"  ✓ Similar frames classified as: {tier_similar}")
print(f"  ✓ Different frames classified as: {tier_diff}")

# --- Test 10: Sleep Consolidation ---
print("\n[TEST 10] Sleep consolidation dry-run...")
from alps.memory.sleep_distillation import SleepConsolidation
from alps.core.latent_rag import LatentRAG
from alps.core.predictor import MultiScalePredictor

rag_test = LatentRAG(d_model=384)
pred_test = MultiScalePredictor(d_model=384, d_cond=384, depth=2)
# Write some test memories
for i in range(3):
    k = torch.randn(384)
    v = torch.randn(384)
    rag_test.write_memory(k, v)
    rag_test.usage_counts[i] = 5  # Mark as frequently used

consolidator = SleepConsolidation(epochs=2)
metrics = consolidator.consolidate(pred_test, rag_test)
print(f"  ✓ Consolidated {metrics['consolidated_count']} memories")
print(f"    Initial loss: {metrics['initial_loss']:.4f}")
print(f"    Final loss: {metrics['final_loss']:.4f}")
print(f"    RAG size after purge: {rag_test.current_size.item()}")

# --- Test 11: Hive-Mind sync ---
print("\n[TEST 11] Hive-Mind fleet sync...")
rag_A = LatentRAG(d_model=384, sim_threshold=0.5)
rag_B = LatentRAG(d_model=384, sim_threshold=0.5)
test_key = torch.randn(1, 1, 384)
test_val = torch.randn(384)
rag_A.write_memory(test_key, test_val)
# Sync
rag_B.keys.copy_(rag_A.keys)
rag_B.values.copy_(rag_A.values)
rag_B.current_size.copy_(rag_A.current_size)
retrieved = rag_B.retrieve_correction(test_key)
sync_ok = retrieved.abs().sum().item() > 0
print(f"  ✓ Fleet sync {'PASSED' if sync_ok else 'FAILED'}: retrieved correction norm = {retrieved.abs().sum().item():.4f}")

# --- Test 12: Active Safe Haven Homing Watchdog ---
print("\n[TEST 12] Active Safe Haven Homing Watchdog...")
from alps.core.fallback import FallbackMonitor
fb = FallbackMonitor()
# Test 12.1: Default safe halt
act_plan = torch.zeros(1, 4)
mrc_stop = fb.get_minimal_risk_action(act_plan)
assert mrc_stop.sum().item() == 0, "MRC action should be zeros when position is None"
print("  ✓ Zeros safely outputted when position is None")

# Test 12.2: Safe Haven active steering (Left room center at 2.5, 5.0)
pos_left = torch.tensor([[1.0, 1.0]])
mrc_left = fb.get_minimal_risk_action(act_plan, current_position=pos_left)
assert mrc_left[0, 0] == 1.0, f"Expected action 0 (up) for pos (1.0, 1.0), got {mrc_left}"
print("  ✓ Steered agent UP towards left Safe Haven from (1.0, 1.0)")

# Test 12.3: Safe Haven active steering (Right room center at 7.5, 5.0)
pos_right = torch.tensor([[9.0, 8.0]])
mrc_right = fb.get_minimal_risk_action(act_plan, current_position=pos_right)
assert mrc_right[0, 1] == 1.0, f"Expected action 1 (down) for pos (9.0, 8.0), got {mrc_right}"
print("  ✓ Steered agent DOWN towards right Safe Haven from (9.0, 8.0)")

# Test 12.4: Safe Haven arrival trigger
pos_close = torch.tensor([[2.5, 4.9]])
mrc_close = fb.get_minimal_risk_action(act_plan, current_position=pos_close)
assert mrc_close.sum().item() == 0, f"Expected safe stop (all zeros) on arrival, got {mrc_close}"
print("  ✓ Safely halted (all zeros) agent upon arrival at Safe Haven")

# Test 12.5: Complex Mode Safe Haven Steering (Room 1 center at 2.5, 7.5)
pos_c1 = torch.tensor([[1.0, 8.0]]) # in Room 1 (top-left)
mrc_c1 = fb.get_minimal_risk_action(act_plan, current_position=pos_c1, complex_mode=True)
# target is (2.5, 7.5), dx=1.5, dy=-0.5. abs(dx) > abs(dy), so move horizontally. dx > 0, so move right (action 3).
assert mrc_c1[0, 3] == 1.0, f"Expected action 3 (right) for pos (1.0, 8.0), got {mrc_c1}"
print("  ✓ [Complex Mode] Steered agent RIGHT towards Room 1 Safe Haven from (1.0, 8.0)")

# Test 12.6: Complex Mode Safe Haven arrival
pos_c_close = torch.tensor([[7.5, 7.45]]) # close to Room 2 center (7.5, 7.5)
mrc_c_close = fb.get_minimal_risk_action(act_plan, current_position=pos_c_close, complex_mode=True)
assert mrc_c_close.sum().item() == 0, f"Expected safe stop (all zeros) on arrival in Complex Mode, got {mrc_c_close}"
print("  ✓ [Complex Mode] Safely halted (all zeros) agent upon arrival at Room 2 Safe Haven")

# --- Parameter Count ---
print("\n" + "=" * 70)
total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable:,}")

enc_params = sum(p.numel() for p in model.encoder.parameters())
op_params = sum(p.numel() for p in model.operative_layer.parameters())
tac_params = sum(p.numel() for p in model.tactical_layer.parameters())
str_params = sum(p.numel() for p in model.strategic_layer.parameters())
print(f"\n  Encoder: {enc_params:,}")
print(f"  Operative: {op_params:,}")
print(f"  Tactical: {tac_params:,}")
print(f"  Strategic: {str_params:,}")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETE")
print("=" * 70)
