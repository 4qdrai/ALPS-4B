"""
Two Rooms ALPS-4B Training Script.

Main training entry point for the Two Rooms navigation benchmark.
Instantiates a smaller ALPS model tailored for 128x128 resolution with
discrete 4-way actions, and trains it using the full hierarchical
JEPA pipeline (System 1/2 dynamic compute gating, SIGReg, VQ, MoE, etc.).

Usage:
    python -m alps.benchmarks.two_rooms.train_two_rooms
    python -m alps.benchmarks.two_rooms.train_two_rooms --epochs 100 --batch-size 32
    python -m alps.benchmarks.two_rooms.train_two_rooms --data-path data/two_rooms.pt --device cuda
"""

import sys
import os

# Ensure the src directory is on the path for module resolution
sys.path.insert(0, "src")

import argparse
import json
import time
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from alps.core.encoders import VisionEncoder
from alps.core.hierarchy import OperativeLayer, TacticalLayer, StrategicLayer
from alps.core.inverse_monitor import InverseMonitor
from alps.core.fallback import FallbackMonitor
from alps.core.checker import BanachChecker
from alps.core.energy import EBMBinder
from alps.core.predictor import LangevinPlanner
from alps.core.sigreg import SIGReg
from alps.training.phase_shifted_scheduler import PhaseShiftedScheduler
from alps.benchmarks.two_rooms.dataset import TwoRoomsDataset


# ════════════════════════════════════════════════════════════════════════════════
#  TwoRoomsALPS — Lightweight ALPS-4B for the Two Rooms Benchmark
# ════════════════════════════════════════════════════════════════════════════════

class TwoRoomsALPS(nn.Module):
    """
    Smaller ALPS-4B model for the Two Rooms 128x128 benchmark.

    Architecture matches ALPSModel but with reduced dimensions:
        - VisionEncoder:  d_model=128, depth=4, num_heads=4, patch_size=(2,16,16)
        - OperativeLayer: d_model=128, d_action=4  (discrete one-hot)
        - TacticalLayer:  d_model=128, num_experts=4, active_experts=2
        - StrategicLayer: d_model=128, num_embeddings=64
        - InverseMonitor: threshold=0.01
        - BanachChecker:  d_model=128, d_cond=128

    Patch layout for 128x128 @ 8 frames with patch_size=(2,16,16):
        temporal: 8/2 = 4 patches
        spatial:  (128/16) * (128/16) = 8 * 8 = 64 patches
        total:    4 * 64 = 256 patches per clip
    """

    def __init__(
        self,
        d_model: int = 128,
        d_action: int = 4,
        num_embeddings: int = 64,
        num_experts: int = 4,
        active_experts: int = 2,
        lambda_sigreg: float = 0.1,
        threshold_op: float = 0.01,
        threshold_tac: float = 0.01,
        var_threshold: float = 1e-4,
        pinning_threshold: float = 0.999,
        encoder_depth: int = 4,
        encoder_num_heads: int = 4,
        encoder_patch_size: tuple = (2, 16, 16),
        encoder_max_patches: int = 512,
        complex_mode: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_action = d_action
        self.complex_mode = complex_mode

        # 1. Vision Encoder (3D tube patch embedding + ViT)
        self.encoder = VisionEncoder(
            d_model=d_model,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            patch_size=encoder_patch_size,
            max_patches=encoder_max_patches,
        )

        # 2. Hierarchical Core Layers
        self.strategic_layer = StrategicLayer(
            d_model=d_model,
            num_embeddings=num_embeddings,
            lambda_sigreg=lambda_sigreg,
        )
        self.tactical_layer = TacticalLayer(
            d_model=d_model,
            num_experts=num_experts,
            active_experts=active_experts,
            lambda_sigreg=lambda_sigreg,
        )
        self.operative_layer = OperativeLayer(
            d_model=d_model,
            d_action=d_action,
            lambda_sigreg=lambda_sigreg,
        )

        # 3. Checker-Refinement (Banach Contraction Loop)
        self.checker = BanachChecker(d_model=d_model, d_cond=d_model)

        # 4. Inverse Monitoring (Efference Copies)
        self.op_monitor = InverseMonitor(threshold=threshold_op)
        self.tac_monitor = InverseMonitor(threshold=threshold_tac)

        # 5. Out-of-Gradient Fallback Watchdog
        self.fallback = FallbackMonitor(
            var_threshold=var_threshold,
            pinning_threshold=pinning_threshold,
        )

        # 6. EBM Binder (unified energy landscape)
        self.ebm = EBMBinder()

        # 7. Langevin Planner (SDE action optimization — default params)
        self.langevin_planner = LangevinPlanner()

    def forward(
        self,
        video_frames: torch.Tensor,
        actions_onehot: torch.Tensor,
        prev_latents: torch.Tensor = None,
        force_system2: bool = False,
        current_position: torch.Tensor = None,
    ) -> dict:
        """
        Full ALPS forward pass with dynamic System 1 / System 2 gating.

        Args:
            video_frames:  [B, 3, T, 128, 128]  raw video input
            actions_onehot:[B, d_action]         one-hot action vector (4D)
            prev_latents:  [B, N, D] or None     previous step latents for pinning check
            force_system2: bool                  force full hierarchical deliberation
            current_position:[B, 2] or None      current 2D position of agent for MRC safe haven homing

        Returns:
            outputs dict with loss, predictions, diagnostics, etc.
        """
        outputs = {}

        # ── 1. ENCODE VISUAL INPUT & SHIFT TARGETS (True temporal dynamic learning) ──
        # JEPA requires context-to-target shift. We split clip into context (0..T-2) and target (1..T-1).
        if video_frames.dim() == 5 and video_frames.shape[2] > 1:
            context_frames = video_frames[:, :, :-1]
            target_frames = video_frames[:, :, 1:]
        else:
            context_frames = video_frames
            target_frames = video_frames

        z_target = self.encoder(target_frames)   # [B, N, D] Target representation at time t+1
        z_t = self.encoder(context_frames)       # [B, N, D] Context representation at time t

        # ── 2. SYSTEM INTEGRITY CHECK (out-of-gradient watchdog) ────────────
        # NaNs/Infs are always catastrophic and must trigger fallback to protect weights
        has_nan_inf = self.fallback.check_nan_inf(z_target) or self.fallback.check_nan_inf(z_t)
        if has_nan_inf:
            outputs["action"] = self.fallback.get_minimal_risk_action(actions_onehot, current_position=current_position, complex_mode=self.complex_mode)
            outputs["fallback_triggered"] = True
            outputs["system_healthy"] = False
            outputs["health_status"] = "NaN/Infinity detected in latent representations"
            outputs["system2_activated"] = False
            outputs["loss"] = torch.tensor(
                0.0, device=video_frames.device, requires_grad=True
            )
            outputs["energy"] = torch.tensor(10.0, device=video_frames.device)
            return outputs

        system_healthy, health_msg = self.fallback.verify_system_health(
            z_t, prev_latents
        )
        outputs["system_healthy"] = system_healthy
        outputs["health_status"] = health_msg

        # During training, we do NOT trigger fallback or early return on variance or pinning triggers.
        # Doing so would stop gradient updates, freezing the model weights in a collapsed state forever.
        # Instead, we let training proceed normally so that SIGReg can generate active gradients to recover the variance!
        # During evaluation (self.training is False), the strict early-return watchdog is fully enforced.
        if not system_healthy and not self.training:
            outputs["action"] = self.fallback.get_minimal_risk_action(actions_onehot, current_position=current_position, complex_mode=self.complex_mode)
            outputs["fallback_triggered"] = True
            outputs["system2_activated"] = False
            outputs["loss"] = torch.tensor(
                0.0, device=video_frames.device, requires_grad=True
            )
            outputs["energy"] = torch.tensor(10.0, device=video_frames.device)
            return outputs

        outputs["fallback_triggered"] = False

        # ── 3. SYSTEM 1: Operative prediction (fast, every step) ────────────
        # Initialize flat subgoal context as zeros for default System 1 pass
        flat_subgoal = torch.zeros_like(z_t)
        z_operative, sigreg_op = self.operative_layer(z_t, flat_subgoal)

        # Predict next latent state conditioned on action
        z_pred = self.operative_layer.predict_next_state(z_operative, actions_onehot)
        pred_loss_op = F.mse_loss(z_pred, z_target.detach())

        # ── 4. DYNAMIC COMPUTE GATING (Inverse Monitor interrupt check) ─────
        div_op, interrupt_op = self.op_monitor(z_pred, z_target)
        outputs["operative_interrupt"] = interrupt_op

        # If prediction error is within bounds and not forced → skip System 2
        if not interrupt_op and not force_system2:
            outputs["system2_activated"] = False
            outputs["z_t"] = z_t
            outputs["z_pred"] = z_pred
            outputs["loss"] = pred_loss_op + sigreg_op
            outputs["pred_loss_op"] = pred_loss_op
            outputs["sigreg_loss"] = sigreg_op
            outputs["vq_loss"] = torch.tensor(0.0, device=video_frames.device)
            outputs["moe_loss"] = torch.tensor(0.0, device=video_frames.device)
            outputs["contraction_loss"] = torch.tensor(
                0.0, device=video_frames.device
            )
            outputs["energy"] = self.ebm(
                torch.tensor(0.0, device=video_frames.device),
                torch.tensor(0.0, device=video_frames.device),
                pred_loss_op,
            )
            return outputs

        # ── 5. SYSTEM 2: Strategic → Tactical top-down cascade ──────────────
        outputs["system2_activated"] = True

        # 5a. Strategic Layer — discrete VQ conceptual bottleneck
        z_strategic, vq_loss, sigreg_str = self.strategic_layer(z_operative)
        outputs["strategic_activated"] = True

        # 5b. Tactical Layer — MoE routing conditioned on strategic guidance
        z_tactical, moe_loss, sigreg_tac = self.tactical_layer(
            z_operative, z_strategic
        )

        # Tactical Inverse Monitor check
        div_tac, interrupt_tac = self.tac_monitor(z_tactical, z_operative)
        outputs["tactical_interrupt"] = interrupt_tac

        # 5c. Banach Contraction Checker-Refinement
        z_refined, check_steps, converged = self.checker(z_tactical, z_strategic)
        contraction_loss = self.checker.compute_contraction_loss(
            z_tactical, z_strategic
        )
        # ── 5d. RE-INTEGRATE SYSTEM 2 GUIDANCE (Critical Fix) ───────────────
        # If System 2 was computed, we must re-calculate the Operative state 
        # using the refined tactical subgoal so the predictor learns to follow top-down plans.
        if force_system2 or interrupt_op:
            # Re-compute z_operative with refined tactical guidance
            z_operative, sigreg_op = self.operative_layer(z_t, z_refined.detach())
            
            # Re-predict next state using the guided operative state
            z_pred = self.operative_layer.predict_next_state(z_operative, actions_onehot)
                
            # Update the operative prediction loss
            pred_loss_op = F.mse_loss(z_pred, z_target.detach())

        # Encode target representations at target scale safely in the same operative domain
        with torch.no_grad():
            z_op_target, _ = self.operative_layer(z_target, torch.zeros_like(z_target))
            z_strategic_target, _, _ = self.strategic_layer(z_op_target)
            z_tactical_target, _, _ = self.tactical_layer(z_op_target, z_strategic_target)
            z_strategic_target = z_strategic_target.detach()
            z_tactical_target = z_tactical_target.detach()

        # ── 6. LOSS AGGREGATION & EBM BINDING ───────────────────────────────
        # Compute prediction energy errors across scales using the top-down predictive cascade (Oracle Fix)
        # 1. Strategic Layer predicts future concept based on current concept (self-conditioned)
        z_str_pred = self.strategic_layer.predict_next_concept(z_strategic, z_strategic)
        # 2. Tactical Layer predicts future subgoal conditioned on the PREDICTED future concept (detached to isolate layers)
        z_tac_pred = self.tactical_layer.predict_next_subgoal(z_tactical, z_str_pred.detach())

        pred_loss_str = F.mse_loss(z_str_pred, z_strategic_target.detach())
        pred_loss_tac = F.mse_loss(z_tac_pred, z_tactical_target.detach())

        total_sigreg = sigreg_op + sigreg_tac + sigreg_str
        loss_total = (
            pred_loss_op
            + pred_loss_tac
            + pred_loss_str
            + total_sigreg
            + vq_loss
            + moe_loss
            + contraction_loss
        )

        outputs["z_t"] = z_t
        outputs["z_pred"] = z_pred
        outputs["z_strategic"] = z_strategic
        outputs["loss"] = loss_total
        outputs["pred_loss_op"] = pred_loss_op
        outputs["sigreg_loss"] = total_sigreg
        outputs["vq_loss"] = vq_loss
        outputs["moe_loss"] = moe_loss
        outputs["contraction_loss"] = contraction_loss

        # Unified energy score
        outputs["energy"] = self.ebm(pred_loss_str, pred_loss_tac, pred_loss_op)

        return outputs


# ════════════════════════════════════════════════════════════════════════════════
#  Training Loop
# ════════════════════════════════════════════════════════════════════════════════

def _generate_synthetic_data(
    num_episodes: int = 10,
    episode_length: int = 50,
) -> dict:
    """
    Generates a small synthetic Two Rooms dataset for testing when no real
    data file is available. Returns a dict matching the .pt file format.
    """
    total = num_episodes * episode_length
    return {
        "frames": torch.rand(total, 3, 128, 128),
        "actions": torch.randint(0, 4, (total,)),
        "positions": torch.rand(total, 2) * 10.0,
        "room_ids": torch.randint(0, 2, (total,)),
        "episode_starts": torch.arange(0, total, episode_length, dtype=torch.long),
    }


def train_two_rooms(
    data_path: str = "data/two_rooms/two_rooms_dataset.pt",
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "results/two_rooms",
    d_model: int = 128,
    num_embeddings: int = 64,
    num_experts: int = 4,
    active_experts: int = 2,
    complex_mode: bool = False,
    lambda_sigreg: float = 0.1,
):
    """
    Main training function for the Two Rooms ALPS-4B benchmark.

    Args:
        data_path:  Path to pre-generated .pt dataset.
        epochs:     Number of training epochs.
        batch_size: Mini-batch size.
        lr:         Base learning rate.
        device:     'cuda' or 'cpu'.
        save_dir:   Directory for saving model checkpoints and logs.
    """
    print("=" * 72)
    print("  ALPS-4B Two Rooms Benchmark — Training")
    print("=" * 72)

    device = torch.device(device)
    print(f"  Device:      {device}")
    print(f"  Data path:   {data_path}")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch size:  {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Lambda SIGReg: {lambda_sigreg}")
    print(f"  Save dir:    {save_dir}")
    print()

    # ── 1. Dataset ──────────────────────────────────────────────────────────
    use_synthetic = False
    if not Path(data_path).exists():
        print(f"[WARNING] Data file not found: {data_path}")
        print("  → Generating synthetic fallback data (10 episodes × 50 frames) ...")
        import tempfile

        synth = _generate_synthetic_data(num_episodes=10, episode_length=50)
        tmp_dir = tempfile.mkdtemp()
        data_path = os.path.join(tmp_dir, "synthetic_two_rooms.pt")
        torch.save(synth, data_path)
        use_synthetic = True

    dataset = TwoRoomsDataset(data_path, clip_length=8, stride=4)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    num_batches = len(dataloader)
    print(f"  Clips:       {len(dataset)}")
    print(f"  Batches/epoch: {num_batches}")

    if len(dataset) < batch_size:
        print(
            f"\n[ERROR] Dataset has fewer clips ({len(dataset)}) than batch_size "
            f"({batch_size}). Reduce batch_size or generate more data."
        )
        return

    # ── 2. Model ────────────────────────────────────────────────────────────
    model = TwoRoomsALPS(
        d_model=d_model,
        d_action=4,
        num_embeddings=num_embeddings,
        num_experts=num_experts,
        active_experts=active_experts,
        lambda_sigreg=lambda_sigreg,
        encoder_depth=4,
        encoder_num_heads=4,
        encoder_patch_size=(2, 16, 16),
        encoder_max_patches=512,
        complex_mode=complex_mode,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ── 3. Optimizers (4 separate, matching train.py pattern) ───────────────
    optimizer_enc = optim.AdamW(model.encoder.parameters(), lr=lr)
    optimizer_op = optim.AdamW(model.operative_layer.parameters(), lr=lr)
    optimizer_tac = optim.AdamW(model.tactical_layer.parameters(), lr=lr)
    optimizer_str = optim.AdamW(model.strategic_layer.parameters(), lr=lr * 0.1)

    # ── 4. Phase-Shifted Scheduler ──────────────────────────────────────────
    scheduler = PhaseShiftedScheduler(k_tactical=4, k_strategic=16)

    # ── 5. Prepare save directory ───────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    training_log = []

    # ── 6. Training Loop ────────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print(f"  Starting training for {epochs} epochs ...")
    print(f"{'─' * 72}\n")

    model.train()
    optimizer_enc.zero_grad(set_to_none=True)
    optimizer_op.zero_grad(set_to_none=True)
    optimizer_tac.zero_grad(set_to_none=True)
    optimizer_str.zero_grad(set_to_none=True)
    
    prev_latents = None  # Track latent history for pinning detection
    global_step = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        # Epoch accumulators
        epoch_total_loss = 0.0
        epoch_pred_loss = 0.0
        epoch_sigreg_loss = 0.0
        epoch_vq_loss = 0.0
        epoch_moe_loss = 0.0
        epoch_system2_count = 0
        epoch_fallback_count = 0
        epoch_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            batch_start = time.perf_counter()
            global_step += 1

            # Move tensors to device
            video_frames = batch["video_frames"].to(device)  # [B, 3, T, H, W]
            actions_onehot = batch["actions_onehot"].to(device)  # [B, T, 4]

            # Use the mean action across the clip as the action conditioning
            # (the operative predictor expects [B, d_action])
            actions_mean = actions_onehot.mean(dim=1)  # [B, 4]

            # Phase-shifted scheduler step
            sched = scheduler.step()

            # Forward pass
            outputs = model(
                video_frames,
                actions_mean,
                prev_latents=prev_latents,
                force_system2=(epoch <= (2 if epochs <= 10 else 5)),  # Force System 2 for early warmup epochs
            )

            # Track latent history for pinning detection
            if outputs.get("z_t") is not None:
                prev_latents = outputs["z_t"].detach()

            # Skip gradient step if fallback triggered
            if outputs.get("fallback_triggered", False):
                epoch_fallback_count += 1
                continue

            loss = outputs["loss"]
            epoch_batches += 1

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Phase-shifted optimizer steps
            optimizer_enc.step()  # Encoder always updates
            optimizer_enc.zero_grad(set_to_none=True)
            
            if sched["update_operative"]:
                optimizer_op.step()
                optimizer_op.zero_grad(set_to_none=True)
                
            if sched["update_tactical"]:
                optimizer_tac.step()
                optimizer_tac.zero_grad(set_to_none=True)
                
            if sched["update_strategic"]:
                optimizer_str.step()
                optimizer_str.zero_grad(set_to_none=True)

            # Accumulate metrics
            loss_val = loss.item()
            epoch_total_loss += loss_val
            epoch_pred_loss += outputs.get(
                "pred_loss_op", torch.tensor(0.0)
            ).item()
            epoch_sigreg_loss += outputs.get(
                "sigreg_loss", torch.tensor(0.0)
            ).item()
            epoch_vq_loss += outputs.get("vq_loss", torch.tensor(0.0)).item()
            epoch_moe_loss += outputs.get("moe_loss", torch.tensor(0.0)).item()
            if outputs.get("system2_activated", False):
                epoch_system2_count += 1

            # Detailed progress every 50 batches
            if batch_idx % 50 == 0:
                batch_time = time.perf_counter() - batch_start
                pred = outputs.get("pred_loss_op", torch.tensor(0.0)).item()
                sigreg = outputs.get("sigreg_loss", torch.tensor(0.0)).item()
                vq = outputs.get("vq_loss", torch.tensor(0.0)).item()
                moe = outputs.get("moe_loss", torch.tensor(0.0)).item()
                s2 = "ON" if outputs.get("system2_activated", False) else "off"

                print(
                    f"  Epoch {epoch:03d}/{epochs:03d} | "
                    f"Batch {batch_idx:04d}/{num_batches:04d} | "
                    f"Loss: {loss_val:.4f} | "
                    f"Pred: {pred:.4f} | "
                    f"SIGReg: {sigreg:.4f} | "
                    f"VQ: {vq:.4f} | "
                    f"MoE: {moe:.4f} | "
                    f"Sys2: {s2} | "
                    f"Time: {batch_time:.3f}s"
                )

        # ── Epoch Summary ───────────────────────────────────────────────────
        epoch_time = time.perf_counter() - epoch_start
        safe_batches = max(1, epoch_batches)

        epoch_metrics = {
            "epoch": epoch,
            "total_loss": epoch_total_loss / safe_batches,
            "pred_loss_op": epoch_pred_loss / safe_batches,
            "sigreg_loss": epoch_sigreg_loss / safe_batches,
            "vq_loss": epoch_vq_loss / safe_batches,
            "moe_loss": epoch_moe_loss / safe_batches,
            "system2_activation_count": epoch_system2_count,
            "fallback_count": epoch_fallback_count,
            "batches_processed": epoch_batches,
            "epoch_time_seconds": round(epoch_time, 2),
        }
        training_log.append(epoch_metrics)

        print(
            f"\n  ── Epoch {epoch:03d} Summary ──"
            f"  Avg Loss: {epoch_metrics['total_loss']:.4f} | "
            f"  Pred: {epoch_metrics['pred_loss_op']:.4f} | "
            f"  SIGReg: {epoch_metrics['sigreg_loss']:.4f} | "
            f"  VQ: {epoch_metrics['vq_loss']:.4f} | "
            f"  MoE: {epoch_metrics['moe_loss']:.4f} | "
            f"  Sys2 activations: {epoch_system2_count}/{epoch_batches} | "
            f"  Time: {epoch_time:.1f}s\n"
        )

        # ── Periodic Checkpoint ─────────────────────────────────────────────
        if epoch % 10 == 0:
            ckpt_path = os.path.join(save_dir, f"two_rooms_model_epoch{epoch:03d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_enc": optimizer_enc.state_dict(),
                    "optimizer_op": optimizer_op.state_dict(),
                    "optimizer_tac": optimizer_tac.state_dict(),
                    "optimizer_str": optimizer_str.state_dict(),
                    "training_log": training_log,
                },
                ckpt_path,
            )
            print(f"  💾 Checkpoint saved: {ckpt_path}")
            
            # Auto-push to GitHub
            print(f"  ⬆️ Pushing checkpoint {epoch} to GitHub...")
            try:
                # Configure git user identity locally in repository if not set
                subprocess.run(["git", "config", "user.email", "alps4b@runpod.io"], check=True, cwd=save_dir)
                subprocess.run(["git", "config", "user.name", "ALPS-4B H100 Runner"], check=True, cwd=save_dir)
                subprocess.run(["git", "add", os.path.basename(ckpt_path)], check=True, cwd=save_dir)
                subprocess.run(["git", "commit", "-m", f"Auto-save checkpoint {os.path.basename(ckpt_path)}"], check=True, cwd=save_dir)
                
                # Push using the PAT from environment
                pat = os.environ.get("GITHUB_PAT")
                if pat:
                    remote_url = f"https://oauth2:{pat}@github.com/4qdrai/4B-JEPA.git"
                    subprocess.run(["git", "push", remote_url, "main"], check=True, cwd=save_dir)
                    print("  ✅ Successfully pushed to GitHub.")
                else:
                    print("  ⚠️ GITHUB_PAT environment variable not set. Skipping push.")
            except subprocess.CalledProcessError as e:
                print(f"  ❌ Failed to push to GitHub: {e}")

    # ── 7. Final Save ───────────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print("  Training Complete!")
    print(f"{'═' * 72}")

    model_name = "two_rooms_model_complex.pt" if complex_mode else "two_rooms_model.pt"
    final_model_path = os.path.join(save_dir, model_name)
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_enc": optimizer_enc.state_dict(),
            "optimizer_op": optimizer_op.state_dict(),
            "optimizer_tac": optimizer_tac.state_dict(),
            "optimizer_str": optimizer_str.state_dict(),
            "training_log": training_log,
        },
        final_model_path,
    )
    print(f"  💾 Final model saved: {final_model_path}")

    # Auto-push final model to GitHub
    print(f"  ⬆️ Pushing final model to GitHub...")
    try:
        # Configure git user identity locally in repository if not set
        subprocess.run(["git", "config", "user.email", "alps4b@runpod.io"], check=True, cwd=save_dir)
        subprocess.run(["git", "config", "user.name", "ALPS-4B H100 Runner"], check=True, cwd=save_dir)
        subprocess.run(["git", "add", os.path.basename(final_model_path)], check=True, cwd=save_dir)
        subprocess.run(["git", "commit", "-m", f"Auto-save final model {os.path.basename(final_model_path)}"], check=True, cwd=save_dir)
        pat = os.environ.get("GITHUB_PAT")
        if pat:
            remote_url = f"https://oauth2:{pat}@github.com/4qdrai/4B-JEPA.git"
            subprocess.run(["git", "push", remote_url, "main"], check=True, cwd=save_dir)
            print("  ✅ Successfully pushed final model to GitHub.")
        else:
            print("  ⚠️ GITHUB_PAT environment variable not set. Skipping push.")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed to push final model to GitHub: {e}")

    print("\n  Training completely finished.\n") if complex_mode else None
    log_name = "training_log_complex.json" if complex_mode else "training_log.json"
    log_path = os.path.join(save_dir, log_name)
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"  Training log saved: {log_path}")

    # Print final summary
    if training_log:
        final = training_log[-1]
        print(f"\n  Final epoch metrics:")
        print(f"    Total Loss:       {final['total_loss']:.6f}")
        print(f"    Pred Loss (Op):   {final['pred_loss_op']:.6f}")
        print(f"    SIGReg Loss:      {final['sigreg_loss']:.6f}")
        print(f"    VQ Loss:          {final['vq_loss']:.6f}")
        print(f"    MoE Loss:         {final['moe_loss']:.6f}")
        print(f"    System 2 count:   {final['system2_activation_count']}")

    return model, training_log


# ════════════════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ALPS-4B Two Rooms Benchmark Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/two_rooms/two_rooms_dataset.pt",
        help="Path to pre-generated .pt dataset file",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Base learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cuda/cpu)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/two_rooms",
        help="Directory for model checkpoints and logs",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Latent model dimension size (e.g. 128 or 384)",
    )
    parser.add_argument(
        "--num-embeddings",
        type=int,
        default=64,
        help="VQ concept codebook size (e.g. 64 or 512)",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=4,
        help="Number of MoE routing experts (e.g. 4 or 8)",
    )
    parser.add_argument(
        "--active-experts",
        type=int,
        default=2,
        help="Active experts per token (e.g. 2)",
    )
    parser.add_argument(
        "--complex-mode",
        action="store_true",
        help="Enable 4-room complex navigation mode with locked doors and keys",
    )
    parser.add_argument(
        "--lambda-sigreg",
        type=float,
        default=0.1,
        help="Collapse prevention weight (sigreg loss multiplier)",
    )
    args = parser.parse_args()

    train_two_rooms(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
        d_model=args.d_model,
        num_embeddings=args.num_embeddings,
        num_experts=args.num_experts,
        active_experts=args.active_experts,
        complex_mode=args.complex_mode,
        lambda_sigreg=args.lambda_sigreg,
    )


if __name__ == "__main__":
    main()
