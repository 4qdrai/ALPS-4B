import torch
import torch.nn as nn
import torch.nn.functional as F

from alps.core.encoders import VisionEncoder
from alps.core.hierarchy import StrategicLayer, TacticalLayer, OperativeLayer
from alps.core.inverse_monitor import InverseMonitor
from alps.core.checker import BanachChecker
from alps.core.fallback import FallbackMonitor
from alps.core.energy import EBMBinder
from alps.core.predictor import LangevinPlanner

from alps.multimodal.sensor_encoders import IMUEncoder, LidarEncoder
from alps.multimodal.modality_router import ModalityRouter

class ALPSModel(nn.Module):
    """
    ALPS-4B (Adaptive Latent Prediction System, Four-Brain)
    
    The master neural network orchestrator for the SPRIND Next Frontier AI Challenge.
    Integrates:
    - Spatiotemporal Video Encoding (ViT, 16 frames x 224x224)
    - 3-Tier Multi-Scale Decoupled JEPA layers (Strategic / Tactical / Operative)
    - Dynamic Compute Gating (System 1 vs System 2 compute allocation)
    - Multimodal selective sensor gating (O(1) modality routing)
    - Banach Contraction Checker-Refinement Loop
    - Bottom-Up Inverse Self-Monitoring (Efference Copy Verification)
    - "No-Retraining" Zero-Shot Self-Learning Loop
    - Out-of-Gradient Fallback Watchdog (NaN, Var, Pinning) with Lyapunov-stable MRC
    - Unified EBM Binding landscape
    """
    def __init__(self, d_model: int = 384, num_embeddings: int = 512, 
                 num_experts: int = 8, active_experts: int = 2, d_action: int = 64,
                 lambda_sigreg: float = 0.1, threshold_op: float = 0.01, 
                 threshold_tac: float = 0.01, var_threshold: float = 1e-4, 
                 pinning_threshold: float = 0.999,
                 encoder_depth: int = 8, encoder_num_heads: int = 6,
                 encoder_patch_size: tuple = (2, 16, 16), encoder_max_patches: int = 2048,
                 use_langevin: bool = False, langevin_steps: int = 5,
                 langevin_lr: float = 0.05, langevin_sigma: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.use_langevin = use_langevin
        
        # 1. Vision Encoder (tube patch embedding + ViT)
        self.encoder = VisionEncoder(
            d_model=d_model, depth=encoder_depth, num_heads=encoder_num_heads,
            patch_size=encoder_patch_size, max_patches=encoder_max_patches
        )
        
        # 2. Hierarchical Core Layers
        self.strategic_layer = StrategicLayer(
            d_model=d_model, num_embeddings=num_embeddings, lambda_sigreg=lambda_sigreg
        )
        self.tactical_layer = TacticalLayer(
            d_model=d_model, num_experts=num_experts, active_experts=active_experts, 
            lambda_sigreg=lambda_sigreg
        )
        self.operative_layer = OperativeLayer(
            d_model=d_model, d_action=d_action, lambda_sigreg=lambda_sigreg
        )
        
        # 3. Multimodal Encoders and Modality Router (O(1) Modality Gating)
        self.imu_encoder = IMUEncoder(d_model=d_model)
        self.lidar_encoder = LidarEncoder(d_model=d_model)
        self.modality_router = ModalityRouter(d_model=d_model)
        
        # 4. Checker-Refinement (Banach Contraction Loop)
        self.checker = BanachChecker(d_model=d_model, d_cond=d_model)
        
        # 5. Inverse Monitoring (Efference Copies)
        self.op_monitor = InverseMonitor(threshold=threshold_op)
        self.tac_monitor = InverseMonitor(threshold=threshold_tac)
        
        # 6. Out-of-Gradient Fallback watchdogs
        self.fallback = FallbackMonitor(var_threshold=var_threshold, pinning_threshold=pinning_threshold)
        
        # 7. EBM Binder
        self.ebm = EBMBinder()
        
        # 8. Langevin Planner SDE Action Optimization
        self.langevin_planner = LangevinPlanner(steps=langevin_steps, lr=langevin_lr, sigma=langevin_sigma)
        
        # 9. Multi-Scale Loss Module
        from alps.training.multi_scale_loss import MultiScaleLoss
        self.criterion = MultiScaleLoss(alpha=1.0, beta=1.0, gamma=1.0)
        
    def forward(self, video_frames: torch.Tensor, actions: torch.Tensor, 
                 prev_latents: torch.Tensor = None, imu_telemetry: torch.Tensor = None,
                 lidar_points: torch.Tensor = None, force_system2: bool = False,
                 mask_indices: torch.Tensor = None, update_tactical: bool = True,
                 update_strategic: bool = True) -> dict:
        """
        Orchestrates a complete multi-scale predictive step with dynamic compute gating
        and multimodal selective gating.
        
        Args:
            video_frames: Raw video inputs, Shape: [B, C, T, H, W]
            actions: Actuator commands, Shape: [B, D_action]
            prev_latents: Previous step's latent representation [B, N, D] for pinning checks
            imu_telemetry: Optional raw IMU inputs, Shape: [B, 6, 100]
            lidar_points: Optional raw LiDAR inputs, Shape: [B, 1, 360]
            force_system2: If True, forces full hierarchical deliberation regardless of confidence
            mask_indices: Optional boolean mask of shape [B, N] where True indicates patches
                          to KEEP (unmasked), and False indicates patches to MASK.
            update_tactical: Flag from scheduler to run backprop for Tactical Layer.
            update_strategic: Flag from scheduler to run backprop for Strategic Layer.
        """
        # --- 1. ENCODE VISUAL INPUT & SHIFT TARGETS (True temporal dynamic learning) ---
        outputs = {}
        
        # JEPA requires context-to-target shift. We split clip into context (0..T-2) and target (1..T-1).
        if video_frames.dim() == 5 and video_frames.shape[2] > 1:
            context_frames = video_frames[:, :, :-1]
            target_frames = video_frames[:, :, 1:]
        else:
            context_frames = video_frames
            target_frames = video_frames
            
        z_target = self.encoder(target_frames) # [B, N, D] Target representation at time t+1
        z_t_full = self.encoder(context_frames) # [B, N, D] Raw context representation at time t
        
        # Apply spatiotemporal tube masking if provided (Phantom Masker Fix)
        # We zero out representations at masked positions to simulate missing inputs.
        # We dynamically pad/align the mask if the encoder added extra tokens (like [CLS] tokens).
        if mask_indices is not None:
            B, N, D = z_t_full.shape
            mask_len = mask_indices.shape[1]
            
            # If the encoder added a CLS token or extra tokens (N > mask_len)
            if N > mask_len:
                diff = N - mask_len
                # Pad the start of the mask with True (Keep the CLS tokens)
                pad_mask = torch.ones(B, diff, dtype=torch.bool, device=mask_indices.device)
                aligned_mask = torch.cat([pad_mask, mask_indices], dim=1)
            else:
                aligned_mask = mask_indices
                
            z_t = z_t_full * aligned_mask[:, :N].unsqueeze(-1)
        else:
            z_t = z_t_full
            
        # --- 2. SYSTEM INTEGRITY VERIFICATION (Fallback watchdog - out of gradient) ---
        # NaNs/Infs are always catastrophic and must trigger fallback to protect weights
        has_nan_inf = self.fallback.check_nan_inf(z_target) or self.fallback.check_nan_inf(z_t)
        if has_nan_inf:
            outputs["action"] = self.fallback.get_minimal_risk_action(actions)
            outputs["fallback_triggered"] = True
            outputs["system_healthy"] = False
            outputs["health_status"] = "NaN/Infinity detected in latent representations"
            outputs["system2_activated"] = False
            outputs["loss"] = torch.tensor(0.0, device=video_frames.device, requires_grad=True)
            outputs["energy"] = torch.tensor(10.0, device=video_frames.device)
            return outputs

        system_healthy, health_msg = self.fallback.verify_system_health(z_t, prev_latents)
        outputs["system_healthy"] = system_healthy
        outputs["health_status"] = health_msg
        
        # During training, we do NOT trigger fallback or early return on variance or pinning triggers.
        # Doing so would stop gradient updates, freezing the model weights in a collapsed state forever.
        # Instead, we let training proceed normally so that SIGReg can generate active gradients to recover the variance!
        # During evaluation (self.training is False), the strict early-return watchdog is fully enforced.
        if not system_healthy and not self.training:
            # Watchdog trigger! Bypasses all forward planning, runs Lyapunov MRC
            outputs["action"] = self.fallback.get_minimal_risk_action(actions)
            outputs["fallback_triggered"] = True
            outputs["system2_activated"] = False
            outputs["loss"] = torch.tensor(0.0, device=video_frames.device, requires_grad=True)
            outputs["energy"] = torch.tensor(10.0, device=video_frames.device)
            return outputs
            
        outputs["fallback_triggered"] = False
        
        # --- 3. SYSTEM 1 PROCESSING (Millisecond-frequency Operative prediction) ---
        # Initialize flat subgoal context as zeros for default System 1 pass
        flat_subgoal = torch.zeros_like(z_t)
        z_operative, sigreg_op = self.operative_layer(z_t, flat_subgoal)
        
        # Multimodal sensor selective gating (O(1) modality routing)
        if imu_telemetry is not None and lidar_points is not None:
            imu_emb = self.imu_encoder(imu_telemetry)
            lidar_emb = self.lidar_encoder(lidar_points)
            
            # Route and fuse dynamically based on top-level sensory attention
            fused_multimodal, active_mask = self.modality_router(z_t.mean(dim=1), [imu_emb, lidar_emb])
            z_operative = z_operative + fused_multimodal
            outputs["active_modalities"] = active_mask
            
        # Predict the next state
        if self.use_langevin:
            actions_refined = self.langevin_planner.plan(
                self.operative_layer.predict_next_state, z_operative, z_target.detach(), actions
            )
            outputs["refined_actions"] = actions_refined
            z_pred = self.operative_layer.predict_next_state(z_operative, actions_refined)
        else:
            z_pred = self.operative_layer.predict_next_state(z_operative, actions)
            
        pred_loss_op = F.mse_loss(z_pred, z_target.detach())
        
        # --- 4. DYNAMIC COMPUTE GATING & INTERRUPT ESCALATION ---
        # Verify prediction divergence using the Efference Copy Inverse Monitor
        div_op, interrupt_op = self.op_monitor(z_pred, z_target)
        outputs["operative_interrupt"] = interrupt_op
        
        # If the predictive error is within bounds and not forced, bypass System 2 completely
        if not interrupt_op and not force_system2:
            outputs["system2_activated"] = False
            outputs["z_t"] = z_t
            outputs["z_pred"] = z_pred
            outputs["loss"] = pred_loss_op + sigreg_op
            outputs["sigreg_loss"] = sigreg_op
            outputs["energy"] = self.ebm(
                torch.tensor(0.0, device=video_frames.device), 
                torch.tensor(0.0, device=video_frames.device), 
                pred_loss_op
            )
            return outputs
            
        # Escalation: System 2 is activated!
        outputs["system2_activated"] = True
        
        # --- 5. SYSTEM 2 PLANNING (Strategic → Tactical top-down cascade) ---
        # 5a. Strategic Layer FIRST (Discrete VQ conceptual planning bottleneck)
        # Actively bypass/run without gradients if the scheduler is not updating the Strategic Layer
        if update_strategic or force_system2:
            z_strategic, vq_loss, sigreg_str = self.strategic_layer(z_operative)
        else:
            with torch.no_grad():
                z_strategic, vq_loss, sigreg_str = self.strategic_layer(z_operative)
            z_strategic = z_strategic.detach()
            vq_loss = vq_loss.detach()
            sigreg_str = sigreg_str.detach()
            
        outputs["strategic_activated"] = True
        
        # 5b. Tactical Layer receives strategic output as top-down guidance
        # Actively bypass/run without gradients if the scheduler is not updating the Tactical Layer
        if update_tactical or force_system2:
            z_tactical, moe_loss, sigreg_tac = self.tactical_layer(z_operative, z_strategic)
        else:
            with torch.no_grad():
                z_tactical, moe_loss, sigreg_tac = self.tactical_layer(z_operative, z_strategic)
            z_tactical = z_tactical.detach()
            moe_loss = moe_loss.detach()
            sigreg_tac = sigreg_tac.detach()
        
        # Tactical Inverse Monitor check
        div_tac, interrupt_tac = self.tac_monitor(z_tactical, z_operative)
        outputs["tactical_interrupt"] = interrupt_tac
            
        # 5c. Banach Contraction Checker-Refinement
        # Refines the tactical sub-goals under strategic constraints
        z_refined, check_steps, converged = self.checker(z_tactical, z_strategic)
        contraction_loss = self.checker.compute_contraction_loss(z_tactical, z_strategic)
        
        # Encode target representations at target scale to calculate top-down predictive target losses
        with torch.no_grad():
            z_strategic_target, _, _ = self.strategic_layer(z_target)
            z_tactical_target, _, _ = self.tactical_layer(z_target, z_strategic_target)
            z_strategic_target = z_strategic_target.detach()
            z_tactical_target = z_tactical_target.detach()
        
        # --- 5d. RE-INTEGRATE SYSTEM 2 GUIDANCE (Critical Fix) ---
        # If System 2 was computed, we must re-calculate the Operative state 
        # using the actual tactical subgoal so the predictor learns to follow top-down plans.
        if update_tactical or force_system2 or interrupt_op:
            # Re-compute z_operative with actual tactical guidance
            z_operative, sigreg_op = self.operative_layer(z_t, z_tactical)
            
            # Re-compute multimodal routing on the updated z_operative
            if imu_telemetry is not None and lidar_points is not None:
                z_operative = z_operative + fused_multimodal
                
            # Re-predict next state using the guided operative state
            if self.use_langevin:
                z_pred = self.operative_layer.predict_next_state(z_operative, actions_refined)
            else:
                z_pred = self.operative_layer.predict_next_state(z_operative, actions)
                
            # Update the operative prediction loss
            pred_loss_op = F.mse_loss(z_pred, z_target.detach())
        
        # --- 6. AUTOMATIC LATENT-RAG WRITE (Self-Learning Loop) ---
        # When operative prediction diverges significantly, auto-write the correction to RAG
        if pred_loss_op.item() > 0.5 and not self.training:
            delta_z = z_t - z_pred  # correction vector
            context_key = z_operative.mean(dim=1)  # [B, D] semantic context
            self.tactical_layer.rag.write_memory(context_key, delta_z.mean(dim=1))
            outputs["rag_auto_write"] = True
        else:
            outputs["rag_auto_write"] = False
        
        # --- 7. LOSS AGGREGATION & EBM BINDING ---
        # Compute prediction energy errors across scales using the top-down predictive cascade (Oracle Fix)
        # 1. Strategic Layer predicts future concept based on current concept (self-conditioned)
        z_str_pred = self.strategic_layer.predict_next_concept(z_strategic, z_strategic)
        # 2. Tactical Layer predicts future subgoal conditioned on the PREDICTED future concept (detached to isolate layers)
        z_tac_pred = self.tactical_layer.predict_next_subgoal(z_tactical, z_str_pred.detach())
        
        # Invoke our dedicated MultiScaleLoss module to cleanly aggregate prediction and regularization errors
        loss_dict = self.criterion(
            z_op=z_target, z_op_pred=z_pred, sigreg_op=sigreg_op,
            z_tac=z_tactical_target, z_tac_pred=z_tac_pred, sigreg_tac=sigreg_tac, moe_loss=moe_loss,
            z_str=z_strategic_target, z_str_pred=z_str_pred, sigreg_str=sigreg_str, vq_loss=vq_loss
        )
        loss_total = loss_dict["loss"] + contraction_loss
        
        outputs["z_t"] = z_t
        outputs["z_pred"] = z_pred
        outputs["z_strategic"] = z_strategic
        outputs["loss"] = loss_total
        outputs["pred_loss_op"] = pred_loss_op
        outputs["sigreg_loss"] = sigreg_op + sigreg_tac + sigreg_str
        outputs["vq_loss"] = vq_loss
        outputs["moe_loss"] = moe_loss
        outputs["contraction_loss"] = contraction_loss
        
        # Unified Energy score
        outputs["energy"] = self.ebm(loss_dict["str_pred_mse"], loss_dict["tac_pred_mse"], loss_dict["op_pred_mse"])
        
        return outputs

    def execute_self_learning_loop(self, video_frames: torch.Tensor, actions: torch.Tensor, 
                                   target_z: torch.Tensor) -> dict:
        """
        Executes the lifelong 'No-Retraining' self-supervised learning write loop.
        
        1. Surprise Detection: Inverse Monitoring triggers on prediction failure.
        2. Mental Simulation: Langevin SDE plans corrected trajectory to minimize energy.
        3. Calculating Correction: Computes mathematical error delta-z.
        4. Database Write: Inserts context and correction directly into Latent-RAG KV store.
        """
        self.eval()
        outputs = {}
        
        # 1. Sense and predict
        z_t = self.encoder(video_frames)
        z_op, _ = self.operative_layer(z_t, torch.zeros_like(z_t))
        z_pred = self.operative_layer.predict_next_state(z_op, actions)
        
        div, interrupt = self.op_monitor(z_pred, target_z)
        outputs["surprise_detected"] = interrupt
        
        if interrupt:
            # 2. SDE Mental Simulation: refine actions to reach target_z
            actions_corrected = self.langevin_planner.plan(
                self.operative_layer.predict_next_state, z_op, target_z, actions
            )
            z_pred_corrected = self.operative_layer.predict_next_state(z_op, actions_corrected)
            
            # 3. Calculate Correction vector (delta-z)
            delta_z = target_z - z_pred_corrected
            
            # 4. Write context-correction directly to Latent-RAG cache
            self.tactical_layer.rag.write_memory(z_op.mean(dim=1), delta_z.mean(dim=1))
            
            outputs["learning_triggered"] = True
            outputs["original_action"] = actions
            outputs["corrected_action"] = actions_corrected
            outputs["written_correction"] = delta_z
        else:
            outputs["learning_triggered"] = False
            
        return outputs
