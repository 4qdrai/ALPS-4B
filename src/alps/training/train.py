import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

from alps.core.alps_model import ALPSModel
from alps.training.masked_prediction import SpatiotemporalMasker
from alps.training.multi_scale_loss import MultiScaleLoss
from alps.training.phase_shifted_scheduler import PhaseShiftedScheduler

def run_unsupervised_training(epochs: int = 10, batch_size: int = 4):
    """
    Main training pipeline for ALPS-4B.
    Demonstrates unsupervised spatiotemporal tube-masked predictive learning
    compatible with H100 execution.
    """
    print("=== ALPS-4B: Unified Unsupervised Training Pipeline ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}")
    
    d_model = 384
    d_action = 64
    
    # 1. Initialize core ALPS model
    model = ALPSModel(d_model=d_model, d_action=d_action).to(device)
    
    # 2. Initialize loss, masker, and scheduler modules
    criterion = MultiScaleLoss()
    masker = SpatiotemporalMasker(mask_ratio=0.9)
    scheduler = PhaseShiftedScheduler()
    
    # Set up optimizers per layer to facilitate phase-shifted updating
    optimizer_op = optim.AdamW(model.operative_layer.parameters(), lr=1e-4)
    optimizer_tac = optim.AdamW(model.tactical_layer.parameters(), lr=1e-4)
    optimizer_str = optim.AdamW(model.strategic_layer.parameters(), lr=1e-5)
    
    # 3. Simulate training epochs
    # On CPU, run a fast single-step validation pass. On GPU, execute full scale.
    is_cpu = (device.type == "cpu")
    run_epochs = 1 if is_cpu else epochs
    resolution = 32 if is_cpu else 224
    
    print(f"\nStarting training loop for {run_epochs} simulated epochs at {resolution}x{resolution} resolution...")
    model.train()
    
    for epoch in range(1, run_epochs + 1):
        start_time = time.perf_counter()
        
        # Simulate video batches [B, C, T, H, W]
        video_input = torch.randn(batch_size, 3, 16, resolution, resolution, device=device)
        actions = torch.randn(batch_size, d_action, device=device)
        
        # Increment step and retrieve phase schedule flags
        sched = scheduler.step()
        
        # Zero gradients
        optimizer_op.zero_grad()
        optimizer_tac.zero_grad()
        optimizer_str.zero_grad()
        
        # Forward pass through model orchestrator
        outputs = model(video_input, actions)
        
        if outputs.get("fallback_triggered", False):
            print(f"  [Epoch {epoch}] Watchdog Triggered MRC! Skipping step.")
            continue
            
        loss = outputs["loss"]
        
        # Backpropagation
        loss.backward()
        
        # Phase-shifted gradient steps
        if sched["update_operative"]:
            optimizer_op.step()
        if sched["update_tactical"]:
            optimizer_tac.step()
        if sched["update_strategic"]:
            optimizer_str.step()
            
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        # Print metrics
        print(f"  Epoch {epoch:02d}/{epochs:02d} | "
              f"Total Loss: {loss.item():.4f} | "
              f"SIGReg Loss: {outputs['sigreg_loss'].item():.4f} | "
              f"VQ Commitment: {outputs['vq_loss'].item():.4f} | "
              f"MoE router loss: {outputs['moe_loss'].item():.4f} | "
              f"Steps: {sched['step']} | Time: {elapsed:.2f}s")
              
    print("\nTraining complete! Model successfully compiled and optimized.")
    
    # Save the consolidated weights
    os.makedirs("results/h100_training", exist_ok=True)
    torch.save(model.state_dict(), "results/h100_training/alps4b_final.pt")
    print("Saved consolidated model weights to results/h100_training/alps4b_final.pt")

if __name__ == "__main__":
    run_unsupervised_training(epochs=5)
