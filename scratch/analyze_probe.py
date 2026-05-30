import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from alps.benchmarks.two_rooms.dataset import TwoRoomsDataset
from alps.benchmarks.two_rooms.train_two_rooms import TwoRoomsALPS

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "results/two_rooms/two_rooms_model_epoch010_384.pt"
    data_path = "data/two_rooms/trajectories.pt"

    # 1. Load model
    print(f"Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # Infer d_model from pos_embed shape
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    d_model = state_dict["encoder.pos_embed"].shape[-1]
    print(f"Inferred d_model: {d_model}")
    
    model = TwoRoomsALPS(
        d_model=d_model,
        d_action=4,
        num_embeddings=64,
        num_experts=4,
        active_experts=2,
        complex_mode=False, # H100 was trained on baseline environment
    ).to(device)
    
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Model loaded successfully with strict=True.")
    except Exception as e:
        print(f"Warning: strict=True failed: {e}")
        print("Retrying load with strict=False...")
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully with strict=False.")
        
    model.eval()

    # 2. Load dataset
    print(f"Loading dataset from {data_path}...")
    dataset = TwoRoomsDataset(data_path, clip_length=8, stride=4)
    print(f"Dataset loaded: {len(dataset)} clips.")

    # 3. Analyze Latent Space Variance and Norms
    print("\nAnalyzing latent space representations...")
    latents = []
    positions = []
    
    # Run encoder on first 20 batches (approx 640 samples)
    with torch.no_grad():
        for i in range(min(20, len(dataset) // 32)):
            batch_frames = []
            batch_pos = []
            for j in range(32):
                idx = i * 32 + j
                sample = dataset[idx]
                batch_frames.append(sample["video_frames"])
                batch_pos.append(sample["positions"][-1])
                
            video_frames = torch.stack(batch_frames).to(device)
            z = model.encoder(video_frames)
            if z.dim() == 3:
                z = z.mean(dim=1)
            latents.append(z.cpu())
            positions.append(torch.stack(batch_pos))

    all_z = torch.cat(latents, dim=0) # [N, D]
    all_y = torch.cat(positions, dim=0) # [N, 2]
    
    print(f"Latents shape: {all_z.shape}")
    print(f"Mean of z: {all_z.mean().item():.6f}")
    print(f"Std of z: {all_z.std().item():.6f}")
    print(f"Min of z: {all_z.min().item():.6f}")
    print(f"Max of z: {all_z.max().item():.6f}")
    
    # Check pairwise distances or cosine similarity to detect representation collapse
    norms = torch.norm(all_z, dim=1)
    print(f"Mean norm of z: {norms.mean().item():.6f}, std of norm: {norms.std().item():.6f}")
    
    normalized_z = F.normalize(all_z, p=2, dim=1)
    cos_sim = torch.mm(normalized_z, normalized_z.t())
    print(f"Mean cosine similarity between random pairs: {cos_sim.mean().item():.6f}")
    print(f"Std of cosine similarity: {cos_sim.std().item():.6f}")
    print(f"Min cosine similarity: {cos_sim.min().item():.6f}")
    print(f"Max cosine similarity: {cos_sim.max().item():.6f}")

    # 4. Check position distributions
    print(f"\nPositions shape: {all_y.shape}")
    print(f"Mean position: X={all_y[:, 0].mean().item():.2f}, Y={all_y[:, 1].mean().item():.2f}")
    print(f"Std position: X={all_y[:, 0].std().item():.2f}, Y={all_y[:, 1].std().item():.2f}")
    
    # Variance of positions (which is the baseline MSE if predicting constant mean)
    mean_y = all_y.mean(dim=0, keepdim=True)
    baseline_mse = torch.mean((all_y - mean_y)**2).item()
    print(f"Baseline MSE (predicting mean): {baseline_mse:.4f}")

if __name__ == "__main__":
    main()
