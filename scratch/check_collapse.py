import sys
import os
sys.path.insert(0, "src")
import torch
from alps.benchmarks.two_rooms.train_two_rooms import TwoRoomsALPS
from alps.benchmarks.two_rooms.dataset import TwoRoomsDataset
from torch.utils.data import DataLoader

def check_model():
    print("Loading model...")
    checkpoint = torch.load("results/two_rooms/two_rooms_model.pt", map_location="cpu")
    
    # Initialize model
    model = TwoRoomsALPS(
        d_model=128,
        d_action=4,
        num_embeddings=64,
        num_experts=4,
        active_experts=2,
        lambda_sigreg=0.1,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    
    # Generate 16 distinct random video clips (each having 3 channels, 8 frames, 128x128 resolution)
    video_frames = torch.rand(16, 3, 8, 128, 128)
    
    with torch.no_grad():
        z_t = model.encoder(video_frames) # [B, N, D]
        
    print(f"z_t shape: {z_t.shape}")
    
    # Calculate variance across batch dimension
    # z_t: [B, N, D] -> [16, 256, 128]
    var_across_batch = torch.var(z_t, dim=0) # [N, D]
    mean_var_across_batch = var_across_batch.mean().item()
    
    # Calculate variance across token dimension
    var_across_tokens = torch.var(z_t, dim=1) # [B, D]
    mean_var_across_tokens = var_across_tokens.mean().item()
    
    # Calculate variance across all combined
    mean_var_combined = torch.var(z_t, dim=(0, 1)).mean().item()
    
    print(f"Mean variance across BATCH dimension (content variance):  {mean_var_across_batch:.8f}")
    print(f"Mean variance across TOKENS dimension (spatial variance):  {mean_var_across_tokens:.8f}")
    print(f"Mean variance across COMBINED B+N (what SIGReg sees):     {mean_var_combined:.8f}")
    
    # Let's check similarity between different inputs in the batch
    # z_t: [B, N, D]. Let's average over tokens to get a single vector per batch element: [B, D]
    z_t_mean = z_t.mean(dim=1)
    norm_z = torch.nn.functional.normalize(z_t_mean, p=2, dim=-1)
    cos_sim_matrix = torch.matmul(norm_z, norm_z.t())
    
    print("\nCosine similarity matrix between first 4 batch elements (different random inputs):")
    print(cos_sim_matrix[:4, :4])

if __name__ == "__main__":
    check_model()
