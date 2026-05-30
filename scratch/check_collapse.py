import sys
import os
sys.path.insert(0, "src")
import torch
from alps.benchmarks.two_rooms.train_two_rooms import TwoRoomsALPS

def check_model():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "results/two_rooms/two_rooms_model.pt"
    if not os.path.exists(model_path):
        print(f"Error: checkpoint {model_path} not found.")
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Analyzing checkpoint: {model_path}")
    print(f"Running on local device: {device.upper()}")
    
    checkpoint = torch.load(model_path, map_location=device)
    sd = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    
    # Dynamically detect d_model from pos_embed shape
    if "encoder.pos_embed" in sd:
        d_model = sd["encoder.pos_embed"].shape[-1]
    else:
        d_model = 128  # fallback
    print(f"Detected d_model: {d_model}")
    
    # Initialize model
    model = TwoRoomsALPS(
        d_model=d_model,
        d_action=4,
        num_embeddings=64,
        num_experts=4,
        active_experts=2,
        lambda_sigreg=0.1,
    )
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    
    # Generate 16 distinct random video clips (each having 3 channels, 8 frames, 128x128 resolution)
    video_frames = torch.rand(16, 3, 8, 128, 128, device=device)
    
    with torch.no_grad():
        z_t = model.encoder(video_frames) # [B, N, D]
        
    print(f"z_t shape: {z_t.shape}")
    
    # Calculate variance across batch dimension
    # z_t: [B, N, D]
    var_across_batch = torch.var(z_t, dim=0) # [N, D]
    mean_var_across_batch = var_across_batch.mean().item()
    
    # Calculate variance across token dimension
    var_across_tokens = torch.var(z_t, dim=1) # [B, D]
    mean_var_across_tokens = var_across_tokens.mean().item()
    
    # Calculate variance across all combined
    mean_var_combined = torch.var(z_t, dim=(0, 1)).mean().item()
    
    print(f"\n============================================================")
    print(f"  REPRESENTATION SPECTRUM & COLLAPSE ANALYSIS")
    print(f"============================================================")
    print(f"Mean variance across BATCH dimension (content variance):  {mean_var_across_batch:.8f}")
    print(f"Mean variance across TOKENS dimension (spatial variance):  {mean_var_across_tokens:.8f}")
    print(f"Mean variance across COMBINED B+N (what SIGReg sees):     {mean_var_combined:.8f}")
    
    # Check similarity between different inputs in the batch
    z_t_mean = z_t.mean(dim=1)
    norm_z = torch.nn.functional.normalize(z_t_mean, p=2, dim=-1)
    cos_sim_matrix = torch.matmul(norm_z, norm_z.t())
    
    print("\nCosine similarity matrix between first 4 batch elements (different random inputs):")
    print(cos_sim_matrix[:4, :4].cpu().numpy())
    
    # Compute rank/sparsity of the latent spectrum
    # SVD on centered representations
    z_flat = z_t.view(-1, d_model)
    z_flat_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
    U, S, V = torch.linalg.svd(z_flat_centered, full_matrices=False)
    singular_values = S.tolist()
    
    print("\nLatent Spectrum (Top 5 Singular Values):")
    print([f"{val:.4f}" for val in singular_values[:5]])
    
    # Check if collapsed
    if mean_var_combined < 1e-4:
        print("\n[ALERT] Model suffers from representation collapse! Variance is close to 0.")
    elif cos_sim_matrix.std().item() < 1e-5 and cos_sim_matrix[0, 1].item() > 0.999:
        print("\n[ALERT] Model suffers from complete dimensional collapse! All elements map to identical vectors.")
    else:
        print("\n[SUCCESS] Model has rich, highly active representations! No collapse detected.")
    print("============================================================\n")

if __name__ == "__main__":
    check_model()
