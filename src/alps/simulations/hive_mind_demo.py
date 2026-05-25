import torch
import json
import os

from alps.core.latent_rag import LatentRAG

def run_hive_mind_demo():
    print("=== ALPS-4B: Hive-Mind Fleet Synchronization Simulation ===")
    
    d_model = 64
    
    # Initialize Robot A and Robot B
    robot_A_rag = LatentRAG(d_model=d_model, sim_threshold=0.5)
    robot_B_rag = LatentRAG(d_model=d_model, sim_threshold=0.5)
    
    # 1. Robot A experiences a failure and learns the correction vector
    z_t = torch.randn(1, 1, d_model)
    target_z_next = torch.randn(1, 1, d_model)
    naive_pred = target_z_next + torch.randn_like(target_z_next) * 0.5
    
    delta_z = target_z_next - naive_pred
    
    print("\nRobot A experiences a new physical context and records a correction...")
    robot_A_rag.write_memory(z_t, delta_z)
    assert robot_A_rag.current_size.item() == 1
    
    # 2. Verify Robot B initially fails on this context
    print("\nRobot B is placed in the same context before synchronization...")
    retrieved_B_pre = robot_B_rag.retrieve_correction(z_t)
    error_B_pre = torch.mean((naive_pred + retrieved_B_pre - target_z_next) ** 2).item()
    print(f"  Robot B Pre-sync Prediction MSE: {error_B_pre:.6f}")
    assert (retrieved_B_pre == 0).all()
    
    # 3. Synchronize Fleet: copy Robot A's RAG databases to Robot B
    print("\nSynchronizing Robot B's database with Robot A's (Hive-Mind KV Transfer)...")
    robot_B_rag.keys.copy_(robot_A_rag.keys)
    robot_B_rag.values.copy_(robot_A_rag.values)
    robot_B_rag.current_size.copy_(robot_A_rag.current_size)
    
    # 4. Verify Robot B now succeeds instantly
    retrieved_B_post = robot_B_rag.retrieve_correction(z_t)
    error_B_post = torch.mean((naive_pred + retrieved_B_post - target_z_next) ** 2).item()
    print(f"  Robot B Post-sync Prediction MSE: {error_B_post:.6f}")
    
    results = {
        "robot_B_pre_sync_error": error_B_pre,
        "robot_B_post_sync_error": error_B_post,
        "sync_success": error_B_post < error_B_pre and error_B_post < 1e-4
    }
    
    os.makedirs("results/simulations", exist_ok=True)
    with open("results/simulations/hive_mind_demo.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nHive-Mind simulation complete! Saved results to results/simulations/hive_mind_demo.json.")

if __name__ == "__main__":
    run_hive_mind_demo()
