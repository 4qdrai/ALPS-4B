import torch
import json
import os

from alps.core.latent_rag import LatentRAG

def run_self_learning_demo():
    print("=== ALPS-4B: Zero-Retraining Self-Learning Simulation ===")
    
    d_model = 64
    rag = LatentRAG(d_model=d_model, sim_threshold=0.5)
    
    # 1. Step 1: Physical encounter with failure state
    print("\nStep 1: System encounters state z_t and predicts next state...")
    z_t = torch.randn(1, 1, d_model)
    target_z_next = torch.randn(1, 1, d_model)
    
    # Predictor makes a naive prediction (initially inaccurate)
    naive_pred = target_z_next + torch.randn_like(target_z_next) * 0.5
    
    initial_error = torch.mean((naive_pred - target_z_next) ** 2).item()
    print(f"  Naive Prediction MSE: {initial_error:.6f}")
    
    # 2. System detects surprise and writes correction (delta-z) to RAG
    delta_z = target_z_next - naive_pred
    # The key signature is the current state z_t
    print("  Surprise triggered! Writing corrective vector to Latent-RAG...")
    rag.write_memory(z_t, delta_z)
    
    # 3. Step 2: System re-encounters identical/similar context
    print("\nStep 2: System re-encounters similar context and queries RAG cache...")
    # Add a little bit of noise to test generalization of retrieval
    z_t_similar = z_t + torch.randn_like(z_t) * 0.02
    
    # naive prediction under same context
    naive_pred2 = naive_pred
    
    # RAG retrieval
    correction = rag.retrieve_correction(z_t_similar)
    corrected_pred = naive_pred2 + correction
    
    corrected_error = torch.mean((corrected_pred - target_z_next) ** 2).item()
    print(f"  Corrected Prediction MSE: {corrected_error:.6f}")
    print(f"  Error reduction: {((initial_error - corrected_error)/initial_error)*100:.2f}%")
    
    results = {
        "initial_error": initial_error,
        "corrected_error": corrected_error,
        "error_reduction_pct": ((initial_error - corrected_error)/initial_error)*100
    }
    
    os.makedirs("results/simulations", exist_ok=True)
    with open("results/simulations/self_learning_demo.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nSelf-Learning simulation complete! Saved details to results/simulations/self_learning_demo.json.")

if __name__ == "__main__":
    run_self_learning_demo()
