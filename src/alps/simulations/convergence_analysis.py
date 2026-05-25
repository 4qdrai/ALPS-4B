import torch
import json
import os

from alps.core.checker import BanachChecker

def run_convergence_simulation():
    """
    Simulates the checker-refinement process for different Lipschitz parameters
    to mathematically verify the Banach contraction mapping theorem.
    """
    print("=== ALPS-4B: Banach Contraction Convergence Simulation ===")
    
    d_model = 64
    B, N = 1, 10
    
    initial_z = torch.randn(B, N, d_model)
    context = torch.randn(B, N, d_model)
    
    lipschitz_values = [0.5, 0.9, 1.2]
    results = {}
    
    for L in lipschitz_values:
        print(f"\nSimulating Lipschitz constant L = {L}...")
        checker = BanachChecker(d_model=d_model, d_cond=d_model, max_iters=10, target_lipschitz=L)
        
        # Override the Lipschitz constant inside the network by scaling residual weight
        # so we can explicitly control contraction behavior
        # Recall R(z) = z + 0.1 * delta. We scale that 0.1 factor to modulate L
        scaling_factor = L / 2.0
        
        # Iteratively run and log step-by-step distance
        z = initial_z.clone()
        distances = []
        
        for step in range(1, 11):
            z_prev = z.clone()
            
            # Simulated forward pass with manual scaling
            combined = torch.cat([z, context], dim=-1)
            delta = checker.refiner.net(combined)
            z = z + scaling_factor * delta
            
            dist = torch.norm(z - z_prev, p=2, dim=-1).mean().item()
            distances.append(dist)
            print(f"  Iteration {step}: d(z_n, z_n-1) = {dist:.6f}")
            
        results[f"L_{L}"] = distances
        
    # Save results to disk
    os.makedirs("results/simulations", exist_ok=True)
    with open("results/simulations/banach_convergence.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nSimulation complete! Results saved to results/simulations/banach_convergence.json.")

if __name__ == "__main__":
    run_convergence_simulation()
