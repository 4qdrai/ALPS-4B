import torch
import json
import time
import os

from alps.core.moe_router import SparseMoERouter

def run_moe_scaling_analysis():
    print("=== ALPS-4B: Sparse MoE O(1) FLOPs Scaling Analysis ===")
    
    d_model = 256
    seq_len = 100
    batch_size = 4
    active_experts = 2
    
    expert_counts = [4, 8, 16, 32, 64]
    moe_times = []
    dense_times = [] # simulated dense layer equivalent scaling linearly
    
    inputs = torch.randn(batch_size, seq_len, d_model)
    
    for num_exp in expert_counts:
        print(f"Profiling MoE with {num_exp} total experts...")
        router = SparseMoERouter(d_model=d_model, num_experts=num_exp, active_experts=active_experts, d_ff=512)
        
        # Warmup
        for _ in range(3):
            _ = router(inputs)
            
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        
        for _ in range(10):
            _ = router(inputs)
            
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()
        
        avg_time = (end - start) / 10.0
        moe_times.append(avg_time)
        print(f"  Avg execution time: {avg_time:.6f} seconds")
        
        # Dense baseline: scales linearly because it activates all experts
        # We mock this execution time by scaling the time of a single expert
        dense_times.append(avg_time * (num_exp / active_experts))
        
    results = {
        "expert_counts": expert_counts,
        "moe_times": moe_times,
        "dense_times": dense_times
    }
    
    os.makedirs("results/simulations", exist_ok=True)
    with open("results/simulations/moe_scaling.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nMoE Scaling benchmark complete! Results saved to results/simulations/moe_scaling.json.")

if __name__ == "__main__":
    run_moe_scaling_analysis()
