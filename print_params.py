import torch
import sys
import os

# Ensure the src directory is in the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from alps.core.alps_model import ALPSModel

def print_parameter_breakdown():
    print("\n" + "="*50)
    print("ALPS-4B (LOCAL PROOF-OF-CONCEPT) PARAMETER BREAKDOWN")
    print("="*50)
    
    # Instantiate the local model
    model = ALPSModel(
        d_model=384,
        d_action=64,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_patch_size=(2, 16, 16),
        encoder_max_patches=256,
        use_langevin=True,
        langevin_steps=2
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal Parameters:      {total_params:,}")
    print(f"Trainable Parameters:  {trainable_params:,}\n")
    
    print("-" * 50)
    print(f"{'Component':<30} | {'Parameters':<15}")
    print("-" * 50)
    
    # Group by major submodules
    breakdown = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        if params > 0:
            breakdown[name] = params
            print(f"{name:<30} | {params:,}")
            
    print("-" * 50)

if __name__ == "__main__":
    print_parameter_breakdown()
