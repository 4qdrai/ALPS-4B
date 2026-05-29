import os
import sys
import torch
import numpy as np
from PIL import Image

def main():
    # Make sure we can find the data file
    data_path = "data/two_rooms/trajectories.pt"
    if not os.path.exists(data_path):
        print(f"[Error] Dataset not found at: {data_path}")
        print("Please wait for the data generator to finish running!")
        return

    print(f"Loading dataset from {data_path} ...")
    # Load on CPU to avoid using GPU memory
    data = torch.load(data_path, map_location="cpu", weights_only=True)
    
    # Extract tensors
    # Check both 'frames' and 'observations' as key names
    frames = data.get("frames", data.get("observations"))
    positions = data.get("positions")
    episode_starts = data.get("episode_starts")
    
    if frames is None or episode_starts is None:
        print("[Error] Loaded dataset structure is invalid or corrupted.")
        return
        
    total_frames = frames.shape[0]
    num_episodes = len(episode_starts)
    
    print("-" * 50)
    print("   DATASET INTEGRITY & VALIDATION REPORT")
    print("-" * 50)
    print(f"Total episodes:       {num_episodes}")
    print(f"Total video frames:   {total_frames}")
    print(f"Frames tensor shape:  {list(frames.shape)}")
    print(f"Positions shape:      {list(positions.shape) if positions is not None else 'N/A'}")
    print("-" * 50)
    
    # Let's save 3 random episodes as animated GIFs to validate!
    output_dir = "results/dataset_validation"
    os.makedirs(output_dir, exist_ok=True)
    
    rng = np.random.RandomState(42)
    episodes_to_visualize = rng.choice(num_episodes - 1, size=3, replace=False)
    
    for ep_idx in episodes_to_visualize:
        start_idx = int(episode_starts[ep_idx])
        end_idx = int(episode_starts[ep_idx + 1])
        ep_frames = frames[start_idx:end_idx]
        
        print(f"Visualizing Episode {ep_idx:03d} (Frames {start_idx} to {end_idx - 1}, length={len(ep_frames)})...")
        
        pil_images = []
        for f in ep_frames:
            # f shape: [3, 128, 128] or [128, 128, 3]
            if f.shape[0] == 3:
                # permute to HWC and scale to 255
                img_np = (f.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            else:
                img_np = (f.numpy() * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))
            
        gif_path = os.path.join(output_dir, f"episode_{ep_idx:03d}_sample.gif")
        pil_images[0].save(
            gif_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=100, # 100ms per frame
            loop=0
        )
        print(f"  ✓ Saved animation to: {gif_path}")
        
    print("-" * 50)
    print(f"Validation complete! You can open the sample GIFs in: {output_dir}/")
    print("-" * 50)

if __name__ == "__main__":
    main()
