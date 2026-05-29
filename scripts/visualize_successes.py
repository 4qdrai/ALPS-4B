import os
import sys
import torch
import numpy as np
from PIL import Image

def main():
    data_path = "data/two_rooms/trajectories.pt"
    if not os.path.exists(data_path):
        print(f"[Error] Dataset not found at: {data_path}")
        return

    print(f"Loading dataset from {data_path} ...")
    data = torch.load(data_path, map_location="cpu", weights_only=True)
    
    frames = data.get("frames", data.get("observations"))
    positions = data.get("positions")
    room_ids = data.get("room_ids")
    episode_starts = data.get("episode_starts")
    
    if frames is None or episode_starts is None:
        print("[Error] Loaded dataset structure is invalid.")
        return
        
    total_frames = frames.shape[0]
    num_episodes = len(episode_starts)
    
    output_dir = "results/dataset_validation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate lengths of each episode
    episode_lengths = []
    for i in range(num_episodes):
        start_idx = int(episode_starts[i])
        if i + 1 < num_episodes:
            end_idx = int(episode_starts[i + 1])
        else:
            end_idx = total_frames
        episode_lengths.append(end_idx - start_idx)
        
    # Successful episodes are those that ended early (length < 100)
    successful_episodes = [ep for ep, length in enumerate(episode_lengths) if length < 100]
    
    print("-" * 60)
    print(f"Total episodes:           {num_episodes}")
    print(f"Successful episodes:      {len(successful_episodes)} ({len(successful_episodes)/num_episodes*100:.1f}%)")
    print("-" * 60)
    
    if len(successful_episodes) == 0:
        print("No successful episodes found in the dataset! (All ran for the full 100 steps)")
        return
        
    # Categorize successful episodes: same-room vs cross-room
    same_room_successes = []
    cross_room_successes = []
    
    for ep_idx in successful_episodes:
        start_idx = int(episode_starts[ep_idx])
        length = episode_lengths[ep_idx]
        end_idx = start_idx + length
        
        ep_rooms = room_ids[start_idx:end_idx]
        start_room = int(ep_rooms[0])
        end_room = int(ep_rooms[-1])
        
        # Check if the agent crossed rooms during the episode
        has_crossed = (ep_rooms.max() != ep_rooms.min())
        
        if has_crossed:
            cross_room_successes.append((ep_idx, length))
        else:
            same_room_successes.append((ep_idx, length))
            
    print(f"Same-room successes:      {len(same_room_successes)}")
    print(f"Cross-room successes:     {len(cross_room_successes)}")
    print("-" * 60)
    
    # Export up to 2 same-room and 2 cross-room successful episodes
    rng = np.random.RandomState(42)
    
    to_export = []
    if same_room_successes:
        selected_same = rng.choice(len(same_room_successes), size=min(2, len(same_room_successes)), replace=False)
        for idx in selected_same:
            to_export.append((same_room_successes[idx][0], "success_same_room"))
            
    if cross_room_successes:
        selected_cross = rng.choice(len(cross_room_successes), size=min(2, len(cross_room_successes)), replace=False)
        for idx in selected_cross:
            to_export.append((cross_room_successes[idx][0], "success_cross_room"))
            
    for ep_idx, prefix in to_export:
        start_idx = int(episode_starts[ep_idx])
        length = episode_lengths[ep_idx]
        end_idx = start_idx + length
        ep_frames = frames[start_idx:end_idx]
        
        print(f"Exporting {prefix} | Episode {ep_idx:03d} (length={length} frames)...")
        
        pil_images = []
        for f in ep_frames:
            if f.dtype == torch.uint8 or f.max() > 1.0:
                img_np = f.permute(1, 2, 0).numpy().astype(np.uint8) if f.shape[0] == 3 else f.numpy().astype(np.uint8)
            else:
                img_np = (f.permute(1, 2, 0).numpy() * 255).astype(np.uint8) if f.shape[0] == 3 else (f.numpy() * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))
            
        gif_path = os.path.join(output_dir, f"{prefix}_episode_{ep_idx:03d}.gif")
        pil_images[0].save(
            gif_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=120,
            loop=0
        )
        print(f"  ✓ Saved to: {gif_path}")
        
    print("-" * 60)
    print("Export complete! Successful episodes exported successfully.")
    print("-" * 60)

if __name__ == "__main__":
    main()
