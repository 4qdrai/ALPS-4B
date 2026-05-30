import os
from PIL import Image

def extract_gif_frames(gif_name, prefix):
    gif_path = f"results/dataset_validation/{gif_name}"
    if not os.path.exists(gif_path):
        print(f"Error: {gif_path} not found")
        return
        
    img = Image.open(gif_path)
    frames = []
    try:
        while True:
            frames.append(img.copy().convert("RGB"))
            img.seek(img.tell() + 1)
    except EOFError:
        pass
        
    out_dir = "C:/Users/User/.gemini/antigravity/brain/6dbc421e-4c7c-4b73-8b06-97eaaf15d2ba"
    os.makedirs(out_dir, exist_ok=True)
    
    frames[0].save(os.path.join(out_dir, f"{prefix}_start.png"))
    frames[len(frames)//2].save(os.path.join(out_dir, f"{prefix}_mid.png"))
    frames[-1].save(os.path.join(out_dir, f"{prefix}_end.png"))
    print(f"Extracted {gif_name} successfully!")

extract_gif_frames("episode_084_sample.gif", "true_frame_084")
extract_gif_frames("episode_2470_sample.gif", "true_frame_2470")
