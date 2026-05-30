import os
import urllib.request
import torch

def download_file(url, filename, token):
    print(f"Downloading {url} to {filename}...")
    headers = {"Authorization": f"Bearer {token}"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print("Download successful!")
    except Exception as e:
        print(f"Error downloading file: {e}")

def main():
    token = os.environ.get("HF_TOKEN", "")
    url = "https://huggingface.co/Free2035/4QDR4BWorldModel/resolve/main/two_rooms_model_epoch010.pt"
    save_path = "results/two_rooms/two_rooms_model_epoch010_384.pt"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path):
        download_file(url, save_path, token)
    else:
        print(f"File already exists at {save_path}")

    # Verify model shape
    print("\nVerifying loaded model...")
    ckpt = torch.load(save_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    print(f"Keys in state_dict: {len(sd.keys())}")
    pos_embed_shape = sd["encoder.pos_embed"].shape
    print(f"encoder.pos_embed shape: {pos_embed_shape}")

if __name__ == "__main__":
    main()
