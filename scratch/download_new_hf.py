import os
import urllib.request

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
    repo_id = "Free2035/4QDR4BWorldModel"
    
    # Let's download two_rooms_model_v1.pt
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/two_rooms_model_v1.pt"
    # Wait, is the repository a model or dataset? 
    # Let's check model resolve url
    model_url = f"https://huggingface.co/{repo_id}/resolve/main/two_rooms_model_v1.pt"
    
    save_path = "results/two_rooms/two_rooms_model_v1.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    download_file(model_url, save_path, token)

if __name__ == "__main__":
    main()
