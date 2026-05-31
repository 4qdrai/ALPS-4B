import os
import sys

# Configure standard output to handle UTF-8 characters on Windows console
os.environ['PYTHONIOENCODING'] = 'utf-8'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Please enter your Hugging Face User Access Token (write permission required):")
        token = input().strip()
        
    repo_id = "Free2035/4QDR4BWorldModel"
    local_file = "results/two_rooms/two_rooms_model_epoch010_384.pt"
    
    if not os.path.exists(local_file):
        print(f"Error: {local_file} not found!")
        return

    print("Uploading model to Hugging Face...")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Upload file
        response = api.upload_file(
            path_or_fileobj=local_file,
            path_in_repo="two_rooms_model_v2_self_supervised.pt",
            repo_id=repo_id,
            token=token
        )
        print(f"\n✨ Upload successful!")
        print(f"Model is live at: https://huggingface.co/{repo_id}/resolve/main/two_rooms_model_v2_self_supervised.pt")
    except ImportError:
        print("huggingface_hub is not installed. Installing it...")
        os.system("pip install huggingface_hub")
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            response = api.upload_file(
                path_or_fileobj=local_file,
                path_in_repo="two_rooms_model_v2_self_supervised.pt",
                repo_id=repo_id,
                token=token
            )
            print(f"\n✨ Upload successful!")
            print(f"Model is live at: https://huggingface.co/{repo_id}/resolve/main/two_rooms_model_v2_self_supervised.pt")
        except Exception as e:
            print(f"Failed to upload model: {e}")
    except Exception as e:
        print(f"Failed to upload model: {e}")

if __name__ == "__main__":
    main()
