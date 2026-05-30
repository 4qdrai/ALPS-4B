import os
from huggingface_hub import HfApi

def upload_to_hf():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable is not set. Please set it before running the script.")
        print("Example: HF_TOKEN=\"your_huggingface_token\" python scratch/upload_hf.py")
        return
        
    repo_id = "Free2035/4QDR4BWorldModel"
    api = HfApi()
    
    # 1. Upload two_rooms_model.pt as two_rooms_model_v1.pt
    model_path = "results/two_rooms/two_rooms_model.pt"
    if os.path.exists(model_path):
        print(f"Uploading {model_path} to HF as two_rooms_model_v1.pt...")
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="two_rooms_model_v1.pt",
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        print("Upload of two_rooms_model_v1.pt successful!")
    else:
        print(f"Error: {model_path} not found.")

    # 2. Upload two_rooms_model_epoch010.pt as two_rooms_model_epoch010_v1.pt
    epoch_path = "results/two_rooms/two_rooms_model_epoch010.pt"
    if os.path.exists(epoch_path):
        print(f"Uploading {epoch_path} to HF as two_rooms_model_epoch010_v1.pt...")
        api.upload_file(
            path_or_fileobj=epoch_path,
            path_in_repo="two_rooms_model_epoch010_v1.pt",
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        print("Upload of two_rooms_model_epoch010_v1.pt successful!")
    else:
        print(f"Error: {epoch_path} not found.")

if __name__ == "__main__":
    upload_to_hf()
