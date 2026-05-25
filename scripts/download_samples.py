import urllib.request
import os

os.makedirs('data', exist_ok=True)

import urllib.request
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# UCF101 sample videos hosted on HuggingFace for testing
url_sunny = "https://huggingface.co/datasets/sayakpaul/ucf101-video-dataset/resolve/main/v_TaiChi_g01_c01.avi"
url_surprise = "https://huggingface.co/datasets/sayakpaul/ucf101-video-dataset/resolve/main/v_Punch_g01_c01.avi"

# Save them to exactly where the inference script expects them
os.makedirs('data/UCF-101/TaiChi', exist_ok=True)
os.makedirs('data/UCF-101/Punch', exist_ok=True)

print("Fetching UCF101 TaiChi (Sunny Case) from HuggingFace...")
urllib.request.urlretrieve(url_sunny, 'data/UCF-101/TaiChi/v_TaiChi_g01_c01.avi')

print("Fetching UCF101 Punch (Surprise Case) from HuggingFace...")
urllib.request.urlretrieve(url_surprise, 'data/UCF-101/Punch/v_Punch_g01_c01.avi')

print("Success! The exact UCF101 semantic videos are now downloaded locally.")
