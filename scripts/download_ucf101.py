import urllib.request
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

print("=== UCF101 Dataset Downloader ===")
os.makedirs("data", exist_ok=True)

url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
out_path = "data/UCF101.rar"

if not os.path.exists(out_path):
    print(f"Downloading {url} to {out_path} (~6.5 GB). This will take a while...")
    urllib.request.urlretrieve(url, out_path)
    print("Download complete!")
else:
    print(f"{out_path} already exists.")

print("\nPlease extract the UCF101.rar file using WinRAR or 7-Zip into data/UCF-101/ before running the inference script.")
