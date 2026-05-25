import urllib.request
import os

os.makedirs('data', exist_ok=True)

import urllib.request
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Real Semantic Video A (People walking smoothly)
url_A = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi"
path_A = "data/sample_A.avi"

# Real Semantic Video B (High action sequence)
url_B = "https://media.w3.org/2010/05/sintel/trailer.mp4"
path_B = "data/sample_B.mp4"

print("Fetching Real Semantic Video A (People Walking)...")
urllib.request.urlretrieve(url_A, path_A)

print("Fetching Real Semantic Video B (High Action)...")
urllib.request.urlretrieve(url_B, path_B)

print("Success! Real semantic videos are now saved in the 'data' folder.")
