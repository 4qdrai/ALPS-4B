import cv2
import numpy as np
import os

os.makedirs('data', exist_ok=True)
width, height = 112, 112
fps = 8

# Generate Video A: Perfectly Static "Sunny Case" (e.g., a blank blue sky)
out_A = cv2.VideoWriter('data/sample_A.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
for i in range(16):
    # Solid blue frame that never changes
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (255, 100, 50) 
    out_A.write(frame)
out_A.release()

# Generate Video B: Chaotic geometric shapes (Surprise Case)
out_B = cv2.VideoWriter('data/sample_B.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
for i in range(16):
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    out_B.write(frame)
out_B.release()

print("Generated actual .mp4 files in 'data/' folder.")
