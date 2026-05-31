import torch
import os

f1 = "results/two_rooms/two_rooms_model_epoch010_384.pt"
f2 = "results/two_rooms/two_rooms_model_v1.pt"

try:
    sd1 = torch.load(f1, map_location="cpu")["model_state_dict"]
    sd2 = torch.load(f2, map_location="cpu")["model_state_dict"]
    
    keys1 = set(sd1.keys())
    keys2 = set(sd2.keys())
    
    print(f"Keys only in v1.pt: {keys2 - keys1}")
    print(f"Keys only in epoch010_384.pt: {keys1 - keys2}")
            
except Exception as e:
    print(f"Error: {e}")
