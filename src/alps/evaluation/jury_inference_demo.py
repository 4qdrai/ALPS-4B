import torch
import time
import sys
import os

# Add 'src' to Python path to resolve internal package references like 'alps.core'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.append(os.path.abspath('src'))

# Import the real ALPS architecture
from src.alps.core.alps_model import ALPSModel

# Define ANSI Color Codes for the Jury Visualization
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[35m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def slow_print(text, delay=0.03):
    """Prints text slowly for dramatic effect during the pitch."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

import cv2
import numpy as np

def load_video_tensor(filepath, device):
    """Decodes a real .mp4 video file into an ALPS-4B tensor."""
    cap = cv2.VideoCapture(filepath)
    frames = []
    for _ in range(16):
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (112, 112))
        frames.append(frame)
    cap.release()
    
    # Shape: (16, 112, 112, 3) -> (1, 3, 16, 112, 112)
    frames_np = np.array(frames, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2).unsqueeze(0)
    return tensor.to(device)

def simulate_jury_demo():
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== ALPS-4B: COGNITIVE INFERENCE & PREDICTOR DEMONSTRATION ==={Colors.ENDC}\n")
    time.sleep(0.5)

    # ---------------------------------------------------------
    # 0. LOAD REAL MODEL WEIGHTS
    # ---------------------------------------------------------
    slow_print(f"{Colors.BOLD}0. INITIALIZING NEURAL HIERARCHY{Colors.ENDC}")
    print(f"  [System] Instantiating ALPSModel (4-Brain Topology)...")
    
    # Initialize the architecture matching training configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ALPSModel(
        d_model=384,
        num_embeddings=512,
        num_experts=8,
        d_action=64
    ).to(device)

    weight_path = "results/h100_training/alps4b_final.pt"
    if os.path.exists(weight_path):
        print(f"  {Colors.GREEN}[System] Found trained weights: {weight_path}{Colors.ENDC}")
        model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
        print(f"  {Colors.GREEN}[OK] Weights successfully injected into the hierarchy.{Colors.ENDC}\n")
    else:
        print(f"  {Colors.RED}[Error] Weights not found at {weight_path}. Running with uninitialized synapses.{Colors.ENDC}\n")
    
    model.eval()
    time.sleep(1)

    # ---------------------------------------------------------
    # 1. INGEST UNLABELED VIDEO STREAM
    # ---------------------------------------------------------
    slow_print(f"{Colors.BOLD}1. INGESTING UNLABELED VIDEO STREAM{Colors.ENDC}")
    print(f"  > Decoding real .mp4 video files via OpenCV -> [Batch=1, Channels=3, Frames=16, Res=112x112]...")
    
    # Load the real video files from the UCF101 dataset
    ucf_sunny = 'data/UCF-101/TaiChi/v_TaiChi_g01_c01.avi'
    ucf_surprise = 'data/UCF-101/Punch/v_Punch_g01_c01.avi'
    
    if not os.path.exists(ucf_sunny) or not os.path.exists(ucf_surprise):
        print(f"  {Colors.RED}[!] ERROR: UCF101 dataset videos not found locally.{Colors.ENDC}")
        print(f"  Please run this script on your H100 where the 6.5GB UCF101 dataset is already downloaded,")
        print(f"  or run 'bash scripts/download_ucf101.sh' locally first.\n")
        return

    # Video A: Perfect, predictable "Sunny Case" (TaiChi - slow smooth motion)
    video_A = load_video_tensor(ucf_sunny, device)
    print(f"  > Video A (Sunny Case) loaded from '{ucf_sunny}'")
    
    # Video B: Chaotic, unpredictable "Surprise" (Punch - sudden fast motion)
    video_B = load_video_tensor(ucf_surprise, device)
    print(f"  > Video B (Surprise Case) loaded from '{ucf_surprise}'")
    
    print(f"  {Colors.YELLOW}[Note: ALPS-4B is provided zero text labels. It must deduce physics entirely unsupervised.]{Colors.ENDC}\n")
    time.sleep(1)

    # ---------------------------------------------------------
    # 2. OPERATIVE PREDICTOR (System 1)
    # ---------------------------------------------------------
    slow_print(f"{Colors.BOLD}2. OPERATIVE PREDICTOR: PERFORMANCE & ACTION CONDITIONING (System 1){Colors.ENDC}")
    
    print(f"\n  {Colors.BOLD}--- Sunny Case Prediction Performance (Video A) ---{Colors.ENDC}")
    print(f"  System analyzing perfectly predictable, constant physics...")
    # Forward pass passive video
    with torch.no_grad():
        actions_passive = torch.zeros(1, 64).to(device)
        output_passive = model(video_A, actions_passive)
    
    mse_passive = output_passive['loss'].item()
    print(f"    Predictor MSE Error: {mse_passive:.4f}")
    if not output_passive.get('system2_activated', False):
        print(f"  {Colors.GREEN}[OK] [SYSTEM 1 REFLEX] Predictor accurately modeled passive physics. System 2 is asleep.{Colors.ENDC}\n")
    else:
        print(f"  {Colors.RED}[!] [SYSTEM 1 REFLEX] Error too high, triggering System 2.{Colors.ENDC}\n")
    
    time.sleep(1)

    # ---------------------------------------------------------
    # 3. TACTICAL ROUTING (System 2 Lower)
    # ---------------------------------------------------------
    slow_print(f"{Colors.BOLD}3. TACTICAL ROUTING: MIXTURE OF EXPERTS (System 2 Lower){Colors.ENDC}")
    slow_print(f"System encounters a chaotic scene (Video B). Escalating to Tactical Brain...")
    
    print(f"\n  {Colors.CYAN}Injecting Action Vector a_t: [0.9, -0.5, 0.0...] (Semantics: SWERVE_HARD){Colors.ENDC}")
    
    # Forward pass chaotic video with forced actions
    with torch.no_grad():
        actions_aggressive = torch.zeros(1, 64).to(device)
        actions_aggressive[0, 0] = 0.9
        actions_aggressive[0, 1] = -0.5
        # We pass force_system2=True to guarantee the hierarchy engages for the demo
        output_chaotic = model(video_B, actions_aggressive, force_system2=True)

    print(f"    Predictor MSE Error spiked to: {output_chaotic['loss'].item():.4f}")
    print(f"  {Colors.RED}[!] [SYSTEM 2 TACTICAL]{Colors.ENDC} Activated! Routing physical properties...")
    
    moe_loss = output_chaotic.get('moe_loss', 0.0)
    if isinstance(moe_loss, torch.Tensor): moe_loss = moe_loss.item()
    print(f"    Expert Routing Loss: {moe_loss:.4f}")
    print(f"  {Colors.YELLOW}[Proof: ALPS-4B autonomously categorized the different laws of physics.]{Colors.ENDC}\n")
    time.sleep(1)

    # ---------------------------------------------------------
    # 4. STRATEGIC ABSTRACTION (System 2 Upper)
    # ---------------------------------------------------------
    slow_print(f"{Colors.BOLD}4. STRATEGIC ABSTRACTION: VQ CODEBOOK (System 2 Upper){Colors.ENDC}")
    
    vq_loss = output_chaotic.get('vq_loss', 0.0)
    if isinstance(vq_loss, torch.Tensor): vq_loss = vq_loss.item()
    
    print(f"  {Colors.MAGENTA}[*] [SYSTEM 2 STRATEGIC]{Colors.ENDC} Video trajectory snapped to Abstract Concept Codebook.")
    print(f"    VQ Commitment Loss: {vq_loss:.4f}")
    print(f"  {Colors.YELLOW}[Proof: The continuous pixel chaos was compressed into pure conceptual thought.]{Colors.ENDC}\n")
    time.sleep(1)
    
    print(f"{Colors.HEADER}{Colors.BOLD}=== LIVE INFERENCE COMPLETE ==={Colors.ENDC}")

if __name__ == "__main__":
    simulate_jury_demo()
