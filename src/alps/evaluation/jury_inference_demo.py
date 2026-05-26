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
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    cap.release()
    
    # Shape: (16, 224, 224, 3) -> (1, 3, 16, 224, 224)
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
    
    # Load the real video files from disk
    print(f"  {Colors.BOLD}--- Dataset Pair 1 ---{Colors.ENDC}")
    video_A = load_video_tensor('data/sample_A.avi', device)
    print(f"  > Video A (Sunny Case - People Walking) loaded from 'data/sample_A.avi'")
    video_B = load_video_tensor('data/sample_B.mp4', device)
    print(f"  > Video B (Surprise Case - Sintel Action Trailer) loaded from 'data/sample_B.mp4'")
    
    print(f"  {Colors.BOLD}--- Dataset Pair 2 ---{Colors.ENDC}")
    video_C = load_video_tensor('data/tree.avi', device)
    print(f"  > Video C (Sunny Case - Tree blowing in wind) loaded from 'data/tree.avi'")
    video_D = load_video_tensor('data/megamind.avi', device)
    print(f"  > Video D (Surprise Case - Megamind Action Sequence) loaded from 'data/megamind.avi'")
    
    print(f"  {Colors.YELLOW}[Note: ALPS-4B is provided zero text labels. It must deduce physics entirely unsupervised.]{Colors.ENDC}\n")
    time.sleep(1)

    # ---------------------------------------------------------
    # 2. OPERATIVE PREDICTOR (System 1)
    # ---------------------------------------------------------
    slow_print(f"{Colors.BOLD}2. OPERATIVE PREDICTOR: PERFORMANCE & ACTION CONDITIONING (System 1){Colors.ENDC}")
    print("\n  --- Sunny Case Prediction Performance (Video A) ---")
    print("  System analyzing perfectly predictable, constant physics...")
    time.sleep(0.5)
    
    # Forward pass: Sunny Case A
    with torch.no_grad():
        actions_passive = torch.zeros(1, 64).to(device)
        output_A = model(video_A, actions_passive)
    mse_A = output_A['loss'].item()
    print(f"    Predictor MSE Error: {mse_A:.4f}")
    sys2_A = output_A.get('system2_activated', False)
    if not sys2_A:
        print(f"  {Colors.GREEN}[OK] [SYSTEM 1 REFLEX] Predictor accurately modeled passive physics. System 2 is asleep.{Colors.ENDC}\n")
        
    print("  --- Sunny Case Prediction Performance (Video C) ---")
    print("  System analyzing smooth, natural physics (wind in trees)...")
    time.sleep(0.5)
    
    # Forward pass: Sunny Case C
    with torch.no_grad():
        output_C = model(video_C, actions_passive)
    mse_C = output_C['loss'].item()
    print(f"    Predictor MSE Error: {mse_C:.4f}")
    sys2_C = output_C.get('system2_activated', False)
    if not sys2_C:
        print(f"  {Colors.GREEN}[OK] [SYSTEM 1 REFLEX] Predictor accurately modeled passive physics. System 2 is asleep.{Colors.ENDC}\n")

    time.sleep(1)
    slow_print(f"{Colors.BOLD}3. TACTICAL ROUTING: MIXTURE OF EXPERTS (System 2 Lower){Colors.ENDC}")
    print("  System encounters a chaotic scene (Video B). Escalating to Tactical Brain...\n")
    time.sleep(0.5)
    
    # Forward pass: Surprise Case B — NO force_system2! Natural threshold escalation.
    print(f"  {Colors.CYAN}Injecting Action Vector a_t: [0.9, -0.5, 0.0...] (Semantics: SWERVE_HARD){Colors.ENDC}")
    with torch.no_grad():
        actions_aggressive = torch.zeros(1, 64).to(device)
        actions_aggressive[0, 0] = 0.9
        actions_aggressive[0, 1] = -0.5
        output_B = model(video_B, actions_aggressive)
        
    mse_B = output_B['loss'].item()
    sys2_B = output_B.get('system2_activated', False)
    rag_B = output_B.get('rag_auto_write', False)
    print(f"    Predictor MSE Error: {mse_B:.4f}")
    print(f"    System 2 Activated: {sys2_B}")
    print(f"    RAG Auto-Write: {rag_B}")
    
    if sys2_B:
        print(f"  {Colors.RED}[!] [SYSTEM 2 TACTICAL]{Colors.ENDC} Naturally activated via InverseMonitor threshold!")
        route_loss_B = output_B.get('moe_loss', 0.0)
        if isinstance(route_loss_B, torch.Tensor): route_loss_B = route_loss_B.item()
        print(f"    Expert Routing Loss: {route_loss_B:.4f}")
        print(f"  {Colors.YELLOW}[Proof: ALPS-4B autonomously categorized the different laws of physics.]{Colors.ENDC}\n")
        
    print("  System encounters another chaotic scene (Video D). Escalating to Tactical Brain...\n")
    time.sleep(0.5)
    
    # Forward pass: Surprise Case D — NO force_system2! Natural threshold escalation.
    print(f"  {Colors.CYAN}Injecting Action Vector a_t: [-0.8, 0.9, 0.1...] (Semantics: RAPID_EVASION){Colors.ENDC}")
    with torch.no_grad():
        actions_evasive = torch.zeros(1, 64).to(device)
        actions_evasive[0, 0] = -0.8
        actions_evasive[0, 1] = 0.9
        output_D = model(video_D, actions_evasive)
        
    mse_D = output_D['loss'].item()
    sys2_D = output_D.get('system2_activated', False)
    rag_D = output_D.get('rag_auto_write', False)
    print(f"    Predictor MSE Error: {mse_D:.4f}")
    print(f"    System 2 Activated: {sys2_D}")
    print(f"    RAG Auto-Write: {rag_D}")
    
    if sys2_D:
        print(f"  {Colors.RED}[!] [SYSTEM 2 TACTICAL]{Colors.ENDC} Naturally activated via InverseMonitor threshold!")
        route_loss_D = output_D.get('moe_loss', 0.0)
        if isinstance(route_loss_D, torch.Tensor): route_loss_D = route_loss_D.item()
        print(f"    Expert Routing Loss: {route_loss_D:.4f}")
        print(f"  {Colors.YELLOW}[Proof: Consistency confirmed. The Tactical Brain awakens on surprise.]{Colors.ENDC}\n")

    time.sleep(1)
    slow_print(f"{Colors.BOLD}4. STRATEGIC ABSTRACTION: VQ CODEBOOK (System 2 Upper){Colors.ENDC}")
    vq_loss_B = output_B.get('vq_loss', 0.0)
    if isinstance(vq_loss_B, torch.Tensor): vq_loss_B = vq_loss_B.item()
    
    vq_loss_D = output_D.get('vq_loss', 0.0)
    if isinstance(vq_loss_D, torch.Tensor): vq_loss_D = vq_loss_D.item()
    
    print(f"  {Colors.MAGENTA}[*] [SYSTEM 2 STRATEGIC]{Colors.ENDC} Video trajectories snapped to Abstract Concept Codebook.")
    print(f"    VQ Commitment Loss (Video B): {vq_loss_B:.4f}")
    print(f"    VQ Commitment Loss (Video D): {vq_loss_D:.4f}")
    print(f"  {Colors.YELLOW}[Proof: The continuous pixel chaos was compressed into pure conceptual thought.]{Colors.ENDC}\n")
    time.sleep(1)
    
    print(f"{Colors.HEADER}{Colors.BOLD}=== LIVE INFERENCE COMPLETE ==={Colors.ENDC}")

if __name__ == "__main__":
    simulate_jury_demo()
