# ALPS-4B: Training Methodology and H100 Recipe

This document provides a comprehensive deep-dive into the unsupervised self-supervised learning methodology, spatiotemporal tube masking, and single-GPU H100 training configurations for **ALPS-4B**.

---

## 1. The Single H100 Training Recipe

To enable rapid, low-cost training that outperforms expensive dense baseline methods, ALPS-4B is optimized for single-GPU execution. The training hyperparameters are as follows:

| Parameter | Configuration | Engineering Rationale |
| :--- | :--- | :--- |
| **Model Size** | ViT-Small (~22M parameters) | Balances representation capacity with rapid execution. |
| **Masking Policy** | Spatiotemporal Tube Masking | 90% masking ratio reduces active token sequence lengths by 10×, saving massive compute. |
| **Optimizer** | AdamW | $\beta_1=0.9, \beta_2=0.95$, weight decay of $0.05$. |
| **Learning Rate Schedule** | Cosine Annealing with Warmup | Linear warmup from $1e-5$ to $1e-3$ over 15 epochs, followed by cosine decay to $1e-6$. |
| **Precision** | BF16 Mixed Precision | Native H100 hardware acceleration, 2× memory reduction. |
| **Attention Kernel** | PyTorch SDPA (Flash Attention v2) | Eliminates $\mathcal{O}(N^2)$ memory consumption, speeding up sequence processing. |
| **Dataset** | UCF101 (Action Recognition) | ~9.5K train videos, 6.5 GB. Standard spatiotemporal benchmark. |
| **Batch Size** | 256 Effective Batch Size | Accumulates gradients from micro-batches of size 32 to ensure stable SIGReg convergence. |
| **Target Encoder** | **None** | Same encoder computes target and source latents; collapse is prevented solely by SIGReg. |

---

## 2. Spatiotemporal Tube Masking Strategy

Standard random patch masking (such as Masked Autoencoders for images) is highly inefficient for video data. Because adjacent video frames are highly correlated, a model can solve random patch masks by simply copying pixels from the preceding or following frame (temporal interpolation). This prevents the network from learning genuine causal dynamics.

ALPS-4B resolves this by employing **Tube Masking**:
1. Spatial coordinate patches $(x, y)$ are randomly selected for masking at a **90% masking ratio**.
2. Once a spatial coordinate is masked, **that identical coordinate is masked across the entire temporal axis** (all 16 frames).
3. This creates a spatiotemporal "tube" of missing information.
4. To predict the masked latents, the model cannot interpolate from adjacent frames; it must understand the physical motion and causal trajectories of the scene.

---

## 3. Phase-Shifted Optimizer Configuration

To prevent high-frequency mechanical gradients from corrupting high-level slow conceptual reasoning, ALPS-4B uses a **Phase-Shifted Update Schedule**:
* **Operative Optimizer**: Active on **every step** ($k=1$), learning millisecond sensorimotor control.
* **Tactical Optimizer**: Active **every 4 steps** ($k=4$), learning episodic sub-goals and expert pathways.
* **Strategic Optimizer**: Active **every 16 steps** ($k=16$), learning discrete, slow-varying conceptual schemas.

During backward passes, stop-gradients prevent gradients from flowing upwards between layers. The optimizers step conditionally based on the phase scheduler, allowing stable convergence at each level of temporal abstraction.
