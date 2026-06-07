"""
HierWorldModel — the FULL ALPS-4B hierarchy, trained as intended.

Unlike `ReprWorldModel` (encoder + operative only), this model instantiates the
learned Strategic and Tactical layers and a Latent-RAG that is queried *in the
inference loop*. It is designed so each layer's claim is independently testable:

  Operative (System 1) : token-level, action-conditioned 1-step dynamics
        z_t, a_t  ->  z_{t+1}
  Tactical            : pooled, predicts a MEDIUM-horizon abstraction and emits
        sub-goals; conditioned (stop-grad) on the strategic concept
        h_t, c_t  ->  h_{t+K_tac}
  Strategic (System 2): pooled, DISCRETE (VQ) concept, predicts a LONG-horizon
        abstraction (slow-varying)
        c_t       ->  c_{t+K_str}
  Goal emission       : goal-conditioned sub-goal head (the learned analog of the
        latent-graph waypoint) trained on goal-directed (progress) transitions
        (h_t, h_goal) -> next sub-goal h*  (decoded to a waypoint)
  Latent-RAG          : surprise-gated correction retrieval during inference

All abstraction is created by TEMPORAL STRIDING (predict K steps ahead), not by
label supervision; whether the strategic concept ends up encoding room identity
is then *tested* by an external probe (see validate_hierarchy.py), not assumed.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from alps.core.encoders import VisionEncoder
from alps.core.predictor import MultiScalePredictor
from alps.core.vq_bottleneck import VectorQuantizer
from alps.core.moe_router import SparseMoERouter
from alps.core.latent_rag import LatentRAG
from alps.core.sigreg import SIGReg


def mlp(d_in, d_hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))


class HierWorldModel(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        d_action: int = 4,
        num_codes: int = 64,
        num_experts: int = 4,
        active_experts: int = 2,
        enc_depth: int = 4,
        enc_heads: int = 4,
        patch_size: tuple = (2, 16, 16),
        max_patches: int = 512,
        op_depth: int = 6,
        lambda_sigreg: float = 0.1,
        sigreg_slices: int = 256,
        rag_sim_threshold: float = 0.75,
    ):
        super().__init__()
        self.d_model = d_model
        # min pseudo-clip length (patch temporal stride 2): T=2 → 1 temporal patch
        # (64 tokens) carries all info for a static frame; old T=8 was 4× redundant.
        self.SINGLE_FRAME_T = 2
        self.lambda_sigreg = lambda_sigreg

        # ── Encoder (shared) ──
        self.encoder = VisionEncoder(d_model=d_model, depth=enc_depth, num_heads=enc_heads,
                                     patch_size=patch_size, max_patches=max_patches)
        # ── Operative (System 1): token-level action-conditioned dynamics ──
        self.op_predictor = MultiScalePredictor(d_model=d_model, d_cond=d_action,
                                                depth=op_depth, num_heads=enc_heads)
        self.pos_head = mlp(d_model, d_model, 2)          # operative position decoder
        # ── Tactical: pooled MoE abstraction + medium-horizon predictor ──
        self.tac_proj = nn.Linear(d_model, d_model)
        self.moe = SparseMoERouter(d_model=d_model, num_experts=num_experts, active_experts=active_experts)
        self.tac_predictor = mlp(d_model + d_model, 2 * d_model, d_model)   # (h, c) -> h'
        self.tac_pos_head = mlp(d_model, d_model, 2)      # tactical position decoder (waypoints)
        self.subgoal_head = mlp(d_model + d_model, 2 * d_model, d_model)    # (h_t, h_goal) -> h*
        # ── Strategic (System 2): pooled discrete VQ concept + long-horizon predictor ──
        self.str_proj = nn.Linear(d_model, d_model)
        self.vq = VectorQuantizer(num_embeddings=num_codes, embedding_dim=d_model)
        self.str_predictor = mlp(d_model, 2 * d_model, d_model)             # c -> c'
        # ── Latent-RAG (queried in the inference loop) ──
        self.rag = LatentRAG(d_model=d_model, sim_threshold=rag_sim_threshold)
        # ── Collapse prevention ──
        self.sigreg = SIGReg(d_model=d_model, num_slices=sigreg_slices)

        self.register_buffer("pos_mean", torch.tensor([5.0, 5.0]))
        self.register_buffer("pos_std", torch.tensor([3.0, 3.0]))

    # ---- encoding / decoding ----
    def encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        clip = frame.unsqueeze(2).expand(-1, -1, self.SINGLE_FRAME_T, -1, -1)
        return self.encoder(clip)                          # [B,N,D]

    @staticmethod
    def pool(z): return z.mean(dim=1)                      # [B,D]

    def decode_pos_norm(self, z): return self.pos_head(self.pool(z) if z.dim() == 3 else z)
    def decode_pos(self, z): return self.decode_pos_norm(z) * self.pos_std + self.pos_mean

    # ---- operative ----
    def op_predict(self, z, a_onehot): return self.op_predictor(z, a_onehot)

    # ---- tactical ----
    def tac_encode(self, z):
        p = self.pool(z) if z.dim() == 3 else z
        h, moe_loss = self.moe(self.tac_proj(p).unsqueeze(1))   # [B,1,D]
        return h.squeeze(1), moe_loss
    def tac_decode_pos_norm(self, h): return self.tac_pos_head(h)
    def tac_decode_pos(self, h): return self.tac_pos_head(h) * self.pos_std + self.pos_mean
    def tac_predict(self, h, c): return self.tac_predictor(torch.cat([h, c], dim=-1))
    def emit_subgoal(self, h_t, h_goal): return self.subgoal_head(torch.cat([h_t, h_goal], dim=-1))

    # ---- strategic ----
    def str_pre(self, z):
        """Continuous pre-VQ strategic projection (used for anti-collapse reg)."""
        return self.str_proj(self.pool(z) if z.dim() == 3 else z)

    def str_encode(self, z):
        p = self.pool(z) if z.dim() == 3 else z
        cq, vq_loss, idx = self.vq(self.str_proj(p).unsqueeze(1))   # [B,1,D],_,[B,1]
        return cq.squeeze(1), vq_loss, idx.squeeze(1)
    def str_predict(self, c): return self.str_predictor(c)

    # ---- RAG (inference) ----
    @torch.no_grad()
    def rag_correct(self, z_pred, z_context):
        """Surprise-gated retrieval is handled by the caller; this just adds the
        retrieved pooled correction broadcast over tokens."""
        q = self.pool(z_context).unsqueeze(1)              # [B,1,D]
        corr = self.rag.retrieve_correction(q)             # [B,1,D]
        return z_pred + corr.expand(-1, z_pred.shape[1], -1)
