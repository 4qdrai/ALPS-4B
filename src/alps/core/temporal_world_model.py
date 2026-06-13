"""
TemporalHierWorldModel — the full ALPS hierarchy with LeWM-style K-frame history
at EVERY scale (operative / tactical / strategic).

Same components as HierWorldModel (encoder, VQ strategic concept, MoE tactical,
goal-conditioned sub-goal head, Latent-RAG) but each predictor is now a
`CausalTemporalPredictor` that conditions on a multi-frame causal history instead
of a single frame. Trained on windows of W frames (teacher-forced), this restores
the LeWM design and extends it to the abstraction layers, where temporal context
is most valuable (momentum, slow concept dynamics, temporal path straightening).
"""

from __future__ import annotations
import torch
import torch.nn as nn

from alps.core.encoders import VisionEncoder
from alps.core.vq_bottleneck import VectorQuantizer
from alps.core.moe_router import SparseMoERouter
from alps.core.latent_rag import LatentRAG
from alps.core.sigreg import SIGReg
from alps.core.temporal_predictor import CausalTemporalPredictor


def mlp(d_in, d_h, d_out):
    return nn.Sequential(nn.Linear(d_in, d_h), nn.GELU(), nn.Linear(d_h, d_out))


class TemporalHierWorldModel(nn.Module):
    def __init__(self, d_model=192, d_action=4, num_codes=64, num_experts=4, active_experts=2,
                 enc_depth=10, enc_heads=8, patch_size=(2, 16, 16), max_patches=512,
                 op_depth=6, abs_depth=4, lambda_sigreg=0.1, sigreg_slices=256,
                 rag_sim_threshold=0.75, k_tac=2, k_str=4, max_frames=12,
                 use_projection_head=True):
        super().__init__()
        self.d_model = d_model
        self.k_tac, self.k_str = k_tac, k_str
        self.SINGLE_FRAME_T = 2
        self.lambda_sigreg = lambda_sigreg

        self.encoder = VisionEncoder(d_model=d_model, depth=enc_depth, num_heads=enc_heads,
                                     patch_size=patch_size, max_patches=max_patches,
                                     use_projection_head=use_projection_head)
        # operative: causal history over spatial tokens, conditioned on action
        self.op_predictor = CausalTemporalPredictor(d_model, d_cond=d_action, depth=op_depth,
                                                    num_heads=enc_heads, max_frames=max_frames)
        self.pos_head = mlp(d_model, d_model, 2)
        # tactical: pooled history, conditioned on strategic concept
        self.tac_proj = nn.Linear(d_model, d_model)
        self.moe = SparseMoERouter(d_model=d_model, num_experts=num_experts, active_experts=active_experts)
        self.tac_predictor = CausalTemporalPredictor(d_model, d_cond=d_model, depth=abs_depth,
                                                     num_heads=enc_heads, max_frames=max_frames)
        self.tac_pos_head = mlp(d_model, d_model, 2)
        self.subgoal_head = mlp(d_model + d_model, 2 * d_model, d_model)
        # strategic: pooled discrete-concept history, self-conditioned
        self.str_proj = nn.Linear(d_model, d_model)
        self.vq = VectorQuantizer(num_embeddings=num_codes, embedding_dim=d_model)
        self.str_predictor = CausalTemporalPredictor(d_model, d_cond=d_model, depth=abs_depth,
                                                     num_heads=enc_heads, max_frames=max_frames)
        self.rag = LatentRAG(d_model=d_model, sim_threshold=rag_sim_threshold)
        self.sigreg = SIGReg(d_model=d_model, num_slices=sigreg_slices)
        # LeWM "the predictor is also followed by a projector": a BatchNorm projector on
        # EACH scale's PREDICTED embedding, so SIGReg can be applied to the predictions
        # (z_t, z_{t+1} AND the prediction) at every level of the hierarchy -- operative,
        # tactical, strategic -- not just the operative encoder. Batch-stat BN (no
        # train/eval gap). Used in --lewm-ssl mode.
        def _pred_proj():
            return nn.Sequential(nn.Linear(d_model, d_model),
                                 nn.BatchNorm1d(d_model, track_running_stats=False))
        self.op_pred_proj = _pred_proj()
        self.tac_pred_proj = _pred_proj()
        self.str_pred_proj = _pred_proj()
        self.register_buffer("pos_mean", torch.tensor([5.0, 5.0]))
        self.register_buffer("pos_std", torch.tensor([3.0, 3.0]))

    # ---- encoding / decoding ----
    def encode_frame(self, frame):
        clip = frame.unsqueeze(2).expand(-1, -1, self.SINGLE_FRAME_T, -1, -1)
        return self.encoder(clip)                       # [B,N,D]

    @staticmethod
    def pool(z): return z.mean(dim=1)

    def decode_pos_norm(self, z): return self.pos_head(self.pool(z) if z.dim() == 3 else z)
    def decode_pos(self, z): return self.decode_pos_norm(z) * self.pos_std + self.pos_mean

    def tac_encode(self, z):
        p = self.pool(z) if z.dim() == 3 else z
        h, moe_loss = self.moe(self.tac_proj(p).unsqueeze(1))
        return h.squeeze(1), moe_loss
    def tac_decode_pos_norm(self, h): return self.tac_pos_head(h)
    def tac_decode_pos(self, h): return self.tac_pos_head(h) * self.pos_std + self.pos_mean
    def emit_subgoal(self, h_t, h_goal): return self.subgoal_head(torch.cat([h_t, h_goal], dim=-1))

    def str_pre(self, z): return self.str_proj(self.pool(z) if z.dim() == 3 else z)
    def str_encode(self, z):
        p = self.pool(z) if z.dim() == 3 else z
        cq, vq_loss, idx = self.vq(self.str_proj(p).unsqueeze(1))
        return cq.squeeze(1), vq_loss, idx.squeeze(1)

    # ---- windowed temporal prediction (training) ----
    def op_predict_window(self, z_win, a_win):
        """z_win [B,W,N,D], a_win [B,W,A] -> pred [B,W,N,D]."""
        return self.op_predictor(z_win, a_win)
    def tac_predict_window(self, h_win, c_win):
        """h_win [B,W,D], c_win [B,W,D] (strategic concept) -> pred [B,W,D]."""
        return self.tac_predictor(h_win.unsqueeze(2), c_win).squeeze(2)
    def str_predict_window(self, c_win):
        """c_win [B,W,D] -> pred [B,W,D] (self-conditioned: zero cond)."""
        cond = torch.zeros_like(c_win)
        return self.str_predictor(c_win.unsqueeze(2), cond).squeeze(2)

    # ---- next-step prediction (inference) ----
    def op_predict_next(self, z_hist, a_hist):
        """z_hist [B,K,N,D], a_hist [B,K,A] -> [B,N,D]."""
        return self.op_predictor.predict_next(z_hist, a_hist)

    @torch.no_grad()
    def rag_correct(self, z_pred, z_context):
        q = self.pool(z_context).unsqueeze(1)
        corr = self.rag.retrieve_correction(q)
        return z_pred + corr.expand(-1, z_pred.shape[1], -1)
