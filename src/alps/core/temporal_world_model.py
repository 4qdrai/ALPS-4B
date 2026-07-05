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
                 use_projection_head=True, use_cls_pool=False, residual_pred=False,
                 film_cond=False, flow_pred=False, slot_mode=False, num_slots=6,
                 slot_dec_hidden=None, slot_dec_depth=2, slot_motion=False):
        super().__init__()
        self.flow_pred = flow_pred
        self.slot_mode = slot_mode
        self.num_slots = num_slots
        self.d_model = d_model
        self.k_tac, self.k_str = k_tac, k_str
        self.SINGLE_FRAME_T = 2
        self.lambda_sigreg = lambda_sigreg
        # CLS-token pooling (LeWM-exact): read the encoder's [CLS] token (index 0) as the
        # pooled latent instead of mean-pooling all tokens. Mean-pool dilutes the spatially
        # localized agent signal -> pure-SSL pooled latent loses position (G1 fails) even
        # though the token grid keeps it; [CLS] attends and selects it. See encoders.py.
        self.use_cls_pool = use_cls_pool

        self.encoder = VisionEncoder(d_model=d_model, depth=enc_depth, num_heads=enc_heads,
                                     patch_size=patch_size, max_patches=max_patches,
                                     use_projection_head=use_projection_head,
                                     use_cls_token=use_cls_pool)
        # operative: causal history over spatial tokens, conditioned on action.
        # In SLOT MODE the same predictor runs over K object slots instead of N grid tokens
        # (CausalTemporalPredictor is token-count agnostic) -> the agent is a DEDICATED slot,
        # so its dynamics are modeled undiluted, independent of pixel size. Binding is learned
        # JOINTLY with prediction + feature-reconstruction (the post-hoc probe ablation showed
        # position-entangled tokens make bolted-on slots bind regions, not objects).
        self.op_predictor = CausalTemporalPredictor(d_model, d_cond=d_action, depth=op_depth,
                                                    num_heads=enc_heads, max_frames=max_frames,
                                                    residual=residual_pred, film_cond=film_cond,
                                                    flow=flow_pred)
        if slot_mode:
            from alps.core.slot_readout import SlotAttention, SlotFeatureDecoder
            n_tok = (128 // patch_size[1]) * (128 // patch_size[2])   # tokens per frame @128px
            self.slot_attn = SlotAttention(d_model, num_slots=num_slots)
            # decoder capacity must scale with token count: the 2-layer/2D default (built for
            # 64 tokens) loses the reconstruction tug-of-war at 256 tokens (patch-8) -> rec
            # rises, slots churn, inv diverges (measured on the first patch-8 pod run).
            self.slot_dec = SlotFeatureDecoder(d_model, n_tok, hidden=slot_dec_hidden,
                                               depth=slot_dec_depth)
            # MOTION-CUED binding (SAVi's flow cue, label-free latent form): slot attention
            # sees each token fused with its TEMPORAL DELTA. Appearance/recon-driven binding
            # demonstrably fails to discover the small moving agent (masks: region tiles at
            # patch-16, whole-frame collapse at patch-8); objects are what MOVE COHERENTLY,
            # and here the agent is the only mover -> the delta channel is a perfect cue.
            self.slot_motion = slot_motion
            if slot_motion:
                self.slot_fuse = nn.Linear(2 * d_model, d_model)
        self.pos_head = mlp(d_model, d_model, 2)
        # tactical: pooled history, conditioned on strategic concept
        self.tac_proj = nn.Linear(d_model, d_model)
        self.moe = SparseMoERouter(d_model=d_model, num_experts=num_experts, active_experts=active_experts)
        self.tac_predictor = CausalTemporalPredictor(d_model, d_cond=d_model, depth=abs_depth,
                                                     num_heads=enc_heads, max_frames=max_frames,
                                                     residual=residual_pred, film_cond=film_cond)
        self.tac_pos_head = mlp(d_model, d_model, 2)
        self.subgoal_head = mlp(d_model + d_model, 2 * d_model, d_model)
        # strategic: pooled discrete-concept history, self-conditioned
        self.str_proj = nn.Linear(d_model, d_model)
        self.vq = VectorQuantizer(num_embeddings=num_codes, embedding_dim=d_model)
        self.str_predictor = CausalTemporalPredictor(d_model, d_cond=d_model, depth=abs_depth,
                                                     num_heads=enc_heads, max_frames=max_frames,
                                                     residual=residual_pred, film_cond=film_cond)
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
        # Inverse-dynamics head: predict the action between two pooled latents. Trained on
        # the NON-detached pooled latent, it FORCES the compact readout (CLS or mean) to
        # encode the controllable agent state -- the pressure that pure next-latent pred +
        # SIGReg lack, which left the SSL pooled latent position-blind (G1 fail). Actions
        # here are the agent's own experience (proprioception), not external supervision,
        # so it stays label-free in the JEPA sense; for passive video this becomes a LATENT
        # action inferred from adjacent frames (AdaWorld/LAM/CLAW). Used in --inv-dyn.
        self.d_action = d_action
        self.inv_head = mlp(2 * d_model, d_model, d_action)
        self.register_buffer("pos_mean", torch.tensor([5.0, 5.0]))
        self.register_buffer("pos_std", torch.tensor([3.0, 3.0]))

    # ---- encoding / decoding ----
    def encode_frame(self, frame):
        clip = frame.unsqueeze(2).expand(-1, -1, self.SINGLE_FRAME_T, -1, -1)
        return self.encoder(clip)                       # [B,N,D]

    def pool(self, z):
        """[B,N,D] -> [B,D]. CLS token (index 0) when use_cls_pool, else mean over tokens."""
        return z[:, 0] if self.use_cls_pool else z.mean(dim=1)

    def slots_of(self, z, slots_init=None, z_prev=None):
        """[B,N,D] token grid -> [B,K,D] object slots (slot mode only). Drops a CLS token if
        present. Slots are the OPERATIVE state: the predictor imagines slot dynamics.
        `slots_init` = previous frame's slots (SAVi recurrent binding). With slot_motion, the
        attention input is each token fused with its TEMPORAL DELTA (zeros when no z_prev)."""
        x = z[:, 1:] if self.use_cls_pool else z
        if getattr(self, "slot_motion", False):
            xp = (z_prev[:, 1:] if self.use_cls_pool else z_prev) if z_prev is not None else None
            delta = (x - xp) if xp is not None else torch.zeros_like(x)
            x = self.slot_fuse(torch.cat([x, delta], dim=-1))
        return self.slot_attn(x, slots_init=slots_init)

    def slots_of_window(self, z_win):
        """[B,W,N,D] -> [B,W,K,D] with RECURRENT binding: slots at frame t initialize from the
        slots at t-1, so slot identity is temporally consistent and slot DYNAMICS are a
        well-posed prediction target (v1's independent per-frame init let identities permute).
        The handoff is DETACHED: identity consistency is a forward-pass property; backprop
        through the W-step slot-attention chain is an RNN that destabilizes training
        (measured: op 0.86->6.4 in 4 epochs with full BPTT). Truncated-BPTT-1, SAVi-style.
        With slot_motion each frame's tokens are fused with their temporal delta."""
        outs, s = [], None
        for t in range(z_win.shape[1]):
            s = self.slots_of(z_win[:, t], slots_init=(s.detach() if s is not None else None),
                              z_prev=(z_win[:, t - 1] if t > 0 else None))
            outs.append(s)
        return torch.stack(outs, dim=1)

    def encode_frame_slots(self, frame, slots_init=None):
        """frame [B,3,H,W] -> [B,K,D] slots (the slot-mode control representation).
        Pass the previous step's slots as `slots_init` for recurrent online binding."""
        return self.slots_of(self.encode_frame(frame), slots_init=slots_init)

    def tok_pool(self, t):
        """[B,W,N,D] -> [B,W,D] (token-dim pool). CLS index 0 or mean, matching pool()."""
        return t[:, :, 0] if self.use_cls_pool else t.mean(dim=2)

    def inverse_action(self, zp_t, zp_next):
        """Inverse dynamics: predict the action from a pair of pooled latents.
        zp_t, zp_next: [...,D] -> action logits [...,d_action]."""
        return self.inv_head(torch.cat([zp_t, zp_next], dim=-1))

    def spatial_readout(self, z, grid=4):
        """Position-faithful COMPACT state: a coarse grid×grid average-pool of the spatial
        tokens, flattened to [B, grid*grid*D]. pool() collapses to one global vector, which
        discards the small agent under pure SSL (G1 random); this keeps the agent's location
        (validated: 4x4 decodes position at R^2 0.92 on the pure-SSL model). The hierarchy's
        graph + control use this so planning is position-faithful WITHOUT supervision. The
        same readout carries to real video (coarse object/scene grid)."""
        spatial = z[:, 1:] if self.use_cls_pool else z        # drop a [CLS] at index 0 if present
        b, N, D = spatial.shape
        s = int(round(N ** 0.5))                              # 8 at patch16, 16 at patch8
        x = spatial.reshape(b, s, s, D).permute(0, 3, 1, 2)   # [b,D,s,s]
        x = torch.nn.functional.adaptive_avg_pool2d(x, (grid, grid))
        return x.reshape(b, -1)                               # [b, grid*grid*D]

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
