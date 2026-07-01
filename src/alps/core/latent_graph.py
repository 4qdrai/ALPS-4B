"""
Latent transition graph for ALPS-4B strategic planning.

The README/paper describe "discrete conceptual landmarks" (VQ codes) and a
"latent graph", but no graph object exists in the codebase — the VQ codebook is
only ever used as a per-frame label. This module builds the missing piece: a
directed graph whose nodes are latent landmarks and whose edges are observed
transitions, then plans shortest paths over it. This is what lets the strategic
layer do real work (turn a cross-room / keyed task into an ordered list of
reachable sub-goals) instead of relying on a hard-coded door image.

Pipeline
--------
1. Encode dataset frames -> pooled latents.
2. Cluster latents into K nodes (landmarks). Each node stores:
     - latent centroid,
     - decoded (x, y) centroid (via the position probe) -> used as a waypoint,
     - mean TRUE (x, y) and room id  -> for validation/visualization only.
3. Edges: for consecutive in-episode frames, accumulate transition counts
   node(z_t) -> node(z_{t+1}). Edge cost = -log(p_transition).
4. Plan: map a start/goal latent to its nearest node, run Dijkstra, return the
   ordered list of decoded waypoints to feed the operative MPC.

Self-learning hook: `add_transition` lets new experience add/strengthen edges
online, so a previously unreachable goal can become reachable after exploration.
"""

from __future__ import annotations

import heapq
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def _kmeans(x: np.ndarray, k: int, iters: int = 50, seed: int = 0):
    """Lightweight k-means (no sklearn dependency required)."""
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(x)
        return km.cluster_centers_, km.labels_
    except Exception:
        rng = np.random.RandomState(seed)
        centers = x[rng.choice(len(x), k, replace=False)].copy()
        labels = np.zeros(len(x), dtype=np.int64)
        x2 = (x ** 2).sum(1)[:, None]                      # [N,1]
        for _ in range(iters):
            # ||x-c||^2 = ||x||^2 - 2 x·c + ||c||^2 via matmul -> [N,k], never the [N,k,D]
            # broadcast tensor (which is 16+ GiB for the 12288-d spatial readout).
            d = x2 - 2.0 * (x @ centers.T) + (centers ** 2).sum(1)[None, :]
            labels = d.argmin(1)
            for j in range(k):
                m = labels == j
                if m.any():
                    centers[j] = x[m].mean(0)
        return centers, labels


@dataclass
class LatentGraph:
    k: int
    centroids: np.ndarray              # [k, D] latent landmark centroids
    decoded_xy: np.ndarray             # [k, 2] decoded waypoint positions
    true_xy: np.ndarray                # [k, 2] mean true positions (validation)
    room_id: np.ndarray                # [k] majority room id (validation)
    edges: np.ndarray                  # [k, k] transition counts
    # --- optional: SEMANTIC graph (centroids live in decoded (x,y,key) feature
    # space, not raw latent space). z_centroids keeps the mean pooled LATENT per
    # node (so the tactical layer can still be conditioned on str_encode(node)).
    z_centroids: np.ndarray = None     # [k, D] mean pooled latent per node
    key_state: np.ndarray = None       # [k] mean has-key score per node
    key_node: int = None               # index of the explicit key-acquisition landmark

    def node_of_latent(self, z_pooled: np.ndarray) -> int:
        d = ((self.centroids - z_pooled[None, :]) ** 2).sum(-1)
        return int(d.argmin())

    def add_transition(self, src: int, dst: int, weight: float = 1.0):
        self.edges[src, dst] += weight

    def _cost_matrix(self) -> np.ndarray:
        row = self.edges.sum(1, keepdims=True)
        p = self.edges / np.clip(row, 1e-8, None)
        with np.errstate(divide="ignore"):
            cost = -np.log(p)
        cost[self.edges <= 0] = np.inf
        return cost

    def shortest_path(self, src: int, dst: int) -> List[int]:
        cost = self._cost_matrix()
        n = self.k
        dist = [float("inf")] * n
        prev = [-1] * n
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                break
            if d > dist[u]:
                continue
            for v in range(n):
                c = cost[u, v]
                if np.isfinite(c) and d + c < dist[v]:
                    dist[v] = d + c
                    prev[v] = u
                    heapq.heappush(pq, (dist[v], v))
        if dist[dst] == float("inf"):
            return []
        path, u = [], dst
        while u != -1:
            path.append(u)
            u = prev[u]
        return path[::-1]

    def waypoints(self, start_z: np.ndarray, goal_z: np.ndarray) -> List[np.ndarray]:
        """Decoded (x, y) sub-goals along the latent-graph shortest path."""
        s, g = self.node_of_latent(start_z), self.node_of_latent(goal_z)
        path = self.shortest_path(s, g)
        if not path:
            return [self.decoded_xy[g].copy()]
        # skip the start node itself; sub-goals are subsequent nodes
        return [self.decoded_xy[n].copy() for n in path[1:]] or [self.decoded_xy[g].copy()]


@torch.no_grad()
def build_latent_graph(
    model, decode_fn, dataset, device, k: int = 16,
    max_frames: int = 6000, seed: int = 0,
) -> LatentGraph:
    """Encode dataset frames, cluster into landmarks, accumulate transition edges."""
    pooled, true_xy, rooms, ep_of_frame = [], [], [], []
    # We need per-episode consecutive structure for edges; iterate clips and use
    # their consecutive frames (clips never cross episode boundaries).
    frame_count = 0
    clip_latents: List[np.ndarray] = []
    clip_break: List[int] = []  # index marking clip boundaries in the flat list
    for i in range(len(dataset)):
        s = dataset[i]
        fr = s["video_frames"].to(device)            # [3,T,H,W]
        T = fr.shape[1]
        z = model.encode_frame(fr.permute(1, 0, 2, 3))  # [T,N,D]
        zp = z.mean(dim=1).cpu().numpy()              # [T,D]
        clip_latents.append(zp)
        clip_break.append(T)
        true_xy.append(s["positions"].numpy())
        rooms.append(s["room_ids"].numpy())
        frame_count += T
        if frame_count >= max_frames:
            break

    Z = np.concatenate(clip_latents, 0)
    XY = np.concatenate(true_xy, 0)
    RM = np.concatenate(rooms, 0)
    centroids, labels = _kmeans(Z, k, seed=seed)

    decoded_xy = np.zeros((k, 2), dtype=np.float32)
    true_centroid = np.zeros((k, 2), dtype=np.float32)
    room_major = np.zeros(k, dtype=np.int64)
    cent_t = torch.tensor(centroids, dtype=torch.float32, device=device)
    # decode the centroid (probe expects [B,N,D]; give it [k,1,D])
    decoded = decode_fn(cent_t.unsqueeze(1)).cpu().numpy()
    decoded_xy[:] = decoded
    for j in range(k):
        m = labels == j
        if m.any():
            true_centroid[j] = XY[m].mean(0)
            room_major[j] = int(np.round(RM[m].mean()))

    # edges from consecutive frames within each clip
    edges = np.zeros((k, k), dtype=np.float64)
    off = 0
    for T in clip_break:
        lab = labels[off:off + T]
        for t in range(T - 1):
            edges[lab[t], lab[t + 1]] += 1.0
        off += T

    return LatentGraph(k=k, centroids=centroids.astype(np.float32),
                       decoded_xy=decoded_xy, true_xy=true_centroid,
                       room_id=room_major, edges=edges)
