"""Parameter-count breakdown across the old and new ALPS-4B models.

Run: PYTHONPATH=src python scripts/param_report.py
"""
import sys, os
sys.path.insert(0, "src")
import torch
from alps.core.alps_model import ALPSModel
from alps.benchmarks.two_rooms.train_two_rooms import TwoRoomsALPS
from alps.evaluation.repr_decoder_gate import ReprWorldModel
from alps.core.hier_world_model import HierWorldModel


def millions(n): return f"{n/1e6:.2f}M"


def report(name, model):
    total = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*64}\n{name}  —  TOTAL {total:,} ({millions(total)})\n{'-'*64}")
    rows = []
    for cname, child in model.named_children():
        p = sum(q.numel() for q in child.parameters())
        if p > 0:
            rows.append((cname, p))
    for cname, p in sorted(rows, key=lambda r: -r[1]):
        print(f"  {cname:<22} {p:>12,}  ({100*p/total:4.1f}%)")
    return total


print("### OLD SETUP (advertised) ###")
report("ALPSModel  (d=384, enc depth=12)  [README '22M config']",
       ALPSModel(d_model=384, d_action=64, encoder_depth=12, encoder_num_heads=6,
                 encoder_patch_size=(2, 16, 16), encoder_max_patches=256))
report("TwoRoomsALPS  (d=128)  [shipped 5%/0% model]",
       TwoRoomsALPS(d_model=128, encoder_depth=4, encoder_num_heads=4,
                    encoder_patch_size=(2, 16, 16), encoder_max_patches=512))

print("\n\n### OUR MODELS ###")
report("ReprWorldModel  (d=128)  [validated foundation, G1 0.19wu]",
       ReprWorldModel(d_model=128))
report("ReprWorldModel  (d=256)  [the A40 DMODEL=256 you ran]",
       ReprWorldModel(d_model=256))
report("HierWorldModel  (d=128)  [full learned hierarchy]",
       HierWorldModel(d_model=128))

# encoder-only sizes for the headline comparison
print(f"\n{'='*64}\nENCODER-ONLY (the dominant component)\n{'-'*64}")
for nm, m in [("ALPSModel d384 depth12", ALPSModel(d_model=384, encoder_depth=12, encoder_num_heads=6)),
              ("ReprWorldModel d128 depth4", ReprWorldModel(d_model=128)),
              ("ReprWorldModel d256 depth4", ReprWorldModel(d_model=256)),
              ("HierWorldModel d128 depth4", HierWorldModel(d_model=128))]:
    e = sum(p.numel() for p in m.encoder.parameters())
    print(f"  {nm:<28} encoder = {e:,} ({millions(e)})")
