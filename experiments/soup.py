"""
E31: average the final weights of several seeds of one recipe into ONE model (a "uniform soup").

Two seeds of one recipe differ by as much as the experiments they are meant to judge (19 section 3.1). Models
fine-tuned from the same pretrained weights usually sit in one basin, so their weights can be averaged: one model,
no extra inference cost, and less of each seed's luck (Wortsman et al., arXiv 2203.05482). Whether that holds
for OUR fine-tunes is the experiment: if the seeds are not in one basin the soup's recall collapses, which the
standard evaluation shows at once.

    C:\\cuas_venv\\Scripts\\python experiments/soup.py E18_soup_s1s2 E18_base_s1 E18_base_s2
    C:\\cuas_venv\\Scripts\\python experiments/e12_eval.py E18_soup_s1s2@last

Writes <runs>/<name>/weights/last.pt (and soup.json, naming the members). Every member's FINAL weights are used
(the rule of 19 section 3.4). Members must share an architecture.
"""
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # a checkpoint may pickle cuas.arch modules
D = Path(os.environ.get("CUAS_DATA", "C:/cuas_data"))


def soup(name, members):
    ckpts = [torch.load(D / "runs" / m / "weights" / "last.pt", map_location="cpu", weights_only=False) for m in members]
    for m, c in zip(members, ckpts):
        if c.get("epoch", -1) != -1:
            raise SystemExit(f"{m}: last.pt is not a finished run (epoch {c['epoch']}); finish training first")
    states = [c["model"].float().state_dict() for c in ckpts]
    if any(s.keys() != states[0].keys() for s in states):
        raise SystemExit("members differ in architecture")
    mean = {k: (sum(s[k] for s in states) / len(states) if v.is_floating_point() else v)      # counters: the first's
            for k, v in states[0].items()}
    out = ckpts[0]
    out["model"].load_state_dict(mean)
    out["model"].half()
    out["train_args"] = dict(out.get("train_args") or {}, name=name)
    folder = D / "runs" / name
    (folder / "weights").mkdir(parents=True, exist_ok=True)
    torch.save(out, folder / "weights" / "last.pt")
    (folder / "soup.json").write_text(json.dumps({"members": [f"{m}@last" for m in members]}, indent=1))
    spread = max(float((a - b).abs().max()) for k in mean if mean[k].is_floating_point()
                 for a, b in [(states[0][k], states[-1][k])])
    print(f"{name}: mean of {len(members)} final weights ({', '.join(members)}); largest single-weight difference "
          f"between the first and last member {spread:.3f}; wrote {folder / 'weights' / 'last.pt'}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    soup(sys.argv[1], sys.argv[2:])
