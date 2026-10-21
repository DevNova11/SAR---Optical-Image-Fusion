"""Train the 5-class land-cover classifier on Dynamic World weak labels.

WEAK SUPERVISION: labels are Dynamic World's (a pretrained model), merged to our
5 classes in semantics/dw_labels.py. The model can at best match DW plus add SAR
robustness; report it that way, never as accuracy vs ground truth.

Input  (B, 7, H, W): S1 [VV,VH] z-scored, S2 [B2,B3,B4,B8] z-scored, optical-validity channel.
        Fixed train-set statistics (saved in the checkpoint), invalid/hidden pixels -> 0.
Target (B, H, W) int64 in 0..4, 255 = ignore (low-confidence DW, nodata).
Augmentation: flips/rot90, plus random optical "cloud" masking on ~half the patches so the
model must learn a SAR fallback (the claim we want to test: SAR+optical beats optical-only under cloud).

Run from CRCD-Net/ :
    python -m semantics.train_classifier --iters 3000
    python -m semantics.train_classifier --eval-only --split test     # touch the test split ONCE, at the end
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fusion.train_fusion import (  # noqa: E402
    compute_norm_stats, list_scenes, load_splits, load_store, normalize, random_hide_mask,
)
from semantics.dw_labels import DEFAULT_MIN_PROB, IGNORE_INDEX, NUM_CLASSES, load_dw_labels  # noqa: E402
from semantics.land_cover_classifier import CLASS_NAMES  # noqa: E402

IN_CH = 7
WEIGHTS = os.path.join(PROJECT_ROOT, "semantics", "weights", "classifier_dw.pt")


# ---------------------------------------------------------------- data
def load_labeled_store(data_dir: str, aois: List[str], min_prob: float):
    """(aoi,date) -> (s1, s2, valid, labels). Scenes without an exported DW raster are skipped (and reported)."""
    store = load_store(data_dir, aois)
    out = {}
    for key, (s1, s2, valid) in store.items():
        lab = load_dw_labels(key[0], key[1], data_dir, valid=valid, min_prob=min_prob)
        if lab is None:
            print(f"SKIP {key}: no DW labels exported yet")
            continue
        out[key] = (s1, s2, valid, lab[0])
    return out


def class_weights(store) -> np.ndarray:
    counts = np.zeros(NUM_CLASSES, np.float64)
    for *_, labels in store.values():
        for c in range(NUM_CLASSES):
            counts[c] += (labels == c).sum()
    freq = counts / max(counts.sum(), 1)
    w = 1.0 / np.sqrt(np.maximum(freq, 1e-4))  # inverse-sqrt: balances without exploding rare classes
    return (w / w[freq > 0].mean()).astype(np.float32), freq


def make_inputs(s1n, s2n, vis_opt):
    """(H,W,*) -> (7,H,W): optical hidden where vis_opt is False."""
    x = np.concatenate([s1n, s2n * vis_opt[..., None], vis_opt[..., None].astype(np.float32)], axis=-1)
    return np.ascontiguousarray(np.transpose(x, (2, 0, 1)), dtype=np.float32)


def make_batch(store, keys, norm, rng, patch: int, batch: int, p_cloud: float = 0.5):
    import torch

    xs, ys = [], []
    for _ in range(batch):
        for _try in range(30):
            key = keys[int(rng.integers(len(keys)))]
            s1, s2, valid, lab = store[key]
            h, w = valid.shape
            if h < patch or w < patch:
                continue
            y, x = int(rng.integers(0, h - patch + 1)), int(rng.integers(0, w - patch + 1))
            sl = (slice(y, y + patch), slice(x, x + patch))
            if (lab[sl] != IGNORE_INDEX).mean() >= 0.2:
                break
        s1p, s2p, vp, lp = s1[sl], s2[sl], valid[sl], lab[sl]
        s1n, s2n = normalize(s1p, s2p, vp, norm)
        vis = vp.copy()
        if rng.random() < p_cloud:
            vis &= ~random_hide_mask(patch, patch, rng, 0.2, 0.7)
        k, flip = int(rng.integers(4)), bool(rng.integers(2))
        arrs = [np.rot90(a, k) for a in (s1n, s2n, vis, lp)]
        if flip:
            arrs = [a[:, ::-1] for a in arrs]
        s1n, s2n, vis, lp = [np.ascontiguousarray(a) for a in arrs]
        xs.append(make_inputs(s1n, s2n, vis))
        ys.append(lp.astype(np.int64))
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


# ---------------------------------------------------------------- model
def build_model(base: int = 24):
    import torch.nn as nn
    import torch.nn.functional as F

    def block(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        )

    class SmallUNet(nn.Module):
        """3-level U-Net: receptive field is far larger than CRCDNet's ~13 px (a lesson from the fusion run)."""

        def __init__(self):
            super().__init__()
            self.e1, self.e2, self.e3 = block(IN_CH, base), block(base, 2 * base), block(2 * base, 4 * base)
            self.mid = block(4 * base, 8 * base)
            self.d3, self.d2, self.d1 = block(8 * base + 4 * base, 4 * base), block(4 * base + 2 * base, 2 * base), block(2 * base + base, base)
            self.head = nn.Conv2d(base, NUM_CLASSES, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            h, w = x.shape[-2:]
            ph, pw = (-h) % 8, (-w) % 8
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph), mode="reflect")
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            m = self.mid(self.pool(e3))
            up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
            d3 = self.d3(torch_cat(up(m), e3))
            d2 = self.d2(torch_cat(up(d3), e2))
            d1 = self.d1(torch_cat(up(d2), e1))
            return self.head(d1)[..., :h, :w]

    import torch

    torch_cat = lambda a, b: torch.cat([a, b], dim=1)
    return SmallUNet()


def predict_probs(model, norm, s1, s2, valid, vis_opt=None, zero_sar=False, device="cpu"):
    """Full-scene softmax probabilities (H, W, 5)."""
    import torch

    s1n, s2n = normalize(s1, s2, valid, norm)
    if zero_sar:
        s1n = np.zeros_like(s1n)
    vis = valid if vis_opt is None else (vis_opt & valid)
    x = torch.from_numpy(make_inputs(s1n, s2n, vis))[None].to(device)
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    return np.transpose(p, (1, 2, 0))


# ---------------------------------------------------------------- metrics
def confusion(pred, lab):
    m = lab != IGNORE_INDEX
    return np.bincount(lab[m].astype(np.int64) * NUM_CLASSES + pred[m].astype(np.int64), minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)


def scores(cm):
    tp = np.diag(cm).astype(float)
    union = cm.sum(0) + cm.sum(1) - tp
    present = cm.sum(1) > 0
    iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
    return {"mIoU": float(np.nanmean(iou[present])) if present.any() else float("nan"),
            "acc": float(tp.sum() / max(cm.sum(), 1)), "iou": iou}


def ece(probs, lab, bins=15, max_pix=300000, seed=0):
    m = lab != IGNORE_INDEX
    p, y = probs[m], lab[m]
    if len(y) > max_pix:
        idx = np.random.default_rng(seed).choice(len(y), max_pix, replace=False)
        p, y = p[idx], y[idx]
    conf, pred = p.max(-1), p.argmax(-1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(correct[s].mean() - conf[s].mean())
    return float(e)


def evaluate(model, store, norm, device="cpu", seed=77, rule_based=True):
    """Held-out scenes. Conditions: clean, and 'cloudy' (random optical regions hidden; scored on the hidden pixels)."""
    import warnings

    cms = {k: np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64) for k in
           ["trained_clean", "trained_noSAR_clean", "trained_noOptical_clean", "trained_cloudy", "trained_noSAR_cloudy"]}
    cm_rule = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    all_p, all_l = [], []
    if rule_based:
        from semantics.land_cover_classifier import LandCoverClassifier

        rule = LandCoverClassifier()
    for i, key in enumerate(sorted(store)):
        s1, s2, valid, lab = store[key]
        p = predict_probs(model, norm, s1, s2, valid, device=device)
        cms["trained_clean"] += confusion(p.argmax(-1), lab)
        all_p.append(p[::4, ::4]), all_l.append(lab[::4, ::4])
        cms["trained_noSAR_clean"] += confusion(predict_probs(model, norm, s1, s2, valid, zero_sar=True, device=device).argmax(-1), lab)
        # no-optical: hide ALL optical
        cms["trained_noOptical_clean"] += confusion(predict_probs(model, norm, s1, s2, valid, vis_opt=np.zeros_like(valid), device=device).argmax(-1), lab)
        hide = random_hide_mask(*valid.shape, np.random.default_rng(seed + i), 0.3, 0.5)
        lab_h = np.where(hide, lab, IGNORE_INDEX).astype(np.uint8)
        cms["trained_cloudy"] += confusion(predict_probs(model, norm, s1, s2, valid, vis_opt=~hide, device=device).argmax(-1), lab_h)
        cms["trained_noSAR_cloudy"] += confusion(predict_probs(model, norm, s1, s2, valid, vis_opt=~hide, zero_sar=True, device=device).argmax(-1), lab_h)
        if rule_based:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s2c = np.where(valid[..., None], s2, 0.0)
                s1c = np.where(valid[..., None], s1, -20.0)
                cm_rule += confusion(rule.predict(s1c, s2c).class_map.astype(np.uint8), lab)
    res = {k: scores(v) for k, v in cms.items()}
    if rule_based:
        res["rule_based_clean"] = scores(cm_rule)
    res["_ece"] = ece(np.concatenate([a.reshape(-1, NUM_CLASSES) for a in all_p]), np.concatenate([a.reshape(-1) for a in all_l]))
    return res


def print_results(res):
    print(f"{'condition':26s} {'mIoU':>6s} {'acc':>6s}   per-class IoU " + " ".join(f"{n[:6]:>6s}" for n in CLASS_NAMES))
    for k, v in res.items():
        if k.startswith("_"):
            continue
        print(f"{k:26s} {v['mIoU']:6.3f} {v['acc']:6.3f}   " + " " * 14 + " ".join(f"{x:6.2f}" for x in v["iou"]))
    print(f"calibration ECE (clean, trained): {res['_ece']:.4f}")


# ---------------------------------------------------------------- train
def main():
    import torch
    import torch.nn.functional as F

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data"))
    ap.add_argument("--out", default=WEIGHTS)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--min-prob", type=float, default=DEFAULT_MIN_PROB)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    splits = load_splits()

    if a.eval_only:
        ck = torch.load(a.out, map_location=a.device)
        model = build_model(ck["config"]["base"]).to(a.device)
        model.load_state_dict(ck["state_dict"])
        store = load_labeled_store(a.data_dir, splits[a.split], ck["config"]["min_prob"])
        print(f"eval on {a.split}: {sorted({k[0] for k in store})} ({len(store)} scenes)")
        print_results(evaluate(model, store, ck["norm"], a.device))
        return

    train = load_labeled_store(a.data_dir, splits["train"], a.min_prob)
    val = load_labeled_store(a.data_dir, splits["val"], a.min_prob)
    if not train or not val:
        sys.exit("No labeled scenes: run `python -m semantics.dw_labels` first to export Dynamic World labels.")
    norm = compute_norm_stats({k: (v[0], v[1], v[2]) for k, v in train.items()})
    w, freq = class_weights(train)
    print(f"train scenes={len(train)} val scenes={len(val)} device={a.device}")
    print("train label freq:", {n: round(float(f), 3) for n, f in zip(CLASS_NAMES, freq)}, "weights:", w.round(2))
    keys = sorted(train)
    model = build_model().to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.iters)
    wt = torch.from_numpy(w).to(a.device)
    cfg = {"base": 24, "min_prob": a.min_prob, "patch": a.patch, "in_ch": IN_CH}
    best, bad, t0 = -1.0, 0, time.time()
    for it in range(1, a.iters + 1):
        model.train()
        x, y = [t.to(a.device) for t in make_batch(train, keys, norm, rng, a.patch, a.batch)]
        loss = F.cross_entropy(model(x), y, weight=wt, ignore_index=IGNORE_INDEX)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 25 == 0:
            print(f"it {it:5d} loss={loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
        if it % a.eval_every == 0 or it == a.iters:
            res = evaluate(model, val, norm, a.device, rule_based=(it == a.eval_every))
            print("--- val ---")
            print_results(res)
            score = res["trained_clean"]["mIoU"]
            if score > best:
                best, bad = score, 0
                os.makedirs(os.path.dirname(a.out), exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "norm": norm, "config": cfg,
                            "meta": {"iter": it, "val_mIoU": score, "seed": a.seed, "label_freq": freq.tolist()}}, a.out)
                print(f"saved best (val mIoU {score:.3f}) -> {a.out}", flush=True)
            else:
                bad += 1
                if bad >= a.patience:
                    print("early stop")
                    break
    print(f"done in {time.time() - t0:.0f}s, best val mIoU {best:.3f}")


if __name__ == "__main__":
    main()
