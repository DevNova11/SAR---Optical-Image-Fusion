# CRCD-Net ML Training Guide (Deep Fusion + Change Detection)

Audience: the 2-person **Fusion team** and the 2-person **Change-Detection team**.
Goal: train real models for the two parts that are currently rule-based or
untrained, **without breaking** the pipeline everyone else already depends on.

Read `DATA_CONTRACT.md` and `docs/RESEARCH_NOVELTY.md` first. This guide only
adds what is specific to training.

---

## 0. What exists today vs. what we are replacing

| Slot in the pipeline | Today | Training target |
|---|---|---|
| Fusion (`fusion/baseline.py`) | Per-channel z-score + weighted blend (no learning) | keep as **fallback + benchmark baseline** |
| Fusion (`fusion/deep_model.py::CRCDNet`) | Architecture exists, **untrained** (`train_fusion_model` is a one-batch stub) | **Fusion team trains this** |
| Reliability fusion (`fusion/reliability_fusion.py::fuse_reliability_aware`) | Hand-built reliability maps; accepts an optional `model=CRCDNet` | trained `CRCDNet` plugs in here |
| Semantic classes (`semantics/land_cover_classifier.py::LandCoverClassifier`) | Hand-tuned rule scores + softmax. Its own docstring says: *not a trained model* | **Change-Detection team trains this** |
| Generic change (`change_detection/`, `compare()`) | Normalize, abs-diff, Otsu, morphology | stays; used as a benchmark baseline |
| Provenance, persistence, confidence (`provenance/`) | Consumes classifier output | unchanged, just gets better input |

Rule: **every trained model must be a drop-in for an existing function with the
same inputs/outputs, with the current rule-based version kept as a fallback.**
If a trained model is worse than the baseline on held-out data, we ship the baseline.

---

## 1. Shared ground rules (both teams)

### 1.1 Data contract (frozen, do not change shapes)
- `s1`: `(H, W, 2)` float32, bands `[VV, VH]`, dB (about -25..0)
- `s2`: `(H, W, 4)` float32, bands `[B2, B3, B4, B8]`, reflectance 0..1
- Channels **last**. 10 m pixels, per-AOI UTM grid. S1 and S2 of the same date are pixel-aligned.
- `NODATA = -9999.0` (not NaN). **Training must mask these pixels out of the loss** and must not feed raw -9999 into a network. A handful of nodata pixels per scene is normal (0.0007%-0.034%).
- Load data only through `handoff.get_training_pair(...)` / `local_preprocessing.load_geotiff`. Do not re-read GeoTIFFs with your own band order.

### 1.2 Normalization: use fixed dataset-level statistics
Per-image z-scoring (what `baseline.py` does) makes the same ground look different on
two dates, which **creates fake change**. For anything you train:
- Compute mean/std per band **once over the training set**, save them next to the weights (`*.json`), and apply the same constants at inference, for every date.
- Never normalize using the statistics of the single image being processed.

### 1.3 Data volume
Cached today: about 25 S1/S2 date-pairs (3 demo AOIs with 4 dates each, a mining AOI, ~10 `custom_*` AOIs). That is enough to debug, not enough to train well.
- Pull more AOIs with the custom-coordinates path (`gee_data_collection.collect_pair`, or the dashboard's Custom Location). Aim for **40+ AOIs across South India**, 3-4 dates each, mixing dry-season and monsoon scenes (monsoon is where SAR matters most).
- Cut scenes into patches (`local_preprocessing.extract_patches`, 128x128 or 256x256 with stride; drop patches > 30% nodata).
- Optional: pretrain on a public S1+S2 dataset (e.g. SEN12MS) and fine-tune on ours.

### 1.4 Splits: by AOI, never by patch
Random patch splits leak (neighbouring patches share pixels/texture) and inflate every metric.
- Hold out **whole AOIs** for validation and test. Agree the list once and write it to `evaluation/splits.json`; nobody edits it later.
- The split lives in `evaluation/splits.json` (**v2**, 25 train / 6 val / 7 test AOIs; every split contains all 5 classes). v1 was replaced because Bare Land was 0.3% of train but 36% of val. Do not edit it; the test AOIs are touched once, at the end.
- Fix random seeds. Report mean +/- std over at least 3 seeds.

### 1.5 Environment
- Train on **Colab (GPU)**. `torch` is not installed locally and Python 3.14 wheels may not exist.
- Keep training code in the repo (`.py`, not only notebooks) so results are reproducible.
- Add `torch` to a separate `requirements-train.txt`, not to `requirements.txt` (the dashboard/backend must keep running without torch, and `reliability_fusion.py` already imports torch lazily for that reason; keep it that way).

### 1.6 Git workflow (important, we have been burned before)
- Branch from the **latest `origin/main`**: `git fetch && git checkout -b ml-fusion origin/main` (or `ml-change-detection`).
- **Never merge or push an old local clone/branch into `main`.** A stale clone once carried a branch that deleted the entire `CRCD-Net/` tree. If your clone is more than a few days old, re-clone.
- Do not commit data, `outputs/`, or large weights. `.gitignore` already excludes `*.db`, `outputs/`. Keep committed weights under about 10 MB; put anything larger on Drive/Release and link it in the PR. (Large pushes have been timing out on our network; split big commits.)
- Run `python tests/test_all_modules.py` before opening a PR. Open PRs into `main`; the other team reviews.
- Do not change a public function signature or the `ChangeDetectionResult` / `SemanticClassificationResult` fields without telling the other team first.

---

## 2. Fusion team: train CRCDNet

### 2.1 Interface you must satisfy
Inference must work through the existing wrappers with no caller changes:

```python
fuse_deep(s1_hwc, s2_hwc, model=trained, device="cuda", data_layout="HWC") -> (H, W, 4) float32
fuse_reliability_aware(s1, s2, model=trained, data_layout="HWC")
    -> (fused, w_sar, w_opt, metrics)
```
- Output: same `(H, W, C_out)`, `float32`, finite, `C_out` equal for both dates (`compare()` requires identical shapes and channel counts).
- Same output **radiometry on both dates** (see 1.2). `compare()` min-max normalizes each fused image separately, so a model whose output scale drifts between dates will show phantom change.
- Add `load_pretrained_crcdnet(path)` that restores weights **and** the normalization stats, so callers cannot forget them.

### 2.2 There is no "ground-truth fused image"; use a self-supervised task
`train_fusion_model` currently regresses against a `target_reference` that does not exist for real data. Replace it with **mask-and-reconstruct**:

1. Take an aligned S1+S2 patch from one date.
2. Randomly blank regions of the **optical** input (synthetic clouds, 10-40% of the patch; also drop single bands sometimes).
3. Train the network to reconstruct the **true optical** pixels from SAR + the remaining optical.
4. Loss (masked to valid, non-nodata pixels):
   - L1 on the hidden optical region (primary)
   - plus a small gradient/edge loss against SAR structure so the output cannot ignore SAR
   - optional SSIM term

This gives a real target (the held-out true optical) and teaches the network when to trust SAR, which is the paper's claim.

Notes:
- `CRCDNet` ends in `Sigmoid`, so train on optical scaled to 0..1 with the fixed stats.
- Add a **temporal-consistency check** to evaluation: fuse the same ground on two dates with no real change (stable AOI areas) and confirm the fused difference stays low.
- Train/validate with a proper `Dataset` + `DataLoader`, mini-batches, an LR schedule and early stopping on the validation AOIs. The existing loop (one full-batch step per epoch) is a placeholder.

### 2.3 Suggested split of work (2 people)
- **Person F1, data + training:** dataset builder, mask-and-reconstruct generator, training script (`fusion/train_fusion.py`), checkpointing, Colab notebook.
- **Person F2, evaluation + integration:** metrics module, held-out evaluation, ablations, wiring the trained model into `fuse_reliability_aware`, dashboard method selector, fallback logic.

### 2.4 Evaluation (F2 owns; no `fusion/metrics.py` exists yet, so create it)
On held-out AOIs, for **baseline weighted, reliability-aware (untrained), and trained CRCDNet**:
- Reconstruction (on masked-optical task): PSNR, SSIM, RMSE, SAM (spectral angle)
- Downstream: change-map quality using the same `compare()` on each fusion's output, vs the Dynamic World deltas in `data/*_land_cover.json`
- Stability: false-change rate on stable regions
- Optional ablations: SAR-only input, optical-only input, no-attention (`SpatialChannelGating` off)

Definition of done: trained model beats both baselines on the held-out test AOIs for the reconstruction metrics **and** does not increase false change on stable regions. Otherwise report it honestly and keep the baseline as default.

---

## 3. Change-Detection team: train the semantic classifier

### 3.1 What to train, and why this slot
The pipeline's change signal comes from **class transitions** (Forest -> Urban etc.) fed into `provenance/trajectory_engine.py` and the persistence verifier. Today those classes come from a hand-tuned rule classifier. Replacing it with a trained model is the highest-value change-detection upgrade, and it plugs in without touching `provenance/`.

Keep `compare()` (generic difference detector) as-is. It remains the "Baseline A" in `evaluation/benchmark_suite.py`.

### 3.2 Interface you must satisfy
```python
class TrainedLandCoverClassifier:
    def predict(self, s1_image, s2_image, fused_image=None) -> SemanticClassificationResult
```
- Same method name, same `SemanticClassificationResult` (`class_map (H,W) int32`, `probabilities (H,W,5) float32`, `confidence_margin`, `class_counts`, `class_fractions`).
- Same 5 classes and order from `LAND_COVER_CLASSES`: `0 Forest, 1 Agriculture, 2 Urban, 3 Bare Land, 4 Water`.
- Keep `LandCoverClassifier` (rule-based) as the fallback; select via a flag/factory, default stays rule-based until the trained model passes the bar below.
- **Train on raw S1+S2 (6 channels), not on fused images.** Then the classifier does not depend on whichever fusion model is current, and the two teams can work in parallel. Treat `fused_image` as optional extra input only if the fusion team's output is stable.
- Probabilities must be real softmax outputs; the confidence engine multiplies `P_margin` into the confidence score, so overconfident outputs distort priority ranking. Calibrate (temperature scaling on validation AOIs) and report calibration error.

### 3.3 Labels: weak labels from Dynamic World
There is no labeled land-cover data. The repo already pulls Google's Dynamic World (a pretrained model) in `land_cover.py`, but only as **scalar per-AOI means**. For training you need **per-pixel** labels:

1. For each cached scene date, export the Dynamic World class raster on the **same UTM grid and extent** as that scene's S1/S2 (reuse `build_dw_snapshot`, take the argmax of the class probabilities or the `label` band, export with `geemap` like `_export_geotiff`). Name it `{aoi}_{date}_DW.tif`.
2. Map DW's 9 classes to our 5 (**proposal, agree as a team and freeze it**):

| Dynamic World | CRCD-Net |
|---|---|
| trees, shrub_and_scrub | Forest |
| grass, crops, flooded_vegetation | Agriculture |
| built | Urban |
| bare, snow_and_ice | Bare Land |
| water | Water |

3. Keep only confident pixels: ignore pixels where the top merged-class probability is below `min_prob` (label set to `ignore_index` in the loss). Default is **0.5**; 0.6 ignored ~60% of pixels on `chimakurthy_quarry` because DW is uncertain there.
4. Mind class imbalance (Urban and Bare are rare): weighted cross-entropy or focal loss.

Be upfront in the report: this is **distillation / weak supervision from Dynamic World**, not human ground truth. The model can at best match DW plus add SAR robustness (clouds, monsoon). Say that, do not claim "accuracy vs ground truth".

### 3.4 Model
Start small: a U-Net (or the existing CNN style) over 6 input channels, 5-class output, 128x128 patches. Only escalate (SegFormer, Siamese change head) if the small model clearly underperforms.
Optional second stage: a Siamese change head taking the two dates' features and predicting "changed vs not", with change labels derived from DW class differences. Do this only after the classifier works.

### 3.5 Evaluation (held-out AOIs)
- Per-class IoU / F1 and mean IoU vs DW labels
- Same, restricted to the SAR-helps cases (patches that are cloudy/hazy in optical): the claim to test is "trained SAR+optical beats optical-only and the rule classifier under cloud"
- Calibration (ECE) of the probabilities
- **Change-level evaluation:** transition accuracy between dates vs DW transitions (e.g. built-up gain on `bengaluru_sarjapur` should be positive; quarry AOI should show tree/vegetation -> bare)
- Run the full ablation in `evaluation/benchmark_suite.py` (Stage 5 to 7) with the trained classifier swapped in, and add a row for it
- Sanity check against the known stories: Bengaluru urbanization, Oragadam industrial, Chimakurthy quarry. If the trained model contradicts them, investigate before trusting it

Definition of done: higher mean IoU than the rule-based classifier on held-out AOIs, better-or-equal calibration, no regression on the three known-story AOIs. Otherwise keep rule-based as default and report the negative result.

### 3.6 Suggested split of work (2 people)
- **C1, labels + data:** DW label export, class mapping, patch dataset, splits file, augmentation.
- **C2, model + evaluation:** training script (`semantics/train_classifier.py`), calibration, `TrainedLandCoverClassifier`, benchmark integration.

---

## 4. Handoffs between the two teams

- Fusion team does not need the classifier, and the classifier does not need the fusion model (it trains on raw S1+S2). The only shared artifacts are the data contract, the splits file, and `evaluation/benchmark_suite.py`.
- Once both exist, run the full benchmark with the 2x2: {baseline fusion, trained fusion} x {rule classifier, trained classifier}. That table is the main result of the paper.
- `data/`, `evaluation/splits.json`, and the DW class mapping are **shared and frozen**; changing them invalidates the other team's numbers. Announce any change.

## 5. Suggested order (no hard deadlines, but do it in this order)

1. **Foundation (everyone):** agree splits and class mapping; pull more AOIs; Colab environment; fixed normalization stats; evaluation harness running on the *untrained* baselines first, so every later number has a reference.
2. **Train v1:** fusion mask-and-reconstruct; classifier on DW labels. Small models, overfit one AOI first to prove the loop works.
3. **Iterate:** augmentation, losses, calibration, ablations; compare to baselines on validation AOIs.
4. **Integrate:** drop-in loaders, dashboard method selector (baseline vs trained), fallbacks, `tests/`.
5. **Final eval + writeup:** test AOIs touched **once**; report mean +/- std; include negative results.

## 6. Honesty rules for the report

- Say "weak supervision from Dynamic World", not "ground truth".
- Report held-out-AOI numbers only. Do not quote validation numbers as the result after tuning on them.
- If the trained model does not beat the baseline, say so and show the baseline is the shipped default.
- Do not describe the rule-based classifier or `generate_direction_labels()` as trained or scientific classification.


---

## 7. Status and findings so far (update this section, do not rewrite history)

### Built
- `fusion/train_fusion.py`, `fusion/trained_fusion.py`, `fusion/metrics.py`: mask-and-reconstruct trainer, `fuse_trained()` drop-in, masked PSNR/SSIM/RMSE/SAM.
- `semantics/dw_labels.py`, `semantics/train_classifier.py`, `semantics/trained_classifier.py` (`get_classifier()` falls back to rule-based): per-pixel Dynamic World labels aligned to each scene's S2 grid, U-Net trainer, drop-in classifier.
- `pull_training_aois.py` (resumable AOI puller), `tests/test_dw_labels.py`.
- Data: 38 AOIs / 142 scenes, all with DW labels.

### Fusion results (validation AOIs, masked-region reconstruction)
Trained CRCDNet with a context-fill prior reaches PSNR 19.33 vs 19.70 for a no-SAR Gaussian fill (worse on PSNR/RMSE, better on SSIM 0.534 vs 0.514). SAR helps structure (SAM 6.7 vs 9.5 deg when SAR is removed) but not absolute reflectance level. **Not ready to be the default fusion.** CRCDNet's receptive field is ~13 px; next step is a larger-receptive-field (U-Net style) model and more data. These numbers used split v1 and are not comparable to v2.

### Classifier results (validation AOIs, split v2, labels = Dynamic World)
| | mIoU | accuracy |
|---|---|---|
| Trained U-Net | 0.722 | 0.863 |
| Rule-based `LandCoverClassifier` | 0.572 | 0.723 |
Trained is better on Forest, Urban, Water and roughly equal on Bare Land; Agriculture is the weakest class (IoU ~0.48). Calibration ECE 0.024.
Test split NOT yet evaluated (reserved for the final run).

### Caveats you must repeat in the report
1. **Scores measure agreement with Dynamic World, not with ground truth.**
2. **The "no SAR" ablation is unfair to the no-SAR case.** It zeroes SAR at test time on a model trained to use it (distribution shift), so 0.722 vs 0.643 overstates SAR's value. A proper comparison needs an optical-only model trained from scratch. The "cloudy, no SAR" row has no information at all and tests nothing; ignore it.
3. The known-story check (Bengaluru Urban up, Chimakurthy Forest down) was run on AOIs that are in the **training** split.

### Mistakes made and lessons (so nobody repeats them)
- AOI coordinates picked from memory were city centers (Ludhiana, Guntur, Thanjavur, Mandya, Anantapur, Ooty, Madikeri, Kolar, Jaisalmer), not farmland/forest/desert. They came out 50-95% Built. Fix: screen every candidate point against ESA WorldCover (`ESA/WorldCover/v200`) **before** downloading imagery; those scenes were renamed `*_city`/`*_town`.
- Check class composition per AOI and per split before the first training run; our first split had almost no Bare Land in train.
