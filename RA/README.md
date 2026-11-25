# RA: Recovery Augmentation for UAV single-object tracking

This folder holds the code for the re-acquisition plug-in (RA) from the NF-UAVTrack paper. It contains the RA module, adapters for eight host trackers, and the evaluation scripts. The NF-UAVTrack dataset itself is documented in the [top-level README](../README.md) and hosted on [Hugging Face](https://huggingface.co/datasets/windowflower/NF-UAVTrack).

## What RA does

RA wraps an existing tracker without retraining it. The tracker is only used through a black-box interface: step on a frame, take or restore a snapshot of its state, and set its box. On every frame:

1. **Pass 1.** The host tracker runs unchanged. Its own confidence `p1` is read (for example, the peak of its score map).
2. **Trigger.** RA acts only if `p1 < tau`.
3. **Candidates.** Full-frame detector boxes (YOLOv12-M in the paper) with confidence ≥ κ0 = 0.25 are considered. A box is kept only if it does not overlap the pass-1 box (IoU ≤ 0.3) and its scale is within [0.5, 2] of the last confident box.
4. **Motion gate.** A constant-velocity Kalman filter predicts the target centre. For a candidate whose Mahalanobis distance d² to the prediction exceeds γ = 135.856 (while the last confident frame is at most L = 30 frames old), there are two cases:
   - with detector confidence ≥ κ1 = 0.5, it is kept but ranked after all gated-in candidates;
   - otherwise it is dropped.
5. **Verification.** For the top k = 3 candidates, the tracker is restored to its pre-pass-1 state, moved to the candidate and stepped again on the same frame. A result is accepted only if all three hold:
   - its score `s'` is above `p1`;
   - `s'` ≥ `tau`;
   - its box agrees with the detection (IoU > 0.3).

   The best accepted result becomes the output and the new tracker state. If none is accepted, the pass-1 result stands.
6. **Kalman update.** The Kalman filter is updated with confident outputs. It is re-initialised after an accepted far jump.

No ground truth is used after the first frame. The ablation variants in the paper are selected with `--variant`: `hard` and `nomo` change the gate, `reset` and `reset_nomo` reset directly to the detector box, and `--trigger periodic --period N` sets a periodic trigger.

## Layout

```text
RA/
├── run_ra.py               runner: one tracker, one RA setting, a list of sequences → predictions + metrics
├── ra/                     RA core
│   ├── recovery.py         trigger, candidate filtering, motion gate, ranking, verified re-search, acceptance
│   ├── kalman.py           fitted constant-velocity Kalman filter and gate threshold γ
│   └── geometry.py         IoU and box centre
├── adapters/               one adapter per host tracker (wraps the official repository, which you clone yourself)
│   ├── hiptrack.py  ostrack.py  stark.py  ortrack.py  mixformerv2.py  tctrackpp.py  siamrpnpp.py (SiamRPN++ and SiamRPN++ LT)
│   ├── openmmlab_siamrpn.py  maps OpenMMLab SiamRPN++ checkpoint keys to PySOT
│   └── compat.py           import shims that let the older tracker code run on current PyTorch
├── eval/
│   ├── eval_uav_sot.py     OPE metrics: success AUC, precision@20 px, normalised precision, absence-aware recall, drift, recovery time
│   ├── label_normalization.py  strict ground-truth / prediction loading (target-absent frames = NaN rows)
│   └── compare.py          paired per-sequence bootstrap CI and sign-flip p, tau selection, plug-in table
├── tools/
│   ├── nfuavtrack_to_sot.py    NF-UAVTrack (images/ + YOLO labels/) → tracking view (img/ + groundtruth.txt)
│   ├── prep_antiuav_rgb.py     Anti-UAV visible stream → tracking view (external benchmark)
│   ├── make_manifest.py        sequence manifest for run_ra.py
│   └── build_yolo_cache.py     cache YOLO detections for every frame
└── configs/
    ├── paper_configs.json      tau per tracker and the shared RA constants used in the paper
    ├── paper_plugin_table.json rows of the plug-in table
    ├── chi2_gate_fit.json      fitted Kalman noise multipliers
    └── splits/                 test (52 sequences), validation (16) and Anti-UAV validation (40) lists
```

## Installation

```sh
git clone https://github.com/windowFlowers/NF-UAVTrack
cd NF-UAVTrack/RA
pip install -r requirements.txt
```

This repository does not include tracker source code or weights. Clone the official repository of each host tracker you want to use, install its requirements, and download its released weights there:

| `--tracker` | Tracker | Official code | Config used (relative to the repo) | `tau` |
| --- | --- | --- | --- | ---: |
| `hiptrack` | HIPTrack | <https://github.com/WenRuiCai/HIPTrack> | `experiments/hiptrack/hiptrack.yaml` | 0.5 |
| `ostrack` | OSTrack-384 | <https://github.com/botaoye/OSTrack> | `experiments/ostrack/vitb_384_mae_ce_32x4_ep300.yaml` | 0.5 |
| `stark` | STARK-ST50 | <https://github.com/researchmm/Stark> | `experiments/stark_st2/baseline.yaml` | 0.5 |
| `ortrack` | ORTrack | <https://github.com/wuyou3474/ORTrack> | `experiments/ortrack/deit_tiny_patch16_224.yaml` | 0.5 |
| `tctrackpp` | TCTrack++ | <https://github.com/vision4robotics/TCTrack> | `experiments/TCTrack++/config.yaml` | 0.7 |
| `mixformer` | MixFormerV2-B | <https://github.com/MCG-NJU/MixFormerV2> | `experiments/mixformer2_vit_online/288_depth8_score.yaml` | 0.7 |
| `siamrpn` | SiamRPN++ R50 | <https://github.com/STVIR/pysot> | `experiments/siamrpn_r50_l234_dwxcorr/config.yaml` | 0.7 |
| `siamrpnlt` | SiamRPN++ LT | <https://github.com/STVIR/pysot> | `experiments/siamrpn_r50_l234_dwxcorr_lt/config.yaml` | 0.7 |

Weights and the commits used are listed in `configs/paper_configs.json`. For `siamrpn`, the paper used the OpenMMLab (MMTracking) SiamRPN++ UAV123 checkpoint, loaded through `adapters/openmmlab_siamrpn.py`. For `ostrack`, it used the ModelScope release of the `vitb_384_mae_ce_32x4_ep300` weights. The trackers were run without modifying their repositories.

## Data

1. **NF-UAVTrack.** Download it from <https://huggingface.co/datasets/windowflower/NF-UAVTrack> (`git lfs install && git clone https://huggingface.co/datasets/windowflower/NF-UAVTrack`). Then convert the test videos into the tracking layout:

   ```sh
   python tools/nfuavtrack_to_sot.py --src /path/to/NF-UAVTrack --out views --split ../splits/test.txt
   ```

   This writes `views/NF_UAVTrack/<video>/img/000001.jpg …` and `groundtruth.txt` (pixel `x,y,w,h`; `nan,nan,nan,nan` marks a target-absent frame).
2. **Other datasets.** The paper's 52-sequence test set also uses DDT, MAV-VID and DUT Anti-UAV; the validation split uses DDT and Purdue. These datasets are not redistributed here; get them from their original sources (see the paper's Data Availability statement). Arrange them in the same layout, `views/<Dataset>/<sequence>/{img/, groundtruth.txt}`, with the sequence names in `configs/splits/`. For the Anti-UAV external benchmark, use `tools/prep_antiuav_rgb.py`.
3. **Detections.** Cache one detection file per sequence:

   ```sh
   python tools/make_manifest.py --views views --cache-dir det_cache --split configs/splits/test.txt --out manifests/test.json --allow-missing
   python tools/build_yolo_cache.py --weights yolov12m.pt --manifest manifests/test.json --out det_cache
   ```

   Any detector can be used if its output is written in the same format, `{"dets": {"000001.jpg": [[x, y, w, h, conf], ...]}}`. The YOLOv12-M weights trained for the paper are not part of this repository.

## Running

Baseline (tracker alone) and tracker + RA:

```sh
python run_ra.py --tracker hiptrack --repo /path/to/HIPTrack --checkpoint /path/to/hiptrack_all_data.pth.tar \
    --manifest manifests/test.json --variant none --output runs/hiptrack_none
python run_ra.py --tracker hiptrack --repo /path/to/HIPTrack --checkpoint /path/to/hiptrack_all_data.pth.tar \
    --manifest manifests/test.json --variant soft --tau 0.5 --output runs/hiptrack_soft_t0.5
```

Each run writes the following per sequence:

- `predictions.txt`: `x,y,w,h,present` per frame;
- `ra_log.npy`: per-frame `p1`, score, triggered, accepted, number of re-searches, candidates and detections;
- `receipt.json`: the metrics.

`summary.json` holds the per-dataset means and the dataset-equal macro mean. Use `--seqs video08,video13` or `--limit-seqs N` for a quick test. Use `--cudnn-tf32` (or `RA_TF32=1`) to allow cuDNN TF32. The paper used it only for the SiamRPN++ runs on Anti-UAV; all other runs are FP32 with TF32 off and seed 42. Some trackers download backbone weights through `torch.hub` when they are built; `--torch-hub-dir` sets the cache directory.

## Reproducing the paper tables

1. **Validation and tau.** For each tracker, run `--variant soft` on the validation manifest for `--tau` ∈ {0.3, 0.4, 0.5, 0.6, 0.7}, then select tau:

   ```sh
   python -m eval.compare select-tau --pattern 'runs_val/ostrack_soft_t{tau}'
   ```

   The rule is the best dataset-equal AUC, with ties within 0.01 points going to the tau closest to 0.5. The values selected in the paper are in `configs/paper_configs.json`.
2. **Test.** On the 52 test sequences, run `--variant none` and `--variant soft --tau <selected>` for each tracker.
3. **Paired comparison** of one tracker:

   ```sh
   python -m eval.compare pair --base runs/hiptrack_none --ra runs/hiptrack_soft_t0.5
   ```

   Add `--exclude-domains DUT_AntiUAV` for the mean without DUT Anti-UAV.
4. **Plug-in table rows:**

   ```sh
   python -m eval.compare table --runs-root runs --spec configs/paper_plugin_table.json
   ```

5. **Ablations.** Use `--variant hard|nomo|reset|reset_nomo`, `--trigger periodic --period 8|15|30`, `--conf-min`, `--kappa1`, `--topk` and `--gate-frames` with all other settings at their defaults.

Gains and AUCs are deterministic for a given environment. CI and p values use fixed random streams (seed 0). The paper's fine-tuned HIPTrack and fine-tuned MixFormerV2 checkpoints and its detector weights are not distributed here.

## Citation

If you use this code, please cite the paper (the full reference will be added after publication):

```bibtex
@article{cai_nfuavtrack_ra,
  author  = {Cai, Zhexuan and others},
  title   = {TBD},
  journal = {Drones},
  year    = {2026},
  note    = {MDPI; manuscript drones-4700775, citation to be updated after publication}
}
```

## License

The code in this folder is released under the MIT License (see [LICENSE](LICENSE)). The NF-UAVTrack dataset is licensed separately under CC BY 4.0 (see the [top-level LICENSE](../LICENSE)). The host trackers, detectors and other datasets are subject to their own licences.
