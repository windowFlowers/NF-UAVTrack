# NF-UAVTrack

English | **[中文说明（完整中文版）](README.zh-CN.md)**

NF-UAVTrack is a **single-class, single-target visible-light UAV dataset** recorded **at night in fog**, created by **caizhexuan (蔡喆炫)**. It comprises **25 video sequences**, **12,127 extracted frames** and **10,936 UAV bounding boxes**; the remaining **1,191 frames are annotated as target-absent**. The only class is **UAV / drone**, with class ID **`0`**. This repository provides the download entry point, the official training/test split, annotation documentation and dataset statistics.

| Sequences | Frames | Bounding boxes | Target-absent frames | Image resolutions |
| ---: | ---: | ---: | ---: | --- |
| 25 | 12,127 | 10,936 | 1,191 | 3840 × 2160 and 1920 × 1080 |

## Preview

| video01 / frame-000001 | video25 / frame-000001 |
| --- | --- |
| ![Original frame from video01](examples/video01-frame-000001.jpg) | ![Original frame from video25](examples/video25-frame-000001.jpg) |

These are unmodified dataset frames. Their matching annotation files are included in [examples/](examples/).

## Download

| Source | Link | Extraction code |
| --- | --- | --- |
| Baidu Netdisk | [Download NF-UAVTrack.zip](https://pan.baidu.com/s/1dzHpwilAFCqUif20WGW_OA?pwd=0854) | `0854` |

1. Open the download link above.
2. Enter `0854` if prompted for an extraction code.
3. Download the shared `NF-UAVTrack.zip` archive and extract it.

The reference archive `NF-UAVTrack.zip` contains extracted JPG frames and TXT annotations in `images/` and `labels/`; it does **not** contain the original video files. Cloning this repository or using GitHub's **Download ZIP** downloads the documentation, the split files and two preview samples, not the full dataset.

Reference archive size: **1,225,785,649 bytes** (approximately 1.23 GB). Its SHA-256 is available in [SHA256SUMS](SHA256SUMS). The reference archive was checked locally; the hosted copy has not been independently verified.

After downloading, compare the archive hash with `SHA256SUMS`:

```powershell
# Windows PowerShell
Get-FileHash .\NF-UAVTrack.zip -Algorithm SHA256
```

```sh
# Linux (run beside the downloaded archive and SHA256SUMS)
sha256sum -c SHA256SUMS
```

## Official split

The dataset is split by video into **14 training videos** and **11 test videos**. This is the split used in the accompanying paper; please use it when you compare results with that paper or with each other. No test video contributes training samples. The lists are also provided as plain-text files in [splits/](splits/).

| Subset | Videos | Frames | Bounding boxes | Target-absent frames | List |
| --- | ---: | ---: | ---: | ---: | --- |
| Training | 14 | 7,426 | 6,645 | 781 | [splits/train.txt](splits/train.txt) |
| Test | 11 | 4,701 | 4,291 | 410 | [splits/test.txt](splits/test.txt) |

- **Training:** video01, video03, video04, video05, video06, video07, video10, video11, video12, video15, video17, video18, video21, video22
- **Test:** video02, video08, video09, video13, video14, video16, video19, video20, video23, video24, video25

No separate validation videos are defined in NF-UAVTrack. If you need a validation set, take it from the training videos (or from other datasets) and report which videos you used. If you use a different split, please report it.

## Dataset information

| Field | Details |
| --- | --- |
| Name | NF-UAVTrack |
| Creator | caizhexuan (蔡喆炫) |
| Video sequences | 25 (`video01`–`video25`) |
| Extracted frames / annotation files | 12,127 / 12,127, paired by filename |
| Bounding boxes | 10,936 |
| Target-absent frames (empty annotation files) | 1,191 |
| Imaging conditions | Night-time, foggy |
| Acquisition devices | Xiaomi 12S; iPhone 15 Pro |
| Acquisition location | Future Campus, Soochow University, Suzhou, Jiangsu, China |
| Recording dates and time | April 2025 and April 2026; 23:00–01:00 the following day (China Standard Time, UTC+8) |
| UAV models | YH F450; DJI Mavic 4 |
| Annotation workflow | Two annotators labelled separate sets of videos and reviewed each other's annotations |
| Download host | Baidu Netdisk |
| Annotation format | YOLO-style normalized bounding boxes: `class_id x_center y_center width height` |
| Dataset type | Single-class, single-target |
| Class ID | `0` = **UAV / drone** (the only class) |
| Train / test split | 14 / 11 videos, see [Official split](#official-split) |

**Annotations:** all 10,936 bounding boxes use class ID `0`; the current dataset contains no class `1` labels. Each sequence concerns a single UAV target. Each non-empty annotation file contains exactly one bounding box. An empty annotation file marks a **target-absent frame**, in which the target has left the field of view or is not visible; 20 of the 25 videos contain such frames, in 108 separate absence intervals. No class-ID remapping is required for single-class training.

**Tracking:** annotations contain per-frame bounding boxes, without object track IDs. Target absence is given by the empty annotation files rather than by a separate visibility flag. A tracker is initialised on the first frame of a sequence that has a bounding box. The accompanying paper evaluates trackers on the 11 test videos together with test sequences from DDT, MAV-VID and DUT Anti-UAV, using one-pass evaluation with success AUC, precision at 20 pixels and normalized precision, and it also measures recovery after target absence; see the paper for the full protocol.

- [Directory structure and annotation format / 标注格式](docs/ANNOTATIONS.md)
- [Per-sequence statistics / 逐序列统计](docs/STATISTICS.md)
- [Machine-readable statistics (CSV)](metadata/sequences.csv)

## Usage terms

No dataset license has been selected yet. This initial publication provides a public download entry point; it does not establish a general reuse or redistribution license. Please contact the maintainer through [GitHub Issues](https://github.com/windowFlowers/NF-UAVTrack/issues) to clarify usage terms before reusing or redistributing the data.

## Citation

When referencing this dataset, credit **caizhexuan (蔡喆炫)** and link to this repository. A repository citation is provided below; a citation of the accompanying paper will be added after publication.

```bibtex
@misc{caizhexuan_nf_uavtrack,
  author       = {Cai, Zhexuan},
  title        = {NF-UAVTrack},
  howpublished = {Dataset repository},
  url          = {https://github.com/windowFlowers/NF-UAVTrack}
}
```

Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).

## Questions and feedback

For download problems, annotation questions, or dataset feedback, please [open an issue](https://github.com/windowFlowers/NF-UAVTrack/issues/new/choose). Include the relevant file or sequence name and a short description of the problem.