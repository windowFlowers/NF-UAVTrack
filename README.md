# NF-UAVTrack

English | **[中文说明（完整中文版）](README.zh-CN.md)**

NF-UAVTrack is a dataset created by **caizhexuan (蔡喆炫)**, comprising **25 video sequences**, **12,127 extracted frames**, and **10,936 UAV bounding boxes**. This repository provides the download entry point, annotation documentation, and dataset statistics for researchers.

| Sequences | Frames | Bounding boxes | Image resolutions |
| ---: | ---: | ---: | --- |
| 25 | 12,127 | 10,936 | 3840 × 2160 and 1920 × 1080 |

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

The reference archive `NF-UAVTrack.zip` contains extracted JPG frames and TXT annotations in `images/` and `labels/`; it does **not** contain the original video files. Cloning this repository or using GitHub's **Download ZIP** downloads the documentation and two preview samples, not the full dataset.

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

## Dataset information

| Field | Details |
| --- | --- |
| Name | NF-UAVTrack |
| Creator | caizhexuan (蔡喆炫) |
| Video sequences | 25 (`video01`–`video25`) |
| Extracted frames / annotation files | 12,127 / 12,127, paired by filename |
| Bounding boxes | 10,936 |
| Empty annotation files | 1,191 |
| Download host | Baidu Netdisk |
| Annotation format | YOLO-style normalized bounding boxes: `class_id x_center y_center width height` |
| Class IDs | Both `0` and `1` represent **UAV / drone**; original IDs are preserved |
| Train / validation / test split | No predefined split is included in the reference archive |

**Class IDs:** there are 10,253 boxes with ID `0` and 683 with ID `1`. They refer to the same semantic category. For a single-class detector, map both IDs to `0` in a separate training copy. Do not interpret them as different object categories.

**Tracking:** annotations contain per-frame bounding boxes, without object track IDs or visibility flags. No official tracking protocol or benchmark scores are provided.

- [Directory structure and annotation format / 标注格式](docs/ANNOTATIONS.md)
- [Per-sequence statistics / 逐序列统计](docs/STATISTICS.md)
- [Machine-readable statistics (CSV)](metadata/sequences.csv)

For experiments, document your sequence-level split and treatment of empty labels. An empty TXT file records no annotated box; it does not by itself explain target absence, occlusion, or annotation policy.

## Usage terms

No dataset license has been selected yet. This initial publication provides a public download entry point; it does not establish a general reuse or redistribution license. Please contact the maintainer through [GitHub Issues](https://github.com/windowFlowers/NF-UAVTrack/issues) to clarify usage terms before reusing or redistributing the data.

## Citation

When referencing this dataset, credit **caizhexuan (蔡喆炫)** and link to this repository. A repository citation is provided below; it is not a citation to a research paper.

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
