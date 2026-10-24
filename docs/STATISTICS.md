# Dataset statistics / 数据集统计

Verified on **2026-10-24** against the reference `NF-UAVTrack.zip` and its extracted files. / 于 **2026-10-24** 根据参考压缩包及解压文件核实。

- 25 sequences; 12,127 JPG images and 12,127 matching TXT files. / 25 条序列，12,127 张 JPG 图像及同名 TXT 标注。
- 10,936 bounding boxes; 1,191 empty label files. / 10,936 个目标框，1,191 个空标注文件。
- Single-class, single-target dataset: all boxes use class ID `0` (UAV). / 单类别、单目标数据集：全部目标框使用类别编号 `0`（无人机）。

| Sequence / 序列 | Frames / 帧数 | Boxes / 框数 | Empty labels / 空标注 | Resolution / 分辨率 | ID 0 |
| --- | ---: | ---: | ---: | --- | ---: |
| video01 | 583 | 333 | 250 | 3840 × 2160 | 333 |
| video02 | 775 | 775 | 0 | 3840 × 2160 | 775 |
| video03 | 2,488 | 2,488 | 0 | 3840 × 2160 | 2,488 |
| video04 | 443 | 401 | 42 | 3840 × 2160 | 401 |
| video05 | 296 | 233 | 63 | 3840 × 2160 | 233 |
| video06 | 335 | 104 | 231 | 3840 × 2160 | 104 |
| video07 | 566 | 557 | 9 | 3840 × 2160 | 557 |
| video08 | 157 | 140 | 17 | 3840 × 2160 | 140 |
| video09 | 520 | 477 | 43 | 3840 × 2160 | 477 |
| video10 | 300 | 299 | 1 | 1920 × 1080 | 299 |
| video11 | 300 | 299 | 1 | 1920 × 1080 | 299 |
| video12 | 300 | 288 | 12 | 1920 × 1080 | 288 |
| video13 | 300 | 300 | 0 | 1920 × 1080 | 300 |
| video14 | 499 | 374 | 125 | 1920 × 1080 | 374 |
| video15 | 257 | 257 | 0 | 1920 × 1080 | 257 |
| video16 | 375 | 234 | 141 | 1920 × 1080 | 234 |
| video17 | 751 | 595 | 156 | 1920 × 1080 | 595 |
| video18 | 193 | 186 | 7 | 1920 × 1080 | 186 |
| video19 | 168 | 138 | 30 | 1920 × 1080 | 138 |
| video20 | 814 | 795 | 19 | 1920 × 1080 | 795 |
| video21 | 319 | 310 | 9 | 1920 × 1080 | 310 |
| video22 | 295 | 295 | 0 | 1920 × 1080 | 295 |
| video23 | 149 | 126 | 23 | 1920 × 1080 | 126 |
| video24 | 261 | 254 | 7 | 1920 × 1080 | 254 |
| video25 | 683 | 678 | 5 | 1920 × 1080 | 678 |
| **Total / 合计** | **12,127** | **10,936** | **1,191** | — | **10,936** |

## Verification scope / 校验范围

Every image header was read to check resolution. Image and label filename sets match in all 25 sequences, and frame numbering is continuous from 1 within each sequence. Every non-empty label row has five fields, a nonnegative integer class ID, and finite normalized coordinates. All box widths and heights are positive; all boxes fall within image boundaries (tolerance: 0.00001).

The archive passes ZIP CRC checks. Every archived file matches the local file size and CRC. These checks establish file consistency and label syntax; they do not measure annotation accuracy or explain empty labels. The hosted Baidu Netdisk copy was not independently checked.

已读取全部图像文件头核对分辨率；25 条序列的图像与标注文件名完全匹配，帧编号均从 1 连续递增。全部非空标注行均为五列，类别为非负整数，坐标为有限的归一化数值，框宽高均大于零；所有框均在图像边界内（容差为 0.00001）。

压缩包通过 ZIP CRC 检查，所有归档文件与本地文件的大小和 CRC 一致。这些检查验证文件一致性与标注语法，不代表对标注精度或空标注原因的判断。网盘副本未独立核实。

Machine-readable files / 机器可读文件：[sequences.csv](../metadata/sequences.csv) · [dataset.json](../metadata/dataset.json) · [SHA256SUMS](../SHA256SUMS)
