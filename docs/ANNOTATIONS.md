# Annotation guide / 标注说明

## Directory structure / 目录结构

Extract `NF-UAVTrack.zip` into a dataset directory. The archive contains these two top-level folders:

将 `NF-UAVTrack.zip` 解压至数据集目录。压缩包包含以下两个顶层文件夹：

```text
NF-UAVTrack/
├── images/
│   ├── video01/
│   │   ├── frame-000001.jpg
│   │   └── ...
│   ├── video02/
│   └── ... video25/
└── labels/
    ├── video01/
    │   ├── frame-000001.txt
    │   └── ...
    ├── video02/
    └── ... video25/
```

Pair an image and its annotation using the sequence name and filename stem. Frame numbering starts at `000001` within each sequence. The original MP4 files are not included in this archive.

按序列名及文件名主干匹配图像与标注，每条序列从 `000001` 开始编号。该压缩包不含原始 MP4 文件。

## Bounding boxes / 目标框

Each non-empty annotation line has five whitespace-separated fields:

每条非空标注行包含五个以空白字符分隔的字段：

```text
class_id x_center y_center width height
```

| Field / 字段 | Meaning / 含义 |
| --- | --- |
| `class_id` | Always `0`, meaning UAV / 固定为 `0`，表示无人机 |
| `x_center` | Horizontal center divided by image width / 中心横坐标除以图像宽度 |
| `y_center` | Vertical center divided by image height / 中心纵坐标除以图像高度 |
| `width` | Box width divided by image width / 框宽除以图像宽度 |
| `height` | Box height divided by image height / 框高除以图像高度 |

This follows the normalized `xywh` convention described in the [Ultralytics YOLO format documentation](https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format). Coordinates are measured from the image's top-left corner.

其坐标含义与 [Ultralytics YOLO 格式说明](https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format)中的归一化 `xywh` 一致，图像坐标原点在左上角。

Example from `labels/video25/frame-000001.txt` / 实际标注示例：

```text
0 0.456982 0.441584 0.085176 0.131808
```

For image dimensions `W × H`, convert to pixel-space corners with:

对于宽高为 `W × H` 的图像，像素坐标转换公式为：

```python
x_min = (x_center - width / 2) * W
y_min = (y_center - height / 2) * H
x_max = (x_center + width / 2) * W
y_max = (y_center + height / 2) * H
```

## Class IDs / 类别编号

NF-UAVTrack is a **single-class, single-target dataset**. Its only class is **UAV / drone**, and all labels use ID `0`:

NF-UAVTrack 是**单类别、单目标数据集**，唯一类别为**无人机**，全部标注使用编号 `0`：

| Class ID / 类别编号 | Category / 类别 | Boxes / 框数 |
| ---: | --- | ---: |
| 0 | UAV / 无人机 | 10,936 |

Use one class named `uav` for training; no class-ID remapping is required. Each sequence concerns one UAV target, and each non-empty label file contains one bounding box. Class ID `0` is a category label, **not a track identity**.

训练时使用一个名为 `uav` 的类别即可，无需进行类别编号映射。每条序列对应一个无人机目标，每个非空标注文件包含一个目标框。`0` 是类别编号，**不是轨迹 ID**。

## Empty labels and evaluation / 空标注与评测

- The reference archive contains 1,191 empty TXT files. These have matching images and no annotated boxes. The files do not encode the reason for the missing box.
- Each of the 10,936 non-empty label files contains exactly one box.
- No object IDs, confidence scores, visibility flags, or occlusion attributes are stored in the TXT files.
- No predefined training/validation/test split is included. Report your sequence-level split and annotation processing when publishing results.

- 参考压缩包中有 1,191 个空 TXT 文件，均有对应图像；空文件没有标注框，也没有记录原因。
- 10,936 个非空标注文件均只有一个框。
- TXT 文件没有保存目标轨迹 ID、置信度、可见性或遮挡属性。
- 压缩包未提供预设训练、验证、测试划分；发表实验结果时，请说明使用的序列划分和标注处理方式。
