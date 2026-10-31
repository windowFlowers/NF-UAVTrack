# NF-UAVTrack 数据集（中文说明）

简体中文 | [English](README.md)

NF-UAVTrack 是由 **caizhexuan（蔡喆炫）** 创建的**单类别、单目标可见光无人机数据集**，全部在**夜间雾天**拍摄，包含 **25 条视频序列**（覆盖**四类场景**：空旷草地、操场、公园、楼宇间；以及**两个录制时期**：2025 年 4 月、2026 年 4 月）、**12,127 张抽帧图像及 10,936 个无人机目标框**，其余 **1,191 帧标注为目标消失**。唯一类别为**无人机**，类别编号为 **`0`**。本仓库提供下载入口、官方训练/测试划分、标注说明和数据统计。

> **数据集已发布在 Hugging Face Hub：** [windowflower/NF-UAVTrack](https://huggingface.co/datasets/windowflower/NF-UAVTrack) —— 可在线浏览文件、无需提取码，支持 `git clone`。详见[数据集下载](#数据集下载)。

| 视频序列 | 图像帧 | 目标框 | 目标消失帧 | 图像分辨率 |
| ---: | ---: | ---: | ---: | --- |
| 25 | 12,127 | 10,936 | 1,191 | 3840 × 2160、1920 × 1080 |

## 数据预览

| video01 / frame-000001 | video25 / frame-000001 |
| --- | --- |
| ![video01 原始图像帧](examples/video01-frame-000001.jpg) | ![video25 原始图像帧](examples/video25-frame-000001.jpg) |

以上为未经修改的原始图像帧，对应标注文件见 [examples/](examples/)。

## 数据集下载

| 下载平台 | 链接 | 获取方式 |
| --- | --- | --- |
| **Hugging Face Hub**（推荐） | [windowflower/NF-UAVTrack](https://huggingface.co/datasets/windowflower/NF-UAVTrack) | 无需提取码，可在线浏览或单独下载文件 |
| 百度网盘 | [下载 NF-UAVTrack.zip](https://pan.baidu.com/s/1dzHpwilAFCqUif20WGW_OA?pwd=0854) | 提取码 `0854` |

两个来源的数据完全相同，均为 `images/` 下的 JPG 抽帧图像和 `labels/` 下的 TXT 标注。

**方式一：Hugging Face Hub**

1. 打开 <https://huggingface.co/datasets/windowflower/NF-UAVTrack>。
2. 在 **Files and versions** 中浏览或单独下载文件，或克隆整个数据集：

   ```sh
   git lfs install
   git clone https://huggingface.co/datasets/windowflower/NF-UAVTrack
   ```

**方式二：百度网盘**

1. 打开上方下载链接。
2. 如页面要求输入提取码，请输入 `0854`。
3. 下载分享文件 `NF-UAVTrack.zip` 并解压。

参考压缩包 `NF-UAVTrack.zip` 包含 `images/` 下的 JPG 抽帧图像和 `labels/` 下的 TXT 标注，**不含原始视频文件**。克隆本仓库或点击 GitHub 的 **Download ZIP** 仅下载文档、划分文件和两组预览样例，不会下载完整数据集。

参考压缩包大小为 **1,225,785,649 字节**（约 1.23 GB），SHA-256 校验值见 [SHA256SUMS](SHA256SUMS)。该参考压缩包已在本地校验；[Hugging Face 副本](https://huggingface.co/datasets/windowflower/NF-UAVTrack)已与解压后的参考数据逐文件比对：24,258 个文件的数量与大小完全一致，抽样文件的 SHA-256 相同。

下载后可计算压缩包哈希值，并与 `SHA256SUMS` 比较：

```powershell
# Windows PowerShell
Get-FileHash .\NF-UAVTrack.zip -Algorithm SHA256
```

```sh
# Linux：在压缩包和 SHA256SUMS 所在目录运行
sha256sum -c SHA256SUMS
```

## 官方划分

数据集按视频划分为 **14 段训练视频**和 **11 段测试视频**，与配套论文使用的划分相同；与该论文或其他工作比较结果时，请使用这一划分。测试视频不提供任何训练样本。视频列表也以纯文本形式放在 [splits/](splits/) 中。

| 子集 | 视频数 | 图像帧 | 目标框 | 目标消失帧 | 列表 |
| --- | ---: | ---: | ---: | ---: | --- |
| 训练 | 14 | 7,426 | 6,645 | 781 | [splits/train.txt](splits/train.txt) |
| 测试 | 11 | 4,701 | 4,291 | 410 | [splits/test.txt](splits/test.txt) |

- **训练：** video01、video03、video04、video05、video06、video07、video10、video11、video12、video15、video17、video18、video21、video22
- **测试：** video02、video08、video09、video13、video14、video16、video19、video20、video23、video24、video25

NF-UAVTrack 本身不单独划出验证视频。如需验证集，请从训练视频（或其他数据集）中选取，并说明所用视频；如使用其他划分，也请在结果中说明。

## 快速使用

解压后，按视频序列名和文件名匹配图像与标注。例如：

```text
NF-UAVTrack/
├── images/
│   ├── video01/
│   │   ├── frame-000001.jpg
│   │   └── ...
│   └── ...（共 25 个序列）
└── labels/
    ├── video01/
    │   ├── frame-000001.txt
    │   └── ...
    └── ...（共 25 个序列）
```

非空 TXT 文件采用 YOLO 风格的五列标注：

```text
类别编号 中心横坐标 中心纵坐标 目标框宽度 目标框高度
```

后四项均为归一化数值：横坐标和框宽除以图像宽度，纵坐标和框高除以图像高度。以下为 `video25/frame-000001.txt` 的实际内容：

```text
0 0.456982 0.441584 0.085176 0.131808
```

唯一类别为 `0`（无人机），单类别训练时无需再合并或转换类别编号。空标注文件表示该帧目标消失。完整格式说明见[标注文档](docs/ANNOTATIONS.md)。

## 数据集信息

| 项目 | 内容 |
| --- | --- |
| 名称 | NF-UAVTrack |
| 创建者 | caizhexuan（蔡喆炫） |
| 视频序列 | 25 条（`video01`–`video25`） |
| 图像帧 / 标注文件 | 12,127 / 12,127，文件名一一对应 |
| 目标框 | 10,936 个 |
| 目标消失帧（空标注文件） | 1,191 个 |
| 拍摄条件 | 夜间、雾天 |
| 拍摄设备 | 小米 12S；iPhone 15 Pro |
| 拍摄地点 | 中国江苏苏州，苏州大学未来校区内四个不同区域 |
| 场景类型 | 空旷草地；操场；公园；楼宇间（每个区域对应一类场景） |
| 拍摄日期与时段 | 2025 年 4 月和 2026 年 4 月；23:00 至次日 01:00（北京时间，UTC+8） |
| 无人机型号 | YH F450；DJI Mavic 4 |
| 标注流程 | 两名标注员分别标注不同视频，并互相复核对方的标注 |
| 下载平台 | Hugging Face Hub；百度网盘 |
| 许可协议 | CC BY 4.0（[LICENSE](LICENSE)） |
| 标注格式 | YOLO 风格的归一化目标框：`class_id x_center y_center width height` |
| 数据集类型 | 单类别、单目标 |
| 类别编号 | `0` = **无人机**（唯一类别） |
| 训练 / 测试划分 | 14 / 11 段视频，见[官方划分](#官方划分) |

**标注说明：**全部 10,936 个目标框的类别编号均为 `0`，当前数据集不含类别 `1` 标注。每条视频序列对应一个无人机目标，每个非空标注文件恰有一个目标框。空标注文件表示**目标消失帧**，即目标离开视野或不可见；25 段视频中有 20 段含有这类帧，共 108 个消失区间。单类别训练时无需进行类别编号映射。

**跟踪说明：**标注提供逐帧目标框，不含目标轨迹 ID；目标消失由空标注文件表示，没有单独的可见性字段。跟踪器在序列中第一个有目标框的帧上初始化。配套论文在 11 段测试视频以及 DDT、MAV-VID 和 DUT Anti-UAV 的测试序列上评测跟踪器，采用一次通过评测（OPE），指标为成功率 AUC、20 像素精度和归一化精度，并单独评测目标消失后的恢复；完整协议见论文。

- [目录结构与标注格式](docs/ANNOTATIONS.md)
- [逐序列统计](docs/STATISTICS.md)
- [机器可读统计（CSV）](metadata/sequences.csv)

## 使用条款

NF-UAVTrack 采用 **知识共享 署名 4.0 国际许可协议（CC BY 4.0）** 发布。您可以复制、分发、改编本数据集，包括用于商业目的，但须注明出处、附上许可协议链接，并说明是否作了修改。完整条款见 [LICENSE](LICENSE)。

建议的署名格式：

> NF-UAVTrack（2025-2026），作者 caizhexuan（蔡喆炫）。
> https://github.com/windowFlowers/NF-UAVTrack —— 依据 CC BY 4.0 授权。

## 引用

引用本数据集时，请注明作者 **caizhexuan（蔡喆炫）** 并提供本仓库链接。以下为数据集仓库的引用信息；配套论文发表后将补充论文引用。

```bibtex
@misc{caizhexuan_nf_uavtrack,
  author       = {Cai, Zhexuan},
  title        = {NF-UAVTrack},
  howpublished = {Dataset repository},
  url          = {https://github.com/windowFlowers/NF-UAVTrack}
}
```

机器可读的引用信息见 [CITATION.cff](CITATION.cff)。

## 问题与反馈

如遇下载问题、标注疑问或希望反馈数据质量问题，请[提交 Issue](https://github.com/windowFlowers/NF-UAVTrack/issues/new/choose)，并注明相关文件或视频序列名称及问题描述。