# EyeTrackUAV2 数据适配与运行说明

## 数据边界

本项目把官方 EyeTrackUAV2 眼动数据与刺激视频视为两个独立来源。官方 FTP 提供眼动原件和派生标注，但没有提供43段刺激视频。刺激来自 UAV123、DTB70 和 VIRAT；只有与实验使用内容、帧区间、分辨率和时长一致的视频才能加入实验清单。

编码和重建算法没有为 EyeTrackUAV2 增加特殊分支。新增适配层只生成现有 ROI 搜索所需的标准 manifest 和 CSV 注视接口。

## 官方眼动数据下载

完整可续传下载命令：

```bash
cd /home/xiaohan/BiSTAR-Benchmark
/home/xiaohan/anaconda3/envs/bistar/bin/python \
  scripts/download_eyetrackuav2.py \
  --output datasets/raw/EyeTrackUAV2
```

下载器先递归读取官方 FTP 清单，随后按远端大小跳过已经完成的文件、从 `.part` 文件续传，并输出：

```text
datasets/raw/EyeTrackUAV2/
├── download_inventory.csv
├── download_manifest.csv
├── download_metadata.json
└── RAW_DATA/...
```

本项目转换器直接使用：

```text
RAW_DATA/RAW_DATA_OrigRes_DomEye/
```

这一路径中的 CSV 已处于原始刺激视频坐标系。不要再次按1920×1080显示器坐标扣除偏移或归一化。ASC/EDF 是采集原件，可用于审计或重新处理，但当前适配器不解析它们。

若通过 wget 镜像，使用以下命令把每个 `.listing` 中的官方文件大小与本地文件逐项核对，并保存 SHA-256：

```bash
python scripts/audit_eyetrackuav2.py --condition TASK --checksums
```

审计结果写入 `datasets/metadata/eyetrackuav2/raw_data_audit.json` 和同名 CSV；只有所有已列出的TASK文件大小一致时，JSON中的 `complete` 才为 `true`。

## 刺激视频配对

生成视频映射模板：

```bash
python scripts/prepare_eyetrackuav2.py --template-only
```

模板字段：

| 字段 | 说明 |
| --- | --- |
| `video_id` | 从官方眼动文件名解析出的序列名，不应修改 |
| `video_path` | 项目内或绝对刺激视频路径 |
| `split` | `val` 用于配置标定，`test` 用于冻结后的留出评价 |
| `source_group` | 原始母视频/场景标识；同一母视频裁剪必须相同 |
| `source_start_frame` | 在源视频或图像序列中的零基起始帧，默认0 |
| `source_end_frame_exclusive` | 独占结束帧；留空表示源内容末尾 |
| `notes` | 获取来源、裁剪范围和人工核验记录 |

映射文件位于 `datasets/metadata/eyetrackuav2/video_map.csv`。不允许用同类别或同名近似视频替代缺失刺激。VIRAT 原片需要先按数据发布使用的准确帧区间裁剪，再填写裁剪后视频路径。
适配器会拒绝同一 `source_group` 同时出现在 `val` 和 `test`，以防同一母视频的不同裁剪泄漏到标定与留出两侧。

`video_path`既可指向视频文件，也可指向按自然数字顺序排列的 JPG/PNG 图像目录。图像序列及设置了帧区间的视频会被无损转换为 `datasets/processed/EyeTrackUAV2/stimuli/*.mkv`，manifest引用该确定性刺激文件。

## 转换协议

正式转换：

```bash
python scripts/prepare_eyetrackuav2.py \
  --conditions TASK \
  --source-fps 30 \
  --target-fps 15 \
  --target-width 224 \
  --target-height 224 \
  --roi-radius-px 64 \
  --gaze-delay-ms 0 \
  --max-hold-ms 100 \
  --strict
```

转换器执行以下操作：

1. 解析无表头的13列 `OrigRes_DomEye` CSV，并验证帧号、数值和边界。
2. 每个观察者导出独立复合 ID，例如 `DTB70_Basketball__TASK__S10`。
3. 评价 ROI 使用当前源帧内有效支配眼样本的中位数。
4. 选择器注视严格使用输出帧时刻减去反馈延迟之前的最后一个有效样本。
5. 追踪短时失效时最多保持 `max_hold_ms`，之后选择器回退中心；不使用未来注视。
6. `roi_valid=0` 的帧继续编码并计入真实码率，但 ROI 指标记为缺失并从聚合中排除。
7. 原始帧号、时间、观察者、条件、文件哈希、有效率和视频时长误差全部写入元数据。

同一刺激视频的多个观察者共享一次预处理和同一无损分段文件，避免在数据接入阶段重复解码并复制整段视频。观察者注视、BiSTAR选择结果和逐观察者质量指标仍保持独立。

默认保持当前工程的224×224拉伸预处理，以保证四种方法读取完全相同的像素和现有 VideoMAE 权重可直接运行。正式论文若采用 letterbox，需要为所有方法同时切换并增加内容掩码，不能只改变 BiSTAR 输入。

## 输出与验收

转换输出：

```text
datasets/manifests/eyetrackuav2.csv
datasets/annotations/eyetrackuav2_task.csv
datasets/metadata/eyetrackuav2/
├── acquisition_inventory.csv
├── observer_inventory.csv
├── preparation_metadata.json
└── video_map.csv
```

`--strict` 会拒绝缺失视频、无法探测的视频和眼动/视频时长不一致的配对。开始大规模搜索前，至少人工检查每个源数据集若干序列的开头、中间和末尾叠加位置。

运行配置检查和正式实验：

```bash
python scripts/run_roi_search.py \
  --config configs/experiments/eyetrackuav2_roi_search.yaml --plan

python scripts/run_roi_search.py \
  --config configs/experiments/eyetrackuav2_roi_search.yaml \
  --device cuda:0 --threads 4
```

配置会在真实应用层码率相近的条件下比较 H.264、H.265、BiSTAR 和 BiSTAR-Atlas。校准与留出必须按原始视频/场景拆分；同一视频的不同观察者不能跨两个 split。

## 试运行与正式运行

首次接入可用每视频一名观察者检查全链路：

```bash
python scripts/prepare_eyetrackuav2.py \
  --observers-per-video 1 --strict
```

正式运行时删除该参数或设为0。TASK 与 FV 是不同受试者群体，应分别转换和报告；不能把两者当成同一观察者的配对条件。反馈时延实验应分别使用0、50、100和200毫秒生成不同注释文件与输出目录，防止结果互相覆盖。
