# BiSTAR Benchmark

BiSTAR Benchmark 是一个可复现的视频传输与重建评测框架，用于公平比较：

- BiSTAR 稀疏图像块传输与 VideoMAE 重建；
- FFmpeg `libx264`（H.264）基线；
- FFmpeg `libx265`（H.265）基线。

项目支持离线率失真评测、目标码率匹配、BiSTAR 组件消融、真实传输字节统计、延迟测量，以及基于 GStreamer 和 Linux `tc/netem` 的实时扩展。正式实验由 YAML 配置和 CSV 数据清单驱动，并保存解析后的配置、环境、Git 状态、逐帧结果和汇总报告。

## 功能

| 功能 | 实现 |
| --- | --- |
| H.264/H.265 | FFmpeg `libx264`、`libx265`，按实际码流文件计费 |
| 码率匹配 | 迭代调整请求码率，默认误差容限 ±5% |
| BiSTAR 选择器 | random、gaze、temporal SSIM、edge、structural 及组合策略 |
| 稀疏传输 | patch 编码、序列化、反序列化、载荷与元数据分别计费 |
| 重建 | previous-frame、interpolation、VideoMAE masked reconstruction |
| 质量指标 | PSNR、SSIM、MS-SSIM、LPIPS、ROI 和时序残差指标 |
| 性能指标 | 编解码时间、重建延迟、FPS、VideoMAE 推理时间 |
| 报告 | CSV、JSON、率失真曲线、运行环境和 Git 元数据 |
| 实时扩展 | 统一实时接口、GStreamer RTP、`tc/netem` 网络模拟 |

## 项目结构

```text
BiSTAR-Benchmark/
├── configs/               # 数据集、编码器、模型、指标和实验配置
├── checkpoints/           # VideoMAE 模型权重
├── datasets/
│   ├── raw/               # 原始视频
│   └── manifests/         # 实验输入清单
├── docs/                  # 架构和验收记录
├── outputs/               # 实验结果（默认不纳入 Git）
├── realtime/
│   ├── gstreamer/         # H.264/H.265 RTP 收发脚本
│   └── network/           # tc/netem 网络整形脚本
├── scripts/               # 用户命令入口
├── src/                   # 核心实现
├── tests/                 # 单元、集成和回归测试
└── third_party/VideoMAE/  # 上游 VideoMAE 源码
```

## 系统和环境

推荐 Ubuntu 22.04。项目需要 Conda、Python 3.10、FFmpeg/FFprobe、`libx264`、`libx265`、PyTorch 2.4.0、torchvision 0.19.0 和 VideoMAE。实时实验还需要 GStreamer、`tc` 及相应插件。

当前已验收环境位于：

```text
/home/xiaohan/anaconda3/envs/bistar
```

当前主机未检测到可用 NVIDIA 设备，验收使用 CPU 推理。若把 `configs/models/videomae.yaml` 中的 `runtime.device` 改为 `cuda:0`，环境检查将要求可用的兼容 GPU 和驱动。

## 安装

在项目根目录执行：

```bash
cd /home/xiaohan/BiSTAR-Benchmark
./scripts/bootstrap.sh
conda activate bistar
```

`bootstrap.sh` 根据 `environment.yml` 创建 `/home/xiaohan/anaconda3/envs/bistar`。环境已经存在时，可按最新依赖文件同步：

```bash
conda env update --prefix /home/xiaohan/anaconda3/envs/bistar \
  --file environment.yml --prune
```

初始化 VideoMAE 子模块：

```bash
git submodule update --init --recursive
git -C third_party/VideoMAE rev-parse HEAD
```

本项目验收使用的 VideoMAE 上游提交为：

```text
14ef8d856287c94ef1f985fe30f958eb4ec2c55d
```

## VideoMAE 权重

将与 `pretrain_videomae_base_patch16_224`、16 帧输入和 tubelet size 2 对应的预训练权重放到：

```text
checkpoints/videomae_base.pth
```

验收权重 SHA-256：

```text
decba3e0264ae77030eadf097403e13051da35e19297d59bb0abf721cf4fa0c2
```

验证文件：

```bash
sha256sum checkpoints/videomae_base.pth
```

模型路径、结构和预期哈希定义在 `configs/models/videomae.yaml`。如使用其他权重，必须同时修改架构字段、路径和 SHA-256，避免错误模型被静默加载。

VideoMAE 重建遵循以下约束：隐藏 patch 在推理输入中置零；选择器的 `visible=True` 掩码转换为 VideoMAE 的 `mask=True` 语义；token 按 `(T, H, W)` 顺序展开；预测 patch 只用已传输像素的统计量反归一化，不读取隐藏区域真值。

## 环境验收

```bash
conda activate bistar
PYTHONNOUSERSITE=1 python scripts/check_environment.py
```

该命令检查 Python、FFmpeg 编码器、Python 依赖、VideoMAE 源码、权重哈希和模型加载。成功时顶层字段为 `"ok": true`，返回码为 0。报告位于：

```text
outputs/environment/check_environment.json
outputs/environment/environment.json
outputs/environment/ffmpeg_buildconf.txt
```

## 快速验证

冒烟测试会生成确定性的 224×224、15 FPS、30 帧视频，并运行 H.264、H.265、BiSTAR 消融和真实 VideoMAE 重建：

```bash
PYTHONNOUSERSITE=1 python scripts/run_smoke_test.py \
  --config configs/experiments/smoke_test.yaml
```

主要输出：

```text
outputs/smoke_test/summary.json
outputs/smoke_test/metrics.csv
outputs/smoke_test/plots/
outputs/smoke_test/bistar/summary.json
outputs/smoke_test/bistar_videomae/summary.json
outputs/smoke_test/bistar_videomae/reconstructed.mp4
```

冒烟测试会重写 `datasets/manifests/bistar_eval.csv`，使其指向自动生成的 `datasets/raw/synthetic.mkv`。运行正式实验前，应重新生成或恢复正式 manifest。

## 准备正式数据

将视频直接放在一个目录中，例如：

```text
datasets/raw/
├── sequence_001.mp4
├── sequence_002.mkv
└── sequence_003.mov
```

构建输入清单：

```bash
python scripts/build_manifest.py \
  --input datasets/raw \
  --output datasets/manifests/bistar_eval.csv \
  --split test
```

构建器扫描目录第一层，支持 `.mp4`、`.mov`、`.mkv`、`.avi` 和 `.webm`，不会递归扫描。CSV 格式：

```csv
sequence_id,video_path,fps,width,height,num_frames,split
sequence_001,datasets/raw/sequence_001.mp4,30,1920,1080,300,test
```

| 字段 | 含义 |
| --- | --- |
| `sequence_id` | 序列唯一标识 |
| `video_path` | 视频路径；相对路径按项目根目录解析 |
| `fps` | FFprobe 探测的源帧率 |
| `width`、`height` | 源视频尺寸 |
| `num_frames` | 视频帧数 |
| `split` | `train`、`val` 或 `test` 等数据划分 |

实验只处理 manifest 中与 `dataset.split` 相同的行。manifest 是实验输入的唯一事实来源，确保三个方法使用相同视频集合。

`configs/datasets/bistar_eval.yaml` 默认把输入统一到 224×224、15 FPS、RGB24。H.264、H.265 和 BiSTAR 使用同一预处理结果，以保证分辨率、帧率、帧区间和帧对齐一致。

## 配置实验

所有命令通过 `--config` 接受 YAML 文件。建议复制现有配置后修改：

```bash
cp configs/experiments/rd_sweep.yaml configs/experiments/rd_formal.yaml
```

实验配置至少包含名称、随机种子、输出目录、数据集配置和 split。所有实验配置的
`experiment` 段均提供 `save_videomae_videos` 开关：

```yaml
experiment:
  name: example
  seed: 0
  output_dir: outputs/example
  save_videomae_videos: true
```

设为 `true` 时保存 VideoMAE 重建视频，设为 `false` 时仍执行重建和指标计算，但跳过
VideoMAE MP4 写出。H.264/H.265 对比视频仍会保存。重复使用同一个 `output_dir` 会更新
其中结果；需要保留多次运行时，请使用不同输出目录。

## 率失真实验

```bash
python scripts/run_rd.py --config configs/experiments/rd_sweep.yaml
```

默认目标码率为 50、75、100、150、200 和 300 kbps。H.264/H.265 迭代调整编码器请求码率，并按实际 bitstream 大小计算码率；BiSTAR 统计序列化包的完整字节，包括压缩 patch 和协议元数据。

```yaml
rd:
  target_bitrates_kbps: [50, 75, 100, 150, 200, 300]
  bitrate_match_tolerance_percent: 5.0
  max_match_iterations: 8
methods:
  h264: {enabled: true, config: configs/codecs/h264.yaml}
  h265: {enabled: true, config: configs/codecs/h265.yaml}
  bistar: {enabled: true, config: configs/codecs/bistar.yaml}
bistar:
  patch_budget_mode: false
  visible_patch_count: 40
quality_metrics: [psnr, ssim, ms_ssim, lpips]
```

VideoMAE 可见 patch 使用布尔开关。按目标码率自动搜索：

```yaml
bistar:
  config: configs/codecs/bistar.yaml
  patch_budget_mode: false
  visible_patch_count: 40  # 此模式下忽略
```

固定为 40 个空间 patch：

```yaml
bistar:
  config: configs/codecs/bistar.yaml
  patch_budget_mode: true
  visible_patch_count: 40
```

224×224 输入和 16×16 patch 对应 196 个空间 patch，因此
`visible_patch_count` 的有效范围为 1～196。固定模式仍保留
`target_bitrates_kbps` 作为结果分组和码率误差参照；若固定 patch 产生的实际码率超出容限，
结果会标记为 `bitrate_unreachable`。

若目标在最大迭代次数或 patch 预算范围内不可达，该点仍会保存，但 `status` 为 `bitrate_unreachable`。分析时必须检查此字段。

```text
outputs/rd_sweep/
├── frame_results.csv
├── sequence_results.csv
├── rd_points.csv
├── metrics.csv
├── plots/bitrate_vs_psnr.png
├── plots/bitrate_vs_ssim.png
├── plots/bitrate_vs_lpips.png
├── reconstructions/<sequence_id>/
│   ├── h264_<target>kbps.mp4
│   ├── h265_<target>kbps.mp4
│   └── bistar_<target>kbps.mp4
├── tables/rate_distortion.csv
├── tables/runtime.csv
└── summary.json
```

## 等码率实验

```bash
python scripts/run_matched_bitrate.py \
  --config configs/experiments/matched_bitrate.yaml
```

默认在 100 kbps 下比较 H.264、H.265 和 `bistar_videomae`，容限 ±5%：

```yaml
matched_bitrate:
  target_kbps: [100]
  tolerance_percent: 5.0
  max_iterations: 8
methods: [h264, h265, bistar_videomae]
bistar:
  config: configs/codecs/bistar.yaml
  patch_budget_mode: false
  visible_patch_count: 40
```

等码率实验使用与 RD 相同的 `bistar.patch_budget_mode`。将其改为 `true`
即可让整个序列的每个16帧 clip 使用相同的可见 patch 数；设为 `false`
则根据目标码率自动搜索。

结果位于 `outputs/matched_bitrate/metrics.csv` 和 `summary.json`：

- `requested_kbps`：最终传给传统编码器的请求值；
- `actual_kbps`：按实际字节数计算的码率；
- `error_percent`：相对目标码率的有符号误差；
- `iterations`：传统编码器的迭代次数；对 BiSTAR 表示可见 patch 数；
- `status`：是否达到配置容限。
- `patch_budget_mode`：布尔值，`true` 表示固定 patch，`false` 表示自动码率匹配；
- `patch_budget_strategy`：结果中记录的可读策略名，值为 `fixed` 或 `bitrate_match`；
- `visible_patch_count`：所有 clip 的平均可见 patch 数；
- `visible_patch_counts`：每个 clip 实际使用的 patch 数列表。

三个方法的完整重建视频保存到：

```text
outputs/matched_bitrate/reconstructions/<sequence_id>/h264_<target>kbps.mp4
outputs/matched_bitrate/reconstructions/<sequence_id>/h265_<target>kbps.mp4
outputs/matched_bitrate/reconstructions/<sequence_id>/bistar_videomae_<target>kbps.mp4
```

CSV 中的 `reconstruction_video` 和 `reconstruction_video_bytes` 分别记录路径和文件大小。

`bistar_videomae` 按 16 帧 clip 处理整个序列；不足 16 帧的尾段会在模型输入端补齐，
写出视频时只保留有效帧。因此三个方法的比较视频具有相同帧数、分辨率和帧率。

## BiSTAR 消融实验

```bash
python scripts/run_ablation.py --config configs/experiments/ablation.yaml
```

默认比较 8 种选择器和 20、40、60 三档可见 patch 数，共 24 个组合：

```yaml
experiment:
  name: ablation
  seed: 0
  output_dir: outputs/ablation
  save_videomae_videos: true
ablation:
  budget_mode: visible_patch_count
  visible_patch_counts: [20, 40, 60]
  variants:
    - random
    - gaze_only
    - temporal_only
    - edge_only
    - structural_only
    - gaze_temporal
    - gaze_edge
    - bistar_full
bistar:
  config: configs/codecs/bistar.yaml
```

三个实验统一从 `configs/codecs/bistar.yaml` 读取 `patch_size`、
`payload_encoding` 和 `reconstruction`。例如：

```yaml
codec:
  patch_size: 16
  payload_encoding: {mode: jpeg, quality: 50}
  reconstruction: {type: previous_frame}
```

修改该文件后，RD、等码率和消融实验都会使用新的载荷编码。结果记录总码率、
payload/metadata 码率、SSIM、LPIPS、中心 ROI 指标、时序误差、重建延迟和 FPS：

```text
outputs/ablation/frame_results.csv
outputs/ablation/sequence_results.csv
outputs/ablation/tables/ablation.csv
outputs/ablation/summary.json
outputs/ablation/videomae_reconstructions/<sequence_id>/<selector>_<patch数量>patches.mp4
```

启用 `save_videomae_videos` 后，每个选择器和 patch 预算组合都会额外执行 VideoMAE
重建并保存完整视频。`sequence_results.csv` 中的
`videomae_reconstruction_video` 和 `videomae_reconstruction_video_bytes`
记录视频路径和大小。

## 编码器参数

H.264/H.265 配置位于 `configs/codecs/h264.yaml` 和 `h265.yaml`：

```yaml
encoder:
  codec_name: libx264       # H.265 使用 libx265
  preset: medium
  mode: bitrate
  bitrate_kbps: 100
  maxrate_kbps: null
  bufsize_kbps: null
  crf: null
  gop_size: 30
  b_frames: 0
  tune: null
```

公平对比时应保持帧率、分辨率、像素格式、GOP 和 B-frame 策略一致。默认关闭 B-frame，便于帧对齐和低延迟扩展。

## 实时模式

检查实时依赖：

```bash
python scripts/run_realtime.py --config configs/experiments/realtime.yaml
cat outputs/realtime/summary.json
```

该命令检查 `gst-launch-1.0` 和 `tc`，写出 `ready` 或 `dependency_missing`，不会自动修改主机网络规则。

本机 H.264 RTP 示例，先启动接收端：

```bash
./realtime/gstreamer/receiver_h264.sh 5000
```

再启动发送端；本地文件使用绝对 `file://` URI：

```bash
./realtime/gstreamer/sender_h264.sh \
  file:///home/xiaohan/BiSTAR-Benchmark/datasets/raw/synthetic.mkv \
  127.0.0.1 5000
```

H.265 对应：

```bash
./realtime/gstreamer/receiver_h265.sh 5001
./realtime/gstreamer/sender_h265.sh \
  file:///home/xiaohan/BiSTAR-Benchmark/datasets/raw/synthetic.mkv \
  127.0.0.1 5001
```

网络整形通常需要 root 权限。以下命令将 `lo` 限制为 200 kbps，并加入 100 ms 延迟、20 ms 抖动和 1% 丢包：

```bash
sudo ./realtime/network/setup_netem.sh lo 200 100 20 1.0
```

实验结束后恢复：

```bash
sudo ./realtime/network/reset_netem.sh lo
```

不要在远程连接依赖的接口上直接试验未经确认的规则。`realtime.yaml` 的网络字段用于记录条件；当 `network.enabled: true` 时，Python runner 会拒绝自动应用规则，要求显式运行脚本。

## 输出与复现

每个 runner 会写入：

| 文件 | 内容 |
| --- | --- |
| `config_resolved.yaml` | 实际使用的完整配置 |
| `environment.json` | Python、平台、依赖、FFmpeg 和模型环境 |
| `git_state.json` | 项目和 VideoMAE 提交及工作区状态 |
| `stdout.log`、`stderr.log` | 运行日志文件 |
| `summary.json` | 完成状态与高层汇总 |
| `metrics.csv` | 主要指标 |

传统编码器中间码流和解码结果位于 `artifacts/`，统一预处理视频位于 `sources/`。码率计算公式：

```text
bitrate_kbps = total_bytes × 8 ÷ duration_seconds ÷ 1000
```

BiSTAR 的 `total_bytes` 同时包含 `payload_bytes` 和 `metadata_bytes`，不会只报告理论 patch 比例。
保存的 VideoMAE MP4 用于人工查看，不参与 PSNR、SSIM、MS-SSIM 或 LPIPS 计算；这些指标直接基于编码 MP4 之前的重建帧计算。

## 测试

```bash
PYTHONNOUSERSITE=1 python -m pytest -q
```

也可分别运行：

```bash
python -m pytest tests/unit
python -m pytest tests/integration
python -m pytest tests/regression
```

当前验收结果为 29 项测试全部通过。版本和验收记录见 `docs/validation.md`。

## 常见问题

### 权重哈希不匹配

运行 `sha256sum checkpoints/videomae_base.pth`。若有意使用其他权重，应更新模型配置中的结构和哈希，不能绕过检查。

### FFmpeg 找不到编码器

```bash
which ffmpeg
ffmpeg -hide_banner -encoders | grep -E 'libx264|libx265'
```

先确认已激活 `bistar` 环境。

### 正式实验仍使用 synthetic 数据

冒烟测试会覆盖 `datasets/manifests/bistar_eval.csv`。重新运行 manifest 构建命令，并确认 CSV 的 `video_path` 和 `split`。

### BiSTAR 显示 `bitrate_unreachable`

最小传输包也包含协议头、patch 索引和压缩载荷，离散 patch 数也限制匹配精度。应保留此状态，不能将该点解释为严格等码率结果。

### CPU 上 LPIPS 或 VideoMAE 较慢

这是完整神经网络推理的预期行为。兼容 GPU 环境可把模型配置的 `runtime.device` 改为 `cuda:0`，随后先运行环境检查。

### GStreamer 无法显示

```bash
gst-inspect-1.0 x264enc
gst-inspect-1.0 x265enc
gst-inspect-1.0 rtph264pay
gst-inspect-1.0 rtph265pay
gst-inspect-1.0 avdec_h264
gst-inspect-1.0 avdec_h265
```

无图形桌面的服务器无法通过 `autovideosink` 打开窗口，需要将接收 pipeline 的末端换成适合服务器的 sink。

## 推荐工作流

```bash
# 1. 激活并验收
conda activate bistar
PYTHONNOUSERSITE=1 python scripts/check_environment.py

# 2. 快速验证
PYTHONNOUSERSITE=1 python scripts/run_smoke_test.py \
  --config configs/experiments/smoke_test.yaml

# 3. 冒烟测试后重新生成正式数据清单
python scripts/build_manifest.py \
  --input datasets/raw \
  --output datasets/manifests/bistar_eval.csv \
  --split test

# 4. 正式实验
python scripts/run_rd.py --config configs/experiments/rd_sweep.yaml
python scripts/run_matched_bitrate.py --config configs/experiments/matched_bitrate.yaml
python scripts/run_ablation.py --config configs/experiments/ablation.yaml

# 5. 回归测试
PYTHONNOUSERSITE=1 python -m pytest -q
```

发布结果前检查每个输出目录中的 `config_resolved.yaml`、`git_state.json`、`environment.json` 和结果行的 `status`，确认使用了预期数据、提交、权重和码率匹配状态。

## 相关文档

- `BiSTAR_Benchmark_Infrastructure.md`：原始工程规格；
- `docs/architecture.md`：掩码转换、稀疏输入和无隐藏真值泄漏说明；
- `docs/validation.md`：已验收环境、版本和命令记录；
- `configs/experiments/README.md`：实验模式简述。
