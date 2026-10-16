# 视觉关键区域配置搜索

这个实验搜索 H.264、H.265、BiSTAR 和 BiSTAR Atlas 的配置，在相同实际码率容限下比较独立标注的逐帧任务 ROI。它选择搜索集合中的最佳配置，不保证 BiSTAR 一定超过传统编码器。若没有优于基线的配置，报告会明确保留负结果。

## 1. 准备逐帧标注

所有帧号和坐标都对应统一预处理后的视频。不要直接使用原始视频的帧号；原视频经过帧率转换后，帧号可能已经不同。

为 `test.mp4` 的现有数据清单生成模板和浏览器标注页面：

```bash
conda activate bistar
cd /home/xiaohan/BiSTAR-Benchmark
python scripts/prepare_task_roi.py \
  --dataset configs/datasets/test_mp4.yaml \
  --split test \
  --output datasets/annotations/task_roi_template.csv
```

在浏览器打开 `datasets/annotations/task_roi_template.html`。拖动鼠标画任务 ROI，选择首尾帧和目标运动关键帧后可执行插值，再逐帧检查并修正。页面只在本地运行。导出 CSV 后保存为 `datasets/annotations/task_roi.csv`。

已生成的模板不能直接用于正式搜索，空坐标会被校验拒绝。每次生成使用新的输出路径，避免覆盖已有标注。标注页面当前每帧支持一个框；CSV读取器支持同一帧多行 ROI，评测时取并集。

CSV 示例：

```csv
sequence_id,frame_id,roi_id,x0,y0,x1,y1,gaze_x,gaze_y,provenance
test_mp4,0,target,0.4,0.3,0.6,0.5,,,manual_boxes
test_mp4,1,target,0.41,0.3,0.61,0.5,,,manual_boxes
```

默认坐标为 `[0,1]` 归一化坐标。也可设置 `roi.coordinate_space: preprocessed_pixels` 使用预处理图像像素坐标。矩形采用左上角包含、右下角排除的范围。框必须非空、位于画面内，所有被评测帧都必须有标注。

`provenance` 必须记录实际来源，例如 `manual_boxes`、`manual_interpolated`、`eye_tracker` 或具体标注系统名称。插值框需要人工检查。没有真实眼动时，手工点击的注视点不能在论文中作为实测眼动。

若使用眼动 ROI，可通过眼动设备导出逐帧框与 `gaze_x/gaze_y`。设置 `roi.use_gaze_for_selection: true` 后，选择器只能读取当前 GOP 首帧的注视点；所有评测帧必须具备注视坐标。默认 `false` 时，任务 ROI 仅用于质量评价，选择器仍使用现有中心注视回退，不会读取任务目标标注来提高评分。

## 2. 搜索与留出验证

多视频正式配置：

```bash
python scripts/run_roi_search.py --config configs/experiments/roi_config_search.yaml --plan
python scripts/run_roi_search.py --config configs/experiments/roi_config_search.yaml
```

数据清单必须同时包含 `val` 和 `test`，视频和 sequence ID 不重叠。`val` 搜索配置，`test` 验证已经冻结的配置。不能因为测试结果更好而重新选择配置。

现有单视频可先做探索性搜索：

```bash
python scripts/run_roi_search.py \
  --config configs/experiments/roi_config_search_single_video.yaml --plan
python scripts/run_roi_search.py \
  --config configs/experiments/roi_config_search_single_video.yaml
```

该配置使用视频前 60% 搜索，间隔 16 帧后用剩余片段验证。它不能证明跨视频、跨场景泛化。正式论文应使用独立多视频划分。

默认模型使用模型 YAML 中的 CUDA 设备。无 GPU 时可显式使用 CPU：

```bash
python scripts/run_roi_search.py \
  --config configs/experiments/roi_config_search_single_video.yaml \
  --device cpu --threads 4
```

设备覆盖只作用于本次运行，并保存到元数据，不修改公共模型配置。

## 3. 配置空间

`methods.<method>.grid` 的键相对于该方法配置的 `codec` 段。所有列表的笛卡尔积构成候选集合，每个候选具有稳定哈希 ID 和独立 YAML 快照。

```yaml
search:
  target_bitrates_kbps: [50, 75, 100, 150, 200, 300]
  tolerance_percent: 1.0
  pair_tolerance_percent: 2.0
  max_match_iterations: 12
  objective: roi_ssim
  resume: true
methods:
  bistar:
    enabled: true
    config: configs/codecs/bistar.yaml
    grid:
      selector.type: [gaze_only, gaze_edge, bistar_full]
      selector.patch_budget_mode: [false]
      payload_encoding.quality: [10, 30, 50, 70, 90]
      reconstruction.type: [videomae]
  bistar_atlas:
    enabled: true
    config: configs/codecs/bistar_atlas.yaml
    grid:
      selector.type: [gaze_only, gaze_edge, bistar_full]
      selector.patch_budget_mode: [true, false]
      selector.num_visible_patches: [16, 36, 64, 100, 144]
      encoder.preset: [ultrafast, fast]
      encoder.pixel_format: [yuv420p]
```

H.264/H.265 同样搜索 preset、GOP，默认关闭 B-frame、采用 zerolatency 和匹配的请求码率/VBV约束。Atlas 示例显式使用同样的 `yuv420p`。BiSTAR 搜索 JPEG 质量和选择器，自动搜索每 GOP 的可见 patch 数；Atlas 联合搜索 patch 数、固定/自适应和编码 preset。重建还可选择 `previous_frame` 或 `interpolation`。

两种稀疏方法也可设置 `grid.gop_size: [4, 8, 16]` 搜索 mask 更新周期。使用 VideoMAE 时 GOP 不超过16帧，短 GOP在模型输入端补齐；这仅验证当前模型对短窗口的适用性，不能假设它具有与16帧输入相同的重建能力。

搜索入口对两种稀疏方法统一使用 GOP 首帧和上一 GOP 末帧选择 mask；GOP 内固定。它与旧 RD BiSTAR 从 clip 末帧选择的离线策略不同，必须作为独立实验协议记录，不应混合两套结果。

正式默认有 87 个配置，每个目标码率都评测，基础搜索为每个标定序列 522 次；单视频默认有 58 个配置，基础搜索 348 次。未匹配候选跳过模型重建。可通过 `--plan` 查看当前数量，或减少 grid 列表进行小规模验证。

## 4. 指标与最优条件

默认按搜索集的平均 `roi_ssim` 选择一个配置，每种方法、每个目标码率各选一次。只有在所有标定序列上都匹配成功的候选才可入选。最佳配置是给定候选空间中的最优，不代表所有可能配置的全局最优。

可选择的 `search.objective`：`roi_ssim`、`roi_psnr`、`critical_patch_retention`（越大越好）或 `roi_normalized_l1`（越小越好）。

`critical_patch_retention` 定义为：在独立标注 ROI 内，按 16×16 网格统计区域质量；同时满足 `metrics.patch_psnr_threshold` 和 `metrics.patch_ssim_threshold` 的 patch，按其 ROI 像素面积加权所得保留比例。默认阈值 30 dB、0.9。它是明确阈值下的视觉保留指标，不是香农互信息，也不能直接替代任务成功率。

ROI SSIM使用全帧 SSIM map 的 ROI 像素平均，ROI PSNR由逐帧 ROI MSE 的平均得到，指标在保存 MP4 前计算。`roi_visible_coverage` 记录 ROI 中实际可见像素比例。

默认单点相对目标实际码率误差不得超过 1%；两个方法的实际码率差不得超过目标的 2%，才允许生成优劣判断。结果保存实际数值。`bitrate_unreachable`、失败或缺少基线的点都不能被当作胜出。

稀疏方法计入完整协议字节和载荷；末 GOP只传输实际帧，模型输入不足16帧时在接收端补齐，不按比例减少已经发送的字节。传统方法计入实际 elementary bitstream。所有方法都尚未统计 UDP/RTP/TCP/IP 开销，正式网络实验需要进一步统计抓包字节。

新搜索入口的稀疏延迟使用每帧 GOP 等待时间加完整 GOP 处理时间。它仍是离线估计，不包含网络/显示延迟。传统编码器只报告完整序列处理时间，没有可直接对比的实时逐帧延迟；报告不会用它宣称低时延优势。

## 5. 输出与复现

```text
outputs/roi_config_search/
├── config_resolved.yaml        # 主实验配置
├── config_snapshot.yaml        # 数据集、模型、所有候选、源文件和代码SHA256
├── code_snapshot/              # 本次运行Python代码副本
├── working_tree.patch
├── environment.json
├── git_state.json
├── annotations.csv
├── manifest.csv
├── source_metadata.json        # 视频SHA256、预处理、帧区间、标注来源
├── candidate_catalog.csv
├── calibration_results.csv
├── best_configs.csv
├── best_configs/<target>/<method>.yaml
├── validation_results.csv
├── validation_patch_results.csv
├── comparisons.csv             # 差值、实际码率差、可比较性和是否胜出
├── report.md
├── summary.json
├── trials/<phase>/<sequence>/<candidate>/<target>/
│   ├── result.json
│   ├── frame_results.csv
│   ├── patch_results.csv
│   ├── gop_results.csv
│   ├── rate_search.csv
│   ├── reconstructed.mp4
│   └── final/wire/              # 稀疏方法实际序列化包
└── plots/
    ├── roi_ssim_rate_comparison.png/.pdf
    ├── roi_psnr_rate_comparison.png/.pdf
    ├── critical_patch_retention_rate_comparison.png/.pdf
    ├── <method>_configuration_heatmap.png
    ├── <method>_vs_<baseline>_gain.png
    └── spatial/                # 留出集空间质量、选择频率、对基线的差值图
```

配置热力图表达超参数与质量的关系；散点图显示全部匹配配置；留出曲线只显示冻结配置；增益图保留负值和未匹配状态。多视频增益柱图的误差线按独立视频计算标准误差，单视频不会产生伪造的统计显著性。

默认可断点续跑。已完成 trial 的 JSON 会复用，失败 trial 会重试。配置、标注、清单、原视频、模型配置、checkpoint 或代码 SHA256 发生变化时，必须更换输出目录，避免混用实验。需要保留不同次运行时使用不同 `experiment.output_dir`。

## 6. 流程示例与正式证据的区别

`scripts/prepare_task_roi.py --example-center` 只生成 `example_center_proxy` 标注。正式搜索默认拒绝这类标注。只有显式 `roi.allow_proxy_annotations: true` 的流程检查才能使用，输出 `summary.json` 会记录 `proxy_ROI_only: true`。这种结果不能作为真实任务关键区域或操控性能的论文证据。

也可以填充已经生成的空模板：

```bash
python scripts/fill_pseudo_roi.py --input datasets/annotations/task_roi_template.csv
python scripts/run_roi_search.py --config configs/experiments/roi_config_search_pseudo.yaml
```

默认使用中心50%宽高的矩形，并把伪注视点设为框中心。`--mode moving` 可生成平滑移动的框用于空间变化测试。重复填充已有标注需要显式 `--overwrite`，每次写入前都会保留备份和 `.pseudo.json` 来源记录。伪标注测试配置只搜索14个候选、100/200/300 kbps，使用独立输出目录；注视点是合成数据，不是实际眼动测量。
