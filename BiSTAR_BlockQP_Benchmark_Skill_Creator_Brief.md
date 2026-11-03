# 给 skill-creator 的创建指令：BiSTAR 块级 QP Benchmark

请根据本文创建一个可指导 Codex 完成代码实现的 skill，建议名称 `bistar-block-qp-benchmark`。本文是 skill 创建需求书，不是已实现的程序或已安装的 skill。目标仓库为 https://github.com/HanXiao485/BiSTAR 。执行时以本地实际 checkout、分支和 HEAD 为准。

## 1. 目标和范围

在现有 BiSTAR 评测框架中新增完整帧 H.264 块级 QP 控制 benchmark：每帧根据可替换策略生成宏块量化偏移，分别控制 ROI 与非 ROI 区域的压缩强度。复用现有数据预处理、码率匹配、质量评价、结果输出和配置机制。为后续用户算法提供稳定接口，后续算法可替换 QP 策略而不改实验运行器或底层编码器。

第一版必须实现可运行的固定 ROI 策略、均匀偏移策略、外部 QP 图读取策略及可替换策略接口。PPO 训练不是第一版要求。没有实现论文的状态、奖励、辅助损失和训练流程时，方法必须命名为 `h264_block_qp` 等，不得标为 RL-RC-DoT 复现。

这里的选择性压缩是在同一完整帧码流中应用不同块的量化强度。默认不裁剪、不拼接 atlas、不删除非 ROI、不用模糊背景冒充 QP 控制，也不逐 patch 独立创建视频流。额外背景降采样或滤波若扩展，必须作为独立消融开关，默认关闭。

## 2. 创建 skill 的要求

skill 的 description 明确触发条件：为 BiSTAR 增加块级 QP benchmark、ROI/背景质量控制、QP 策略扩展、编码器适配与兼容性验证。SKILL.md 保持精简，按“核查仓库→选择后端→实现接口→接入评测→验证→交付”组织；详细接口、配置和验收表放入 references，并从 SKILL.md 链接。按执行环境实际 skill-creator 的规范生成、验证和保存 skill，不硬编码本需求书所在机器的路径。

该 skill 被调用后必须完成程序实现和必要验证，不能只输出方案。不得自动启动大规模正式实验或 PPO 训练。保存用户未提交修改；不自动推送主分支，不替换用户已有输出。

## 3. 已核对的仓库基础与重新核查

以下来自本次对 main 分支源码的只读检查，后续可能变化，必须重新核对：

| 路径 | 已观察到的职责 | 接入要求 |
|---|---|---|
| `src/codecs/base.py` | 编解码适配器契约 | 保持旧签名可用 |
| `src/codecs/h264.py` | 继承 FFmpegAdapter，指定 libx264 | 保留普通 H.264 基线 |
| `src/codecs/ffmpeg_common.py` | subprocess 调用 FFmpeg；bitrate/crf 模式；码流字节和解码统计 | 当前已读实现没有逐帧块级 QP 图输入 |
| `src/common/config.py` | 配置读取与验证 | 新字段应默认兼容旧 YAML |
| `src/common/types.py` | 帧结果等共享数据类型 | 新字段可选，保持原字段语义 |
| `src/experiments/rd_runner.py` | RD 扫描，ADAPTERS 字典，码率匹配，空间质量 | 注册新方法，不能只增加配置而忘记调度 |
| `src/experiments/matched_bitrate_runner.py` | 等码率评测 | 核查注册和计费逻辑后接入 |
| `src/experiments/search_codecs.py` | 搜索用编码执行路径 | 核查 evaluate_candidate 等分派 |
| `src/experiments/roi_search_runner.py` | ROI 选参与留出评价 | 支持新方法及缓存隔离 |
| `src/datasets/task_roi.py`、`eyetrackuav2.py` | ROI/眼动输入 | 重用时间和空间对齐规则 |
| `src/metrics/spatial_quality.py` | patch 空间质量聚合 | 保留现有输出，补充背景和 QP 日志 |
| `src/metrics/task_fidelity.py` | 任务区域质量 | 核查后复用 |
| `configs/codecs/h264.yaml` | 普通编码参数 | 新建配置，不修改默认基线行为 |

首次执行：读取适用 AGENTS.md、README、pyproject.toml、上述代码及相关 runner；记录 git HEAD、工作区修改、FFmpeg/x264 版本和编译选项、Python 版本。用 rg 搜索所有方法白名单、分派、结果列和绘图入口，列出最小改动清单。路径不存在时查找当前等价模块，不凭记忆新建平行框架。

## 4. 后端技术选择与真实性

概念关系为 `QP[t,i] ≈ frame_base_qp[t] + delta_qp[t,i]`。这是接口语义，实际 QP 还可能受舍入、裁剪、AQ、VBV、帧类型与编码器内部规则影响。禁止把请求值直接当成实际使用值。

优先评估一个小型原生 libx264 bridge，通过当前版本公开头文件中的逐宏块量化偏移接口注入逐帧地图。核实 `x264_picture_t` 相关 `quant_offsets`、AQ 前置条件、生命周期、网格顺序和延迟输出，不凭字段名称假定行为。若需要补丁，固定上游版本、单独保存最小补丁和构建说明，不替换系统 FFmpeg/x264。

另一个可选方案是 FFmpeg AVFrame ROI side data，但仅在核实当前 libx264 wrapper 支持、偏移换算和逐帧动态注入能力后采用。FFmpeg 的全局 `-qp`、CRF 或逐帧 qpfile 不能替代每帧宏块地图。普通 addroi 命令行配置不能未经验证就宣称支持任意动态网格。

第一版只承诺 H.264、8-bit、YUV420p、16×16 宏块。H.265 保持原有基线功能，不把 H.264 宏块假设硬套 CTU。禁用新方法时，不导入或加载新增原生依赖。

后端提供 capabilities：是否支持 delta/absolute、宏块尺寸、实时状态、实际 QP tracing、帧级基准控制。只实现 delta 时，absolute 请求必须明确报不支持；不能用未知 base_qp 伪造换算。

基础 QP 由 x264 码率控制决定时，不保证 Python 策略在编码当前帧之前能拿到它。接口中 base_qp 必须允许为空，支持直接输出 delta；未来策略需要当前 base_qp 或 lookahead 特征时，必须增加相应的编码器内部回调或提取接口并记录代价。

## 5. 稳定接口契约

建议放在 `src/codecs/qp_control/`，最终遵循仓库现有命名风格。避免通用插件框架和不必要依赖。公开策略接口可用 ABC/Protocol 或现有工厂模式实现，需兼容项目 Python 版本。

```python
class QPMapProvider:
    def reset(self, sequence_info, config): ...
    def select(self, context): ...  # 返回 QPDecision
    def update(self, feedback): ... # 默认空操作
    def close(self): ...
```

`context` 至少包含 sequence_id、源 frame_id、PTS/time_base、原始尺寸、padding 后尺寸、网格尺寸、目标码率、当前可用 ROI 覆盖率、可选当前 RGB 和编码器特征、仅过去已完成编码帧的反馈。各特征带可用性信息，策略声明 required_features；缺失时显式失败或采用配置指定的回退，不补虚假零值。

`QPDecision` 包含 mode（delta/absolute）、shape 为 `[ceil(H/16), ceil(W/16)]` 的数组、策略元数据、坐标系版本。数组必须有限值、维度正确、范围合法，行优先索引 i=row*ncols+col。缓存、外部地图须绑定源 frame_id，不能按编码器输出顺序直接索引。

`feedback` 包含对应源帧 ID、实际输出字节、编码耗时、累计/滚动码率、可获取的基础 QP 和实际块 QP、可用性标志。B 帧或 lookahead 造成延迟时，按 PTS 映射；没有真实逐帧数据就留空，不能将序列总字节均分后反馈给策略。

策略生命周期与序列/码率匹配试次隔离，reset 后不能继承上一试次的码率或运动状态。外部策略以显式注册或受控 module:Class 加载，禁止 eval 配置文本。未来 RL 策略仅替换 select/update，完整 RL-RC-DoT 复现是独立扩展。

## 6. ROI 和非 ROI 压缩策略

复用因果 ROI 来源，按有效图像面积计算宏块 ROI 覆盖率 a[t,i]∈[0,1]。默认混合块只要与 ROI 相交即保护，提供 coverage_threshold 和 soft_blend 可选方式；soft_blend 示例：delta=a*delta_roi+(1-a)*delta_background。padding 像素不影响覆盖率，解码后去除 padding。

至少支持：uniform（所有偏移为零或相同值）、fixed_roi（ROI/背景固定偏移）、external_map（按帧读取用户地图）。非 ROI 可选 fixed_delta、inherit（不施加背景额外偏移）、external；明确最终地图只由一个组合阶段生成，防止背景偏移重复叠加。背景和 ROI 可各自设置偏移边界。

后续算法允许直接接管整张图，也允许仅产生 ROI 或非 ROI 子区域调整。配置必须规定优先级，例如 external 完整图 > 区域策略组合；冲突配置直接拒绝。每个区域的原始策略值、组合结果及裁剪结果可追踪。

评价 ROI 与编码可见 ROI 分开保存。当前真实眼动仅用于评价时不能传给编码策略；若用 oracle ROI，方法名称和结果必须明确标识 oracle，不能与因果方法混同。缺失/无效 ROI 默认显式采用 uniform 回退并记录次数；评价有效性沿用既有规则。

## 7. 配置示例（实施时适配现有 schema）

```yaml
codec:
  name: h264_block_qp
  implementation: libx264_qp_bridge
  pixel_format: yuv420p
  encoder:
    preset: medium
    mode: bitrate
    bitrate_kbps: 100
    gop_size: 30
    b_frames: 0
    tune: zerolatency
  qp_control:
    provider: fixed_roi
    mode: delta
    block_size: 16
    roi_delta: -4.0
    non_roi:
      policy: fixed_delta
      delta: 4.0
    delta_limits: [-12.0, 12.0]
    mixed_block_policy: any_overlap
    missing_roi_policy: uniform
    external_map_path: null
    trace_requested: true
    trace_applied: false
  output:
    container: h264
    extension: h264
```

数值仅为初始实验配置，不能预设最优。原生后端的 AQ、VBV、lookahead、线程、scenecut 等必须有清晰默认值并写入 resolved config。未知 provider、不支持模式、非法地图、错帧外部图应立即报错。新配置关闭时旧配置、命令和输出仍有效。

## 8. 评测、计费与兼容性

新增方法接入 RD、等码率和 ROI 搜索；复用已有搜索和报告逻辑，只做必要扩展。配置选参只使用现有校准/验证划分，留出集不能调策略。分辨率、帧率、帧区间、像素格式和源视频必须一致。

增加同原生后端、同配置、zero-delta 的控制组以隔离后端差异。继续保留已有 FFmpeg H.264 基线。只在同一后端中比较零/非零地图，才能把差异可靠归因于地图；跨后端比较必须记录所有参数差异。

码率按实际 elementary stream 完整字节计费，包含 NAL 头和 QP 信令。发送端生成且不额外传输的地图日志不计网络字节；如果实现额外 side channel，应计入完整传输字节。可视化 MP4 与研究日志不能算成编码码率。

码率匹配每次重置策略和编码器，使用真实输出字节；继承 tolerance、max_iterations 和 bitrate_unreachable 语义。QP 地图会改变实际码率，不能认为相同 target 就公平。实时部署与离线迭代匹配区分，不能把多次离线编码的选参能力描述为在线控制。

保留现有结果字段；追加 provider、backend/version、ROI 来源、实际码率、ROI/non-ROI/global 质量、策略耗时、编码耗时、QP trace 路径。QP CSV 至少包括 frame_id、row、col、coverage、requested_delta、requested_absolute（可空）、base_qp（可空）、applied_qp（可空）及 available。不把空值写成 0。记录裁剪比例、无效 ROI 回退次数、地图校验和。

加入地图、ROI/背景质量、RD 图及其原始 CSV/NPZ。缓存签名包含源数据、ROI、策略配置/代码、后端版本、编码参数、外部地图哈希；新增方法不得误用旧缓存。保持原有结果字段和脚本退出码含义，不重写全部 runner。

## 9. 实现阶段与规范

1. 先完成仓库审计与后端可行性小样，读取公开接口源码确认 QP 注入机制。
2. 建立网格/ROI 映射、策略契约和 fixed/uniform/external 策略。
3. 实现编码 bridge 和适配器，生成可由标准解码器读取的 H.264 流。
4. 接入配置验证、runner、码率匹配、指标、缓存与报告。
5. 提供 `scripts/run_block_qp_benchmark.sh`，定位仓库根目录、使用当前 Python 环境，接收 --config，严格传递退出码；支持小规模 smoke 配置和正式 manifest 配置。依赖构建提供显式独立入口，不覆盖系统库。
6. 完成下列验收和用户文档，给出一键运行示例。

遵循项目缩进、类型标注、导入、Path、日志、返回字典及 YAML 风格。subprocess 使用参数数组并保留 stderr，禁止 shell 拼接。资源异常时关闭编码器并清理临时文件；原子写结果。新增依赖可选，原生编译失败只阻止新方法。禁止静默退回普通 H.264 并继续标记新方法成功。

## 10. 必须完成的验证

| 验证 | 验收要求 |
|---|---|
| 真实 QP 注入 | 使用左右等复杂度的合成纹理视频，交换两侧 delta；通过编码器实际 QP trace 或解码码流解析证明对应宏块 QP 随之变化。仅文件大小或 PSNR 改变不足以单独证明 |
| 零偏移控制 | 同后端相同参数下零地图与无自定义偏移路径一致或差异有可复现解释；不要求不同后端比特流相同 |
| 动态地图 | ROI 左右交替，确认每帧地图及源帧 ID 正确，无一帧错位；若宣称支持 B 帧则验证重排与 flush |
| 边界尺寸 | 非16整除尺寸正确 padding/crop；输出尺寸、帧数、PTS 保持对齐；不兼容的奇数 YUV420 尺寸有明确处理 |
| 输入错误 | 错尺寸、NaN、超范围、缺失外部帧、重复 frame_id、未知策略均产生可定位错误 |
| ROI 边界 | 全 ROI、无 ROI、无效 ROI、混合块按配置处理；无评价标签泄漏 |
| 码率与状态 | 真实字节计费可复算，匹配失败明确标记，重试 reset 生效 |
| 向后兼容 | 原 H.264/H.265、BiSTAR/Atlas 相关小规模路径通过；关闭新方法时无原生库仍能导入和执行旧路径 |
| 扩展性 | 一个最小自定义 provider 在不改后端和 runner 的情况下运行；外部地图策略可重放 |
| 结果完整性 | 配置/版本/日志/地图和图表元数据完整；断点续跑与新旧缓存隔离 |

不预设 ROI 画质或 BD-rate 必然提升；小样验证正确性，正式性能结论须由实际留出实验得出。运行吞吐与端到端延迟分开报告，lookahead 缓冲、策略和 bridge 开销均纳入说明。未具备 trace 时明确标为待验证，不能声称实现通过全部验收。

## 11. 最终交付

交付适配器/后端、策略与配置、单命令入口、兼容性改动、必要测试、运行文档和小规模结果。报告实现路径、已验证能力、未支持能力、测试环境与剩余阻塞。若构建环境受限，保留完整实现和可复现命令并如实记录未运行项目，不能输出虚构成功或性能收益。

创建的 skill 应把本需求书中的接口语义、真实 QP 验证、因果性和旧路径兼容性作为不可省略的约束；目录和类名可依据仓库当前结构调整。

## 12. 依据与核查入口

- 仓库：https://github.com/HanXiao485/BiSTAR
- 编码适配器：https://github.com/HanXiao485/BiSTAR/blob/main/src/codecs/ffmpeg_common.py
- RD 运行器：https://github.com/HanXiao485/BiSTAR/blob/main/src/experiments/rd_runner.py
- ROI 搜索：https://github.com/HanXiao485/BiSTAR/blob/main/src/experiments/roi_search_runner.py
- 论文：https://openaccess.thecvf.com/content/CVPR2025/html/Gadot_RL-RC-DoT_A_Block-level_RL_agent_for_Task-Aware_Video_Compression_CVPR_2025_paper.html

仓库能力描述依据本会话源码读取；本文建议的新接口、配置和后端均属于待实现设计，不代表仓库已具备。实施者必须以固定提交对应的源码和后端头文件为最终依据。
