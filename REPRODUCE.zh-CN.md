# Weave 复现准备与监控

## 正式训练：每卡 10,240（2026-10-19 19:28）

按用户选择，从头启动完整九物体 D_train 的统一策略训练，GPU 0、1 各 10,240 环境，总计 20,480，
目标 100,000 次迭代，使用 Isaac Sim 5.1 和原 PPO / Muon / SimBa / torch.compile 配置。
用户随后明确允许停止旧任务，已停止两个 RoboNaldo 训练及其自动接续调度器，旧 checkpoint 保留。
因此本轮为两卡独占，显存条件与此前共用 GPU 的 smoke 不同。

后台会话为 `weave-train-dual10240`，日志为 `logs/training/latest-dual/train.log`，
实际任务目录记录在 `logs/preparation/dual10240-training-job.path`。
启动验证和旧任务停止记录见 [正式训练记录](outputs/training-starts/2026-10-19/dual10240/README.md)。

## 按作者 issue #2 更新训练 recipe（2026-10-19）

作者确认使用 [8 × 16,384 = 131,072 个并行环境](https://github.com/xiaohu-art/Weave/issues/2)。
此前双卡总计 4,096 的配置仅为其 1/32，不能称为论文训练规模。
新增 `train_paper.yaml` 和 `scripts/reproduce/train_paper.sh`，默认从头训练完整九物体 D_train，
保留公开 PPO / Muon / SimBa 参数及已完成的 bug 修复，默认使用上游安装脚本的 Isaac Sim 5.1 环境。
详细参数、八卡启动、断点约束及双卡缩小版本见 [论文训练 recipe](scripts/reproduce/PAPER_TRAINING.zh-CN.md)。

```bash
bash scripts/reproduce/train_paper.sh --dry-run
```

上述命令仅打印计划。当前机器仅可见两张 GPU，不能运行完整八卡配置；recipe 修改阶段仅做静态检查。
后续容量 smoke 已确认：**每卡 8,192 通过 5 个完整训练迭代、至少约 5 GiB 空闲；10,240 余量不足 1 GiB；
12,288 和 16,384 均 OOM**。当前共享 GPU 推荐每卡 8,192，正式启动需显式覆盖环境数量，详见
[容量测试报告](outputs/capacity-smoke/2026-10-19_18-58-49/README.md)及[启动说明](scripts/reproduce/PAPER_TRAINING.zh-CN.md#本机共享-gpu-的容量-smoke2026-10-19)。
双卡脚本现为 2 × 16,384 = 32,768 环境，即作者规模的 1/4；以下历史训练记录的 4,096 数字仍指当时设置。

## 训练问题修复（2026-10-19）

已修复新版多物体克隆与参考动作错配、拾取支撑桌配对和接触力接口含义变化，
并加固优化器断点恢复：PPO 自适应学习率直接更新实际 Muon / AdamW。
每次创建环境都会核验所有物体的 USD 来源，发现错配会停止启动。
修复说明、回归验证及后续训练命令见 [训练修复说明](scripts/reproduce/TRAINING_FIXES.zh-CN.md)。

此前 `model_64500.pt` / `model_99999.pt` 的训练已受问题影响；修复代码不会自动恢复权重质量。
旧双卡诊断恢复脚本默认使用首次受影响恢复前保存的 `model_300.pt`，运行名为 `reference-scaled-dual-isaac62-corrected`。
该权重来自旧小规模训练，不作为新论文 recipe 的起点；论文启动器默认从头训练。
本次只做修复和短验证，没有启动全量重训，原有训练任务保持运行。

## 最终统一策略评估（2026-10-19）

100,000 次迭代已经完成，最终权重为 `2026-10-15_19-21-47_paper-reference-isaac62-resumed/model_99999.pt`。
这是修复前的问题训练记录，不能视为修复后策略的评估。
已按用户要求在 GPU 1 与现有训练共享运行完整 train/test split 评估；此前的 CPU 诊断结果单独保留。
目标、评估口径、软件版本差异、运行状态与复跑命令见
[论文评估说明](scripts/reproduce/PAPER_EVAL.zh-CN.md)。

## Isaac Sim 6.2 与 Viser（2026-10-15）

新增独立 uv 环境，固定 Isaac Sim 6.2.0.0、Isaac Lab `release/3.0.0` 的提交 `7dd5688`、Python 3.12。
`eval.py` 和 `play.py` 支持 `--viser`。安装、SSH 连接、四类拾取任务与验证记录见
[新版环境 README](environments/isaac62/README.md)。

```bash
cd /mnt/data/ruofei/zhangyu/code/Weave
source scripts/reproduce/activate_isaac62.sh
python scripts/reproduce/viser_pickups.py --sequence sub1_tripod_075
```

本地使用 `ssh -N -L 8080:127.0.0.1:8080 <服务器SSH别名>` 转发，打开 `http://localhost:8080`，
点击 **开始 / 重播** 执行。侧栏可搜索全部 9,474 个参考片段，选择后直接切换任务。
默认加载最近保存的统一策略 checkpoint，切换任务共享同一份权重。
评估结束保留终止姿态与指标，按 `Ctrl+C` 退出。
旧的根目录 `.venv` 与 Isaac Sim 5.1 环境继续保留，以下历史记录中的版本和命令仍指旧环境。

## 当前 PAI 机器的 uv 项目环境（2026-10-13）

仓库根 `pyproject.toml` 现已包含复现依赖，`uv.lock` 固定完整依赖树。
环境位于 `/mnt/data/ruofei/zhangyu/code/Weave/.venv`，使用 uv 托管的 Python 3.11.14，
版本通过 `.python-version` 固定；
IsaacLab 固定提交仍为 `e17312889676ed229b986d56c9e0b23a01cf0ab7`，位于仓库同级。
本机驱动 570.124.04 使用 PyTorch 2.11.0 + CUDA 12.8、torchvision 0.26.0、
NumPy 1.26.4 和 RSL-RL 3.1.2。普通包使用阿里云镜像，PyTorch 使用上海交大 CUDA 12.8 镜像。
Isaac Sim 5.1.0 是仓库同级的独立安装，不属于 `uv sync` 下载的 Python 包；
CUDA 12.8 编译工具位于同级 `cuda-12.8/`。

```bash
cd /mnt/data/ruofei/zhangyu/code/Weave
uv sync --locked
source scripts/reproduce/activate.sh
# GPU 0 空闲时，运行单 GPU 的最小训练检查：
bash scripts/reproduce/run_gpu0.sh scripts/rsl_rl/train.py \
  --task G1-Inspire-HOI-v0 --headless \
  --config-dir ./configs/track --config-name smoke
```

最终安装日志见 `logs/preparation/uv-sync-managed.log`；
首次依赖解析和下载日志见 `logs/preparation/uv-sync.log`。
本机数据和 Git LFS 资产的重新校验结果分别见
`logs/preparation/dataset.current-check.json` 和 `logs/preparation/assets.current-check.json`。
下文的 `/home/sherlock/code/` 路径、`sim51` 环境与旧 `SUCCESS` 状态来自迁移前的机器，
不能作为当前 `.venv` 安装和运行成功的证据。

本机检查结果：`uv sync --locked` 成功安装 160 个包，再次同步只需约 0.2 秒；
`uv pip check`、GPU 矩阵乘法、cuDNN 卷积、56 个数据文件的 SHA256、22 个 LFS 资产的
SHA256 和数据格式检查均通过。`G1-Inspire-HOI-v0` 能正常注册。
环境快照见 `logs/preparation/environment.uv.json` 与 `environment.uv.freeze.txt`，
当前检查摘要见 `logs/preparation/readiness.uv.json`。
激活 Isaac Sim 后会优先加载其预打包 NumPy 1.26.0，uv 环境本身安装的是 1.26.4。

**Vulkan 已修复并验证：** 容器已有的 `libEGL_nvidia.so.0` 可以提供 Vulkan，
但 GLX Vulkan 入口加载失败，且缺少 NVIDIA ICD 配置。
仓库内新增 `scripts/reproduce/vulkan/nvidia_icd.json`，激活脚本为当前 shell 设置
`VK_DRIVER_FILES` 和 `VK_ICD_FILENAMES`，通过 EGL 驱动访问 Vulkan；
没有修改系统驱动、系统 ICD 配置或容器启动参数，也没有重启容器。
NVIDIA 官方对 EGL Vulkan ICD 的说明见
<https://download.nvidia.com/XFree86/Linux-x86_64/570.124.04/README/installedcomponents.html>。

`vulkaninfo` 已识别 8 张 NVIDIA GPU，Isaac Sim 启动时使用 GPU 0，
多 GPU 渲染已关闭。`smoke` 的 32 个环境、2 次训练迭代通过，生成：
`logs/rsl_rl/g1_inspire_hoi/2026-10-13_16-46-38_preparation-vulkan-gpu0/model_1.pt`。
日志见 `logs/preparation/vulkan-summary.log`、`vulkan-isaacsim-check.log` 和 `vulkan-smoke.log`。
测试结束后 GPU 0 已释放；GPU 2–7 的已有训练 PID 和显存占用在检查前后保持一致。

共享机器运行训练、评估或播放时，使用 `scripts/reproduce/run_gpu0.sh`：
它会检查物理 GPU 0 的已有进程、显存和利用率，忙碌时拒绝启动，
并将 CUDA 访问限制到 GPU 0、关闭多 GPU 渲染。
文件锁只能协调使用该脚本启动的任务，GPU 检查不会预留设备；
其他用户在检查后启动新任务仍需自行协调。
单卡完整训练集实验可将上面命令中的 `smoke` 换为 `train_reference`，但其并行规模仍小于论文；
完整论文规模请使用文首的 `train_paper.sh`。
此次两迭代检查只验证安装与运行流程，不代表论文性能。

### GPU 0–1 双卡历史实验

2026-10-13 17:04（北京时间）启动完整训练参考集的统一策略训练，
原会话为 `weave-reference-dual`，现已按用户要求停止，保留 `model_300.pt`。
配置 `train_reference_dual` 继承作者训练参数，
使用九种物体、7,869 个训练片段和 100,000 次迭代；
每卡 2,048 个环境，双卡合计 4,096，保存间隔为 100 次迭代。
这是使用完整训练参考集的小规模统一策略实验，其并行规模仅为作者后来确认规模的 1/32。

启动脚本为 `scripts/reproduce/train_gpu01.sh`，仅允许空闲的物理 GPU 0–1，
通过 UUID 限制 CUDA 可见设备，每个 rank 使用一张卡，关闭多 GPU 渲染，
使用自动选择的独立通信端口。CPU 亲和性限制为 16 个核、nice 为 10，
PyTorch 编译并行度为每 rank 2，以减少对已有任务的 CPU 争用。
启动前的 GPU 和其他训练进程快照保存在任务日志目录。
各 rank 使用统一运行时间戳，配置快照只由 rank 0 写入。

```bash
cd /mnt/data/ruofei/zhangyu/code/Weave
tail -f logs/training/latest-dual/train.log
cat logs/training/latest-dual/status
tmux attach -t weave-pickups-then-paper
```

`latest-dual` 始终指向最新双卡任务；原任务日志为
`logs/training/2026-10-13_17-04-51_reference-dual/`，状态为 `STOPPED_BY_USER`。
详细启动参数见各任务目录的 `launch.json`，逐 rank 日志在 `workers/`。
checkpoint 与 TensorBoard 事件保存在
`logs/rsl_rl/g1_inspire_hoi/2026-10-13_17-04-51_paper-reference-dual/`。
训练在 tmux 中运行，关闭聊天或退出查看日志不会终止训练。
启动脚本的 `status` 会在进程结束后更新为 `COMPLETED` 或 `FAILED exit=...`。
17:10 的启动验证确认训练已进行到第 49 次迭代，`model_0.pt` 已保存；
两个 rank 的设备和全训练集配置正确，GPU 2–7 原有六个进程的 PID 和显存占用保持不变。
验证快照见任务目录的 `verified-start.json`。此时每张训练卡使用约 22 GiB 显存，
稳定迭代耗时约 2.6 秒；按此速度估算 100,000 次迭代约需 3 天，实际耗时以训练日志为准。

### 拾取专项训练后自动接续全量训练

2026-10-13 17:42（北京时间）启动 tmux 会话 `weave-pickups-then-paper`。
队列脚本为 `scripts/reproduce/pickup_then_full.py`，先执行双卡拾取专项训练，
再逐物体生成评估报告，最后接续原统一策略训练。队列任一阶段执行失败时停止，
评估成功率低本身不阻止接续。3,000 次迭代是首轮训练预算，不保证动作已经学会。

`scripts/reproduce/prepare_pickups.py` 从作者已发布的参考文件提取下列四条序列，
包含其发布的变体，共 12 个片段；原始训练集和测试集不被改写。

| 序列 | 来源 | 片段数 |
| --- | --- | --- |
| `sub1_tripod_075` | train | 3 |
| `sub16_clothesstand_012` | test | 5 |
| `sub9_clothesstand_039` | train | 3 |
| `sub7_whitechair_072` | train | 1 |

未在已发布的 Weave 参考文件中找到 `sub6_woodchair_066`，没有用其他动作代替。
白椅子参考第一帧已经存在手部接触，不能据此验证从无接触状态接近并抓住物体的全过程。
作者场景没有桌面，新增的 `G1-Inspire-HOI-Pickup-v0` 仅在白椅子环境加入固定承托面：
根据初始物体网格拟合，顶部高度约 0.278 米；尺寸和原始 OMOMO 桌面不等价。
三脚架和衣架环境没有该承托面的碰撞，原论文任务的场景保持原样。
提取清单、来源 split 和承托面参数见 `data/pickups/manifest.json`。

专项配置 `train_pickups_dual` 使用每卡 2,048 个环境、总计 4,096 个环境，
保存间隔 100 次迭代；保留原策略、优化器和网络编译配置。完成 3,000 次迭代后，
配置 `eval_pickups` 从动作开头评估全部 12 个片段，报告写入队列目录的 `eval/`。
全量阶段从原 `model_300.pt` 恢复，增加 99,700 次迭代，目标累计 100,000 次。
专项包含一条 test 序列，因此专项权重仅用于专项评估，不能用来初始化论文全量训练。

```bash
cd /mnt/data/ruofei/zhangyu/code/Weave
cat logs/training/latest-chain/state.json
cat logs/training/latest-chain/plan.json
tail -f logs/training/latest-pickups/train.log
tmux attach -t weave-pickups-then-paper
```

`latest-chain` 指向队列目录，记录阶段、接续计划和评估结果；
`latest-pickups` 保持指向专项任务，`latest-dual` 在全量阶段开始后切换到全量任务。
队列和 GPU 启动器使用文件锁，并在每阶段启动前检查 GPU 空闲状态；
已有其他进程占用 GPU 0–1 时等待或停止，不终止其他任务。
所有训练均限制在物理 GPU 0–1，GPU 2–7 原有任务保持运行。

仅在现有队列结束或已停止时重新启动：

```bash
source scripts/reproduce/activate.sh
python -I scripts/reproduce/pickup_then_full.py \
  --paper-checkpoint "$PWD/logs/rsl_rl/g1_inspire_hoi/2026-10-13_17-04-51_paper-reference-dual/model_300.pt" \
  --specialist-iterations 3000
```

启动前检查已完成：双卡小规模专项训练、白椅子 checkpoint 加载与评估，
以及三项队列测试（执行顺序、专项训练失败时停止、评估异常时停止）。
试运行的低成功率只验证基础流程，不作为正式训练结果。
专项训练现已完成，专项 checkpoint 为
`logs/rsl_rl/g1_inspire_hoi/2026-10-13_17-42-07_pickups-specialist-dual/model_2999.pt`。
全量训练已于 2026-10-13 20:01 自动从原 `model_300.pt` 接续。
17:44 的正式启动检查确认已完成第 10 次迭代并保存 `model_0.pt`，
两个 rank 的承托面碰撞分配和策略编译已确认，原有六个 GPU 任务的 PID、显存保持一致。
检查证据见专项任务目录的 `verified-start.json`；当前迭代约 2.9–3.1 秒，
据此估算专项训练约 2.5 小时，随后还需评估和全量训练初始化。

### 无界面录制专项评估视频

2026-10-14 已使用专项 `model_2999.pt` 为四条序列各录制一个参考变体的首次执行。
输出位于 `outputs/pickup-videos/`：`overview.mp4` 为四宫格总览，
四个以序列名命名的 MP4 为单独视频，说明文件为 `outputs/pickup-videos/README.md`，不提供 ZIP 包。
视频为 50 fps 的仿真时间，字幕标明本次执行的进度、终止原因；终止姿态额外停留一秒。
各任务的 `metrics.json` 与 `manifest.json` 保存本次录像结果，
不能替代先前的 12 片段整组评估。白椅子仍使用拟合承托面。

录像接口新增在 `scripts/rsl_rl/eval.py`，使用 `--video`，可用 `--video_path` 指定 MP4 路径；
要求单环境和单参考片段。它在自动重置前保存最后姿态，并检查黑帧。
Isaac Sim 5.1 的 headless 配置关闭了 Replicator 调度，
录像过程显式初始化调度并逐帧请求新的图像，避免黑屏或重复初始姿态。
只有录像进程启用这些设置，原训练进程不改变。

通常应在 GPU 0 空闲时运行：

```bash
source scripts/reproduce/activate.sh
python scripts/reproduce/record_pickups.py
python scripts/reproduce/compose_pickup_videos.py outputs/pickup-videos
```

本次录像使用 GPU 0 的剩余显存，单环境顺序执行，CPU 限制四核、nice 19。
`record_pickups.py --share-weave-gpu0` 仅用于本机已确认运行的双 rank Weave 任务；
检查到其他额外 GPU 0 计算进程时拒绝启动。
默认模式使用 GPU 0 文件锁并拒绝共享占用中的 GPU。
运行后检查视频能正常解码、有非黑画面和动作变化，录像 GPU 已释放，
全部原有 GPU 任务的 PID 与显存占用保持一致，全量训练继续运行。
验证结果在 `outputs/pickup-videos/verification.json`。

准备基于当前 Weave 提交 `d839eb298358b71aedd0407c7ba9bbb05c14e05e`。
本文记录配置和操作方式；**是否安装成功，以后台任务状态和日志为准**。

## 后台任务

```bash
tmux attach -t weave-env       # 环境安装；checks 窗口等待安装/数据完成后检查
tmux attach -t weave-data      # 参考数据下载；assets 窗口是 Git LFS 资产下载
```

按 `Ctrl+b`、`d` 退出查看，后台任务继续。按 `Ctrl+b`、`n` 切换窗口。
tmux 和 git-lfs 安装在 `~/.local/bin`，底层文件在 `/home/sherlock/code/.weave-tools`；不需要 sudo。

```bash
cd /home/sherlock/code/Weave
watch -n 5 bash scripts/reproduce/status.sh
tail -f logs/preparation/env.log
tail -f logs/preparation/data.log
tail -f logs/preparation/validate.log
```

状态含义：`RUNNING` 正在执行（validate 可能在等待）；`SUCCESS` 已完成；
`FAILED exit=...` 已失败。`local` 是启用编译的本机训练检查，`eval` 是 checkpoint 重新加载和评估检查。
日志以追加方式保留历次尝试，请结合最新状态和最后一次启动记录查看。

## 环境

- Ubuntu 22.04、RTX 4070 系列 GPU，12 GB 显存，约 64 GB 内存。
- uv 创建 Python 3.11 环境：`/home/sherlock/code/sim51`。
- Isaac Sim 5.1.0：`/home/sherlock/code/isaacsim`。
- IsaacLab 固定提交：`e17312889676ed229b986d56c9e0b23a01cf0ab7`，目录 `/home/sherlock/code/IsaacLab`。
- 作者安装脚本按当前驱动选择 CUDA 13，并安装支持 Muon 的 PyTorch >= 2.10 和 NumPy < 2。
- 普通 Python 包使用清华 PyPI 镜像；PyTorch CUDA wheel 使用作者脚本指定的 PyTorch 官方源。
  可以在重跑前设置 `WEAVE_PYPI_INDEX=https://pypi.org/simple` 改回 PyPI。
- cmake 安装在虚拟环境中，避免 IsaacLab 安装步骤调用 sudo。

安装完成后：

```bash
cd /home/sherlock/code/Weave
source scripts/reproduce/activate.sh
python scripts/list_envs.py
```

本项目原来的根 `pyproject.toml` 只配置代码检查，并不是 uv 项目依赖清单。
这里按作者要求使用 `uv venv` 和 `uv pip` 管理仿真环境。激活脚本还会设置 Isaac Sim 的库路径；
`source scripts/reproduce/activate.sh` 同时暴露已安装的 nvcc 和 `CUDA_HOME`，供编译路径使用。
运行仿真前应先 `source`，而不是只调用虚拟环境中的 Python。

环境成功安装后会生成：

- `logs/preparation/environment.freeze.txt`：实际安装包版本和 editable 源码位置。
- `logs/preparation/environment.json`：Python、Torch、CUDA、GPU、核心包版本和仓库提交。

这些文件是本次环境的快照，不是跨机器通用的 `uv.lock`。
当前 PyTorch 为 `2.14.1+cu130`，IsaacLab 为 `0.54.3`，RSL-RL 为 `3.1.2`。
已将 wheel 固定到 `0.45.1`，兼容仿真环境的 packaging 23.2。
激活 Isaac Sim 后 NumPy 从其预打包目录加载，实际运行版本为 1.26.0；
uv 环境安装的是 1.26.4，两个版本都符合作者的 NumPy < 2 约束。环境快照同时记录模块的实际加载位置。

## 数据与资产

下载源：[作者数据集 appolyn/Weave](https://huggingface.co/datasets/appolyn/Weave)。
固定修订：`db4a90f705f06667bd6b200053237e69644f0064`。
默认下载参考动作和物体资产，共 56 个文件，约 11.59 GB。

| 远端目录 | 本地目录 | 用途 |
| --- | --- | --- |
| `reference/train/` | `Weave/data/train/` | 7,869 个训练参考片段 |
| `reference/test/` | `Weave/data/test/` | 1,605 个测试参考片段 |
| `objects/` | `Weave/data/objects/` | 作者发布的物体资产备份 |

仿真运行使用 `source/g1_hoi_learning/g1_hoi_learning/` 下的机器人、物体和地面资产。
Git LFS 已通过 `git lfs pull` 补齐，并用 `git lfs fsck` 检查；SDF、surface 和 BPS 文件原本就在仓库中。
动作下载会校验文件大小和作者发布的 LFS SHA256。中断文件以 `.part` 保存，重新运行会续传。

- `logs/preparation/dataset.remote.json`：远端路径、文件大小、作者校验信息。
- `logs/preparation/dataset.verified.json`：全部下载成功后生成的本地 SHA256 清单。
- `logs/preparation/dataset.validation.json`：数据格式检查通过后生成的片段数、帧数、浮点数组内存统计。

作者数据卡说明，`rollout/` 是训练策略实际执行的轨迹，contact label 与参考数据的语义不同。
参考跟踪训练使用 `reference/`。如后续要研究策略执行轨迹，可单独增加下载：

```bash
python3 scripts/reproduce/download_data.py --include-rollouts
```

这会额外下载约 11 GB；请在当前下载任务结束后执行，避免同时写同一文件。
本次仓库与数据集文件清单中未发现作者的 `.pt` 策略权重；评估需要训练得到的 checkpoint。
准备检查生成的 2 次迭代 checkpoint 仅用于验证保存/加载流程，不能用于对照论文性能。

## 检查与实验配置

后台 `checks` 窗口会等待 `env`、`data`、`assets` 三项成功，然后：

1. 检查两个 split 的片段数、50 fps、53 个关节、54 个刚体、各数组形状和 dtype。
2. 检查 512 个物体 surface 点、128³ SDF 和 128 个 BPS 点。
3. 用 `smoke.yaml` 执行 32 个并行环境、2 次迭代的测试，并设置 20 分钟超时。

原始 `configs/track/train.yaml` 实际指向 `data/test/`，对应论文中直接训练测试参考集的实验；
`eval.yaml` 默认只评估 smalltable。新配置保留这些原始配置，分别提供：

| 配置名 | 用途 |
| --- | --- |
| `smoke` | 单物体、32 环境、2 次迭代，检查环境可运行 |
| `train_local` | smallbox 训练集、256 环境、3,000 次迭代，用于本机单物体实验 |
| `train_reference` | 九物体完整训练集，继承作者的其余训练参数 |
| `train_reference_dual` | 双卡缩小实验，每卡 16,384，总计 32,768，仅为作者规模的 1/4 |
| `train_paper` | 九物体完整训练集，8 × 16,384 环境；使用 `train_paper.sh` 启动并校验 |
| `eval_reference` | 单物体测试配置，完整测试集需按物体分别运行 |

本机显存 12 GB，低于 [Isaac Sim 5.1 官方列出的 16 GB 最低配置](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)。
是否能运行具体任务，应以 smoke 检查结果为准。
已测得完整训练集的浮点动作数组合计 **11.477 GiB**，完整测试集合计 **2.151 GiB**，
这些数字还不包括仿真、策略、优化器和临时张量。
源码 `MotionLoader` 会把全部参考动作数组加载到 GPU，并在拼接时额外分配内存。
因此仅降低 `num_envs` 不能解决完整训练集的动作缓冲显存需求。
`train_local` 是缩小范围的本机实验；完整九物体实验应结合 validation 输出的内存统计规划 GPU。

准备完成后，手动启动本机实验：

```bash
cd /home/sherlock/code/Weave
source scripts/reproduce/activate.sh
python scripts/rsl_rl/train.py --task G1-Inspire-HOI-v0 \
  --headless --config-dir ./configs/track --config-name train_local
```

单卡完整训练集实验（需要足够显存，非论文并行规模）：

```bash
python scripts/rsl_rl/train.py --task G1-Inspire-HOI-v0 \
  --headless --config-dir ./configs/track --config-name train_reference
```

单物体评估（默认 smalltable），替换为自己训练得到的实际权重路径：

```bash
python scripts/rsl_rl/eval.py --task G1-Inspire-HOI-v0 \
  --headless --config-dir ./configs/track --config-name eval_reference \
  --checkpoint /absolute/path/to/model_100000.pt
```

`MotionCommand.eval_mode` 强制要求单物体参考集。完整测试集需按物体分别评估；
例如 smallbox：

```bash
python scripts/rsl_rl/eval.py --task G1-Inspire-HOI-v0 \
  --headless --config-dir ./configs/track --config-name eval_reference \
  --checkpoint /absolute/path/to/model_100000.pt \
  'env.commands.motion.motion_files=[./data/test/smallbox_24clip_var5.npz]'
```

全部九物体可按以下方式逐个评估，报告保存在各物体的目录中：

```bash
for motion in data/test/*.npz; do
  object="$(basename "$motion" .npz)"
  python scripts/rsl_rl/eval.py --task G1-Inspire-HOI-v0 \
    --headless --config-dir ./configs/track --config-name eval_reference \
    --checkpoint /absolute/path/to/model_100000.pt \
    --output_dir "outputs/eval/$object" \
    "env.commands.motion.motion_files=[$motion]"
done
```

这些准备覆盖已发布参考动作上的策略学习与仿真评估；原始人类动作到参考动作的完整前处理链路，
还需要另行确认作者是否提供对应代码和输入数据。
实验拆分与论文指标见 [论文](https://arxiv.org/html/2609.16683v1)。

## 失败后的重跑

确认同名任务已停止后，在 tmux 的空闲 shell 中运行：

```bash
cd /home/sherlock/code/Weave
bash scripts/reproduce/run_job.sh env bash scripts/reproduce/install_env.sh
bash scripts/reproduce/run_job.sh data python3 -u scripts/reproduce/download_data.py
bash scripts/reproduce/run_job.sh assets bash scripts/reproduce/download_assets.sh
bash scripts/reproduce/run_job.sh validate bash scripts/reproduce/validate_when_ready.sh
```

环境已经安装、只有最终检查失败时，可用以下命令修复并检查，避免重复安装 IsaacLab 和 PyTorch：

```bash
bash scripts/reproduce/run_job.sh env bash scripts/reproduce/repair_env.sh
bash scripts/reproduce/run_job.sh local bash scripts/reproduce/check_local.sh
bash scripts/reproduce/run_job.sh eval bash scripts/reproduce/check_eval.sh
```

只重跑失败项。若 checks 因前置任务失败而退出，修复前置任务后也要重跑 validate。
保留 Isaac Sim ZIP 和 `.part` 可避免重新下载；首次解压、安装和编译缓存也会占用额外磁盘。

## 代理与正式实验前的核对项

此次下载继承本机 `HTTP_PROXY` 和 `HTTPS_PROXY`，地址为 `http://localhost:7890`。
参考数据及物体资产共 11,590,792,152 字节（11.59 GB，约 10.79 GiB），
其中参考动作 11.53 GB，物体资产约 60 MB。Isaac Sim ZIP 另外约 8.77 GB；
可选 rollout 轨迹未下载。

原始 `eval.py` 的聚合跟踪误差按帧加权，而论文给出的公式按片段平均。
环境准备检查使用作者现有实现；正式逐项对照论文指标时，应先核对并统一这个口径。

## 本次准备结果

以下检查已完成，六项状态均为 `SUCCESS`：

- Python/Isaac Sim/IsaacLab 安装与包依赖兼容检查，CUDA 和 cuDNN 可用。
- 56 个数据文件全部下载并校验；Git LFS 资产完整。
- 7,869 个训练片段、1,605 个测试片段及物体几何格式验证。
- 32 环境、2 次迭代的基础训练，保存 checkpoint。
- smallbox 训练集、256 环境、2 次迭代，开启 `torch.compile` 的训练检查。
- 加载基础检查的 checkpoint，完成 smallbox 测试集全部 120 个片段并输出评估报告。

机器可读汇总：`logs/preparation/readiness.json`。
评估检查报告：`logs/preparation/eval-smoke/metrics.json`。
这些短训练是准备检查，尚未执行 3,000/100,000 次迭代的正式实验。

继续本机训练可直接运行：

```bash
cd /home/sherlock/code/Weave
source scripts/reproduce/activate.sh
python scripts/rsl_rl/train.py --task G1-Inspire-HOI-v0 \
  --headless --config-dir ./configs/track --config-name train_local
```
