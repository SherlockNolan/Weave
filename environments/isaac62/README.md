# Isaac Sim 6.2 与 Viser 可视化

本目录是 Weave 的新版 uv 项目。旧的根目录 `.venv`、Isaac Sim 5.1 和 Isaac Lab 2.3 环境继续保留。
此次升级未安装系统驱动、重启机器或停止其他训练进程。

## 固定版本

| 组件 | 版本 |
| --- | --- |
| Python | 3.12.13 |
| Isaac Sim | 6.2.0.0 |
| Isaac Lab | `release/3.0.0`，提交 `7dd56885594d4109d8677eb35cfae0cbf5e2aa76`；Python 包版本 27.2.0 |
| PyTorch | 2.11.0 + CUDA 12.8 |
| Warp / Viser | 1.17.0 / 1.1.1 |
| NumPy / packaging | 2.3.1 / 26.0 |
| RSL-RL | 3.1.2，保留既有 MuonPPO / SimBa checkpoint 的兼容性 |

版本以 2026-10-15 安装时的发布版本为准，依赖由本目录 `uv.lock` 固定。
参见 [Isaac Lab 版本兼容关系](https://github.com/isaac-sim/IsaacLab)。

## 安装与激活

当前机器已经安装完成，直接激活即可：

```bash
cd /mnt/data/ruofei/zhangyu/code/Weave
source scripts/reproduce/activate_isaac62.sh
```

容器重建后如果本地环境丢失，运行以下命令恢复：

```bash
bash scripts/reproduce/setup_isaac62.sh
source scripts/reproduce/activate_isaac62.sh
```

项目 `.venv` 链接到本地 SSD 的 `/var/tmp/weave-isaac62-*/venv`，避免共享盘安装、导入数万个小文件过慢。
新建环境时可用 `WEAVE_RUNTIME_ROOT` 指定本地盘根目录。
大 wheel 的持久缓存位于 `/mnt/data/ruofei/zhangyu/cache/weave-wheels`，可用 `WEAVE_WHEEL_CACHE` 覆盖。
安装脚本校验 SHA256、支持并行分段下载与续传，再执行 `uv sync --locked`。
普通 Python 包优先使用阿里云镜像；完整下载地址与哈希见 `large-wheels.json`。

Isaac Lab 源码放在同级 `IsaacLab-3.0-latest`。安装脚本检查固定提交，不会覆盖已有的其他版本。
激活脚本仅为当前 shell 选择与主机内核驱动匹配的 CUDA 动态库，没有修改系统链接。

依赖解析沿用 Isaac Lab 3 上游对 `fsspec>=2026.6.0`、`urllib3>=2.8.0` 的覆盖设置。
因此 `uv pip check` 会报告这两项与 Isaac Sim 元数据中的旧精确版本冲突；这是明确保留的上游覆盖，
不是遗漏安装。NumPy 与 packaging 按 Isaac Sim 的版本约束固定，并验证了 Numba 导入和编译。

## 搜索并切换全部任务

启动三脚架任务：

```bash
python scripts/reproduce/viser_pickups.py --sequence sub1_tripod_075
```

在**本地电脑**另开终端建立 SSH 转发，服务器别名替换成平时登录这台机器的地址或配置名：

```bash
ssh -N -L 8080:127.0.0.1:8080 <服务器SSH别名>
```

打开 <http://localhost:8080>，等机器人和物体加载后点击 **开始 / 重播**。
可在浏览器中旋转、平移、缩放视角；不需要 X11、桌面 GUI 或 RTX 视频编码。
默认仅监听服务器 `127.0.0.1`。端口被占用会明确报错，不会自动换到另一个端口。

侧栏的 **任务选择 / Tasks** 包含本地训练集和测试集全部 **9,474 个参考片段、9 类物体**。
在 **搜索全部任务** 中输入序列 ID、物体英文名或中文名，例如 `sub1_tripod_075`、`whitechair`、`衣架`。
支持多个关键词，也可以通过 **数据范围** 和 **物体** 下拉框筛选。
选择 **任务 / Task** 中的片段后会直接切换并开始执行，不需要刷新网页或重启命令。
匹配超过 200 项时显示前 200 项；继续输入关键词即可搜索到其余片段。

相同物体的任务复用场景，切换不同物体时重建对应场景，浏览器连接保持有效。
切换物体只替换场景和图表数据，不重建侧栏；选中的任务、搜索词、筛选条件和面板布局保持不变。
修改搜索或筛选时也保留当前选择，只有明确选择另一片段才会切换任务。
每次只向 GPU 加载当前选中的片段，全部任务共享启动时加载的同一份 policy。
**开始 / 重播** 可以重新执行当前片段，**暂停 / 继续** 可以控制执行过程。
关闭浏览器后默认暂停执行，重新连接可继续；`--viser-no-wait` 可关闭连接等待。
同一场景内的切换和重播复用初始化时采样的物理参数，观测噪声和随机扰动仍然保留。
重播次数不能直接当作独立随机场景下的成功率统计。

下面四类拾取任务以 ★ 标出，也可通过 **数据范围 → 关注拾取** 筛出，共 12 个片段：

| 任务 | `--sequence` | `--variant` 可选值 |
| --- | --- | --- |
| 三脚架地面拾取 | `sub1_tripod_075` | 0、1、2 |
| 衣架地面拾取 | `sub16_clothesstand_012` | 0、1、2、3、4 |
| 衣架放倒再拾起 | `sub9_clothesstand_039` | 0、1、2 |
| 白椅子桌面取下 | `sub7_whitechair_072` | 0 |

`--sequence` 和 `--variant` 设置启动时首先显示的片段，例如：

```bash
python scripts/reproduce/viser_pickups.py --sequence sub16_clothesstand_012 --variant 3
```

默认加载本机最近保存、修改时间超过 5 秒的 `*paper-reference*` 统一策略权重，
跳过正在写入的 checkpoint。侧栏显示当前权重名称，切换任务不会更换权重。
可通过 `--checkpoint /绝对路径/model_xxx.pt` 明确指定。

其他选项：

- `--device cuda:0`：选择 GPU；也可先设置 `CUDA_VISIBLE_DEVICES`。
- `--viser-port 8081`：更换服务端口；同时修改 SSH 转发两端端口。
- `--viser-no-wait`：立即开始执行，无需浏览器连接或点击启动。
- `--viser-exit`：评估结束后自动退出服务。
- `--viser-catalog all|train|test|pickups`：限制提供给浏览器的任务目录，默认 `all`。
- `--single-task`：使用原来的单片段评估，不显示任务选择器。
- `--original-scene`：使用统一策略训练时的原始场景，不添加白椅子的支撑桌。

默认只有 `sub7_whitechair_072` 使用根据初始姿态拟合的支撑桌，其余片段使用原始任务场景。
这张支撑桌是本地评估的附加几何，并非 OMOMO 发布的原始桌面，也不在此前统一策略训练场景中。
若需比较原训练场景表现，使用 `--original-scene`。

## 回合结束与结果

运行到参考动作末尾，或触发原有失败终止条件时，该片段的评估结束。
没有新增截短回合的播放时限，也没有改变原有 reward、失败阈值或成功判定。
浏览器随后保留**自动 reset 前的终止姿态**，侧栏显示成功与否、完成比例及终止原因。
`bad_contact` 等原因来自原任务终止项；`clip_end` 且无失败终止才计为完整成功。

任务选择模式的结果写入 `outputs/viser/<checkpoint名称>/tasks/<train或test>/<物体>/<片段名称>/metrics.json`。
`--single-task` 仍使用 `outputs/viser/<checkpoint名称>/<片段名称>/metrics.json`。
同一片段重播会覆盖该文件；切换时中断的回合不写入完整评估结果。
默认结束后继续保留场景，可以在侧栏重播或选择其他任务。按 `Ctrl+C` 退出服务。
如服务运行在后台的 `weave-viser` 会话，可用 `tmux attach -t weave-viser` 进入，
再按 `Ctrl+C` 停止当前服务。`Ctrl+B` 后按 `D` 可只退出 tmux 画面并保持服务运行。

## 原始 eval / play 接口

`scripts/rsl_rl/eval.py` 和 `play.py` 都新增了 `--viser`。
`eval.py` 额外支持 `--viser-tasks --num_envs 1`，开启上述搜索和任务切换界面。
例如，直接评估整个三脚架拾取文件的三个片段：

```bash
python scripts/rsl_rl/eval.py --viser --headless \
  --task G1-Inspire-HOI-Pickup-v0 \
  --config-dir ./configs/track --config-name eval_pickups \
  --checkpoint logs/rsl_rl/g1_inspire_hoi/2026-10-13_20-01-41_paper-reference-resumed/model_64500.pt \
  --output_dir outputs/viser/tripod-all
```

不传 `--num_envs` 时 eval 根据片段数创建环境。Viser 最多显示 12 个环境。
`play.py` 使用相同可视化参数并持续执行；有限片段指标请使用 `eval.py`。
`--viser` 与 `--video` 分开运行。旧版环境仍可通过原来的 `activate.sh` 执行原视频流程。

## 新版双卡断点续训

从之前的统一策略 `model_64500.pt` 继续，在 GPU 0、1 上联合训练至原定约 100,000 次迭代：

```bash
bash scripts/reproduce/resume_isaac62_gpu01.sh
```

该脚本要求两张卡空闲，不会停止已有任务；需要先关闭占用 GPU 的可视化服务。
每卡 2,048 个环境，共 4,096 个，训练数据仍为九类物体的完整训练集。
保留原 PPO、Muon / AdamW、SimBa 和 `torch.compile` 设置，恢复权重、归一化统计及优化器状态。
PPO 的自适应学习率标量同步为 checkpoint 中的优化器学习率。
同时修复优化器包装器加载后参数组引用失效的问题，确保自适应学习率实际更新 Muon / AdamW；
CPU 状态恢复检查记录在 `logs/preparation/isaac62-optimizer-resume-check.json`。
由于 Isaac Sim 已从 5.1 升级至 6.2，此阶段属于仿真版本改变后的继续训练。

默认 checkpoint 固定为 `model_64500.pt`；后续再次恢复时应明确传入新的 checkpoint：

```bash
bash scripts/reproduce/resume_isaac62_gpu01.sh /绝对路径/model_最新迭代.pt
```

最新启动目录记录在 `logs/preparation/isaac62-resume-job.path`；该目录包含 `launch.json`、
`train.log`、各 rank 日志和 `status`。模型与 TensorBoard 数据写入独立的
`logs/rsl_rl/g1_inspire_hoi/<启动时间>_paper-reference-isaac62-resumed/`。
新目录不覆盖旧 checkpoint，每 100 次迭代保存一次。

TensorBoard：

```bash
source scripts/reproduce/activate_isaac62.sh
tensorboard --logdir logs/rsl_rl/g1_inspire_hoi --host 127.0.0.1 --port 6006
```

本地 SSH 增加 `-L 6006:127.0.0.1:6006`，再打开 <http://localhost:6006>。

## 迁移说明与验证

Isaac Lab 3 使用 xyzw 四元数与新的状态数组接口。本项目在读取参考动作时转换四元数，
在算法入口转换状态数组，原始 NPZ 数据保持不变。
机器人使用此前 Isaac Sim 5.1 导出的 USD，保留原始 capsule、mimic joint 和关节顺序；
缓存位于 `data/isaac62/g1_inspire_usd`，来源及哈希见 `robot-assets.json`。
如迁移这份缓存，可执行 `python scripts/reproduce/prepare_isaac62_robot.py --source /原始robot_usd目录`。

验证记录：

- `logs/preparation/isaac62-orientation-check.json`：4,629 帧跨版本旋转、局部速度等计算对照。
- `logs/preparation/isaac62-viser-button-check.json`：真实 HTTP / WebSocket 连接，点击启动按钮并接收动态几何变换。
- `logs/preparation/isaac62-pickups-check.json`：四类任务分别加载统一策略完成 GPU eval 的检查结果。
- `logs/preparation/isaac62-train-smoke-check.json`：16 个环境、两次 PPO 迭代，完成 1,024 步采样、反向传播与 checkpoint 保存。
- `logs/preparation/isaac62-dependencies-check.json`：最终版本、Numba JIT 及两项明确的上游依赖覆盖。
- `logs/preparation/isaac62-viser-final-stream-check.json`：最终锁定环境中的三片段并行 eval 与浏览器通信检查。
- `logs/preparation/viser-task-catalog-check.json`：全量目录、关键词搜索与单片段提取校验。
- `logs/preparation/viser-task-switch-check.json`：真实 WebSocket 下的任务切换、重播、暂停及连接保持检查。
- `logs/preparation/viser-task-final-check.json`：最终服务的数据范围/物体筛选、运行中切换及重播检查。
- `logs/preparation/viser-stable-ui-check.json`：跨物体切换时控件、选择值、筛选与布局保持，及加载中的连续选择检查。

这些检查证明新版环境和可视化链路可运行。物理引擎升级可能改变轨迹与成功率；
它们不等同于完整复现论文指标，也不能用之前专项策略的成功数代替本次统一策略的结果。
