# 论文统一策略训练 recipe（2026-10-19）

## 作者确认与配置差异

作者在 [issue #2](https://github.com/xiaohu-art/Weave/issues/2) 中确认使用 **8 × 16,384 个并行环境**，
即 **131,072 个环境**，并建议增加仿真并行数量以改善训练平台期。
此前本机双卡每卡 2,048、总计 4,096 的训练只有该规模的 **1/32**。
公开 `train.yaml` 的 4,096 默认值不能作为论文实际训练规模的依据。

本地将该规模实现为 **8 个 DDP 进程，每进程一张 GPU、16,384 个环境**，共同训练同一个策略。
这是启动器的实现方式；issue 没有明确作者的 GPU 型号或硬件拓扑。
issue 新确认的是并行规模，其余训练参数继续采用公开配置，不能据此声称已获得作者未公开的全部设置。

| 项目 | 当前论文配置 |
| --- | --- |
| 配置 / 启动脚本 | `train_paper.yaml` / `train_paper.sh` |
| 数据 | `data/train/` 九个完整文件，7,869 个片段，单个统一策略 |
| 初始化 | 默认从头训练；不加载此前的小规模 `model_300.pt` 或问题权重 |
| 并行环境 | 8 × 16,384 = 131,072 |
| 每环境 rollout 长度 | 32 步 |
| 每次迭代的全局采样量 | 4,194,304 条 transition |
| PPO epochs / mini-batches | 5 / 4（各进程独立分批、DDP 同步梯度） |
| 每个 mini-batch 对应的全局样本数 | 1,048,576 |
| 迭代 / 保存间隔 | 100,000 / 100 |
| 初始学习率 / 调度 / KL | `1e-4` / adaptive / `0.01` |
| 网络、优化器、奖励、终止、RSI、编译 | 保留公开 SimBa、Muon / AdamW 与原配置；保留已完成的兼容性 bug 修复 |

没有用增加 rollout 长度、累积梯度或增加迭代数代替并行规模。
不同并行规模的训练不应直接声称是同一 recipe；此前低成功率同时存在已核实的实现问题，不能仅归因于环境数量。

## 查看配置与启动

以下命令只打印计划，不激活 Isaac、不创建 GPU 训练任务：

```bash
cd /mnt/data/ruofei/zhangyu/code/Weave
bash scripts/reproduce/train_paper.sh --dry-run
```

准备好 **8 张空闲 GPU** 后正式启动：

```bash
bash scripts/reproduce/train_paper.sh
```

默认选择物理 GPU `0,1,2,3,4,5,6,7`；用 `WEAVE_GPU_INDICES` 指定其他八个不同编号。
启动前检查设备是否存在、Weave 锁和已有 GPU 占用，条件不满足就退出，不停止其他任务，也不自动减少环境数。
当前服务器仅可见物理 GPU 0、1，因此不能在本机按此启动器运行完整八卡配置。
recipe 修改阶段没有启动训练；后续按用户要求执行的双卡容量 smoke 结果见文末。

默认激活 `scripts/reproduce/activate.sh`，使用保留的 Isaac Sim **5.1.0** / Python **3.11**，
Isaac Lab 固定提交为上游 `install.sh` 指定的 `e17312889676ed229b986d56c9e0b23a01cf0ab7`。
不修改系统驱动；已安装的 Isaac Sim 6.2 / Viser 环境继续保留供可视化等使用。
若显式选择以下环境，属于软件版本不同的实验，应在结果中注明：

```bash
WEAVE_TRAIN_ACTIVATION=scripts/reproduce/activate_isaac62.sh \
  bash scripts/reproduce/train_paper.sh --dry-run
```

## 断点恢复与记录

训练入口增加 `--paper_recipe` 校验：检查 8 个进程、每进程 16,384 环境、32 步 rollout、
5 epochs / 4 mini-batches、完整 D_train、RSI、统一任务和 100,000 次目标迭代。
启动器自动携带该选项；不要仅在少数卡上加载 `train_paper.yaml` 后称为论文规模。

每次运行在 checkpoint 目录写入 `training_recipe.json`，记录实际解析后的规模、参考文件、软件版本、
初始迭代及断点路径。严格模式仅接受同规模 recipe 记录的断点，不接受旧 4,096 环境权重：

```bash
bash scripts/reproduce/train_paper.sh --resume \
  --load_run /绝对路径/同规模训练目录 --checkpoint model_10000.pt
```

恢复时保留已修复的优化器学习率逻辑，总迭代目标仍为 100,000，不额外增加 100,000 次。
checkpoint 文件的迭代编号从 0 开始，恢复从其下一次迭代继续。
训练日志位于 `logs/training/latest-paper/`；模型和 TensorBoard 位于对应的
`logs/rsl_rl/g1_inspire_hoi/<时间>_paper-issue2-full/`。

## 双卡缩小实验

`train_reference_dual.yaml` / `train_gpu01.sh` 现在明确采用 **2 × 16,384 = 32,768** 个环境，
为作者并行规模的 **1/4**，默认名称 `reference-scaled-dual`。它不是完整论文 recipe。
可用以下命令查看，不启动训练：

```bash
bash scripts/reproduce/train_gpu01.sh --dry-run
```

启动器默认要求所选 GPU 空闲。明确接受共享算力时，可显式指定：

```bash
WEAVE_ALLOW_GPU_SHARING=1 WEAVE_MIN_FREE_GPU_MIB=35000 \
  bash scripts/reproduce/train_gpu01.sh
```

共享模式保留已有进程，仍使用 Weave 锁、UUID 设备限制及空闲显存门槛；
会竞争算力，35,000 MiB 门槛不保证训练峰值显存足够。
本机本次共享启动的授权、启动器和实测记录见 `outputs/training-starts/2026-10-19/`。

旧 `resume_isaac62_gpu01.sh` 仍供双卡诊断续训使用，默认加载旧 `model_300.pt`，
运行名为 `reference-scaled-dual-isaac62-corrected`。
该路径同时存在旧小规模预训练、并行规模改变和仿真版本改变，不作为论文从头训练的起点。

上述 recipe 修改阶段仅执行静态检查；后续用户明确要求容量 smoke，实测记录如下。

## 本机共享 GPU 的容量 smoke（2026-10-19）

每档均使用完整九物体 D_train、原网络、`torch.compile`、32 步 rollout、5 epochs / 4 mini-batches，
从头运行最多 5 个训练迭代；已有任务约占 7 GiB/卡，持续保留。

| 每卡环境 | 双卡总量 | 完整训练迭代 | 观测到的最小空闲显存 | 结论 |
| ---: | ---: | ---: | --- | --- |
| 8,192 | 16,384 | 5 | 约 5.06 / 5.18 GiB | 推荐 |
| 10,240 | 20,480 | 5 | 约 0.97 / 0.99 GiB | 能跑，但共享时余量偏小 |
| 12,288 | 24,576 | 0 | OOM | 不采用 |
| 16,384 | 32,768 | 0 | OOM（前一次启动） | 不采用 |

推荐当前共享条件下使用 **每卡 8,192**。8,192 档的 checkpoint 浮点张量全部有限，
模型从第 0 到第 4 次迭代有实际更新。此规模为作者总并行量的 **1/8**，不能作为相同规模的论文复现。
所有 smoke 进程已退出，显存恢复至测试前水平。
逐档日志、显存采样、checkpoint 及核对记录见
[容量测试报告](../../outputs/capacity-smoke/2026-10-19_18-58-49/README.md)。

正式训练仍需显式指定本机验证过的环境数量，默认双卡配置中的 16,384/卡属于上文的缩小配方声明，
在本机共享条件下已经证实会 OOM。以下命令使用本次实测的 5.1 激活方式，并从头训练 100,000 次迭代：

```bash
cd /mnt/data/ruofei/zhangyu/code/Weave
WEAVE_TRAIN_ACTIVATION=outputs/training-starts/2026-10-19/activate_sim51_driver.sh \
WEAVE_ALLOW_GPU_SHARING=1 WEAVE_MIN_FREE_GPU_MIB=36000 \
WEAVE_TRAIN_ENVS_PER_GPU=8192 \
  bash scripts/reproduce/train_gpu01.sh
```

本轮仅执行容量 smoke，没有自动开启以上正式训练。Isaac 6.2 下的峰值显存需要另外验证。

## 随后正式启动：每卡 10,240

2026-10-19 19:28 用户选择每卡 **10,240**、双卡总计 **20,480** 的正式训练，
并明确允许停止已有训练。两个 RoboNaldo 训练及其自动调度器已停止，原日志和 checkpoint 保留。
本轮从头训练完整九物体 D_train，目标 100,000 次迭代，沿用上述 5.1 环境和公开网络 / PPO 参数。

此前 10,240 smoke 的约 1 GiB 余量来自两卡仍有其他任务共用的条件；
本轮独占两卡，已开始实际 PPO 更新，当前观测空闲显存约 8 GiB/卡。
此规模是作者确认总并行量的 5/32。

启动器、任务路径、进程与 checkpoint 验证见
[正式训练记录](../../outputs/training-starts/2026-10-19/dual10240/README.md)。
查看日志：

```bash
tail -f logs/training/latest-dual/train.log
```
