# 统一策略的论文成功率评估

本轮权重为
`logs/rsl_rl/g1_inspire_hoi/2026-10-15_19-21-47_paper-reference-isaac62-resumed/model_99999.pt`。
迭代从 0 编号；最后一次为 99999，对应完成 100,000 次迭代，训练任务状态为 `COMPLETED`。
评估不会继续训练或修改 checkpoint。
该历史权重使用总计 4,096 个环境，并非作者在 [issue #2](https://github.com/xiaohu-art/Weave/issues/2)
后来确认的 8 × 16,384 环境。还存在已核实的训练 bug，因此本轮结果不能作为相同 recipe 的复现结论。
新的训练配置见 [论文训练 recipe](PAPER_TRAINING.zh-CN.md)。

## 对照目标

按[论文 V-B](https://arxiv.org/html/2609.16683v1#S5.SS2)使用同一个训练集策略：

| 评估 split | 参考片段数 | 论文成功率 SR | 论文完成比例 PR |
| --- | ---: | ---: | ---: |
| train | 7,869 | 92.5% | 96.3% |
| test | 1,605 | 65.0% | 84.5% |

95.3% 属于另一项直接在 test split 上训练的实验。本次权重不属于该实验，不能将其作为本轮目标。
九物体分别评估，再按片段数汇总；不会把九个物体的成功率简单平均。

## 已启动的任务

```bash
cd /mnt/data/ruofei/zhangyu/code/Weave
cat outputs/paper-eval/2026-10-19_model99999_gpu/status.json
cat outputs/paper-eval/2026-10-19_model99999_cpu-diagnostic/status.json
```

- `weave-paper-eval-gpu`：按用户要求改为在 GPU 1 与已有训练共享运行。
  每个物体启动前要求至少 16 GiB 空闲显存；保留已有训练进程，共享算力可能降低其速度。
  CUDA 通过 UUID 只开放物理 GPU 1，因此进程内及结果中的 `cuda:0` 对应物理 GPU 1；
  各目录 `command.json` 和主 `status.json` 记录实际物理 GPU 编号。
- CPU 诊断已停止，已完成的数据保留在 CPU 输出目录。之前最多三个 CPU 评估进程并行，
  每进程 4 个计算线程、nice 19；正式结果以 GPU 评估为准。

CPU 模式仍需要为 Isaac Lab 的 pinned-memory 分配建立 CUDA 上下文，每进程约占 458 MiB 显存，
没有使用 GPU PhysX 或 GPU policy 推理。CPU / GPU PhysX 的接触求解可能不同，CPU 结果会单独标识。

任务通过 tmux 保持运行。查看控制器日志：

```bash
tail -f logs/preparation/paper-eval-gpu-runner.log
tail -f logs/preparation/paper-eval-cpu-runner.log
```

每完成一个物体，就更新输出目录中的 `README.md` 和 `summary.json`。
`complete: false` 表示只完成部分物体，不能将此时的数值当作完整 split 成功率。
各物体目录保存 `metrics.json`（逐片段数据）、`eval.log`、`command.json`。

## 评估口径与差异

沿用[作者 eval 配置](https://github.com/xiaohu-art/Weave/blob/main/configs/track/eval.yaml)：
每个片段一个环境，使用第 0 帧初始化，关闭随机起始帧 RSI；保留配置原有的物理随机化、观测噪声和推扰。
随机种子固定为 42。论文没有明确给出多 seed 重复次数，本轮先评估完整 split 一次。
使用 `G1-Inspire-HOI-v0` 原始场景，不添加拾取专项的支撑桌。

成功要求到达参考片段末尾且没有失败终止。失败阈值与作者 YAML 相同：
骨盆位置 0.25 m、物体位置 0.30 m、骨盆 / 物体投影重力差 0.8 / 0.3、腕踝垂直误差 0.25 m，
预期手部接触连续丢失 10 帧也会终止。没有放宽这些条件。

评估统计的修正：

1. 完成与失败同时触发时按失败；原脚本仅使用 timeout mask，可能误记成功。
   `success_rate_official_legacy` 保留原统计，`simultaneous_clip_end_and_failure` 记录差异片段数。
2. 训练集有 21 个参考片段长于默认 20 秒；对应物体的评估时限延长到最长片段加 5 帧，允许完整执行。
3. 跟踪误差保留原来的按帧加权结果，另提供论文定义的按片段平均结果。

本地训练中途从 Isaac Sim 5.1 / Isaac Lab 2.3 迁移至 6.2 / 3.0。
这属于与论文原始软件环境的差异；本轮用于检验当前已训练策略，不能声称软件环境完全一致。

## 已完成的 CPU 测试集诊断

seed 42、全部 1,605 个 test 片段已完成：6 个成功，SR 为 **0.3738%**，PR 为 **45.66%**。
原统计与修正后的成功数一致；差距不是同时触发完成 / 失败的统计修正造成的。
这与论文测试集目标相差很大，但 CPU 数据不能作为最终 GPU 复现结论。

18 个参考文件已重新校验 SHA256，全部通过；checkpoint 可正常加载，权重均为有限值。
训练结束前的平均训练回合长度为 17.70 步，升级前日志末尾为 85.86 步。
这些是不同训练阶段的日志，不是受控对比，不能据此直接断言升级是原因。
原始检查保存在输出目录的 `dataset-current-check.json`、`checkpoint-check.json`、
`training-before-after.json`、`protocol-audit.json`。

## 复跑

下面的命令可以复用已完整生成的逐物体结果；会校验 checkpoint 哈希、seed、版本、片段名和完成状态。
已有输出的设置与本次命令不一致时会报错，避免混合结果。

```bash
source scripts/reproduce/activate_isaac62.sh
python scripts/reproduce/evaluate_paper.py --share-gpu --gpu-indices 1 --min-free-gib 16 \
  --checkpoint logs/rsl_rl/g1_inspire_hoi/2026-10-15_19-21-47_paper-reference-isaac62-resumed/model_99999.pt \
  --output outputs/paper-eval/2026-10-19_model99999_gpu
```

不传 `--share-gpu` 时仍使用等待空闲 GPU 的默认模式。
CPU 诊断可移除共享 GPU 选项，增加 `--device cpu --workers 3`，并将输出改为 CPU 目录。
并行进程不会拆分或重抽样片段，
每个物体仍然以一个环境对应一个片段的方式评估。

同一目录不允许并发运行两个控制器。GPU 空闲检查不能阻止其他用户之后启动新任务；
它不会暂停、重启或修改其他训练。
