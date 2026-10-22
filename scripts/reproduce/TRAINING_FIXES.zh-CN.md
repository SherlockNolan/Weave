# 训练兼容问题修复（2026-10-19）

## 修复内容

1. **物体与参考动作保持一致。** Isaac Lab 3 显式采用轮换克隆策略，环境 `i` 对应物体 `i % O`。
   场景创建后逐一核验所有物体的实际 USD 来源；任何错配都会终止启动。
   拾取任务的物体与支撑桌变体也成对选择，避免新版克隆器独立组合出错误桌面。
2. **断点恢复后学习率实际生效。** 恢复后刷新子优化器参数组引用，并从实际 Muon / AdamW 参数组恢复 PPO 学习率标量。
   自适应学习率直接写入子优化器；恢复到不同学习率或无效学习率时明确报错。
3. **保留原接触定义。** 接触观测、接触奖励、接触丢失终止及足部滑动项统一读取法向接触力。
   Lab 3 使用 `normal_force_matrix_w` / `net_normal_forces_w_history`，Lab 2 保留原字段。
   不再把新版接口中额外包含的摩擦力代入旧公式。
4. **固定物理参数。** `enable_external_forces_every_iteration=False` 与旧版配置保持一致。

网络结构、奖励权重、终止阈值、数据集和正式训练的 `torch.compile` 设置保持原配置。
这些修复不能自动恢复此前训练权重的质量，也不构成达到论文成功率的证明。

## 验证

CPU 回归覆盖克隆布局、支撑桌配对、全部环境的启动校验、接触力含义，以及真实 Muon / AdamW 恢复后的参数更新。
恢复后继续更新的参数与不中断更新的参数逐元素一致。

```bash
source scripts/reproduce/activate_isaac62.sh
python -m unittest discover -s tests -v
```

小场景验证脚本从实际数据中各取一个完整片段，检查每个环境的 USD 来源、真实接触力接口和多步物理计算。
它不会更新策略或修改原始数据。请在已选择的 GPU 上运行：

```bash
python scripts/reproduce/check_training_compat.py --headless --device cuda:0 \
  --num-envs 19 --output outputs/compat/unified.json \
  --config-dir ./configs/track --config-name train_reference

python scripts/reproduce/check_training_compat.py --headless --device cuda:0 \
  --task G1-Inspire-HOI-Pickup-v0 --num-envs 19 \
  --output outputs/compat/pickups.json \
  --config-dir ./configs/track --config-name train_pickups_dual
```

本机此次验证结果写入 `outputs/bugfix-checks/2026-10-19/`，汇总见该目录 `README.md`。
短 PPO 续训仅用于验证采样、反向传播、checkpoint 保存和学习率恢复；不能将它当作新策略成功率评估。

## 后续完整训练

此前的 `model_64500.pt` 已经历优化器恢复问题，后续 `model_99999.pt` 还经历了物体错配训练。
它们继续保留供故障对照；不要把它们当作修复后的干净基线。

根据作者 [issue #2](https://github.com/xiaohu-art/Weave/issues/2) 的回复，训练规模改为 **8 × 16,384** 个环境。
论文 recipe 默认从头训练，沿用这些兼容性修复，具体参数和环境见 [论文训练 recipe](PAPER_TRAINING.zh-CN.md)。
以下命令仅打印计划，正式训练需要八张空闲 GPU：

```bash
bash scripts/reproduce/train_paper.sh --dry-run
```

旧双卡恢复脚本仅作为缩小实验，默认使用首次受影响恢复前保存的
`logs/rsl_rl/g1_inspire_hoi/2026-10-13_17-04-51_paper-reference-dual/model_300.pt`：

```bash
bash scripts/reproduce/resume_isaac62_gpu01.sh
```

双卡现为每卡 16,384、总计 32,768 环境，仅为作者规模的 1/4；
运行名为 `reference-scaled-dual-isaac62-corrected`，不能把旧小规模 checkpoint 续训作为论文从头训练。
启动脚本保留 GPU 空闲检查；不会暂停其他任务。本次没有自动启动训练。

修复前问题的证据见 `outputs/code-audit/2026-10-19/README.md`，旧评估结果保持为历史记录。
