# Inspire → Revo 2 小样本数据处理

本流程使用官方 [dex-retargeting](https://github.com/dexsuite/dex-retargeting)，先转换四段拾取参考，再独立检查运动学和几何。现有训练、原始数据和 Inspire 机器人配置不切换。

工具位于 `/mnt/data/ruofei/zhangyu/code/LocoManipulation/repo/hand_retargeting/`，上游 checkout 在其 `dex-retargeting/` 子目录。固定 commit 为 `3f56141bc8bd2760d5e452e382937269554ebb21`。该目录的 `pyproject.toml`、`uv.lock`、`.venv` 管理独立 CPU 环境。

共享文件系统解压 PyTorch 的大量小文件很慢，因此 `.venv` 链接到本机 `/var/tmp/weave-hand-retargeting-3f56141bc8bd/venv`，与项目的 uv 配置配套管理。CPU PyTorch wheel 保存在工具目录的 `wheels/`，已按官方 SHA256 校验。该环境使用 CPU PyTorch 2.9.1、NumPy 2.4.6、Pinocchio 4.1.0，未更新训练环境的依赖。

## 首批数据

| 片段 | 原始划分 | 帧数 | 时长 |
| --- | --- | ---: | ---: |
| `sub1_tripod_075_v00` | train | 417 | 8.34 s |
| `sub16_clothesstand_012_v00` | test | 354 | 7.08 s |
| `sub9_clothesstand_039_v00` | train | 467 | 9.34 s |
| `sub7_whitechair_072` | train | 207 | 4.14 s |

每种方法处理全部 1,445 帧，保留 50 Hz 时序。时长按帧数除以 FPS 计，最后一帧时间比时长少一个采样间隔。`sub16` 始终标为 test，仅用于诊断，不并入训练集。

## 数据格式审计

原 NPZ 不包含关节名称。通过原始 USD 关节图、历史 Isaac Sim 5.1/6.2 运行记录和逐帧 FK，确认数据是固定关节合并后的广度优先、兄弟关节按名称排序，**不是 URDF 文件声明顺序**。原数据为 53 个运动关节、54 个刚体。

四段全部帧的原模型 FK 与保存位置最大差约 `1.31e-6 m`、旋转最大差约 `2.95e-6 rad`；mimic 残差小于 `1.91e-7 rad`。接触标签含义是 `+1` 期望接触、`-1` 避免接触、`0` 不约束。原始关节、刚体和接触数组保存在输出的 `source_*` 字段，四元数采用 wxyz。

证据：`outputs/revo2-retargeting/schema/{README.md,source_layout.json,clip_audit.json}`。

## 转换方法

- `position`：用官方位置优化器匹配两手十个指尖在机器人根坐标系的位置。
- `vector`：用官方向量优化器匹配掌根到指尖的向量。
- 两种方法均只优化双手 12 个主动关节，使用新手 10 个 mimic 关系。身体 29 个关节的位置及原始速度按核实后的关节名复制。
- 根据从动关节限位收紧主动关节范围；优化后平滑并约束相邻帧速度，同时满足主动和从动关节的速度限制。片段时长不改变。
- 另计算按旧／新关节范围线性缩放的基线，作为对照。
- 接触标签按对应手指语义映射，拇指合并时 `+1` 优先、全部 `-1` 才视为避免接触；这不等于重新证明了物理接触。

输出分别在 `data/revo2/pilot_position/` 和 `data/revo2/pilot_vector/`。每段有 NPZ、JSON，目录有总 manifest，记录来源哈希、上游 commit、方法、完整帧数和运行环境。

输出明确含 `preview_only=True`。它使用显式 `joint_names` 和 `body_names`：51 个运动关节、83 个完整 URDF link。固定关节帧没有冒充 PhysX 刚体，**不能把这些文件直接传给原 `G1-Inspire-HOI-v0` 训练入口**。训练接入还需要按实际导入的刚体布局转换、适配奖励和观测。

## 已测指尖位置误差

以下均为十个指尖、整段所有帧的平均欧氏距离，单位 mm；这些数值不是拾取成功率，也不代表接触稳定。

| 片段 | 线性角度映射 | Position | Vector |
| --- | ---: | ---: | ---: |
| 三脚架拾取 | 42.8 | 22.0 | 25.2 |
| 衣架地面拾取 sub16 | 43.4 | 23.8 | 26.5 |
| 衣架放倒再拾取 sub9 | 40.4 | 21.0 | 25.1 |
| 白椅子桌面取下 | 39.0 | 19.6 | 24.6 |

实际几何结果见 `outputs/revo2-retargeting/pilot_position/`、`pilot_vector/` 下的 `*.review.json`、指标图和对照快照。

在最终独立 CPU 环境复跑中，所有帧的优化器状态均为 `3`（满足函数收敛条件），没有负状态。Position 在速度投影前的平均误差仍为约 `18.4–23.7 mm`，投影后约 `19.6–23.8 mm`；主要残差不能仅归因于速度约束。每段 JSON 保留了投影前误差和优化器状态统计。

## 几何检查和全量门槛

检查工具在 CPU 上使用各机器人碰撞形状的凸包，以及原物体三角网格的 FCL BVH；不会把三脚架、衣架、椅子整体凸包化。固定连接及直接相邻刚体之间的自碰撞被排除。对象三角网格的 FCL 接触深度不作为可信的全局穿透量，因此物体接触采用碰撞布尔值和分离距离。

全帧检查数组、关节限位、速度和联动；碰撞、分离距离均匀抽查每段 32 帧。使用以下工程筛查阈值，它们不是论文指标：

- 新手关节位置和速度不得新增明显超限。
- 对应手指／掌部／身体组之间，相比源参考新增自碰撞深度超过 5 mm，需要修正。
- 原来期望接触、源手距离不超过 10 mm 的帧，新手若分离超过 20 mm，需要修正。
- 即使通过抽样几何检查，也需要全帧接触审查、PhysX 初始化和动态抓握验证后，才能称为训练数据。

目前保持 `full_dataset_promoted=False`。已观察到部分地面拾取帧接触丢失，以及新增手与膝部／髋部的碰撞，不能仅凭指尖误差降低就推广。

原参考本身也有问题，单独记录为 inherited：例如 sub16 的右膝存在明显位置、速度超限，速度约 `153.93 rad/s`，而原 URDF 速度上限为 `20 rad/s`，不能简单解释成安全的 ±π 包裹。旧轨迹在不开自碰撞的配置下生成；开启自碰撞后需要考虑身体、手臂和手腕的可行性。下一阶段应加入物体接触约束、受限腕部／手臂调整，并筛查或修复原有身体异常，再评估全量处理。

一个具体例子是 sub16 第 148 帧：所选凸包模型中，右掌根与右髋部重叠约 `122 mm`。掌根固定在腕部，它的位置不受手指角度影响，因此仅更换手指优化器无法消除这类问题。这个数值是离线碰撞几何诊断，不能直接当作实机或 PhysX 的穿透测量。

## 重跑与检查

```bash
source /mnt/data/ruofei/zhangyu/code/LocoManipulation/repo/hand_retargeting/activate.sh
cd /mnt/data/ruofei/zhangyu/code/Weave

# 默认处理上述四段完整 v00 参考，不覆盖原数据。
python scripts/reproduce/retarget_revo2.py --method position --output data/revo2/pilot_position
python scripts/reproduce/retarget_revo2.py --method vector --output data/revo2/pilot_vector

python scripts/reproduce/check_retargeted.py \
  --motion data/revo2/pilot_position/sub1_tripod_075_v00.npz \
  --samples 32 \
  --output outputs/revo2-retargeting/pilot_position/sub1_tripod_075_v00.review.json

python -m unittest discover -s tests -p test_retarget_dataset.py -v
```

`--max-frames` 仅用于调试，默认不截短片段。原始 NPZ 不改写，结果放在新的 `data/revo2/` 下。不要把 test 片段作为训练参考加入统一策略。

## Viser 对照回放

新增的 [Viser 数据集入口](VISER_DATASET.zh-CN.md) 可以在同一页面搜索 train / test 全部原始轨迹，并切换 Inspire、Revo2、两种手对照及 Position / Vector 方法：

```bash
source /mnt/data/ruofei/zhangyu/code/LocoManipulation/repo/hand_retargeting/activate.sh
cd /mnt/data/ruofei/zhangyu/code/Weave
python scripts/reproduce/viser_dataset.py --port 8083
```

以下单文件入口继续可用：

```bash
source /mnt/data/ruofei/zhangyu/code/LocoManipulation/repo/hand_retargeting/activate.sh
cd /mnt/data/ruofei/zhangyu/code/Weave
python scripts/reproduce/viser_retargeted.py \
  --motion data/revo2/pilot_position/sub1_tripod_075_v00.npz \
  --port 8082
```

本地电脑：

```bash
ssh -N -L 8082:127.0.0.1:8082 <服务器SSH别名>
```

打开 <http://localhost:8082>。可完整播放、逐帧检查、调整速度、切换视觉／碰撞网格或叠加两手。更换 `--motion` 可查看另三段。左侧为原始 Inspire、右侧为 Revo 2；这是参考数据回放，不是 policy eval，不显示虚构的成功率。
