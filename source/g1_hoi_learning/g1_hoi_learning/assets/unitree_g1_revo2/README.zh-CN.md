# G1 + BrainCo Revo 2 双手模型

该目录保存官方资源和装配结果。沿用 Weave 的 G1 rev 1.0 身体，移除原 Inspire 双手，装上 BrainCo 当前官方左右手及宇树腕部连接件。现有 Inspire 任务、checkpoint 和正在运行的训练不切换模型。

## 文件

| 文件 | 用途 |
| --- | --- |
| `g1_29dof_rev_1_0_with_revo2.urdf` | 标准装配模型，保留 URDF mimic 联动定义 |
| `g1_29dof_rev_1_0_with_revo2_software_mimic.urdf` | Isaac 软件联动控制入口，去掉 mimic 标签，避免导入器约束与软件目标重复 |
| `assembly.json` | 腕部安装、主动关节、联动比例、限位和初始姿态 |
| `sources.json` | 官方仓库、固定 commit、逐文件 Git blob 和 SHA256 校验值 |
| `validation.json` | CPU 结构、坐标、限位、凸包碰撞和独立解析器检查结果 |
| `vendor/brainco/` | BrainCo 原始 URDF、引用的 visual/collision STL、上游说明 |
| `vendor/unitree/` | 宇树整机参考 URDF、左右腕部连接件及 LICENSE |

模型的 G1 网格通过相对路径引用相邻的 `../unitree_g1/meshes/`，Revo 2 和连接件引用本目录 `vendor/`。移动模型时需要保留这些相对目录关系。

## 来源与安装对齐

- BrainCo：[brainco-description](https://github.com/BrainCoTech/brainco-description)，固定 commit `6db7e2ed1d864ee0a3ab6ff62d43a303787137f6`。采用 `revo2_system` 的当前左右手模型，不使用旧版指尖关节。
- 宇树：[g1_with_brainco_hand](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_with_brainco_hand)，固定 commit `5994d4faef0a9cadd3287f8de0199a67eeb2a259`。参考腕部安装关系并保留连接件的几何、质量和惯量，不继承该旧模型的手指关节定义。
- G1 身体：相邻 `unitree_g1/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf`。保留身体的所有 link、joint、质量、惯量、碰撞和驱动参数；仅重写网格相对路径。

| 安装 | 父 link | 平移 xyz / m | 旋转 rpy / rad |
| --- | --- | --- | --- |
| 左掌根 | `left_wrist_yaw_link` | `0.0591 0 0` | `-1.570796326 0 -1.570796326` |
| 右掌根 | `right_wrist_yaw_link` | `0.0591 0 0` | `1.570796326 0 1.570796326` |
| 左右连接件 | 各侧 `wrist_yaw_link` | `0.0415 0 0` | `0 0 0` |

连接件与掌根均固定在腕部，遵循宇树原始装配方式。已核对两套模型的腕部 roll/pitch/yaw 安装坐标，以及新旧掌根下各手指根部坐标。最大锚点误差约 `2.76e-7 m`，来自上游数值精度差异。

当前官方 URDF 每手质量合计约 `0.390 kg`，每侧连接件 `0.02280635 kg`。这些是模型中的数值，实机参数仍需按具体硬件校准。

## 关节与控制

- 身体：29 个独立关节。
- 双手：22 个运动关节，12 个独立控制关节、10 个 mimic 从动关节。
- 整机：83 个 link、82 个 joint，其中 51 个运动关节、41 个独立控制量。
- 每手控制：拇指 metacarpal、拇指 proximal，以及食指、中指、无名指、小指 proximal。
- 拇指 distal = proximal × `1.0`；其他四指 distal = proximal × `1.155`，均无偏移。

新增的 `robots/g1_revo2.py` 提供 `G1_REVO2_CFG`、`REVO2_ACTIVE_JOINTS` 和 `REVO2_MIMIC`。它采用 URDF 中的 effort/velocity 限位，手部初始 PD 为 `stiffness=5.0, damping=0.5, armature=0.01`；这些是仿真候选参数，尚未做动力学标定。使用软件联动 URDF 时，控制端必须通过 `MimicJointPositionAction` 或等价逻辑写入从动关节目标。

新配置在导入器设置 `self_collision=True`，并设置 articulation 的 `enabled_self_collisions=True`。不会把该配置自动替换到现有 Inspire 训练任务。

## 碰撞检查的实际结果

CPU 检查使用 FCL，对每个碰撞几何取凸包，排除固定连接的同一刚体及相邻运动刚体；这不是 PhysX 运行验证，也不测量接触力。

- 共检查 78 个碰撞定义、6 组姿态。
- 全零张手姿态：左掌与拇指凸包有约 `0.202 mm` 重叠。
- 身体默认站姿、手部控制范围的 25% 和 50%：未检出上述规则下的非相邻凸包穿透。
- 全部手指同时到 75% / 100%：存在拇指与其他手指的接触或穿透，最大约 `18.5 mm`。单关节限位的笛卡尔积并不保证整手姿态无碰撞。

因此新配置以 25% 轻微屈指作为初始姿态，并按照联动关系初始化全部从动关节。保留相关碰撞，不通过删除手部自碰撞来掩盖几何问题。Viser 查看器只显示运动学姿态，滑块可进入穿透姿态；实际闭合行为还需要在 PhysX 中检查。

## 重建和检查

在 Weave 根目录执行：

```bash
# 首次下载；固定来源版本并校验每个文件，然后组装。
python scripts/reproduce/prepare_revo2.py

# 已有资源时离线重新组装。
python scripts/reproduce/prepare_revo2.py --offline

# 使用现有 .venv 完成基础 CPU 检查。
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/reproduce/check_revo2.py
```

本次额外检查工具安装在 `outputs/revo2-tools/`，没有更改训练 `.venv` 的包：

```bash
uv pip install --python .venv/bin/python --target outputs/revo2-tools --no-deps \
  python-fcl==0.7.0.11 yourdfpy==0.0.60 \
  --index-url https://mirrors.aliyun.com/pypi/simple/

PYTHONPATH=outputs/revo2-tools OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/reproduce/check_revo2.py --contacts --independent-parser
```

## CPU Viser 查看

```bash
cd /mnt/data/ruofei/zhangyu/code/Weave
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  environments/isaac62/.venv/bin/python scripts/reproduce/viser_revo2.py --port 8081
```

本地电脑另开终端：

```bash
ssh -N -L 8081:127.0.0.1:8081 <服务器SSH别名>
```

浏览器打开 <http://localhost:8081>。可选择身体零位/站姿、分别控制两手各 6 个主动关节、切换 visual/collision 网格和显示 link 坐标轴。该查看器不加载 Isaac Sim、不使用 CUDA，也不执行训练 policy。

## 接入 Weave policy 的边界

本次完成资产下载、装配、坐标对齐、独立机器人配置和 CPU 资产查看。尚未将现有动作参考、手部 link 索引、奖励及接触标签重定向到 Revo 2，未注册新的 HOI 训练任务，也未开始新的训练。已有 Inspire checkpoint 不能因为控制量同为 41 就直接视为兼容。

物理验证需检查实际 USD 的关节、联动、过滤关系、接触稳定性和闭合过程，再决定 Revo 2 的训练环境数量。新手和自碰撞的显存容量不能直接由旧训练外推。

上游资源版权归各作者所有。宇树 LICENSE 随文件保留；BrainCo 当前资源 README 中的许可说明尚待上游补充，没有将旧版 ROS 包的 Apache 许可套用到新资源。
