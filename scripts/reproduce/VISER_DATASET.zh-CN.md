# Viser 数据集轨迹回放

从浏览器选择 train / test 数据中录制好的完整轨迹，查看 G1 + Inspire 或 G1 + Revo2，并可对同一段动作进行双手对照。

## 启动

使用已经配置好的 uv CPU 环境：

```bash
source /mnt/data/ruofei/zhangyu/code/LocoManipulation/repo/hand_retargeting/activate.sh
cd /mnt/data/ruofei/zhangyu/code/Weave
python scripts/reproduce/viser_dataset.py --port 8083
```

在本地电脑建立 SSH 转发：

```bash
ssh -N -L 8083:127.0.0.1:8083 <服务器SSH别名>
```

然后打开 <http://localhost:8083>。默认监听服务器的 `127.0.0.1`；启动日志会打印实际端口。停止时在服务器终端按 `Ctrl+C`。

已有 `viser_pickups.py` 入口也支持：

```bash
python scripts/reproduce/viser_pickups.py --dataset --viser-port 8083
```

## 界面操作

1. 在「手型 / Hand」选择 Inspire 原始手、Revo2 重定向手或两种手对照。
2. 用「数据范围」选择 train / test；可以继续按物体、序列 ID 或物体中英文名搜索。Revo2 支持 Position / Vector 方法。
3. 在「轨迹 / Motion」选择片段后会自动加载；「加载 / 重播」可重新加载已选片段。
4. 点击「播放」，可调速度、循环播放，或暂停后拖动帧滑条／点击上一帧和下一帧。状态栏显示完整帧数、FPS、时长及真实来源文件。
5. 「显示」中可以切换视觉／碰撞网格；对照模式支持左右并排或叠加显示。「重置视角」会重新对准当前轨迹范围。

筛选条件不会截短轨迹。匹配超过 200 项时，继续输入关键词可搜索整个目录。

## 当前可用数据

| 手型 | train | test | 数据来源 |
| --- | ---: | ---: | --- |
| Inspire | 7,869 | 1,605 | `data/train/*.npz`、`data/test/*.npz` |
| Revo2 / Position | 3 | 1 | `data/revo2/pilot_position/` |
| Revo2 / Vector | 3 | 1 | `data/revo2/pilot_vector/` |

以上是当前本地数据的数量，启动时会重新扫描。Revo2 和双手对照只列出已经实际转换的片段：`sub1_tripod_075_v00`、`sub16_clothesstand_012_v00`、`sub9_clothesstand_039_v00`、`sub7_whitechair_072`。其中 `sub16_clothesstand_012_v00` 保持为 test。

两种手对照读取重定向 NPZ 中保存的原始 Inspire 轨迹和 Revo2 轨迹，保持同一物体和帧时间。后续按当前格式生成新的重定向文件并更新对应 `manifest.json` 后，重启查看器即可纳入目录。

## 指定初始片段

```bash
# 原始 Inspire 测试集
python scripts/reproduce/viser_dataset.py \
  --hand inspire --split test --sequence sub16_clothesstand_012_v00 --port 8083

# Revo2 Vector，并排对照两种手
python scripts/reproduce/viser_dataset.py \
  --hand both --method vector --sequence sub1_tripod_075_v00 --port 8083
```

`--sequence` 可以是完整片段名或搜索关键词。界面中仍可更换手型、划分和片段。更多参数见 `--help`。

## 回放含义与资源使用

这是数据集中保存的机器人／物体姿态回放；任务执行效果需要另外运行 policy eval。Revo2 当前为重定向预览数据，界面显示其离线检查状态，详情见 [手部重定向报告](HAND_RETARGETING.zh-CN.md)。数据中未记录的桌子等支撑场景不会额外推测生成。

查看器只使用 CPU，不启动 Isaac Sim 或 policy。目录只读小型元数据；切换到某片段时，用固定大小的数据块读取其姿态，缓存到 `outputs/viser/dataset_replay/clips/`，不改写原始数据。大型压缩文件中的片段首次读取可能需要等待，后续会使用缓存。
