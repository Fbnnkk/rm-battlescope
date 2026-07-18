# RM BattleScope

RM BattleScope 是面向 RoboMaster 机甲大师超级对抗赛（RMUC）的赛事数据解析、战术回放与策略研究工具。当前版本围绕 RMUC 2026 区域赛公开 SQLite 数据集构建，提供只读数据查询、比赛检索、轨迹重建、状态可视化和启发式战斗关系推断。

项目不会修改源数据库。所有回放、审计报告和导出文件默认写入本地 `outputs/`。

RM BattleScope 目前处于早期开发阶段。部分战斗关系和规则状态由规则、秒级遥测与事件组合推断，使用时请结合原始事件和比赛录像复核。欢迎通过 Issue 提交问题、异常案例和改进建议。

## 演示视频

[在哔哩哔哩观看 RM BattleScope 演示视频](https://www.bilibili.com/video/BV1UvKN6LENV/)

[在线体验：北部赛区第 90 场第 1 局静态回放](https://ezthor.github.io/rm-battlescope/replays/north-region-match-90-game-1/)

## 功能概览

### 稳定功能

- 按学校、赛区、赛程或 `game_id` 检索和浏览比赛。
- 对 SQLite 数据库执行只读查询，查看数据概览、表结构和逐秒状态。
- 读取秒级轨迹并完成基础清洗，保留原始坐标用于复核。
- 生成静态图和 HTML 交互回放，同步展示轨迹、血量、热量、朝向、发弹、增益、判罚、战亡与复活状态。
- 将查询结果导出为 CSV、JSON 或 JSONL。
- 批量审计越界、缺失、跳点、插值和清洗后残余异常，生成逐局复核报告。

### 实验功能

- 单帧跳点识别、短缺口插值和轨迹平滑。
- 根据发弹与受击窗口、阵营、口径、枪口朝向和距离匹配攻击关系。
- 展示累计经济、当前经济、科技核心装配和其他状态时间线，并合并临近的装配标记以减少遮挡。

### 规则推断

- 立即复活、英雄部署吊射和堡垒占领。
- 基地护甲展开、前哨站旋转、飞镖致盲、无人机反制，以及其他缺少原生字段的机制。

这些结果来自规则、遥测和事件的启发式组合，不等同于裁判系统确认结果。

回放同时支持按帧速播、1×/1.5×/2×/3×/5×实时播放和 `0.25×–8×` 无极调速。按秒播放使用浮点播放游标，可在高刷新率显示器上连续运行。

## 适用范围

本项目适合战术复盘、赛事数据浏览、特征工程和候选样本生成，不适合机器人实时控制、碰撞判断或毫米级路径分析。推断结果需要结合原始事件和比赛录像复核。

## 快速开始

1. 克隆仓库并进入项目目录。

```powershell
git clone git@github.com:ezthor/rm-battlescope.git
Set-Location rm-battlescope
```

2. 创建并激活 Conda 环境。

```powershell
conda env create --prefix .\.conda\envs\rmuc2026 --file environment.yml
conda activate .\.conda\envs\rmuc2026
```

3. 将下载的数据库保存为 `rmuc_2026_region_dataset/rmuc_2026_region_dataset.sqlite`。
4. 启动 Web 页面。

```powershell
python .\scripts\rmuc_web.py
```

5. 打开 <http://127.0.0.1:8765/>。

数据库的下载地址和文件名要求见下一节。

## 数据集准备

用于自动化与策略训练的 RMUC 2026 区域赛部分赛事数据已发布在 RoboMaster 论坛：

<https://bbs.robomaster.com/article/1936220>

从上述页面下载数据集后，将 SQLite 文件放在以下位置，并保持文件名一致：

```text
rm-battlescope/
└─ rmuc_2026_region_dataset/
   └─ rmuc_2026_region_dataset.sqlite
```

数据库、数据压缩包和规则手册 PDF 不包含在本仓库中，也不会被 Git 跟踪。若希望把数据库放在其他位置，可在命令行中使用 `--db <路径>`。

## 安装

推荐使用 Conda：

```powershell
git clone git@github.com:ezthor/rm-battlescope.git
Set-Location rm-battlescope
conda env create --prefix .\.conda\envs\rmuc2026 --file environment.yml
conda activate .\.conda\envs\rmuc2026
```

环境包含 Python 3.13、NumPy、Matplotlib 和 Pillow。SQLite、HTTP 服务和大部分查询能力使用 Python 标准库。

也可以使用已有 Python 3.13 环境安装依赖：

```powershell
python -m pip install -r requirements.txt
```

## 启动比赛浏览器

```powershell
python .\scripts\rmuc_web.py
```

打开 <http://127.0.0.1:8765/>，搜索学校或选择比赛，设置轨迹清洗参数后开始解析。生成完成后页面会自动进入交互回放。

可选参数：

```powershell
python .\scripts\rmuc_web.py --db D:\data\rmuc.sqlite --port 9000
```

## 命令行使用

查看数据库概览和结构：

```powershell
python .\scripts\rmuc_sqlite.py summary
python .\scripts\rmuc_sqlite.py schema
```

查询比赛、事件和逐秒状态：

```powershell
python .\scripts\rmuc_sqlite.py matches --school "学校名" --limit 10
python .\scripts\rmuc_sqlite.py events --game-id 1779323658229 --event-type 受击 --limit 20
python .\scripts\rmuc_sqlite.py timeseries --game-id 1779323658229 --robot-type 英雄 --start 60 --end 90 --limit 50
```

生成一局完整回放：

```powershell
python .\scripts\rmuc_trajectory.py --game-id 1779323658229
```

生成指定时间窗与兵种的回放和 GIF：

```powershell
python .\scripts\rmuc_trajectory.py --game-id 1779323658229 `
  --start 0 --end 120 --robot-type 英雄 --robot-type 步兵3 --gif
```

批量审计数据库中的全部比赛：

```powershell
python .\scripts\rmuc_trajectory_audit.py
```

## 数据与计算内容

公开数据主要由三张表组成：

| 表 | 粒度 | 主要内容 |
|---|---|---|
| `matches` | 每局一行 | 对阵、赛区、赛程、胜方、比赛时间与 `game_id` |
| `timeseries` | 每局、每秒、每实体 | 血量、位置、朝向、功率、热量、累计发弹、经济与易伤标志 |
| `events` | 每次事件一行 | 发弹、受击、装配、增益、能量机关、飞镖与无人机反制等事件 |

轨迹管线按以下顺序处理：

```text
SQLite 秒级状态
  → 比赛/阵营/机器人/时间窗切片
  → 场外点与缺失值标记
  → 单帧定位跳点识别
  → 短缺口插值
  → 连续段加权平滑
  → 残余超速段断线
  → events 时间对齐
  → HTML 回放、静态图与质量报告
```

场地坐标按官方 `28 m × 15 m` 尺寸映射。背景画布以停机坪和有效场地的内侧角点标定，不把外围挡板计入坐标范围。堡垒占领优先使用原始坐标，轨迹绘制使用清洗坐标。

攻击关系、立即复活、部署吊射、堡垒占领和基地护甲展开中有部分状态无法从数据集原生字段直接取得。项目会在界面和报告中标注推断置信度；这些结果适用于战术复盘、特征工程和候选样本生成，不等同于裁判系统确认结果。

## 输出目录

```text
outputs/
├─ trajectories/<game_id>/
│  ├─ trajectory.html
│  ├─ trajectory.png
│  └─ quality_report.json
├─ web_replays/
└─ trajectory_audit/
```

`outputs/` 已加入 `.gitignore`。原始数据、查询导出和生成回放不会进入版本库。

## 工程结构

```text
rmuc_trajectory/                 核心解析、规则推断与渲染模块
rmuc_web/                        比赛选择前端页面
scripts/rmuc_web.py              本地 Web 服务入口
scripts/rmuc_sqlite.py           SQLite 只读查询工具
scripts/rmuc_trajectory.py       单局轨迹与回放生成器
scripts/rmuc_trajectory_audit.py 全库质量审计入口
docs/                            规则、字段映射和轨迹处理说明
tests/                           单元测试
assets/                          回放所需场地画布
environment.yml                 Conda 环境定义
```

进一步说明：

- [规则与数据索引](docs/RMUC_2026_规则与数据索引.md)
- [轨迹解析与标定说明](docs/RMUC_2026_轨迹解析说明.md)

## 测试

```powershell
python -m unittest discover -s tests -v
```

## 已知边界

- 数据是秒级遥测，不适用于控制环、碰撞或毫米级路径分析。
- 数据允许缺失值，且大表没有数据库级主键和外键约束。
- 全国赛 V2.x 规则晚于区域赛数据发生时间，不能直接用于解释全部区域赛历史机制。
- 场地图是规则手册俯视渲染图，不是测绘底图。
- 规则推断结果应结合原始事件、质量报告和比赛录像复核。

## 参与贡献

欢迎提交无法正确解析的比赛案例、数据字段解释修正、推断规则反例、可视化改进，以及测试和文档改进。建议先通过 Issue 描述问题和复现条件，再提交 Pull Request（合并请求）。

## 许可证

代码以 [MIT License](LICENSE) 开源。场地图等第三方材料不在 MIT 授权范围内，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。RoboMaster 及相关名称和材料的权利归其各自权利人所有。
