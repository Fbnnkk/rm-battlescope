# RM BattleScope

面向 RoboMaster 机甲大师超级对抗赛（RMUC）的比赛数据浏览、战术回放与个人战绩评分工具。本阶段正式开源版本为 **v1.0.0**，自定义职责评分规则为 **v3.2**；软件发布状态不代表评分经过官方认可或人工准确率校准。

项目基于 RMUC 2026 区域赛公开 SQLite 数据，以只读方式加载原始数据。可以搜索学校与比赛、生成包含地图和逐秒状态的回放，在战绩页查看个人贡献、队伍表现、评分依据和关键事件。

## 安装与启动

需要 Python 3.12或3.13，Conda模板使用3.13。纯复盘只安装 NumPy、Matplotlib、Pillow，不需要 PyTorch 或训练环境。

```bash
git clone https://github.com/Fbnnkk/rm-battlescope.git
cd rm-battlescope
python -m venv .venv
```

Windows PowerShell 激活：`.\.venv\Scripts\Activate.ps1`。Linux / macOS 激活：`source .venv/bin/activate`。然后：

```bash
python -m pip install -r requirements.txt
python scripts/rmuc_web.py
```

打开 [本地比赛库](http://127.0.0.1:8765/)。先按下一节准备数据库，再启动服务。已有 Conda 用户也可使用 `environment.yml`，通过 `conda env create --prefix ./.conda/envs/rmuc2026 --file environment.yml` 创建环境。

## 数据集

[RMUC 2026 区域赛公开数据来源：RoboMaster 论坛](https://bbs.robomaster.com/article/1936220)。下载解压后，将 SQLite 文件保存为 `dataset/rmuc_2026_region_dataset.sqlite`。数据库、压缩包、规则手册和训练产物不随源码分发；遵循原发布者的数据使用要求。

数据库放在其他位置时使用：

```bash
python scripts/rmuc_web.py --db /path/to/rmuc.sqlite --port 9000
```

默认只监听 `127.0.0.1`，用于本机复盘；本地HTTP服务没有公共多用户服务的认证与隔离设计。

## 个人战绩与评分

总览首先显示比赛结果、双方表现和个人榜单。点击兵种查看得分构成、表现曲线、数据覆盖与贡献依据，再跳到地图核对。队伍均分只统计已评级实体，页面同时显示分母和暂不评级数量。

v3.2 根据可观测的职责贡献折算输出、攻坚、协同、控场、机动、支援和代价：英雄攻坚，步兵交战与机动，哨兵防守，空中出动输出，工程装配，飞镖命中，以及雷达标记与反制。配置见 [configs/scoring_v3.json](configs/scoring_v3.json)。不同兵种职责与数据覆盖不同，分数只能作为复盘线索。

有效遥测已观测到活动但没有可计分贡献时，可以给出中性表现；缺少足够证据时显示“暂不评级”，分数与等级导出为空。例如只记录飞镖开闸，无法确认发射、未命中或伤害，不能自动给5分或C级。工程运输、兑换等未完整观测职责也不能凭空计分。

伤害量和来源分开核对：原始命中事件、原始受击遥测、高可信归因、中可信归因、低可信分摊、未归因有明确标签。高可信仍是遥测启发式推断，不等于裁判系统确认。低可信可按配置折算输出分，但不产生击杀或控场；逐目标核对高＋中＋低＋未归因＝原始弹丸受击HP，原始飞镖命中事件另计，避免混淆。

前压控制、迫退／火力受限、跨地形骚扰和掩护是基于位置与交战窗口的观测迹象，不能确认视线遮挡、操作者意图或战术因果。未观测贡献不等于没有贡献。规则权重与等级阈值尚未通过同步录像、人工标注和独立比赛校准；数据覆盖率、伤害守恒和分数分布均不能替代准确率验证。

## 回放与导出

地图上方提供播放／暂停、前后5秒、常用倍速与时间轴。“回放设置”展开后可切换按帧播放、无极调速、原始点、尾迹、轨迹／事件／攻击置信度过滤以及时刻链接。空格播放／暂停，左右方向键前后5秒，Shift加方向键前后15秒。

关键事件可以按类别、阵营与关键词筛选，点击后从事件前3秒跳转并显示上下文。连续能量机关击打合并为过程，原始记录可展开；击打不自动判为激活成功，未知字段含义不猜测。

本地后端生成的回放支持将战绩JSON和笔记保存到配置的输出目录，并下载评分CSV。笔记也保存在当前浏览器中，可通过JSON导出／导入备份。独立静态回放使用浏览器下载JSON，不请求Python后端；CSV仅在生成文件存在时显示。若内嵌浏览器不支持下载，请在Chrome／Edge中打开回放。

## 输出配置

优先级：**CLI `--output-dir` > `BATTLESCOPE_OUTPUT_DIR` 环境变量 > 项目 `.battlescope.local.json` > 项目 `outputs/`**。普通生成目录均被Git忽略，明确选择的公开演示在 `docs/replays/`。

Windows PowerShell：

```powershell
$env:BATTLESCOPE_OUTPUT_DIR = 'E:\outputs\RM-BattleScope'
python scripts/rmuc_web.py
```

Linux / macOS：

```bash
export BATTLESCOPE_OUTPUT_DIR="$HOME/rm-battlescope-results"
python scripts/rmuc_web.py
```

持久的项目本地配置：复制 `.battlescope.example.json` 为 `.battlescope.local.json`，编辑 `output_dir`。该本地文件被忽略，不进入开源提交；相对路径相对于项目根目录。配置错误或目标无法写入时会报错，不会静默改存其他位置。

`--output-dir` 指定当前命令的实际目标：Web存储回放子目录，单局CLI存储单局文件。省略时分别放在输出根目录的 `web_replays/` 和 `trajectories/<game_id>-<唯一后缀>/`。已有单局回放不会静默覆盖。

## 命令行

```bash
python scripts/rmuc_sqlite.py summary
python scripts/rmuc_sqlite.py schema
python scripts/rmuc_sqlite.py matches --school "浙江大学" --limit 10
python scripts/rmuc_trajectory.py --game-id 1779427046868
python scripts/rmuc_trajectory.py --game-id 1779427046868 --start 60 --end 180 --min-confidence high --output-dir ./outputs/clip
python scripts/rmuc_trajectory_audit.py
python scripts/batch_score.py --help
```

单局CLI默认生成 `trajectory.html`、`trajectory.png`、`quality_report.json`、`scores.json`、`scores.csv`、`review.json` 与 `timeseries_scores.json`。`--no-scores` 只生成轨迹，`--gif` 可额外生成GIF。`--return-url ../../index.html` 可为静态导出配置相对演示目录链接。

## 静态演示与 GitHub Pages

[当前源码中的v1.0.0静态演示](docs/replays/east-region-match-27-game-2/index.html)：浙江大学对东南大学，东部第27场第2局。克隆后可直接打开，也可运行 `python -m http.server 8080 --directory docs`，访问 `http://127.0.0.1:8080/`。此命令仅用于浏览静态示例，不能搜索数据库或重新解析比赛。

GitHub Pages只能托管预生成的静态页面，不能运行Python比赛库后端。可将仓库的 `docs/` 配置为Pages发布目录；实际启用状态以仓库设置和部署结果为准。静态页面采用相对目录链接和浏览器下载，适配 `/rm-battlescope/` 项目子路径。仓库保留 [北部第90场历史示例](docs/replays/north-region-match-90-game-1/index.html)，它使用旧版界面与规则，不作为最新版评分案例。

[原作者的历史在线演示](https://ezthor.github.io/rm-battlescope/replays/north-region-match-90-game-1/)与[演示视频](https://www.bilibili.com/video/BV1UvKN6LENV/)保留作来源参考，可能与当前功能不同。

## 结构与验证

核心源码在 `rmuc_trajectory/`，比赛选择与回放前端在 `rmuc_web/`，命令行入口在 `scripts/`，自定义规则在 `configs/scoring_v3.json`。测试覆盖轨迹处理、伤害归因、评分、缺证据实体、回放摘要和导出。CI在Windows、Linux、macOS上安装复盘依赖并执行全部测试与CLI启动检查。

```bash
python -m unittest discover -s tests -v
```

[规则与数据索引](docs/RMUC_2026_规则与数据索引.md)、[轨迹标定说明](docs/RMUC_2026_轨迹解析说明.md)、[本阶段发布说明](docs/RELEASE_v1.0.0.md)。源数据为秒级遥测，场地图为规则手册渲染图，不适用于实时控制、碰撞判断或毫米级路径分析。部分规则状态由遥测与事件组合推断，应结合比赛录像复核。

离线强化学习研究、数据集、模型权重和实验图不属于本次已验收的复盘发布范围，不提供训练性能结论。

## 来源、贡献与许可证

本项目沿用 [ezthor/rm-battlescope](https://github.com/ezthor/rm-battlescope) 的赛事解析与回放基础，保留原作者版权、MIT许可证、第三方素材声明和历史演示。本仓库增加个人评分、贡献依据与复盘体验改进，欢迎通过Issue提供具体比赛与复现步骤，并通过Pull Request提交修改。

代码以 [MIT License](LICENSE) 开源。场地图等第三方材料不在MIT授权范围内，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。RoboMaster及相关名称、规则、数据和素材权利归各自权利人所有，不暗示官方认可。
