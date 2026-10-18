# LLM-KT

独立的大模型知识追踪推理与评估框架，基础版本 v0.1。默认使用此前的真实作答预测当前作答，不把预测概率回填历史。

仓库地址：[brothore/LLM-KT](https://github.com/brothore/LLM-KT)。支持决策模型和普通大模型，复用 pykt 数据协议，并提供学生级评估及离线成本估算。

Qwen阶段现已暂停付费工作。最新数据、推荐配置、成果、失败记录和未结事项统一见 [Qwen阶段复盘与交接](docs/Qwen阶段复盘与交接.md)。当前采用按学生保留完整评价序列的开发集；旧0.1%稀疏时间步子集仅保留作历史探索记录。

Jev采用独立 `decision_file` 和 `state/questions/noul`，不复用Chat prompt；[两样本适配验收](docs/Jev独立决策适配与冒烟验收.md)后已完成[四人492目标完整对比](docs/Jev完整开发集与Qwen对比.md)。Jev整体AUC=0.756513、学生mean=0.672589，费用约￥0.4424；高于Qwen baseline，低于R2和省钱版。大验证集尚未执行。

```powershell
git clone https://github.com/brothore/LLM-KT.git
cd LLM-KT
```

仓库只发布代码、文档、公开价格快照和合成测试数据。真实数据集、运行日志、模型响应和密钥不随仓库发布。数据在本地准备，密钥保存在项目外。

每次提交和推送前必须按 [AGENTS.md](AGENTS.md) 审查敏感信息。克隆后可启用项目提供的 Git 钩子：

```powershell
git config core.hooksPath .githooks
python tools/audit_sensitive.py --worktree --staged --history
```

## 已实现

- YAML/JSON 配置：数据快照、训练/验证折、测试集、历史协议、模型与执行参数。
- pykt 完整学生序列 CSV 的项目内复制、哈希校验及训练测试学生隔离检查。
- 题目级多知识点集合、单知识点数据、概念展开序列的受限重建。
- `pykt_window`、`pykt_segmented`、`full_prefix` 目标及历史构造；稳定 sample/event ID，标签与模型输入分离。
- OpenRouter Decisions Noul/Choice、OpenAI 兼容 Chat JSON/小数/整数概率解析，以及离线 Mock。整数输出虽已支持，但Qwen降本实验效果下降，不推荐。
- 异步有界并发、请求速率控制、可恢复错误重试、SQLite 缓存、进程锁、断点续跑。
- 整体 AUC/ACC/LogLoss/Brier、学生 AUC mean/std/IQR、有效学生和目标覆盖率；另有历史统计口径。
- 相同任务上的多模型比较；不完整结果仅报告共同成功目标的探索性比较。

## 安装

在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .
```

以下示例在激活虚拟环境后使用 `python -m llm_kt`。也可以直接调用安装后的 `llm-kt`。不需要安装 pykt、Torch、CUDA 或 WandB。

如当前 Python 已具备依赖，开发时可用 `$env:PYTHONPATH = "$PWD/src"` 后执行模块命令。

## 无密钥跑通示例

使用项目中的合成数据，不发网络请求：

```powershell
python -m llm_kt import-pykt --test tests/fixtures/test_quelevel.csv --dataset demo --output data/snapshots/demo
python -m llm_kt prepare --config configs/demo.yaml --output data/prepared/demo
python -m llm_kt validate --config configs/demo.yaml --prepared data/prepared/demo
python -m llm_kt infer --config configs/demo.yaml --prepared data/prepared/demo --run runs/demo --dry-run
python -m llm_kt infer --config configs/demo.yaml --prepared data/prepared/demo --run runs/demo
python -m llm_kt evaluate --prepared data/prepared/demo --run runs/demo
python -m llm_kt resume --config configs/demo.yaml --prepared data/prepared/demo --run runs/demo
```

快照和 prepared 输出目录必须不存在；重新导入使用新目录。重复 infer/resume 复用相同 run manifest，只补齐未完成目标，不重复调用成功目标。

合成数据含 3 名学生、10 个目标，其中 1 名学生只有单一标签，标准 AUC 不可计算；Mock 仅用于检查链路，结果不代表大模型 KT 能力。

## 导入实际数据

本地三个数据集的目录、数量、处理来源和配置见 [三个 KT 数据集的处理与存放](docs/三个KT数据集的处理与存放.md)。批处理脚本为 `tools/prepare_local_datasets.py`，运行它需安装 `pip install -e ".[preprocess]"` 所列可选依赖。

首版接收 **完整原始学生序列 CSV**，例如 pykt 的 `test_quelevel.csv` 和 `train_valid_quelevel.csv`。不要传入 `test_*_sequences.csv` 等已切片文件。所选 CSV 会复制到快照中，运行时不再读取来源路径。

```powershell
python -m llm_kt import-pykt --test D:/datasets/assist2009/test_quelevel.csv --train-valid D:/datasets/assist2009/train_valid_quelevel.csv --dataset assist2009 --representation question_native --output data/snapshots/assist2009
python -m llm_kt prepare --config configs/jev.yaml --output data/prepared/assist2009
```

CSV 每行对应唯一学生，必需 `fold,uid,concepts,responses`，题目列 `questions` 可缺省。序列逗号分隔，多知识点如 `2_7`；原始文件不能含 padding。测试 fold=-1，训练验证 fold>=0。

项目已提供 ASSIST2009/NIPS Task 3/4 原始数据处理脚本，以及培优已有产物复用流程，详情见数据集说明。通用 `import-pykt` 命令本身只负责导入已有完整序列。

`--representation` 的含义：

| 取值 | 输入与限制 |
|---|---|
| `question_native` | 一次作答一项，知识点为集合，适合 quelevel 原始文件 |
| `single_concept` | 每次作答一个知识点，无重复展开 |
| `concept_expanded` | 通过 is_repeat 将同题概念合并；首版只允许 full_prefix，不能声称等价于上游题目专用窗口 |

配置中的 `dataset.split` 可选 `test/train/validation`；`validation_fold` 指定验证折。训练和验证只读取 train_valid，框架本身不训练权重。数据已经分好 fold 时直接复用；首版不负责为全新未划分日志自动生成训练测试划分。

## 切换模型

`configs/jev.yaml` 使用 Decisions API，`configs/chat.yaml` 使用 Chat Completions API；两者的数据与任务配置相同，可共用 prepared 目录。

密钥保存在项目外的 `D:/1017KT/.secrets/openrouter.key`，只写一行真实密钥，不加引号、变量名或 `Bearer`。使用文本编辑器粘贴保存，不要在终端命令中写真实密钥。两个示例模型配置已指定此文件（路径相对于配置文件）。父目录与项目忽略规则排除了密钥目录和 `.key` 文件。

仍支持环境变量，优先级高于密钥文件。框架只将文件路径或环境变量名写入运行配置，不将密钥内容写入配置和请求日志。测试命令：

```powershell
python -m llm_kt infer --config configs/jev.yaml --prepared data/prepared/assist2009 --run runs/assist2009_jev --dry-run
python -m llm_kt infer --config configs/jev.yaml --prepared data/prepared/assist2009 --run runs/assist2009_jev
```

`.env.example` 仅列变量名；框架不自动读取 `.env`。模型 ID、endpoint、API key 环境变量和参数均可配置。参数支持情况需根据所选模型确认；`validate` 检查本地配置与数据，尚不进行供应商能力探测或上下文 token 计算。Chat 默认要求最终 JSON `{"p_correct":0.73}`，不从推理文本提取数字。

正式实验避免使用 latest 别名；不同模型调用用不同 run 目录。要独立重复推理，修改 `execution.replicate_id` 并使用新 run；缓存不算新的重复实验。

## 执行限制和恢复

`execution.max_requests` 是**本次命令的尝试次数上限**，包含重试；不是美元预算。相同配置继续运行 resume，可继续补齐目标。`max_attempts` 是每个目标在整个 run 的累计尝试上限，达到后不再重试。永久失败保留原因；更改配置需新建 run。

普通 CLI infer/resume 默认未绑定金额预算。付费实验工具 `tools/run_prompt_experiment.py` 支持 `--budget` 与 `--budget-limit-usd`，通过持久化账本预留、结算和未知费用占额控制金额；共享账本仅支持依次执行，不支持多进程并写。金额触顶停止新派发并等待在途请求记录完成。实际 usage.cost 缺失时保留预留，不按零计算。429/部分 5xx/连接异常可重试；鉴权、模型与请求参数等严重 HTTP 错误停止新派发。网络不确定状态下重试可能产生额外计费。

中断后成功结果以 SQLite 为准；resume 会重新导出预测文件。一个 run 同时只允许一个执行进程。请求和响应保存在 run 中，Authorization 不落盘；数据可能包含学生作答历史，应按实验数据管理。

## 离线成本估算

模型配置通过 `pricing_file` 记录带来源和日期的输入、输出、缓存价格。无需密钥、无需生成全部推理输入即可估算所选数据集 split：

```powershell
python -m llm_kt estimate --config configs/jev.yaml --snapshot data/snapshots/peiyou --output data/reports/costs/peiyou_jev.json
python -m llm_kt estimate --config configs/chat.yaml --provider darkbloom/fp4 --output-tokens 256 --output data/reports/costs/assist2009_chat.json
```

输出 JSON 与 Markdown，默认不计缓存折扣。token 采用明确的字符比率假设，可用既有真实调用用量校准；结果不是精确账单或花费上限。详细参数和计价规则见 [API 价格与成本估算器](docs/API价格与成本估算器.md)。

## 指标和结果

```powershell
python -m llm_kt evaluate --prepared data/prepared/assist2009 --run runs/assist2009_jev
python -m llm_kt evaluate --prepared data/prepared/assist2009 --run runs/assist2009_jev --profile pykt_legacy_v1
python -m llm_kt compare --prepared data/prepared/assist2009 --runs runs/assist2009_jev runs/assist2009_chat --output runs/comparison.json
```

标准口径每个目标一次，单类学生 AUC=null，仅可计算学生参与 mean/std。默认样本标准差 ddof=1，有效学生少于 2 时 std=null。整体 AUC 与学生平均 AUC 是不同指标。部分完成的评估明确报告覆盖率。

`pykt_legacy_v1` 提供二次滑窗 200、单类赋 0.5、样本标准差以及 Gini/EAWI 的历史统计规则。它不自动将原始概率舍入为上游输出精度，也不补齐旧模型的专有融合路径，因此不能仅凭开启此选项宣称严格复现旧实验。

产物包括 `manifest.json`、`state.sqlite`、`requests/`、`responses/`、`predictions.jsonl/csv`、`failures.jsonl`、`execution_summary.json`、`metrics_<profile>.json`、`per_student_<profile>.csv` 和 Markdown 报告。只改指标可离线重算，无需再次调用模型。

## 测试

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m unittest discover -s tests -v
```

测试不访问网络、不需要密钥，HTTP 返回通过模拟检查。覆盖标签隔离、历史更新、序列边界、同题多知识点、严格概率解析、学生指标、请求限额和恢复。

## 代码和来源

`data.py` 管导入；`protocols.py` 管目标和可见历史；`providers.py` 管 API 数据格式；`runtime.py` 管执行状态；`evaluation.py` 管离线统计；CLI 只编排用例。领域接口在 `domain.py`。

所需 pykt 来源已复制到 `third_party/pykt/reference`，并在 `provenance.json` 记录上游提交和哈希。运行使用项目内 `vendor/pykt` 的纯索引协议和学生统计函数；reference 仅作为来源审阅材料，不导入原仓库。许可见 `third_party/pykt/LICENSE`。

完整设计和后续范围见 [设计方案](docs/大模型KT独立项目设计方案.md)。当前版本不包含模型训练、few-shot 检索、多步预测、自动模型能力发现或完整历史结果导入。

历史阶段完成过NIPS 0.1%稀疏预测目标抽样及Jev/Qwen冒烟测试，见 [NIPS 子集与全流程冒烟验收](docs/NIPS子集与全流程冒烟验收.md)。该抽样未保留每学生完整评价目标，现已退出正式评估；当前两级完整学生验证集见 [两级验证集说明](docs/两级完整学生验证集.md)。真实数据和运行产物只保存在本地忽略目录。

本地结果缓存、服务端 prompt 缓存的区别、供应商配置、较早的重复请求对照及复现命令见 [缓存用法与小样本成本实测](docs/缓存用法与小样本成本实测.md)。较早的重复请求小样本未观察到远端命中，不能据此判断相邻目标是否可以复用历史。

已完成 [KT 相邻历史前缀缓存探索](docs/KT相邻历史前缀缓存探索.md)：Qwen3.8/Chutes增长样本观察到输入折扣，滑窗滚动后未命中。后续R2已实现可配置的历史在前、无损紧凑编码和可选学生串行；具体效果见 [R2降Token结果](docs/R2降Token优化结果.md)，不能将增长样本折扣外推整个数据集。

已完成 [三轮完整 Prompt AUC 迭代](docs/Prompt三轮迭代结果.md)：Tenstorrent 成本更低；验证集最佳R2 AUC为0.798241，但独立测试为0.766683，未超过基线0.777541，因此未替换默认prompt。每轮假设、复盘、费用、配置与完整本地追溯均已保留。

以上三轮使用旧稀疏目标集。后续已完成 [四人完整序列重验](docs/四人开发集Prompt重新验证结果.md) 和 [R2降Token优化](docs/R2降Token优化结果.md)：推荐配置 `configs/experiments/nips_validation_iteration_02cny_v1_r2_cost_optimized.yaml`，4人492目标AUC=0.782239，学生mean=0.716313；输入/输出token分别少29.79%/45.82%，本次费用约￥0.0631。只验证了小开发集，大验证集尚未推理，原基线配置保留。


## Jev迭代阶段交接（2026-10-18）

四轮候选、最佳完整复跑和原模板对照已完成，固定4学生492目标。推荐配置 `configs/experiments/nips_jev_optimized_validation.yaml`。最佳整体AUC .786681、复跑.787856；已知费用￥2.375435，未知预留￥.070965，停止付费。详见 [Jev多轮适配结果与交接](docs/Jev多轮适配结果与交接.md)。待核实未知账单；大验证未运行，新付费工作需另行授权。
