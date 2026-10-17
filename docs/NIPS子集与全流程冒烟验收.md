# NIPS 0.1% 子集与全流程冒烟验收

## 数据集与抽样口径

2026-10-18，基于项目已有的 `nips_task34` 原始序列快照生成独立派生数据集 `nips_task34_0p1pct`。按用户确认的口径，在测试集及每个训练验证 fold 内分别抽取约 0.1% 的**可预测目标**，不是抽取学生，也不是直接截取 CSV 的前 0.1%。

| 分区 | 原目标数 | 抽样目标数 | 覆盖学生数 |
|---|---:|---:|---:|
| test | 271833 | 272 | 228 |
| fold 0 | 221348 | 221 | 179 |
| fold 1 | 226311 | 226 | 186 |
| fold 2 | 215220 | 215 | 182 |
| fold 3 | 218892 | 219 | 180 |
| fold 4 | 224205 | 224 | 194 |
| 合计 | 1377809 | 1377 | 1149 |

使用 `pykt_window`、question 级别、maxlen=200、min_seq_len=3。每 fold 目标数量四舍五入；配置 validation_fold=0 时，train 为其余四个 fold 共 884 个目标，validation 为 221 个，test 为 272 个。

分层维度为目标正负标签、历史长度区间（1–9、10–49、50–99、100–199）及当前题是否有多个知识点。最大余数法分配层内名额，再按 seed=42 与父目标身份的 SHA256 排序抽样，避免源文件行序偏差。小层可能因比例舍入没有样本；这是比例分层样本，不保证覆盖所有题目和知识点。

抽样使用标签进行离线分层；标签只保存在目标文件，不进入请求。保留选中学生完整的原序列，预测时再严格按原协议取当前目标之前的历史，当前及未来标签均不可见。`selection.jsonl` 明确哪些目标计分，因此不能将保留序列里的所有事件再次当成子集目标。

本地路径：`data/snapshots/nips_task34_0p1pct/`。其中 `manifest.json`、`selection.jsonl`、`sampling_report.json`、完整上下文 CSV、阶段哈希清单共同描述派生关系。父快照与抽样清单均参与身份计算。构建时的上下文中间 CSV 保存在相邻 `.building` 目录，完成状态由阶段清单标记。

配置入口：`configs/datasets/nips_task34_0p1pct.yaml`；可修改 split 和 validation_fold 使用不同分区。成本估算器读取同一目标清单，不会误算为所有保留学生的全部事件。

## 实际 API 冒烟结果

两个模型使用相同两条真实目标：一正一负、来自两名学生。该两条输入向 OpenRouter 发送已获用户明确授权，只含题目/知识点 ID 与此前作答，不含学生 ID、当前或未来答案。

| 模型 | 成功数 | 本次远程请求 | 接口返回费用 USD | 整体 AUC | ACC |
|---|---:|---:|---:|---:|---:|
| typesafe/jev-1.14 | 2/2 | 2 | 0.000286734 | 1.0 | 1.0 |
| qwen/qwen3.6-35b-a3b | 2/2 | 2 | 0.000461500 | 1.0 | 1.0 |

合计费用 0.000748234 USD，来自响应 usage.cost。两条样本仅验证工程链路，不能据此比较模型性能。每名学生只有一个目标，学生 AUC 无法定义，mean/std 均正确输出 null，有效学生 AUC 覆盖率为 0；没有把它们替换成 0 或 0.5。

Qwen 配置关闭 reasoning，max_tokens=256；最终内容必须为精确的 `{"p_correct": number}`，且正常结束。Jev 读取 noul 概率。配置分别是 `configs/smoke/nips_jev.yaml` 与 `configs/smoke/nips_qwen.yaml`，target_limit=2，max_requests=2，max_attempts=1，concurrency=1。target_limit 的确定性选择策略交替取正负池，优先接近 100 长度的历史，再按父事件身份排序；它专用于小样本验收。

两个运行目录：

- `runs/nips_0p1pct_jev_smoke_v1/`
- `runs/nips_0p1pct_qwen_smoke_v1/`

再次调用 infer 后，两者均成功复用已有两条结果，`attempts_this_invocation=0`，不再读取凭据或发送请求。评估重复执行命中缓存；两模型共同目标比较、每个运行的请求响应哈希检查全部通过。验收汇总保存在 `data/reports/nips_smoke_acceptance.json`。

初始字符代理估价低估了实际输入计费，因此另存了利用真实 usage 校准的报告 `data/reports/nips_{jev,qwen}_smoke_calibrated.json`。两个样本只能提供粗略校准，不能代替完整数据集的 token 测量。远端 prompt cache 的输入重排优化没有在本次实施。

## 缓存与追溯结构

新增 `artifacts.py` 统一记录 SHA256、阶段身份、父产物、UTC 时间、Python/平台、Git HEAD 及实际 Python 源码指纹。源码指纹补足未提交改动无法由 Git HEAD 描述的问题。

| 阶段 | 保留产物与身份 |
|---|---|
| 子集构建 | 父快照身份、抽样规格、选择清单、分层统计、上下文 CSV、阶段清单 |
| prepare | 目标标签、模型可见输入、学生源行映射、task_id、文件哈希、缓存调用事件 |
| preview / validate | 确切请求示例，按内容哈希保存在 prepared/previews 下 |
| estimate | JSON/Markdown 报告及内容寻址的 `.artifacts` 存档 |
| infer | run_id；每次调用独立 invocation_id；每次请求独立 attempt_id；SQLite 作业状态；请求 JSON 与 HTTP 编码正文；响应正文及 JSON；状态码、允许列表响应头、时间、耗时、用量、错误类别、请求及结果哈希 |
| 导出 | 每次 invocation 独立保存 predictions、failures、summary 与 attempt 元数据哈希索引；根目录文件提供最新视图 |
| evaluate | 以 task、预测文件哈希、评估配置、源码指纹寻址，保存 metrics、逐学生 CSV、Markdown 与阶段清单 |
| compare | 共同目标比较报告及各预测文件哈希，保存内容寻址存档 |

运行目录复制 prepared 输入和标签，保留自包含的输入来源。请求不包含 Authorization 头；响应正文在落盘前移除已知密钥及 OpenRouter token 形式，保留是否发生脱敏的标记。HTTP 失败也保留返回正文，网络异常没有收到响应时记录异常类别。请求尝试预先落盘，进程中断可留下明确的 running 状态；这些记录不能视为已完成响应，也不能推断远端是否已计费。

完成的阶段通过 `artifact_manifest.json` 校验内容；成功结果恢复前校验对应请求/响应哈希及结果哈希，发现修改即报错。运行配置或源码指纹变化需要新运行目录。该缓存是实验本地恢复缓存，不是跨模型、跨运行目录的全局响应共享服务；随机模型的一次成功结果仍代表一次固定 replicate。

`verify-run` 校验 prepared、副本、各 invocation 阶段、attempt 索引、请求响应及最终预测的哈希。哈希用于发现意外修改，不是防止具有本地写权限的人同时改写文件与清单的签名机制。

## 复现命令

先按 README 安装项目，或在 PowerShell 设置 `$env:PYTHONPATH="$PWD/src"`。

```powershell
python -m llm_kt subset --parent data/snapshots/nips_task34 --output data/snapshots/nips_task34_0p1pct --dataset nips_task34_0p1pct --fraction 0.001 --seed 42
python -m llm_kt prepare --config configs/smoke/nips_jev.yaml --output data/prepared/nips_0p1pct_smoke_v1
python -m llm_kt infer --config configs/smoke/nips_jev.yaml --prepared data/prepared/nips_0p1pct_smoke_v1 --run runs/nips_0p1pct_jev_smoke_v1
python -m llm_kt infer --config configs/smoke/nips_qwen.yaml --prepared data/prepared/nips_0p1pct_smoke_v1 --run runs/nips_0p1pct_qwen_smoke_v1
python -m llm_kt evaluate --prepared data/prepared/nips_0p1pct_smoke_v1 --run runs/nips_0p1pct_jev_smoke_v1
python -m llm_kt verify-run --run runs/nips_0p1pct_jev_smoke_v1
```

现有缓存对应本次源码版本。今后修改代码时，新建版本化的 prepared/run 目录，避免混淆不同实现。

## 本地验证与存储

24 项离线测试通过，新增验证抽样可复现、估算目标数与 prepare 一致、完整历史不变、缓存恢复不读取密钥、不重复请求、产物篡改被发现。已有测试继续覆盖真实历史更新、边界、标签隔离、失败重试、指标及隐私扫描。敏感信息扫描通过。

真实数据、模型输入输出、SQLite、各阶段中间文件与运行元数据全部位于 Git 忽略的 data/runs 下。API key 继续位于项目外；本次没有提交或推送改动。
