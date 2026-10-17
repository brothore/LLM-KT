# API 价格记录与离线成本估算器

更新日期：2026-10-17。工具只读取数据快照和价格配置，不读取 API key，不调用模型，不修改请求内容、历史协议、路由或缓存实现。

## 1. 价格记录在哪里

此前模型配置没有记录价格；运行时只保存供应商响应中的实际 usage/cost。现已增加模型价格文件引用，例如：

```yaml
model:
  model_id: typesafe/jev-1.14
  pricing_file: pricing/jev-1.14.json
```

`pricing_file` 相对于 YAML 所在目录解析。加载时验证模型 ID、币种、计价单位和单价，并将完整 `pricing_snapshot` 加入解析后的配置。新推理运行的 manifest 因而保留当时使用的价格快照，而非只留一个可能改变的文件路径。价格不会发送给模型，也不改变供应商收费。

已有运行配置与新增价格字段不同，继续复用旧 run 可能被原配置一致性检查拒绝；新实验使用新 run 目录。修改价格不是刷新旧运行的实际账单。

价格文件为 `configs/pricing/jev-1.14.json` 和 `configs/pricing/qwen3.6-35b-a3b.json`，包括：

- 模型 ID、USD、每百万 token 的计价单位。
- 核实日期和官方来源 URL。
- 各供应商名称、端点 tag、输入和输出价格。
- 缓存读取/写入价格及隐式缓存能力标记；`null` 表示未知或未公布，不能解释为免费。

记录当日 Jev 输入为 $0.042/百万 token、输出免费；其端点没有公布缓存折扣。Qwen 的供应商价格不同，不能只用一个最低单价代表自动路由的实际费用。快照只列核实时状态为正常的端点，不保证将来可用或实际被路由选中。

来源：[Jev 端点元数据](https://openrouter.ai/api/v1/models/typesafe/jev-1.14/endpoints)、[Qwen 端点元数据](https://openrouter.ai/api/v1/models/qwen/qwen3.6-35b-a3b/endpoints)。后续更换模型时新增对应格式的价格文件，不能借用其他模型的价格。

## 2. 如何使用

在项目根目录运行，已安装包时可将 `python -m llm_kt` 换成 `llm-kt`。源码运行需先设置 `$env:PYTHONPATH = "$PWD/src"`。

```powershell
# 配置中指定的数据集、split 和协议
python -m llm_kt estimate --config configs/jev.yaml --output data/reports/costs/assist2009_jev.json

# 只为本次估算覆盖数据快照，不修改 YAML
python -m llm_kt estimate --config configs/jev.yaml --snapshot data/snapshots/peiyou --output data/reports/costs/peiyou_jev.json

# 普通模型：选定一个供应商价格，每题预计输出 256 token（含推理 token）
python -m llm_kt estimate --config configs/chat.yaml --snapshot data/snapshots/nips_task34 --provider darkbloom/fp4 --output-tokens 256 --output data/reports/costs/nips_qwen.json

# 假设输入 token 中 50% 以缓存读取价格收费，仅用于情景分析
python -m llm_kt estimate --config configs/chat.yaml --provider darkbloom/fp4 --cache-read-fraction 0.5 --output data/reports/costs/qwen_cache_scenario.json
```

输出 JSON 和同名 Markdown。报告包括请求总数、学生数、历史长度、输入字符总量、输入/输出 token 估计、各供应商输入/输出/总费用、敏感性范围及全部假设。

未指定 `--provider` 时分别计算价格文件里的候选端点，给出价格差异范围；若配置显式限定了 `provider.only`，或以 `allow_fallbacks=false` 限定 order，则按配置筛选。普通自动路由的候选价格范围不是供应商配比预测。`--provider` 只改变估算，不会替你固定真实调用端点。

## 3. 数量和输入 token 如何估算

数据量来自实际快照：校验哈希，读取配置指定的 test/train/validation 和 fold，按同一 `target_ranges` 窗口/分段/完整前缀协议计算目标。不是简单用“学生数 × 固定题数”估计。

工具流式读取学生，不生成 prepared 输入，不保留所有学生的展开历史。利用单条历史记录的序列化长度前缀和，计算所有目标的字符总量，因此不必生成数 GB 的重复窗口 JSON。

字符统计严格对应当前适配器的文本代理：

- Chat：消息 content 文本，外加存在时的 response_format 序列化长度。
- Decisions：state 和 questions 的规范 JSON 文本。
- 不把 HTTP headers、API key 或模型名称计入提示字符。

**字符总量可以逐请求核对，但供应商实际计费 token 仍是估计。**Chat 模板包装、schema 编码、分词器以及 Decisions 内部序列化可能与代理不同。

默认中心假设为 `3 characters/token`，另以 `2` 和 `4` 给出输入 token 敏感性范围：

```text
中心输入 token ≈ 输入字符数 / 3
较低输入 token ≈ 输入字符数 / 4
较高输入 token ≈ 输入字符数 / 2
```

该范围是人为设定的敏感性分析，不是统计置信区间，不保证包含真实账单。数字和 JSON 较多的输入不能套用普通英文的经验比例后就当作精确分词。

可使用 `--chars-per-token`、`--chars-per-token-low`、`--chars-per-token-high` 修改假设。实际分词信息不明时应保留这一不确定性。

## 4. 输出与推理 token

普通模型默认假设每题总输出 32 token，可用 `--output-tokens` 覆盖。这是情景假设，尚未经真实调用校准。它必须包含供应商计费的 reasoning token，不能只计算最终 JSON 的长度。

Decisions 默认输出 token 估计为 0；对当前 Jev 快照，由于输出价格为 0，输出用量并不影响费用。这不表示协议真实响应没有输出 token。

配置中的 max_tokens 是请求上限，不能当成平均实际输出。本工具也不将默认的 32 token 解释为普通推理模型的可靠平均数；开启较多推理时，应使用更保守的输出假设或先校准。

## 5. 成本公式与缓存情景

没有缓存、重试或本地命中时：

```text
总费用 USD = 输入 token / 1,000,000 × 输入单价
           + 输出 token / 1,000,000 × 输出单价
```

缓存情景将输入 token 划分为普通输入、缓存读取和缓存写入三部分，分别乘相应单价。缓存读/写比例指 token 占比，不是“多少请求命中”；两者之和不能超过 1。

- `--cache-read-fraction` / `--cache-write-fraction` 默认 0，不预设缓存收益。
- 缓存价格缺失或当前能力信息不支持指定情景时直接报错，不使用零价格替代未知值。
- 自动缓存可能有额外写入收费，未公布或未配置的费用不在基础估计中；估算结果不是金额上限。
- `--local-hit-fraction` 假设同一比例目标已有本地结果，默认 0。它假设命中目标与剩余目标的平均长度相同，并不读取 SQLite 来精确扣除已完成目标。
- `--extra-attempt-fraction 0.1` 表示额外尝试数为基准远程请求的 10%，默认 0。按相同平均用量计费，是保守的简化；真实失败请求未必按完整请求收费。

工具估算的是**整个所选数据集 split**。`execution.max_requests=20` 只限制一次 infer/resume 命令的尝试数，不会把完整数据集估算缩成 20 次请求，也不会据此自动限制美元花费。

税费、充值/支付费用、批量折扣、未列出的附加费用和未来价格变化不计入。报告币种为 USD，不自动换算人民币。

## 6. 用真实调用校准

不需要为估算器提供 key；使用已经存在的真实运行产物：

```powershell
python -m llm_kt estimate --config configs/chat.yaml --calibration-run runs/your_small_pilot --output data/reports/costs/calibrated.json
```

工具核对模型、适配器、endpoint、参数和概率提取策略，读取成功预测的实际 input/prompt token 及保存的请求，计算“总字符 / 实际输入 token”的经验比率。输出用量齐全时同时取得平均 output/completion token，`--output-tokens` 仍可覆盖。

校准后敏感性范围采用经验比率的 0.8–1.2 倍，仍不是置信区间。样本应覆盖短历史、窗口增长和滚动阶段；若供应商不回报全部推理用量、模板改变或实际供应商变化，该校准可能不适用。模型配置匹配不能证明所有内部计费条件相同。

## 7. 验证与现有报告

新增测试检查了全部协议下的高效字符统计与真实构造请求的一致性、手算成本、缓存读写计价、未知价格错误、供应商筛选、输出免费、离线执行和用量校准。连同原有测试共 19 项通过。

已为三个本地数据集分别生成 Jev 和 Qwen 的无缓存默认情景报告，位于 `data/reports/costs/<dataset>_jev.json/.md` 和 `<dataset>_chat.json/.md`。这些报告未进行任何付费调用。

估算器实现：`src/llm_kt/costs.py`；价格校验：`pricing.py`；命令入口：`cli.py`。缓存实际优化仍按 `docs/TODO.md` 单独安排，本次没有实施。
