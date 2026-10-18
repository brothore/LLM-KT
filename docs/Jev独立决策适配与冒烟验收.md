# Jev独立决策适配与两样本冒烟验收

后续状态：用户已另行授权并完成4人492目标的完整Jev验证，见 [完整开发集与Qwen对比](Jev完整开发集与Qwen对比.md)。本文“未进行完整评测”描述冒烟阶段当时范围，不是当前最新状态。

2026-10-18，用户要求以R2的信息组织为基础，针对Jev的决策特性独立适配，并先用1～2个样本跑通。本轮完成适配及两次真实请求，不包含完整验证集评测或AUC优化结论。Qwen阶段已冻结，未知费用账本未动用。

## 查阅依据与适配原则

阅读了 [TypeSafe构建指南](https://docs.typesafe.ai/concepts/how-to-build-with-system-one)、[Noul定义](https://docs.typesafe.ai/primitives/noul) 和 [OpenRouter Decisions Python接口](https://openrouter.ai/docs/client-sdks/python/sdks/decisions/README.md)，并只读核对 [Jev1.14端点元数据](https://openrouter.ai/api/v1/models/typesafe/jev-1.14/endpoints)。官方模型采用state与类型化问题，窄问题与明确判定标准适合由代码组合；不是Chat文本生成接口。

本轮将预测事件明确为“这个学生对当前题目的真实作答是否正确”，使用一个 `noul` 问题，直接取 `answers.will_answer_correctly.noul` 为预测概率。true/false分别对应当前响应1/0，不能改成“是否确定掌握知识点”或只输出一个正确/错误分类。没有要求Jev输出JSON、四位小数、解释或思维过程，也没有传Chat消息、temperature、reasoning、max_tokens或Chat缓存字段。

## 与R2共享的信息、独立的模板

- `state.history`：全部协议允许的历史对象，时间有序，correct是真实历史作答。
- `state.current_item`：当前题目和知识点ID，不包含当前答案。
- `state.evidence`：近期10/20题、整体、相同题目、共享知识点、各当前知识点统计，及原训练fold1–4的平滑难度先验。
- `state.interpretation`：字段语义、ID不代表题目文本、训练先验来源和高p代表易题等说明。
- `questions.will_answer_correctly`：独立的问题指令和对立criteria，通过路径引用state中的证据。保留R2的题目难度与个人近期表现互补、稀疏证据收缩、避免重复计数等原则。

本轮优先使用自解释的结构化对象，没有直接搬用Qwen的逐行压缩prompt。是否压缩Jev输入以及如何组合多个辅助问题，应在后续独立检验；同一请求的多个问题只能共享同一个合法时间点的state，不能放入未来作答。

模板文件是 `configs/decisions/jev_r2_v1.yaml`，加载入口为 `model.decision_file`；Chat仍使用 `model.prompt_file`。配置校验拒绝混用。`decision_prompts.py`只负责决策state/问题，不导入Chat renderer。两者共享纯KT事实计算模块 `kt_features.py`；原prompts模块保留兼容导出，原Qwen模板未修改。

## 两样本范围与实测结果

从既定四人完整开发集中，按学生ID稳定排序取前两人，分别选择原事件index20和80；不按标签、模型输出或效果选样。仅生成独立接口冒烟prepared，原四人492个完整目标和五人1466个目标均未修改。这种两时间步抽取只用于接口验收，不是新的正式验证数据集。

| 冒烟请求 | 合法历史长度 | Jev返回概率 | 输入token | API报告output_tokens | 实际费用USD |
|---|---:|---:|---:|---:|---:|
| 1 | 20 | 0.43 | 1610 | 24 | 0.000067620 |
| 2 | 80 | 0.87 | 3507 | 24 | 0.000147294 |

两次均来自TypeSafe，全部成功，无重试、无未知费用，完整执行约3.11秒。费用合计 **$0.000214914**，按规划汇率7约 **￥0.001504398**。两条概率保留API原始数值，不人为增加小数精度。usage报告的output_tokens是决策响应计量，不能理解为Jev生成了24个解释文本token；端点价格快照的输出价格为0。

独立账本上限$0.01，单次配置最多两次尝试、每目标最多一次，不复用Qwen的余额或未知预留。端点本轮价格输入$0.042/M、输出0，元数据 `supports_implicit_caching=false`，没有假设缓存折扣。

两个学生各一个冒烟目标，因此学生AUC均不定义，报告明确0/2有效；这不是正式数据抽样问题复发，而是刻意限定接口测试范围。不能凭这两条概率判断JeV优于R2、已完成KT概率校准或已提高AUC。

## 代码、配置与追溯

- `src/llm_kt/decision_prompts.py`：独立加载和state/问题渲染。
- `src/llm_kt/kt_features.py`：Chat/Decisions共享的纯历史和先验计算。
- `providers.py`：Decisions适配器支持冻结decision_snapshot，保留旧无模板路径。
- `config.py`：独立decision_file、Chat与Decisions模板隔离校验。
- `costs.py`：决策模板有动态证据时走完整请求渲染的离线字符统计，不漏算先验。
- `configs/pricing/jev-1.14-20261018.json`：新日期快照，未覆盖旧价格记录。
- `configs/experiments/nips_jev_r2.yaml`：两请求冒烟配置；用 `tools/smoke_jev_r2.py`生成隔离的冒烟scope。
- `configs/experiments/nips_jev_r2_full_validation.yaml`：完整492目标的待用配置，**没有执行**。
- `data/prepared/jev_r2_smoke_20261018_v1`：源task/sample映射、两条合法输入及本地标签、stage哈希。
- `runs/jev_r2_smoke_20261018_v1`：冻结计划、独立budget、原始请求/响应及HTTP body、attempt/invocation、SQLite、预测、覆盖率和summary。

原始作答、学生身份及模型输出仅在被Git忽略的data/runs中。密钥由运行时在内存读取项目外路径，不写入请求存档、文档或Git。

## 验收与边界

49项离线测试通过，覆盖独立模板请求形状、配置混用拒绝、合法概率解析和原功能。共享KT特征抽取重构后，将原Qwen R2的492条渲染请求与旧存档逐条比较，完全一致；两个Jev样本的history/current_item/evidence也逐字段等于对应R2合法输入。

两次真实请求已验证结构化state、noul响应解析、概率记录、费用结算、评估落盘和原始证据哈希；prepared、plan、总stage和verify_run均通过。接口链路已跑通，完整序列效果验证尚未开展。

后续应先在固定完整开发集上建立Jev基线，再单独比较问题措辞、state结构、多个原子问题组合或输入压缩，并与训练题目先验的简单基线对比。不能直接继承Qwen的token优化收益或AUC结论，也不能用未知预算预留自动扩展付费范围。

## 与Qwen测试对齐的单轮费用估算

2026-10-18离线渲染完整目标，并使用本轮两次真实请求校准：字符/token约2.00977，平均输出计量24token、输出单价0；不假设缓存或本地答案复用，美元按人民币7规划换算。只估算，没有执行完整推理。

| 固定目标集合 | 学生/目标 | Jev当前独立R2模板估算 | token假设敏感性范围 | 含输入保守余量及10%重试的预算参考 |
|---|---|---:|---:|---:|
| 四人开发集 | 4 / 492 | ￥0.4385 | ￥0.3654～0.5482 | 约￥0.603，建议￥0.65 |
| 大验证集 | 5 / 1466 | ￥2.1731 | ￥1.8110～2.7164 | 约￥2.988，建议￥3 |

敏感性区间由实测字符/token比率上下浮动20%得到，不是统计置信区间；只有两条校准样本，长历史分词和内部序列化仍可能偏离。预算参考取上侧token估算再加10%重试，不保证任何网络故障下都能完成。

四人同492目标的Qwen既有实测：B0约￥0.065257，R2约￥0.092180，R2省钱版约￥0.063137。新跑一次Jev可直接与这些已保存预测比较，无需重跑Qwen。若四组都重新运行，参考合计约￥0.6591，但Qwen未来缓存温度和输出可能变化。

“￥1大验证集”是此前按Qwen价格定义的数据规模名称，不代表换成Jev后仍需￥1。当前Jev模板保留结构化历史和全部证据，计量按其自身端点校准，且没有已验证的输入缓存折扣。

估算JSON/Markdown、源运行哈希和汇总位于 `data/reports/jev_qwen_aligned_cost_20261018_v1`；真实数据仍不上传。这里的Jev估算是一种已适配模板的一轮，不是已经实现了Jev baseline/R2/省钱三种模板的总价。
