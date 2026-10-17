# 三个 KT 数据集的处理与存放

处理日期：2026-10-17。全部数据位于独立项目内，未调用模型接口。

| 数据集 | 训练/验证学生 | 测试学生 | 总作答次数（题目级） | 测试窗口目标数 |
|---|---:|---:|---:|---:|
| peiyou | 14,453 | 3,613 | 5,549,635 | 1,099,197 |
| assist2009 | 3,082 | 770 | 282,619 | 52,378 |
| nips_task34 | 3,935 | 983 | 1,382,727 | 271,833 |

## 数据来源与处理规则

- 培优压缩包已经包含完整 pykt 处理产物，复用原文件，保留原有学生划分、fold、ID 映射，不重新随机切分。
- ASSIST2009 原始 CSV 通过项目内复制的 pykt 数据处理器排序、过滤、生成六行格式，再做概念展开/题目级映射和划分。
- NIPS 仅处理 Task 3/4，关联作答时间和三级知识点；Task 1/2 不参与。本地原始压缩包仍完整保留。
- 新处理数据采用 pykt 默认的 random_state=1024、测试比例 0.2、训练验证 5 折、maxlen=200、min_seq_len=3。
- 概念级和题目级分别生成映射，保存为 keyid2idx_concept.json / keyid2idx_question.json；通用 keyid2idx.json 对应题目级。培优保留包内映射。
- NIPS 处理器将解析元数据列表的 eval 改为 ast.literal_eval，并缓存重复时间值的原函数转换结果；排序、时间转换规则、三级知识点及响应转换沿用 pykt。

## 项目内目录

```text
data/
  raw/archives/                    # 三个完整来源压缩包
  raw/assist2009/                  # 解压的原始 ASSIST2009 CSV
  raw/nips_task34/                 # Task 3/4 训练日志及相应 metadata
  processed/<dataset>/            # pykt 完整序列、普通分段、映射和处理清单
  snapshots/<dataset>/            # 本框架直接使用的题目级数据快照
  reports/                        # 汇总、校验结果和处理日志
```

每个 processed 目录包含 train_valid.csv、test.csv、train_valid_quelevel.csv、test_quelevel.csv、
train_valid_sequences.csv、test_sequences.csv、train_valid_sequences_quelevel.csv、test_sequences_quelevel.csv，
以及 ID 映射和 processing_manifest.json。新处理数据还保存 data.txt。

snapshots 目录保存 test.csv 和 train_valid.csv（内容为题目级表示）、keyid2idx.json 和 manifest.json。
manifest 记录来源、表示、数据哈希与快照身份，运行时无需访问原 pykt 仓库。

## 窗口文件与推理输入

没有额外展开巨大的 test_window 或 test_question_window CSV。培优包内这些文件仍保留在原始归档中，
独立项目按 pykt 题目原生窗口协议，从完整学生序列重建目标和对应历史。普通分段文件则已经物化保存。
这不同于概念展开后的题目专用融合窗口，不将两种历史口径混称为相同协议。

当前尚未为全部目标生成 inputs.jsonl 或发起推理。百万级目标的历史 JSON 会产生大量重复数据，
测试应先选定一个小样本任务；现有 prepare 会展开所有选择的学生目标，不应直接拿全量数据做第一次测试。

## 配置与验证

三个入口配置位于 configs/datasets/peiyou.yaml、assist2009.yaml、nips_task34.yaml。
默认 split=test、validation_fold=0、题目级 pykt_window。可切换 split=train/validation 读取现有折号。
API key 路径指向项目外的 D:/1017KT/.secrets/openrouter.key；数据与密钥位置分开。

完成的校验包括：字段长度、二元响应、单学生唯一完整行、训练测试 UID 无交叉、训练验证 fold 0–4、测试 fold=-1、
来源归档/复制产物的 SHA-256，以及普通题目分段与原始序列逐项、mask、学生及 fold 的一致性。
校验明细见 data/reports/verification.json。

data/raw、data/processed、data/snapshots、data/prepared、data/reports、runs、密钥目录和压缩包已加入 Git 忽略规则。
通过临时 Git 仓库的 check-ignore 验证了这些路径被排除；本次没有上传任何数据。

处理脚本：tools/prepare_local_datasets.py；离线校验：tools/verify_local_datasets.py。
处理源代码及哈希位于 third_party/pykt/provenance.json。
