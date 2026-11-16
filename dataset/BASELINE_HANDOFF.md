# 新闻标题分类基线交接

三个目录 `dataset/`、`tf-idf_baseline/`、`transformer_baseline/` 直接放在仓库根目录。以下命令均从仓库根目录运行。

## 数据与评估协议

- `dataset/clean_split/train.txt`：646,510 条，训练模型及拟合 TF-IDF 词表。
- `dataset/clean_split/dev.txt`：68,736 条，比较模型与超参数。
- `dataset/clean_split/test.txt`：71,826 条，仅标题，按原顺序生成预测。
- `dataset/clean_split/test_labeled.txt`：与测试标题逐行对应的真实类别，只用于最终本地评分。
- 14 个类别；精确重复和冲突标题已处理。划分依据、随机种子和类别数量见 `dataset/clean_split/README.md`、`report.json`、`class_counts.csv`。

两套现有模型均只用 `train.txt` 训练，在 `dev.txt` 上验证；已保存的预测文件在读取 `test_labeled.txt` 之前生成。下面的本地测试分数是固定基线的记录。后续优化在 `dev.txt` 上进行；若反复用测试分数选参数，测试集就不能再视作独立评估。

## 已完成结果

| 模型 | 配置摘要 | Dev Accuracy | Dev Macro-F1 | Test Accuracy | Test Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| TF-IDF + LinearSVC | 字符 1–2 gram，`min_df=5`，`C=1.0` | 94.62% | 93.73% | **94.74%** | **93.94%** |
| Transformer Tiny | `uer/chinese_roberta_L-2_H-128`，长度 64，批次 64，1 轮 | 91.24% | 89.64% | 91.29% | 89.87% |

测试分数来自完整的 71,826 条本地测试标题。类别指标、混淆矩阵和预测文件哈希分别在：

- `tf-idf_baseline/outputs/min_df_5/test_metrics.json`
- `transformer_baseline/outputs/tiny/test_metrics.json`

两个模型在“游戏”类的测试 F1 都相对较低。TF-IDF 的主要混淆包括“股票→科技”“科技→股票”“财经→股票”；Transformer Tiny 在这些类别之间的混淆更多。细节见两份 `test_metrics.json` 的 `largest_confusions`。

## 复现命令

TF-IDF：

```powershell
python -m pip install -r tf-idf_baseline/requirements.txt
python tf-idf_baseline/train.py
python tf-idf_baseline/train.py --predict-only
python dataset/evaluate_test.py --predictions tf-idf_baseline/outputs/min_df_5/predictions.txt
```

Transformer Tiny：

```powershell
python -m pip install -r transformer_baseline/requirements.txt
python -u transformer_baseline/train.py
python -u transformer_baseline/train.py --predict-only
python dataset/evaluate_test.py --predictions transformer_baseline/outputs/tiny/predictions.txt
```

`transformer_baseline/pretrained/chinese_roberta_tiny/` 中的本地预训练权重与 tokenizer 必须保留，或从模型来源重新取得。当前环境使用 CPU 训练 Transformer；小样本试运行位于 `transformer_baseline/outputs/tiny_pilot/`，其分数不可与完整数据结果直接比较。

## 交接给优化者

- 保持这份数据划分与测试标题顺序不变。每次实验使用独立的 `--output-dir`，保留上述固定基线产物。
- TF-IDF 可在验证集上比较 `min_df`、`C`、字符片段长度等设置，并同时记录词表大小、Accuracy 与 Macro-F1。
- Transformer Tiny 目前只训练 1 轮；可在验证集上比较训练轮数、学习率、输入长度等设置。若更换预训练模型，tokenizer 必须与模型配套。
- 模型和参数定稿后再运行本地测试评分。已经公开的这两份测试分数只作为基线记录，不用于后续参数选择。
