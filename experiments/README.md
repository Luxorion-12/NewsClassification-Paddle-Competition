# 验证集实验记录

实验统一使用 `database/train.txt`（646,510 条）和 `database/dev.txt`（68,736 条）。按验证 Macro-F1 比较配置。优化模型的本地留出集评估尚未执行。

| 实验 | 结果 |
| --- | --- |
| CPU 恒定学习率与续训 | [5e-5](../transformer_baseline/outputs/lr_search_20261116_132433/lr_5e-5/metrics.json)、[1e-4](../transformer_baseline/outputs/lr_search_20261116_132433/lr_1e-4_restart/metrics.json)、[2e-4（7 轮）](../transformer_baseline/outputs/lr_search_20261116_132433/lr_2e-4_restart/metrics.json) |
| CPU 单轮 Batch Size、TF-IDF 的 C 与字符片段范围 | [汇总](batch_and_tfidf_20261117_124422/summary.json) |
| GPU 七轮恒定与线性调度、三轮 Batch Size | [汇总](transformer_followup_gpu_20261117/summary.json) |
| CPU 线性调度试跑（第 4 轮后切到 GPU 重跑对照） | [逐轮指标](transformer_followup_20261117/linear_bs64_7e/history.json) |
| GPU Batch Size 32、七轮线性调度 | [逐轮指标](batch32_and_min_df_20261117_144354/batch_32_7e/metrics.json) |
| TF-IDF 的 min_df（C=1.0，字符 1–3 gram） | [2](batch32_and_min_df_20261117_144354/min_df_2/metrics.json)、[5](batch_and_tfidf_20261117_124422/svm_c1.0_char1-3/metrics.json)、[10](batch32_and_min_df_20261117_144354/min_df_10/metrics.json)、[20](batch32_and_min_df_20261117_144354/min_df_20/metrics.json) |

| 模型与配置 | 验证 Accuracy | 验证 Macro-F1 |
| --- | ---: | ---: |
| Transformer Tiny，Batch Size 32，七轮，峰值学习率 2e-4，5% 预热后线性衰减 | 93.91% | 92.90% |
| TF-IDF + LinearSVC，C=1.0，字符 1–3 gram，min_df=5 | 94.75% | 93.83% |
| TF-IDF + LinearSVC，C=1.0，字符 1–3 gram，min_df=2 | 94.79% | 93.87% |

最后两行的模型文件体积分别约为 76.2 MiB 和 240.0 MiB，因此当前推荐 `min_df=5`。Transformer 实验在 RTX 4060 Laptop GPU 上运行。启用 `--early-stop-patience 3 --early-stop-min-epochs 7` 时监控验证 Macro-F1，七轮实验跑满预算。

CPU 短时测速在 Batch Size 64、8 线程下测得训练约 26.36 ms/批、验证约 6.91 ms/批。该测速用 200 个训练批次和 100 个验证批次外推，持续运行的耗时见逐轮指标。

以下命令从仓库根目录运行，会在新目录生成模型和指标。

```powershell
python -u transformer_baseline/train.py --data-dir database --output-dir experiments/reproduce_tiny_bs32 --epochs 7 --batch-size 32 --max-length 64 --learning-rate 2e-4 --threads 8 --seed 2026 --lr-schedule linear --warmup-ratio 0.05 --early-stop-patience 3 --early-stop-min-epochs 7
python -u tf-idf_baseline/train.py --data-dir database --output-dir experiments/reproduce_tfidf --c 1.0 --ngram-range 1 3 --min-df 5 --dev-only
```
