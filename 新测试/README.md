# 新闻标题分类复现实验

本目录在原基线上加入特征融合、MacBERT-base 微调和训练集检索。算法原理、原基线对照、结果限制及 Git LFS 下载说明见 [算法更新说明.md](算法更新说明.md)，实测结果见 [RESULTS.md](RESULTS.md)。

本目录复用项目原来的 `../database` 固定划分。训练读 `train.txt`，选择配置和检查点读 `dev.txt`，预测读不含标签的 `test.txt`。最后由独立评分脚本读取 `test_labeled.txt`，逐行验证标题顺序并计算 Accuracy、Macro-F1 和混淆矩阵。官方无标签测试集未用于本次实验。

## 模型

- **TF-IDF + LinearSVC**。融合原始字符 1–3 gram 特征和归一化字符 1–4 gram 特征的两个线性 SVM。对大小写、全半角、数字串和空白做归一化。检索通道以训练集中的字符片段构建索引，根据各类别最相似训练标题的余弦相似度调整分数。另比较标题首尾和格式特征、以及常见错误类别对专用的二分类 SVM。
- **Transformer**。将 2 层 Tiny 换为 12 层 MacBERT-base。使用 BF16、按长度分桶、AdamW、6.5% 预热和余弦学习率。对同一验证集按 Accuracy 选检查点。单独报告 MacBERT，并对照只加入训练集相似标题检索的版本。[MacBERT 预训练权重](https://huggingface.co/hfl/chinese-macbert-base/tree/a986e004d2a7f2a1c2f5a3edef4e20604a974ed1)

## 运行与评分

以下命令在本目录用 PowerShell 执行。两个解释器和 Transformer 包的路径对应本机已有环境。更换机器时将它们改为对应环境路径即可，运行所需包见下方。

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_HOME = Join-Path $PWD 'cache'
$svmPython = 'C:/shuju/envs/torch/python.exe'
$gpuPython = 'C:/shuju/envs/shenduxuexi/python.exe'
$env:PYTHONPATH = (Resolve-Path '../experiments/transformer_followup_20261117/gpu_packages').Path

# 核对数据
& $svmPython -X utf8 verify_split.py

# 训练 TF-IDF 路线
& $svmPython -X utf8 -u tfidf_retrieval.py
& $svmPython -X utf8 -u tfidf_pairwise.py
& $svmPython -X utf8 -u tfidf_position.py

# 训练 Transformer 路线。沿用原参数时，可追加 --resume 从最近完整轮次恢复
& $gpuPython -X utf8 -u train_macbert.py
& $svmPython -X utf8 -u macbert_retrieval.py
```

`baseline_svm.joblib` 是从项目中已有的 `min_df_2` 模型复制而来。它按交接记录由相同的 646,510 条训练标题拟合。`pretrained/macbert-base` 中的模型文件已下载到本目录，离线微调和预测可直接使用。当前使用 scikit-learn 1.7.2、SciPy 1.15.3、joblib，以及 GPU 环境的 PyTorch 2.8.0+cu126、Transformers 4.46.3、scikit-learn 1.6.1。BF16 训练使用支持 BF16 的 CUDA GPU。

从落盘模型重出预测时，给相应训练命令追加 `--predict-only`。MacBERT 检索对照命令直接读取落盘后的 Dev 和测试 logits。评分示例如下，`outputs/tfidf` 可替换为某次运行的输出目录。

末两轮权重平均的对照可在训练完成后运行 `& $gpuPython -X utf8 -u macbert_average.py`。实验已经保存了第 2 轮的权重。若从头重训，在第 2 轮的 `last_checkpoint.pt` 保存后、第 3 轮覆盖该文件前，用另一终端运行 `macbert_average.py --snapshot-epoch2`。

```powershell
& $svmPython -X utf8 evaluate_test.py --data-dir ../database --predictions outputs/tfidf/predictions.txt
```

`metrics.json` 记录 Dev 结果及配置。`test_metrics.json` 记录固定测试的准确率、Macro-F1、各类别指标和预测文件哈希。测试集有 71,826 条，达到 98% 至少需答对 70,390 条。

