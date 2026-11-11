# 中文新闻标题分类：PyTorch baseline

这个独立 baseline 使用字符 unigram/bigram、`EmbeddingBag(mode="mean")` 和线性分类层预测 14 个新闻类别。训练时仅从训练集建立词表；验证集只用于选取模型；测试集只用于生成预测。

## 数据

默认读取 `../SkrnjMg2/` 中的 `train.txt`、`dev.txt`、`test.txt`：

- `train.txt`、`dev.txt`：每行 `新闻标题<TAB>类别`。
- `test.txt`：每行一个新闻标题。

当前本地文件分别有 752,471、80,000、83,599 行。原始文件不应移动到本目录。

## 运行

在仓库根目录执行：

```powershell
python -m pip install -r NewsClassification-Paddle-Competition/news_baseline/requirements.txt
python NewsClassification-Paddle-Competition/news_baseline/train.py --train-limit 50000 --dev-limit 10000 --epochs 2 --output-dir NewsClassification-Paddle-Competition/news_baseline/output_smoke
python NewsClassification-Paddle-Competition/news_baseline/train.py --epochs 5 --output-dir NewsClassification-Paddle-Competition/news_baseline/output
```

第一条训练命令是快速检查流程，所得分数不能当作完整训练结果。第二条使用全部数据。CPU 上训练较慢；若有 CUDA，脚本会自动使用 GPU。可用 `--data-dir` 指定其他数据目录，`--help` 查看其余参数。

输出：`metrics.json`（每轮损失、验证 Accuracy、Macro-F1 和各类别报告）、`model.pt`、`vocab.json`、`result.txt`、`submission.zip`。`result.txt` 与测试文件顺序一致，一行一个中文类别；ZIP 内仅含这个文件。提交前请以竞赛页面当前要求为准，特别核对文件名和压缩格式。

本地数据中有 37,951 条验证标题在训练集中原样出现，完整验证集的 Accuracy 因而偏高。脚本额外报告 `unseen_dev_accuracy` 和 `unseen_dev_macro_f1`，只评估训练集中未出现过的验证标题，并用 `unseen_dev_accuracy` 选择最佳 epoch。完整验证集分数仍保留，便于和其他 baseline 对比。这个划分只排除完全相同的标题，不能排除近似重复。

本机 CPU 上用默认参数完整运行的结果：最佳 epoch 为 3；完整验证集 Accuracy 0.9643、Macro-F1 0.9610；未见标题的 42,049 条验证样本上 Accuracy 0.9675、Macro-F1 0.8762。未见标题的类别分布很偏，其中一个类别只有 1 条，因此这个子集的 Macro-F1 波动较大。第 5 轮的完整验证集 Accuracy 虽然达到 0.9700，但未见标题 Accuracy 略低于第 3 轮；脚本按事先约定的未见标题 Accuracy 保存第 3 轮模型。这些分数是本地验证结果，不是竞赛测试集得分。

## 作业报告可以解释的部分

1. 任务：根据新闻标题做 14 类单标签分类；指标主要看验证集 Accuracy，同时报告 Macro-F1。
2. 特征：中文字符及相邻双字符，不依赖分词词典；未登录特征映射到 `<UNK>`。
3. 模型：对一个标题内的特征向量取平均，再用线性层预测类别；损失函数为交叉熵。
4. 实验：固定随机种子；只在训练文件上更新参数，并明确处理训练与验证文件间的重复标题；依据未见标题的验证准确率选择 epoch。
5. 分析：从 `metrics.json` 找最弱类别，观察易混类别，再考虑加入词级特征或改成 TextCNN。

数据来源：[飞桨学习赛：中文新闻文本标题分类](https://aistudio.baidu.com/competition/detail/809/0/related-material)。竞赛页面正文由动态页面加载；提交文件的细节可参考公开的[社区 baseline](https://github.com/livingbody/Chinese_New_Title_Classification)，以竞赛页面当前规则为准。
