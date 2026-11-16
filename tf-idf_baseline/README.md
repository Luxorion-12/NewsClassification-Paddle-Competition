# TF-IDF 传统分类基线

三个文件夹放到仓库根目录后，代码位于 `tf-idf_baseline/train.py`，默认读取 `dataset/clean_split`。

## 数据流

1. 从 `train.txt`、`dev.txt` 读取标题与真实类别；从 `test.txt` 只读取标题。
2. 用训练集类别建立固定的“中文类别 → 整数编号”映射。
3. 用训练标题学习字符 1-gram、2-gram 词表和 IDF，生成稀疏矩阵 `X_train`。
4. 用同一个 TF-IDF 转换器生成 `X_dev`、`X_test`；验证集和测试集不重新建立词表。
5. 用 `X_train`、`y_train` 训练 LinearSVC；在 `dev` 上计算 Accuracy 和 Macro-F1。
6. 按 `test.txt` 原顺序输出预测类别。训练脚本不读取 `test_labeled.txt`；训练结束后由独立评分脚本计算本地测试指标。

当前基线保留标题里的数字、英文、标点和大小写，使用 `min_df=5`、`C=1.0`。后续优化请只用验证集选择参数。

## 运行

在项目根目录执行：

```powershell
python -m pip install -r tf-idf_baseline/requirements.txt
python tf-idf_baseline/train.py
```

若要比较 `min_df=2`：

```powershell
python tf-idf_baseline/train.py --min-df 2
```

结果分别写入 `tf-idf_baseline/outputs/min_df_5` 和 `tf-idf_baseline/outputs/min_df_2`：

- `metrics.json`：验证集指标、各类别指标、词表大小和配置。
- `model.joblib`：分类器、TF-IDF 转换器和类别顺序，可用于之后预测。
- `predictions.txt`：与 `dataset/clean_split/test.txt` 逐行对应的中文类别预测。

脚本也支持 `--data-dir`、`--output-dir`、`--c` 参数。运行完整训练可能需要一些时间和内存。

训练完成后，若只需用已保存模型重新生成本地测试预测，无需重新训练：

```powershell
python tf-idf_baseline/train.py --predict-only
```

## 本地测试

在模型生成 `predictions.txt` 后，从项目根目录运行：

```powershell
python dataset/evaluate_test.py --predictions tf-idf_baseline/outputs/min_df_5/predictions.txt
```

评分脚本会核对预测条数以及 `test.txt` 与 `test_labeled.txt` 的逐行对应关系，再写出 `tf-idf_baseline/outputs/min_df_5/test_metrics.json`。当前固定基线的本地测试 Accuracy 为 **94.74%**，Macro-F1 为 **93.94%**。
