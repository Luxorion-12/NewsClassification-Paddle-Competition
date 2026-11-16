# 小型预训练 Transformer 基线

三个文件夹放到仓库根目录后，代码位于 `transformer_baseline/train.py`。默认模型是本地下载的
[`uer/chinese_roberta_L-2_H-128`](https://huggingface.co/uer/chinese_roberta_L-2_H-128)，
一个两层、隐藏维度 128 的中文预训练模型。它的分类头在本任务训练时新建。

## 数据流

1. 读取 `dataset/clean_split/train.txt` 和 `dev.txt` 的标题与类别。
2. 用训练类别建立固定编号表。训练批次提供类别编号；验证批次只提供标题输入，预测后与真实类别比较。
3. 使用与预训练模型配套的 tokenizer，把标题转换为 `input_ids` 和 `attention_mask`。批次内动态填充；最多保留 64 个词元。
4. 训练时根据损失更新模型；每轮在 `dev` 上计算 Accuracy 和 Macro-F1，保存 Macro-F1 最好的模型。
5. 可选 `--predict-test`，按 `dataset/clean_split/test.txt` 顺序输出预测。训练脚本不读取 `test_labeled.txt`；训练结束后由独立评分脚本计算本地测试指标。

## 运行

在项目根目录执行完整数据训练：

```powershell
python -m pip install -r transformer_baseline/requirements.txt
python -u transformer_baseline/train.py
```

本机只有 CPU。可以先用一个明确标记为**小样本试运行**的配置确认速度：

```powershell
python -u transformer_baseline/train.py --train-limit 20000 --dev-limit 5000 --output-dir transformer_baseline/outputs/tiny_pilot
```

`metrics.json` 保存验证指标；`model/` 保存微调后的模型及 tokenizer。小样本分数不能与使用全部数据的 TF-IDF 分数直接比较。

完整训练后，可以直接使用保存的模型生成测试标题预测，无需重新训练：

```powershell
python -u transformer_baseline/train.py --predict-only
```

输出 `transformer_baseline/outputs/tiny/predictions.txt`，行顺序与 `dataset/clean_split/test.txt` 相同。

## 本地测试

在模型生成 `predictions.txt` 后，从项目根目录运行：

```powershell
python dataset/evaluate_test.py --predictions transformer_baseline/outputs/tiny/predictions.txt
```

评分结果写入 `transformer_baseline/outputs/tiny/test_metrics.json`。当前模型使用完整训练集、训练 1 轮，测试 Accuracy 为 **91.29%**，Macro-F1 为 **89.87%**。这是小型预训练模型的初始基线，后续优化请只用验证集选择参数。
