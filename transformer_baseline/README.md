# С��Ԥѵ�� Transformer ����

�����ļ��зŵ��ֿ��Ŀ¼�󣬴���λ�� `transformer_baseline/train.py`��Ĭ��ģ���Ǳ������ص�
[`uer/chinese_roberta_L-2_H-128`](https://huggingface.co/uer/chinese_roberta_L-2_H-128)��
һ�����㡢����ά�� 128 ������Ԥѵ��ģ�͡����ķ���ͷ�ڱ�����ѵ��ʱ�½���

## ������

1. ��ȡ `dataset/clean_split/train.txt` �� `dev.txt` �ı��������
2. ��ѵ��������̶���ű���ѵ�������ṩ����ţ���֤����ֻ�ṩ�������룬Ԥ�������ʵ���Ƚϡ�
3. ʹ����Ԥѵ��ģ�����׵� tokenizer���ѱ���ת��Ϊ `input_ids`��`attention_mask` �� `token_type_ids`�������ڶ�̬��䣻��ౣ�� 64 ����Ԫ��
4. ѵ��ʱ������ʧ����ģ�ͣ�ÿ���� `dev` �ϼ��� Accuracy �� Macro-F1���� `--epochs` ���겢�������һ��ģ�͡�
5. ��ѡ `--predict-test`���� `dataset/clean_split/test.txt` ˳�����Ԥ�⡣ѵ���ű�����ȡ `test_labeled.txt`��ѵ���������ɶ������ֽű����㱾�ز���ָ�ꡣ

�ű���ʹ����ͣ����֤����ֻ���ڹ۲�ÿ��Ч�������ı�ѵ��������

## ����

����Ŀ��Ŀ¼ִ����������ѵ����

```powershell
python -m pip install -r transformer_baseline/requirements.txt
python -u transformer_baseline/train.py
```

����ֻ�� CPU����������һ����ȷ���Ϊ**С����������**������ȷ���ٶȣ�

```powershell
python -u transformer_baseline/train.py --train-limit 20000 --dev-limit 5000 --output-dir transformer_baseline/outputs/tiny_pilot
```

`metrics.json` ������ָ֤�ꣻ`model/` ����΢�����ģ�ͼ� tokenizer��С��������������ʹ��ȫ�����ݵ� TF-IDF ����ֱ�ӱȽϡ�

����ѵ���󣬿���ֱ��ʹ�ñ����ģ�����ɲ��Ա���Ԥ�⣬��������ѵ����

```powershell
python -u transformer_baseline/train.py --predict-only
```

��� `transformer_baseline/outputs/tiny/predictions.txt`����˳���� `dataset/clean_split/test.txt` ��ͬ��

## ���ز���

��ģ������ `predictions.txt` �󣬴���Ŀ��Ŀ¼���У�

```powershell
python dataset/evaluate_test.py --predictions transformer_baseline/outputs/tiny/predictions.txt
```

���ֽ��д�� `transformer_baseline/outputs/tiny/test_metrics.json`����ǰģ��ʹ������ѵ������ѵ�� 1 �֣����� Accuracy Ϊ **91.29%**��Macro-F1 Ϊ **89.87%**������С��Ԥѵ��ģ�͵ĳ�ʼ���ߣ������Ż���ֻ����֤��ѡ�������
