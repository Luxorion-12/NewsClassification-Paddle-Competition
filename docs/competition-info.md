# 比赛基础信息

## 概览

| 项目 | 内容 |
| --- | --- |
| 比赛名称 | 飞桨学习赛：中文新闻文本标题分类 |
| 平台 | 百度飞桨 AI Studio 星河社区 |
| 任务类型 | 中文新闻标题多类别分类 |
| 数据来源说明 | 基于 THUCNews |
| 页面显示周期 | 2023-02-17 至 2030-12-31 |
| 页面显示评比方式 | 月度评比；每月为一个竞赛周期 |
| 页面显示奖项 | 积分 |

以上信息根据比赛介绍页和赛题定义页整理，页面信息可能更新，参赛前应以平台实时展示为准。

## 赛题简介

参赛者根据新闻标题内容训练分类模型，并为测试标题预测所属类别。AI Studio 的赛题定义页说明，该任务基于 THUCNews；该语料由新浪新闻 RSS 频道 2005 至 2011 年间的历史数据筛选整理，包含约 74 万篇新闻文档。

比赛介绍列出的 14 个类别为：

财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐。

## 参赛与提交要点

平台页面说明：

- 仅限个人参赛。
- 需要使用 PaddlePaddle 2.0 或更高版本。
- 提交结果文件名为 `result.txt`，压缩为 ZIP 后上传；文件每行一个类别标签。
- 页面列有积分奖励，并说明提供 Tesla V100 GPU 算力。算力的具体领取条件及当下可用性以 AI Studio 页面为准。
- 榜单按月评比，记录参赛者提交的最高成绩。

公开页面当前可读取的内容没有明确确认底层评分公式，因此本仓库不预设 Accuracy 等指标。请在提交前查看比赛规则、提交说明及评测反馈。

## 官方链接

- [比赛介绍](https://aistudio.baidu.com/competition/detail/809/0/introduction)
- [赛题定义](https://aistudio.baidu.com/competition/detail/809/0/task-definition)
- [相关学习资料](https://aistudio.baidu.com/competition/detail/809/0/related-material)
- [提交结果](https://aistudio.baidu.com/competition/detail/809/0/submit-result)
- [排行榜](https://aistudio.baidu.com/competition/detail/809/0/leaderboard)
- [飞桨官网](https://www.paddlepaddle.org.cn/)
- [AI Studio 平台](https://aistudio.baidu.com/)
- [PaddleNLP 基线项目](https://aistudio.baidu.com/aistudio/projectdetail/1709811)
- [AI Studio 基线项目：中文新闻文本标题分类](https://aistudio.baidu.com/aistudio/projectdetail/2311230)

## 来源说明

主要信息来自上述百度 AI Studio 官方页面。平台可能要求登录后才能访问数据、完整规则或提交功能。具体字段、标签编码、数据拆分和文件格式应以实际下载的数据包及赛事说明为准。
