# 模块七：动作诊断与建议

模块七负责读取模块六的评分结果，并结合模块五的单样本技术指标，输出面向上传者的动作问题诊断与训练建议。

当前能力：

- 读取模块六结构化评分结果
- 读取模块五单样本技术指标
- 识别最弱阶段与前三个优先问题
- 输出结构化 JSON
- 输出汇总 CSV
- 生成 Markdown 报告

默认输入：

- `../6_技术评价与反馈/output/json/score_results_all.json`
- `../5_特征融合/output/csv/technical/*_technical_metrics.csv`

默认输出：

- `output/json/*_diagnosis.json`
- `output/json/diagnosis_all.json`
- `output/csv/diagnosis_summary.csv`
- `output/reports/*_advice.md`

运行方式：

```bash
python3 src/run.py
```

只跑单个样本：

```bash
python3 src/run.py --video-id 039
```
