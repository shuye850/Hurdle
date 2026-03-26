# 模块六：技术评价与反馈

模块六负责把模块五输出的技术指标转成可交付的评分结果。

当前能力：

- 加载已确定好的部署模型
- 输出单项分、阶段分、总分
- 导出评分 CSV / JSON
- 生成评分可视化图片
- 生成 HTML 报告
- 输出规则版基础建议

默认输入：

- `../5_特征融合/output/csv/technical/*_technical_metrics.csv`

默认输出：

- `output/csv/score_results.csv`
- `output/json/*_result.json`
- `output/json/score_results_all.json`
- `output/visualization/*_summary.png`
- `output/reports/*_report.html`

运行方式：

```bash
python3 src/run.py
```

只跑单个样本：

```bash
python3 src/run.py --video-id 001
```
