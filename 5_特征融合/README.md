# 模块五：特征融合与指标计算

## 输入
本模块优先读取以下上游输出：
- 栏架识别：`001.csv`
- 人体关键点：`001_keypoints.csv`
- 阶段划分事件：`001_events.json`
- 阶段划分阶段：`001_phase_segments.json`

其中：
- 事件 JSON 中已包含 `direction_sign / swing_leg / takeoff_leg / takeoff_touchdown / takeoff_toeoff / swing_touchdown / swing_toeoff / cross_bar_row`
- 阶段 JSON 中已包含 `起跨 / 腾空 / 落地` 三段

## 输出目录

技术指标 CSV 现包含腿长指标，定义为 `大腿长度 + 小腿长度`，并同时输出像素值与按栏高换算后的米制值。

### 单样本 CSV
- `output/csv/technical/*_technical_metrics.csv`
- `output/csv/event/*_event_frames.csv`
- `output/csv/frame/*_frame_features.csv`

### 汇总 CSV
- `output/csv/summary/technical_metrics_all.csv`
- `output/csv/summary/event_frames_all.csv`
- `output/csv/summary/frame_features_all.csv`

### 视频
- `output/video/*_fusion_result.mp4`

### 可视化
- `output/visualization/*_fusion_summary.png`

## 运行
```bash
python run.py
