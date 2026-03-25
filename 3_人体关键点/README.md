# 模块3 MediaPipe 增强版

基于你提供的原始模块3代码重构，保持 MediaPipe Tasks 后端不变，增强点如下：

1. 可选 ROI top-down 前置裁剪
2. ROI 平滑与周期性整帧刷新
3. 可选轻量预处理（默认关闭）
4. raw / clean 两套 CSV
5. overlay_raw / overlay_smooth 两版标定视频
6. 起跨前支撑脚轻量约束（默认开启，可关闭）

## 运行
python src/run.py
或
python src/run.py /path/to/video.mp4
