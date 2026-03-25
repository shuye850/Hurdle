毕业论文Ⅱ/
├── 0_项目流程总结/
│   ├── 0_技术细节文本.docx
│   ├── 0_开发流程指南.docx
│   └── 0_术语与字段规范.md
│
├── 1_视频预处理/
│   ├── raw_videos/
│   ├── clips_one_hurdle/
│   ├── scripts/
│   ├── outputs/
│   │   ├── meta/
│   │   ├── normalized_videos/
│   │   └── frames_optional/
│   └── README.md
│
├── 2_栏架识别/
│   ├── data/
│   ├── weights/
│   ├── runs/
│   ├── infer/
│   │   ├── src/
│   │   │   ├── run.py
│   │   │   ├── postprocess.py
│   │   │   └── visualize.py
│   │   └── outputs/
│   │       ├── det_jsonl/
│   │       ├── hurdle_top_jsonl/
│   │       └── vis_overlay/
│   └── README.md
│
├── 3_人体关键点/
│   ├── config/
│   │   ├── pose.yaml
│   │   └── keypoints_mediapipe33.yaml
│   ├── models/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── run.py
│   │   ├── io_video.py
│   │   ├── pose_mediapipe.py
│   │   ├── export_csv.py
│   │   ├── postprocess_optional.py
│   │   └── visualize_optional.py
│   ├── outputs/
│   │   ├── pose_raw/
│   │   ├── pose_clean/
│   │   └── vis_overlay/
│   └── README.md
│
├── 4_阶段划分与轻量融合/
│   ├── config/
│   │   ├── thresholds.yaml
│   │   └── phase_rules.md
│   ├── src/
│   │   ├── run.py
│   │   ├── events.py
│   │   ├── fuse_minimal.py
│   │   └── visualize.py
│   ├── outputs/
│   │   ├── fusion_minimal/
│   │   ├── phases_json/
│   │   └── vis_overlay/
│   └── README.md
│
├── 5_特征提取与评价特征构建/
│   ├── config/
│   │   ├── feature_list.yaml
│   │   └── filters.yaml
│   ├── src/
│   │   ├── metrics.py
│   │   ├── features.py
│   │   ├── align_window.py
│   │   └── visualize.py
│   ├── outputs/
│   │   ├── metrics/
│   │   ├── features/
│   │   └── plots/
│   └── README.md
│
├── 6_人工标注与教师评价/
│   ├── templates/
│   ├── labels_raw/
│   ├── labels_clean/
│   └── README.md
│
├── 7_评价模型训练与推理/
│   ├── config/
│   │   ├── split.yaml
│   │   └── model.yaml
│   ├── src/
│   │   ├── train_baseline.py
│   │   ├── train_sequence.py
│   │   ├── eval.py
│   │   └── infer.py
│   ├── models/
│   ├── reports/
│   └── README.md
│
├── 8_系统集成与总入口/
│   ├── run_analysis.py
│   ├── configs/
│   │   ├── default.yaml
│   │   └── paths_win.yaml
│   ├── outputs/
│   │   ├── per_video/
│   │   └── debug/
│   └── README.md
│
├── 9_实验记录与版本管理/
│   ├── changelog.md
│   ├── experiments/
│   └── notes/
│
├── outputs/
│   └── _README_outputs.md
│
└── README.md