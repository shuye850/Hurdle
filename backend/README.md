# 跨栏动作分析后端

后端将现有七模块流水线封装为视频分析任务。LLM 默认且固定为关闭状态。

## 启动

```bash
.venv_rtmpose/bin/python3.11 -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

启动后可访问：

- 分析页面：`http://127.0.0.1:8000/`
- API 文档：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/api/health`

## 接口

- `POST /api/jobs`：使用 `video` 字段上传视频并创建分析任务。
- `GET /api/jobs`：查看任务历史。
- `GET /api/jobs/{job_id}`：查看任务状态、进度与结果文件。
- `GET /api/jobs/{job_id}/log`：查看流水线日志。
- `GET /api/jobs/{job_id}/preview`：读取 YOLO 检测过程的最新标注帧。
- `GET /api/jobs/{job_id}/previews/{kind}`：分别读取 `hurdle` 栏架检测或 `pose` 人体姿态最新标注帧。
- `GET /api/jobs/{job_id}/artifacts/{path}`：下载或预览任务结果。

任务保存在 `runtime/jobs/{job_id}`。由于原有算法模块使用公共中间目录，任务会自动排队并依次执行。
