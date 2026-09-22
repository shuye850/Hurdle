# 跨栏动作技术分析系统

系统通过单栏视频完成栏架识别、人体关键点检测、动作阶段划分、技术指标计算、评分和规则诊断。目前不启用 LLM。

## 启动分析网页

macOS 可直接双击项目根目录中的 `一键启动.command`。

也可以在终端运行：

```bash
cd /Users/xisenberg/test/system2
.venv_rtmpose/bin/python3.11 WEB.py
```

浏览器访问：

- 分析页面：`http://127.0.0.1:8000/`
- API 文档：`http://127.0.0.1:8000/docs`

网页支持视频上传、分析进度、历史任务、技术评分、问题诊断、训练建议以及报告和数据文件查看。任务结果保存在 `runtime/jobs/{任务ID}`。

## 命令行运行

```bash
.venv_rtmpose/bin/python3.11 RUN.py
```

默认读取 `input/` 中的视频，并将最终结果写入 `output/`。

