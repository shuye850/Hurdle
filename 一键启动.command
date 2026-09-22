#!/bin/zsh

set -u
unsetopt BG_NICE

PROJECT_DIR="${0:A:h}"
PYTHON_PATH="$PROJECT_DIR/.venv_rtmpose/bin/python3.11"
WEB_URL="http://127.0.0.1:8000/"
HEALTH_URL="http://127.0.0.1:8000/api/health"

cd "$PROJECT_DIR" || exit 1

echo "========================================"
echo "  跨栏动作技术分析系统"
echo "========================================"
echo ""

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "启动失败：未找到项目 Python 环境"
  echo "$PYTHON_PATH"
  echo ""
  read "?按回车键关闭窗口..."
  exit 1
fi

if /usr/bin/curl -fsS --max-time 1 "$HEALTH_URL" >/dev/null 2>&1; then
  echo "系统已经运行，正在打开网页..."
  /usr/bin/open "$WEB_URL"
  exit 0
fi

echo "正在启动分析服务..."
"$PYTHON_PATH" "$PROJECT_DIR/WEB.py" &
SERVER_PID=$!

stop_server() {
  if /bin/kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    /bin/kill "$SERVER_PID" >/dev/null 2>&1
    wait "$SERVER_PID" 2>/dev/null
  fi
}

trap stop_server EXIT INT TERM

READY=0
for _ in {1..60}; do
  if /usr/bin/curl -fsS --max-time 1 "$HEALTH_URL" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! /bin/kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    break
  fi
  /bin/sleep 0.5
done

if [[ "$READY" -ne 1 ]]; then
  echo ""
  echo "启动失败，请查看上方错误信息。"
  echo ""
  read "?按回车键关闭窗口..."
  exit 1
fi

echo ""
echo "系统已启动：$WEB_URL"
echo "正在打开浏览器..."
echo ""
echo "请保持此窗口开启。"
echo "按 Control + C 或关闭窗口可停止系统。"
echo ""

/usr/bin/open "$WEB_URL"
wait "$SERVER_PID"
