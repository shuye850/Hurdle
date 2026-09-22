from __future__ import annotations

import argparse

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动跨栏动作技术分析网页")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--reload", action="store_true", help="开发时自动重载")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"分析页面: http://{args.host}:{args.port}/")
    print(f"API 文档: http://{args.host}:{args.port}/docs")
    uvicorn.run("backend.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
