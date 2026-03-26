from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="llama.cpp 文案生成 worker")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-threads", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.load(sys.stdin)
    messages = payload["messages"]

    from llama_cpp import Llama

    llm = Llama(
        model_path=args.model_path,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=0,
        verbose=False,
    )
    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.4,
        max_tokens=1200,
        response_format={"type": "json_object"},
    )
    message = (response.get("choices") or [{}])[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        content = "\n".join(text_parts)
    json.dump({"content": content}, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
