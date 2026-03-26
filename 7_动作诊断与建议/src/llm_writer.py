from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import requests


DEFAULT_TIMEOUT = 60
DEFAULT_LLAMA_PYTHON = Path(__file__).resolve().parents[2] / ".venv_llama_cpp" / "bin" / "python"
DEFAULT_LLAMA_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"


def _clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _extract_text_from_response(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
            if chunks:
                return "\n".join(chunks)
    return ""


def _build_prompt_payload(diagnosis: dict[str, Any]) -> dict[str, Any]:
    problems = []
    for item in diagnosis.get("top_problems", []):
        problems.append(
            {
                "task_id": item.get("task_id"),
                "task_name": item.get("task_name"),
                "stage": item.get("stage"),
                "score": item.get("score"),
                "score_round": item.get("score_round"),
                "problem_title": item.get("problem_title"),
                "diagnosis": item.get("diagnosis"),
                "impact": item.get("impact"),
                "evidence": item.get("evidence", []),
                "recommended_drills": item.get("recommended_drills", []),
                "next_focus": item.get("next_focus"),
            }
        )

    return {
        "sample_name": diagnosis.get("sample_name"),
        "video_id": diagnosis.get("video_id"),
        "overall_score": (diagnosis.get("overall_score") or {}).get("score"),
        "overall_summary": diagnosis.get("overall_summary"),
        "weakest_stage": (diagnosis.get("weakest_stage") or {}).get("stage_name"),
        "stage_diagnosis": [
            {
                "stage": item.get("stage"),
                "stage_name": item.get("stage_name") or item.get("task_name"),
                "score": item.get("score"),
                "score_round": item.get("score_round"),
                "summary": item.get("summary"),
            }
            for item in diagnosis.get("stage_diagnosis", [])
        ],
        "top_problems": problems,
        "recommended_drills": diagnosis.get("recommended_drills", []),
        "next_focus": diagnosis.get("next_focus", []),
    }


def _build_messages(diagnosis: dict[str, Any]) -> list[dict[str, str]]:
    prompt_payload = _build_prompt_payload(diagnosis)
    system_prompt = (
        "你是一名专业、克制、表达清晰的跨栏技术诊断教练。"
        "你的任务不是重新判分，而是在不改变事实的前提下，把已有规则诊断改写成更自然、可读、适合报告展示的中文文案。"
        "请严格基于输入信息，不要捏造新的技术问题、指标数值或训练练习。"
        "输出必须是 JSON，不要输出 JSON 之外的任何说明。"
    )
    user_prompt = (
        "请根据以下结构化诊断结果，生成更自然的中文报告文案。\n"
        "要求：\n"
        "1. overall_summary 写成 1 句总体判断，控制在 35 字以内。\n"
        "2. stage_diagnosis 中每个阶段保留原 stage，输出更自然的 summary，控制在 30 字以内。\n"
        "3. 对 top_problems 中每个问题，保留原 task_id，输出更自然的 problem_title、diagnosis、impact、recommended_drills、next_focus。\n"
        "4. problem_title 控制在 16 字以内；diagnosis 和 impact 各写 1 句，尽量不超过 40 字。\n"
        "5. 每个问题的 recommended_drills 只保留 2 条，表达为简洁短语，不要长句。\n"
        "6. 每个问题的 next_focus 只写 1 句，尽量不超过 40 字。\n"
        "7. recommended_drills 为 3-5 条去重后的训练建议，表达为简洁短语。\n"
        "8. next_focus 为 3-5 条下次训练观察重点，表达为完整短句，但尽量简短。\n"
        "9. 输出要明显比输入更短，不要复述长段解释。\n"
        "10. 所有内容都用中文。\n"
        "11. 不要出现“根据数据可知”“AI”“模型认为”等措辞。\n"
        "12. 如果原始信息不足，就保守润色，不要编造。\n\n"
        "请按以下 JSON 结构返回：\n"
        "{\n"
        '  "overall_summary": "string",\n'
        '  "stage_diagnosis": [\n'
        "    {\n"
        '      "stage": "string",\n'
        '      "summary": "string"\n'
        "    }\n"
        "  ],\n"
        '  "top_problems": [\n'
        "    {\n"
        '      "task_id": "string",\n'
        '      "problem_title": "string",\n'
        '      "diagnosis": "string",\n'
        '      "impact": "string",\n'
        '      "recommended_drills": ["string"],\n'
        '      "next_focus": "string"\n'
        "    }\n"
        "  ],\n"
        '  "recommended_drills": ["string"],\n'
        '  "next_focus": ["string"]\n'
        "}\n\n"
        f"输入数据：\n{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _build_summary_messages(diagnosis: dict[str, Any]) -> list[dict[str, str]]:
    payload = {
        "sample_name": diagnosis.get("sample_name"),
        "overall_score": (diagnosis.get("overall_score") or {}).get("score"),
        "overall_summary": diagnosis.get("overall_summary"),
        "weakest_stage": (diagnosis.get("weakest_stage") or {}).get("stage_name"),
        "stage_diagnosis": [
            {
                "stage": item.get("stage"),
                "stage_name": item.get("stage_name") or item.get("task_name"),
                "score": item.get("score"),
                "summary": item.get("summary"),
            }
            for item in diagnosis.get("stage_diagnosis", [])
        ],
        "recommended_drills": diagnosis.get("recommended_drills", []),
        "next_focus": diagnosis.get("next_focus", []),
    }
    system_prompt = (
        "你是一名专业、克制、表达清晰的跨栏技术诊断教练。"
        "请把已有规则诊断改写成更自然、简洁、适合报告展示的中文文案。"
        "严格基于输入，不要编造。输出必须是 JSON。"
    )
    user_prompt = (
        "请将以下结构化诊断结果改写为简洁版总评。\n"
        "要求：\n"
        "1. overall_summary 1句，35字以内。\n"
        "2. 每个 stage_diagnosis.summary 1句，30字以内。\n"
        "3. recommended_drills 输出3-5条简洁短语。\n"
        "4. next_focus 输出3-5条简短完整句。\n"
        "5. 不要重复解释，不要长段落。\n\n"
        "返回 JSON 结构：\n"
        "{\n"
        '  "overall_summary": "string",\n'
        '  "stage_diagnosis": [{"stage": "string", "summary": "string"}],\n'
        '  "recommended_drills": ["string"],\n'
        '  "next_focus": ["string"]\n'
        "}\n\n"
        f"输入数据：\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _build_problem_messages(problem: dict[str, Any]) -> list[dict[str, str]]:
    payload = {
        "task_id": problem.get("task_id"),
        "task_name": problem.get("task_name"),
        "stage": problem.get("stage"),
        "score": problem.get("score"),
        "problem_title": problem.get("problem_title"),
        "diagnosis": problem.get("diagnosis"),
        "impact": problem.get("impact"),
        "evidence": problem.get("evidence", []),
        "recommended_drills": problem.get("recommended_drills", []),
        "next_focus": problem.get("next_focus"),
    }
    system_prompt = (
        "你是一名专业、克制、表达清晰的跨栏技术诊断教练。"
        "请把单个问题的规则诊断改写成更自然、简洁、适合报告展示的中文文案。"
        "严格基于输入，不要编造。输出必须是 JSON。"
    )
    user_prompt = (
        "请改写这个单项问题。\n"
        "要求：\n"
        "1. 保留 task_id。\n"
        "2. problem_title 16字以内。\n"
        "3. diagnosis 和 impact 各1句，尽量不超过40字。\n"
        "4. recommended_drills 只保留2条简洁短语。\n"
        "5. next_focus 只写1句，尽量不超过40字。\n\n"
        "返回 JSON 结构：\n"
        "{\n"
        '  "task_id": "string",\n'
        '  "problem_title": "string",\n'
        '  "diagnosis": "string",\n'
        '  "impact": "string",\n'
        '  "recommended_drills": ["string"],\n'
        '  "next_focus": "string"\n'
        "}\n\n"
        f"输入数据：\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _request_llm(
    *,
    api_key: str,
    base_url: str,
    model: str,
    diagnosis: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": 0.4,
            "response_format": {"type": "json_object"},
            "messages": _build_messages(diagnosis),
        },
        timeout=timeout,
    )
    response.raise_for_status()
    text = _extract_text_from_response(response.json())
    if not text:
        raise ValueError("LLM 返回内容为空")
    return json.loads(_clean_json_text(text))


def _request_llama_cpp(
    *,
    python_path: str,
    model_path: str,
    messages: list[dict[str, str]],
    timeout: int,
) -> dict[str, Any]:
    worker_path = Path(__file__).resolve().parent / "llama_cpp_worker.py"
    proc = subprocess.run(
        [python_path, str(worker_path), "--model-path", model_path],
        input=json.dumps({"messages": messages}, ensure_ascii=False),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "llama.cpp worker 执行失败")
    payload = json.loads(proc.stdout)
    text = str(payload.get("content", "")).strip()
    if not text:
        raise ValueError("llama.cpp 返回内容为空")
    return json.loads(_clean_json_text(text))


def enrich_diagnosis_with_llm(
    diagnosis: dict[str, Any],
    *,
    enabled: bool = False,
    backend: str = "openai",
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    llama_python_path: str | None = None,
    llama_model_path: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    enriched = copy.deepcopy(diagnosis)
    enriched["llm_status"] = "disabled"

    if not enabled:
        return enriched

    backend = (backend or os.getenv("LLM_BACKEND", "openai")).strip().lower()
    base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    llama_python_path = llama_python_path or os.getenv("LLAMA_CPP_PYTHON", "")
    llama_model_path = llama_model_path or os.getenv("LLAMA_MODEL_PATH", "")
    if not llama_python_path and DEFAULT_LLAMA_PYTHON.exists():
        llama_python_path = str(DEFAULT_LLAMA_PYTHON)
    if not llama_model_path and DEFAULT_LLAMA_MODEL_PATH.exists():
        llama_model_path = str(DEFAULT_LLAMA_MODEL_PATH)

    try:
        if backend == "llama_cpp":
            if not llama_python_path:
                enriched["llm_status"] = "missing_llama_python"
                return enriched
            if not llama_model_path:
                enriched["llm_status"] = "missing_model_path"
                return enriched
            summary_output = _request_llama_cpp(
                python_path=llama_python_path,
                model_path=llama_model_path,
                messages=_build_summary_messages(diagnosis),
                timeout=timeout,
            )
            problem_outputs = []
            for problem in diagnosis.get("top_problems", []):
                problem_outputs.append(
                    _request_llama_cpp(
                        python_path=llama_python_path,
                        model_path=llama_model_path,
                        messages=_build_problem_messages(problem),
                        timeout=timeout,
                    )
                )
            llm_output = {
                "overall_summary": summary_output.get("overall_summary"),
                "stage_diagnosis": summary_output.get("stage_diagnosis", []),
                "recommended_drills": summary_output.get("recommended_drills", []),
                "next_focus": summary_output.get("next_focus", []),
                "top_problems": problem_outputs,
            }
            enriched["llm_python_path"] = llama_python_path
            enriched["llm_model_path"] = llama_model_path
        else:
            if not api_key:
                enriched["llm_status"] = "missing_api_key"
                return enriched
            llm_output = _request_llm(
                api_key=api_key,
                base_url=base_url,
                model=model,
                diagnosis=diagnosis,
                timeout=timeout,
            )
    except Exception as exc:
        enriched["llm_status"] = "error"
        enriched["llm_error"] = str(exc)
        return enriched

    enriched["rule_based_overall_summary"] = diagnosis.get("overall_summary", "")
    enriched["rule_based_top_problems"] = copy.deepcopy(diagnosis.get("top_problems", []))
    enriched["rule_based_recommended_drills"] = list(diagnosis.get("recommended_drills", []))
    enriched["rule_based_next_focus"] = list(diagnosis.get("next_focus", []))

    overall_summary = llm_output.get("overall_summary")
    if isinstance(overall_summary, str) and overall_summary.strip():
        enriched["overall_summary"] = overall_summary.strip()

    llm_problem_map: dict[str, dict[str, Any]] = {}
    for item in llm_output.get("top_problems", []):
        if isinstance(item, dict):
            task_id = str(item.get("task_id", "")).strip()
            if task_id:
                llm_problem_map[task_id] = item

    llm_stage_map: dict[str, dict[str, Any]] = {}
    for item in llm_output.get("stage_diagnosis", []):
        if isinstance(item, dict):
            stage = str(item.get("stage", "")).strip()
            if stage:
                llm_stage_map[stage] = item

    rewritten_stage_diagnosis = []
    for item in diagnosis.get("stage_diagnosis", []):
        stage = str(item.get("stage", "")).strip()
        llm_item = llm_stage_map.get(stage, {})
        merged = copy.deepcopy(item)
        summary = llm_item.get("summary")
        if isinstance(summary, str) and summary.strip():
            merged["summary"] = summary.strip()
        rewritten_stage_diagnosis.append(merged)
    enriched["stage_diagnosis"] = rewritten_stage_diagnosis

    rewritten_problems = []
    for item in diagnosis.get("top_problems", []):
        task_id = str(item.get("task_id", "")).strip()
        llm_item = llm_problem_map.get(task_id, {})
        merged = copy.deepcopy(item)
        for key in ("problem_title", "diagnosis", "impact", "next_focus"):
            value = llm_item.get(key)
            if isinstance(value, str) and value.strip():
                merged[key] = value.strip()
        drills = llm_item.get("recommended_drills")
        if isinstance(drills, list):
            cleaned = [str(x).strip() for x in drills if str(x).strip()]
            if cleaned:
                merged["recommended_drills"] = cleaned
        rewritten_problems.append(merged)
    enriched["top_problems"] = rewritten_problems

    recommended_drills = llm_output.get("recommended_drills")
    if isinstance(recommended_drills, list):
        cleaned = [str(x).strip() for x in recommended_drills if str(x).strip()]
        if cleaned:
            enriched["recommended_drills"] = list(dict.fromkeys(cleaned))[:5]

    next_focus = llm_output.get("next_focus")
    if isinstance(next_focus, list):
        cleaned = [str(x).strip() for x in next_focus if str(x).strip()]
        if cleaned:
            enriched["next_focus"] = list(dict.fromkeys(cleaned))[:5]

    enriched["llm_status"] = "ok"
    enriched["llm_backend"] = backend
    if backend == "openai":
        enriched["llm_model"] = model
        enriched["llm_base_url"] = base_url
    return enriched
