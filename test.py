# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib import import_module, metadata
from pathlib import Path
from typing import Any


def get_dist_version(pkg_name: str) -> str:
    try:
        return metadata.version(pkg_name)
    except metadata.PackageNotFoundError:
        return "未安装"
    except Exception as e:
        return f"获取失败: {e}"


def try_import(module_name: str) -> tuple[bool, str]:
    try:
        mod = import_module(module_name)
        ver = getattr(mod, "__version__", None)
        if ver is None:
            try:
                ver = get_dist_version(module_name)
            except Exception:
                ver = "未知"
        return True, str(ver)
    except Exception as e:
        return False, f"导入失败: {type(e).__name__}: {e}"


def get_torch_details() -> dict[str, Any]:
    info: dict[str, Any] = {}

    try:
        import torch

        info["torch"] = getattr(torch, "__version__", "未知")
        info["cuda_is_available"] = torch.cuda.is_available()
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["cuDNN_version"] = (
            torch.backends.cudnn.version()
            if hasattr(torch.backends, "cudnn")
            else None
        )

        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            devices = []
            for i in range(torch.cuda.device_count()):
                try:
                    devices.append(
                        {
                            "index": i,
                            "name": torch.cuda.get_device_name(i),
                            "capability": torch.cuda.get_device_capability(i),
                        }
                    )
                except Exception as e:
                    devices.append({"index": i, "error": str(e)})
            info["cuda_devices"] = devices
        else:
            info["cuda_device_count"] = 0
            info["cuda_devices"] = []

        # Apple Silicon / macOS MPS
        try:
            info["mps_is_available"] = bool(torch.backends.mps.is_available())
            info["mps_is_built"] = bool(torch.backends.mps.is_built())
        except Exception:
            info["mps_is_available"] = False
            info["mps_is_built"] = False

    except Exception as e:
        info["torch"] = f"导入失败: {type(e).__name__}: {e}"

    return info


def run_cmd(cmd: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return True, result.stdout.strip()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def print_block(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    output_dir = Path.cwd() / "env_check_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print_block("1. Python 与系统信息")
    print(f"Python 可执行文件: {sys.executable}")
    print(f"Python 版本: {sys.version}")
    print(f"平台: {platform.platform()}")
    print(f"系统: {platform.system()} {platform.release()}")
    print(f"机器架构: {platform.machine()}")

    print_block("2. 关键库版本检查")

    packages_to_check: list[tuple[str, str]] = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("torchaudio", "torchaudio"),
        ("mmcv", "mmcv"),
        ("mmengine", "mmengine"),
        ("mmdet", "mmdet"),
        ("mmpose", "mmpose"),
        ("openmim", "mim"),
        ("ultralytics", "ultralytics"),
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("onnx", "onnx"),
        ("onnxruntime", "onnxruntime"),
    ]

    results: dict[str, dict[str, Any]] = {}

    for module_name, dist_name in packages_to_check:
        imported, detail = try_import(module_name)
        dist_ver = get_dist_version(dist_name)
        results[module_name] = {
            "distribution_name": dist_name,
            "distribution_version": dist_ver,
            "import_status": "成功" if imported else "失败",
            "import_detail": detail,
        }

    for module_name, info in results.items():
        print(f"{module_name:<12} | pip版本: {info['distribution_version']}")
        print(f"{'':<12} | 导入状态: {info['import_status']}")
        print(f"{'':<12} | 详情: {info['import_detail']}")

    print_block("3. Torch / CUDA / MPS 详细信息")
    torch_info = get_torch_details()
    print(json.dumps(torch_info, ensure_ascii=False, indent=2))

    print_block("4. pip / conda 信息")

    ok, pip_v = run_cmd([sys.executable, "-m", "pip", "--version"])
    print("[pip --version]")
    print(pip_v if ok else f"执行失败: {pip_v}")

    ok, pip_list = run_cmd([sys.executable, "-m", "pip", "list"])
    print("\n[pip list 前若干行]")
    if ok:
        pip_list_lines = pip_list.splitlines()
        print("\n".join(pip_list_lines[:40]))
        if len(pip_list_lines) > 40:
            print("... (已省略，完整内容已写入文件)")
    else:
        print(f"执行失败: {pip_list}")

    ok, conda_v = run_cmd(["conda", "--version"])
    print("\n[conda --version]")
    print(conda_v if ok else f"未检测到 conda 或执行失败: {conda_v}")

    print_block("5. 导出完整报告")

    report = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "packages": results,
        "torch_details": torch_info,
    }

    json_path = output_dir / "env_report.json"
    txt_path = output_dir / "env_report.txt"
    pip_list_path = output_dir / "pip_list.txt"

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    txt_lines: list[str] = []
    txt_lines.append("Python 与系统信息")
    txt_lines.append(f"Python 可执行文件: {sys.executable}")
    txt_lines.append(f"Python 版本: {sys.version}")
    txt_lines.append(f"平台: {platform.platform()}")
    txt_lines.append(f"系统: {platform.system()} {platform.release()}")
    txt_lines.append(f"机器架构: {platform.machine()}")
    txt_lines.append("")

    txt_lines.append("关键库版本检查")
    for module_name, info in results.items():
        txt_lines.append(f"{module_name}")
        txt_lines.append(f"  distribution_name: {info['distribution_name']}")
        txt_lines.append(f"  distribution_version: {info['distribution_version']}")
        txt_lines.append(f"  import_status: {info['import_status']}")
        txt_lines.append(f"  import_detail: {info['import_detail']}")
        txt_lines.append("")

    txt_lines.append("Torch / CUDA / MPS 详细信息")
    txt_lines.append(json.dumps(torch_info, ensure_ascii=False, indent=2))

    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    if ok:
        pip_list_path.write_text(pip_list, encoding="utf-8")
    else:
        pip_list_path.write_text(f"获取失败: {pip_list}", encoding="utf-8")

    print(f"JSON 报告: {json_path}")
    print(f"TXT 报告:  {txt_path}")
    print(f"pip 列表:   {pip_list_path}")

    print_block("6. 建议你重点关注")
    print("1) mmpose / mmdet / mmcv / mmengine 是否都已安装且能成功导入")
    print("2) torch / torchvision 版本是否成对")
    print("3) 如果是 macOS，重点看 mps_is_available")
    print("4) 如果是 CPU 环境，cuda_is_available=False 是正常的")


if __name__ == "__main__":
    main()