from __future__ import annotations

from typing import Any

try:
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    np = None


def normalize_bbox_current(raw_bbox: Any) -> list[float]:
    if raw_bbox is None:
        return [0.0, 0.0, 0.0, 0.0]

    if isinstance(raw_bbox, list):
        if len(raw_bbox) == 4 and not isinstance(raw_bbox[0], list):
            return [float(v) for v in raw_bbox]
        if len(raw_bbox) > 0 and isinstance(raw_bbox[0], list):
            return [float(v) for v in raw_bbox[0][:4]]

    return [0.0, 0.0, 0.0, 0.0]


def normalize_bbox_trial(raw_bbox: Any) -> list[float]:
    if raw_bbox is None:
        return [0.0, 0.0, 0.0, 0.0]

    if np is not None and isinstance(raw_bbox, np.ndarray):
        if raw_bbox.ndim == 1 and raw_bbox.size >= 4:
            return [float(v) for v in raw_bbox[:4].tolist()]
        if raw_bbox.ndim >= 2 and raw_bbox.shape[0] > 0 and raw_bbox.shape[1] >= 4:
            return [float(v) for v in raw_bbox[0, :4].tolist()]
        return [0.0, 0.0, 0.0, 0.0]

    if isinstance(raw_bbox, (list, tuple)):
        nested_types = (list, tuple) if np is None else (list, tuple, np.ndarray)
        if len(raw_bbox) == 4 and not isinstance(raw_bbox[0], nested_types):
            return [float(v) for v in raw_bbox]
        if len(raw_bbox) > 0 and isinstance(raw_bbox[0], nested_types):
            first = raw_bbox[0]
            return [float(v) for v in list(first)[:4]]

    return [0.0, 0.0, 0.0, 0.0]


def main() -> None:
    samples = {
        "none": None,
        "list_1d": [10, 20, 110, 220],
        "list_2d": [[10, 20, 110, 220, 0.9]],
        "tuple_1d": (10, 20, 110, 220),
    }
    if np is not None:
        samples["ndarray_1d"] = np.array([10, 20, 110, 220], dtype=np.float32)
        samples["ndarray_2d"] = np.array([[10, 20, 110, 220, 0.9]], dtype=np.float32)

    print("name".ljust(12), "current".ljust(28), "trial")
    for name, sample in samples.items():
        current = normalize_bbox_current(sample)
        trial = normalize_bbox_trial(sample)
        print(name.ljust(12), str(current).ljust(28), trial)


if __name__ == "__main__":
    main()
