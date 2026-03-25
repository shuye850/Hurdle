# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def _interp_nan_1d(arr: np.ndarray, max_gap: int) -> np.ndarray:
    x = arr.astype(np.float32).copy()
    n = x.shape[0]
    isn = np.isnan(x)
    if not isn.any():
        return x

    idx = np.arange(n)
    valid = ~isn
    if valid.sum() < 2:
        return x

    x_interp = np.interp(idx, idx[valid], x[valid]).astype(np.float32)

    i = 0
    while i < n:
        if not isn[i]:
            i += 1
            continue
        j = i
        while j < n and isn[j]:
            j += 1
        gap = j - i
        if gap <= max_gap:
            x[i:j] = x_interp[i:j]
        else:
            x[i:j] = np.nan
        i = j
    return x


def _ffill_bfill(v: np.ndarray) -> np.ndarray:
    out = v.copy()
    last = np.nan
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    last = np.nan
    for i in range(len(out) - 1, -1, -1):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out


def _savgol_safe(series: np.ndarray, win: int, poly: int) -> np.ndarray:
    nan_mask = np.isnan(series)
    temp = _ffill_bfill(series) if nan_mask.any() else series
    filtered = savgol_filter(temp, window_length=win, polyorder=poly, mode="interp").astype(np.float32)
    filtered[nan_mask] = np.nan
    return filtered


def _normalized_window(series_len: int, window: int, poly: int) -> tuple[int, bool]:
    win = int(window)
    if win % 2 == 0:
        win += 1
    if win > series_len:
        win = series_len if series_len % 2 == 1 else max(1, series_len - 1)
    do_filter = win >= 5 and win > poly
    return win, do_filter


def _apply_support_foot_constraint(y_out: np.ndarray, conf_all: np.ndarray, conf_th: float, max_upward_jump_px: float = 18.0) -> np.ndarray:
    protected_indices = [27, 28, 29, 30, 31, 32]
    y = y_out.copy()
    t = y.shape[0]
    if t < 3:
        return y

    for k in protected_indices:
        for i in range(1, t - 1):
            if conf_all[i, k] < conf_th:
                continue
            if not (np.isfinite(y[i - 1, k]) and np.isfinite(y[i, k]) and np.isfinite(y[i + 1, k])):
                continue
            upward = y[i - 1, k] - y[i, k]
            if upward > max_upward_jump_px and abs(y[i + 1, k] - y[i - 1, k]) < (0.7 * upward):
                y[i, k] = np.float32(0.5 * (y[i - 1, k] + y[i + 1, k]))
    return y


def postprocess_pose(
    x_all: np.ndarray,
    y_all: np.ndarray,
    conf_all: np.ndarray,
    conf_th: float = 0.6,
    max_interp_gap: int = 5,
    savgol_window: int = 9,
    savgol_polyorder: int = 2,
    enable_support_foot_constraint: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t, k = x_all.shape
    x = x_all.astype(np.float32).copy()
    y = y_all.astype(np.float32).copy()
    c = conf_all.astype(np.float32).copy()

    bad = c < conf_th
    x[bad] = np.nan
    y[bad] = np.nan

    win, do_filter = _normalized_window(t, savgol_window, savgol_polyorder)
    x_out = np.empty_like(x, dtype=np.float32)
    y_out = np.empty_like(y, dtype=np.float32)

    for idx in range(k):
        xk = _interp_nan_1d(x[:, idx], max_gap=max_interp_gap)
        yk = _interp_nan_1d(y[:, idx], max_gap=max_interp_gap)
        if do_filter:
            xk = _savgol_safe(xk, win, savgol_polyorder)
            yk = _savgol_safe(yk, win, savgol_polyorder)
        x_out[:, idx] = xk
        y_out[:, idx] = yk

    if enable_support_foot_constraint:
        y_out = _apply_support_foot_constraint(y_out, c, conf_th=conf_th)

    return x_out, y_out, c


def postprocess_pose3d(
    x3d_all: np.ndarray,
    y3d_all: np.ndarray,
    z3d_all: np.ndarray,
    conf_all: np.ndarray,
    conf_th: float = 0.6,
    max_interp_gap: int = 5,
    savgol_window: int = 9,
    savgol_polyorder: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_clean, y_clean, c_clean = postprocess_pose(
        x3d_all,
        y3d_all,
        conf_all,
        conf_th=conf_th,
        max_interp_gap=max_interp_gap,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
        enable_support_foot_constraint=False,
    )

    t, k = z3d_all.shape
    z = z3d_all.astype(np.float32).copy()
    z[c_clean < conf_th] = np.nan

    win, do_filter = _normalized_window(t, savgol_window, savgol_polyorder)
    z_out = np.empty_like(z, dtype=np.float32)
    for idx in range(k):
        zk = _interp_nan_1d(z[:, idx], max_gap=max_interp_gap)
        if do_filter:
            zk = _savgol_safe(zk, win, savgol_polyorder)
        z_out[:, idx] = zk

    return x_clean, y_clean, z_out, c_clean
