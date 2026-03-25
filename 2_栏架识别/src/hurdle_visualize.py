# -*- coding: utf-8 -*-
from __future__ import annotations

import cv2
import numpy as np


def draw_hurdle_result(frame: np.ndarray, bbox_xyxy, bar_mid, post_x, conf, polygon=None, alpha: float = 0.35):
    vis = frame.copy()

    # 1) 先画 mask（半透明填充 + 轮廓）
    if polygon is not None and len(polygon) >= 3:
        pts = np.round(np.asarray(polygon)).astype(np.int32)

        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 255))   # 黄色半透明区域
        vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

        cv2.polylines(vis, [pts], isClosed=True, color=(0, 200, 255), thickness=2)

    # 2) 再画 bbox
    if bbox_xyxy is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"hurdle {conf:.2f}" if conf is not None and conf == conf else 'hurdle'
        cv2.putText(vis, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 3) 画横杆中点
    if bar_mid is not None and bar_mid[0] == bar_mid[0] and bar_mid[1] == bar_mid[1]:
        bx, by = [int(round(v)) for v in bar_mid]
        cv2.circle(vis, (bx, by), 5, (0, 0, 255), -1)
        cv2.putText(vis, 'bar_mid', (bx + 6, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 4) 画立柱 x
    if post_x is not None and post_x == post_x:
        px = int(round(post_x))
        cv2.line(vis, (px, 0), (px, vis.shape[0] - 1), (255, 0, 0), 2)
        cv2.putText(vis, 'post_x', (px + 6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return vis