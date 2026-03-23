"""
VLA 추론 결과 시각화 유틸리티
smolvla_infer_live.py / xvla_infer_live.py 에서 공통으로 사용.

레이아웃:
┌─────────────────────────────────────────────┐
│   Top Cam [320×320]  │  Wrist Cam [320×320] │
├─────────────────────────────────────────────┤
│  Task: "Pick up the cube..."                │
│  J1(base)     :   0.60° →   2.30°  Δ+1.70° │
│  J2(shoulder) : 177.20° → 160.50°  Δ-16.70°│
│  ...                                        │
│  J6(gripper)  :   0.10° →   2.30°  Δ+2.20° │
│                               [ModelName]   │
└─────────────────────────────────────────────┘
"""

import os
import cv2
import numpy as np
from datetime import datetime


JOINT_NAMES = ["J1(base)", "J2(shoulder)", "J3(elbow)",
               "J4(wrist_tilt)", "J5(wrist_rot)", "J6(gripper)"]

# 색상 (BGR)
WHITE  = (255, 255, 255)
YELLOW = (0, 220, 255)
GREEN  = (80, 220, 80)
RED    = (80, 80, 255)
GRAY   = (160, 160, 160)
DARK   = (30, 30, 30)
CYAN   = (220, 200, 0)


def _resize_display(img_bgr: np.ndarray, size: int = 320) -> np.ndarray:
    """표시용 정사각형 리사이즈 (패딩 포함)"""
    h, w = img_bgr.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img_bgr, (nw, nh))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    t, l = (size - nh) // 2, (size - nw) // 2
    canvas[t:t+nh, l:l+nw] = resized
    return canvas


def _draw_text(canvas, text, x, y, color=WHITE, scale=0.55, thickness=1):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def build_result_image(
    top_bgr: np.ndarray,
    wrist_bgr: np.ndarray,
    task: str,
    state_values: list[float],
    action_values: list[float],
    model_name: str = "Model",
) -> np.ndarray:
    """
    시각화 이미지 생성.

    Returns:
        np.ndarray: BGR 이미지 (OpenCV 표시 / 저장용)
    """
    IMG_SIZE  = 320   # 카메라 표시 크기
    INFO_H    = 240   # 하단 텍스트 영역 높이
    PANEL_W   = IMG_SIZE * 2  # 전체 너비 = 640

    # ── 상단: 카메라 2개 ──
    top_disp   = _resize_display(top_bgr,   IMG_SIZE)
    wrist_disp = _resize_display(wrist_bgr, IMG_SIZE)

    # 카메라 라벨
    cv2.rectangle(top_disp,   (0, 0), (120, 22), DARK, -1)
    cv2.rectangle(wrist_disp, (0, 0), (140, 22), DARK, -1)
    _draw_text(top_disp,   "Top Cam",   5, 16, CYAN)
    _draw_text(wrist_disp, "Wrist Cam", 5, 16, CYAN)

    cam_row = np.hstack([top_disp, wrist_disp])   # [320, 640, 3]

    # ── 하단: 텍스트 정보 ──
    info = np.full((INFO_H, PANEL_W, 3), 20, dtype=np.uint8)  # 어두운 배경

    # 구분선
    cv2.line(info, (0, 0), (PANEL_W, 0), (60, 60, 60), 1)

    y = 28
    # Task
    task_short = task if len(task) <= 62 else task[:59] + "..."
    _draw_text(info, f"Task: \"{task_short}\"", 10, y, YELLOW, scale=0.5)
    y += 22

    # 헤더
    _draw_text(info, f"{'Joint':<16}  {'Current':>8}    {'Action':>8}    {'Delta':>8}", 10, y, GRAY, scale=0.45)
    y += 4
    cv2.line(info, (10, y), (PANEL_W - 10, y), (60, 60, 60), 1)
    y += 14

    # Joint별 행
    action_dim = min(len(state_values), len(action_values), len(JOINT_NAMES))
    for i in range(action_dim):
        name  = JOINT_NAMES[i]
        cur   = state_values[i]
        pred  = action_values[i]
        delta = pred - cur
        delta_color = GREEN if delta >= 0 else RED
        line  = f"{name:<16}  {cur:>7.2f}deg  {pred:>7.2f}deg"
        _draw_text(info, line, 10, y, WHITE, scale=0.47)
        _draw_text(info, f"  {delta:>+7.2f}deg", 10 + int(PANEL_W * 0.62), y, delta_color, scale=0.47)
        y += 20

    # 모델명 + 타임스탬프
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _draw_text(info, f"{model_name}  |  {ts}", 10, INFO_H - 12, GRAY, scale=0.42)

    return np.vstack([cam_row, info])   # [560, 640, 3]


def show_result(
    top_bgr: np.ndarray,
    wrist_bgr: np.ndarray,
    task: str,
    state_values: list[float],
    action_values: list[float],
    model_name: str = "Model",
    save_path: str | None = None,
    window_title: str = "VLA Inference Result",
) -> None:
    """
    결과 이미지를 화면에 표시하고 선택적으로 저장.

    Args:
        save_path: None이면 저장 안 함. 디렉터리 경로면 타임스탬프 파일명 자동 생성.
    """
    img = build_result_image(top_bgr, wrist_bgr, task, state_values, action_values, model_name)

    # 저장
    if save_path is not None:
        if os.path.isdir(save_path) or not os.path.splitext(save_path)[1]:
            os.makedirs(save_path, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(save_path, f"infer_{ts}.png")
        else:
            fname = save_path
        cv2.imwrite(fname, img)
        print(f"  결과 이미지 저장: {fname}")

    # 화면 표시 (GUI 환경에서만 동작)
    try:
        cv2.imshow(window_title, img)
        print("  [결과 창] 아무 키나 누르면 닫힘")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("  [결과 창] GUI 환경 없음 — 이미지 파일로만 저장됨")
