"""
X-VLA 라이브 추론 스크립트
실제 카메라 이미지 + 자연어 태스크 + 로봇 상태를 직접 입력으로 받아 action을 예측한다.
데이터셋 없이 동작 가능.

Usage:
  # 이미지 파일 + 텍스트 입력
  python xvla_infer_live.py \
    --model-path outputs/xvla_proper/checkpoints/last/pretrained_model \
    --top-image /path/to/top.jpg \
    --wrist-image /path/to/wrist.jpg \
    --task "Pick up the cube and place it in the box" \
    --state "0.6,177.2,164.6,72.3,82.5,0.1"

  # 웹캠 직접 입력 (상단 카메라: 0번, 손목 카메라: 1번)
  python xvla_infer_live.py \
    --model-path outputs/xvla_proper/checkpoints/last/pretrained_model \
    --top-cam 0 \
    --wrist-cam 1 \
    --task "Pick up the cube and place it in the box" \
    --state "0.6,177.2,164.6,72.3,82.5,0.1"
  # 결과 이미지 화면 표시 + 저장
  python xvla_infer_live.py ... --show-result --save-dir outputs/infer_results
"""

import argparse
import numpy as np
import torch
import cv2
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from infer_viz import show_result


TARGET_SIZE = 224   # 모델 입력 해상도


def resize_with_padding(img_bgr: np.ndarray, size: int) -> np.ndarray:
    """가로세로 비율 유지 + 패딩으로 size×size 정사각형으로 변환. [0,255] uint8 반환."""
    h, w = img_bgr.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas


def load_image_from_file(path: str) -> np.ndarray:
    """이미지 파일 → BGR numpy [H,W,3] uint8"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {path}")
    return img


def capture_from_camera(cam_index: int) -> np.ndarray:
    """웹캠에서 단일 프레임 캡처 → BGR numpy [H,W,3] uint8"""
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: index={cam_index}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"카메라에서 프레임을 읽을 수 없습니다: index={cam_index}")
    return frame


def bgr_to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """BGR numpy [H,W,3] → RGB tensor [H,W,3] uint8"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img_rgb)   # [H,W,3], uint8, [0,255]


def parse_args():
    parser = argparse.ArgumentParser(description="X-VLA Live Inference")
    parser.add_argument("--model-path", type=str,
                        default="outputs/xvla_proper/checkpoints/last/pretrained_model")
    parser.add_argument("--task", type=str,
                        default="Pick up the cube and place it in the box",
                        help="자연어 태스크 지시")
    parser.add_argument("--state", type=str,
                        default="0.6,177.2,164.6,72.3,82.5,0.1",
                        help="로봇 현재 joint angles (쉼표 구분, degree 단위)")
    # 이미지 소스: 파일 또는 카메라 인덱스
    parser.add_argument("--top-image", type=str, default=None,
                        help="상단 카메라 이미지 파일 경로 (.jpg/.png)")
    parser.add_argument("--wrist-image", type=str, default=None,
                        help="손목 카메라 이미지 파일 경로 (.jpg/.png)")
    parser.add_argument("--top-cam", type=int, default=None,
                        help="상단 카메라 인덱스 (예: 0)")
    parser.add_argument("--wrist-cam", type=int, default=None,
                        help="손목 카메라 인덱스 (예: 1)")
    parser.add_argument("--show-result", action="store_true",
                        help="결과 이미지를 OpenCV 창으로 표시")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="결과 이미지 저장 디렉터리 (예: outputs/infer_results)")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def get_image(file_path: str | None, cam_index: int | None, name: str) -> np.ndarray:
    """파일 또는 카메라에서 이미지 획득"""
    if file_path is not None:
        print(f"  [{name}] 파일에서 로드: {file_path}")
        return load_image_from_file(file_path)
    elif cam_index is not None:
        print(f"  [{name}] 카메라 {cam_index}번에서 캡처")
        return capture_from_camera(cam_index)
    else:
        # 아무것도 지정 안 했으면 검정 더미 이미지 (테스트용)
        print(f"  [{name}] 입력 없음 → 더미 이미지(검정) 사용")
        return np.zeros((480, 640, 3), dtype=np.uint8)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. 모델 로드
    print(f"모델 로드: {args.model_path}")
    policy = XVLAPolicy.from_pretrained(args.model_path).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config, args.model_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print(f"파라미터: {sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M")
    print("=" * 60)

    # 2. 이미지 획득
    print("[입력 이미지 준비]")
    top_bgr   = get_image(args.top_image,   args.top_cam,   "top")
    wrist_bgr = get_image(args.wrist_image, args.wrist_cam, "wrist")

    # 3. 리사이즈 (학습 조건과 동일하게 224×224 + 패딩)
    top_resized   = resize_with_padding(top_bgr,   TARGET_SIZE)
    wrist_resized = resize_with_padding(wrist_bgr, TARGET_SIZE)

    top_tensor   = bgr_to_tensor(top_resized)    # [224,224,3] uint8
    wrist_tensor = bgr_to_tensor(wrist_resized)  # [224,224,3] uint8

    # 4. 로봇 상태 파싱
    state_values = [float(v) for v in args.state.split(",")]
    state_tensor = torch.tensor(state_values, dtype=torch.float32)

    # 5. 프레임 딕셔너리 구성
    # preprocessor가 top→image, wrist→image2 로 rename함
    frame = {
        "observation.images.top":   top_tensor,
        "observation.images.wrist": wrist_tensor,
        "observation.state":        state_tensor,
        "task":                     args.task,
    }

    # 6. 전처리 → 추론 → 후처리
    batch = preprocess(frame)

    with torch.no_grad():
        raw_action = policy.select_action(batch)
        action = postprocess(raw_action)

    # 7. 결과 출력
    print()
    print("[추론 결과]")
    print(f"  태스크:     \"{args.task}\"")
    print(f"  현재 상태:   {state_values}")
    action_list = action[0].tolist() if action.dim() > 1 else action.tolist()
    print(f"  예측 action: {[round(v, 4) for v in action_list]}")
    print()
    print("  Joint별 해석 (SO-100 기준):")
    joint_names = ["J1(base)", "J2(shoulder)", "J3(elbow)", "J4(wrist_tilt)",
                   "J5(wrist_rot)", "J6(gripper)"]
    for name, cur, pred in zip(joint_names, state_values, action_list):
        delta = pred - cur
        print(f"    {name:18s}: {cur:7.2f}° → {pred:7.2f}°  (Δ{delta:+.2f}°)")

    # 8. 시각화
    if args.show_result or args.save_dir:
        model_tag = f"X-VLA 879M | {args.model_path.split('/')[-3]}"
        show_result(
            top_bgr=top_bgr,
            wrist_bgr=wrist_bgr,
            task=args.task,
            state_values=state_values,
            action_values=action_list,
            model_name=model_tag,
            save_path=args.save_dir,
            window_title="X-VLA Inference Result",
        )

    print()
    print("=" * 60)
    print("추론 완료.")


if __name__ == "__main__":
    main()
