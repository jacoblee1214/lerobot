"""
SmolVLA 라이브 추론 스크립트
실제 카메라 이미지 + 자연어 태스크 + 로봇 상태를 직접 입력으로 받아 action chunk를 예측한다.
데이터셋 없이 동작 가능.

Usage:
  # 이미지 파일 입력
  python smolvla_infer_live.py \
    --model-path outputs/train/checkpoints/last/pretrained_model \
    --top-image /path/to/top.jpg \
    --wrist-image /path/to/wrist.jpg \
    --task "Pick up the cube and place it in the box" \
    --state "0.6,177.2,164.6,72.3,82.5,0.1"

  # 웹캠 직접 입력
  python smolvla_infer_live.py \
    --model-path outputs/train/checkpoints/last/pretrained_model \
    --top-cam 0 \
    --wrist-cam 1 \
    --task "Pick up the cube and place it in the box" \
    --state "0.6,177.2,164.6,72.3,82.5,0.1"

  # action chunk 전체 출력
  python smolvla_infer_live.py ... --show-chunk

  # 결과 이미지 화면 표시 + 저장
  python smolvla_infer_live.py ... --show-result --save-dir outputs/infer_results
"""

import argparse
import numpy as np
import torch
import cv2
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from infer_viz import show_result


def resize_with_padding(img_bgr: np.ndarray, size: int) -> np.ndarray:
    """가로세로 비율 유지 + 패딩으로 size×size 정사각형으로 변환."""
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
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {path}")
    return img


def capture_from_camera(cam_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: index={cam_index}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"카메라에서 프레임을 읽을 수 없습니다: index={cam_index}")
    return frame


def bgr_to_float_chw(img_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 [H,W,3] → RGB float [3,H,W] in [0,1]
    SmolVLA는 [0,1] float CHW를 입력으로 받고 내부에서 [-1,1]로 변환한다."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).float() / 255.0  # [H,W,3], [0,1]
    return tensor.permute(2, 0, 1)                       # [3,H,W]


def get_image(file_path: str | None, cam_index: int | None, name: str) -> np.ndarray:
    if file_path is not None:
        print(f"  [{name}] 파일에서 로드: {file_path}")
        return load_image_from_file(file_path)
    elif cam_index is not None:
        print(f"  [{name}] 카메라 {cam_index}번에서 캡처")
        return capture_from_camera(cam_index)
    else:
        print(f"  [{name}] 입력 없음 → 더미 이미지(검정) 사용")
        return np.zeros((480, 640, 3), dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description="SmolVLA Live Inference")
    parser.add_argument("--model-path", type=str,
                        default="outputs/train/checkpoints/last/pretrained_model")
    parser.add_argument("--task", type=str,
                        default="Pick up the cube and place it in the box")
    parser.add_argument("--state", type=str,
                        default="0.6,177.2,164.6,72.3,82.5,0.1",
                        help="현재 joint angles (쉼표 구분, degree 단위)")
    parser.add_argument("--top-image", type=str, default=None)
    parser.add_argument("--wrist-image", type=str, default=None)
    parser.add_argument("--top-cam", type=int, default=None)
    parser.add_argument("--wrist-cam", type=int, default=None)
    parser.add_argument("--show-chunk", action="store_true",
                        help="action chunk 전체 출력 (chunk_size=50 steps)")
    parser.add_argument("--show-result", action="store_true",
                        help="결과 이미지를 OpenCV 창으로 표시")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="결과 이미지 저장 디렉터리 (예: outputs/infer_results)")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. 모델 로드
    print(f"모델 로드: {args.model_path}")
    policy = SmolVLAPolicy.from_pretrained(args.model_path).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config, args.model_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    chunk_size = policy.config.chunk_size
    print(f"파라미터: {sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M  |  chunk_size={chunk_size}")
    print("=" * 60)

    # 2. 이미지 획득
    print("[입력 이미지 준비]")
    top_bgr   = get_image(args.top_image,   args.top_cam,   "top")
    wrist_bgr = get_image(args.wrist_image, args.wrist_cam, "wrist")

    # 3. float CHW [0,1]로 변환
    # SmolVLA는 내부 prepare_images()에서 resize + [-1,1] 변환을 직접 수행하므로
    # 여기서는 원본 해상도 그대로 float CHW [0,1]만 맞춰주면 된다.
    top_tensor   = bgr_to_float_chw(top_bgr)    # [3,H,W] float [0,1]
    wrist_tensor = bgr_to_float_chw(wrist_bgr)  # [3,H,W] float [0,1]

    # 4. 로봇 상태 파싱
    state_values = [float(v) for v in args.state.split(",")]
    state_tensor = torch.tensor(state_values, dtype=torch.float32)

    # 5. 프레임 딕셔너리 구성
    # preprocessor가 top→camera1, wrist→camera2 로 rename함
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
    # select_action은 chunk 중 첫 번째 action 반환하거나 chunk 전체 반환할 수 있음
    if action.dim() == 1:
        first_action = action
    else:
        first_action = action[0]

    print()
    print("[추론 결과]")
    print(f"  태스크:     \"{args.task}\"")
    print(f"  현재 상태:   {[round(v, 3) for v in state_values]}")
    print(f"  다음 action: {[round(v, 4) for v in first_action.tolist()]}")
    print()
    print("  Joint별 해석 (SO-100 기준):")
    joint_names = ["J1(base)", "J2(shoulder)", "J3(elbow)", "J4(wrist_tilt)",
                   "J5(wrist_rot)", "J6(gripper)"]
    for name, cur, pred in zip(joint_names, state_values, first_action.tolist()):
        delta = pred - cur
        print(f"    {name:18s}: {cur:7.2f}° → {pred:7.2f}°  (Δ{delta:+.2f}°)")

    if args.show_chunk and action.dim() > 1:
        print()
        print(f"  [Action Chunk 전체 {action.shape[0]}스텝 @ 30Hz = {action.shape[0]/30:.1f}초]")
        for step_i, step_action in enumerate(action):
            print(f"    step {step_i:03d}: {[round(v, 2) for v in step_action.tolist()]}")

    # 8. 시각화
    if args.show_result or args.save_dir:
        model_tag = f"SmolVLA 450M | {args.model_path.split('/')[-3]}"
        show_result(
            top_bgr=top_bgr,
            wrist_bgr=wrist_bgr,
            task=args.task,
            state_values=state_values,
            action_values=first_action.tolist(),
            model_name=model_tag,
            save_path=args.save_dir,
            window_title="SmolVLA Inference Result",
        )

    print()
    print("=" * 60)
    print("추론 완료.")


if __name__ == "__main__":
    main()
