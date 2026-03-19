"""
SmolVLA 단일 프레임 추론 스크립트
Usage: python scripts/smolvla_infer.py --model-path <checkpoint_path> --dataset-repo <repo_id> --frames 0 100 200
"""

import argparse
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args():
    parser = argparse.ArgumentParser(description="SmolVLA Inference")
    parser.add_argument("--model-path", type=str, default="outputs/train/checkpoints/last/pretrained_model")
    parser.add_argument("--dataset-repo", type=str, default="lerobot/svla_so100_pickplace")
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 50, 100, 200, 300])
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading model from: {args.model_path}")
    policy = SmolVLAPolicy.from_pretrained(args.model_path).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config, args.model_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    print(f"Loading dataset: {args.dataset_repo}")
    dataset = LeRobotDataset(args.dataset_repo)
    print(f"Dataset: {len(dataset)} frames, {dataset.meta.episodes.shape[0]} episodes")
    print(f"Model: {sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M params")
    print("=" * 60)

    for i in args.frames:
        if i >= len(dataset):
            print(f"Frame {i} out of range (max: {len(dataset)-1}), skipping")
            continue

        frame = dict(dataset[i])
        batch = preprocess(frame)

        with torch.no_grad():
            raw_action = policy.select_action(batch)
            action = postprocess(raw_action)

        gt = frame["action"]

        print(f"\n--- Frame {i} ---")
        print(f"  Predicted: {action[0][:6].tolist()}")
        print(f"  GT:        {gt[:6].tolist()}")

        # 각 차원별 오차
        if action.dim() > 1:
            action_flat = action[0][:len(gt)]
        else:
            action_flat = action[:len(gt)]
        error = (action_flat.cpu() - gt).abs()
        print(f"  Abs Error: {error.tolist()}")
        print(f"  Mean Error: {error.mean().item():.4f}")

    print("\n" + "=" * 60)
    print("Inference complete!")


if __name__ == "__main__":
    main()
