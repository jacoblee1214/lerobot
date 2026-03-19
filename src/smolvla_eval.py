"""
SmolVLA Open-loop 평가 스크립트
에피소드별로 predicted action vs GT action을 비교하고 통계를 출력합니다.
Usage: python scripts/smolvla_eval.py --model-path <checkpoint_path> --dataset-repo <repo_id>
"""

import argparse
import torch
import numpy as np
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args():
    parser = argparse.ArgumentParser(description="SmolVLA Open-loop Evaluation")
    parser.add_argument("--model-path", type=str, default="outputs/train/checkpoints/last/pretrained_model")
    parser.add_argument("--dataset-repo", type=str, default="lerobot/svla_so100_pickplace")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--samples-per-episode", type=int, default=10, help="Frames to sample per episode")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-file", type=str, default=None, help="Save results to file")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 모델 로드
    print(f"Loading model from: {args.model_path}")
    policy = SmolVLAPolicy.from_pretrained(args.model_path).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config, args.model_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # 데이터셋 로드
    print(f"Loading dataset: {args.dataset_repo}")
    dataset = LeRobotDataset(args.dataset_repo)
    num_episodes = min(args.num_episodes, dataset.meta.episodes.shape[0])
    print(f"Evaluating {num_episodes} episodes, {args.samples_per_episode} samples each")
    print("=" * 60)

    all_errors = []
    episode_results = []

    for ep_idx in range(num_episodes):
        from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
        to_idx = dataset.meta.episodes["dataset_to_index"][ep_idx]
        ep_length = to_idx - from_idx

        # 에피소드 내에서 균등하게 샘플링
        sample_indices = np.linspace(from_idx, to_idx - 1, args.samples_per_episode, dtype=int)

        ep_errors = []
        for frame_idx in sample_indices:
            frame = dict(dataset[int(frame_idx)])
            batch = preprocess(frame)

            with torch.no_grad():
                raw_action = policy.select_action(batch)
                action = postprocess(raw_action)

            gt = frame["action"]

            if action.dim() > 1:
                action_flat = action[0][:len(gt)]
            else:
                action_flat = action[:len(gt)]

            error = (action_flat.cpu() - gt).abs().mean().item()
            ep_errors.append(error)

        ep_mean_error = np.mean(ep_errors)
        ep_std_error = np.std(ep_errors)
        all_errors.extend(ep_errors)

        print(f"Episode {ep_idx:3d} | Length: {ep_length:4d} | "
              f"Mean Error: {ep_mean_error:8.4f} | Std: {ep_std_error:8.4f}")
        episode_results.append({
            "episode": ep_idx,
            "length": ep_length,
            "mean_error": ep_mean_error,
            "std_error": ep_std_error,
        })

    # 전체 통계
    print("=" * 60)
    print(f"Overall Mean Error: {np.mean(all_errors):.4f}")
    print(f"Overall Std Error:  {np.std(all_errors):.4f}")
    print(f"Min Error:          {np.min(all_errors):.4f}")
    print(f"Max Error:          {np.max(all_errors):.4f}")
    print(f"Total samples:      {len(all_errors)}")

    # 결과 저장
    if args.output_file:
        import json
        results = {
            "model_path": args.model_path,
            "dataset": args.dataset_repo,
            "overall_mean_error": float(np.mean(all_errors)),
            "overall_std_error": float(np.std(all_errors)),
            "episodes": episode_results,
        }
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
