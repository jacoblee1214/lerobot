"""
SO-100 데이터셋 3개 병합 스크립트
lerobot/svla_so100_pickplace (50) + stacking (56) + sorting (52) → 158 episodes

Usage:
    cd /home/jake/lerobot/src
    conda run -n smolvla python merge_so100.py
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets

print("Loading datasets...")
ds1 = LeRobotDataset("lerobot/svla_so100_pickplace")
ds2 = LeRobotDataset("lerobot/svla_so100_stacking")
ds3 = LeRobotDataset("lerobot/svla_so100_sorting")

print(f"  pickplace : {ds1.meta.total_episodes} episodes, {len(ds1)} frames")
print(f"  stacking  : {ds2.meta.total_episodes} episodes, {len(ds2)} frames")
print(f"  sorting   : {ds3.meta.total_episodes} episodes, {len(ds3)} frames")
print()

print("Merging...")
merged = merge_datasets(
    datasets=[ds1, ds2, ds3],
    output_repo_id="local/so100_merged",
    output_dir="datasets/so100_merged",
)

print()
print("=== 병합 완료 ===")
print(f"  총 episodes : {merged.meta.total_episodes}")
print(f"  총 frames   : {len(merged)}")
print(f"  저장 위치   : datasets/so100_merged")
print()
print("다음 단계: 아래 커맨드로 학습")
print("""
cd /home/jake/lerobot/src && PYTORCH_ALLOC_CONF=expandable_segments:True \\
conda run -n smolvla python -u -m lerobot.scripts.lerobot_train \\
  --policy.path=lerobot/xvla-base \\
  --dataset.repo_id=local/so100_merged \\
  --dataset.root=datasets/so100_merged \\
  --output_dir=outputs/xvla_158ep \\
  --policy.push_to_hub=false \\
  --job_name=xvla_158ep \\
  --steps=50000 \\
  --batch_size=4 \\
  --policy.dtype=bfloat16 \\
  --policy.action_mode=auto \\
  --policy.freeze_vision_encoder=false \\
  --policy.freeze_language_encoder=false \\
  --policy.train_policy_transformer=true \\
  --policy.train_soft_prompts=true \\
  --policy.resize_imgs_with_padding="[224,224]" \\
  --rename_map='{"observation.images.top": "observation.images.image", "observation.images.wrist": "observation.images.image2"}'
""")
