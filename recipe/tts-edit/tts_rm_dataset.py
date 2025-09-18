#!/usr/bin/env python
"""
Preprocess the TTS Edit Score dataset for multi-image VLM training with VERL PPO.
This script handles multi-image inputs (input_image and output_image) for image editing evaluation.
"""

import argparse
import glob
import json
import os
import random
from string import Template
from typing import Any, Optional, Tuple

import datasets
import numpy as np
from PIL import Image
from tqdm import tqdm

from verl.utils.hdfs_io import copy, makedirs

# ================== Constants ==================
SYSTEM_PROMPT = """你是一个AI助理。

在正式回答之前，你总是可以使用<think></think>标签进行详细地思考。
请保持<think></think>标签内的内容与问题相关且有意义，不要随意编造，也不要重复冗余。
若需要思考再回答，请保持答案与思考过程一致。
"""

SCORER_PROMPT_TEMPLATE = Template(
    """你是一个图像编辑评分模型。你的任务是根据给定的编辑指令，评估编辑后的图像相对于输入图像的质量。
你的回答必须是0到9之间的整数，0表示编辑效果非常差，9表示编辑效果非常好。
你必须用<answer></answer>标签括起你的最终答案。

编辑指令： $instruction
输入图像：
"""
)

DATA_SOURCE = "tts_edit_score"

# ================== Data Loading Functions ==================


def load_tts_edit_dataset(data_dir: str, split: str = "train") -> list[dict[str, Any]]:
    """
    Load TTS Edit dataset from directory structure.

    Expected structure:
    - *_input.jpg: Input images
    - *_output.jpg: Output (edited) images
    - *_meta.json: Contains instruction
    - *_score.json: Contains ground truth score
    """
    dataset_items = []

    # Find all input images
    print(f"Searching for images in {data_dir}...")
    input_images = glob.glob(
        os.path.join(data_dir, "**", "*_input.jpg"), recursive=True
    )
    print(f"Found {len(input_images)} input images")

    # Statistics
    image_sizes = []
    instruction_lengths = []
    scores = []
    skipped_count = 0

    for idx, input_image_path in enumerate(tqdm(input_images, desc="Loading dataset")):
        # Construct paths for related files
        output_image_path = input_image_path.replace("_input.jpg", "_output.jpg")
        meta_path = input_image_path.replace("_input.jpg", "_meta.json")
        score_path = input_image_path.replace("_input.jpg", "_score.json")

        # Skip if any required file is missing
        if not all(
            os.path.exists(p) for p in [output_image_path, meta_path, score_path]
        ):
            skipped_count += 1
            continue

        # Load metadata
        with open(meta_path) as f:
            metadata = json.load(f)
            instruction = metadata.get("instruction", "")
            instruction_lengths.append(len(instruction))

        # Load score
        with open(score_path) as f:
            score_data = json.load(f)
            score = score_data.get("score", 5.0)
            scores.append(score)

        # For statistics, sample first few images to avoid loading all into memory
        if idx < 10:  # Only load first 10 images for statistics
            input_image = Image.open(input_image_path).convert("RGB")
            output_image = Image.open(output_image_path).convert("RGB")

            # Record original sizes
            image_sizes.append(
                {
                    "input": (input_image.width, input_image.height),
                    "output": (output_image.width, output_image.height),
                }
            )

            # Close images to free memory immediately
            input_image.close()
            output_image.close()

        # Store paths instead of loaded images (images will be loaded in mapping function)
        dataset_items.append(
            {
                "idx": idx,
                "input_image_path": input_image_path,
                "output_image_path": output_image_path,
                "instruction": instruction,
                "score": score,
                "split": split,
            }
        )

    # Print statistics
    if skipped_count > 0:
        print(f"Skipped {skipped_count} samples due to missing files")

    # Don't print statistics here, will print after sampling
    return dataset_items


def print_score_distribution(
    scores: list[float], title: str = "Score Distribution"
) -> None:
    """Print score distribution statistics."""
    if not scores:
        return

    print(f"\n{title}:")
    print(f"  Total samples: {len(scores)}")
    print(f"  Mean: {np.mean(scores):.2f}, Std: {np.std(scores):.2f}")

    # Score histogram
    hist, bins = np.histogram(scores, bins=10)
    print("  Distribution:")
    for i in range(len(hist)):
        bar = "█" * int(hist[i] * 30 / max(hist)) if max(hist) > 0 else ""
        print(f"    [{bins[i]:.2f}-{bins[i + 1]:.2f}]: {bar} ({hist[i]})")


def print_sample_examples(samples: list[dict[str, Any]]) -> None:
    """Print sample examples from the dataset."""
    print("\nSample Examples:")
    for i, sample in enumerate(samples[:2], 1):
        print(f"  Example {i}:")
        print(
            f"    Instruction: {sample['instruction'][:80]}{'...' if len(sample['instruction']) > 80 else ''}"
        )
        print(f"    Score: {sample['score']:.2f}")


def normalize_score(score: float, median: float = 0.9, mad: float = 0.1) -> int:
    """Normalize score to 0-9 range using median and MAD."""
    normalized = (score - median) / mad
    normalized = normalized * 4.5 + 4.5
    return int(max(0, min(9, normalized)))


def balanced_sample_by_score(
    data: list[dict[str, Any]], score_median: float, score_mad: float, seed: int = 42
) -> list[dict[str, Any]]:
    """
    Perform balanced sampling where each normalized score group contributes an equal number of samples.
    """
    import collections

    # Group samples by normalized score
    score_groups = collections.defaultdict(list)

    for item in data:
        # Calculate normalized score for this item
        normalized_score = normalize_score(item["score"], score_median, score_mad)
        score_groups[normalized_score].append(item)

    # Find the minimum group size
    group_sizes = {score: len(items) for score, items in score_groups.items()}
    min_size = min(group_sizes.values())

    print("\nBalanced Sampling (by normalized scores 0-9):")
    print("  Groups before sampling:")
    for score in sorted(score_groups.keys()):
        print(f"    Score {score}: {group_sizes[score]} samples")
    print(f"  → Taking {min_size} samples from each group (minimum group size)")
    print(
        f"  → Total after balancing: {min_size * len(score_groups)} (from {len(data)})"
    )

    # Sample min_size items from each group
    random.seed(seed)
    balanced_data = []

    for score in sorted(score_groups.keys()):
        group_items = score_groups[score]
        sampled_items = random.sample(group_items, min_size)
        balanced_data.extend(sampled_items)

    # Shuffle the final dataset to mix scores
    random.shuffle(balanced_data)

    return balanced_data


# ================== Visualization Functions ==================


def print_final_statistics(
    train_dataset: datasets.Dataset,
    test_dataset: datasets.Dataset,
    normalized_scores: list[int],
) -> None:
    """Print final statistics about the processed datasets."""
    print("\n" + "=" * 60)
    print(f"Final Dataset: {len(train_dataset)} train, {len(test_dataset)} test")

    # Final normalized score distribution
    if normalized_scores:
        print("\nFinal Normalized Score Distribution (0-9):")
        score_counts = {i: normalized_scores.count(i) for i in range(10)}
        for score in range(10):
            count = score_counts.get(score, 0)
            bar = (
                "█" * int(count * 30 / max(score_counts.values()))
                if max(score_counts.values()) > 0
                else ""
            )
            print(f"  Score {score}: {bar} ({count})")

    print("=" * 60)


# ================== Dataset Transformation Functions ==================


def make_map_fn(
    split: str, data_source: str, score_median: float = 0.9, score_mad: float = 0.1
) -> callable:
    """
    Create a mapping function for dataset processing.
    This handles multi-image inputs for VLM training.
    """

    def process_fn(example: dict[str, Any], idx: int) -> dict[str, Any]:
        # Extract data from example
        instruction = example.get("instruction", "")
        score = example.get("score", 5.0)
        input_image_path = example.get("input_image_path")
        output_image_path = example.get("output_image_path")

        # Load and process images
        input_image = Image.open(input_image_path).convert("RGB")
        output_image = Image.open(output_image_path).convert("RGB")

        # Resize images to keep both images the same size (following qwen-vl-grpo approach)
        target_width = output_image.width // 2
        target_height = output_image.height // 2

        input_image = input_image.resize((target_width, target_height), Image.LANCZOS)
        output_image = output_image.resize((target_width, target_height), Image.LANCZOS)

        # Normalize score
        normalized_score = normalize_score(score, score_median, score_mad)

        # Construct the prompt with instruction following
        scorer_prompt = SCORER_PROMPT_TEMPLATE.substitute(instruction=instruction)
        full_prompt = scorer_prompt + "\n" + "<image>" + "\n编辑后的图像:\n" + "<image>"

        # Create the conversation format for chat template
        # Note: For multi-image VLM, we need to structure the content properly
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": full_prompt,  # Text prompt
            },
        ]

        # Prepare data in VERL format
        data = {
            "data_source": data_source,
            "prompt": prompt_messages,
            "images": [input_image, output_image],  # Multi-image input
            "ability": "image_editing_evaluation",
            "reward_model": {
                "style": "rule",
                "ground_truth": str(
                    normalized_score
                ),  # Store as string for consistency
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "instruction": instruction,
                "original_score": score,
                "normalized_score": normalized_score,
                "input_image_path": input_image_path,
            },
        }

        return data

    return process_fn


# ================== CLI and Helper Functions ==================


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess TTS Edit Score dataset for multi-image VLM training"
    )

    # Data paths
    parser.add_argument(
        "--data_dir",
        default="/mnt/data2/datasets/tts-edit/edited_images/",
        help="Directory containing the TTS edit dataset",
    )
    parser.add_argument(
        "--local_dir",
        default="~/data/tts_edit",
        help="Local directory to save processed parquet files",
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="Optional HDFS directory to copy the processed files",
    )

    # Dataset configuration
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.9,
        help="Proportion of data to use for training",
    )

    # Score normalization
    parser.add_argument(
        "--score_median", type=float, default=0.9, help="Median score for normalization"
    )
    parser.add_argument(
        "--score_mad",
        type=float,
        default=0.1,
        help="MAD (Median Absolute Deviation) for score normalization",
    )

    # Sampling configuration
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to randomly select from the dataset",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--balanced_sampling",
        action="store_true",
        help="Use balanced sampling where each score group contributes equal samples",
    )

    return parser.parse_args()


def apply_sampling_strategy(
    data: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    """Apply the specified sampling strategy to the dataset."""
    if args.balanced_sampling:
        print("\n⚖️  Applying balanced sampling across score groups...")
        original_size = len(data)
        data = balanced_sample_by_score(
            data, args.score_median, args.score_mad, args.seed
        )
        print(f"Dataset reduced from {original_size} to {len(data)} samples")
    elif args.sample_size is not None and args.sample_size < len(data):
        print(
            f"\n🎲 Randomly sampling {args.sample_size} samples from {len(data)} total"
        )
        random.seed(args.seed)
        data = random.sample(data, args.sample_size)
        print(f"Using {len(data)} samples after sampling (seed={args.seed})")

    return data


def split_dataset(
    data: list[dict[str, Any]], train_split: float
) -> Tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split dataset into train and test sets."""
    train_size = int(len(data) * train_split)
    train_data = data[:train_size]
    test_data = data[train_size:]
    print(f"\n📂 Dataset split: {len(train_data)} train, {len(test_data)} test samples")
    return train_data, test_data


def process_datasets(
    train_data: list[dict[str, Any]],
    test_data: list[dict[str, Any]],
    data_source: str,
    score_median: float,
    score_mad: float,
) -> Tuple[datasets.Dataset, datasets.Dataset]:
    """Convert data to HuggingFace datasets and apply mapping."""
    print("🔄 Converting to HuggingFace datasets...")
    train_dataset = datasets.Dataset.from_list(train_data)
    print("✅ Train dataset created")

    test_dataset = datasets.Dataset.from_list(test_data)
    print("✅ Test dataset created")

    # Apply mapping function
    print("\n🔄 Processing train dataset...")
    train_dataset = train_dataset.map(
        function=make_map_fn("train", data_source, score_median, score_mad),
        with_indices=True,
        num_proc=8,
        desc="Processing train samples",
    )

    print("\n🔄 Processing test dataset...")
    test_dataset = test_dataset.map(
        function=make_map_fn("test", data_source, score_median, score_mad),
        with_indices=True,
        num_proc=8,
        desc="Processing test samples",
    )

    return train_dataset, test_dataset


def get_output_directory(
    local_dir: str, balanced_sampling: bool, sample_size: Optional[int]
) -> str:
    """Determine the output directory based on sampling configuration."""
    local_dir = os.path.expanduser(local_dir)

    if balanced_sampling:
        base_dir = local_dir.rstrip("/")
        if "tts_edit" in base_dir:
            local_dir = base_dir.replace("tts_edit", "tts_edit_balanced")
        else:
            local_dir = f"{base_dir}_balanced"
    elif sample_size is not None:
        base_dir = local_dir.rstrip("/")
        if "tts_edit" in base_dir:
            local_dir = base_dir.replace("tts_edit", f"tts_edit{sample_size}")
        else:
            local_dir = f"{base_dir}{sample_size}"

    return local_dir


def save_datasets(
    train_dataset: datasets.Dataset,
    test_dataset: datasets.Dataset,
    local_dir: str,
    hdfs_dir: Optional[str] = None,
) -> None:
    """Save datasets to local directory and optionally copy to HDFS."""
    os.makedirs(local_dir, exist_ok=True)
    print(f"\n📁 Output directory: {local_dir}")

    print("💾 Saving datasets...")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        print(f"\n☁️  Copying to HDFS: {hdfs_dir}")
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)


# ================== Main Function ==================


def main():
    args = parse_arguments()

    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    all_data = load_tts_edit_dataset(args.data_dir)

    if not all_data:
        raise ValueError(f"No valid data found in {args.data_dir}")

    print(f"Found {len(all_data)} valid samples")

    # Save original scores before sampling
    original_scores = [item["score"] for item in all_data]

    # Print original distribution if using balanced sampling
    if args.balanced_sampling:
        print_score_distribution(
            original_scores, "Original Score Distribution (before sampling)"
        )

    # Apply sampling strategy
    all_data = apply_sampling_strategy(all_data, args)

    # Print statistics after sampling
    sampled_scores = [item["score"] for item in all_data]
    if args.balanced_sampling:
        print_score_distribution(
            sampled_scores, "Raw Score Distribution (after balanced sampling)"
        )

        # Also show normalized score distribution to confirm balance
        normalized_scores_check = [
            normalize_score(item["score"], args.score_median, args.score_mad)
            for item in all_data
        ]
        print("\nNormalized Score Distribution (0-9 scale):")
        score_counts = {i: normalized_scores_check.count(i) for i in range(10)}
        for score in range(10):
            count = score_counts.get(score, 0)
            if count > 0:
                print(f"  Score {score}: {count} samples")
    else:
        print_score_distribution(sampled_scores, "Score Distribution")

    print_sample_examples(all_data)

    # Split dataset
    train_data, test_data = split_dataset(all_data, args.train_split)

    # Process datasets
    train_dataset, test_dataset = process_datasets(
        train_data, test_data, DATA_SOURCE, args.score_median, args.score_mad
    )

    # Determine output directory
    local_dir = get_output_directory(
        args.local_dir, args.balanced_sampling, args.sample_size
    )

    # Save datasets
    save_datasets(train_dataset, test_dataset, local_dir, args.hdfs_dir)

    # Print final statistics
    normalized_scores = [
        normalize_score(item["score"], args.score_median, args.score_mad)
        for item in all_data
    ]
    print_final_statistics(train_dataset, test_dataset, normalized_scores)

    print("\n✅ Data preprocessing completed successfully!")


if __name__ == "__main__":
    main()
