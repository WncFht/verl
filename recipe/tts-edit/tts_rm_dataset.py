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
from typing import Any

import datasets
import numpy as np
from PIL import Image
from tqdm import tqdm

from verl.utils.hdfs_io import copy, makedirs

# System prompt for the AI assistant
SYSTEM_PROMPT = """你是一个AI助理。

在正式回答之前，你总是可以使用<think></think>标签进行详细地思考。
请保持<think></think>标签内的内容与问题相关且有意义，不要随意编造，也不要重复冗余。
若需要思考再回答，请保持答案与思考过程一致。
"""

# Scorer prompt template for image editing evaluation
SCORER_PROMPT_TEMPLATE = Template(
    """你是一个图像编辑评分模型。你的任务是根据给定的编辑指令，评估编辑后的图像相对于输入图像的质量。
你的回答必须是0到9之间的整数，0表示编辑效果非常差，9表示编辑效果非常好。
你必须用<answer></answer>标签括起你的最终答案。

编辑指令： $instruction
输入图像：
"""
)


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


def print_dataset_statistics(
    image_sizes: list[dict], instruction_lengths: list[int], scores: list[float]
):
    """Print statistics about the dataset."""
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    # Image size statistics
    if image_sizes:
        input_widths = [s["input"][0] for s in image_sizes]
        input_heights = [s["input"][1] for s in image_sizes]
        output_widths = [s["output"][0] for s in image_sizes]
        output_heights = [s["output"][1] for s in image_sizes]

        print("\n📊 Image Dimensions:")
        print("  Input images:")
        print(
            f"    Width:  min={min(input_widths)}, max={max(input_widths)}, mean={np.mean(input_widths):.1f}"
        )
        print(
            f"    Height: min={min(input_heights)}, max={max(input_heights)}, mean={np.mean(input_heights):.1f}"
        )
        print("  Output images:")
        print(
            f"    Width:  min={min(output_widths)}, max={max(output_widths)}, mean={np.mean(output_widths):.1f}"
        )
        print(
            f"    Height: min={min(output_heights)}, max={max(output_heights)}, mean={np.mean(output_heights):.1f}"
        )

    # Instruction length statistics
    if instruction_lengths:
        print("\n📝 Instruction Lengths:")
        print(f"    Min: {min(instruction_lengths)} chars")
        print(f"    Max: {max(instruction_lengths)} chars")
        print(f"    Mean: {np.mean(instruction_lengths):.1f} chars")
        print(f"    Median: {np.median(instruction_lengths):.1f} chars")

    # Score statistics
    if scores:
        print("\n🎯 Score Distribution:")
        print(f"    Min: {min(scores):.2f}")
        print(f"    Max: {max(scores):.2f}")
        print(f"    Mean: {np.mean(scores):.2f}")
        print(f"    Std: {np.std(scores):.2f}")
        print(f"    Median: {np.median(scores):.2f}")

        # Score histogram
        hist, bins = np.histogram(scores, bins=10)
        print("    Distribution (10 bins):")
        for i in range(len(hist)):
            bar = "█" * int(hist[i] * 20 / max(hist))
            print(f"      [{bins[i]:.2f}-{bins[i + 1]:.2f}]: {bar} ({hist[i]})")

    print("=" * 60 + "\n")


def print_sample_examples(samples: list[dict[str, Any]]):
    """Print a few sample examples from the dataset."""
    print("\n" + "=" * 60)
    print("Sample Examples")
    print("=" * 60)

    for i, sample in enumerate(samples, 1):
        print(f"\n📌 Sample {i}:")
        print(f"  File: {os.path.basename(sample['input_image_path'])}")
        print(
            f"  Instruction: {sample['instruction'][:100]}{'...' if len(sample['instruction']) > 100 else ''}"
        )
        print(f"  Score: {sample['score']:.2f}")
        print(f"  Input image path: {sample['input_image_path']}")
        print(f"  Output image path: {sample['output_image_path']}")

    print("=" * 60 + "\n")


def normalize_score(score: float, median: float = 0.9, mad: float = 0.1) -> int:
    """Normalize score to 0-9 range using median and MAD."""
    normalized = (score - median) / mad
    normalized = normalized * 4.5 + 4.5
    return int(max(0, min(9, normalized)))


def print_final_statistics(train_dataset, test_dataset, args):
    """Print final statistics about the processed datasets."""
    print("\n" + "=" * 60)
    print("Final Dataset Summary")
    print("=" * 60)

    print("\n📈 Dataset Sizes:")
    print(f"  Train dataset: {len(train_dataset)} samples")
    print(f"  Test dataset: {len(test_dataset)} samples")
    print(f"  Total: {len(train_dataset) + len(test_dataset)} samples")

    if args.sample_size is not None:
        print("\n🎲 Sampling Info:")
        print(f"  Sampled: {args.sample_size} samples")
        print(f"  Random seed: {args.seed}")

    # Analyze prompts
    print("\n💬 Prompt Analysis:")
    if len(train_dataset) > 0:
        train_sample = train_dataset[0]
        if "prompt" in train_sample:
            prompt_example = train_sample["prompt"]
            print(f"  Number of messages per prompt: {len(prompt_example)}")
            print(f"  Message roles: {[msg['role'] for msg in prompt_example]}")

            # Estimate token count (rough approximation)
            total_text_length = sum(
                len(str(msg.get("content", ""))) for msg in prompt_example
            )
            estimated_tokens = (
                total_text_length // 4
            )  # Rough estimate: 4 chars per token
            print(f"  Estimated tokens per prompt: ~{estimated_tokens}")

    # Score normalization info
    print("\n🎯 Score Normalization:")
    print(f"  Median: {args.score_median}")
    print(f"  MAD: {args.score_mad}")
    print("  Output range: 0-9")

    # Sample normalized scores
    if len(train_dataset) > 0 and "reward_model" in train_dataset[0]:
        sample_scores = [
            train_dataset[i]["reward_model"]["ground_truth"]
            for i in range(min(10, len(train_dataset)))
        ]
        print(f"  Sample normalized scores: {sample_scores}")

    # Data format info
    print("\n📦 Data Format:")
    print("  Image key: 'images'")
    print("  Images per sample: 2 (input + output)")
    print(
        "  Image resize strategy: Both images resized to output_size//2 (same dimensions)"
    )
    print("  Ability: 'image_editing_evaluation'")
    print("  Data source: 'tts_edit_score'")

    print("=" * 60 + "\n")


def make_map_fn(
    split: str, data_source: str, score_median: float = 0.9, score_mad: float = 0.1
):
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


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.9,
        help="Proportion of data to use for training",
    )
    parser.add_argument(
        "--score_median", type=float, default=0.9, help="Median score for normalization"
    )
    parser.add_argument(
        "--score_mad",
        type=float,
        default=0.1,
        help="MAD (Median Absolute Deviation) for score normalization",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to randomly select from the dataset. If None, use all samples.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")

    args = parser.parse_args()

    data_source = "tts_edit_score"

    # Load dataset from directory
    print(f"Loading dataset from {args.data_dir}...")
    all_data = load_tts_edit_dataset(args.data_dir, split="all")

    if not all_data:
        raise ValueError(f"No valid data found in {args.data_dir}")

    print(f"Found {len(all_data)} valid samples")

    # Random sampling if sample_size is specified
    if args.sample_size is not None and args.sample_size < len(all_data):
        print(
            f"\n🎲 Randomly sampling {args.sample_size} samples from {len(all_data)} total samples"
        )
        random.seed(args.seed)
        all_data = random.sample(all_data, args.sample_size)
        print(f"Using {len(all_data)} samples after sampling (seed={args.seed})")

    # Calculate statistics for the (possibly sampled) dataset
    print("\n📊 Computing dataset statistics...")
    image_sizes = []
    instruction_lengths = []
    scores = []

    for i, item in enumerate(all_data):
        instruction_lengths.append(len(item["instruction"]))
        scores.append(item["score"])

        # Load images for size statistics
        if i < 10:  # Only load first 10 images for size stats
            input_image = Image.open(item["input_image_path"]).convert("RGB")
            output_image = Image.open(item["output_image_path"]).convert("RGB")
            image_sizes.append(
                {
                    "input": (input_image.width, input_image.height),
                    "output": (output_image.width, output_image.height),
                }
            )
            input_image.close()
            output_image.close()

    # Print statistics for the sampled dataset
    if all_data:
        print_dataset_statistics(image_sizes, instruction_lengths, scores)
        print_sample_examples(all_data[:2])  # Print first 2 samples

    # Split into train and test
    train_size = int(len(all_data) * args.train_split)
    train_data = all_data[:train_size]
    test_data = all_data[train_size:]

    print(f"\n📂 Dataset split: {len(train_data)} train, {len(test_data)} test samples")

    # Convert to HuggingFace datasets (this step can be slow with images)
    print("🔄 Converting to HuggingFace datasets (this may take a while)...")
    train_dataset = datasets.Dataset.from_list(train_data)
    print("✅ Train dataset created")

    test_dataset = datasets.Dataset.from_list(test_data)
    print("✅ Test dataset created")

    # Apply mapping function
    print("\n🔄 Processing train dataset...")
    train_dataset = train_dataset.map(
        function=make_map_fn("train", data_source, args.score_median, args.score_mad),
        with_indices=True,
        num_proc=8,
        desc="Processing train samples",
    )

    print("\n🔄 Processing test dataset...")
    test_dataset = test_dataset.map(
        function=make_map_fn("test", data_source, args.score_median, args.score_mad),
        with_indices=True,
        num_proc=8,
        desc="Processing test samples",
    )

    # Create local directory if it doesn't exist
    # Append sample size to directory name if sampling was used
    local_dir = os.path.expanduser(args.local_dir)
    if args.sample_size is not None:
        # Extract base name and add sample size
        base_dir = local_dir.rstrip("/")
        if "tts_edit" in base_dir:
            local_dir = base_dir.replace("tts_edit", f"tts_edit{args.sample_size}")
        else:
            local_dir = f"{base_dir}{args.sample_size}"

    os.makedirs(local_dir, exist_ok=True)
    print(f"\n📁 Output directory: {local_dir}")

    # Save to parquet format
    print("💾 Saving datasets...")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    # Print final statistics
    print_final_statistics(train_dataset, test_dataset, args)

    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        print(f"\n☁️  Copying to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)

    print("\n✅ Data preprocessing completed successfully!")


if __name__ == "__main__":
    main()
