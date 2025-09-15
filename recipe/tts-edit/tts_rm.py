#!/usr/bin/env python
"""
Preprocess the TTS Edit Score dataset for multi-image VLM training with VERL PPO.
This script handles multi-image inputs (input_image and output_image) for image editing evaluation.
"""

import argparse
import glob
import json
import os
from string import Template
from typing import Any

import datasets
from PIL import Image

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

# Instruction following prompt for the model
INSTRUCTION_FOLLOWING = (
    "请首先使用<think></think>标签进行思考，分析编辑前后的图像变化，然后在<answer></answer>标签中给出0-9的评分。"
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
    input_images = glob.glob(os.path.join(data_dir, "**", "*_input.jpg"), recursive=True)

    for idx, input_image_path in enumerate(input_images):
        # Construct paths for related files
        output_image_path = input_image_path.replace("_input.jpg", "_output.jpg")
        meta_path = input_image_path.replace("_input.jpg", "_meta.json")
        score_path = input_image_path.replace("_input.jpg", "_score.json")

        # Skip if any required file is missing
        if not all(os.path.exists(p) for p in [output_image_path, meta_path, score_path]):
            continue

        # Load metadata
        with open(meta_path) as f:
            metadata = json.load(f)
            instruction = metadata.get("instruction", "")

        # Load score
        with open(score_path) as f:
            score_data = json.load(f)
            score = score_data.get("score", 5.0)

        # Load images
        input_image = Image.open(input_image_path).convert("RGB")
        output_image = Image.open(output_image_path).convert("RGB")

        # Resize images for efficiency (optional, adjust as needed)
        max_size = 512
        if input_image.width > max_size or input_image.height > max_size:
            input_image.thumbnail((max_size, max_size), Image.LANCZOS)
        if output_image.width > max_size or output_image.height > max_size:
            output_image.thumbnail((max_size, max_size), Image.LANCZOS)

        dataset_items.append(
            {
                "idx": idx,
                "input_image": input_image,
                "output_image": output_image,
                "instruction": instruction,
                "score": score,
                "input_image_path": input_image_path,
                "split": split,
            }
        )

    return dataset_items


def normalize_score(score: float, median: float = 0.9, mad: float = 0.1) -> int:
    """Normalize score to 0-9 range using median and MAD."""
    normalized = (score - median) / mad
    normalized = normalized * 4.5 + 4.5
    return int(max(0, min(9, normalized)))


def make_map_fn(split: str, data_source: str, score_median: float = 0.9, score_mad: float = 0.1):
    """
    Create a mapping function for dataset processing.
    This handles multi-image inputs for VLM training.
    """

    def process_fn(example: dict[str, Any], idx: int) -> dict[str, Any]:
        # Extract data from example
        instruction = example.get("instruction", "")
        score = example.get("score", 5.0)
        input_image = example.get("input_image")
        output_image = example.get("output_image")

        # Normalize score
        normalized_score = normalize_score(score, score_median, score_mad)

        # Construct the prompt with instruction following
        scorer_prompt = SCORER_PROMPT_TEMPLATE.substitute(instruction=instruction)
        full_prompt = scorer_prompt + "\n" + INSTRUCTION_FOLLOWING

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
                "ground_truth": str(normalized_score),  # Store as string for consistency
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "instruction": instruction,
                "original_score": score,
                "normalized_score": normalized_score,
                "input_image_path": example.get("input_image_path", ""),
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
        "--local_dir", default="~/data/tts_edit", help="Local directory to save processed parquet files"
    )
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy the processed files")
    parser.add_argument("--train_split", type=float, default=0.9, help="Proportion of data to use for training")
    parser.add_argument("--score_median", type=float, default=0.9, help="Median score for normalization")
    parser.add_argument(
        "--score_mad", type=float, default=0.1, help="MAD (Median Absolute Deviation) for score normalization"
    )

    args = parser.parse_args()

    data_source = "tts_edit_score"

    # Load dataset from directory
    print(f"Loading dataset from {args.data_dir}...")
    all_data = load_tts_edit_dataset(args.data_dir, split="all")

    if not all_data:
        raise ValueError(f"No valid data found in {args.data_dir}")

    print(f"Found {len(all_data)} valid samples")

    # Split into train and test
    train_size = int(len(all_data) * args.train_split)
    train_data = all_data[:train_size]
    test_data = all_data[train_size:]

    # Convert to HuggingFace datasets
    train_dataset = datasets.Dataset.from_list(train_data)
    test_dataset = datasets.Dataset.from_list(test_data)

    # Apply mapping function
    print("Processing train dataset...")
    train_dataset = train_dataset.map(
        function=make_map_fn("train", data_source, args.score_median, args.score_mad), with_indices=True, num_proc=8
    )

    print("Processing test dataset...")
    test_dataset = test_dataset.map(
        function=make_map_fn("test", data_source, args.score_median, args.score_mad), with_indices=True, num_proc=8
    )

    # Create local directory if it doesn't exist
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Save to parquet format
    print(f"Saving datasets to {local_dir}...")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        print(f"Copying to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)

    print("Data preprocessing completed successfully!")


if __name__ == "__main__":
    main()
