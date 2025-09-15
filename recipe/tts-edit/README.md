# TTS Edit VLM Training with VERL

This recipe provides scripts for training a Vision-Language Model (VLM) for image editing evaluation using the TTS Edit dataset with VERL's PPO implementation.

## Overview

This training pipeline implements:
- Multi-image input support (input image + edited image)
- Custom reward function for edit quality scoring
- GRPO (Group Relative Policy Optimization) advantage estimation
- LoRA fine-tuning for efficient training

## Files

- `tts_rm.py`: Data preprocessing script that converts TTS Edit dataset to VERL-compatible parquet format
- `tts_edit_reward.py`: Custom reward function for evaluating model outputs
- `run_qwen2_5_vl-3b_lora.sh`: Main training script for Qwen2.5-VL-3B model

## Setup

### 1. Prepare Dataset

First, preprocess your TTS Edit dataset:

```bash
python tts_rm.py \
    --data_dir /mnt/data2/datasets/tts-edit/edited_images/ \
    --local_dir ~/data/tts_edit \
    --train_split 0.9 \
    --score_median 0.9 \
    --score_mad 0.1
```

This will create:
- `~/data/tts_edit/train.parquet`: Training data
- `~/data/tts_edit/test.parquet`: Validation data

### 2. Run Training

Execute the training script:

```bash
bash run_qwen2_5_vl-3b_lora.sh
```

## Configuration

### Key Parameters

- **Model**: Qwen2.5-VL-3B-Instruct
- **LoRA Config**:
  - Rank: 128
  - Alpha: 32
  - Target modules: all-linear (excluding visual modules)
- **Training**:
  - Batch size: 256 (global)
  - Learning rate: 1e-6
  - Epochs: 10
  - Rollout samples (n): 4
- **Memory**:
  - GPU memory utilization: 0.5
  - FSDP with parameter offloading enabled

### Reward Function

The custom reward function (`tts_edit_reward.py`) evaluates:
1. **Format compliance**: Checks for `<think>` and `<answer>` tags (1 point)
2. **Score accuracy**: Compares predicted score with ground truth (±2 tolerance) (1 point)

Maximum reward: 2.0 points per response

## Multi-Image Support

The pipeline handles multi-image inputs natively:
- Input image: Original image before editing
- Output image: Edited image result
- Both images are processed together through the VLM processor
- Images are stored in the `images` field as a list

## Monitoring

Training progress can be monitored through:
- Console output
- WandB (if configured)
- Generated samples saved to:
  - `./rollout_generations/`: Training rollout samples
  - `./validation_generations/`: Validation samples

## Dataset Format

The preprocessed parquet files contain:
- `prompt`: Chat-formatted messages with system and user prompts
- `images`: List containing [input_image, output_image]
- `reward_model.ground_truth`: Normalized score (0-9)
- `data_source`: "tts_edit_score"
- `ability`: "image_editing_evaluation"

## Customization

To modify the training:

1. **Adjust hyperparameters**: Edit parameters in `run_qwen2_5_vl-3b_lora.sh`
2. **Change reward function**: Modify `tts_edit_reward.py`
3. **Update data preprocessing**: Modify `tts_rm.py` for different data formats

## Requirements

- VERL framework installed
- Qwen2.5-VL model weights
- TTS Edit dataset
- 4x GPUs with sufficient memory (recommended: A100 or similar)

## Notes

- The model is trained to output evaluations in a specific format with `<think>` tags for reasoning and `<answer>` tags for scores
- Scores are normalized to 0-9 range using median and MAD
- Visual modules are excluded from LoRA training to preserve visual understanding capabilities