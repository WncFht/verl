import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Compute reward score for TTS edit evaluation task.

    Args:
        data_source: Dataset identifier (e.g., "tts_edit_score")
        solution_str: Model's generated response
        ground_truth: Ground truth score (as string, "0"-"9")
        extra_info: Additional information (optional)

    Returns:
        float: Reward score (0.0 to 2.0)
    """
    # 提取<think>和<answer>标签内容
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

    # 检查是否有think标签
    has_think = bool(think_pattern.search(solution_str))

    # 检查是否有answer标签
    answer_match = answer_pattern.search(solution_str)

    # 如果缺少必要的标签，返回0分
    if not answer_match:
        return 0.0

    # 格式奖励：有think标签给1分，没有给0分
    format_reward = 1.0 if has_think else 0.0

    # 准确性奖励
    accuracy_reward = 0.0
    try:
        # 提取预测分数（支持整数或浮点数）
        predicted_str = answer_match.group(1).strip()
        predicted_score = float(predicted_str)

        # 四舍五入到最近的整数
        predicted_score = round(predicted_score)

        # 确保在有效范围内
        if 0 <= predicted_score <= 9:
            # 将ground truth转换为数字
            true_score = float(ground_truth)
            true_score = round(true_score)

            # 如果预测分数与真实分数差距在2以内，给1分
            if abs(predicted_score - true_score) <= 2:
                accuracy_reward = 1.0
    except (ValueError, TypeError):
        # 如果无法解析分数，准确性奖励为0
        pass

    # 总奖励 = 格式奖励 + 准确性奖励
    total_reward = format_reward + accuracy_reward

    return total_reward
