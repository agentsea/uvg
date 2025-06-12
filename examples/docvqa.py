import re

from datasets import load_dataset
from qwen_vl_utils import process_vision_info

from uvg import Config, train

SYSTEM_PROMPT = """Answer the questions.

Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>"""


def collate_fn(batch: list[dict]) -> list[dict]:
    processed_samples = []
    for sample in batch:
        messages = []
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
        content_block = []
        content_block.append({"type": "text", "text": sample["question"]})
        content_block.append(
            {
                "type": "image",
                "image": sample["image"],  # only one image in this ds
                "resized_height": 768,  # XGA resolution
                "resized_width": 1024,
            }
        )
        messages.append({"role": "user", "content": content_block})
        processed_images, *_ = process_vision_info(  # process with qwen utils
            messages.copy()
        )
        sample["prompt"] = messages
        sample["images"] = processed_images
        sample["answer"] = sample["answers"]
        processed_samples.append(sample)
    return processed_samples


dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")


def format_reward_func(
    completions: list[list[dict[str, str]]], **kwargs
) -> list[float]:
    def get_assistant_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return [msg for msg in messages if msg.get("role") == "assistant"]

    def parse_xml_content(text: str, tag: str, strip: bool = True) -> str | None:
        pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            content = match.group(1)
            return content.strip() if strip else content
        return None

    def check_message_format(content: str) -> float:
        think_content = parse_xml_content(content, "think")
        answer_content = parse_xml_content(content, "answer")
        think_content_no_strip = parse_xml_content(content, "think", strip=False)
        answer_content_no_strip = parse_xml_content(content, "answer", strip=False)
        score = 0.0
        fields_present = 0
        if think_content is not None:
            fields_present += 1
        if answer_content is not None:
            fields_present += 1
        field_ratio = fields_present / 2.0
        score += 0.4 * field_ratio
        has_correct_spacing = True
        if think_content is not None and think_content_no_strip is None:
            has_correct_spacing = False
        if answer_content is not None and answer_content_no_strip is None:
            has_correct_spacing = False
        if has_correct_spacing:
            score += 0.2
        if content.strip().startswith("<think>"):
            score += 0.2
        if content.strip().endswith("</answer>"):
            score += 0.2
        return score

    def score_single_completion(completion: list[dict[str, str]]) -> float:
        assistant_messages = get_assistant_messages(completion)
        if not assistant_messages:
            return 0.0
        format_scores = []
        for msg in assistant_messages:
            content = msg.get("content", "")
            message_score = check_message_format(content)
            format_scores.append(message_score)
        if not format_scores:
            return 0.0
        return sum(format_scores) / len(format_scores)

    batch_scores = []
    for completion in completions:
        score = score_single_completion(completion)
        batch_scores.append(score)

    return batch_scores


def correctness_reward_func(
    completions: list[list[dict[str, str]]], **kwargs
) -> list[float]:
    def get_assistant_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return [msg for msg in messages if msg.get("role") == "assistant"]

    def parse_xml_content(text: str, tag: str, strip: bool = True) -> str | None:
        pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            content = match.group(1)
            return content.strip() if strip else content
        return None

    scores = []
    for i, completion in enumerate(completions):
        assistant_messages = get_assistant_messages(completion)
        if assistant_messages is None:
            scores.append(0.0)
            continue
        msgs_scores = []
        for msg in assistant_messages:
            content = msg.get("content", "")
            answer = parse_xml_content(content, "answer")
            if answer is None:
                continue
            gt_answers = kwargs["answer"][i]
            mean_gt_len = sum([len(gt_answer) for gt_answer in gt_answers]) / len(
                gt_answers
            )
            diff_from_mean = min(
                mean_gt_len / len(answer), 1.0
            )  # penalize long answers
            if answer in gt_answers:
                msgs_scores.append(2.0)
            elif answer.lower() in [ans.lower() for ans in gt_answers]:
                msgs_scores.append(1.0)
            elif any(ans.lower() in answer.lower() for ans in gt_answers):
                msgs_scores.append(diff_from_mean)
        if msgs_scores == []:
            scores.append(0.0)
        else:
            scores.append((sum(msgs_scores) / len(msgs_scores)) / 2.0)
    return scores


config = Config(
    model_id="Qwen/Qwen2.5-VL-32B-Instruct",
    collate_fn=collate_fn,
    bf16=True,
    log_completions=True,
    wandb_project="docvqa",
    use_wandb=True,
)

train(
    reward_funcs=[format_reward_func, correctness_reward_func],
    cfg=config,
    train_dataset=dataset,
)
