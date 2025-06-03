import copy
from dataclasses import dataclass
from typing import Callable


SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def gsm8k_data_collator(batch: list[dict]) -> list[dict]:
    processed_samples = []
    for original_sample in batch:
        raw_question = original_sample['prompt']
        raw_answer_text = original_sample['answer']
        formatted_prompt = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': raw_question}
        ]
        cleaned_answer = extract_hash_answer(raw_answer_text)
        processed_sample = {
            'prompt': formatted_prompt,
            'answer': cleaned_answer
        }
        for key, value in original_sample.items():
            if key not in ['prompt', 'answer']:
                processed_sample[key] = value
        processed_samples.append(processed_sample)
    return processed_samples


@dataclass
class Config:
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    dataset_id: str = "openai/gsm8k"
    collate_fn: Callable[[list[dict]], list[dict]] | None = gsm8k_data_collator
    no_apply_chat_template: bool = False
    extra_columns: str | None = "answer"
    batch_size: int = 8
    max_completion_len: int = 200
    num_generations: int = 8
    num_epochs: int = 1
    learning_rate: float = 5e-6
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    grad_norm: float = 0.1
    epsilon: float = 0.2
    epsilon_high: float = 0.2
    beta: float = 0.04
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    min_p: float | None = None
    repetition_penalty: float = 1.0
    use_peft: bool = False
    use_fsdp: bool = False
    bf16: bool = False
    fsdp_bf16: bool = False
    gradient_checkpoint: bool = False
    log_steps: int = 1
    save_steps: int = 250
    use_wandb: bool = False
    wandb_project: str = "test"
    push_to_hub: bool = False
    hub_repo_id: str = "nph4rd/test_save"
    hub_private: bool = True
    seed: int = 42
    dtype: str = "bfloat16"
    use_cache: bool = False
    use_unsloth: bool = False
