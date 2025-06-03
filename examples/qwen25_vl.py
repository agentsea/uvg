from datasets import load_dataset
from uvg import Config, trainer
from peft import LoraConfig
from qwen_vl_utils import process_vision_info
import copy

SYSTEM_PROMPT = f"""Answer the questions.

Respond in the following format:
<think></think><answer></answer>"""

def collate_fn(batch: list[dict]) -> list[dict]:
    processed_samples = []
    for sample in batch:
        messages = []
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
        prompt_data = sample["question"]
        processed_prompt = copy.deepcopy(prompt_data)
        processed_images = []
        if "images" in sample:
            image_data = sample["images"]
            image_index = 0
            for message in processed_prompt:
                for content in message["content"]:
                    if isinstance(content, dict) and content.get("type") == "image":
                        content["image"] = image_data[image_index]
                        image_index += 1
            processed_images, *_ = process_vision_info(processed_prompt)
        processed_sample = {"prompt": processed_prompt, "images": processed_images}
        for key, value in sample.items():
            if key not in ["prompt", "images"]:
                processed_sample[key] = value
        processed_samples.append(processed_sample)
    return processed_samples


dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")

import pdb;pdb.set_trace()

lora_config = LoraConfig(lora_alpha=64, lora_dropout=0.05, r=32, bias="none", target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")

def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

config = Config(model="Qwen/Qwen2.5-VL-3B-Instruct")

trainer(
    reward_funcs=reward_len,
    config=config,
    train_dataset=dataset,
    lora_config=lora_config,
)