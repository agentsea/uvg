from datasets import load_dataset
from uvg import Config, trainer
from peft import LoraConfig

dataset = load_dataset()

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