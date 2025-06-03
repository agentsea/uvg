# UVG - Unsloth VLM GRPO

A Python library for training VLMs using GRPO with Unsloth acceleration.

## Installation

```bash
git clone https://github.com/agentsea/uvg.git
cd uvg
uv sync && uv pip install flash-attn --no-build-isolation && uv pip install -e .
```

To add Qwen-specific deps:

```bash
uv pip install -e ".[qwen]"
```

## Usage

```python
from datasets import load_dataset
from uvg import Config, trainer
from peft import LoraConfig

dataset = load_dataset(..)

lora_config = LoraConfig(..)

def reward_len(completions, **kwargs):
    ...
    return [...]

config = Config()

trainer(
    reward_funcs=reward_len,
    config=config,
    train_dataset=dataset,
    lora_config=lora_config,
)
```
