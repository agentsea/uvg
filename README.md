# UVG - Unsloth VLM GRPO

A Python library for training VLMs using GRPO with Unsloth acceleration.

## Installation

```bash
git clone https://github.com/agentsea/uvg.git
cd uvg
uv sync && uv pip install flash-attn --no-build-isolation && uv pip install -e
```

To add Qwen-specific deps:

```bash
uv pip install -e ".[qwen]"
```

## Usage

```python
from uvg import Config, train

# Create configuration
config = Config(
    model_id="Qwen/Qwen2.5-3B-Instruct",
    dataset_id="openai/gsm8k",
    use_wandb=True,
    wandb_project="my-project"
)

# Start training
train(config)
```
