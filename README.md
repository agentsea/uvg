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

Check the `examples/` folder.


## TODO:

- update save_checkpoint function to comply with unsloth
- maybe replace RepeatSampler / build_batch_sampler
