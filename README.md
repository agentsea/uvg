# vgrpo

## install

```bash
uv sync
uv pip install flash-attn --no-build-isolation
```

## vllm server

```bash
VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0 uv run vllm_server.py --model "Qwen/Qwen2.5-3B-Instruct"
```

## training

```bash
CUDA_VISIBLE_DEVICES=1 uv run unsloth_train.py
CUDA_VISIBLE_DEVICES=1 uv run torchrun --nproc_per_node=1 unsloth_train_dist.py --use_unsloth
```

```
uv run unsloth_train.py \
    --use_wandb

```

TODO:

check why loss is being computed as 0 always
