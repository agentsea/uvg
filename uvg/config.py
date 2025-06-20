from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Config:
    model_id: str
    collate_fn: Callable[[list[dict]], list[dict]] | None = None
    no_apply_chat_template: bool = False
    max_completion_len: int = 2048
    num_generations: int = 8
    num_epochs: int = 1
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_norm: float = 0.001
    epsilon: float = 0.2
    epsilon_high: float = 0.2
    beta: float = 0.001
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float = 1.0
    bf16: bool = False
    gradient_checkpoint: bool = False
    log_steps: int = 1
    save_steps: int = 250
    use_wandb: bool = False
    wandb_project: str | None = None
    push_to_hub: bool = False
    hub_repo_id: str | None = None
    hub_private: bool = True
    seed: int = 42
    dtype: str = "bfloat16"
    use_cache: bool = False
    lora_target_modules: list = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    lora_alpha: int = 64
    lora_rank: int = 64
    log_completions: bool = False
    lr_scheduler: str = "cosine"
    eval_steps: int = 100
