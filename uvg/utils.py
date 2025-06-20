import inspect
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import unsloth  # noqa
from huggingface_hub import HfApi, create_repo
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from torch import Tensor
from torch.utils.data import Sampler

import wandb

from .config import Config


def log_completions(
    prompts: list[str],
    completions: list[str],
    rewards: dict[str, list[float]],
    advantages: Tensor,
) -> None:
    console = Console()
    table = Table(show_header=True, header_style="bold white", expand=True)
    table.add_column("Prompt", style="bright_yellow")
    table.add_column("Completion", style="bright_green")
    for reward_name in rewards.keys():
        table.add_column(reward_name, style="bold cyan", justify="right")
    table.add_column("Advantage", style="bold magenta", justify="right")
    for i in range(len(prompts)):
        reward_values = [f"{rewards[key][i]:.2f}" for key in rewards.keys()]
        table.add_row(
            Text(prompts[i]),
            Text(completions[i]),
            *reward_values,
            f"{advantages[i]:.2f}",
        )
        table.add_section()
    panel = Panel(table, expand=False, border_style="bold white")
    console.print(panel)


def accepts_kwarg(fn, name: str) -> bool:
    try:
        inspect.signature(fn).bind_partial(**{name: None})
        return True
    except TypeError:
        return False


def save_checkpoint(
    model: Any,
    processor: Any,
    output_dir: str = "checkpoint",
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
    hub_private: bool = False,
    commit_msg: str = "checkpoint",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(output_path)
    model.save_pretrained(output_path)
    if push_to_hub:
        _push_folder_to_hub(
            folder=output_path,
            repo_id=hub_repo_id or output_path.name,
            private=hub_private,
            commit_message=commit_msg,
        )


def _push_folder_to_hub(folder: Path, repo_id: str, private: bool, commit_message: str):
    api = HfApi()
    if not api.repo_exists(repo_id):
        create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(folder), repo_id=repo_id, commit_message=commit_message
    )


def init_wandb(model_id: str, wandb_project: str | None) -> None:
    run_name = f"{model_id.split('/')[-1]}"
    wandb.init(project=wandb_project, name=run_name)
    if getattr(wandb, "define_metric", None):
        wandb.define_metric("train/global_step") 
        wandb.define_metric("*", step_metric="train/global_step", step_sync=True)


def log_wandb(metrics: defaultdict[str, list[float]], step: int, prefix: str) -> None:
    wandb_log_payload = {f"{prefix}/{k}": v[-1] for k, v in metrics.items() if v}
    wandb.log({**wandb_log_payload, **{"train/global_step": step}}, step=step)


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


class RepeatSampler(Sampler):
    def __init__(
        self,
        data_source,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed
        if shuffle:
            self.generator = torch.Generator()
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            indexes = torch.randperm(
                self.num_samples, generator=self.generator
            ).tolist()
        else:
            indexes = list(range(self.num_samples))
        indexes = [
            indexes[i : i + self.batch_size]
            for i in range(0, len(indexes), self.batch_size)
        ]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]
        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


def validate_cfg(cfg: Config) -> Config:
    cfg.dtype = getattr(torch, cfg.dtype)
    if cfg.use_wandb:
        assert cfg.wandb_project is not None
    if cfg.push_to_hub:
        assert cfg.hub_repo_id is not None
    if cfg.gradient_checkpoint:
        cfg.use_cache = False
    if cfg.collate_fn is None:
        cfg.collate_fn = lambda batch: batch
    return cfg
