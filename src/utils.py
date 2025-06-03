import argparse
import inspect
import os
import random
import re
from collections import defaultdict
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import HfApi, create_repo
from peft.tuners.lora import LoraLayer
from torch import Tensor, nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import BatchSampler, Sampler
import unsloth
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    PreTrainedModel,
)

import wandb
from config import TrainConfig
from vllm_client import VLLMClient


def accepts_kwarg(fn, name: str) -> bool:
    try:
        inspect.signature(fn).bind_partial(**{name: None})
        return True
    except TypeError:
        return False


def smart_load(model_id: str, **hf_kwargs) -> PreTrainedModel:
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    for arch in cfg.architectures or []:
        try:
            cls = getattr(import_module("transformers"), arch)
            return cls.from_pretrained(
                model_id,
                trust_remote_code=True,
                **hf_kwargs,
            )
        except (AttributeError, ImportError, ValueError):
            pass

    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoModelForVision2Seq,
    )

    for auto_cls in (
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoModelForVision2Seq,
        AutoModel,
    ):
        try:
            return auto_cls.from_pretrained(
                model_id,
                trust_remote_code=True,
                **hf_kwargs,
            )
        except ValueError:
            continue

    raise RuntimeError(f"No suitable loader found for model type {cfg.model_type!r}")


def init_distributed() -> tuple[int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    return local_rank, device


def sync_fsdp_params_to_vllm(
    module: nn.Module,
    vllm_client: VLLMClient | None,
    prefix: str = "",
    visited: set[str] | None = None,
    peft: bool = False,
) -> None:
    LORA_PAT = re.compile(r"\.lora_[AB]\.")
    rank = dist.get_rank()
    if visited is None:
        visited = set()
    for child_name, child_module in module.named_children():
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        sync_fsdp_params_to_vllm(
            child_module, vllm_client, prefix=child_prefix, visited=visited, peft=peft
        )
    if isinstance(module, FSDP):
        with FSDP.summon_full_params(module, recurse=False, writeback=False):
            merged = []
            if peft:
                for m in module.modules():
                    if isinstance(m, LoraLayer):
                        m.merge()
                        merged.append(m)
            for param_name, param in module.named_parameters():
                full_name = f"{prefix}.{param_name}" if prefix else param_name
                subs = ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.")
                if FSDP:
                    if LORA_PAT.search(full_name):
                        continue
                    subs = (
                        "base_model.model.",
                        "base_model.",
                        "_fsdp_wrapped_module.",
                        "_checkpoint_wrapped_module.",
                        ".base_layer",
                        "modules_to_save.default.",
                    )
                for extra in subs:
                    full_name = full_name.replace(extra, "")
                if full_name in visited:
                    continue
                visited.add(full_name)
                if rank == 0:
                    vllm_client.update_named_param(full_name, param.data)
            for m in merged:
                m.unmerge()


def gather(tensor: Tensor) -> Tensor:
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor.unsqueeze(0) if tensor.dim() == 0 else tensor
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    if tensor.dim() == 0:
        return torch.stack(tensor_list)
    else:
        return torch.cat(tensor_list, dim=0)


def save_checkpoint(
    model: FSDP | DDP,
    processor: AutoProcessor,
    output_dir: str = "checkpoint",
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
    hub_private: bool = False,
    commit_msg: str = "checkpoint",
) -> None:
    rank = dist.get_rank() if dist.is_initialized() else 0
    if isinstance(model, DDP):
        model = model.module
    if rank == 0:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        processor.save_pretrained(output_dir)
        model.config.save_pretrained(output_dir)
    opts = StateDictOptions(
        full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True
    )
    state_dict, _ = get_state_dict(model, {}, options=opts)
    if rank == 0:
        model.save_pretrained(output_dir, state_dict=state_dict)
    if push_to_hub and rank == 0:
        _push_folder_to_hub(
            folder=output_dir,
            repo_id=hub_repo_id or output_dir.name,
            private=hub_private,
            commit_message=commit_msg,
        )
    dist.barrier() if dist.is_initialized() else None


def _push_folder_to_hub(folder: Path, repo_id: str, private: bool, commit_message: str):
    api = HfApi()
    if not api.repo_exists(repo_id):
        create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(folder), repo_id=repo_id, commit_message=commit_message
    )


def init_wandb(model_id: str, wandb_project: str) -> None:
    run_name = f"{model_id.split('/')[-1]}"
    wandb.init(project=wandb_project, name=run_name)


def log_wandb(metrics: defaultdict[str, list[float]]) -> None:
    wandb_log_payload = {f"train/{k}": v[-1] for k, v in metrics.items() if v}
    wandb.log(wandb_log_payload)


def gather_object(obj: Any) -> list[Any]:
    world_size = dist.get_world_size()
    obj_list = [None] * world_size
    dist.all_gather_object(obj_list, obj)
    if isinstance(obj_list[0], list):
        return sum(obj_list, [])
    return obj_list


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


def create_reference_model(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    parameter_names = [n for n, _ in model.named_parameters()]
    ref_model = deepcopy(model)
    for param_name in parameter_names:
        param = ref_model.get_parameter(param_name)
        param.requires_grad = False
    return ref_model.eval()


def build_batch_sampler(
    sampler: Sampler,
    batch_size: int,
    num_replicas: int,
    rank: int,
    drop_last: bool = False,
) -> Sampler:
    batch_sampler = BatchSampler(
        sampler=sampler, batch_size=batch_size, drop_last=drop_last
    )
    dist_batch_sampler = DistBatchSampler(
        batch_sampler=batch_sampler,
        num_replicas=num_replicas,
        rank=rank,
        drop_last=drop_last,
    )
    return dist_batch_sampler


class DistBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        batch_sampler: BatchSampler,
        num_replicas: int,
        rank: int,
        drop_last: bool = False,
    ):
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in [0, {num_replicas - 1}]"
            )
        self.batch_sampler = batch_sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.epoch = 0
        if self.drop_last:
            self.num_samples = len(self.batch_sampler) // self.num_replicas
        else:
            self.num_samples = (
                len(self.batch_sampler) + self.num_replicas - 1
            ) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        if hasattr(self.batch_sampler.sampler, "set_epoch"):
            self.batch_sampler.sampler.set_epoch(self.epoch)
        elif (
            hasattr(self.batch_sampler.sampler, "generator")
            and hasattr(self.batch_sampler.sampler, "seed")
            and self.batch_sampler.sampler.generator is not None
        ):
            self.batch_sampler.sampler.generator.manual_seed(
                self.batch_sampler.sampler.seed + self.epoch
            )
        idx = 0
        for i, batch in enumerate(self.batch_sampler):
            if i % self.num_replicas == self.rank:
                yield batch
                idx += 1
            if self.drop_last and idx >= self.num_samples:
                break

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        if hasattr(self.batch_sampler.sampler, "set_epoch"):
            self.batch_sampler.sampler.set_epoch(epoch)
        elif (
            hasattr(self.batch_sampler.sampler, "generator")
            and hasattr(self.batch_sampler.sampler, "seed")
            and self.batch_sampler.sampler.generator is not None
        ):
            self.batch_sampler.sampler.generator.manual_seed(
                self.batch_sampler.sampler.seed + epoch
            )


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
    os.environ["HCCL_DETERMINISTIC"] = "1"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    cfg = TrainConfig()
    for field in cfg.__dataclass_fields__.values():
        name = field.name.lower()
        default = getattr(cfg, field.name)
        t = type(default)
        if t is bool:
            parser.add_argument(
                f"--{name}",
                action="store_true" if not default else "store_false",
                help=f"(default: {default})",
            )
        else:
            parser.add_argument(
                f"--{name}", type=t, default=default, help=f"(default: {default})"
            )
    args = parser.parse_args()
    cfg = TrainConfig(
        **{
            f.name: getattr(args, f.name.lower())
            for f in cfg.__dataclass_fields__.values()
        }
    )
    world_size = dist.get_world_size()
    assert cfg.num_generations in [
        n_gen
        for n_gen in range(2, (world_size * cfg.batch_size) + 1)
        if (world_size * cfg.batch_size) % n_gen == 0
    ]
    cfg.dtype = getattr(torch, cfg.dtype)
    if cfg.gradient_checkpoint:
        cfg.use_cache = False
    if cfg.use_fsdp and world_size == 1:
        raise Exception("FSDP should not be used with just one GPU")
    if cfg.fsdp_bf16 and cfg.use_fsdp:
        cfg.bf16 = True
        if cfg.dtype == torch.bfloat16:
            cfg.dtype = torch.float32
    if cfg.collate_fn is None:
        cfg.collate_fn = lambda batch: batch
    return cfg

def unsloth_parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    cfg = TrainConfig()
    for field in cfg.__dataclass_fields__.values():
        name = field.name.lower()
        default = getattr(cfg, field.name)
        t = type(default)
        if t is bool:
            parser.add_argument(
                f"--{name}",
                action="store_true" if not default else "store_false",
                help=f"(default: {default})",
            )
        else:
            parser.add_argument(
                f"--{name}", type=t, default=default, help=f"(default: {default})"
            )
    args = parser.parse_args()
    cfg = TrainConfig(
        **{
            f.name: getattr(args, f.name.lower())
            for f in cfg.__dataclass_fields__.values()
        }
    )
    assert cfg.num_generations in [
        n_gen
        for n_gen in range(2, (cfg.batch_size) + 1)
        if (cfg.batch_size) % n_gen == 0
    ]
    cfg.dtype = getattr(torch, cfg.dtype)
    if cfg.gradient_checkpoint:
        cfg.use_cache = False
    if cfg.collate_fn is None:
        cfg.collate_fn = lambda batch: batch
    return cfg