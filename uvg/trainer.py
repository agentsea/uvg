import inspect
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any
import sys

import torch
import torch.nn.functional as F
import unsloth  # noqa
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION
from transformers.trainer_utils import SchedulerType
from unsloth import FastVisionModel

from .config import Config
from .utils import (
    RepeatSampler,
    accepts_kwarg,
    init_wandb,
    log_completions,
    log_wandb,
    nanmax,
    nanmin,
    save_checkpoint,
    validate_cfg,
)


def score_completions(
    prompts: list[str],
    completions: list[str],
    completion_ids_list: list[list[int]],
    decoded_completion_ids_list: list[list[str]],
    completion_probs_list: list[list[float]],
    reward_funcs: list[Callable[..., list[float]]],
    cfg: Config,
    **reward_kwargs,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    output_reward_func = [
        torch.tensor(
            reward(
                prompts=prompts,
                completions=completions,
                completion_ids=completion_ids_list,
                decoded_completion_ids=decoded_completion_ids_list,
                completion_probs=completion_probs_list,
                **reward_kwargs,
            ),
            dtype=torch.float32,
            device="cuda",
        )
        for reward in reward_funcs
    ]
    rewards_per_func = torch.stack(output_reward_func, dim=1)
    rewards = rewards_per_func.nansum(dim=1)
    mean_grouped_rewards = rewards.view(-1, cfg.num_generations).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, cfg.num_generations).std(dim=1)
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
        cfg.num_generations, dim=0
    )
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(
        cfg.num_generations, dim=0
    )
    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
    return advantages, rewards, rewards_per_func, std_grouped_rewards


def get_log_probs(
    model: Any,
    input_ids: Tensor,
    attention_mask: Tensor,
    logits_to_keep: int,
    cfg: Config,
    maybe_cast_to_f32: bool = True,
    **model_kwargs,
) -> Tensor:
    forward_model = model.module if hasattr(model, "module") else model
    forward = (
        forward_model.get_base_model().forward
        if hasattr(forward_model, "get_base_model")
        else forward_model.forward
    )
    if accepts_kwarg(forward, "logits_to_keep"):
        model_kwargs["logits_to_keep"] = logits_to_keep + 1
    logits = model(
        input_ids=input_ids, attention_mask=attention_mask, **model_kwargs
    ).logits
    if cfg.bf16 and maybe_cast_to_f32:
        logits = logits.float()
    logits = logits[:, :-1, :]
    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:]
    logits = logits / cfg.temperature
    index = input_ids
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(
            logits, dim=-1, index=index.unsqueeze(-1)
        ).squeeze(-1)
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values
    else:
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):
            row_logps = F.log_softmax(
                row_logits,
                dim=-1,
                dtype=torch.bfloat16 if cfg.bf16 and not maybe_cast_to_f32 else None,
            )
            row_per_token_logps = row_logps.gather(
                dim=-1, index=row_labels.unsqueeze(-1)
            ).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def prepare_inputs(
    batch: list[dict[str, Any]],
    policy_model: Any,
    processor: Any,
    reward_funcs: list[Callable[..., list[float]]],
    metrics: defaultdict[str, list[float]],
    cfg: Config,
) -> tuple[dict[str, Tensor], defaultdict[str, list[float]]]:
    mode = "train" if policy_model.training else "eval"
    prompts = [x["prompt"] for x in batch]
    images = [x["images"] for x in batch if "images" in x]
    if cfg.no_apply_chat_template:
        prompts_text = prompts
    else:
        prompts_text = [
            processor.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            for prompt in prompts
        ]
    prompt_inputs = processor(
        text=prompts_text.copy(),
        images=images,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    ).to("cuda")
    prompt_ids, prompt_mask = (
        prompt_inputs["input_ids"],
        prompt_inputs["attention_mask"],
    )
    remaining_prompt_inputs = {
        k: v
        for k, v in prompt_inputs.items()
        if k not in ["input_ids", "attention_mask"]
    }
    pad_token_id = (
        processor.tokenizer.pad_token_id
        if images is not None
        else processor.pad_token_id
    )
    eos_token_id = (
        processor.tokenizer.eos_token_id
        if images is not None
        else processor.eos_token_id
    )
    bos_token_id = (
        processor.tokenizer.bos_token_id
        if images is not None
        else processor.bos_token_id
    )
    if mode == "train":
        FastVisionModel.for_inference(policy_model)
    outputs = policy_model.generate(
        prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=cfg.max_completion_len,
        do_sample=True,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        min_p=cfg.min_p,
        repetition_penalty=cfg.repetition_penalty,
        cache_implementation=None,
        output_scores=True,
        return_dict_in_generate=True,
        **remaining_prompt_inputs,
    )
    if mode == "train": # return to train mode
        FastVisionModel.for_training(policy_model)
    prompt_length = prompt_ids.size(1)
    prompt_completion_ids = outputs.sequences
    prompt_ids = prompt_completion_ids[:, :prompt_length]
    completion_ids = prompt_completion_ids[:, prompt_length:]
    is_eos = completion_ids == eos_token_id
    eos_idx = torch.full(
        (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device="cuda"
    )
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device="cuda").expand(
        is_eos.size(0), -1
    )
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
    completion_probs_list = []
    if cfg.compute_completion_probs:
        with torch.no_grad():
            prompt_completion_scores = outputs.scores
            for i in range(completion_ids.size(0)):
                example_completion_ids = completion_ids[i]
                example_mask = completion_mask[i]
                valid_length = example_mask.sum().item()
                if valid_length == 0:
                    completion_probs_list.append([])
                    continue
                valid_tokens = example_completion_ids[:valid_length]
                example_probs = []
                for token_idx in range(valid_length):
                    if token_idx < len(prompt_completion_scores):
                        token_logits = prompt_completion_scores[token_idx][i:i+1]  # [1, vocab_size]
                        token_logits = token_logits / cfg.temperature
                        token_id = valid_tokens[token_idx].item()
                        log_probs = F.log_softmax(token_logits, dim=-1, dtype=torch.float32)
                        token_prob = torch.exp(log_probs[0, token_id]).item()
                        example_probs.append(token_prob)
                completion_probs_list.append(example_probs)
            del prompt_completion_scores
    completion_ids_list = [
        [id.item() for id, m in zip(row, mask_row) if m]
        for row, mask_row in zip(completion_ids, completion_mask)
    ]
    decoded_completion_ids_list = [
        [processor.tokenizer.decode([token_id]) for token_id in seq_ids]
        for seq_ids in completion_ids_list
    ]
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    completion_texts = processor.batch_decode(completion_ids, skip_special_tokens=True)
    if cfg.no_apply_chat_template:
        completions = completion_texts
    else:
        completions = []
        for prompt, completion in zip(prompts, completion_texts):
            bootstrap = (
                prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
            )
            completions.append(
                [{"role": "assistant", "content": bootstrap + completion}]
            )
    keys = [
        key for key in batch[0] if key not in ["prompt", "completion", "completion_ids"]
    ]
    reward_kwargs = {key: [example[key] for example in batch] for key in keys}
    advantages, rewards, rewards_per_func, std_grouped_rewards = score_completions(
        prompts, completions, completion_ids_list, decoded_completion_ids_list, completion_probs_list, reward_funcs, cfg, **reward_kwargs
    )
    all_advantages = advantages.clone()
    metrics["num_tokens"] = [
        attention_mask.sum().sum().item()
        + (metrics["num_tokens"][0] if metrics["num_tokens"] else 0)
    ]
    agg_completion_mask = (completion_mask.sum(1)).tolist()
    metrics["completions/mean_length"].append(
        sum(agg_completion_mask) / len(agg_completion_mask)
    )
    metrics["completions/min_length"].append(min(agg_completion_mask))
    metrics["completions/max_length"].append(max(agg_completion_mask))
    for i, reward_func in enumerate(reward_funcs):
        mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
        metrics[f"rewards/{reward_func.__name__}"].append(mean_rewards)
    metrics["reward"].append(rewards.mean().item())
    metrics["reward_std"].append(std_grouped_rewards.mean().item())
    if cfg.log_completions and mode == "train":
        reward_func_logs = defaultdict(list)
        for i, reward_func in enumerate(reward_funcs):
            reward_func_logs[reward_func.__name__].extend(
                rewards_per_func[:, i].tolist()
            )
        log_completions(
            prompts_text,
            completion_texts,
            reward_func_logs,
            all_advantages,
        )
    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "advantages": advantages,
        **remaining_prompt_inputs,
    }, metrics


def compute_loss(
    policy_model: Any,
    inputs: dict[str, Tensor],
    metrics: defaultdict[str, list[float]],
    cfg: Config,
) -> tuple[Tensor, defaultdict[str, list[float]]]:
    prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
    completion_ids, completion_mask = (
        inputs["completion_ids"],
        inputs["completion_mask"],
    )
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)
    model_kwarg_keys = (
        inspect.signature(policy_model.forward).parameters.keys()
        if not hasattr(policy_model, "get_base_model")
        else inspect.signature(policy_model.get_base_model().forward).parameters.keys()
    )
    remaining_kwargs = {k: inputs[k] for k in model_kwarg_keys if k in inputs}
    per_token_logps = get_log_probs(
        policy_model,
        input_ids,
        attention_mask,
        logits_to_keep,
        cfg,
        maybe_cast_to_f32=True,
        **remaining_kwargs,
    )
    with torch.no_grad():
        with policy_model.disable_adapter():
            ref_per_token_logps = get_log_probs(
                policy_model,
                input_ids,
                attention_mask,
                logits_to_keep,
                cfg,
                maybe_cast_to_f32=True,
                **remaining_kwargs,
            )
    per_token_kl = (
        torch.exp(ref_per_token_logps - per_token_logps)
        - (ref_per_token_logps - per_token_logps)
        - 1
    )
    advantages = inputs["advantages"]
    old_per_token_logps = per_token_logps.detach()
    coef_1 = torch.exp(per_token_logps - old_per_token_logps)
    coef_2 = torch.clamp(coef_1, 1 - cfg.epsilon, 1 + cfg.epsilon_high)
    per_token_loss1 = coef_1 * advantages.unsqueeze(1)
    per_token_loss2 = coef_2 * advantages.unsqueeze(1)
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
    per_token_loss = per_token_loss + cfg.beta * per_token_kl
    loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(
        min=1.0
    )
    metrics["kl"].append(
        ((per_token_kl * completion_mask).sum() / completion_mask.sum())
        .nanmean()
        .item()
    )
    is_low_clipped = (coef_1 < 1 - cfg.epsilon) & (advantages.unsqueeze(1) < 0)
    is_high_clipped = (coef_1 > 1 + cfg.epsilon_high) & (advantages.unsqueeze(1) > 0)
    is_region_clipped = is_low_clipped | is_high_clipped
    low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
    high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
    clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()
    gathered_low_clip = low_clip
    metrics["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
    metrics["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
    gathered_high_clip = high_clip
    metrics["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
    metrics["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
    gathered_clip_ratio = clip_ratio
    metrics["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
    return loss, metrics


def init_dataloader(dataset, cfg: Config) -> DataLoader:
    sampler = RepeatSampler(
        data_source=dataset,
        mini_repeat_count=cfg.num_generations,
        seed=cfg.seed,
    )
    batch_sampler = BatchSampler(
        sampler=sampler,
        batch_size=cfg.num_generations,
        drop_last=False,
    )
    return DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=cfg.collate_fn,
        pin_memory=True,
    )


def init_models(cfg: Config) -> tuple[PreTrainedModel, Any]:
    policy_model, processor = FastVisionModel.from_pretrained(
        cfg.model_id,
        dtype=cfg.dtype,
        use_cache=cfg.use_cache,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
    )  # TODO: check padding side
    policy_model = FastVisionModel.get_peft_model(
        policy_model,
        lora_alpha=cfg.lora_alpha,
        r=cfg.lora_rank,
        target_modules=cfg.lora_target_modules,
        random_state=3407,
    )
    policy_model.print_trainable_parameters()
    policy_model.to("cuda")
    policy_model.train()
    return policy_model, processor


def evaluate(
    policy_model: Any,
    processor: Any,
    eval_dataloader: DataLoader,
    reward_funcs: list[Callable[..., list[float]]],
    cfg: Config,
) -> defaultdict[str, list[float]]:
    FastVisionModel.for_inference(policy_model)
    eval_metrics = defaultdict(list)
    total_batches = len(eval_dataloader)
    for i, batch in enumerate(eval_dataloader):
        with (
            torch.no_grad(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if cfg.bf16
            else nullcontext(),
        ):
            _, eval_metrics = prepare_inputs(
                batch,
                policy_model,
                processor,
                reward_funcs,
                eval_metrics,
                cfg,
            )
        sys.stdout.write(f"\rEvaluating batch {i + 1}/{total_batches}...")
        sys.stdout.flush()
    sys.stdout.write("\n")
    FastVisionModel.for_training(policy_model)
    return eval_metrics


def train(
    cfg: Config,
    reward_funcs: list[Callable[..., list[float]]],
    train_dataset: Any,
    eval_dataset: Any | None = None,
) -> None:
    cfg = validate_cfg(cfg)
    metrics = defaultdict(list)
    if cfg.use_wandb:
        init_wandb(cfg.model_id, cfg.wandb_project)
    policy_model, processor = init_models(cfg)
    train_dataloader = init_dataloader(train_dataset, cfg)
    eval_dataloader = init_dataloader(eval_dataset, cfg) if eval_dataset else None
    optimizer = AdamW(
        [p for _, p in policy_model.named_parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    num_training_steps = cfg.num_epochs * len(train_dataloader)
    num_warmup_steps = int(num_training_steps * cfg.warmup_ratio)
    name = SchedulerType(cfg.lr_scheduler)
    scheduler_fn = TYPE_TO_SCHEDULER_FUNCTION[name]
    scheduler = scheduler_fn(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    for epoch in range(cfg.num_epochs):
        for step, batch in enumerate(train_dataloader):
            policy_model.train()
            with (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if cfg.bf16
                else nullcontext()
            ):
                inputs, metrics = prepare_inputs(
                    batch,
                    policy_model,
                    processor,
                    reward_funcs,
                    metrics,
                    cfg,
                )
                loss, metrics = compute_loss(policy_model, inputs, metrics, cfg)
            loss.backward()
            metrics["loss"].append(round(loss.mean().item(), 4))
            grad_norm_to_log = torch.as_tensor(
                clip_grad_norm_(policy_model.parameters(), cfg.grad_norm)
            )
            metrics["grad_norm"].append(grad_norm_to_log.mean().item())
            metrics["learning_rate"].append(scheduler.get_last_lr()[0])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if step % cfg.log_steps == 0:
                prefix = "train"
                metrics_str = " | ".join(f"{prefix}/{k}: {v[-1]}" for k, v in metrics.items())
                print(f"epoch {epoch} | step: {step + 1} | {metrics_str}")
                if cfg.use_wandb:
                    log_wandb(metrics, step + 1, prefix)
            if eval_dataloader and (step + 1) % cfg.eval_steps == 0:
                print(f"\nRunning eval at step {step+1}...")
                eval_metrics = evaluate(
                    policy_model,
                    processor,
                    eval_dataloader,
                    reward_funcs,
                    cfg,
                )
                prefix = "eval"
                eval_metrics_str = " | ".join(
                    f"{prefix}/{k}: {v[-1]}" for k, v in eval_metrics.items()
                )
                print(f"epoch {epoch} | step: {step + 1} | {eval_metrics_str}")
                if cfg.use_wandb:
                    log_wandb(eval_metrics, step + 1, prefix)
                print("\nFinished eval...")
            if (step + 1) % cfg.save_steps == 0 or (step + 1) == len(train_dataloader):
                save_checkpoint(
                    model=policy_model,
                    processor=processor,
                    push_to_hub=cfg.push_to_hub,
                    hub_repo_id=cfg.hub_repo_id,
                    hub_private=cfg.hub_private,
                    commit_msg=f"checkpoint at step {step + 1}"
                    if (step + 1) % cfg.save_steps == 0
                    else "final checkpoint",
                )
