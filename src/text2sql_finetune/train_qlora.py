"""Train a LoRA adapter with SQL-completion loss and auditable data filtering."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .sft_encoding import encode_sql_completion


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SFTDataset:
    """JSONL dataset; discard over-budget examples rather than truncate SQL."""

    def __init__(self, path: str | Path, tokenizer: Any, max_length: int):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Training/validation JSONL is missing: {self.path}")
        self.samples: list[dict[str, list[int]]] = []
        self.filtered_sample_ids: list[str] = []
        self.db_ids: set[str] = set()
        self.sample_ids: set[str] = set()
        retained_db_ids: set[str] = set()
        raw_count = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("db_id"), str):
                    raise ValueError(f"Missing db_id at {self.path}:{line_number}; regenerate prepared data")
                if not row["db_id"]:
                    raise ValueError(f"Empty db_id at {self.path}:{line_number}")
                sample_id = str(row.get("id", f"line-{line_number}"))
                if sample_id in self.sample_ids:
                    raise ValueError(f"Duplicate sample id {sample_id!r} in {self.path}")
                self.sample_ids.add(sample_id)
                self.db_ids.add(row["db_id"])
                raw_count += 1
                try:
                    sample = encode_sql_completion(tokenizer, row["prompt"], row["completion"], max_length)
                except (KeyError, ValueError) as error:
                    raise ValueError(f"Invalid SFT example at {self.path}:{line_number}: {error}") from error
                if sample is None:
                    self.filtered_sample_ids.append(sample_id)
                    continue
                retained_db_ids.add(row["db_id"])
                self.samples.append(sample)
        if not self.samples:
            raise ValueError(f"No usable examples remain in {self.path}; check max_seq_length/data")
        self.stats = {
            "path": str(self.path.resolve()),
            "sha256": file_sha256(self.path),
            "input_sample_count": raw_count,
            "retained_sample_count": len(self.samples),
            "filtered_over_length_count": len(self.filtered_sample_ids),
            "filtered_sample_ids": self.filtered_sample_ids,
            "max_seq_length": max_length,
            "input_db_ids": sorted(self.db_ids),
            "retained_db_ids": sorted(retained_db_ids),
            "minimum_supervised_tokens_including_eos": min(
                sum(token != -100 for token in sample["labels"]) for sample in self.samples
            ),
            "maximum_retained_sequence_length": max(len(sample["input_ids"]) for sample in self.samples),
            "length_policy": "discard whole example; never truncate prompt or SQL completion",
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.samples[index]


class DataCollatorForCausalLM:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        if not features:
            raise ValueError("Cannot collate an empty batch")
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id")
        batch_length = max(len(feature["input_ids"]) for feature in features)
        batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = batch_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.tokenizer.pad_token_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def validate_training_config(config: dict[str, Any]) -> tuple[int, int]:
    eval_steps = int(config.get("eval_steps", 200))
    save_steps = int(config.get("save_steps", 200))
    if eval_steps < 1 or save_steps < 1 or eval_steps != save_steps:
        raise ValueError("eval_steps and save_steps must be equal positive integers for best-model selection")
    if int(config.get("max_seq_length", 2048)) < 1:
        raise ValueError("max_seq_length must be positive")
    return eval_steps, save_steps


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(f"Config file is missing: {args.config}")

    import torch
    import yaml
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers import Trainer, TrainingArguments, set_seed

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Training config must be a YAML mapping")
    for key in ("model_name_or_path", "train_file", "validation_file", "output_dir"):
        if key not in config:
            raise ValueError(f"Required config field is missing: {key}")
    eval_steps, save_steps = validate_training_config(config)
    dtype_name = config.get("compute_dtype", "float16")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported compute_dtype: {dtype_name}")
    compute_dtype = dtype_map[dtype_name]
    use_4bit = bool(config.get("use_4bit", True))
    if use_4bit and not torch.cuda.is_available():
        raise RuntimeError("This QLoRA configuration requires a CUDA GPU; use_4bit=false for a CPU fixture only")
    seed = int(config.get("seed", 42))
    set_seed(seed)
    if config.get("wandb_project"):
        os.environ["WANDB_PROJECT"] = str(config["wandb_project"])
    output_dir = Path(config["output_dir"])
    if output_dir.is_dir() and any(output_dir.iterdir()) and not config.get("resume_from_checkpoint"):
        raise FileExistsError(
            f"Output directory is nonempty: {output_dir}. Choose a new run directory "
            "or explicitly resume a saved checkpoint."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = config["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.eos_token_id is None:
        raise ValueError("Base tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    max_length = int(config.get("max_seq_length", 2048))
    train_dataset = SFTDataset(config["train_file"], tokenizer, max_length)
    valid_dataset = SFTDataset(config["validation_file"], tokenizer, max_length)
    overlap = train_dataset.db_ids & valid_dataset.db_ids
    if overlap:
        raise ValueError(f"Training/validation databases overlap: {sorted(overlap)}")
    if train_dataset.sample_ids & valid_dataset.sample_ids:
        raise ValueError("Training/validation sample IDs overlap")

    packages = {}
    for package in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "PyYAML"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    runtime = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "config_path": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config),
        "config": config,
        "selection_protocol": "best checkpoint chosen by validation loss only; no test-set access",
    }
    local_rank = runtime["local_rank"]
    if use_4bit:
        torch.cuda.set_device(local_rank)
    # In DDP every process owns one quantized model on its assigned GPU.
    # device_map='auto' would spread one training model over several devices.
    model_kwargs: dict[str, Any] = {"torch_dtype": compute_dtype}
    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(config.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs["device_map"] = {"": local_rank}
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    gradient_checkpointing = bool(config.get("gradient_checkpointing", True))
    if use_4bit:
        preparation_kwargs: dict[str, Any] = {"use_gradient_checkpointing": gradient_checkpointing}
        if "gradient_checkpointing_kwargs" in inspect.signature(prepare_model_for_kbit_training).parameters:
            preparation_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
        model = prepare_model_for_kbit_training(model, **preparation_kwargs)
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=int(config.get("lora_r", 64)),
        lora_alpha=int(config.get("lora_alpha", 128)),
        target_modules=config.get("lora_target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ]),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    runtime["trainable_parameters"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    runtime["total_parameters"] = sum(parameter.numel() for parameter in model.parameters())

    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(config.get("num_train_epochs", 3)),
        "max_steps": int(config.get("max_steps", -1)),
        "per_device_train_batch_size": int(config.get("per_device_train_batch_size", 2)),
        "per_device_eval_batch_size": int(config.get("per_device_eval_batch_size", 2)),
        "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 8)),
        "learning_rate": float(config.get("learning_rate", 2e-4)),
        "warmup_ratio": float(config.get("warmup_ratio", 0.03)),
        "weight_decay": float(config.get("weight_decay", 0)),
        "lr_scheduler_type": config.get("lr_scheduler_type", "cosine"),
        "logging_steps": int(config.get("logging_steps", 10)),
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "save_strategy": "steps",
        "save_total_limit": int(config.get("save_total_limit", 3)),
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "fp16": dtype_name == "float16",
        "bf16": dtype_name == "bfloat16",
        "optim": config.get("optim", "paged_adamw_8bit" if use_4bit else "adamw_torch"),
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
        "gradient_checkpointing": gradient_checkpointing,
        "report_to": config.get("report_to", "none"),
        "seed": seed,
        "data_seed": seed,
    }
    argument_parameters = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = "eval_strategy" if "eval_strategy" in argument_parameters else "evaluation_strategy"
    training_kwargs[strategy_key] = "steps"
    if "gradient_checkpointing_kwargs" in argument_parameters:
        training_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    training_args = TrainingArguments(**training_kwargs)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": valid_dataset,
        "data_collator": DataCollatorForCausalLM(tokenizer),
    }
    tokenizer_key = "processing_class" if "processing_class" in inspect.signature(Trainer.__init__).parameters else "tokenizer"
    trainer_kwargs[tokenizer_key] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    if trainer.is_world_process_zero():
        write_json(output_dir / "training_runtime.json", runtime)
        write_json(output_dir / "training_data_audit.json", {
            "train": train_dataset.stats, "validation": valid_dataset.stats,
            "database_disjoint": True,
        })
    training_result = trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    trainer.log_metrics("train", training_result.metrics)
    trainer.save_metrics("train", training_result.metrics)
    trainer.save_state()
    validation_metrics = trainer.evaluate()
    trainer.log_metrics("eval", validation_metrics)
    trainer.save_metrics("eval", validation_metrics)
    trainer.save_model(str(output_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)
        write_json(output_dir / "training_summary.json", {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "train_metrics": training_result.metrics,
            "validation_metrics": validation_metrics,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_validation_loss": trainer.state.best_metric,
            "global_step": trainer.state.global_step,
            "test_set_used": False,
        })


if __name__ == "__main__":
    main()
