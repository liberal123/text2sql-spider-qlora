from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from text2sql_finetune.formatting import build_prompt, build_repair_prompt, extract_sql
from text2sql_finetune.schema_utils import load_spider_tables, spider_schema_to_context


def load_model(base_model: str, adapter_path: str | None = None, load_4bit: bool = True) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if load_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=False,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_from_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=False)
    return extract_sql(generated)


def generate_sql(
    model: Any,
    tokenizer: Any,
    question: str,
    schema_context: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    prompt = build_prompt(question, schema_context)
    return generate_from_prompt(model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=temperature)


def repair_sql(
    model: Any,
    tokenizer: Any,
    question: str,
    schema_context: str,
    bad_sql: str,
    error: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    prompt = build_repair_prompt(question, schema_context, bad_sql, error)
    return generate_from_prompt(model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=temperature)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--tables", default="data/raw/spider/tables.json")
    parser.add_argument("--db-id", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    schemas = load_spider_tables(args.tables)
    schema_context = spider_schema_to_context(schemas[args.db_id])
    model, tokenizer = load_model(args.base_model, args.adapter_path, load_4bit=not args.no_4bit)
    sql = generate_sql(model, tokenizer, args.question, schema_context)
    print(json.dumps({"question": args.question, "sql": sql}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
