import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from common.thyroid_data import load_labels
from common.vlm_sft import (
    build_dataset_pair,
    build_training_args,
    load_model_with_lora,
    parse_target_modules,
    qwen3_collate_fn,
    save_processor,
    ThyroidTrainer,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--image_dir", type=str, required=True)
    ap.add_argument("--train_json", type=str, required=True)
    ap.add_argument("--test_json", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--max_seq_length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--use_qlora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--eval_strategy", type=str, default="epoch")
    ap.add_argument("--save_strategy", type=str, default="epoch")
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--load_best_model_at_end", action="store_true")
    ap.add_argument("--attn_impl", type=str, default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager"])
    return ap.parse_args()


def main():
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.model_dir, use_fast=False)
    train_records = load_labels(args.train_json)
    test_records = load_labels(args.test_json)
    train_dataset, test_dataset = build_dataset_pair(train_records, test_records, args.image_dir, answer_with_space=True)

    attn_impl = None if args.attn_impl == "auto" else args.attn_impl
    model = load_model_with_lora(
        model_cls=Qwen3VLForConditionalGeneration,
        model_dir=args.model_dir,
        dtype=args.dtype,
        use_qlora=args.use_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=parse_target_modules(args.target_modules, ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        gradient_checkpointing=args.gradient_checkpointing,
        attn_implementation=attn_impl,
    )
    model.print_trainable_parameters()

    trainer = ThyroidTrainer(
        model=model,
        args=build_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=lambda batch: qwen3_collate_fn(processor, batch),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    save_processor(processor, args.output_dir)
    print(f"Saved Qwen3-VL adapter to: {args.output_dir}")


if __name__ == "__main__":
    main()
