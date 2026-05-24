import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, Trainer, TrainingArguments

from common.thyroid_data import ThyroidBinaryDataset, find_filename_overlap, print_dataset_report
from common.thyroid_metrics import compute_metrics
from common.thyroid_prompts import MEDGEMMA_PROMPT, build_qwen3_messages


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


class ThyroidTrainer(Trainer):
    def __init__(self, *args, prediction_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.prediction_callback = prediction_callback

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        if self.prediction_callback is not None:
            extra_metrics = self.prediction_callback(self.model)
            for key, value in extra_metrics.items():
                metrics[f"{metric_key_prefix}_{key}"] = value
            self.log(metrics)
        return metrics


def parse_target_modules(value: Optional[str], defaults: Sequence[str]) -> List[str]:
    if not value:
        return list(defaults)
    return [item.strip() for item in value.split(",") if item.strip()]


def maybe_build_quant_config(use_qlora: bool) -> Optional[BitsAndBytesConfig]:
    if not use_qlora:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_model_with_lora(
    model_cls,
    model_dir: str,
    dtype: str,
    use_qlora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: Sequence[str],
    gradient_checkpointing: bool,
    attn_implementation: Optional[str] = None,
):
    model_kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": DTYPE_MAP[dtype],
    }
    quantization_config = maybe_build_quant_config(use_qlora)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = model_cls.from_pretrained(model_dir, **model_kwargs)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if use_qlora:
        model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=list(target_modules),
    )
    model = get_peft_model(model, lora_config)
    return model


def load_peft_adapter_if_needed(model, adapter_dir: Optional[str]):
    if not adapter_dir:
        return model
    return PeftModel.from_pretrained(model, adapter_dir)


def medgemma_collate_fn(processor, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    images = [item["image"] for item in batch]
    prefixes = [MEDGEMMA_PROMPT for _ in batch]
    full_texts = [MEDGEMMA_PROMPT + item["answer"] for item in batch]

    full_inputs = processor(images=images, text=full_texts, return_tensors="pt", padding=True)
    prefix_inputs = processor(images=images, text=prefixes, return_tensors="pt", padding=True)

    labels = full_inputs["input_ids"].clone()
    labels[full_inputs["attention_mask"] == 0] = -100
    prefix_lengths = prefix_inputs["attention_mask"].sum(dim=1)
    for i, prefix_len in enumerate(prefix_lengths.tolist()):
        labels[i, :prefix_len] = -100

    full_inputs["labels"] = labels
    return full_inputs


def qwen3_collate_fn(processor, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    messages = [build_qwen3_messages(item["image"], item["answer"]) for item in batch]
    full_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )

    prefix_messages = [build_qwen3_messages(item["image"], None) for item in batch]
    prefix_inputs = processor.apply_chat_template(
        prefix_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )

    labels = full_inputs["input_ids"].clone()
    labels[full_inputs["attention_mask"] == 0] = -100
    prefix_lengths = prefix_inputs["attention_mask"].sum(dim=1)
    for i, prefix_len in enumerate(prefix_lengths.tolist()):
        labels[i, :prefix_len] = -100

    full_inputs["labels"] = labels
    return full_inputs


def build_training_args(args, metric_for_best_model: str = "eval_auroc") -> TrainingArguments:
    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        bf16=args.dtype == "bf16",
        fp16=args.dtype == "fp16",
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
    )


def build_dataset_pair(
    train_records,
    test_records,
    train_image_dir: str,
    test_image_dir: str,
    answer_with_space: bool = True,
):
    print_dataset_report("train", train_records, train_image_dir)
    print_dataset_report("test", test_records, test_image_dir)
    overlap = find_filename_overlap(train_records, test_records)
    if overlap:
        print(f"[warning] train/test filename overlap: {len(overlap)} samples")
    train_dataset = ThyroidBinaryDataset(train_records, image_dir=train_image_dir, answer_with_space=answer_with_space)
    test_dataset = ThyroidBinaryDataset(test_records, image_dir=test_image_dir, answer_with_space=answer_with_space)
    return train_dataset, test_dataset


def score_binary_from_logits(logits: np.ndarray, answer_token_ids: Sequence[int]) -> np.ndarray:
    probs = []
    for row in logits:
        token_logits = row[-1]
        selected = token_logits[list(answer_token_ids)]
        selected = selected - np.max(selected)
        selected = np.exp(selected)
        prob_1 = float(selected[1] / (selected[0] + selected[1] + 1e-12))
        probs.append(prob_1)
    return np.asarray(probs)


def summarize_predictions(labels: Sequence[int], probabilities: Sequence[float], threshold: float) -> Dict[str, float]:
    preds = [1 if p >= threshold else 0 for p in probabilities]
    metrics = compute_metrics(list(labels), list(probabilities), preds)
    return {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in metrics.items()}


def save_processor(processor, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    processor.save_pretrained(output_dir)
