import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, Trainer, TrainingArguments

from common.thyroid_data import (
    ThyroidBinaryDataset,
    find_filename_overlap,
    print_dataset_report,
    resolve_image_path,
    safe_open_image,
)
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


@torch.inference_mode()
def score_string_logprob(model, tokenizer, base_inputs: Dict[str, torch.Tensor], target_text: str) -> float:
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    if not target_ids:
        raise ValueError(f"target_text tokenized to empty sequence: {target_text!r}")

    out = model(**base_inputs, use_cache=True)
    next_logits = out.logits[:, -1, :]
    past = out.past_key_values
    logprob = 0.0

    for token_id in target_ids:
        step_logprob = torch.log_softmax(next_logits, dim=-1)[0, token_id].item()
        logprob += step_logprob
        input_ids = torch.tensor([[token_id]], device=next_logits.device, dtype=torch.long)
        out = model(input_ids=input_ids, past_key_values=past, use_cache=True)
        next_logits = out.logits[:, -1, :]
        past = out.past_key_values

    return float(logprob)


def build_qwen3_vl_inputs(processor, image: Image.Image) -> Dict[str, torch.Tensor]:
    messages = build_qwen3_messages(image, None)
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


@torch.inference_mode()
def prob_malignant_from_qwen3_logits(model, processor, image: Image.Image) -> float:
    tokenizer = processor.tokenizer
    device = model.device
    base_inputs = build_qwen3_vl_inputs(processor, image)
    base_inputs = {k: v.to(device) for k, v in base_inputs.items()}

    candidate_pairs = [(" 0", " 1"), ("0", "1"), ("\n0", "\n1"), (" 0\n", " 1\n")]
    last_err = None
    for c0, c1 in candidate_pairs:
        try:
            lp0 = score_string_logprob(model, tokenizer, base_inputs, c0)
            lp1 = score_string_logprob(model, tokenizer, base_inputs, c1)
            m = max(lp0, lp1)
            p0 = torch.exp(torch.tensor(lp0 - m)).item()
            p1 = torch.exp(torch.tensor(lp1 - m)).item()
            return float(p1 / (p0 + p1 + 1e-12))
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"All Qwen3 candidate pairs failed. Last error: {last_err}")


@torch.inference_mode()
def prob_malignant_from_medgemma_logits(model, processor, image: Image.Image, prefer_leading_space: bool = True) -> float:
    tokenizer = processor.tokenizer
    device = model.device
    base_inputs = processor(images=image, text=MEDGEMMA_PROMPT, return_tensors="pt")
    base_inputs = {k: v.to(device) for k, v in base_inputs.items()}
    c0 = " 0" if prefer_leading_space else "0"
    c1 = " 1" if prefer_leading_space else "1"
    lp0 = score_string_logprob(model, tokenizer, base_inputs, c0)
    lp1 = score_string_logprob(model, tokenizer, base_inputs, c1)
    m = max(lp0, lp1)
    p0 = torch.exp(torch.tensor(lp0 - m)).item()
    p1 = torch.exp(torch.tensor(lp1 - m)).item()
    return float(p1 / (p0 + p1 + 1e-12))


def build_prediction_callback(
    records,
    image_dir: str,
    processor,
    probability_fn: Callable[[Any, Any, Image.Image], float],
    threshold: float = 0.5,
):
    def callback(model) -> Dict[str, float]:
        labels: List[int] = []
        probabilities: List[float] = []
        for record in records:
            filename = str(record["filename"])
            image_path = resolve_image_path(image_dir, filename)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Missing evaluation image: {image_path}")
            image = safe_open_image(image_path)
            try:
                probability = probability_fn(model, processor, image)
            finally:
                image.close()
            labels.append(int(record["malignancy"]))
            probabilities.append(float(probability))
        return summarize_predictions(labels, probabilities, threshold=threshold)

    return callback


def save_processor(processor, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    processor.save_pretrained(output_dir)
