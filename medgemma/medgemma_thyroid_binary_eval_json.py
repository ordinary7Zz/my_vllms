import os
import sys
import json
import argparse
from typing import List, Dict, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, AutoModelForImageTextToText
from common.thyroid_metrics import compute_metrics, bootstrap_metric_cis
from common.thyroid_prompts import MEDGEMMA_PROMPT
from common.vlm_sft import load_peft_adapter_if_needed


def load_labels(label_json: str, label_key: str = "malignancy") -> List[Dict]:
    with open(label_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Label JSON must be a list of records.")
    normalized = []
    for r in data:
        if "filename" not in r or label_key not in r:
            raise ValueError(f"Each record must contain keys: filename, {label_key}.")
        malignancy = int(r[label_key])
        if malignancy not in (0, 1):
            raise ValueError(f"{label_key} must be 0/1, got {malignancy}")
        normalized.append({"filename": str(r["filename"]), "malignancy": malignancy})
    return normalized


def safe_open_image(path: str) -> Image.Image:
    img = Image.open(path)
    return img.convert("RGB")


@torch.inference_mode()
def score_string_logprob(
    model: AutoModelForImageTextToText,
    tokenizer,
    base_inputs: Dict[str, torch.Tensor],
    target_text: str,
) -> float:
    tgt_ids = tokenizer.encode(target_text, add_special_tokens=False)
    if len(tgt_ids) == 0:
        raise ValueError("target_text tokenized to empty sequence.")

    out = model(**base_inputs, use_cache=True)
    logits = out.logits
    past = out.past_key_values

    next_logits = logits[:, -1, :]
    logprob = 0.0

    for tid in tgt_ids:
        step_logprob = torch.log_softmax(next_logits, dim=-1)[0, tid].item()
        logprob += step_logprob

        input_ids = torch.tensor([[tid]], device=next_logits.device, dtype=torch.long)
        out2 = model(input_ids=input_ids, past_key_values=past, use_cache=True)
        next_logits = out2.logits[:, -1, :]
        past = out2.past_key_values

    return float(logprob)


def prob_malignant_from_logits(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    img: Image.Image,
    prompt: str,
    prefer_leading_space: bool = True,
) -> Tuple[float, Dict[str, float]]:
    tokenizer = processor.tokenizer
    device = model.device

    inputs = processor(images=img, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    c0 = " 0" if prefer_leading_space else "0"
    c1 = " 1" if prefer_leading_space else "1"

    lp0 = score_string_logprob(model, tokenizer, inputs, c0)
    lp1 = score_string_logprob(model, tokenizer, inputs, c1)

    m = max(lp0, lp1)
    p0 = torch.exp(torch.tensor(lp0 - m)).item()
    p1 = torch.exp(torch.tensor(lp1 - m)).item()
    p1_norm = p1 / (p0 + p1 + 1e-12)

    return float(p1_norm), {"logp_0": lp0, "logp_1": lp1, "cand0": c0, "cand1": c1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True,
                    help="Local path to medgemma model dir, e.g. /mnt/wangbd8/workspace/medgemma-4b-it")
    ap.add_argument("--adapter_dir", type=str, default=None,
                    help="Optional LoRA adapter directory saved by the training script")
    ap.add_argument("--image_dir", type=str, required=True,
                    help="Directory containing images referenced by filename in label json")
    ap.add_argument("--label_json", type=str, required=True,
                    help="Label json path, e.g. /mnt/data/tn3k_test_label.json")
    ap.add_argument("--label_key", type=str, default="malignancy",
                    help="Label field name in label json, default: malignancy")
    ap.add_argument("--filename", type=str, default="medgemma_preds",
                    help="Base filename for outputs, without extension")
    ap.add_argument("--out_path", type=str, default=".",
                    help="Directory to save evaluation outputs")
    ap.add_argument("--use_fast_processor", action="store_true",
                    help="Use fast processor (default: slow for stability)")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"],
                    help="Computation dtype on GPU")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Decision threshold on P(malignant)")
    ap.add_argument("--limit", type=int, default=-1,
                    help="Run only first N samples for quick test (-1 for all)")
    ap.add_argument("--ci_bootstrap", type=int, default=2000,
                    help="Number of bootstrap resamples for confidence intervals")
    ap.add_argument("--ci_alpha", type=float, default=0.95,
                    help="Confidence level for intervals, e.g. 0.95")
    ap.add_argument("--ci_seed", type=int, default=42,
                    help="Random seed for bootstrap confidence intervals")
    args = ap.parse_args()

    out_json = os.path.join(args.out_path, f"{args.filename}.json")
    out_metrics = os.path.join(args.out_path, f"{args.filename}.txt")
    os.makedirs(args.out_path, exist_ok=True)

    labels = load_labels(args.label_json, label_key=args.label_key)

    if args.limit is not None and args.limit > 0:
        labels = labels[: args.limit]

    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if args.adapter_dir and os.path.exists(os.path.join(args.adapter_dir, "preprocessor_config.json")):
        processor_source = args.adapter_dir
    else:
        processor_source = args.model_dir
    processor = AutoProcessor.from_pretrained(processor_source, use_fast=args.use_fast_processor)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir,
        torch_dtype=torch_dtype,
        device_map="cuda",
    )
    model = load_peft_adapter_if_needed(model, args.adapter_dir)
    model.eval()

    prompt = MEDGEMMA_PROMPT

    y_true: List[int] = []
    y_prob: List[float] = []
    y_pred: List[int] = []
    results = []

    print(f"Model dir:         {args.model_dir}")
    print(f"Adapter dir:       {args.adapter_dir or 'none'}")
    print(f"Processor source:  {processor_source}")
    print(f"Output dir:        {args.out_path}")
    print(f"Output filename:   {args.filename}")
    print(f"Predictions JSON:  {out_json}")
    print(f"Metrics TXT:       {out_metrics}")

    missing_files = 0
    bad_images = 0

    for r in tqdm(labels, desc="Infer"):
        fn = r["filename"]
        gt = int(r["malignancy"])

        img_path = os.path.join(args.image_dir, fn)
        if not os.path.exists(img_path):
            missing_files += 1
            continue

        try:
            img = safe_open_image(img_path)
        except Exception:
            bad_images += 1
            continue

        try:
            try:
                p1, dbg = prob_malignant_from_logits(
                    model=model,
                    processor=processor,
                    img=img,
                    prompt=prompt,
                    prefer_leading_space=True,
                )
            except Exception:
                p1, dbg = prob_malignant_from_logits(
                    model=model,
                    processor=processor,
                    img=img,
                    prompt=prompt,
                    prefer_leading_space=False,
                )
        finally:
            img.close()

        pred = 1 if p1 >= args.threshold else 0
        p0 = 1.0 - p1

        y_true.append(gt)
        y_prob.append(p1)
        y_pred.append(pred)

        results.append({
            "record_type": "sample",
            "image_file": img_path,
            "image_name": fn,
            "filename": fn,
            "selected_model": "medgemma",
            "predicted_class": pred,
            "confidence": float(max(p0, p1)),
            "prob_class_0": float(p0),
            "prob_class_1": float(p1),
            "true_label": gt,
            "logp_0": float(dbg["logp_0"]),
            "logp_1": float(dbg["logp_1"]),
            "cand0": dbg["cand0"],
            "cand1": dbg["cand1"],
        })

    if len(y_true) == 0:
        raise RuntimeError("No valid samples were evaluated (all missing/bad?). Check paths.")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    metrics = compute_metrics(y_true, y_prob, y_pred)
    cis = bootstrap_metric_cis(
        y_true=y_true,
        y_prob=y_prob,
        threshold=args.threshold,
        n_bootstrap=args.ci_bootstrap,
        alpha=args.ci_alpha,
        seed=args.ci_seed,
    )

    summary_lines = [
        "==== MedGemma Thyroid Binary Classification Metrics (pos=malignant=1) ====",
        f"Evaluated samples: {len(y_true)}",
        f"Missing files:     {missing_files}",
        f"Bad images:        {bad_images}",
        f"AUROC:             {metrics['auroc']:.6f} (95% CI {cis['auroc'][0]:.6f}-{cis['auroc'][1]:.6f})",
        f"AUPRC:             {metrics['auprc']:.6f} (95% CI {cis['auprc'][0]:.6f}-{cis['auprc'][1]:.6f})",
        f"Acc:               {metrics['accuracy']:.6f} (95% CI {cis['accuracy'][0]:.6f}-{cis['accuracy'][1]:.6f})",
        f"F1:                {metrics['f1']:.6f} (95% CI {cis['f1'][0]:.6f}-{cis['f1'][1]:.6f})",
        f"Sensitivity:       {metrics['sensitivity']:.6f} (95% CI {cis['sensitivity'][0]:.6f}-{cis['sensitivity'][1]:.6f})",
        f"Specificity:       {metrics['specificity']:.6f} (95% CI {cis['specificity'][0]:.6f}-{cis['specificity'][1]:.6f})",
        f"Confusion (tn fp fn tp): {metrics['tn']} {metrics['fp']} {metrics['fn']} {metrics['tp']}",
        f"Saved per-image predictions to: {out_json}",
        f"Saved metrics to: {out_metrics}",
    ]
    summary_text = "\n".join(summary_lines)

    print("\n" + summary_text)
    with open(out_metrics, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"Saved metrics to: {out_metrics}")


if __name__ == "__main__":
    main()
