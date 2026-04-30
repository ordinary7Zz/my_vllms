import os
import json
import csv
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, AutoModelForImageTextToText
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ----------------------------
# Utils
# ----------------------------
def load_labels(label_json: str) -> List[Dict]:
    """
    Expected label format (list of dict):
      [{"filename": "...", "malignancy": 0/1}, ...]
    """
    with open(label_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Label JSON must be a list of records.")
    for r in data:
        if "filename" not in r or "malignancy" not in r:
            raise ValueError("Each record must contain keys: filename, malignancy.")
        r["malignancy"] = int(r["malignancy"])
        if r["malignancy"] not in (0, 1):
            raise ValueError(f"malignancy must be 0/1, got {r['malignancy']}")
    return data


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


def compute_metrics(y_true: List[int], y_prob: List[float], y_pred: List[int]) -> Dict[str, float]:
    auroc = roc_auc_score(y_true, y_prob) if len(set(y_true)) == 2 else float("nan")
    auprc = average_precision_score(y_true, y_prob) if len(set(y_true)) == 2 else float("nan")

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "accuracy": float(acc),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_metric_cis(
    y_true: List[int],
    y_prob: List[float],
    threshold: float,
    n_bootstrap: int = 2000,
    alpha: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    n = len(y_true_arr)
    rng = np.random.default_rng(seed)

    metric_samples = {
        "auroc": [],
        "auprc": [],
        "accuracy": [],
        "f1": [],
        "sensitivity": [],
        "specificity": [],
    }

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        sample_y_true = y_true_arr[indices]
        sample_y_prob = y_prob_arr[indices]
        sample_y_pred = (sample_y_prob >= threshold).astype(int)

        metrics = compute_metrics(
            sample_y_true.tolist(),
            sample_y_prob.tolist(),
            sample_y_pred.tolist(),
        )

        for key in metric_samples:
            value = metrics[key]
            if not np.isnan(value):
                metric_samples[key].append(value)

    lower_q = (1.0 - alpha) / 2.0
    upper_q = 1.0 - lower_q
    cis: Dict[str, Tuple[float, float]] = {}
    for key, values in metric_samples.items():
        if len(values) == 0:
            cis[key] = (float("nan"), float("nan"))
        else:
            cis[key] = (
                float(np.quantile(values, lower_q)),
                float(np.quantile(values, upper_q)),
            )
    return cis


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True,
                    help="Local path to Llama-3.2 model dir, e.g. /mnt/models/llama-3.2-vision")
    ap.add_argument("--image_dir", type=str, required=True,
                    help="Directory containing images referenced by filename in label json")
    ap.add_argument("--label_json", type=str, required=True,
                    help="Label json path, e.g. /mnt/data/thyroid_labels.json")
    ap.add_argument("--out_csv", type=str, default="llama_preds.csv",
                    help="Output CSV with per-image predictions")
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

    labels = load_labels(args.label_json)
    if args.limit is not None and args.limit > 0:
        labels = labels[: args.limit]

    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    processor = AutoProcessor.from_pretrained(args.model_dir, use_fast=args.use_fast_processor)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir,
        dtype=torch_dtype,
        device_map="cuda",
    )
    model.eval()

    prompt = (
    "<start_of_image>\n"
    "You are a medical imaging assistant.\n"
    "Task: Thyroid ultrasound nodule malignancy classification.\n"
    "Output exactly one character: 0 or 1.\n"
    "0 = benign, 1 = malignant.\n"
    "Answer:"
    )

    y_true: List[int] = []
    y_prob: List[float] = []
    y_pred: List[int] = []

    missing_files = 0
    bad_images = 0

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "gt_malignant", "p_malignant", "pred_malignant", "logp_0", "logp_1", "cand0", "cand1"])

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
                p1, dbg = prob_malignant_from_logits(model, processor, img, prompt)
            except Exception:
                p1, dbg = prob_malignant_from_logits(model, processor, img, prompt, prefer_leading_space=False)

            pred = 1 if p1 >= args.threshold else 0

            y_true.append(gt)
            y_prob.append(p1)
            y_pred.append(pred)

            w.writerow([fn, gt, f"{p1:.6f}", pred, f"{dbg['logp_0']:.6f}", f"{dbg['logp_1']:.6f}", dbg["cand0"], dbg["cand1"]])

    if len(y_true) == 0:
        raise RuntimeError("No valid samples were evaluated. Check paths.")

    metrics = compute_metrics(y_true, y_prob, y_pred)
    cis = bootstrap_metric_cis(
        y_true=y_true,
        y_prob=y_prob,
        threshold=args.threshold,
        n_bootstrap=args.ci_bootstrap,
        alpha=args.ci_alpha,
        seed=args.ci_seed,
    )

    print("\n==== Llama-3.2 Thyroid Binary Classification Metrics ====")
    print(f"Evaluated samples: {len(y_true)}")
    print(f"Missing files: {missing_files}")
    print(f"Bad images: {bad_images}")
    print(f"AUROC: {metrics['auroc']:.6f} (95% CI {cis['auroc'][0]:.6f}-{cis['auroc'][1]:.6f})")
    print(f"AUPRC: {metrics['auprc']:.6f} (95% CI {cis['auprc'][0]:.6f}-{cis['auprc'][1]:.6f})")
    print(f"Acc: {metrics['accuracy']:.6f} (95% CI {cis['accuracy'][0]:.6f}-{cis['accuracy'][1]:.6f})")
    print(f"F1: {metrics['f1']:.6f} (95% CI {cis['f1'][0]:.6f}-{cis['f1'][1]:.6f})")
    print(f"Sensitivity: {metrics['sensitivity']:.6f} (95% CI {cis['sensitivity'][0]:.6f}-{cis['sensitivity'][1]:.6f})")
    print(f"Specificity: {metrics['specificity']:.6f} (95% CI {cis['specificity'][0]:.6f}-{cis['specificity'][1]:.6f})")
    print(f"Confusion (tn fp fn tp): {metrics['tn']} {metrics['fp']} {metrics['fn']} {metrics['tp']}")
    print(f"Saved per-image predictions to: {args.out_csv}")


if __name__ == "__main__":
    main()