import os
import sys
import json
import csv
import argparse
from typing import List, Dict, Tuple, Optional

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
    # ultrasound images might be grayscale; convert to RGB to be safe
    return img.convert("RGB")


@torch.inference_mode()
def score_string_logprob(
    model: AutoModelForImageTextToText,
    tokenizer,
    base_inputs: Dict[str, torch.Tensor],
    target_text: str,
) -> float:
    """
    Compute log P(target_text | base_inputs) using cached KV for efficiency.
    Supports multi-token target strings robustly.

    Steps:
      1) Forward pass on base inputs -> get logits for next token + past_key_values
      2) For each token in target tokens:
          add logprob of token given current logits
          feed token with past_key_values to get next logits
    """
    # tokenize target string without special tokens
    tgt_ids = tokenizer.encode(target_text, add_special_tokens=False)
    if len(tgt_ids) == 0:
        raise ValueError("target_text tokenized to empty sequence.")

    # 1) forward base
    out = model(**base_inputs, use_cache=True)
    logits = out.logits  # [B, T, V]
    past = out.past_key_values

    # next-token logits are last position
    next_logits = logits[:, -1, :]  # [1, V]
    logprob = 0.0

    for tid in tgt_ids:
        # log softmax for this step
        step_logprob = torch.log_softmax(next_logits, dim=-1)[0, tid].item()
        logprob += step_logprob

        # feed this token to advance
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
    """
    Return P(malignant=1) by comparing log-probabilities of " 0" vs " 1" (or "0"/"1").
    We compute:
        p1 = exp(lp1) / (exp(lp0) + exp(lp1))
    Also returns raw logprobs for debugging.
    """
    tokenizer = processor.tokenizer
    device = model.device

    inputs = processor(images=img, text=prompt, return_tensors="pt")
    # move tensors
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Candidate strings: leading-space variant often matches model output after "Answer:"
    c0 = " 0" if prefer_leading_space else "0"
    c1 = " 1" if prefer_leading_space else "1"

    lp0 = score_string_logprob(model, tokenizer, inputs, c0)
    lp1 = score_string_logprob(model, tokenizer, inputs, c1)

    # stable softmax for two values
    m = max(lp0, lp1)
    p0 = torch.exp(torch.tensor(lp0 - m)).item()
    p1 = torch.exp(torch.tensor(lp1 - m)).item()
    p1_norm = p1 / (p0 + p1 + 1e-12)

    return float(p1_norm), {"logp_0": lp0, "logp_1": lp1, "cand0": c0, "cand1": c1}


# ----------------------------
# Main
# ----------------------------
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
    ap.add_argument("--out_csv", type=str, default="medgemma_preds.csv",
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

    # dtype
    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # load processor/model
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

    print(f"Model dir:         {args.model_dir}")
    print(f"Adapter dir:       {args.adapter_dir or 'none'}")
    print(f"Processor source:  {processor_source}")

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
                p1, dbg = prob_malignant_from_logits(
                    model=model,
                    processor=processor,
                    img=img,
                    prompt=prompt,
                    prefer_leading_space=True,
                )
            except Exception:
                # fallback: try without leading space
                p1, dbg = prob_malignant_from_logits(
                    model=model,
                    processor=processor,
                    img=img,
                    prompt=prompt,
                    prefer_leading_space=False,
                )

            pred = 1 if p1 >= args.threshold else 0

            y_true.append(gt)
            y_prob.append(p1)
            y_pred.append(pred)

            w.writerow([fn, gt, f"{p1:.6f}", pred, f"{dbg['logp_0']:.6f}", f"{dbg['logp_1']:.6f}", dbg["cand0"], dbg["cand1"]])

    if len(y_true) == 0:
        raise RuntimeError("No valid samples were evaluated (all missing/bad?). Check paths.")

    metrics = compute_metrics(y_true, y_prob, y_pred)
    cis = bootstrap_metric_cis(
        y_true=y_true,
        y_prob=y_prob,
        threshold=args.threshold,
        n_bootstrap=args.ci_bootstrap,
        alpha=args.ci_alpha,
        seed=args.ci_seed,
    )

    out_metrics = os.path.splitext(args.out_csv)[0] + ".txt"
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
        f"Saved per-image predictions to: {args.out_csv}",
    ]
    summary_text = "\n".join(summary_lines)

    print("\n" + summary_text)
    with open(out_metrics, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"Saved metrics to: {out_metrics}")


if __name__ == "__main__":
    main()

