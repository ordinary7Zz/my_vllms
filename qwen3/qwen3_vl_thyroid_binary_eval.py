import os
import sys
import csv
import argparse
from typing import List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from common.thyroid_data import load_labels, safe_open_image
from common.thyroid_metrics import compute_metrics, bootstrap_metric_cis
from common.vlm_sft import (
    load_peft_adapter_if_needed,
    maybe_build_quant_config,
    prob_malignant_from_qwen3_logits,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_dir",
        type=str,
        default="/mnt/wangbd8/workspace/qwen3_vl_8b_instruct",
        help="Local path to Qwen3-VL-8B-Instruct model dir",
    )
    ap.add_argument(
        "--adapter_dir",
        type=str,
        default=None,
        help="Optional LoRA adapter directory saved by the training script",
    )
    ap.add_argument("--image_dir", type=str, required=True,
                    help="Directory containing images referenced by filename in label json")
    ap.add_argument("--label_json", type=str, required=True,
                    help="Label json path, e.g. /mnt/data/tn3k_test_label.json")
    ap.add_argument("--out_csv", type=str, default="qwen3_vl_8b_preds.csv",
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
    ap.add_argument("--attn_impl", type=str, default="auto",
                    choices=["auto", "flash_attention_2", "sdpa", "eager"],
                    help="Attention implementation; use flash_attention_2 if installed")
    ap.add_argument("--use_qlora", action="store_true",
                    help="Load the base model with the same 4-bit QLoRA quantization used during training")
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

    if args.adapter_dir and os.path.exists(os.path.join(args.adapter_dir, "preprocessor_config.json")):
        processor_source = args.adapter_dir
    else:
        processor_source = args.model_dir
    processor = AutoProcessor.from_pretrained(processor_source, use_fast=args.use_fast_processor)

    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch_dtype,
    }
    quantization_config = maybe_build_quant_config(args.use_qlora)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    if args.attn_impl != "auto":
        model_kwargs["attn_implementation"] = args.attn_impl

    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_dir, **model_kwargs)
    model = load_peft_adapter_if_needed(model, args.adapter_dir)
    model.eval()

    y_true: List[int] = []
    y_prob: List[float] = []
    y_pred: List[int] = []

    print(f"Model dir:         {args.model_dir}")
    print(f"Adapter dir:       {args.adapter_dir or 'none'}")
    print(f"Processor source:  {processor_source}")
    print(f"Use QLoRA:         {args.use_qlora}")
    print(f"Attention impl:    {args.attn_impl}")

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
                p1, dbg = prob_malignant_from_qwen3_logits(model=model, processor=processor, image=img, return_debug=True)
            finally:
                img.close()
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

    print("\n==== Qwen3-VL-8B-Instruct Thyroid Binary Classification Metrics (pos=malignant=1) ====")
    print(f"Model dir:         {args.model_dir}")
    print(f"Evaluated samples: {len(y_true)}")
    print(f"Missing files:     {missing_files}")
    print(f"Bad images:        {bad_images}")
    print(f"AUROC:             {metrics['auroc']:.6f} (95% CI {cis['auroc'][0]:.6f}-{cis['auroc'][1]:.6f})")
    print(f"AUPRC:             {metrics['auprc']:.6f} (95% CI {cis['auprc'][0]:.6f}-{cis['auprc'][1]:.6f})")
    print(f"Acc:               {metrics['accuracy']:.6f} (95% CI {cis['accuracy'][0]:.6f}-{cis['accuracy'][1]:.6f})")
    print(f"F1:                {metrics['f1']:.6f} (95% CI {cis['f1'][0]:.6f}-{cis['f1'][1]:.6f})")
    print(f"Sensitivity:       {metrics['sensitivity']:.6f} (95% CI {cis['sensitivity'][0]:.6f}-{cis['sensitivity'][1]:.6f})")
    print(f"Specificity:       {metrics['specificity']:.6f} (95% CI {cis['specificity'][0]:.6f}-{cis['specificity'][1]:.6f})")
    print(f"Confusion (tn fp fn tp): {metrics['tn']} {metrics['fp']} {metrics['fn']} {metrics['tp']}")
    print(f"Saved per-image predictions to: {args.out_csv}")


if __name__ == "__main__":
    main()
