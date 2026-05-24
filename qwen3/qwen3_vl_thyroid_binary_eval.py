import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from common.thyroid_data import load_labels, safe_open_image
from common.thyroid_metrics import bootstrap_metric_cis, compute_metrics
from common.thyroid_prompts import build_qwen3_messages
from common.vlm_sft import DTYPE_MAP, load_peft_adapter_if_needed


@torch.inference_mode()
def score_string_logprob(model, tokenizer, base_inputs: Dict[str, torch.Tensor], target_text: str) -> float:
    tgt_ids = tokenizer.encode(target_text, add_special_tokens=False)
    if len(tgt_ids) == 0:
        raise ValueError(f"target_text tokenized to empty sequence: {repr(target_text)}")
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


def build_qwen3_vl_inputs(processor, img) -> Dict[str, torch.Tensor]:
    messages = build_qwen3_messages(img, None)
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


def prob_malignant_from_logits(model, processor, img) -> Tuple[float, Dict[str, float]]:
    tokenizer = processor.tokenizer
    device = model.device
    base_inputs = build_qwen3_vl_inputs(processor, img)
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
            p1_norm = p1 / (p0 + p1 + 1e-12)
            return float(p1_norm), {"logp_0": lp0, "logp_1": lp1, "cand0": c0, "cand1": c1}
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"All candidate pairs failed. Last error: {last_err}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--image_dir", type=str, required=True)
    ap.add_argument("--label_json", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default="qwen3_vl_8b_preds.csv")
    ap.add_argument("--adapter_dir", type=str, default="")
    ap.add_argument("--use_fast_processor", action="store_true")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--ci_bootstrap", type=int, default=2000)
    ap.add_argument("--ci_alpha", type=float, default=0.95)
    ap.add_argument("--ci_seed", type=int, default=42)
    ap.add_argument("--attn_impl", type=str, default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager"])
    return ap.parse_args()


def main():
    args = parse_args()
    labels = load_labels(args.label_json)
    if args.limit is not None and args.limit > 0:
        labels = labels[: args.limit]

    processor = AutoProcessor.from_pretrained(args.model_dir, use_fast=args.use_fast_processor)
    model_kwargs = {"device_map": "cuda", "torch_dtype": DTYPE_MAP[args.dtype]}
    if args.attn_impl != "auto":
        model_kwargs["attn_implementation"] = args.attn_impl
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_dir, **model_kwargs)
    model = load_peft_adapter_if_needed(model, args.adapter_dir or None)
    model.eval()

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
            p1, dbg = prob_malignant_from_logits(model=model, processor=processor, img=img)
            pred = 1 if p1 >= args.threshold else 0
            y_true.append(gt)
            y_prob.append(p1)
            y_pred.append(pred)
            w.writerow([fn, gt, f"{p1:.6f}", pred, f"{dbg['logp_0']:.6f}", f"{dbg['logp_1']:.6f}", dbg["cand0"], dbg["cand1"]])

    if len(y_true) == 0:
        raise RuntimeError("No valid samples were evaluated (all missing/bad?). Check paths.")

    metrics = compute_metrics(y_true, y_prob, y_pred)
    cis = bootstrap_metric_cis(y_true=y_true, y_prob=y_prob, threshold=args.threshold, n_bootstrap=args.ci_bootstrap, alpha=args.ci_alpha, seed=args.ci_seed)
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
