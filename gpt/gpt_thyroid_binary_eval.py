"""
GPT-4V/GPT-4o Thyroid Binary Classification Evaluation Script

This script uses OpenAI's vision model to perform thyroid ultrasound nodule 
malignancy classification and compute comprehensive metrics with bootstrap 
confidence intervals.

Requirements:
  - openai>=1.0.0
  - numpy, pandas, sklearn
  
Environment:
  - Set OPENAI_API_KEY environment variable with your API key
"""

import os
import json
import csv
import argparse
import base64
import time
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

try:
    from openai import OpenAI, RateLimitError
except ImportError:
    raise ImportError("Please install openai: pip install openai")


# ----------------------------
# Utils
# ----------------------------
def load_labels(label_json: str) -> List[Dict]:
    """
    Load label JSON file.
    Expected format (list of dict):
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
    """Safely open and convert image to RGB."""
    img = Image.open(path)
    return img.convert("RGB")


def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def classify_with_gpt(
    client: OpenAI,
    image_path: str,
    model: str = "gpt-4-vision-preview",
    temperature: float = 0.0,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> Tuple[float, Dict[str, str]]:
    """
    Classify thyroid nodule malignancy using OpenAI's GPT model with vision capability.
    
    Args:
        client: OpenAI client instance
        image_path: Path to the image file
        model: Model name (gpt-4-vision-preview, gpt-4o, etc.)
        temperature: Model temperature (0.0 for deterministic)
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    
    Returns:
        Tuple of (malignancy_probability: float, debug_info: Dict)
        
    The function extracts probability from the model's response using regex patterns.
    If the model responds with a confidence score, it's normalized to [0,1].
    """
    system_prompt = (
        "You are an expert radiologist specializing in thyroid ultrasound analysis. "
        "Your task is to classify thyroid nodules as benign (0) or malignant (1) based on ultrasound images. "
        "Provide your classification and a confidence score.\n\n"
        "IMPORTANT: You MUST respond in JSON format like this:\n"
        '{"prediction": 0 or 1, "confidence": 0.0-1.0, "reasoning": "brief explanation"}\n'
        "The confidence field should be a probability between 0 and 1, where 0=certain benign, 1=certain malignant."
    )

    user_prompt = (
        "Please analyze this thyroid ultrasound image and classify the nodule malignancy. "
        "Respond ONLY in JSON format with prediction (0=benign, 1=malignant), "
        "confidence (0.0-1.0), and brief reasoning."
    )

    # Encode image to base64
    image_data = image_to_base64(image_path)
    
    # Get file extension to determine media type
    ext = Path(image_path).suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(ext, "image/jpeg")

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_data}",
                                },
                            },
                            {
                                "type": "text",
                                "text": user_prompt,
                            },
                        ],
                    },
                ],
                temperature=temperature,
                max_tokens=256,
            )

            response_text = response.choices[0].message.content.strip()

            # Try to parse JSON response
            try:
                # Find JSON in response (in case of extra text)
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    result = json.loads(json_str)
                    confidence = float(result.get("confidence", 0.5))
                    # Ensure confidence is in [0, 1]
                    confidence = max(0.0, min(1.0, confidence))
                    
                    return confidence, {
                        "raw_response": response_text,
                        "parsed_prediction": result.get("prediction", -1),
                        "parsed_confidence": confidence,
                        "reasoning": result.get("reasoning", ""),
                    }
            except (json.JSONDecodeError, ValueError, AttributeError):
                # If JSON parsing fails, return 0.5 (neutral)
                return 0.5, {
                    "raw_response": response_text,
                    "parsed_prediction": -1,
                    "parsed_confidence": 0.5,
                    "reasoning": "Failed to parse response",
                }

        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"  Rate limited. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
            else:
                raise

    raise RuntimeError(f"Failed to classify image after {max_retries} attempts")


def compute_metrics(y_true: List[int], y_prob: List[float], y_pred: List[int]) -> Dict[str, float]:
    """Compute classification metrics."""
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
    """
    Compute bootstrap confidence intervals for metrics.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        threshold: Decision threshold
        n_bootstrap: Number of bootstrap samples
        alpha: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed
    
    Returns:
        Dictionary mapping metric names to (lower, upper) CI bounds
    """
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
    ap = argparse.ArgumentParser(
        description="Evaluate thyroid nodule malignancy classification using GPT vision models"
    )
    ap.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing images referenced by filename in label json",
    )
    ap.add_argument(
        "--label_json",
        type=str,
        required=True,
        help="Label json path, e.g. /path/to/test_label.json",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="gpt_thyroid_preds.csv",
        help="Output CSV with per-image predictions",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        choices=["gpt-4-vision-preview", "gpt-4o", "gpt-4o-mini"],
        help="GPT model to use",
    )
    ap.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (default: read from OPENAI_API_KEY env var)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold on P(malignant)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Run only first N samples for quick test (-1 for all)",
    )
    ap.add_argument(
        "--ci_bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap resamples for confidence intervals",
    )
    ap.add_argument(
        "--ci_alpha",
        type=float,
        default=0.95,
        help="Confidence level for intervals, e.g. 0.95",
    )
    ap.add_argument(
        "--ci_seed",
        type=int,
        default=42,
        help="Random seed for bootstrap confidence intervals",
    )
    ap.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retries for API calls",
    )
    ap.add_argument(
        "--retry_delay",
        type=float,
        default=2.0,
        help="Delay between retries in seconds",
    )
    args = ap.parse_args()

    # Initialize OpenAI client
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not provided. Set via --api_key or OPENAI_API_KEY env var")
    
    client = OpenAI(api_key=api_key)

    # Load labels
    labels = load_labels(args.label_json)
    if args.limit is not None and args.limit > 0:
        labels = labels[: args.limit]

    print(f"Loaded {len(labels)} labels from {args.label_json}")
    print(f"Using model: {args.model}")

    y_true: List[int] = []
    y_prob: List[float] = []
    y_pred: List[int] = []

    missing_files = 0
    bad_images = 0
    api_errors = 0

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "filename",
                "gt_malignant",
                "p_malignant",
                "pred_malignant",
                "parsed_prediction",
                "reasoning",
            ]
        )

        for r in tqdm(labels, desc="Infer"):
            fn = r["filename"]
            gt = int(r["malignancy"])

            img_path = os.path.join(args.image_dir, fn)
            if not os.path.exists(img_path):
                missing_files += 1
                print(f"  Missing: {img_path}")
                continue

            try:
                img = safe_open_image(img_path)
            except Exception as e:
                bad_images += 1
                print(f"  Bad image: {img_path} ({e})")
                continue

            try:
                p1, dbg = classify_with_gpt(
                    client=client,
                    image_path=img_path,
                    model=args.model,
                    temperature=0.0,
                    max_retries=args.max_retries,
                    retry_delay=args.retry_delay,
                )
            except Exception as e:
                api_errors += 1
                print(f"  API error for {fn}: {e}")
                p1 = 0.5  # Default to uncertain
                dbg = {"reasoning": f"API error: {str(e)}", "parsed_prediction": -1}

            pred = 1 if p1 >= args.threshold else 0

            y_true.append(gt)
            y_prob.append(p1)
            y_pred.append(pred)

            w.writerow(
                [
                    fn,
                    gt,
                    f"{p1:.6f}",
                    pred,
                    dbg.get("parsed_prediction", -1),
                    dbg.get("reasoning", ""),
                ]
            )

    if len(y_true) == 0:
        raise RuntimeError(
            "No valid samples were evaluated (all missing/bad?). Check paths."
        )

    metrics = compute_metrics(y_true, y_prob, y_pred)
    cis = bootstrap_metric_cis(
        y_true=y_true,
        y_prob=y_prob,
        threshold=args.threshold,
        n_bootstrap=args.ci_bootstrap,
        alpha=args.ci_alpha,
        seed=args.ci_seed,
    )

    print("\n" + "=" * 80)
    print("GPT Thyroid Binary Classification Metrics (pos=malignant=1)")
    print("=" * 80)
    print(f"Model:             {args.model}")
    print(f"Evaluated samples: {len(y_true)}")
    print(f"Missing files:     {missing_files}")
    print(f"Bad images:        {bad_images}")
    print(f"API errors:        {api_errors}")
    print(f"Decision threshold: {args.threshold}")
    print("-" * 80)
    print(
        f"AUROC:             {metrics['auroc']:.6f} (95% CI {cis['auroc'][0]:.6f}-{cis['auroc'][1]:.6f})"
    )
    print(
        f"AUPRC:             {metrics['auprc']:.6f} (95% CI {cis['auprc'][0]:.6f}-{cis['auprc'][1]:.6f})"
    )
    print(
        f"Acc:               {metrics['accuracy']:.6f} (95% CI {cis['accuracy'][0]:.6f}-{cis['accuracy'][1]:.6f})"
    )
    print(
        f"F1:                {metrics['f1']:.6f} (95% CI {cis['f1'][0]:.6f}-{cis['f1'][1]:.6f})"
    )
    print(
        f"Sensitivity:       {metrics['sensitivity']:.6f} (95% CI {cis['sensitivity'][0]:.6f}-{cis['sensitivity'][1]:.6f})"
    )
    print(
        f"Specificity:       {metrics['specificity']:.6f} (95% CI {cis['specificity'][0]:.6f}-{cis['specificity'][1]:.6f})"
    )
    print(f"Confusion (tn fp fn tp): {metrics['tn']} {metrics['fp']} {metrics['fn']} {metrics['tp']}")
    print("-" * 80)
    print(f"Saved per-image predictions to: {args.out_csv}")


if __name__ == "__main__":
    main()
