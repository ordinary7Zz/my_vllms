"""
OpenAI-compatible Thyroid Binary Classification Evaluation Script

This script uses an OpenAI-compatible vision model to perform thyroid ultrasound
nodule malignancy classification and compute comprehensive metrics with bootstrap
confidence intervals.

Requirements:
  - openai>=1.0.0
  - numpy, pandas, sklearn

Environment:
  - Set OPENAI_API_KEY or POE_API_KEY environment variable with your API key
"""

import os
import json
import argparse
import base64
import re
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

from debug_print_response_once import print_raw_response_once


# ----------------------------
# Utils
# ----------------------------
def load_labels(label_json: str, label_key: str) -> List[Dict]:
    """
    Load label JSON file.
    Expected format (list of dict):
      [{"filename": "...", "<label_key>": 0/1}, ...]
    """
    with open(label_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Label JSON must be a list of records.")
    for r in data:
        if "filename" not in r or label_key not in r:
            raise ValueError(f"Each record must contain keys: filename, {label_key}.")
        r["label"] = int(r[label_key])
        if r["label"] not in (0, 1):
            raise ValueError(f"{label_key} must be 0/1, got {r['label']}")
    return data


def safe_open_image(path: str) -> Image.Image:
    """Safely open and convert image to RGB."""
    with Image.open(path) as img:
        return img.convert("RGB")


TASK_CONFIGS = {
    "malignancy": {
        "label_key": "malignancy",
        "display_name": "thyroid ultrasound benign-vs-malignant classification",
        "positive_label": "malignant",
        "negative_label": "benign",
        "visual_cues": (
            "Focus on irregular or infiltrative margins, marked hypoechogenicity, microcalcifications, "
            "taller-than-wide shape, and other suspicious malignant features."
        ),
    },
    "lymph_node_metastasis": {
        "label_key": "lymph_node_metastasis",
        "display_name": "thyroid ultrasound cervical lymph node metastasis classification",
        "positive_label": "yes",
        "negative_label": "no",
        "visual_cues": (
            "Focus on suspicious nodal features such as round shape, loss of fatty hilum, cystic change, "
            "microcalcifications, abnormal cortical thickening, and abnormal vascularity."
        ),
    },
    "ftc_ptc": {
        "label_key": "ftc_ptc",
        "display_name": "thyroid ultrasound FTC-vs-PTC classification",
        "positive_label": "FTC",
        "negative_label": "PTC",
        "visual_cues": (
            "PTC often shows microcalcifications, marked hypoechogenicity, irregular margins, and taller-than-wide shape, "
            "whereas FTC may appear more circumscribed or encapsulated-like and may lack classic PTC signs."
        ),
    },
}


GEMINI_MODELS = {"gemini-3.5-flash", "gemini-3.1-pro"}


def _stringify_detail(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _gemini_retry_reason(response_text: str, response_status: Optional[str], incomplete_details) -> str:
    text = (response_text or "").strip()
    if not text:
        if response_status == "incomplete":
            return "response_status=incomplete with empty text"
        return "empty text"

    if response_status == "incomplete":
        if re.search(r'"?prediction"?\s*[:=]\s*$', text, flags=re.IGNORECASE):
            return "response_status=incomplete with missing prediction value"
        if re.search(r'"?confidence"?\s*[:=]\s*$', text, flags=re.IGNORECASE):
            return "response_status=incomplete with missing confidence value"
        if text.count("{") > text.count("}") or text.count("[") > text.count("]"):
            return "response_status=incomplete with truncated JSON"
        return "response_status=incomplete"

    if text.count("{") > text.count("}") or text.count("[") > text.count("]"):
        return "truncated JSON"
    if text.startswith("```") and not text.rstrip().endswith("```"):
        return "truncated code fence"
    if re.search(r'"?prediction"?\s*[:=]\s*$', text, flags=re.IGNORECASE):
        return "missing prediction value"
    if re.search(r'"?confidence"?\s*[:=]\s*$', text, flags=re.IGNORECASE):
        return "missing confidence value"

    return ""


def build_prompts(task_name: str) -> Tuple[str, str]:
    cfg = TASK_CONFIGS[task_name]
    system_prompt = (
        "You are an expert medical image classifier for thyroid ultrasound images. "
        f"The current task is {cfg['display_name']}. "
        f"Label mapping: 1 = {cfg['positive_label']}, 0 = {cfg['negative_label']}. "
        f"{cfg['visual_cues']} "
        "Use only visual evidence from the single input image. "
        'Output valid JSON only in this exact form: {"prediction": 0 or 1, "risk_score": 0.00 to 100.00}. '
        "risk_score is a continuous malignancy suspicion score, not a probability. "
        "Use fine-grained decimals across the full range and avoid coarse bins or stock values. "
        "Do not output markdown, code fences, explanations, or extra keys."
    )
    user_prompt = (
        f"Task: {cfg['display_name']}.\n"
        f"1 = {cfg['positive_label']}; 0 = {cfg['negative_label']}.\n"
        "Classify the image using only visual evidence from the thyroid ultrasound.\n"
        'Return JSON only: {"prediction": 0 or 1, "risk_score": 0.00 to 100.00}. '
        "risk_score should be a continuous malignancy suspicion score."
    )
    return system_prompt, user_prompt


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _json_fragment_candidates(text: str) -> List[str]:
    candidates = [text]
    start_positions = [idx for idx in (text.find("{"), text.find("[")) if idx != -1]
    if start_positions:
        start = min(start_positions)
        fragment = text[start:].strip()
        candidates.append(fragment)

        open_braces = fragment.count("{") - fragment.count("}")
        open_brackets = fragment.count("[") - fragment.count("]")
        if open_braces > 0 or open_brackets > 0:
            repaired = fragment + ("]" * max(open_brackets, 0)) + ("}" * max(open_braces, 0))
            candidates.append(repaired)

    return candidates


def _parse_json_candidate(candidate: str) -> Optional[Dict]:
    candidate = candidate.strip()
    if not candidate:
        return None
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_binary_fields(parsed: Dict) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[str]]:
    try:
        pred = int(parsed.get("prediction"))
    except (TypeError, ValueError):
        return None, None, None, "prediction_missing_or_invalid"

    if pred not in (0, 1):
        return None, None, None, "prediction_out_of_range"

    score_raw = parsed.get("risk_score")
    if score_raw is not None:
        try:
            risk_score = float(score_raw)
        except (TypeError, ValueError):
            return None, None, None, "risk_score_invalid"
        if not 0.0 <= risk_score <= 100.0:
            return None, None, None, "risk_score_out_of_range"
        prob_malignant = risk_score / 100.0
        return pred, prob_malignant, risk_score, ""

    prob_raw = parsed.get("prob_malignant", parsed.get("malignant_probability"))
    if prob_raw is not None:
        try:
            prob_malignant = float(prob_raw)
        except (TypeError, ValueError):
            return None, None, None, "prob_malignant_invalid"
        if not 0.0 <= prob_malignant <= 1.0:
            return None, None, None, "prob_malignant_out_of_range"
        return pred, prob_malignant, prob_malignant * 100.0, ""

    conf_raw = parsed.get("confidence")
    try:
        confidence = float(conf_raw)
    except (TypeError, ValueError):
        return None, None, None, "confidence_missing_or_invalid"

    if not 0.0 <= confidence <= 1.0:
        return None, None, None, "confidence_out_of_range"

    prob_malignant = confidence if pred == 1 else 1.0 - confidence
    return pred, prob_malignant, prob_malignant * 100.0, ""


def _summarize_parse_failure(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "empty_response"
    if stripped.startswith("```") and not stripped.rstrip().endswith("```"):
        return "truncated_code_fence"
    if re.search(r'"?prediction"?\s*[:=]\s*$', stripped, flags=re.IGNORECASE):
        return "prediction_value_missing"
    if re.search(r'"?(?:risk_score|confidence)"?\s*[:=]\s*$', stripped, flags=re.IGNORECASE):
        return "risk_score_value_missing"
    if stripped.count("{") > stripped.count("}") or stripped.count("[") > stripped.count("]"):
        return "truncated_json"
    if "prediction" in stripped.lower():
        return "prediction_not_recovered"
    if "risk_score" in stripped.lower() or "confidence" in stripped.lower():
        return "risk_score_not_recovered"
    return "unrecoverable_json"


def parse_binary_response(response_text: str) -> Tuple[Optional[int], Optional[float], Optional[float], str]:
    text = _strip_code_fences(response_text or "")
    if not text:
        return None, None, None, "empty_response"

    parsed = None
    for candidate in _json_fragment_candidates(text):
        parsed = _parse_json_candidate(candidate)
        if parsed is not None:
            break

    if isinstance(parsed, dict):
        pred, prob_malignant, parsed_risk_score, reason = _extract_binary_fields(parsed)
        if pred is not None and prob_malignant is not None:
            return pred, prob_malignant, parsed_risk_score, ""
        return None, None, None, reason or _summarize_parse_failure(text)

    match = re.search(r'"?prediction"?\s*[:=]\s*["\']?([01])["\']?', text, flags=re.IGNORECASE)
    if match:
        pred = int(match.group(1))
        risk_match = re.search(r'"?risk_score"?\s*[:=]\s*["\']?([0-9]*\.?[0-9]+)', text, flags=re.IGNORECASE)
        if risk_match:
            risk_score = float(risk_match.group(1))
            if 0.0 <= risk_score <= 100.0:
                prob_malignant = risk_score / 100.0
                return pred, prob_malignant, risk_score, ""
            return None, None, None, "risk_score_out_of_range"
        legacy_match = re.search(r'"?(?:confidence|prob_malignant)"?\s*[:=]\s*["\']?([0-9]*\.?[0-9]+)', text, flags=re.IGNORECASE)
        if legacy_match:
            confidence = float(legacy_match.group(1))
            if 0.0 <= confidence <= 1.0:
                prob_malignant = confidence if pred == 1 else 1.0 - confidence
                return pred, prob_malignant, confidence * 100.0, ""
            return None, None, None, "confidence_out_of_range"
        prob_malignant = 1.0 if pred == 1 else 0.0
        return pred, prob_malignant, None, ""

    match = re.match(r"^\s*([01])(?:\s|$)", text)
    if match:
        pred = int(match.group(1))
        prob_malignant = 1.0 if pred == 1 else 0.0
        return pred, prob_malignant, None, ""

    return None, None, None, _summarize_parse_failure(text)


def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def extract_response_text(response) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    output = getattr(response, "output", None)
    if output:
        parts: List[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if content is None and isinstance(item, dict):
                content = item.get("content")

            if isinstance(content, list):
                for block in content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict):
                        if block.get("type") in {"output_text", "text"}:
                            parts.append(str(block.get("text", "")))
                        elif "text" in block:
                            parts.append(str(block.get("text", "")))
                    else:
                        block_text = getattr(block, "text", None)
                        if block_text is not None:
                            parts.append(str(block_text))
            elif isinstance(content, str):
                parts.append(content)

        text = "".join(parts).strip()
        if text:
            return text

    choices = getattr(response, "choices", None)
    if choices:
        for choice in choices:
            message = getattr(choice, "message", None)
            if message is None and isinstance(choice, dict):
                message = choice.get("message")
            if message is None:
                continue
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()

    return ""


def classify_with_gpt(
    client: OpenAI,
    image_path: str,
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4-vision-preview",
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> Tuple[float, Dict[str, object]]:
    """
    Classify a thyroid ultrasound image using an OpenAI-compatible model.
    """
    image_data = image_to_base64(image_path)

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
            response_text = ""
            response_status = ""
            response_incomplete_details = ""

            if model in GEMINI_MODELS:
                response = client.responses.create(
                    model=model,
                    instructions=system_prompt,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "image_url": f"data:{media_type};base64,{image_data}",
                                },
                                {
                                    "type": "input_text",
                                    "text": user_prompt,
                                },
                            ],
                        }
                    ],
                    max_output_tokens=512,
                    reasoning={"effort": "none"},
                )
                print_raw_response_once(response)
                response_text = extract_response_text(response)
                response_status = str(getattr(response, "status", "") or "")
                response_incomplete_details = _stringify_detail(getattr(response, "incomplete_details", None))

                retry_reason = _gemini_retry_reason(response_text, response_status, response_incomplete_details)
                if retry_reason and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    detail_suffix = f"; details={response_incomplete_details}" if response_incomplete_details else ""
                    print(f"  Gemini response looks truncated ({retry_reason}{detail_suffix}). Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue

            if not response_text:
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
                    max_tokens=128,
                )
                print_raw_response_once(response)
                response_text = extract_response_text(response)
                response_status = str(getattr(response, "status", "") or "chat_completions")
                response_incomplete_details = _stringify_detail(getattr(response, "incomplete_details", None))

            pred, prob_malignant, parsed_risk_score, parse_reason = parse_binary_response(response_text)
            if pred is None or prob_malignant is None:
                failure_reason = parse_reason or _summarize_parse_failure(response_text)
                if response_status:
                    status_bits = [f"response_status={response_status}"]
                    if response_incomplete_details:
                        status_bits.append(f"incomplete_details={response_incomplete_details}")
                    failure_reason = f"{failure_reason} ({', '.join(status_bits)})"
                return 0.5, {
                    "raw_response": response_text,
                    "parsed_prediction": -1,
                    "parsed_confidence": 0.5,
                    "parsed_risk_score": 50.0,
                    "parsed_prob_malignant": 0.5,
                    "prob_malignant": 0.5,
                    "reasoning": "Failed to parse response",
                    "parse_failure_reason": failure_reason,
                    "response_status": response_status,
                    "response_incomplete_details": response_incomplete_details,
                }

            return prob_malignant, {
                "raw_response": response_text,
                "parsed_prediction": pred,
                "parsed_confidence": prob_malignant,
                "parsed_risk_score": parsed_risk_score if parsed_risk_score is not None else prob_malignant * 100.0,
                "parsed_prob_malignant": prob_malignant,
                "prob_malignant": prob_malignant,
                "reasoning": "",
                "parse_failure_reason": "",
                "response_status": response_status,
                "response_incomplete_details": response_incomplete_details,
            }

        except RateLimitError:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"  Rate limited. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
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
        "--task",
        type=str,
        required=True,
        choices=list(TASK_CONFIGS.keys()),
        help="Classification task to run",
    )
    ap.add_argument(
        "--label_key",
        type=str,
        default=None,
        help="Label field name inside label_json (default depends on --task)",
    )
    ap.add_argument(
        "--out_json",
        type=str,
        default="gpt_thyroid_preds.json",
        help="Output JSON with per-sample predictions",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        choices=[
            "gpt-4-vision-preview",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-5.5",
            "gemini-3.5-flash",
            "gemini-3.1-pro",
        ],
        help="OpenAI-compatible model to use",
    )
    ap.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key (default: read from OPENAI_API_KEY or POE_API_KEY env var)",
    )
    ap.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL (e.g. https://api.poe.com/v1)",
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
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("POE_API_KEY")
    if not api_key:
        raise ValueError(
            "API key not provided. Set via --api_key, OPENAI_API_KEY, or POE_API_KEY"
        )

    base_url = args.base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("POE_API_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)


    # Load labels
    task_config = TASK_CONFIGS[args.task]
    label_key = args.label_key or task_config["label_key"]
    system_prompt, user_prompt = build_prompts(args.task)
    labels = load_labels(args.label_json, label_key)
    if args.limit is not None and args.limit > 0:
        labels = labels[: args.limit]

    print(f"Loaded {len(labels)} labels from {args.label_json} (label_key={label_key})")
    print(f"Task:              {args.task}")
    print(f"Using model:       {args.model}")

    y_true: List[int] = []
    y_prob: List[float] = []
    y_pred: List[int] = []
    results: List[Dict[str, object]] = []

    missing_files = 0
    bad_images = 0
    api_errors = 0

    for r in tqdm(labels, desc="Infer"):
            fn = r["filename"]
            gt = int(r["label"])

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
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=args.model,
                    max_retries=args.max_retries,
                    retry_delay=args.retry_delay,
                )
            except Exception as e:
                api_errors += 1
                print(f"  API error for {fn}: {e}")
                p1 = 0.5  # Default to uncertain
                dbg = {
                    "reasoning": f"API error: {str(e)}",
                    "parsed_prediction": -1,
                    "parsed_confidence": 0.5,
                    "parsed_risk_score": 50.0,
                    "parsed_prob_malignant": 0.5,
                    "prob_malignant": 0.5,
                    "parse_failure_reason": "api_error",
                    "response_status": "",
                    "response_incomplete_details": "",
                }

            pred = 1 if p1 >= args.threshold else 0
            p0 = 1.0 - p1

            y_true.append(gt)
            y_prob.append(p1)
            y_pred.append(pred)
            results.append(
                {
                    "record_type": "sample",
                    "image_file": img_path,
                    "image_name": fn,
                    "filename": fn,
                    "selected_task": args.task,
                    "label_key": label_key,
                    "selected_model": args.model,
                    "predicted_class": pred,
                    "confidence": float(dbg.get("parsed_confidence", p1)),
                    "risk_score": float(dbg.get("parsed_risk_score", p1 * 100.0)),
                    "prob_malignant": float(dbg.get("parsed_prob_malignant", p1)),
                    "prob_class_0": float(p0),
                    "prob_class_1": float(p1),
                    "true_label": gt,
                    "parsed_prediction": dbg.get("parsed_prediction", -1),
                    "parsed_confidence": dbg.get("parsed_confidence", p1),
                    "parsed_risk_score": dbg.get("parsed_risk_score", p1 * 100.0),
                    "parsed_prob_malignant": dbg.get("parsed_prob_malignant", p1),
                    "parse_failure_reason": dbg.get("parse_failure_reason", ""),
                    "response_status": dbg.get("response_status", ""),
                    "response_incomplete_details": dbg.get("response_incomplete_details", ""),
                    "raw_response": dbg.get("raw_response", ""),
                    "reasoning": dbg.get("reasoning", ""),
                }
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
    print(f"GPT Thyroid Binary Classification Metrics ({args.task}, 1={task_config['positive_label']}, 0={task_config['negative_label']})")
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
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved per-sample predictions to: {args.out_json}")


if __name__ == "__main__":
    main()
