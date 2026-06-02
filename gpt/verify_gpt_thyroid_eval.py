#!/usr/bin/env python3
"""
Verify whether the current machine can successfully send an OpenAI-compatible request.

This is a minimal connectivity check for the API configuration used by gpt_thyroid_eval.sh.
It does not require local image or label data.
"""

import argparse
import os
import sys
from typing import Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    print("FAIL: openai package is not installed. Run: pip install openai")
    sys.exit(1)

DEFAULT_MODEL = "gpt-5.5"
DEFAULT_BASE_URL = "https://api.poe.com/v1"
TEXT_MODELS = {"gpt-5.5", "gemini-3.5-flash", "gemini-3.1-pro"}


def resolve_api_key(cli_api_key: Optional[str]) -> Tuple[Optional[str], str]:
    if cli_api_key:
        return cli_api_key, "--api_key"
    if os.getenv("POE_API_KEY"):
        return os.getenv("POE_API_KEY"), "POE_API_KEY"
    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY"
    return None, ""


def resolve_base_url(cli_base_url: Optional[str]) -> Tuple[Optional[str], str]:
    if cli_base_url:
        return cli_base_url, "--base_url"
    if os.getenv("POE_API_BASE_URL"):
        return os.getenv("POE_API_BASE_URL"), "POE_API_BASE_URL"
    if os.getenv("OPENAI_BASE_URL"):
        return os.getenv("OPENAI_BASE_URL"), "OPENAI_BASE_URL"
    return DEFAULT_BASE_URL, "default"


def mask_secret(secret: str) -> str:
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}...{secret[-4:]}"


def send_test_request(client: OpenAI, model: str) -> str:
    prompt = "Reply with exactly: pong"

    if model in TEXT_MODELS:
        response = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=16,
        )
        return response.output_text.strip()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=16,
    )
    return response.choices[0].message.content.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify whether gpt_thyroid_eval.sh API settings can send a request"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--base_url", default=None)
    args = parser.parse_args()

    print("[1/2] Resolving API configuration...")
    api_key, api_key_source = resolve_api_key(args.api_key)
    if not api_key:
        print("FAIL: API key not configured.")
        print("Set one of: --api_key, OPENAI_API_KEY, POE_API_KEY")
        return 1

    base_url, base_url_source = resolve_base_url(args.base_url)
    print(f"  API key source: {api_key_source}")
    print(f"  API key: {mask_secret(api_key)}")
    print(f"  Base URL source: {base_url_source}")
    print(f"  Base URL: {base_url}")
    print(f"  Model: {args.model}")

    print("[2/2] Sending one test request...")
    try:
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        output = send_test_request(client, args.model)
    except Exception as exc:
        print(f"FAIL: API request failed: {exc}")
        return 1

    print("SUCCESS: Request completed.")
    print(f"  Response: {output}")
    print("  This machine can reach the configured OpenAI-compatible endpoint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
