"""Utilities for printing a raw API response exactly once."""

from __future__ import annotations

import json


def print_raw_response_once(response) -> None:
    if getattr(print_raw_response_once, "_printed", False):
        return

    print("\nRAW RESPONSE OBJECT (first call):")
    try:
        if hasattr(response, "model_dump_json"):
            print(response.model_dump_json(indent=2))
        elif hasattr(response, "model_dump"):
            print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2, default=str))
        else:
            print(repr(response))
    except Exception as e:
        print(f"<failed to print response: {e}>")
        print(repr(response))

    print_raw_response_once._printed = True
