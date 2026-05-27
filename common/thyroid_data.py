import json
import os
from collections import Counter
from typing import Dict, Iterable, List, Tuple, Union

from PIL import Image
from torch.utils.data import Dataset


Record = Dict[str, Union[int, str]]


def load_labels(label_json: str, label_key: str = "malignancy") -> List[Record]:
    with open(label_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Label JSON must be a list of records.")
    records: List[Record] = []
    for r in data:
        if not isinstance(r, dict):
            raise ValueError("Each record must be a dict.")
        if "filename" not in r or label_key not in r:
            raise ValueError(f"Each record must contain keys: filename, {label_key}.")
        filename = str(r["filename"])
        malignancy = int(r[label_key])
        if malignancy not in (0, 1):
            raise ValueError(f"{label_key} must be 0/1, got {malignancy}")
        records.append({"filename": filename, "malignancy": malignancy})
    return records


def resolve_image_path(image_dir: str, filename: str) -> str:
    return os.path.join(image_dir, filename)


def safe_open_image(path: str) -> Image.Image:
    img = Image.open(path)
    return img.convert("RGB")


def class_distribution(records: Iterable[Record]) -> Dict[int, int]:
    counts = Counter(int(r["malignancy"]) for r in records)
    return {0: counts.get(0, 0), 1: counts.get(1, 0)}


def dataset_summary(records: List[Record]) -> Dict[str, Union[int, Dict[int, int]]]:
    return {
        "num_samples": len(records),
        "class_distribution": class_distribution(records),
    }


def find_filename_overlap(records_a: List[Record], records_b: List[Record]) -> List[str]:
    names_a = {str(r["filename"]) for r in records_a}
    names_b = {str(r["filename"]) for r in records_b}
    return sorted(names_a & names_b)


def validate_image_records(records: List[Record], image_dir: str) -> Dict[str, int]:
    missing_files = 0
    bad_images = 0
    for record in records:
        image_path = resolve_image_path(image_dir, str(record["filename"]))
        if not os.path.exists(image_path):
            missing_files += 1
            continue
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception:
            bad_images += 1
    return {
        "num_samples": len(records),
        "missing_files": missing_files,
        "bad_images": bad_images,
        "valid_files": len(records) - missing_files - bad_images,
    }


def print_dataset_report(name: str, records: List[Record], image_dir: str) -> None:
    summary = dataset_summary(records)
    validation = validate_image_records(records, image_dir)
    dist = summary["class_distribution"]
    print(f"[{name}] samples={summary['num_samples']} benign={dist[0]} malignant={dist[1]}")
    print(
        f"[{name}] valid={validation['valid_files']} missing={validation['missing_files']} bad_images={validation['bad_images']}"
    )


class ThyroidBinaryDataset(Dataset):
    def __init__(self, records: List[Record], image_dir: str, answer_with_space: bool = True):
        self.records = records
        self.image_dir = image_dir
        self.answer_with_space = answer_with_space

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        filename = str(record["filename"])
        label = int(record["malignancy"])
        image_path = resolve_image_path(self.image_dir, filename)
        image = safe_open_image(image_path)
        answer = (" " if self.answer_with_space else "") + str(label)
        return {
            "filename": filename,
            "label": label,
            "answer": answer,
            "image": image,
        }
