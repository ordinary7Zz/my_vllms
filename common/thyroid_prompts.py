from typing import Optional

MEDGEMMA_PROMPT = (
"<start_of_image>\n"
    "You are a medical imaging assistant.\n"
    "Task: Thyroid ultrasound nodule malignancy classification.\n"
    "Output exactly one character: 0 or 1.\n"
    "0 = benign, 1 = malignant.\n"
    "Answer:"
)

QWEN3_SYSTEM_TEXT = (
    "You are a medical imaging assistant. "
    "You will be given a thyroid ultrasound image. "
    "Your task is binary malignancy classification."
)

QWEN3_USER_TEXT = (
    "Task: Thyroid ultrasound nodule malignancy classification.\n"
    "Output exactly one character: 0 or 1.\n"
    "0 = benign, 1 = malignant.\n"
    "Answer:"
)


def label_to_answer_text(label: int, leading_space: bool = True) -> str:
    return (" " if leading_space else "") + str(int(label))


def build_qwen3_messages(image, answer_text: Optional[str] = None):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": QWEN3_SYSTEM_TEXT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": QWEN3_USER_TEXT},
            ],
        },
    ]
    if answer_text is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer_text}]})
    return messages
