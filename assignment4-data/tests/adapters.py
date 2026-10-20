from __future__ import annotations

import os
from typing import Any


from scripts.extract_text import extract_text_from_html
from scripts.predict_language import predict
from scripts.mask_pii import mask_email, mask_phone, mask_ip
from scripts.harmful_content_detector import classify_nsfw, classify_hate

def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return extract_text_from_html(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    return predict(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    return mask_email(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return mask_phone(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return mask_ip(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return classify_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return classify_hate(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    raise NotImplementedError


def run_gopher_quality_filter(text: str) -> bool:
    raise NotImplementedError


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    raise NotImplementedError


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    raise NotImplementedError
