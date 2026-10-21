from __future__ import annotations

import os
from typing import Any


from scripts.preprocessing.extract_text import extract_text_from_html
from scripts.preprocessing.predict_language import predict
from scripts.preprocessing.mask_pii import mask_email, mask_phone, mask_ip
from scripts.preprocessing.harmful_content_detector import classify_nsfw, classify_hate
from scripts.preprocessing.gopher_quality_filters import is_high_quality
from scripts.preprocessing.quality_classifier import predict_quality
from scripts.preprocessing.exact_deduplication import dedup
from scripts.preprocessing.minhashing import minhash_dedup

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
    return predict_quality(text)

def run_gopher_quality_filter(text: str) -> bool:
    return is_high_quality(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    return dedup(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    minhash_dedup(input_files, num_hashes, num_bands, ngrams, jaccard_threshold, output_directory)
