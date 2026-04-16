"""Shared text preprocessing utilities used across the project."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TOKEN_PATTERN = re.compile(r"[a-z]+")
stemmer = PorterStemmer()
stop_words = set(ENGLISH_STOP_WORDS)


def tokenize(text: str) -> list[str]:
    """Normalize text into lowercase alphabetic tokens."""
    normalized = str(text).lower()
    return TOKEN_PATTERN.findall(normalized)


def normalize_tokens(tokens: Iterable[str]) -> list[str]:
    """Remove stop words and stem remaining tokens for sparse text models."""
    return [stemmer.stem(token) for token in tokens if token not in stop_words]


def clean_text(text: str) -> str:
    """Convert complaint text into a compact normalized string."""
    tokens = tokenize(text)
    normalized_tokens = normalize_tokens(tokens)
    return " ".join(normalized_tokens)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = "complaint_text",
) -> pd.DataFrame:
    """Return a copy with cleaned complaint text ready for vectorization."""
    processed = df.copy()
    processed["clean_text"] = processed[text_column].astype(str).map(clean_text)
    return processed
