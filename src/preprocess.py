"""Shared text-cleaning helpers used by both training and inference scripts."""

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download once on import so any standalone script can rely on the resources.
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Normalize complaint text into a lightweight bag-of-words friendly form."""
    # Lowercasing collapses duplicate features such as "Debt" and "debt".
    text = str(text).lower()
    # Strip punctuation and digits because the TF-IDF models are word-based.
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = text.split()
    # Remove common filler words so the models focus on complaint-specific terms.
    words = [w for w in words if w not in stop_words]
    # Lemmatization helps map simple word variants to one base form.
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)


def preprocess_dataframe(df: pd.DataFrame, text_column: str = "complaint_text") -> pd.DataFrame:
    """Return a copy with a cleaned text column expected by model scripts."""
    # Work on a copy so callers do not get accidental in-place mutations.
    df = df.copy()
    df["clean_text"] = df[text_column].apply(clean_text)
    return df
