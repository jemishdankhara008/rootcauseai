"""Prepare filtered training datasets from the raw complaint export."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from src.config import TrainingConfig, get_project_paths


@dataclass(frozen=True)
class PreparedDatasets:
    """Prepared training splits persisted to disk for downstream stages."""

    cleaned_rows: int
    product_rows: int
    issue_rows: int
    product_label_count: int
    issue_label_count: int


def _clean_raw_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Select and normalize the columns required by the project."""
    cleaned = df[
        ["Consumer complaint narrative", "Issue", "Product"]
    ].dropna()

    cleaned = cleaned.rename(
        columns={
            "Consumer complaint narrative": "complaint_text",
            "Issue": "issue",
            "Product": "product",
        }
    )

    for column in ("complaint_text", "issue", "product"):
        cleaned = cleaned[cleaned[column].astype(str).str.strip() != ""]

    return cleaned


def prepare_datasets(config: TrainingConfig | None = None) -> PreparedDatasets:
    """Create product and issue training datasets from the raw export."""
    config = config or TrainingConfig()
    paths = get_project_paths()
    paths.data_dir.mkdir(parents=True, exist_ok=True)

    if not paths.raw_dataset.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {paths.raw_dataset}. "
            "Place the Kaggle complaint export there before training."
        )

    raw_df = pd.read_csv(paths.raw_dataset, low_memory=False)
    cleaned_df = _clean_raw_dataset(raw_df)

    top_products = cleaned_df["product"].value_counts().head(config.top_product_count).index
    product_df = cleaned_df[cleaned_df["product"].isin(top_products)].copy()

    product_sample_size = min(config.product_sample_size, len(product_df))
    product_df = product_df.sample(n=product_sample_size, random_state=config.random_state)

    issue_counts = product_df["issue"].value_counts()
    supported_issues = issue_counts[issue_counts >= config.min_issue_frequency].index
    issue_df = product_df[product_df["issue"].isin(supported_issues)].copy()

    issue_sample_size = min(config.issue_sample_size, len(issue_df))
    issue_df = issue_df.sample(n=issue_sample_size, random_state=config.random_state)

    cleaned_output = issue_df.copy()

    product_df.to_csv(paths.product_dataset, index=False)
    issue_df.to_csv(paths.issue_dataset, index=False)
    cleaned_output.to_csv(paths.cleaned_dataset, index=False)

    summary = PreparedDatasets(
        cleaned_rows=len(cleaned_df),
        product_rows=len(product_df),
        issue_rows=len(issue_df),
        product_label_count=product_df["product"].nunique(),
        issue_label_count=issue_df["issue"].nunique(),
    )

    print("Prepared datasets:")
    print(asdict(summary))
    print(f"Saved product dataset to {paths.product_dataset}")
    print(f"Saved issue dataset to {paths.issue_dataset}")
    print(f"Saved compatibility dataset to {paths.cleaned_dataset}")
    return summary


if __name__ == "__main__":
    prepare_datasets()
