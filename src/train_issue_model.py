"""Train the issue classifier used by the inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.config import TrainingConfig, get_project_paths
from src.preprocess import preprocess_dataframe


@dataclass
class IssueTrainingArtifacts:
    """Trained issue model, vectorizer, evaluation slice, and compatibility map."""

    model: CalibratedClassifierCV
    vectorizer: TfidfVectorizer
    processed_df: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    metrics: dict
    compatibility_map: dict[str, dict[str, float]]


def _build_compatibility_map(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Create normalized issue priors per product from the training data."""
    compatibility: dict[str, dict[str, float]] = {}
    grouped = df.groupby(["product", "issue"]).size().reset_index(name="count")
    for product, product_rows in grouped.groupby("product"):
        total = float(product_rows["count"].sum())
        compatibility[product] = {
            row["issue"]: round(float(row["count"] / total), 6)
            for _, row in product_rows.iterrows()
        }
    return compatibility


def train_issue_model(config: TrainingConfig | None = None) -> IssueTrainingArtifacts:
    """Train and persist the calibrated issue classifier."""
    config = config or TrainingConfig()
    paths = get_project_paths()
    paths.models_dir.mkdir(parents=True, exist_ok=True)

    if not paths.issue_dataset.exists():
        raise FileNotFoundError(
            f"Missing {paths.issue_dataset}. Run `python -m src.prepare_dataset` first."
        )

    df = pd.read_csv(paths.issue_dataset)
    processed_df = preprocess_dataframe(df, text_column="complaint_text")
    vectorizer = TfidfVectorizer(
        max_features=config.issue_max_features,
        ngram_range=(config.ngram_min, config.ngram_max),
        min_df=config.min_document_frequency,
    )

    x_all = vectorizer.fit_transform(processed_df["clean_text"])
    y_all = processed_df["issue"]
    indices = processed_df.index.to_numpy()

    x_train, x_test, y_train, y_test, train_indices, test_indices = train_test_split(
        x_all,
        y_all,
        indices,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y_all,
    )

    base_model = LogisticRegression(
        max_iter=config.logistic_max_iter,
        class_weight="balanced",
    )
    model = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=3)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "label_count": int(y_all.nunique()),
        "train_rows": int(x_train.shape[0]),
        "test_rows": int(x_test.shape[0]),
    }

    joblib.dump(model, paths.issue_model)
    joblib.dump(vectorizer, paths.issue_vectorizer)

    train_df = processed_df.loc[train_indices].reset_index(drop=True)
    test_df = processed_df.loc[test_indices].reset_index(drop=True)
    compatibility_map = _build_compatibility_map(train_df)

    print("Issue model metrics:")
    print(metrics["classification_report"])

    return IssueTrainingArtifacts(
        model=model,
        vectorizer=vectorizer,
        processed_df=processed_df,
        train_df=train_df,
        test_df=test_df,
        metrics=metrics,
        compatibility_map=compatibility_map,
    )


if __name__ == "__main__":
    train_issue_model()
