"""Generate submission-ready charts, tables, and architecture artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from src.artifacts import load_artifacts
from src.config import TrainingConfig, get_project_paths
from src.preprocess import preprocess_dataframe


def _ensure_report_directories() -> tuple[Path, Path]:
    """Create report directories if they do not already exist."""
    paths = get_project_paths()
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_assets_dir.mkdir(parents=True, exist_ok=True)
    return paths.reports_dir, paths.reports_assets_dir


def _load_metrics() -> dict:
    """Load the persisted metrics file written by evaluation."""
    paths = get_project_paths()
    if not paths.metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found at {paths.metrics_path}. Run `python -m src.train_model` first."
        )
    with paths.metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_metrics_summary(metrics: dict, reports_dir: Path) -> list[dict[str, str]]:
    """Write a compact metrics comparison table in JSON, CSV, and Markdown."""
    high_band_accuracy = metrics["end_to_end"]["confidence_band_summary"].get("High", {}).get("joint_accuracy", 0.0)
    rows = [
        {"metric": "Product Accuracy", "value": f"{metrics['product_model']['accuracy']:.4f}"},
        {"metric": "Issue Accuracy", "value": f"{metrics['issue_model']['accuracy']:.4f}"},
        {"metric": "Joint Accuracy", "value": f"{metrics['end_to_end']['joint_accuracy']:.4f}"},
        {"metric": "High-Confidence Accuracy", "value": f"{high_band_accuracy:.4f}"},
    ]

    json_path = reports_dir / "metrics_summary.json"
    csv_path = reports_dir / "metrics_summary.csv"
    md_path = reports_dir / "metrics_summary.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Metric | Value |",
        "|---|---:|",
        *[f"| {row['metric']} | {row['value']} |" for row in rows],
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def _plot_accuracy_comparison(rows: list[dict[str, str]], assets_dir: Path) -> Path:
    """Create a bar chart comparing the key submission metrics."""
    labels = [row["metric"] for row in rows]
    values = [float(row["value"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2E86AB", "#F18F01", "#C73E1D", "#2A9D8F"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("RootCause AI Model Accuracy Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    plt.xticks(rotation=15, ha="right")

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")

    output_path = assets_dir / "accuracy_comparison.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _plot_confidence_distribution(metrics: dict, assets_dir: Path) -> Path:
    """Create a combined chart of confidence-band volume and accuracy."""
    summary = metrics["end_to_end"]["confidence_band_summary"]
    bands = ["Low", "Medium", "High"]
    counts = [summary[band]["count"] for band in bands]
    accuracies = [summary[band]["joint_accuracy"] for band in bands]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    count_bars = ax1.bar(bands, counts, color=["#D7263D", "#F4A261", "#2A9D8F"], alpha=0.85)
    ax1.set_ylabel("Prediction Count")
    ax1.set_title("Confidence Band Distribution and Joint Accuracy")
    ax1.grid(axis="y", linestyle="--", alpha=0.25)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    ax2.plot(bands, accuracies, color="#264653", marker="o", linewidth=2)
    ax2.set_ylabel("Joint Accuracy")
    ax2.set_ylim(0, 1)

    for bar, count in zip(count_bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, count + max(counts) * 0.015, str(count), ha="center")
    for band, accuracy in zip(bands, accuracies):
        ax2.text(band, accuracy + 0.03, f"{accuracy:.3f}", ha="center", color="#264653")

    output_path = assets_dir / "confidence_band_analysis.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _plot_product_confusion_matrix(assets_dir: Path) -> Path:
    """Generate a confusion matrix image for the product classifier."""
    paths = get_project_paths()
    config = TrainingConfig()
    artifacts = load_artifacts()

    df = pd.read_csv(paths.product_dataset)
    processed_df = preprocess_dataframe(df, text_column="complaint_text")
    x_all = artifacts.product_vectorizer.transform(processed_df["clean_text"])
    y_all = processed_df["product"]

    _, x_test, _, y_test = train_test_split(
        x_all,
        y_all,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y_all,
    )

    predictions = artifacts.product_model.predict(x_test)
    labels = list(artifacts.product_model.classes_)
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Product Model Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    output_path = assets_dir / "product_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def _write_architecture_artifacts(reports_dir: Path) -> tuple[Path, Path]:
    """Write Mermaid and Markdown architecture artifacts for reports."""
    mermaid = """flowchart LR
    A[Input Complaint]
    B[Preprocessing]
    C[Product Model]
    D[Issue Model]
    E[Retrieval System]
    F[Confidence Engine]
    G[LLM Insight Layer]
    H[Final Output]
    I[Streamlit UI]
    J[FastAPI API]

    A --> B
    B --> C
    B --> D
    B --> E
    C --> F
    D --> F
    E --> F
    F --> G
    F --> H
    G --> H
    H --> I
    H --> J
"""
    markdown = "\n".join(
        [
            "# Architecture Diagram",
            "",
            "```mermaid",
            mermaid.rstrip(),
            "```",
            "",
            "This diagram can be pasted directly into GitHub Markdown, Mermaid Live Editor, or most report tooling.",
            "",
        ]
    )

    mermaid_path = reports_dir / "architecture.mmd"
    markdown_path = reports_dir / "ARCHITECTURE.md"
    mermaid_path.write_text(mermaid, encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    return mermaid_path, markdown_path


def generate_report_artifacts() -> dict[str, str]:
    """Create deterministic report and presentation support artifacts."""
    reports_dir, assets_dir = _ensure_report_directories()
    metrics = _load_metrics()
    rows = _write_metrics_summary(metrics, reports_dir)
    accuracy_chart = _plot_accuracy_comparison(rows, assets_dir)
    confidence_chart = _plot_confidence_distribution(metrics, assets_dir)
    confusion_matrix_path = _plot_product_confusion_matrix(assets_dir)
    mermaid_path, architecture_md_path = _write_architecture_artifacts(reports_dir)

    outputs = {
        "metrics_summary_json": str(reports_dir / "metrics_summary.json"),
        "metrics_summary_csv": str(reports_dir / "metrics_summary.csv"),
        "metrics_summary_markdown": str(reports_dir / "metrics_summary.md"),
        "accuracy_chart": str(accuracy_chart),
        "confidence_chart": str(confidence_chart),
        "product_confusion_matrix": str(confusion_matrix_path),
        "architecture_mermaid": str(mermaid_path),
        "architecture_markdown": str(architecture_md_path),
    }
    print(json.dumps(outputs, indent=2))
    return outputs


if __name__ == "__main__":
    generate_report_artifacts()
