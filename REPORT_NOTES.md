# Report Notes

## Purpose

- RootCause AI addresses customer complaint triage for financial-service complaint narratives.
- The project focuses on practical NLP decision support rather than raw label prediction alone.
- The main goal is to classify complaints, surface similar past cases, estimate confidence, and support reviewer decisions.

## Solution Approach

- Clean raw complaint text and prepare focused training datasets from the larger CFPB-style complaint export.
- Train separate calibrated TF-IDF + Logistic Regression models for product and issue prediction.
- Use a product-aware issue reranking step so final issue predictions are more consistent with the predicted product.
- Retrieve similar historical complaints with TF-IDF cosine similarity to improve interpretability.
- Blend model confidence, compatibility, and retrieval similarity into a final confidence band.
- Optionally generate analyst-facing summaries with OpenAI while preserving a graceful offline fallback.

## Technical Implementation & Architecture

- Central path/config management lives in `src/config.py`.
- Dataset preparation is handled by `src/prepare_dataset.py`.
- Preprocessing is handled by `src/preprocess.py`.
- Model training is orchestrated by `src/train_model.py`.
- Evaluation is written to `models/metrics.json` and `models/metadata.json`.
- Inference is handled by `src/predict.py`.
- Similar-case retrieval is handled by `src/retrieve_similar.py`.
- UI is provided by `app/app.py`.
- API is provided by `api/app.py`.
- Architecture artifact is saved in `reports/ARCHITECTURE.md` and `reports/architecture.mmd`.

## Core Functionalities

- End-to-end dataset preparation, training, evaluation, and inference.
- Product classification.
- Issue classification.
- Product-aware issue refinement.
- Similar complaint retrieval.
- Confidence-aware decision logic with low/medium/high bands.
- Streamlit-based interactive interface.
- FastAPI REST interface with Swagger docs.
- Optional LLM-generated summary, explanation, urgency, and recommended action.

## Result Analysis

- Product accuracy is stronger than issue accuracy because the issue label space is larger and noisier.
- Joint accuracy is a better measure of end-to-end usefulness than issue accuracy alone.
- The high-confidence band performs much better than the low-confidence band, which supports the confidence-aware design.
- Saved metrics and charts:
- `models/metrics.json`
- `reports/metrics_summary.md`
- `reports/assets/accuracy_comparison.png`
- `reports/assets/confidence_band_analysis.png`
- `reports/assets/product_confusion_matrix.png`

## Limitations

- Retrieval is lexical TF-IDF rather than semantic embedding search.
- Issue classification remains challenging because of label noise and class imbalance.
- OpenAI enrichment depends on API availability and is intentionally optional.
- The system is a reviewer-assist tool, not a replacement for human decision-making.

## Future Work

- Replace lexical retrieval with embedding-based retrieval.
- Explore transformer-based classifiers for the issue prediction task.
- Add human feedback loops for continuous improvement.
- Add more robust experiment tracking and automated test coverage.
- Add optional containerized demo data bootstrapping for faster evaluation.
