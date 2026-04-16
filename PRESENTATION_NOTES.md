# Presentation Notes

## 5-Slide Outline

### Slide 1: Problem and Goal

- Introduce complaint triage as an NLP problem with noisy text and many labels.
- Explain that the project goal is not just classification, but confidence-aware decision support.

### Slide 2: Solution Architecture

- Show `reports/ARCHITECTURE.md`.
- Explain the flow:
- input complaint
- preprocessing
- product and issue models
- retrieval system
- confidence engine
- optional LLM insights
- final output in Streamlit or API

### Slide 3: Implementation Highlights

- Mention calibrated TF-IDF + Logistic Regression models.
- Mention product-aware issue reranking.
- Mention similar historical complaint retrieval.
- Mention optional OpenAI enrichment with safe fallback.

### Slide 4: Results

- Show `reports/assets/accuracy_comparison.png`.
- Show `reports/assets/confidence_band_analysis.png`.
- Explain that high-confidence predictions are much more reliable than low-confidence predictions.
- Mention current key metrics:
- product accuracy `0.7179`
- issue accuracy `0.3762`
- joint accuracy `0.3986`
- high-confidence joint accuracy `0.7111`

### Slide 5: Demo and Takeaways

- Show Streamlit running on one complaint.
- Optionally show FastAPI Swagger docs.
- Conclude that the project is an end-to-end, reproducible NLP solution with strong software-engineering structure.

## Live Demo Sequence

1. Show `README.md` and the one-command training pipeline.
2. Show `reports/ARCHITECTURE.md`.
3. Run a CLI prediction with `python -m src.predict --text "..." --skip-llm`.
4. Open Streamlit and analyze a complaint interactively.
5. Open `/docs` in the FastAPI app and show the REST endpoint.

## What to Say During the Demo

- “We separated product and issue prediction because they operate at different granularities.”
- “We added retrieval so the prediction is easier to interpret.”
- “We use confidence bands to decide when automation should be trusted.”
- “The OpenAI layer is optional and never blocks the core ML pipeline.”

## Expected Q&A

### Why is issue accuracy lower than product accuracy?

- The issue label space is much larger and more imbalanced, so it is a harder classification problem.

### Why use TF-IDF instead of transformers?

- TF-IDF + Logistic Regression is a strong, interpretable baseline and easier to train reproducibly for a capstone submission.

### What happens if OpenAI is unavailable?

- The app still works. It returns a documented fallback response and preserves the classification output.

### Why include retrieval?

- Retrieval improves interpretability by showing similar historical complaints that support or challenge the prediction.

### What is the main engineering improvement?

- Centralized config, reproducible training/evaluation, and report-ready artifacts that make the solution easier to validate and present.
