# Project architecture

Cette page résume le pipeline technique. Les détails du projet et les résultats sont dans le `README.md`.

```text
Raw Sephora data
        │
        ▼
Data audit / EDA
        │
        ▼
Cleaning, deduplication and segmentation
        │
        ▼
9-aspect ABSA taxonomy + structured output contract
        │
        ▼
LLM benchmark
Qwen / Llama / Nemotron / GPT-OSS
        │
        ▼
GPT-OSS 20B annotation
2,000 sampled segments
batching + retries + checkpoints
        │
        ├──────────────► Business aggregation
        │                category / brand / product / skin type
        │
        ▼
DistilBERT V1
        │
        ▼
DistilBERT V2
class weighting + tuned thresholds
        │
        ▼
Model evaluation
aspect detection + sentiment + error analysis + speed
        │
        ▼
Consumer insights
        │
        ▼
Cost and scale review
        │
        ▼
Streamlit demo
```

## Main components

### `src/`
Reusable code for:
- data loading;
- text cleaning and segmentation;
- ABSA prompt/output validation;
- LLM inference;
- sampling;
- business aggregation.

### `notebooks/`
Exploration, experiments, model training and evaluation.

### `prompts/`
System prompt used for GPT-OSS 20B ABSA annotation.

### `reports/`
Selected figures and evaluation outputs kept for the portfolio.

### `app/`
Streamlit demo using the saved DistilBERT V2 model.

## Design choices

The full review corpus is not sent directly to GPT-OSS 20B.

The LLM is used on a controlled POC sample to generate pseudo-labels. A smaller DistilBERT model is then trained on those labels to reduce inference cost and latency while keeping the pipeline usable locally.

The reported DistilBERT metrics measure agreement with the GPT-OSS pseudo-labels. No independent human gold dataset is used in the current version.
