from pathlib import Path
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="Sephora Consumer Voice Intelligence",
    page_icon="💄",
    layout="wide",
)

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ASPECTS = [
    "efficacy_results",
    "hydration_dryness",
    "texture_finish",
    "irritation_sensitivity",
    "acne_breakouts",
    "fragrance_smell",
    "application_absorption",
    "packaging",
    "price_value",
]

ASPECT_LABELS = {
    "efficacy_results": "Efficacy & results",
    "hydration_dryness": "Hydration / dryness",
    "texture_finish": "Texture & finish",
    "irritation_sensitivity": "Irritation & sensitivity",
    "acne_breakouts": "Acne & breakouts",
    "fragrance_smell": "Fragrance & smell",
    "application_absorption": "Application & absorption",
    "packaging": "Packaging",
    "price_value": "Price & value",
}

SENTIMENT_LABELS = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


# ============================================================
# Project paths
# ============================================================

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent

ARTIFACTS_DIR = ROOT / "artifacts"
BUSINESS_DIR = ROOT / "data" / "processed" / "sample_2000_business"


def find_path(base_dir: Path, name: str, is_dir: bool = False):
    if not base_dir.exists():
        return None

    for root, dirs, files in os.walk(base_dir):
        if is_dir and name in dirs:
            return Path(root) / name
        if not is_dir and name in files:
            return Path(root) / name

    return None


ASPECT_MODEL_DIR = find_path(
    ARTIFACTS_DIR,
    "student_aspect_distilbert_v2",
    is_dir=True,
)

SENTIMENT_MODEL_DIR = find_path(
    ARTIFACTS_DIR,
    "student_sentiment_distilbert_v2",
    is_dir=True,
)

THRESHOLDS_PATH = find_path(
    ARTIFACTS_DIR,
    "aspect_thresholds_v2.csv",
)


# ============================================================
# Loaders
# ============================================================

@st.cache_resource
def load_models():
    if ASPECT_MODEL_DIR is None:
        raise FileNotFoundError(
            "student_aspect_distilbert_v2 introuvable dans artifacts/"
        )

    if SENTIMENT_MODEL_DIR is None:
        raise FileNotFoundError(
            "student_sentiment_distilbert_v2 introuvable dans artifacts/"
        )

    if THRESHOLDS_PATH is None:
        raise FileNotFoundError(
            "aspect_thresholds_v2.csv introuvable dans artifacts/"
        )

    aspect_tokenizer = AutoTokenizer.from_pretrained(ASPECT_MODEL_DIR)
    aspect_model = AutoModelForSequenceClassification.from_pretrained(
        ASPECT_MODEL_DIR
    ).to(DEVICE)
    aspect_model.eval()

    sentiment_tokenizer = AutoTokenizer.from_pretrained(
        SENTIMENT_MODEL_DIR
    )
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL_DIR
    ).to(DEVICE)
    sentiment_model.eval()

    thresholds_df = pd.read_csv(THRESHOLDS_PATH)
    threshold_map = dict(
        zip(
            thresholds_df["aspect_id"],
            thresholds_df["best_threshold"],
        )
    )

    thresholds = np.array(
        [threshold_map[a] for a in ASPECTS],
        dtype=float,
    )

    return (
        aspect_tokenizer,
        aspect_model,
        sentiment_tokenizer,
        sentiment_model,
        thresholds,
    )


@st.cache_data
def load_business_tables():
    tables = {}

    files = {
        "overall": "agg_overall_aspect.parquet",
        "category": "agg_category_aspect.parquet",
        "brand": "agg_brand_aspect.parquet",
        "product": "agg_product_aspect.parquet",
        "skin_type": "agg_skin_type_aspect.parquet",
        "category_long": "category_aspect_sentiment_long.parquet",
    }

    for key, filename in files.items():
        path = BUSINESS_DIR / filename
        if path.exists():
            tables[key] = pd.read_parquet(path)

    return tables


# ============================================================
# Inference
# ============================================================

def tokenize_for_distilbert(tokenizer, texts):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=160,
        return_tensors="pt",
        return_token_type_ids=False,
    )

    # Extra safety for tokenizer/model version differences
    encoded.pop("token_type_ids", None)

    return {
        k: v.to(DEVICE)
        for k, v in encoded.items()
    }


def predict_review(
    text,
    aspect_tokenizer,
    aspect_model,
    sentiment_tokenizer,
    sentiment_model,
    thresholds,
):
    enc = tokenize_for_distilbert(
        aspect_tokenizer,
        [text],
    )

    with torch.no_grad():
        logits = aspect_model(**enc).logits

    probs = (
        torch.sigmoid(logits)
        .cpu()
        .numpy()[0]
    )

    detected = []

    for i, aspect_id in enumerate(ASPECTS):
        detected_flag = probs[i] >= thresholds[i]

        if detected_flag:
            sentiment_text = (
                f"[ASPECT] {aspect_id} "
                f"[TEXT] {text}"
            )

            sent_enc = tokenize_for_distilbert(
                sentiment_tokenizer,
                [sentiment_text],
            )

            with torch.no_grad():
                sent_logits = sentiment_model(
                    **sent_enc
                ).logits

            sent_probs = (
                torch.softmax(
                    sent_logits,
                    dim=-1,
                )
                .cpu()
                .numpy()[0]
            )

            sent_id = int(
                np.argmax(sent_probs)
            )

            detected.append(
                {
                    "aspect_id": aspect_id,
                    "aspect": ASPECT_LABELS[aspect_id],
                    "aspect_probability": float(probs[i]),
                    "threshold": float(thresholds[i]),
                    "sentiment": SENTIMENT_LABELS[sent_id],
                    "sentiment_confidence": float(
                        sent_probs[sent_id]
                    ),
                }
            )

    all_aspects = pd.DataFrame(
        {
            "aspect_id": ASPECTS,
            "aspect": [
                ASPECT_LABELS[a]
                for a in ASPECTS
            ],
            "probability": probs,
            "threshold": thresholds,
        }
    )

    detected_df = pd.DataFrame(detected)

    return detected_df, all_aspects


# ============================================================
# Helpers for business visuals
# ============================================================

def first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def prepare_overall_table(df):
    if df is None or df.empty:
        return None

    x = df.copy()

    if "aspect_id" in x.columns:
        x["aspect"] = x["aspect_id"].map(
            ASPECT_LABELS
        ).fillna(x["aspect_id"])

    return x


def prepare_category_table(df):
    if df is None or df.empty:
        return None

    x = df.copy()

    if "aspect_id" in x.columns:
        x["aspect"] = x["aspect_id"].map(
            ASPECT_LABELS
        ).fillna(x["aspect_id"])

    return x


# ============================================================
# UI
# ============================================================

st.title("💄 Sephora Consumer Voice Intelligence")

st.caption(
    "Aspect-Based Sentiment Analysis powered by a lightweight "
    "DistilBERT V2 trained on GPT-OSS 20B pseudo-labels."
)

tab_analyze, tab_insights = st.tabs(
    [
        "🔎 Analyze a review",
        "📊 Consumer insights",
    ]
)


# ============================================================
# TAB 1 — Individual inference
# ============================================================

with tab_analyze:
    st.subheader("Analyze a customer review")

    default_text = (
        "This moisturizer leaves my skin really hydrated, "
        "but the fragrance is much too strong for me."
    )

    review_text = st.text_area(
        "Customer review",
        value=default_text,
        height=150,
        placeholder="Paste a Sephora review here...",
    )

    analyze = st.button(
        "Analyze review",
        type="primary",
        use_container_width=False,
    )

    if analyze:
        if not review_text.strip():
            st.warning("Enter a review first.")

        else:
            try:
                (
                    aspect_tokenizer,
                    aspect_model,
                    sentiment_tokenizer,
                    sentiment_model,
                    thresholds,
                ) = load_models()

                detected_df, all_aspects = predict_review(
                    review_text.strip(),
                    aspect_tokenizer,
                    aspect_model,
                    sentiment_tokenizer,
                    sentiment_model,
                    thresholds,
                )

                if detected_df.empty:
                    st.info(
                        "No aspect exceeded its V2 decision threshold."
                    )
                else:
                    st.success(
                        f"{len(detected_df)} aspect(s) detected."
                    )

                    display_df = detected_df[
                        [
                            "aspect",
                            "sentiment",
                            "aspect_probability",
                            "sentiment_confidence",
                        ]
                    ].copy()

                    display_df[
                        "aspect_probability"
                    ] = display_df[
                        "aspect_probability"
                    ].round(3)

                    display_df[
                        "sentiment_confidence"
                    ] = display_df[
                        "sentiment_confidence"
                    ].round(3)

                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                st.markdown("#### Aspect probabilities")

                plot_df = all_aspects.sort_values(
                    "probability",
                    ascending=True,
                )

                fig = px.bar(
                    plot_df,
                    x="probability",
                    y="aspect",
                    orientation="h",
                    hover_data=["threshold"],
                    labels={
                        "probability": "Probability",
                        "aspect": "",
                    },
                )

                fig.add_scatter(
                    x=plot_df["threshold"],
                    y=plot_df["aspect"],
                    mode="markers",
                    name="Decision threshold",
                )

                fig.update_layout(
                    height=470,
                    xaxis_range=[0, 1],
                    legend_title_text="",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

                st.caption(
                    "The probability is the DistilBERT model output. "
                    "The threshold is aspect-specific and was selected "
                    "on validation data."
                )

            except Exception as exc:
                st.error(
                    "Model loading or inference failed."
                )
                st.exception(exc)


# ============================================================
# TAB 2 — Business insights
# ============================================================

with tab_insights:
    st.subheader("Consumer insights from the 2,000-segment POC sample")

    tables = load_business_tables()

    if not tables:
        st.warning(
            "No processed business aggregation found in "
            "`data/processed/sample_2000_business/`."
        )

    else:
        overall = prepare_overall_table(
            tables.get("overall")
        )
        category = prepare_category_table(
            tables.get("category")
        )

        # ----------------------------------------------------
        # 1. Most-mentioned aspects
        # ----------------------------------------------------
        if overall is not None:
            mentions_col = first_existing_column(
                overall,
                [
                    "weighted_mentions",
                    "n_mentions_weighted",
                    "n_mentions",
                ],
            )

            if mentions_col is not None:
                st.markdown(
                    "### Most discussed aspects"
                )

                top_mentions = (
                    overall
                    .sort_values(
                        mentions_col,
                        ascending=True,
                    )
                )

                fig = px.bar(
                    top_mentions,
                    x=mentions_col,
                    y="aspect",
                    orientation="h",
                    labels={
                        mentions_col: "Mentions",
                        "aspect": "",
                    },
                )

                fig.update_layout(
                    height=430
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        # ----------------------------------------------------
        # 2. Positive vs negative rates
        # ----------------------------------------------------
        if overall is not None:
            if {
                "positive_rate",
                "negative_rate",
            }.issubset(
                overall.columns
            ):
                st.markdown(
                    "### Positive vs negative sentiment by aspect"
                )

                sentiment_plot = overall[
                    [
                        "aspect",
                        "positive_rate",
                        "negative_rate",
                    ]
                ].melt(
                    id_vars="aspect",
                    var_name="sentiment",
                    value_name="rate",
                )

                sentiment_plot[
                    "sentiment"
                ] = sentiment_plot[
                    "sentiment"
                ].replace(
                    {
                        "positive_rate": "Positive",
                        "negative_rate": "Negative",
                    }
                )

                fig = px.bar(
                    sentiment_plot,
                    x="aspect",
                    y="rate",
                    color="sentiment",
                    barmode="group",
                    labels={
                        "rate": "Weighted rate",
                        "aspect": "",
                        "sentiment": "",
                    },
                )

                fig.update_layout(
                    xaxis_tickangle=-35,
                    height=500,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        # ----------------------------------------------------
        # 3. Category comparison
        # ----------------------------------------------------
        if category is not None:
            category_col = first_existing_column(
                category,
                [
                    "secondary_category",
                    "category",
                ],
            )

            if category_col is not None:
                st.markdown(
                    "### Compare core skincare categories"
                )

                available_categories = sorted(
                    category[
                        category_col
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                selected_categories = st.multiselect(
                    "Categories",
                    options=available_categories,
                    default=available_categories[:3],
                )

                category_filtered = category[
                    category[
                        category_col
                    ]
                    .astype(str)
                    .isin(selected_categories)
                ].copy()

                metric_col = first_existing_column(
                    category_filtered,
                    [
                        "net_sentiment",
                        "positive_rate",
                        "negative_rate",
                    ],
                )

                if (
                    metric_col is not None
                    and not category_filtered.empty
                ):
                    fig = px.bar(
                        category_filtered,
                        x="aspect",
                        y=metric_col,
                        color=category_col,
                        barmode="group",
                        labels={
                            "aspect": "",
                            metric_col: metric_col.replace(
                                "_",
                                " ",
                            ).title(),
                            category_col: "Category",
                        },
                    )

                    fig.update_layout(
                        xaxis_tickangle=-35,
                        height=520,
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

        # ----------------------------------------------------
        # 4. Main irritants
        # ----------------------------------------------------
        if overall is not None and "negative_rate" in overall.columns:
            st.markdown(
                "### Main consumer irritants"
            )

            irritants = (
                overall
                .sort_values(
                    "negative_rate",
                    ascending=False,
                )
                .head(6)
                .sort_values(
                    "negative_rate",
                    ascending=True,
                )
            )

            fig = px.bar(
                irritants,
                x="negative_rate",
                y="aspect",
                orientation="h",
                labels={
                    "negative_rate": "Weighted negative rate",
                    "aspect": "",
                },
            )

            fig.update_layout(
                height=380
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    st.caption(
        "Business rates are based on the POC sample and its "
        "post-stratification weights; they are not full-population "
        "Sephora estimates."
    )


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("Scope")

    st.write(
        "Core skincare categories: "
        "Moisturizers, Treatments & Cleansers."
    )