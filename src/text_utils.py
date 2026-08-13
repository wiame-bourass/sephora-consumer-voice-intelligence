import html
import re
import unicodedata

import pandas as pd


# ============================================================
# CONTRAST MARKERS
# ============================================================

# Marqueurs relativement fiables d'opposition / concession.
# On évite volontairement :
# yet, while, unfortunately, except, despite
# car ils produisent beaucoup de faux positifs.
CONTRAST_RE = re.compile(
    r"\b(?:but|however|although|though|whereas)\b",
    flags=re.IGNORECASE
)


# ============================================================
# BASIC TEXT NORMALIZATION
# ============================================================

def normalize_whitespace(text):
    """
    Normalisation légère du texte sans modifier son contenu
    linguistique.
    """
    text = "" if pd.isna(text) else str(text)

    # Convertit par exemple &amp; -> &
    text = html.unescape(text)

    # Normalisation Unicode
    text = unicodedata.normalize("NFKC", text)

    # Espaces multiples -> un seul espace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# NORMALIZATION FOR DUPLICATE DETECTION
# ============================================================

def normalize_for_dedup(text):
    """
    Normalisation plus agressive utilisée uniquement
    pour comparer des textes et détecter des doublons.
    """

    text = normalize_whitespace(text).lower()

    # Suppression des URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Suppression de la ponctuation / caractères spéciaux
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Espaces multiples
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# WORD COUNT
# ============================================================

def word_count(text):
    """
    Compte approximativement le nombre de mots.
    """
    return len(
        re.findall(
            r"\b\w+\b",
            normalize_whitespace(text)
        )
    )


# ============================================================
# CONTRAST DETECTION
# ============================================================

def contains_contrast(text):
    """
    Retourne True si le texte contient un marqueur explicite
    d'opposition ou de concession.
    """
    text = normalize_whitespace(text)

    return bool(
        CONTRAST_RE.search(text)
    )


# ============================================================
# SENTENCE SEGMENTATION
# ============================================================

def sentence_split_series(
    df,
    text_col="review_text"
):
    """
    Segmentation pragmatique des reviews en phrases.

    Une phrase peut toujours contenir plusieurs aspects.
    Exemple :
    "Very hydrating but the smell is awful."

    Cette phrase reste un seul segment.
    """

    out = df.copy()

    # Retours ligne -> séparation potentielle de phrases
    out[text_col] = (
        out[text_col]
        .fillna("")
        .astype(str)
        .str.replace(
            r"[\r\n]+",
            ". ",
            regex=True
        )
    )

    # Découpage après . ! ?
    out["segment_text"] = out[text_col].str.split(
        r"(?<=[.!?])\s+",
        regex=True
    )

    # Une ligne par segment
    out = out.explode(
        "segment_text",
        ignore_index=False
    )

    out["segment_text"] = (
        out["segment_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Retirer segments vides
    out = out[
        out["segment_text"].ne("")
    ].copy()

    # Numéro du segment dans la review
    out["segment_index"] = (
        out
        .groupby("review_id")
        .cumcount()
        .astype("int16")
    )

    # ID unique du segment
    out["segment_id"] = (
        out["review_id"].astype(str)
        + "::s"
        + out["segment_index"].astype(str)
    )

    # Nombre de mots du segment
    out["segment_word_count"] = (
        out["segment_text"]
        .str.findall(r"\b\w+\b")
        .str.len()
        .astype("int16")
    )

    # Présence d'un marqueur de contraste
    out["has_contrast_marker"] = (
        out["segment_text"]
        .str.contains(
            CONTRAST_RE,
            regex=True,
            na=False
        )
    )

    return out


# ============================================================
# ASPECT KEYWORD PATTERNS
# ============================================================

def compile_aspect_patterns(
    taxonomy_dict,
    include_watchlist=False
):
    """
    Compile les mots-clés associés aux aspects.
    """

    aspects = dict(
        taxonomy_dict["core_aspects"]
    )

    if include_watchlist:
        aspects.update(
            taxonomy_dict.get(
                "watchlist",
                {}
            )
        )

    patterns = {}

    for aspect_id, meta in aspects.items():

        keywords = sorted(
            meta.get("keywords", []),
            key=len,
            reverse=True
        )

        escaped_keywords = []

        for keyword in keywords:

            token = (
                re.escape(keyword)
                .replace(
                    r"\ ",
                    r"\s+"
                )
            )

            escaped_keywords.append(
                token
            )

        if escaped_keywords:

            pattern = re.compile(
                r"(?i)(?:\b"
                + r"\b|\b".join(
                    escaped_keywords
                )
                + r"\b)"
            )

        else:

            # Regex qui ne matche jamais
            pattern = re.compile(
                r"$^"
            )

        patterns[aspect_id] = pattern

    return patterns


# ============================================================
# CANDIDATE ASPECT DETECTION
# ============================================================

def candidate_aspects(
    text,
    patterns
):
    """
    Renvoie les aspects candidats détectés par mots-clés.

    Attention :
    ce sont des heuristiques EDA / pré-annotation,
    pas des labels gold.
    """

    text = normalize_whitespace(text)

    return [
        aspect
        for aspect, pattern in patterns.items()
        if pattern.search(text)
    ]