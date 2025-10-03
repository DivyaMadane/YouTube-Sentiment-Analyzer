"""
Multilingual text cleaning and sentiment analysis utilities.

Adds language detection (langdetect), translation to English (googletrans),
and emoji-aware preprocessing. Performs sentiment using VADER or TextBlob on
English text (translated when necessary). Also prepares text for wordclouds.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect, DetectorFactory
from googletrans import Translator

# Make langdetect deterministic
DetectorFactory.seed = 0

_vader = SentimentIntensityAnalyzer()
_translator = Translator()

# Simple emoji sentiment hints to influence VADER/TextBlob
_EMOJI_SENTIMENT_MAP = {
    "😀": " positive ", "😃": " positive ", "😄": " positive ", "😁": " positive ",
    "😊": " positive ", "🙂": " positive ", "😍": " positive ", "❤️": " positive ",
    "👍": " positive ", "🔥": " positive ", "⭐": " positive ", "😂": " positive ", "🤣": " positive ", "😆": " positive ",
    "😭": " negative ", "😢": " negative ", "😡": " negative ", "😠": " negative ",
    "👎": " negative ", "💔": " negative ", "🤮": " negative ", "🤬": " negative ",
    "😐": " neutral ",  "😑": " neutral ",  "😶": " neutral ",
}


def _replace_emojis(text: str) -> str:
    if not isinstance(text, str):
        return ""
    out = []
    for ch in text:
        out.append(_EMOJI_SENTIMENT_MAP.get(ch, ch))
    return "".join(out)


def clean_text(text: str) -> str:
    """Conservative cleaning for wordclouds: lowercase, remove URLs/handles, strip non-alphanum except spaces."""
    if not isinstance(text, str):
        return ""
    s = text.lower()
    # Remove URLs
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    # Remove mentions/hashtags
    s = re.sub(r"[@#]\w+", " ", s)
    # Remove non-alphanumeric (keep spaces)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def prepare_for_sentiment(text: str) -> str:
    """Preprocessing for sentiment: keep punctuation and emojis, remove URLs and handles."""
    if not isinstance(text, str):
        return ""
    s = text.lower()
    # Remove URLs and common artifacts
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"[@#]\w+", " ", s)
    # Replace emojis with sentiment hints (keeps the emoji effect)
    s = _replace_emojis(s)
    # Remove control characters
    s = re.sub(r"[\r\n\t]", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def label_from_compound(score: float) -> str:
    """Map VADER/TextBlob-like compound score to Positive/Neutral/Negative."""
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"


def detect_language(text: str) -> str:
    """Detect language using langdetect; returns ISO 639-1 code like 'en', or 'und' if unknown."""
    try:
        lang = detect(text) if isinstance(text, str) and text.strip() else "und"
        return lang or "und"
    except Exception:
        return "und"


def translate_to_english(text: str, src_lang: str) -> Tuple[str, bool]:
    """Translate to English using googletrans; returns (translated_text, had_error)."""
    if not isinstance(text, str) or not text.strip():
        return "", False
    if src_lang.lower() in ("en", "und"):
        return text, False
    try:
        res = _translator.translate(text, src=src_lang, dest="en")
        return res.text, False
    except Exception:
        return text, True


def analyze_sentiment_text(text_en: str, method: str = "vader") -> Tuple[float, str]:
    """Analyze sentiment on English text (already translated if needed)."""
    prepared = prepare_for_sentiment(text_en)
    if not prepared:
        return 0.0, "Neutral"
    m = method.lower()
    if m == "vader":
        scores = _vader.polarity_scores(prepared)
        c = float(scores.get("compound", 0.0))
        return c, label_from_compound(c)
    elif m == "textblob":
        polarity = float(TextBlob(prepared).sentiment.polarity)
        return polarity, label_from_compound(polarity)
    else:
        scores = _vader.polarity_scores(prepared)
        c = float(scores.get("compound", 0.0))
        return c, label_from_compound(c)


def analyze_comments_to_df(comments: List[Dict], method: str = "vader") -> pd.DataFrame:
    """
    Convert a list of comment dicts into a DataFrame and annotate sentiment.

    Adds: language, translated_text (English), analysis_text (preprocessed), wc_text (for wordclouds),
    sentiment_score, sentiment.

    Expected comment keys: comment_id, author, published_at, like_count, text, video_id
    """
    base_cols = [
        "comment_id",
        "author",
        "published_at",
        "like_count",
        "text",
        "video_id",
    ]

    if not comments:
        return pd.DataFrame(
            columns=base_cols
            + ["language", "translated_text", "analysis_text", "wc_text", "sentiment_score", "sentiment"]
        )

    df = pd.DataFrame(comments)

    # Ensure required columns exist
    for col in base_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Language detection and translation
    df["language"] = df["text"].astype(str).apply(detect_language)

    def _translate_row(txt: str, lang: str) -> Tuple[str, bool]:
        return translate_to_english(txt, lang)

    trans = df.apply(lambda r: _translate_row(str(r.get("text", "")), str(r.get("language", "und"))), axis=1)
    df["translated_text"] = trans.apply(lambda x: x[0])
    df["translation_error"] = trans.apply(lambda x: x[1])

    # Analysis text (English) and wordcloud text
    df["analysis_text"] = df["translated_text"].astype(str).apply(prepare_for_sentiment)
    # For wordclouds, use a more aggressive clean on translated (English) text so clouds are meaningful
    df["wc_text"] = df["translated_text"].astype(str).apply(clean_text)

    # Sentiment on English text
    scores_labels = df["analysis_text"].apply(lambda t: analyze_sentiment_text(t, method))
    df["sentiment_score"] = scores_labels.apply(lambda x: x[0])
    df["sentiment"] = scores_labels.apply(lambda x: x[1])

    # Parse published_at to datetime when present
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

    # Sort newest first when possible
    if df["published_at"].notna().any():
        df = df.sort_values(by=["published_at"], ascending=False)

    return df.reset_index(drop=True)
