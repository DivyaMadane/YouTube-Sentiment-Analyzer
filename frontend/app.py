from __future__ import annotations

import base64
import io
import os
import sys
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, Response, redirect, render_template, request, url_for, jsonify
from wordcloud import STOPWORDS, WordCloud

# Ensure the project root (one level up from this file) is on sys.path so we can import backend/*
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.fetch_comments import extract_video_id, fetch_comments
from backend.sentiment_analysis import analyze_comments_to_df

app = Flask(__name__, template_folder="templates")


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def _gen_wordcloud_b64(text: str, stopwords: set) -> str:
    # Optimized wordcloud settings for speed
    wc = WordCloud(
        width=400, height=200,  # Reduced size for faster generation
        background_color="white", 
        stopwords=stopwords,
        max_words=50,  # Limit words for faster processing
        relative_scaling=0.5,
        min_font_size=10
    )
    img = wc.generate(text)
    fig, ax = plt.subplots(figsize=(6, 3), dpi=72)  # Smaller figure, lower DPI
    ax.imshow(img, interpolation="nearest")  # Faster interpolation
    ax.axis("off")
    return _fig_to_base64(fig)


@app.route("/", methods=["GET"]) 
def index():
    # Initial page load - just render the form
    return render_template("index.html")


@app.route("/analyze", methods=["POST"]) 
def analyze():
    """Handle POST requests for sentiment analysis and return JSON results."""
    print("[DEBUG] Analyze endpoint called")  # Debug log
    
    try:
        data = request.get_json()
        print(f"[DEBUG] Received data: {data}")  # Debug log
        
        video_input = (data.get("video_input") or "").strip()
        method = (data.get("method") or "vader").lower()
        if method not in {"vader", "textblob"}:
            method = "vader"
        try:
            max_comments = int(data.get("max_comments", 100))  # Reduced default from 300 to 100
        except (ValueError, TypeError):
            max_comments = 100
        max_comments = max(25, min(500, max_comments))  # Reduced max from 1000 to 500
        
        print(f"[DEBUG] Processing: video={video_input}, method={method}, max_comments={max_comments}")
        
        if not video_input:
            print("[DEBUG] No video URL provided")
            return jsonify({"error": "No video URL provided"}), 400
        
        # Extract video ID
        video_id = extract_video_id(video_input)
        if not video_id:
            print(f"[DEBUG] Could not extract video ID from: {video_input}")
            return jsonify({"error": "Could not extract a valid YouTube video ID from the provided input."}), 400
        
        print(f"[DEBUG] Extracted video ID: {video_id}")
        
        # Fetch comments and analyze sentiment
        raw_comments = fetch_comments(video_input, max_comments=max_comments)
        print(f"[DEBUG] Fetched {len(raw_comments)} raw comments")
        
        df = analyze_comments_to_df(raw_comments, method=method)
        print(f"[DEBUG] Analyzed comments, DataFrame shape: {df.shape}")
        
        if df.empty:
            print("[DEBUG] DataFrame is empty - no comments found")
            return jsonify({"error": "No comments found or failed to fetch comments."}), 400
        
        # Calculate statistics
        counts = (
            df["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0).astype(int)
        )
        stats = {
            "total": int(df.shape[0]),
            "positive": int(counts.get("Positive", 0)),
            "neutral": int(counts.get("Neutral", 0)),
            "negative": int(counts.get("Negative", 0)),
        }
        
        print(f"[DEBUG] Stats: {stats}")
        
        # Generate charts
        charts = {}
        
        # Pie chart - optimized size and settings
        fig, ax = plt.subplots(figsize=(4, 4), dpi=72)  # Reduced size and DPI for speed
        labels = ["Positive", "Neutral", "Negative"]
        sizes = [counts.get(l, 0) for l in labels]
        colors = ["#2ecc71", "#95a5a6", "#e74c3c"]
        ax.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=140, colors=colors)  # Reduced decimal precision
        ax.axis("equal")
        charts["pie"] = _fig_to_base64(fig)
        
        # Bar chart - optimized size
        fig, ax = plt.subplots(figsize=(5, 3), dpi=72)  # Reduced size and DPI for speed
        ax.bar(labels, sizes, color=["#2ecc71", "#95a5a6", "#e74c3c"])  # Use simple bar instead of seaborn
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        ax.set_title("Sentiment Distribution")
        charts["bar"] = _fig_to_base64(fig)
        
        # WordClouds - only generate if sufficient data
        stopwords = set(STOPWORDS)
        stopwords.update({"https", "http", "www", "youtube", "video"})
        pos_text = " ".join(df.loc[df["sentiment"] == "Positive", "wc_text"].tolist())
        neg_text = " ".join(df.loc[df["sentiment"] == "Negative", "wc_text"].tolist())
        
        # Only generate wordclouds if enough text (minimum 20 words for better performance)
        if pos_text.strip() and len(pos_text.split()) >= 20:
            charts["wc_pos"] = _gen_wordcloud_b64(pos_text, stopwords)
        if neg_text.strip() and len(neg_text.split()) >= 20:
            charts["wc_neg"] = _gen_wordcloud_b64(neg_text, stopwords)
        
        # Table data (limit 100 rows for faster processing)
        display_cols = ["published_at", "author", "language", "like_count", "text", "translated_text", "sentiment", "sentiment_score"]
        for c in display_cols:
            if c not in df.columns:
                df[c] = np.nan
        table_records = (
            df[display_cols].fillna("").head(100).to_dict(orient="records")
        )
        
        # Convert datetime objects to strings and handle NaN values for JSON serialization
        for record in table_records:
            if pd.notna(record.get("published_at")):
                record["published_at"] = str(record["published_at"])
            # Replace any NaN values with empty strings or appropriate defaults
            for key, value in record.items():
                if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
                    if key == "like_count":
                        record[key] = 0
                    else:
                        record[key] = ""
        
        result = {
            "success": True,
            "video_id": video_id,
            "stats": stats,
            "charts": charts,
            "table_records": table_records
        }
        
        print(f"[DEBUG] Returning successful result with {len(table_records)} table records")
        print(f"[DEBUG] Result keys: {list(result.keys())}")
        print(f"[DEBUG] Charts keys: {list(result['charts'].keys())}")
        return jsonify(result)
        
    except Exception as e:
        print(f"[DEBUG] Error in analyze endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/download.csv", methods=["GET"]) 
def download_csv():
    video_input = (request.args.get("video_input") or "").strip()
    method = (request.args.get("method") or "vader").lower()
    try:
        max_comments = int(request.args.get("max_comments", 300))
    except ValueError:
        max_comments = 300
    max_comments = max(50, min(1000, max_comments))

    raw_comments = fetch_comments(video_input, max_comments=max_comments)
    df = analyze_comments_to_df(raw_comments, method=method)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    vid = extract_video_id(video_input) or "video"
    return Response(
        csv_bytes,
        headers={
            "Content-Disposition": f"attachment; filename=sentiment_{vid}.csv",
            "Content-Type": "text/csv",
        },
    )


if __name__ == "__main__":
    # Run Flask dev server
    app.run(host="127.0.0.1", port=5000, debug=True)
