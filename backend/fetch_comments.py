"""
YouTube comments fetcher and video ID extractor.

This module uses YouTube Data API v3 to fetch the latest comments for a given
YouTube video URL or video ID.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

#API key
API_KEY: str = "YOUR_API_KEY_HERE"


def extract_video_id(url_or_id: str) -> Optional[str]:
    """
    Extract a YouTube video ID from a full URL or return the ID if already provided.

    Supports common URL formats like:
    - https://www.youtube.com/watch?v=VIDEOID
    - https://youtu.be/VIDEOID
    - https://www.youtube.com/shorts/VIDEOID
    - https://www.youtube.com/live/VIDEOID
    - https://www.youtube.com/embed/VIDEOID

    Returns None if no valid ID is found.
    """
    if not url_or_id:
        return None

    candidate = url_or_id.strip()

    # If the user pasted a raw ID (11 chars, valid charset), accept it directly.
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", candidate):
        return candidate

    patterns = [
        r"(?:v=)([0-9A-Za-z_-]{11})",           # watch?v=VIDEOID
        r"youtu\.be/([0-9A-Za-z_-]{11})",      # youtu.be/VIDEOID
        r"/shorts/([0-9A-Za-z_-]{11})",         # /shorts/VIDEOID
        r"/live/([0-9A-Za-z_-]{11})",           # /live/VIDEOID
        r"/embed/([0-9A-Za-z_-]{11})",          # /embed/VIDEOID
        r"/v/([0-9A-Za-z_-]{11})",              # /v/VIDEOID
        r"/videos/([0-9A-Za-z_-]{11})",         # /videos/VIDEOID
    ]

    for pat in patterns:
        m = re.search(pat, candidate)
        if m:
            return m.group(1)

    m = re.search(r"([0-9A-Za-z_-]{11})", candidate)
    if m:
        return m.group(1)

    return None


def _get_youtube_service():
    """Build and return the YouTube Data API client service."""
    return build("youtube", "v3", developerKey=API_KEY)


def fetch_comments(url_or_id: str, max_comments: int = 200) -> List[Dict]:
    """
    Fetch up to max_comments of the latest top-level comments for the given YouTube video.

    Args:
        url_or_id: A full YouTube URL or raw video ID.
        max_comments: Maximum number of comments to fetch (API returns up to 100 per page).

    Returns:
        A list of dictionaries with keys: comment_id, author, published_at, like_count, text.
    """
    video_id = extract_video_id(url_or_id)
    if not video_id:
        raise ValueError("Could not extract a valid YouTube video ID from the provided input.")

    comments: List[Dict] = []
    service = _get_youtube_service()

    page_token: Optional[str] = None
    fetched = 0

    try:
        while True:
            req = service.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                textFormat="plainText",
                order="time",
                pageToken=page_token,
            )
            resp = req.execute()

            items = resp.get("items", [])
            for it in items:
                snip = it.get("snippet", {})
                top = snip.get("topLevelComment", {}).get("snippet", {})
                if not top:
                    continue
                comments.append(
                    {
                        "comment_id": it.get("id"),
                        "author": top.get("authorDisplayName"),
                        "published_at": top.get("publishedAt"),
                        "like_count": top.get("likeCount", 0),
                        # textOriginal is plain text when textFormat=plainText
                        "text": top.get("textOriginal") or top.get("textDisplay") or "",
                        "video_id": video_id,
                    }
                )
                fetched += 1
                if fetched >= max_comments:
                    break

            if fetched >= max_comments:
                break

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

    except HttpError as e:
        # Return what we have and attach an error marker as a final entry 
        comments.append(
            {
                "comment_id": None,
                "author": None,
                "published_at": None,
                "like_count": None,
                "text": f"YouTube API error: {e}",
                "video_id": video_id,
            }
        )

    return comments
