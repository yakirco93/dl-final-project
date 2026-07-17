"""Thumbnail image download + local caching.

This ports the download logic already validated during the diagnostic phase
(96.5% success rate, n=995, when run from a non-cloud network). Key lessons
baked in here:

  1. Run this from a local/office network, NOT from Google Colab — mako's
     image CDN blocks Google Cloud IP ranges (~60% failure rate confirmed).
  2. Local disk caching means re-running only fetches what's still missing.
  3. Retries with backoff handle transient failures; failures are classified
     (403/404/429/timeout) so you can tell "blocked" apart from "genuinely
     missing" at scale.

Usage:
    python -m src.data.download_images --csv data/raw/articles.csv \
        --url-column pic_furl --cache-dir data/images/
"""
import argparse
import hashlib
import io
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from PIL import Image


def _cache_path(url: str, cache_dir: str) -> str:
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{h}.jpg")


def _classify_error(e: Exception) -> str:
    msg = str(e)
    if "429" in msg:
        return "rate_limited (429)"
    if "403" in msg:
        return "forbidden (403)"
    if "404" in msg:
        return "not_found (404)"
    if "timeout" in msg.lower():
        return "timeout"
    return "other"


def download_images(urls, cache_dir="data/images/", max_workers=8, max_retries=3):
    """Download+cache a list of image URLs. Returns (n_success, error_counter)."""
    os.makedirs(cache_dir, exist_ok=True)
    error_types = Counter()
    n_success = 0

    def task(url):
        path = _cache_path(url, cache_dir)
        if os.path.exists(path):
            return True, None
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        }
        last_err = None
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, timeout=10, headers=headers)
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                img.save(path, "JPEG")
                return True, None
            except Exception as e:
                last_err = e
                if attempt < max_retries - 1:
                    time.sleep(1.5 * (attempt + 1))
        return False, _classify_error(last_err)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(task, u) for u in urls]
        for i, future in enumerate(as_completed(futures), 1):
            ok, err = future.result()
            if ok:
                n_success += 1
            else:
                error_types[err] += 1
            if i % 100 == 0 or i == len(urls):
                print(f"  {i}/{len(urls)} processed ({n_success} cached, {sum(error_types.values())} failed)")

    if error_types:
        print("\nFailure breakdown:", dict(error_types))
        if error_types.get("forbidden (403)", 0) > 10:
            print("  \u26a0\ufe0f  Many 403s — likely running from a blocked (cloud) network. "
                  "Re-run this script from a local machine instead.")

    return n_success, error_types


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--url-column", default="pic_furl")
    parser.add_argument("--cache-dir", default="data/images/")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    urls = df[args.url_column].dropna().unique().tolist()
    print(f"Downloading {len(urls)} unique images...")
    download_images(urls, cache_dir=args.cache_dir, max_workers=args.max_workers)
