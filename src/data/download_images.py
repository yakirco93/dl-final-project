"""Thumbnail image download + local caching.

This ports the download logic already validated during the diagnostic phase
(96.5% success rate, n=995, when run from a non-cloud network). Key lessons
baked in here:

  1. Run this from a local/office network, NOT from Google Colab -- mako's
     image CDN blocks Google Cloud IP ranges (~60% failure rate confirmed).
  2. Local disk caching means re-running only fetches what's still missing.
  3. Retries with backoff handle transient failures; failures are classified
     (403/404/429/timeout) so you can tell "blocked" apart from "genuinely
     missing" at scale.
  4. A full-scale run (71.5k unique URLs) at high concurrency from a home
     network tripped mako's Radware/Akamai bot-mitigation (consistent 403s,
     confirmed via direct curl). The office network did not trip it. Defaults
     here stay moderately conservative (delay + circuit breaker) as a safety
     net in case any network gets flagged again.
  5. IMPORTANT -- `pic_furl` as stored in Snowflake (e.g. `..._autoOrient_a.jpg`)
     is a tiny 82x62px preview, confirmed by direct pixel inspection -- not
     an artifact of any block. Stripping the `_autoOrient_<letter>` suffix
     (e.g. `....jpg`) resolves to the original full-resolution photo
     (thousands of px on a side) in ~65% of cases; the rest 404 because the
     original was apparently deleted/replaced after the thumbnail was
     generated. The `_autoOrient_b` variant (232x175px) reliably exists as a
     fallback in those cases. This module now tries, per URL: full-res
     original -> `_b` (232x175) -> the raw `_a` URL (82x62, last resort,
     always present). Downloaded images are resized down to `max_side` on
     save to keep disk usage sane.

Usage:
    python -m src.data.download_images --csv data/raw/articles.csv \
        --url-column pic_furl --cache-dir data/images/
"""
import argparse
import hashlib
import io
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from PIL import Image

_THUMB_SUFFIX_RE = re.compile(r"^(.*)_autoOrient_[a-zA-Z]\.jpg$")


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


def _candidate_urls(pic_url: str):
    """Yield URLs for this article's image, best quality first.

    `pic_url` (the raw `pic_furl` value) is always tried last -- it's
    guaranteed to exist, so it anchors the fallback chain.
    """
    m = _THUMB_SUFFIX_RE.match(pic_url)
    if m:
        base = m.group(1)
        yield f"{base}.jpg"                 # full-res original
        yield f"{base}_autoOrient_b.jpg"     # 232x175 fallback
    yield pic_url


def _fetch(url, headers, max_retries):
    """GET a single URL. Returns (content_bytes_or_None, error_label_or_None).
    404s are not retried (no point); other failures use backoff retries."""
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=30, headers=headers)
            if resp.status_code == 404:
                return None, "not_found (404)"
            resp.raise_for_status()
            return resp.content, None
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
    return None, _classify_error(last_err)


def download_images(urls, cache_dir="data/images/", max_workers=6, max_retries=3,
                     delay_range=(0.3, 0.9), circuit_break_after=25, cooldown_seconds=600,
                     max_side=512):
    """Download+cache a list of image URLs. Returns (n_success, error_counter).

    delay_range: random per-request delay (seconds) before each network call,
        so the run doesn't look like a burst scrape.
    circuit_break_after: consecutive failures (across all workers) that
        trigger a pause -- a wall of failures almost always means "blocked",
        and hammering a block harder only reinforces it.
    cooldown_seconds: how long to pause when the circuit breaker trips.
    max_side: downloaded images are resized (preserving aspect ratio) so
        their longer side is at most this many pixels, to keep disk usage
        reasonable -- full-res originals can be several MB each.
    """
    os.makedirs(cache_dir, exist_ok=True)
    error_types = Counter()
    n_success = 0
    consecutive_failures = 0
    state_lock = threading.Lock()

    def task(url):
        nonlocal consecutive_failures
        path = _cache_path(url, cache_dir)
        if os.path.exists(path):
            return True, None

        with state_lock:
            trip_cooldown = consecutive_failures >= circuit_break_after
            if trip_cooldown:
                consecutive_failures = 0
        if trip_cooldown:
            print(f"  Warning: {circuit_break_after}+ consecutive failures -- likely blocked. "
                  f"Pausing {cooldown_seconds}s before resuming.", flush=True)
            time.sleep(cooldown_seconds)

        time.sleep(random.uniform(*delay_range))

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Referer": "https://www.mako.co.il/",
        }
        last_err = None
        for candidate in _candidate_urls(url):
            content, err = _fetch(candidate, headers, max_retries)
            if content is None:
                last_err = err
                continue
            try:
                img = Image.open(io.BytesIO(content)).convert("RGB")
                if max(img.size) > max_side:
                    img.thumbnail((max_side, max_side), Image.LANCZOS)
                img.save(path, "JPEG")
            except Exception:
                last_err = "corrupt_image"
                continue
            else:
                with state_lock:
                    consecutive_failures = 0
                return True, None
        with state_lock:
            consecutive_failures += 1
        return False, last_err

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(task, u) for u in urls]
        for i, future in enumerate(as_completed(futures), 1):
            ok, err = future.result()
            if ok:
                n_success += 1
            else:
                error_types[err] += 1
            if i % 100 == 0 or i == len(urls):
                print(f"  {i}/{len(urls)} processed ({n_success} cached, {sum(error_types.values())} failed)",
                      flush=True)

    if error_types:
        print("\nFailure breakdown:", dict(error_types))
        if error_types.get("forbidden (403)", 0) > 10:
            print("  Warning: many 403s -- likely running from a blocked network. "
                  "Try a different network, or wait for the block to lift.")

    return n_success, error_types


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--url-column", default="pic_furl")
    parser.add_argument("--cache-dir", default="data/images/")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--delay-min", type=float, default=0.3,
                         help="Minimum random delay (seconds) before each request.")
    parser.add_argument("--delay-max", type=float, default=0.9,
                         help="Maximum random delay (seconds) before each request.")
    parser.add_argument("--circuit-break-after", type=int, default=25,
                         help="Consecutive failures before pausing the whole run.")
    parser.add_argument("--cooldown-seconds", type=int, default=600,
                         help="Pause duration (seconds) when the circuit breaker trips.")
    parser.add_argument("--max-side", type=int, default=512,
                         help="Resize downloaded images so their longer side is at most this many pixels.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = [str(c).strip().lower() for c in df.columns]
    urls = df[args.url_column].dropna().unique().tolist()
    print(f"Downloading {len(urls)} unique images "
          f"(max_workers={args.max_workers}, delay={args.delay_min}-{args.delay_max}s, "
          f"circuit_break_after={args.circuit_break_after}, cooldown={args.cooldown_seconds}s, "
          f"max_side={args.max_side})...")
    download_images(
        urls,
        cache_dir=args.cache_dir,
        max_workers=args.max_workers,
        delay_range=(args.delay_min, args.delay_max),
        circuit_break_after=args.circuit_break_after,
        cooldown_seconds=args.cooldown_seconds,
        max_side=args.max_side,
    )
