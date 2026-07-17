"""Article-level data extraction — MANUAL EXPORT workflow.

No live Snowflake connection from code (by design — avoids ever needing
DB credentials inside this repo). Instead:

  1. Open a Snowflake SQL worksheet.
  2. Paste ARTICLE_LEVEL_QUERY below (fill in start_date).
  3. Run it, then use Snowflake's "Download Results" -> CSV.
  4. Save the file as data/raw/articles.csv (this path is already the
     default in configs/base_config.yaml -> data.raw_csv).
  5. Run validate_extract() (below) once to sanity-check the export before
     moving on to image downloading / EDA.

Grain: ONE ROW PER item_id (not per raw event) — the query already handles
this with MIN_BY(field, server_ts) for metadata fields, since the same
item_id can have multiple values across raw events.
"""
import pandas as pd

ARTICLE_LEVEL_QUERY = """
WITH base AS (
    SELECT
        item_id, item_name, tags, pic_furl, site,
        main_channel_id, main_channel_name, page_url,
        display_date, server_ts
    FROM mako_data_lake.public.combined_events_enriched
    WHERE event_name = 'page_view'
      AND site IN ('n12', 'mako')
      AND content_type IN ('article', 'recipe')
      AND display_date >= '2023-09-01'   -- <- edit start date here
      AND item_id IS NOT NULL AND item_id != 'unknown'
      AND item_name IS NOT NULL
      AND pic_furl IS NOT NULL
),
item_48h AS (
    SELECT
        item_id,
        MIN_BY(item_name, server_ts) AS item_name,
        MIN_BY(tags, server_ts) AS tags,
        MIN_BY(pic_furl, server_ts) AS pic_furl,
        MIN_BY(site, server_ts) AS site,
        MIN_BY(main_channel_id, server_ts) AS main_channel_id,
        MIN_BY(main_channel_name, server_ts) AS main_channel_name,
        MIN_BY(page_url, server_ts) AS page_url,
        MIN(display_date) AS display_date,
        DATE(MIN(display_date)) AS publish_date,
        DATE_TRUNC('month', MIN(display_date)) AS publish_month,
        COUNT(*) AS pageviews_48h
    FROM base
    WHERE server_ts >= display_date
      AND server_ts < DATEADD(hour, 48, display_date)
    GROUP BY item_id
),
ranked_items AS (
    SELECT
        item_48h.*,
        COUNT(*) OVER (PARTITION BY site, publish_month) AS items_in_site_month,
        PERCENT_RANK() OVER (
            PARTITION BY site, publish_month ORDER BY pageviews_48h
        ) AS pct_rank_site_month
    FROM item_48h
)
SELECT
    item_id, item_name, tags, pic_furl, site,
    main_channel_name, display_date, publish_month,
    pageviews_48h, pct_rank_site_month,
    CASE WHEN pct_rank_site_month >= 0.90 THEN 1 ELSE 0 END AS is_top_10_site_month,
    CASE WHEN pct_rank_site_month >= 0.80 THEN 1 ELSE 0 END AS is_top_20_site_month
FROM ranked_items
WHERE items_in_site_month >= 100
ORDER BY publish_date, site;
"""

EXPECTED_COLUMNS = [
    "item_id", "item_name", "tags", "pic_furl", "site",
    "main_channel_name", "display_date", "publish_month",
    "pageviews_48h", "pct_rank_site_month",
    "is_top_10_site_month", "is_top_20_site_month",
]


def validate_extract(csv_path: str = "data/raw/articles.csv") -> pd.DataFrame:
    """Sanity-check a manually-exported CSV before using it downstream.
    Run this once, right after exporting, in notebooks/01_eda.ipynb or a
    plain python shell — catches column mismatches or export mistakes early.
    """
    df = pd.read_csv(csv_path)

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. "
                          f"Found columns: {list(df.columns)}")

    print(f"Rows: {len(df):,}")
    print(f"Unique item_id: {df['item_id'].nunique():,}  "
          f"(should equal row count — duplicates would mean a grain bug)")
    print(f"Sites: {df['site'].value_counts().to_dict()}")
    print(f"Positive rate (is_top_10_site_month): {df['is_top_10_site_month'].mean():.1%}")
    print(f"Null pic_furl: {df['pic_furl'].isna().sum()}  (should be 0 — query already filters these)")
    print(f"Date range: {df['display_date'].min()} to {df['display_date'].max()}")

    return df


if __name__ == "__main__":
    validate_extract()
