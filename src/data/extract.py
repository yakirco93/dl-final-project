"""Article-level data extraction — MANUAL EXPORT workflow.

No live Snowflake connection from code (by design). Instead:
  1. Run ARTICLE_LEVEL_QUERY (below) in a Snowflake SQL worksheet.
  2. Download results as CSV.
  3. Save as data/raw/articles.csv (default path in configs/base_config.yaml).
  4. Run validate_extract() to sanity-check the export.

NOTE on the `target` definition in this version: positives are the top 10%
by 48h pageviews, but ONLY among articles with pageviews_2d > 500 (see the
HAVING clause) -- a per-article floor, not a peer-group-size floor. Flagged
explicitly because it changes what "success" means: this excludes
low-traffic articles from training entirely (rather than labeling them
negative), which narrows the population the model is evaluated against.
Confirm this is intentional before using it for the main model.
"""
import pandas as pd

ARTICLE_LEVEL_QUERY = """
WITH base AS (
    SELECT
        a.item_id,
        a.item_name,
        main_external_title as teaser_titel,
        a.tags,
        a.pic_furl,
        a.site,
        a.display_time,
        a.event_time
    FROM mako_data_lake.public.combined_events_enriched a
    JOIN MAKO_DATA_LAKE.PUBLIC.dataenrichment b
    ON a.item_id = b.item_id AND a.channel_id = b.channel_id
    WHERE a.event_name = 'page_view'
      AND a.site IN ('n12', 'mako')
      AND a.content_type IN ('article', 'recipe')
      AND a.display_time >= '2024-01-01'
      AND a.publish_state = 'published'
      AND a.pic_furl IS NOT NULL
      AND a.site = a.canonical_site
      AND a.main_channel_id NOT IN (
          'd0289dfc85dab610VgnVCM200000650a10acRCRD',
          '44460a2610f26110VgnVCM1000005201000aRCRD',
          '87b50a2610f26110VgnVCM1000005201000aRCRD',
          '737434c1c4e7c210VgnVCM2000002a0c10acRCRD'
      )
),
item_2d AS (
    SELECT
        item_id,
        MIN_BY(teaser_titel, event_time) AS teaser_titel,
        MIN_BY(tags, event_time) AS tags,
        MIN_BY(pic_furl, event_time) AS pic_furl,
        MIN_BY(site, event_time) AS site,
        MIN(display_time) AS display_time,
        DATE_TRUNC('month', MIN(display_time)) AS publish_month,
        COUNT(*) AS pageviews_2d
    FROM base
    WHERE event_time >= display_time
      AND event_time < DATEADD(hour,48,display_time)
    GROUP BY item_id
    HAVING COUNT(*) > 500
),
ranked AS (
    SELECT
        *,
        PERCENT_RANK() OVER (
            PARTITION BY site,publish_month
            ORDER BY pageviews_2d
        ) AS pct_rank_site_month
    FROM item_2d
)
SELECT
    item_id,
    teaser_titel,
    tags,
    pic_furl,
    site,
    CASE
        WHEN pct_rank_site_month >= .90 THEN 1
        ELSE 0
    END AS target
FROM ranked
ORDER BY display_time;
"""

EXPECTED_COLUMNS = ["item_id", "teaser_titel", "tags", "pic_furl", "site", "target"]


def validate_extract(csv_path: str = "data/raw/articles.csv") -> pd.DataFrame:
    """Sanity-check a manually-exported CSV before using it downstream."""
    df = pd.read_csv(csv_path)

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. "
                          f"Found columns: {list(df.columns)}")

    print(f"Rows: {len(df):,}")
    print(f"Unique item_id: {df['item_id'].nunique():,}  "
          f"(should equal row count -- duplicates would mean a grain bug)")
    print(f"Sites: {df['site'].value_counts().to_dict()}")
    print(f"Positive rate (target): {df['target'].mean():.1%}  "
          f"(expected close to 10% by construction, but the >500-views "
          f"floor may shift this -- worth checking)")
    print(f"Null pic_furl: {df['pic_furl'].isna().sum()}  (should be 0)")
    print(f"Null teaser_titel: {df['teaser_titel'].isna().sum()}  "
          f"(check this -- the enrichment join could produce nulls if a "
          f"channel_id match is missing)")

    return df


if __name__ == "__main__":
    validate_extract()
