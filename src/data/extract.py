"""Article-level data extraction -- MANUAL EXPORT workflow.

No live Snowflake connection from code (by design). Instead:
  1. Run ARTICLE_LEVEL_QUERY (below) in a Snowflake SQL worksheet.
  2. Download results as CSV.
  3. Save as data/raw/articles.csv (default path in configs/base_config.yaml).
  4. Run validate_extract() to sanity-check the export.

Filters applied in this version, and why:
  - publish_state = 'published', site = canonical_site: exclude drafts and
    duplicate/mirrored site entries.
  - sponsored_articles IS NULL: exclude sponsored content (no editorial
    control over title/image for these).
  - item_name != 'test': best-effort exclusion of obvious QA/test entries.
    NOTE: this is an exact-match filter and will miss variants like
    "test article", "asdasd", Hebrew "בדיקה", etc. -- flagged as a known
    limitation, not a thorough test-data filter.
  - main_channel_id NOT IN (...): excludes specific non-editorial channels.
  - HAVING COUNT(*) > 10: floor on 48h pageviews, intended to drop
    negligible/junk items. NOTE: also known to imperfectly correct for a
    suspected cross-site event-attribution bug (some items showing near-zero
    views on their real site due to a tracking bug misattributing views to
    another site) -- accepted based on manual sampling of dozens of cases,
    not a full quantified audit. Documented here as a conscious, stated
    tradeoff.
"""
import pandas as pd

ARTICLE_LEVEL_QUERY = """
WITH base AS (
    SELECT
        a.item_id,
        a.item_name,
        main_external_title as teaser_title,
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
      AND a.sponsored_articles is null
      AND a.item_name != 'test'
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
        MIN_BY(teaser_title, event_time) AS teaser_title,
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
    HAVING COUNT(*) > 10
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
    teaser_title,
    tags,
    pic_furl,
    site,
    TO_VARCHAR(display_time, 'YYYY-MM-DD HH24:MI:SS') AS display_time,
    CASE
        WHEN pct_rank_site_month >= .90 THEN 1
        ELSE 0
    END AS target
FROM ranked
ORDER BY display_time;
"""

EXPECTED_COLUMNS = ["item_id", "teaser_title", "tags", "pic_furl", "site", "display_time", "target"]


def validate_extract(csv_path: str = "data/raw/articles.csv") -> pd.DataFrame:
    """Sanity-check a manually-exported CSV before using it downstream.
    Column names are matched case-insensitively -- Snowflake's CSV export
    uppercases column names by default."""
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. "
                          f"Found columns: {list(df.columns)}")

    print(f"Rows: {len(df):,}")
    print(f"Unique item_id: {df['item_id'].nunique():,}  "
          f"(should equal row count -- duplicates would mean a grain bug)")
    print(f"Sites: {df['site'].value_counts().to_dict()}")
    print(f"Positive rate (target): {df['target'].mean():.1%}  "
          f"(expected close to 10% by construction)")
    print(f"Null pic_furl: {df['pic_furl'].isna().sum()}  (should be 0)")
    print(f"Null teaser_title: {df['teaser_title'].isna().sum()}  "
          f"(check this -- the enrichment join could produce nulls if a "
          f"channel_id match is missing)")

    # Fail fast on a malformed/mangled display_time column (e.g. a spreadsheet
    # app reinterpreting a timestamp as a bare "time of day" cell on save,
    # which silently truncates the date/hour and previously slipped through
    # here undetected).
    parsed_dates = pd.to_datetime(df["display_time"], errors="coerce")
    n_unparseable = parsed_dates.isna().sum()
    if n_unparseable > 0:
        raise ValueError(
            f"{n_unparseable}/{len(df)} display_time values failed to parse as "
            f"real timestamps. Example raw values: "
            f"{df['display_time'].head(3).tolist()}. This usually means the CSV "
            f"was opened/saved in a spreadsheet app (Excel/Sheets/Numbers) "
            f"between export and use, which can silently mangle timestamp "
            f"columns. Re-export directly from Snowflake without opening it in "
            f"a spreadsheet app first."
        )
    min_sane_year, max_sane_year = 2020, 2030
    if parsed_dates.dt.year.min() < min_sane_year or parsed_dates.dt.year.max() > max_sane_year:
        raise ValueError(
            f"display_time parsed, but with an implausible year range "
            f"({parsed_dates.dt.year.min()}\u2013{parsed_dates.dt.year.max()}) -- "
            f"likely still corrupted, just not in a way that raised a parse error."
        )
    print(f"Date range: {parsed_dates.min()} to {parsed_dates.max()}")

    return df


if __name__ == "__main__":
    validate_extract()
