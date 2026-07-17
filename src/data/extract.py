"""Extraction of article-level data from Keshet's Snowflake warehouse.

This is the article-level counterpart of the daily-aggregated validation query
already developed during the diagnostic phase. Grain: ONE ROW PER item_id
(not per event) — use MIN_BY(field, server_ts) for metadata fields, since the
same item_id can have multiple values across raw events.

TODO (Step 1): run this against Snowflake and save the result to
data/raw/articles.csv. This module currently only holds the query template
and a thin wrapper — actual DB connection code depends on the credentials
setup available on your machine (not run from this environment).
"""

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
      AND display_date >= %(start_date)s
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


def extract_to_csv(start_date: str, output_path: str) -> None:
    """TODO: connect to Snowflake (e.g. via snowflake-connector-python),
    run ARTICLE_LEVEL_QUERY with the given start_date, and write the result
    to output_path as CSV. Left unimplemented here — fill in together in Step 1."""
    raise NotImplementedError("Wire up your Snowflake connection here (Step 1).")
