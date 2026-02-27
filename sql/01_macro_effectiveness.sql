-- Help Center Intelligence Analyzer
-- Macro Effectiveness Scoring Queries
-- Equivalent to src/macro_effectiveness.py calculations

-- ============================================
-- 1. Calculate baseline metrics per contact driver
-- Used to normalize effectiveness scores
-- ============================================
WITH baseline_metrics AS (
    SELECT
        contact_driver,
        AVG(csat_score) AS baseline_csat,
        AVG(handle_time_seconds) AS baseline_handle_time,
        AVG(CASE WHEN was_reopened THEN 1.0 ELSE 0.0 END) AS baseline_reopen_rate
    FROM fact_tickets
    WHERE csat_score IS NOT NULL
    GROUP BY contact_driver
),

-- ============================================
-- 2. Calculate raw metrics per macro
-- ============================================
macro_raw_metrics AS (
    SELECT
        m.macro_id,
        m.macro_name,
        m.category,
        t.contact_driver,
        COUNT(*) AS usage_count,
        AVG(t.csat_score) AS avg_csat,
        AVG(t.handle_time_seconds) AS avg_handle_time,
        AVG(CASE WHEN t.was_reopened THEN 1.0 ELSE 0.0 END) AS reopen_rate
    FROM dim_macros m
    JOIN bridge_macro_usage u ON m.macro_id = u.macro_id
    JOIN fact_tickets t ON u.ticket_id = t.ticket_id
    WHERE t.csat_score IS NOT NULL
    GROUP BY m.macro_id, m.macro_name, m.category, t.contact_driver
    HAVING COUNT(*) >= 5  -- MIN_USAGE_FOR_SCORING threshold
),

-- ============================================
-- 3. Calculate normalized component scores (0-100 scale)
-- ============================================
macro_normalized AS (
    SELECT
        mr.macro_id,
        mr.macro_name,
        mr.category,
        mr.usage_count,
        mr.avg_csat,
        mr.avg_handle_time,
        mr.reopen_rate,

        -- CSAT component: higher is better, scale 1-5 to 0-100
        ((mr.avg_csat - 1) / 4.0) * 100 AS csat_score_normalized,

        -- Handle time component: lower is better relative to baseline
        -- Score = 100 * (1 - (macro_time / baseline_time)), capped at 0-100
        GREATEST(0, LEAST(100,
            100 * (1 - (mr.avg_handle_time / NULLIF(b.baseline_handle_time, 0)))
        )) AS handle_time_score_normalized,

        -- Reopen rate component: lower is better
        -- Score = 100 * (1 - reopen_rate)
        (1 - mr.reopen_rate) * 100 AS reopen_score_normalized

    FROM macro_raw_metrics mr
    JOIN baseline_metrics b ON mr.contact_driver = b.contact_driver
),

-- ============================================
-- 4. Calculate weighted effectiveness index
-- Weights: CSAT 40%, Handle Time 30%, Reopen 30%
-- ============================================
macro_effectiveness AS (
    SELECT
        macro_id,
        macro_name,
        category,
        usage_count,
        avg_csat,
        avg_handle_time,
        reopen_rate,
        csat_score_normalized,
        handle_time_score_normalized,
        reopen_score_normalized,

        -- Weighted effectiveness index
        (0.40 * csat_score_normalized) +
        (0.30 * handle_time_score_normalized) +
        (0.30 * reopen_score_normalized) AS effectiveness_index

    FROM macro_normalized
)

-- ============================================
-- Final output: Macro effectiveness scores
-- ============================================
SELECT
    macro_id,
    macro_name,
    category,
    usage_count,
    ROUND(avg_csat, 2) AS avg_csat,
    ROUND(avg_handle_time, 0) AS avg_handle_time_sec,
    ROUND(reopen_rate * 100, 1) AS reopen_rate_pct,
    ROUND(effectiveness_index, 1) AS effectiveness_score,

    -- Categorize effectiveness
    CASE
        WHEN effectiveness_index >= 75 THEN 'High Performer'
        WHEN effectiveness_index >= 50 THEN 'Moderate'
        WHEN effectiveness_index >= 25 THEN 'Needs Review'
        ELSE 'Low Performer'
    END AS effectiveness_tier

FROM macro_effectiveness
ORDER BY effectiveness_index DESC;


-- ============================================
-- BONUS: Identify "Underused Gems"
-- High effectiveness + low usage
-- ============================================
SELECT
    macro_id,
    macro_name,
    category,
    usage_count,
    ROUND(effectiveness_index, 1) AS effectiveness_score,
    'Underused Gem - Promote!' AS recommendation
FROM macro_effectiveness
WHERE effectiveness_index >= 75  -- Top quartile
  AND usage_count < 50           -- Low usage threshold
ORDER BY effectiveness_index DESC;
