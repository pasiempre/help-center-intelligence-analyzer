-- Help Center Intelligence Analyzer
-- Recommendation Queries
-- Equivalent to src/evaluation.py logic

-- ============================================
-- 1. Unused Macros - Candidates for Deprecation
-- ============================================
SELECT
    m.macro_id,
    m.macro_name,
    m.category,
    m.created_at,
    COALESCE(COUNT(u.usage_id), 0) AS total_uses,
    'Archive - No usage in analysis period' AS recommendation
FROM dim_macros m
LEFT JOIN bridge_macro_usage u ON m.macro_id = u.macro_id
WHERE m.is_active = TRUE
GROUP BY m.macro_id, m.macro_name, m.category, m.created_at
HAVING COALESCE(COUNT(u.usage_id), 0) = 0
ORDER BY m.created_at;


-- ============================================
-- 2. Low-Usage Macros (< 5 uses)
-- May need review or promotion
-- ============================================
SELECT
    m.macro_id,
    m.macro_name,
    m.category,
    COUNT(u.usage_id) AS total_uses,
    'Review - Insufficient data for scoring' AS recommendation
FROM dim_macros m
LEFT JOIN bridge_macro_usage u ON m.macro_id = u.macro_id
WHERE m.is_active = TRUE
GROUP BY m.macro_id, m.macro_name, m.category
HAVING COUNT(u.usage_id) BETWEEN 1 AND 4
ORDER BY COUNT(u.usage_id) DESC;


-- ============================================
-- 3. Cluster Analysis for Consolidation
-- Identify categories with redundant macros
-- ============================================
WITH category_stats AS (
    SELECT
        m.category,
        COUNT(DISTINCT m.macro_id) AS macro_count,
        COUNT(DISTINCT u.ticket_id) AS total_tickets,
        AVG(t.csat_score) AS avg_csat,
        AVG(t.handle_time_seconds) AS avg_handle_time
    FROM dim_macros m
    LEFT JOIN bridge_macro_usage u ON m.macro_id = u.macro_id
    LEFT JOIN fact_tickets t ON u.ticket_id = t.ticket_id
    WHERE m.is_active = TRUE
    GROUP BY m.category
)
SELECT
    category,
    macro_count,
    total_tickets,
    ROUND(avg_csat, 2) AS avg_csat,
    ROUND(avg_handle_time, 0) AS avg_handle_time_sec,
    CASE
        WHEN macro_count > 10 AND avg_csat < 3.5
            THEN 'Consolidate - Too many low-performing macros'
        WHEN macro_count > 15
            THEN 'Review - High macro count may indicate redundancy'
        ELSE 'OK'
    END AS recommendation
FROM category_stats
WHERE macro_count > 5
ORDER BY macro_count DESC;


-- ============================================
-- 4. Similarity Detection (requires text functions)
-- Find macros with similar names for manual review
-- ============================================
SELECT
    m1.macro_id AS macro_1_id,
    m1.macro_name AS macro_1_name,
    m2.macro_id AS macro_2_id,
    m2.macro_name AS macro_2_name,
    m1.category,
    'Potential duplicate - review for consolidation' AS recommendation
FROM dim_macros m1
JOIN dim_macros m2
    ON m1.category = m2.category
    AND m1.macro_id < m2.macro_id  -- Avoid duplicates
    AND (
        -- Simple similarity: shared words in name
        LOWER(m1.macro_name) LIKE '%' || LOWER(SPLIT_PART(m2.macro_name, ' ', 1)) || '%'
        OR LOWER(m2.macro_name) LIKE '%' || LOWER(SPLIT_PART(m1.macro_name, ' ', 1)) || '%'
    )
WHERE m1.is_active = TRUE AND m2.is_active = TRUE
LIMIT 50;


-- ============================================
-- 5. Executive Summary Metrics
-- ============================================
WITH macro_status AS (
    SELECT
        m.macro_id,
        COALESCE(COUNT(u.usage_id), 0) AS usage_count
    FROM dim_macros m
    LEFT JOIN bridge_macro_usage u ON m.macro_id = u.macro_id
    WHERE m.is_active = TRUE
    GROUP BY m.macro_id
)
SELECT
    COUNT(*) AS total_active_macros,
    SUM(CASE WHEN usage_count = 0 THEN 1 ELSE 0 END) AS unused_macros,
    SUM(CASE WHEN usage_count BETWEEN 1 AND 4 THEN 1 ELSE 0 END) AS low_usage_macros,
    SUM(CASE WHEN usage_count >= 5 THEN 1 ELSE 0 END) AS scoreable_macros,
    ROUND(100.0 * SUM(CASE WHEN usage_count = 0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS unused_pct,
    ROUND(AVG(usage_count), 1) AS avg_usage_per_macro
FROM macro_status;


-- ============================================
-- 6. Action Items Summary
-- ============================================
SELECT 'Macros to Archive (unused)' AS action_type,
       COUNT(*) AS count
FROM dim_macros m
LEFT JOIN bridge_macro_usage u ON m.macro_id = u.macro_id
WHERE m.is_active = TRUE
GROUP BY m.macro_id
HAVING COUNT(u.usage_id) = 0

UNION ALL

SELECT 'Macros Needing Review (low usage)' AS action_type,
       COUNT(*) AS count
FROM dim_macros m
LEFT JOIN bridge_macro_usage u ON m.macro_id = u.macro_id
WHERE m.is_active = TRUE
GROUP BY m.macro_id
HAVING COUNT(u.usage_id) BETWEEN 1 AND 4

UNION ALL

SELECT 'Categories for Consolidation Review' AS action_type,
       COUNT(*) AS count
FROM (
    SELECT category
    FROM dim_macros
    WHERE is_active = TRUE
    GROUP BY category
    HAVING COUNT(*) > 10
) subq;
