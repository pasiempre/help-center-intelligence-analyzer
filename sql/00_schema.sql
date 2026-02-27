-- Help Center Intelligence Analyzer
-- Schema DDL for analytical tables
-- These would be created in a data warehouse (Snowflake, BigQuery, Redshift)

-- ============================================
-- DIMENSION: Macros
-- ============================================
CREATE TABLE dim_macros (
    macro_id VARCHAR(50) PRIMARY KEY,
    macro_name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    macro_body TEXT,
    owner_team VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    language VARCHAR(10) DEFAULT 'en',
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE INDEX idx_macros_category ON dim_macros(category);
CREATE INDEX idx_macros_active ON dim_macros(is_active);

-- ============================================
-- FACT: Tickets
-- ============================================
CREATE TABLE fact_tickets (
    ticket_id VARCHAR(50) PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,
    resolved_at TIMESTAMP,
    csat_score INTEGER CHECK (csat_score BETWEEN 1 AND 5),
    handle_time_seconds INTEGER,
    contact_driver VARCHAR(100),
    priority VARCHAR(20),
    channel VARCHAR(50),
    agent_id VARCHAR(50),
    first_response_time_seconds INTEGER,
    was_reopened BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_tickets_created ON fact_tickets(created_at);
CREATE INDEX idx_tickets_driver ON fact_tickets(contact_driver);
CREATE INDEX idx_tickets_agent ON fact_tickets(agent_id);

-- ============================================
-- BRIDGE: Macro Usage (many-to-many)
-- ============================================
CREATE TABLE bridge_macro_usage (
    usage_id SERIAL PRIMARY KEY,
    ticket_id VARCHAR(50) REFERENCES fact_tickets(ticket_id),
    macro_id VARCHAR(50) REFERENCES dim_macros(macro_id),
    used_at TIMESTAMP,
    sequence_order INTEGER DEFAULT 1
);

CREATE INDEX idx_usage_ticket ON bridge_macro_usage(ticket_id);
CREATE INDEX idx_usage_macro ON bridge_macro_usage(macro_id);

-- ============================================
-- VIEW: Macro Usage Aggregates
-- ============================================
CREATE OR REPLACE VIEW v_macro_usage_stats AS
SELECT
    m.macro_id,
    m.macro_name,
    m.category,
    COUNT(DISTINCT u.ticket_id) AS total_uses,
    COUNT(DISTINCT t.agent_id) AS unique_agents,
    MIN(u.used_at) AS first_used,
    MAX(u.used_at) AS last_used
FROM dim_macros m
LEFT JOIN bridge_macro_usage u ON m.macro_id = u.macro_id
LEFT JOIN fact_tickets t ON u.ticket_id = t.ticket_id
GROUP BY m.macro_id, m.macro_name, m.category;

-- ============================================
-- VIEW: Ticket-Macro Denormalized
-- ============================================
CREATE OR REPLACE VIEW v_ticket_macro_detail AS
SELECT
    t.ticket_id,
    t.created_at,
    t.csat_score,
    t.handle_time_seconds,
    t.contact_driver,
    t.was_reopened,
    m.macro_id,
    m.macro_name,
    m.category
FROM fact_tickets t
JOIN bridge_macro_usage u ON t.ticket_id = u.ticket_id
JOIN dim_macros m ON u.macro_id = m.macro_id;
