"""
Generate synthetic ticket, macro, and macro usage data.

This module creates realistic CX data that simulates a customer support ticketing system.
It generates patterns where certain macros correlate with better outcomes.
"""

import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from src.config import (
    CHANNELS,
    CONTACT_DRIVERS,
    DEFAULT_NUM_AGENTS,
    DEFAULT_NUM_MACROS,
    DEFAULT_NUM_TICKETS,
    DEFAULT_TIME_RANGE_DAYS,
    LANGUAGES,
    MACRO_CATEGORIES,
    OWNER_TEAMS,
    PRIORITIES,
    RAW_MACROS_FILE,
    RAW_MACRO_USAGE_FILE,
    RAW_TICKETS_FILE,
    RANDOM_SEED,
    STATUSES,
)
from src.utils import set_random_seed


def generate_macros(num_macros: int = DEFAULT_NUM_MACROS) -> pd.DataFrame:
    """
    Generate synthetic macro data.

    Args:
        num_macros: Number of macros to generate

    Returns:
        DataFrame with macro information
    """
    set_random_seed()

    macros = []
    base_date = datetime.now() - timedelta(days=365)

    for i in range(num_macros):
        category = random.choice(MACRO_CATEGORIES)
        macro_id = f"{category.upper()[:4]}_{i+1:03d}"

        # Generate macro text based on category
        macro_body = _generate_macro_text(category, i)

        macro = {
            "macro_id": macro_id,
            "macro_name": f"{category.title()} Response {i+1}",
            "category": category,
            "macro_body": macro_body,
            "created_at": base_date + timedelta(days=random.randint(0, 300)),
            "updated_at": base_date + timedelta(days=random.randint(301, 365)),
            "owner_team": random.choice(OWNER_TEAMS),
            "is_active": random.random() > 0.1,  # 90% active
            "is_internal_only": random.random() < 0.15,  # 15% internal only
        }
        macros.append(macro)

    return pd.DataFrame(macros)


def _generate_macro_text(category: str, index: int) -> str:
    """Generate synthetic macro text based on category."""
    templates = {
        "billing": [
            "Thank you for contacting us about your billing inquiry. I've reviewed your account and can confirm that {detail}. Your updated balance is reflected in your account.",
            "I understand your concern about the billing issue. After checking, I can see that {detail}. I've processed a refund which will appear in 3-5 business days.",
            "Regarding your payment question, {detail}. Please let me know if you need any clarification on your invoice.",
        ],
        "technical": [
            "I'm sorry you're experiencing technical difficulties. To resolve this issue, please try the following: {detail}. This should resolve the problem.",
            "Thank you for reporting this technical error. Our team has identified that {detail}. The fix has been deployed.",
            "Based on the error you described, {detail}. Please clear your cache and try again.",
        ],
        "account": [
            "I can help you with your account settings. {detail}. Your changes have been saved successfully.",
            "Thank you for reaching out about your account. I've updated {detail} as requested.",
            "Regarding your account inquiry, {detail}. Let me know if you need further assistance.",
        ],
        "policy": [
            "Thank you for your question about our policy. {detail}. You can find more information in our help center.",
            "I understand your concern. Our policy states that {detail}. Please let me know if you have additional questions.",
            "Regarding our terms, {detail}. We appreciate your understanding.",
        ],
        "product": [
            "Thank you for your interest in our product features. {detail}. Would you like to learn more?",
            "I'm happy to explain how this works. {detail}. Feel free to ask if you have more questions.",
            "Great question! {detail}. Let me know if you'd like additional details.",
        ],
        "escalation": [
            "I understand this is important to you. I'm escalating your case to our specialist team. {detail}.",
            "Thank you for your patience. I've forwarded this to our senior support team who will contact you within 24 hours. {detail}.",
            "I apologize for the inconvenience. {detail}. Our management team will review this personally.",
        ],
    }

    template = random.choice(templates.get(category, templates["billing"]))
    detail_placeholder = random.choice([
        "the issue has been resolved",
        "your request has been processed",
        "we've made the necessary updates",
        "this is working as expected",
        "we've identified the root cause",
    ])

    return template.format(detail=detail_placeholder)


def generate_tickets(
    num_tickets: int = DEFAULT_NUM_TICKETS,
    num_agents: int = DEFAULT_NUM_AGENTS,
    time_range_days: int = DEFAULT_TIME_RANGE_DAYS,
) -> pd.DataFrame:
    """
    Generate synthetic ticket data.

    Args:
        num_tickets: Number of tickets to generate
        num_agents: Number of agents
        time_range_days: Range of days for ticket timestamps

    Returns:
        DataFrame with ticket information
    """
    set_random_seed()

    tickets = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=time_range_days)

    # Generate contact drivers based on weighted distribution
    contact_driver_list = []
    for driver, weight in CONTACT_DRIVERS.items():
        contact_driver_list.extend([driver] * int(weight * num_tickets))

    # Pad or trim to exact number
    contact_driver_list = (contact_driver_list + [random.choice(list(CONTACT_DRIVERS.keys()))] * num_tickets)[:num_tickets]
    random.shuffle(contact_driver_list)

    for i in range(num_tickets):
        ticket_id = f"TKT-{i+1:06d}"
        contact_driver = contact_driver_list[i]
        priority = random.choice(PRIORITIES)
        channel = random.choice(CHANNELS)

        # Generate timestamps
        created_at = start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )

        # Resolve time depends on priority and contact driver
        base_resolve_hours = _get_base_resolve_time(priority, contact_driver)
        resolve_hours = max(0.5, random.gauss(base_resolve_hours, base_resolve_hours * 0.3))
        resolved_at = created_at + timedelta(hours=resolve_hours)

        # Handle time
        base_handle = _get_base_handle_time(contact_driver, channel)
        handle_time_minutes = max(5, random.gauss(base_handle, base_handle * 0.4))

        # First response time
        first_response_minutes = max(1, random.gauss(15, 10))

        # Replies
        replies_count = random.randint(1, 6)

        # Reopens (some tickets reopen)
        reopens_count = 1 if random.random() < 0.15 else 0

        # CSAT (will be influenced by macro effectiveness later)
        base_csat = _get_base_csat(contact_driver, priority)
        csat_score = np.clip(int(random.gauss(base_csat, 0.8)), 1, 5)

        ticket = {
            "ticket_id": ticket_id,
            "created_at": created_at,
            "resolved_at": resolved_at,
            "status": random.choice(STATUSES),
            "channel": channel,
            "priority": priority,
            "contact_driver": contact_driver,
            "first_response_time_minutes": first_response_minutes,
            "total_handle_time_minutes": handle_time_minutes,
            "replies_count": replies_count,
            "reopens_count": reopens_count,
            "csat_score": csat_score,
            "csat_response": None,  # Optional free text
            "agent_id": f"AGENT_{random.randint(1, num_agents):03d}",
            "macro_sequence": None,  # Will be populated by macro_usage
            "final_macro_id": None,  # Will be populated by macro_usage
            "language": random.choice(LANGUAGES),
        }
        tickets.append(ticket)

    return pd.DataFrame(tickets)


def _get_base_resolve_time(priority: str, contact_driver: str) -> float:
    """Get base resolution time in hours based on priority and contact driver."""
    priority_multipliers = {"urgent": 0.5, "high": 1.0, "normal": 2.0, "low": 3.0}
    driver_base_hours = {
        "billing_issue": 24,
        "login_problem": 4,
        "device_error": 12,
        "account_management": 8,
        "payment_failed": 6,
        "feature_request": 48,
        "refund_request": 24,
        "technical_error": 16,
        "general_inquiry": 12,
    }
    return driver_base_hours.get(contact_driver, 12) * priority_multipliers.get(priority, 1.0)


def _get_base_handle_time(contact_driver: str, channel: str) -> float:
    """Get base handle time in minutes."""
    driver_times = {
        "billing_issue": 35,
        "login_problem": 15,
        "device_error": 45,
        "account_management": 25,
        "payment_failed": 30,
        "feature_request": 20,
        "refund_request": 40,
        "technical_error": 50,
        "general_inquiry": 20,
    }
    channel_multipliers = {"chat": 0.8, "email": 1.0, "phone": 1.2, "webform": 0.9}

    base = driver_times.get(contact_driver, 30)
    return base * channel_multipliers.get(channel, 1.0)


def _get_base_csat(contact_driver: str, priority: str) -> float:
    """Get base CSAT score (1-5 scale)."""
    driver_scores = {
        "billing_issue": 3.2,
        "login_problem": 3.8,
        "device_error": 3.0,
        "account_management": 4.0,
        "payment_failed": 3.3,
        "feature_request": 4.2,
        "refund_request": 3.5,
        "technical_error": 3.1,
        "general_inquiry": 3.9,
    }
    priority_adjustments = {"urgent": -0.3, "high": -0.1, "normal": 0.0, "low": 0.1}

    base = driver_scores.get(contact_driver, 3.5)
    adjustment = priority_adjustments.get(priority, 0.0)
    return base + adjustment


def generate_macro_usage(
    tickets_df: pd.DataFrame,
    macros_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic macro usage data and update tickets with macro sequences.

    This function creates realistic patterns where:
    - Some macros are correlated with better outcomes (high CSAT, lower handle time)
    - Some macros are used frequently but have poor outcomes
    - Some macros are underused but effective

    Args:
        tickets_df: DataFrame of tickets
        macros_df: DataFrame of macros

    Returns:
        Tuple of (macro_usage_df, updated_tickets_df)
    """
    set_random_seed()

    # Identify "effective" and "ineffective" macro groups
    macro_ids = macros_df["macro_id"].tolist()
    effective_macros = random.sample(macro_ids, k=int(len(macro_ids) * 0.2))  # Top 20%
    ineffective_macros = random.sample(
        [m for m in macro_ids if m not in effective_macros], k=int(len(macro_ids) * 0.15)
    )  # Bottom 15%

    macro_usage_records = []
    tickets_df = tickets_df.copy()

    for idx, ticket in tickets_df.iterrows():
        # Determine if this ticket uses macros (80% chance)
        if random.random() > 0.8:
            continue

        # Number of macros to use (1-3)
        num_macros = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]

        # Filter macros by category matching contact driver
        category_map = {
            "billing_issue": "billing",
            "payment_failed": "billing",
            "refund_request": "billing",
            "login_problem": "technical",
            "device_error": "technical",
            "technical_error": "technical",
            "account_management": "account",
            "feature_request": "product",
            "general_inquiry": "policy",
        }

        preferred_category = category_map.get(ticket["contact_driver"], "policy")
        category_macros = macros_df[macros_df["category"] == preferred_category]["macro_id"].tolist()

        if not category_macros:
            category_macros = macro_ids

        # Select macros (with potential bias toward effective/ineffective)
        selected_macros = []
        for position in range(1, num_macros + 1):
            # Bias selection
            if random.random() < 0.3 and effective_macros:
                macro = random.choice(effective_macros)
            elif random.random() < 0.15 and ineffective_macros:
                macro = random.choice(ineffective_macros)
            else:
                macro = random.choice(category_macros if category_macros else macro_ids)

            selected_macros.append(macro)

            # Create usage record
            applied_at = ticket["created_at"] + timedelta(
                minutes=random.uniform(0, ticket["total_handle_time_minutes"])
            )

            usage = {
                "ticket_id": ticket["ticket_id"],
                "macro_id": macro,
                "applied_at": applied_at,
                "position_in_thread": position,
                "agent_id": ticket["agent_id"],
                "channel": ticket["channel"],
            }
            macro_usage_records.append(usage)

        # Update ticket with macro sequence
        tickets_df.at[idx, "macro_sequence"] = ",".join(selected_macros)
        tickets_df.at[idx, "final_macro_id"] = selected_macros[-1]

        # Adjust CSAT and handle time based on macro effectiveness
        if any(m in effective_macros for m in selected_macros):
            # Effective macros improve outcomes
            tickets_df.at[idx, "csat_score"] = min(5, tickets_df.at[idx, "csat_score"] + random.randint(0, 1))
            tickets_df.at[idx, "total_handle_time_minutes"] *= random.uniform(0.8, 0.95)
            tickets_df.at[idx, "reopens_count"] = 0 if random.random() < 0.8 else tickets_df.at[idx, "reopens_count"]

        if any(m in ineffective_macros for m in selected_macros):
            # Ineffective macros worsen outcomes
            tickets_df.at[idx, "csat_score"] = max(1, tickets_df.at[idx, "csat_score"] - random.randint(0, 1))
            tickets_df.at[idx, "total_handle_time_minutes"] *= random.uniform(1.05, 1.25)
            if random.random() < 0.3:
                tickets_df.at[idx, "reopens_count"] = 1

    macro_usage_df = pd.DataFrame(macro_usage_records)

    return macro_usage_df, tickets_df


def generate_all_data(
    num_tickets: int = DEFAULT_NUM_TICKETS,
    num_macros: int = DEFAULT_NUM_MACROS,
    num_agents: int = DEFAULT_NUM_AGENTS,
    time_range_days: int = DEFAULT_TIME_RANGE_DAYS,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate all synthetic data (macros, tickets, macro_usage) and optionally save to CSV.

    Args:
        num_tickets: Number of tickets to generate
        num_macros: Number of macros to generate
        num_agents: Number of agents
        time_range_days: Range of days for timestamps
        save: Whether to save CSVs to data/raw/

    Returns:
        Tuple of (macros_df, tickets_df, macro_usage_df)
    """
    logger.info("Generating macros...")
    macros_df = generate_macros(num_macros)
    logger.info(f"  Generated {len(macros_df)} macros")

    logger.info("Generating tickets...")
    tickets_df = generate_tickets(num_tickets, num_agents, time_range_days)
    logger.info(f"  Generated {len(tickets_df)} tickets")

    logger.info("Generating macro usage...")
    macro_usage_df, tickets_df = generate_macro_usage(tickets_df, macros_df)
    logger.info(f"  Generated {len(macro_usage_df)} macro usage records")

    if save:
        logger.info(f"Saving to {RAW_MACROS_FILE.parent}...")
        macros_df.to_csv(RAW_MACROS_FILE, index=False)
        tickets_df.to_csv(RAW_TICKETS_FILE, index=False)
        macro_usage_df.to_csv(RAW_MACRO_USAGE_FILE, index=False)
        logger.info("✓ Data generation complete!")

    return macros_df, tickets_df, macro_usage_df


if __name__ == "__main__":
    generate_all_data()
