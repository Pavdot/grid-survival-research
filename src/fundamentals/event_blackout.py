from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config_loader import project_path


EVENT_COLUMNS = [
    "event_time_utc",
    "known_time_utc",
    "category",
    "severity",
    "source",
    "title",
    "is_scheduled",
    "is_surprise",
]

WINDOW_COLUMNS = [
    "mode",
    "window_start_utc",
    "window_end_utc",
    *EVENT_COLUMNS,
]


def _parse_utc(value: object, column: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{column} must be timezone-aware UTC, got naive timestamp {value!r}")
    return timestamp.tz_convert("UTC")


def default_fundamental_events() -> pd.DataFrame:
    """Curated MVP seed: major scheduled macro events plus crypto surprise proxies."""
    rows = [
        ("2024-04-20T00:09:00Z", "2024-04-20T00:09:00Z", "halving", 4, "seed", "Bitcoin halving execution window", True, False),
        ("2024-06-07T12:30:00Z", "2024-06-07T12:30:00Z", "macro_jobs", 4, "seed", "US employment situation release", True, False),
        ("2024-06-12T12:30:00Z", "2024-06-12T12:30:00Z", "macro_cpi", 5, "seed", "US CPI release", True, False),
        ("2024-06-12T18:00:00Z", "2024-06-12T18:00:00Z", "macro_fomc", 5, "seed", "FOMC decision and projections", True, False),
        ("2024-07-05T00:00:00Z", "2024-07-05T00:00:00Z", "major_liquidation", 4, "seed", "German government BTC transfer pressure", False, True),
        ("2024-07-11T12:30:00Z", "2024-07-11T12:30:00Z", "macro_cpi", 5, "seed", "US CPI release", True, False),
        ("2024-07-31T18:00:00Z", "2024-07-31T18:00:00Z", "macro_fomc", 5, "seed", "FOMC decision", True, False),
        ("2024-08-02T12:30:00Z", "2024-08-02T12:30:00Z", "macro_jobs", 4, "seed", "US employment situation release", True, False),
        ("2024-08-05T00:00:00Z", "2024-08-05T00:00:00Z", "major_liquidation", 5, "seed", "Global risk-off crypto liquidation shock", False, True),
        ("2024-09-06T12:30:00Z", "2024-09-06T12:30:00Z", "macro_jobs", 4, "seed", "US employment situation release", True, False),
        ("2024-09-11T12:30:00Z", "2024-09-11T12:30:00Z", "macro_cpi", 5, "seed", "US CPI release", True, False),
        ("2024-09-18T18:00:00Z", "2024-09-18T18:00:00Z", "macro_fomc", 5, "seed", "FOMC decision and projections", True, False),
        ("2024-10-04T12:30:00Z", "2024-10-04T12:30:00Z", "macro_jobs", 4, "seed", "US employment situation release", True, False),
        ("2024-10-10T12:30:00Z", "2024-10-10T12:30:00Z", "macro_cpi", 5, "seed", "US CPI release", True, False),
        ("2024-11-01T12:30:00Z", "2024-11-01T12:30:00Z", "macro_jobs", 4, "seed", "US employment situation release", True, False),
        ("2024-11-07T19:00:00Z", "2024-11-07T19:00:00Z", "macro_fomc", 5, "seed", "FOMC decision", True, False),
        ("2024-11-13T13:30:00Z", "2024-11-13T13:30:00Z", "macro_cpi", 5, "seed", "US CPI release", True, False),
        ("2024-12-06T13:30:00Z", "2024-12-06T13:30:00Z", "macro_jobs", 4, "seed", "US employment situation release", True, False),
        ("2024-12-11T13:30:00Z", "2024-12-11T13:30:00Z", "macro_cpi", 5, "seed", "US CPI release", True, False),
        ("2024-12-18T19:00:00Z", "2024-12-18T19:00:00Z", "macro_fomc", 5, "seed", "FOMC decision and projections", True, False),
        ("2025-01-10T13:30:00Z", "2025-01-10T13:30:00Z", "macro_jobs", 4, "seed", "US employment situation release", True, False),
        ("2025-01-15T13:30:00Z", "2025-01-15T13:30:00Z", "macro_cpi", 5, "seed", "US CPI release", True, False),
        ("2025-01-29T19:00:00Z", "2025-01-29T19:00:00Z", "macro_fomc", 5, "seed", "FOMC decision", True, False),
        ("2025-02-07T13:30:00Z", "2025-02-07T13:30:00Z", "macro_jobs", 4, "seed", "US employment situation release", True, False),
        ("2025-02-12T13:30:00Z", "2025-02-12T13:30:00Z", "macro_cpi", 5, "seed", "US CPI release", True, False),
        ("2025-02-21T14:00:00Z", "2025-02-21T14:00:00Z", "exchange_hack", 5, "seed", "Major exchange hack reported", False, True),
        ("2025-03-02T00:00:00Z", "2025-03-02T00:00:00Z", "crypto_regulatory", 4, "seed", "US strategic crypto reserve announcement risk", False, True),
        ("2025-03-07T13:30:00Z", "2025-03-07T13:30:00Z", "macro_jobs", 4, "seed", "US employment situation release", True, False),
        ("2025-03-12T12:30:00Z", "2025-03-12T12:30:00Z", "macro_cpi", 5, "seed", "US CPI release", True, False),
        ("2025-03-19T18:00:00Z", "2025-03-19T18:00:00Z", "macro_fomc", 5, "seed", "FOMC decision and projections", True, False),
        ("2025-04-04T12:30:00Z", "2025-04-04T12:30:00Z", "macro_jobs", 4, "seed", "US employment situation release", True, False),
        ("2025-04-10T12:30:00Z", "2025-04-10T12:30:00Z", "macro_cpi", 5, "seed", "US CPI release", True, False),
        ("2025-05-02T12:30:00Z", "2025-05-02T12:30:00Z", "macro_jobs", 4, "seed", "US employment situation release", True, False),
        ("2025-05-07T18:00:00Z", "2025-05-07T18:00:00Z", "macro_fomc", 5, "seed", "FOMC decision", True, False),
        ("2025-05-13T12:30:00Z", "2025-05-13T12:30:00Z", "macro_cpi", 5, "seed", "US CPI release", True, False),
        ("2025-06-06T12:30:00Z", "2025-06-06T12:30:00Z", "macro_jobs", 4, "seed", "US employment situation release", True, False),
        ("2025-06-11T12:30:00Z", "2025-06-11T12:30:00Z", "macro_cpi", 5, "seed", "US CPI release", True, False),
        ("2025-06-18T18:00:00Z", "2025-06-18T18:00:00Z", "macro_fomc", 5, "seed", "FOMC decision and projections", True, False),
        ("2025-07-03T12:30:00Z", "2025-07-03T12:30:00Z", "macro_jobs", 4, "seed", "US employment situation release", True, False),
        ("2025-07-15T12:30:00Z", "2025-07-15T12:30:00Z", "macro_cpi", 5, "seed", "US CPI release", True, False),
        ("2025-07-30T18:00:00Z", "2025-07-30T18:00:00Z", "macro_fomc", 5, "seed", "FOMC decision", True, False),
        ("2025-08-01T12:30:00Z", "2025-08-01T12:30:00Z", "macro_jobs", 4, "seed", "US employment situation release", True, False),
        ("2025-08-12T12:30:00Z", "2025-08-12T12:30:00Z", "macro_cpi", 5, "seed", "US CPI release", True, False),
        ("2025-09-05T12:30:00Z", "2025-09-05T12:30:00Z", "macro_jobs", 4, "seed", "US employment situation release", True, False),
        ("2025-09-11T12:30:00Z", "2025-09-11T12:30:00Z", "macro_cpi", 5, "seed", "US CPI release", True, False),
        ("2025-09-17T18:00:00Z", "2025-09-17T18:00:00Z", "macro_fomc", 5, "seed", "FOMC decision and projections", True, False),
        ("2025-10-03T12:30:00Z", "2025-10-03T12:30:00Z", "macro_jobs", 4, "seed", "US employment situation release", True, False),
        ("2025-10-15T12:30:00Z", "2025-10-15T12:30:00Z", "macro_cpi", 5, "seed", "US CPI release", True, False),
        ("2025-10-29T18:00:00Z", "2025-10-29T18:00:00Z", "macro_fomc", 5, "seed", "FOMC decision", True, False),
        ("2025-11-07T13:30:00Z", "2025-11-07T13:30:00Z", "macro_jobs", 4, "seed", "US employment situation release", True, False),
        ("2025-11-13T13:30:00Z", "2025-11-13T13:30:00Z", "macro_cpi", 5, "seed", "US CPI release", True, False),
        ("2025-12-05T13:30:00Z", "2025-12-05T13:30:00Z", "macro_jobs", 4, "seed", "US employment situation release", True, False),
        ("2025-12-10T13:30:00Z", "2025-12-10T13:30:00Z", "macro_cpi", 5, "seed", "US CPI release", True, False),
        ("2025-12-10T19:00:00Z", "2025-12-10T19:00:00Z", "macro_fomc", 5, "seed", "FOMC decision and projections", True, False),
        ("2026-01-09T13:30:00Z", "2026-01-09T13:30:00Z", "macro_jobs", 4, "seed", "US employment situation release", True, False),
        ("2026-01-13T13:30:00Z", "2026-01-13T13:30:00Z", "macro_cpi", 5, "seed", "US CPI release", True, False),
        ("2026-01-28T19:00:00Z", "2026-01-28T19:00:00Z", "macro_fomc", 5, "seed", "FOMC decision", True, False),
        ("2026-02-06T13:30:00Z", "2026-02-06T13:30:00Z", "macro_jobs", 4, "seed", "US employment situation release", True, False),
        ("2026-02-12T13:30:00Z", "2026-02-12T13:30:00Z", "macro_cpi", 5, "seed", "US CPI release", True, False),
        ("2026-03-06T13:30:00Z", "2026-03-06T13:30:00Z", "macro_jobs", 4, "seed", "US employment situation release", True, False),
        ("2026-03-18T18:00:00Z", "2026-03-18T18:00:00Z", "macro_fomc", 5, "seed", "FOMC decision and projections", True, False),
        ("2026-04-03T12:30:00Z", "2026-04-03T12:30:00Z", "macro_jobs", 4, "seed", "US employment situation release", True, False),
        ("2026-04-10T12:30:00Z", "2026-04-10T12:30:00Z", "macro_cpi", 5, "seed", "US CPI release", True, False),
        ("2026-05-01T12:30:00Z", "2026-05-01T12:30:00Z", "macro_jobs", 4, "seed", "US employment situation release", True, False),
        ("2026-05-06T18:00:00Z", "2026-05-06T18:00:00Z", "macro_fomc", 5, "seed", "FOMC decision", True, False),
        ("2026-05-12T12:30:00Z", "2026-05-12T12:30:00Z", "macro_cpi", 5, "seed", "US CPI release", True, False),
        ("2026-06-05T12:30:00Z", "2026-06-05T12:30:00Z", "macro_jobs", 4, "seed", "US employment situation release", True, False),
        ("2026-06-10T12:30:00Z", "2026-06-10T12:30:00Z", "macro_cpi", 5, "seed", "US CPI release", True, False),
    ]
    return normalize_fundamental_events(pd.DataFrame(rows, columns=EVENT_COLUMNS))


def normalize_fundamental_events(events: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in EVENT_COLUMNS if column not in events.columns]
    if missing:
        raise ValueError(f"Missing fundamental event columns: {missing}")
    normalized = events[EVENT_COLUMNS].copy()
    normalized["event_time_utc"] = [_parse_utc(value, "event_time_utc") for value in normalized["event_time_utc"]]
    normalized["known_time_utc"] = [_parse_utc(value, "known_time_utc") for value in normalized["known_time_utc"]]
    normalized["category"] = normalized["category"].astype(str)
    normalized["severity"] = normalized["severity"].astype(int)
    normalized["source"] = normalized["source"].astype(str)
    normalized["title"] = normalized["title"].astype(str)
    normalized["is_scheduled"] = normalized["is_scheduled"].astype(bool)
    normalized["is_surprise"] = normalized["is_surprise"].astype(bool)
    if (normalized["known_time_utc"] < normalized["event_time_utc"]).any():
        invalid = normalized[normalized["known_time_utc"] < normalized["event_time_utc"]].iloc[0]
        raise ValueError(f"known_time_utc cannot be earlier than event_time_utc for {invalid['title']!r}")
    return normalized.sort_values(["event_time_utc", "known_time_utc"]).reset_index(drop=True)


def load_fundamental_events(config: dict[str, Any]) -> pd.DataFrame:
    blackout_config = config.get("fundamental_blackout", {})
    events_path = blackout_config.get("events_path")
    events: list[pd.DataFrame] = []
    if bool(blackout_config.get("use_default_seed", True)):
        events.append(default_fundamental_events())
    if events_path:
        path = Path(events_path)
        if not path.is_absolute():
            path = project_path(str(path))
        if path.exists():
            events.append(normalize_fundamental_events(pd.read_csv(path)))
        else:
            raise FileNotFoundError(f"Configured fundamental events_path does not exist: {path}")
    if not events:
        raise ValueError("No fundamental events configured")
    return normalize_fundamental_events(pd.concat(events, ignore_index=True).drop_duplicates())


def _event_allowed(event: pd.Series, blackout_config: dict[str, Any]) -> bool:
    categories = set(str(value) for value in blackout_config.get("categories", []))
    min_severity = int(blackout_config.get("min_severity", 1))
    return (not categories or str(event["category"]) in categories) and int(event["severity"]) >= min_severity


def build_blackout_windows(events: pd.DataFrame, blackout_config: dict[str, Any], mode: str) -> pd.DataFrame:
    if mode not in {"realistic", "oracle"}:
        raise ValueError("mode must be realistic or oracle")
    events = normalize_fundamental_events(events)
    pre = pd.Timedelta(hours=float(blackout_config.get("pre_event_hours", 12)))
    post = pd.Timedelta(hours=float(blackout_config.get("post_event_hours", 6)))
    surprise_reaction = pd.Timedelta(hours=float(blackout_config.get("surprise_reaction_hours", 24)))
    oracle_pre = pd.Timedelta(hours=float(blackout_config.get("oracle_pre_event_hours", 12)))
    rows: list[dict[str, object]] = []
    for _, event in events.iterrows():
        if not _event_allowed(event, blackout_config):
            continue
        if bool(event["is_scheduled"]):
            start = event["event_time_utc"] - pre
            end = event["event_time_utc"] + post
        elif mode == "realistic":
            start = event["known_time_utc"]
            end = event["known_time_utc"] + surprise_reaction
        else:
            start = event["event_time_utc"] - oracle_pre
            end = event["event_time_utc"] + post
        if end <= start:
            raise ValueError("blackout window end must be after start")
        rows.append({"mode": mode, "window_start_utc": start, "window_end_utc": end, **event.to_dict()})
    if not rows:
        return pd.DataFrame(columns=WINDOW_COLUMNS)
    return pd.DataFrame(rows, columns=WINDOW_COLUMNS).sort_values("window_start_utc").reset_index(drop=True)


def blackout_mask(index: pd.Index, windows: pd.DataFrame) -> pd.Series:
    if index.tz is None:
        raise ValueError("blackout index must be timezone-aware")
    mask = pd.Series(False, index=index, dtype=bool)
    if windows.empty:
        mask.name = "blackout"
        return mask
    for _, window in windows.iterrows():
        start = _parse_utc(window["window_start_utc"], "window_start_utc")
        end = _parse_utc(window["window_end_utc"], "window_end_utc")
        mask |= (index >= start) & (index <= end)
    mask.name = "blackout"
    return mask


def build_blackout_bundle(
    index: pd.Index,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    events = load_fundamental_events(config)
    blackout_config = config.get("fundamental_blackout", {})
    realistic_windows = build_blackout_windows(events, blackout_config, "realistic")
    oracle_windows = build_blackout_windows(events, blackout_config, "oracle")
    windows = pd.concat([realistic_windows, oracle_windows], ignore_index=True)
    masks = {
        "realistic": blackout_mask(index, realistic_windows),
        "oracle": blackout_mask(index, oracle_windows),
    }
    return events, windows, masks
