"""
Feature engineering for the AML model.

Every feature here is computable at scoring time using only information
available up to and including the current transaction (rolling windows on
past data) — no look-ahead into the future, since the whole point is
real-time / pre-transaction scoring.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CTR_THRESHOLD = 1_000_000
HIGH_RISK_COUNTRIES = {"KP", "IR", "MM", "AF"}


def _data_path(name: str) -> Path:
    return BASE_DIR / name


def load():
    txns = pd.read_csv(_data_path("raw_transactions.csv"))
    txns["timestamp"] = pd.to_datetime(txns["timestamp"], format="mixed")
    customers = pd.read_csv(_data_path("customers.csv"))
    txns = txns.merge(customers, on="customer_id", how="left")
    txns = txns.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
    return txns


def build_features(txns: pd.DataFrame) -> pd.DataFrame:
    df = txns.copy()

    # ---- 1. Profile-deviation features -----------------------------------
    df["amount_vs_baseline"] = df["amount"] / df["baseline_avg_amount"].clip(lower=1)

    # ---- 2. Structuring ----------------------------------------------------
    df["near_threshold"] = ((df["amount"] >= CTR_THRESHOLD * 0.8) &
                             (df["amount"] < CTR_THRESHOLD)).astype(int)

    # ---- rolling, per-customer, time-ordered window features --------------
    struct_counts = []
    fan_in_counts = []
    fan_out_counts = []
    uniq_cp_30d = []
    new_cp_ratio = []
    freq_change = []
    velocity_hours = []
    geo_mismatch = []
    high_risk_geo = []
    odd_hour = []

    for cust_id, g in df.groupby("customer_id", sort=False):
        g = g.sort_values("timestamp")
        idx = g.index
        times = g["timestamp"].values  # numpy datetime64 array
        times_ts = g["timestamp"]       # pandas Series of Timestamps, for comparisons
        amounts = g["amount"].values
        directions = g["direction"].values
        counterparties = g["counterparty_id"].values
        cp_countries = g["counterparty_country"].values
        home = g["home_country"].iloc[0]
        near_thr = g["near_threshold"].values

        seen_cp = set()
        last_in_time = None
        last_in_amount = 0

        times_np = times_ts.to_numpy()
        for i in range(len(g)):
            t = times_ts.iloc[i]
            window7 = (times_np >= np.datetime64(t - pd.Timedelta(days=7))) & (times_np <= np.datetime64(t))
            window1 = (times_np >= np.datetime64(t - pd.Timedelta(days=1))) & (times_np <= np.datetime64(t))
            window30 = (times_np >= np.datetime64(t - pd.Timedelta(days=30))) & (times_np <= np.datetime64(t))

            # structuring: count of near-threshold txns in trailing 7d (incl. current)
            struct_counts.append(int(near_thr[window7].sum()))

            # fan-in: distinct senders paying IN to this account in trailing 24h
            in_mask = window1 & (directions == "in")
            fan_in_counts.append(len(set(counterparties[in_mask])))

            # fan-out: distinct recipients this account paid OUT to in trailing 24h
            out_mask = window1 & (directions == "out")
            fan_out_counts.append(len(set(counterparties[out_mask])))

            # unique counterparties in trailing 30d
            uniq_cp_30d.append(len(set(counterparties[window30])))

            # new-counterparty ratio: is this counterparty new to the customer?
            is_new = 1 if counterparties[i] not in seen_cp else 0
            new_cp_ratio.append(is_new)
            seen_cp.add(counterparties[i])

            # frequency change: txns in trailing 7d vs customer's overall average weekly rate
            freq_change.append(int(window7.sum()))

            # velocity: hours since the most recent inbound deposit (pass-through signal)
            if directions[i] == "out" and last_in_time is not None:
                velocity_hours.append((t - last_in_time) / pd.Timedelta(hours=1))
            else:
                velocity_hours.append(np.nan)
            if directions[i] == "in":
                last_in_time = t
                last_in_amount = amounts[i]

            # geographic
            geo_mismatch.append(int(cp_countries[i] != home))
            high_risk_geo.append(int(cp_countries[i] in HIGH_RISK_COUNTRIES))

            # odd hour
            odd_hour.append(int(t.hour < 6 or t.hour > 22))

    df["structuring_count_7d"] = struct_counts
    df["fan_in_24h"] = fan_in_counts
    df["fan_out_24h"] = fan_out_counts
    df["unique_counterparties_30d"] = uniq_cp_30d
    df["is_new_counterparty"] = new_cp_ratio
    df["txn_count_7d"] = freq_change
    df["hours_since_last_deposit"] = velocity_hours
    df["rapid_passthrough"] = ((df["direction"] == "out") &
                                (df["hours_since_last_deposit"] <= 6)).astype(int)
    df["geo_mismatch"] = geo_mismatch
    df["high_risk_geo"] = high_risk_geo
    df["odd_hour"] = odd_hour

    # frequency vs customer's own baseline monthly rate (normalized to weekly)
    df["freq_vs_baseline"] = df["txn_count_7d"] / (df["baseline_monthly_freq"] / 4.0).clip(lower=0.5)

    # fan-in/fan-out combined burst score
    df["fan_in_out_combo"] = ((df["fan_in_24h"] >= 5) & (df["fan_out_24h"] >= 1)).astype(int)

    # dormant reactivation: long account age but first activity burst in the data window
    df["dormant_reactivation_flag"] = ((df["account_age_days"] > 365) &
                                        (df["txn_count_7d"] >= 4) &
                                        (df["amount_vs_baseline"] > 3)).astype(int)

    # contextual risk multipliers (static, from KYC/onboarding — not real-time triggers alone)
    df["risk_context_score"] = (df["is_pep"].astype(int) +
                                 df["sanctions_match"].astype(int) * 3 +
                                 df["high_risk_entity"].astype(int))

    df["channel_is_cash_or_wire"] = df["channel"].isin(["cash_deposit", "wire_intl"]).astype(int)

    return df


FEATURE_COLS = [
    "amount", "amount_vs_baseline", "near_threshold", "structuring_count_7d",
    "fan_in_24h", "fan_out_24h", "fan_in_out_combo", "unique_counterparties_30d",
    "is_new_counterparty", "txn_count_7d", "freq_vs_baseline",
    "hours_since_last_deposit", "rapid_passthrough", "geo_mismatch", "high_risk_geo",
    "odd_hour", "dormant_reactivation_flag", "risk_context_score",
    "channel_is_cash_or_wire", "account_age_days",
]


if __name__ == "__main__":
    txns = load()
    feat = build_features(txns)
    feat.to_csv(_data_path("features.csv"), index=False)
    print(feat[FEATURE_COLS + ["is_laundering", "timestamp"]].describe(include="all").T)
    print("\nSaved features.csv with", len(feat), "rows and", len(FEATURE_COLS), "features.")
