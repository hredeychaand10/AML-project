"""
Synthetic bank-transaction data generator for an AML detection model.

Simulates ~2,500 customer accounts transacting over 120 days, with a small
fraction of accounts carrying out realistic money-laundering patterns drawn
from the red-flag taxonomy (structuring, smurfing, rapid fund movement /
pass-through, fan-in/fan-out, circular movement, geographic anomaly,
sudden behavioural change). Everything else is "normal" background noise.

Output: raw_transactions.csv (one row per transaction, ground-truth label
is_laundering included for training/evaluation only — a real system would
never have this at scoring time).
"""

from datetime import datetime, timedelta
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
rng = np.random.default_rng(42)

N_CUSTOMERS = 2500
N_DAYS = 120
START_DATE = datetime(2026, 1, 1)
CTR_THRESHOLD = 1_000_000  # illustrative reporting threshold (INR 10L, PMLA-style)

COUNTRIES = ["IN", "US", "UK", "AE", "SG", "DE", "FR", "JP"]
HIGH_RISK_COUNTRIES = {"KP", "IR", "MM", "AF"}  # sanctioned / high-risk corridor
ALL_COUNTRIES = COUNTRIES + list(HIGH_RISK_COUNTRIES)
COUNTRY_WEIGHTS = [0.55, 0.12, 0.08, 0.08, 0.06, 0.04, 0.04, 0.03] + [0.0] * len(HIGH_RISK_COUNTRIES)
COUNTRY_WEIGHTS = np.array(COUNTRY_WEIGHTS) / sum(COUNTRY_WEIGHTS)

OCCUPATIONS = ["salaried", "business_owner", "trader", "freelancer", "retired", "student"]
CHANNELS = ["NEFT", "IMPS", "UPI", "RTGS", "cash_deposit", "wire_intl"]


def make_customers(n):
    home_country = rng.choice(COUNTRIES, size=n, p=COUNTRY_WEIGHTS[: len(COUNTRIES)] / COUNTRY_WEIGHTS[: len(COUNTRIES)].sum())
    occupation = rng.choice(OCCUPATIONS, size=n)
    # baseline "typical" transaction size varies a lot by occupation/wealth
    base_amount = np.where(
        occupation == "business_owner", rng.lognormal(11.5, 0.9, n),
        np.where(occupation == "trader", rng.lognormal(11.0, 1.0, n),
                 rng.lognormal(9.5, 0.7, n)),
    )
    base_freq = rng.poisson(6, n) + 1  # avg transactions/month
    is_pep = rng.random(n) < 0.01
    sanctions_match = rng.random(n) < 0.003
    high_risk_entity = (rng.random(n) < 0.02) | sanctions_match
    account_age_days = rng.integers(5, 2500, n)

    df = pd.DataFrame({
        "customer_id": [f"C{100000+i}" for i in range(n)],
        "home_country": home_country,
        "occupation": occupation,
        "baseline_avg_amount": base_amount,
        "baseline_monthly_freq": base_freq,
        "is_pep": is_pep,
        "sanctions_match": sanctions_match,
        "high_risk_entity": high_risk_entity,
        "account_age_days": account_age_days,
    })
    return df


def normal_transactions_for_customer(cust, n_days=N_DAYS):
    """Generate background-normal activity for one customer over the period."""
    rows = []
    n_txn = max(1, int(rng.poisson(cust.baseline_monthly_freq * n_days / 30)))
    # a stable little circle of counterparties this customer usually deals with
    n_regulars = rng.integers(2, 8)
    regulars = [f"P{uuid.uuid4().hex[:8]}" for _ in range(n_regulars)]

    for _ in range(n_txn):
        day_offset = rng.integers(0, n_days)
        ts = START_DATE + timedelta(days=int(day_offset), hours=int(rng.integers(7, 22)),
                                     minutes=int(rng.integers(0, 60)))
        amount = max(500, rng.lognormal(np.log(max(cust.baseline_avg_amount, 1000)), 0.5))
        counterparty = rng.choice(regulars) if rng.random() < 0.85 else f"P{uuid.uuid4().hex[:8]}"
        direction = rng.choice(["in", "out"])
        cp_country = cust.home_country if rng.random() < 0.9 else rng.choice(COUNTRIES)
        rows.append({
            "customer_id": cust.customer_id,
            "timestamp": ts,
            "amount": round(amount, 2),
            "direction": direction,
            "counterparty_id": counterparty,
            "counterparty_country": cp_country,
            "channel": rng.choice(CHANNELS, p=[0.25, 0.25, 0.3, 0.1, 0.07, 0.03]),
            "is_laundering": 0,
            "pattern": "normal",
        })
    return rows


def inject_structuring(cust, n_days=N_DAYS):
    """Multiple transactions just under the reporting threshold, over several days."""
    rows = []
    n_txn = rng.integers(4, 9)
    start = rng.integers(0, n_days - 10)
    cp = f"P{uuid.uuid4().hex[:8]}"
    for i in range(n_txn):
        ts = START_DATE + timedelta(days=int(start + i), hours=int(rng.integers(9, 18)))
        amount = CTR_THRESHOLD * rng.uniform(0.85, 0.98)
        rows.append({
            "customer_id": cust.customer_id, "timestamp": ts, "amount": round(amount, 2),
            "direction": "in", "counterparty_id": cp, "counterparty_country": cust.home_country,
            "channel": "cash_deposit", "is_laundering": 1, "pattern": "structuring",
        })
    return rows


def inject_smurfing(cust, n_days=N_DAYS):
    """Many distinct small senders funnel into this account within a short window."""
    rows = []
    start = rng.integers(0, n_days - 5)
    n_senders = rng.integers(8, 20)
    for _ in range(n_senders):
        ts = START_DATE + timedelta(days=int(start + rng.integers(0, 3)), hours=int(rng.integers(8, 20)))
        amount = rng.uniform(15000, 60000)
        rows.append({
            "customer_id": cust.customer_id, "timestamp": ts, "amount": round(amount, 2),
            "direction": "in", "counterparty_id": f"P{uuid.uuid4().hex[:8]}",
            "counterparty_country": cust.home_country, "channel": rng.choice(["UPI", "IMPS"]),
            "is_laundering": 1, "pattern": "smurfing",
        })
    # then consolidated rapid outflow
    ts_out = START_DATE + timedelta(days=int(start + 3), hours=int(rng.integers(8, 20)))
    rows.append({
        "customer_id": cust.customer_id, "timestamp": ts_out,
        "amount": round(sum(r["amount"] for r in rows) * rng.uniform(0.9, 0.99), 2),
        "direction": "out", "counterparty_id": f"P{uuid.uuid4().hex[:8]}",
        "counterparty_country": rng.choice(list(HIGH_RISK_COUNTRIES) + COUNTRIES),
        "channel": "wire_intl", "is_laundering": 1, "pattern": "smurfing_outflow",
    })
    return rows


def inject_rapid_movement(cust, n_days=N_DAYS):
    """Large deposit followed within hours by near-total withdrawal (pass-through)."""
    rows = []
    start = rng.integers(0, n_days - 2)
    amount = rng.uniform(300000, 3000000)
    ts_in = START_DATE + timedelta(days=int(start), hours=int(rng.integers(8, 14)))
    ts_out = ts_in + timedelta(hours=float(rng.uniform(0.5, 6)))
    cp_in = f"P{uuid.uuid4().hex[:8]}"
    cp_out = f"P{uuid.uuid4().hex[:8]}"
    rows.append({"customer_id": cust.customer_id, "timestamp": ts_in, "amount": round(amount, 2),
                 "direction": "in", "counterparty_id": cp_in, "counterparty_country": cust.home_country,
                 "channel": "RTGS", "is_laundering": 1, "pattern": "rapid_movement"})
    rows.append({"customer_id": cust.customer_id, "timestamp": ts_out, "amount": round(amount * rng.uniform(0.9, 0.99), 2),
                 "direction": "out", "counterparty_id": cp_out,
                 "counterparty_country": rng.choice(list(HIGH_RISK_COUNTRIES) + COUNTRIES),
                 "channel": "wire_intl", "is_laundering": 1, "pattern": "rapid_movement"})
    return rows


def inject_fan_out(cust, n_days=N_DAYS):
    """One account sending to many unique counterparties in a short burst."""
    rows = []
    start = rng.integers(0, n_days - 3)
    n_targets = rng.integers(10, 25)
    for _ in range(n_targets):
        ts = START_DATE + timedelta(days=int(start + rng.integers(0, 2)), hours=int(rng.integers(8, 20)))
        amount = rng.uniform(20000, 90000)
        rows.append({
            "customer_id": cust.customer_id, "timestamp": ts, "amount": round(amount, 2),
            "direction": "out", "counterparty_id": f"P{uuid.uuid4().hex[:8]}",
            "counterparty_country": rng.choice(COUNTRIES), "channel": rng.choice(["UPI", "IMPS"]),
            "is_laundering": 1, "pattern": "fan_out",
        })
    return rows


def inject_circular(cust, n_days=N_DAYS):
    """A -> B -> C -> back to A, within days."""
    rows = []
    start = rng.integers(0, n_days - 5)
    b = f"P{uuid.uuid4().hex[:8]}"
    c = f"P{uuid.uuid4().hex[:8]}"
    amount = rng.uniform(200000, 1500000)
    ts0 = START_DATE + timedelta(days=int(start), hours=10)
    rows.append({"customer_id": cust.customer_id, "timestamp": ts0, "amount": round(amount, 2),
                 "direction": "out", "counterparty_id": b, "counterparty_country": cust.home_country,
                 "channel": "NEFT", "is_laundering": 1, "pattern": "circular"})
    ts2 = ts0 + timedelta(days=float(rng.uniform(2, 4)))
    rows.append({"customer_id": cust.customer_id, "timestamp": ts2, "amount": round(amount * rng.uniform(0.9, 0.97), 2),
                 "direction": "in", "counterparty_id": c, "counterparty_country": cust.home_country,
                 "channel": "NEFT", "is_laundering": 1, "pattern": "circular"})
    return rows


def inject_dormant_reactivation(cust, n_days=N_DAYS):
    """Long-dormant account suddenly bursts into high-value activity."""
    rows = []
    start = rng.integers(n_days - 20, n_days - 2)
    for i in range(rng.integers(3, 7)):
        ts = START_DATE + timedelta(days=int(start + i), hours=int(rng.integers(8, 20)))
        amount = rng.uniform(500000, 2000000)
        rows.append({
            "customer_id": cust.customer_id, "timestamp": ts, "amount": round(amount, 2),
            "direction": rng.choice(["in", "out"]), "counterparty_id": f"P{uuid.uuid4().hex[:8]}",
            "counterparty_country": rng.choice(COUNTRIES), "channel": "RTGS",
            "is_laundering": 1, "pattern": "dormant_reactivation",
        })
    return rows


def inject_smart_launderer(cust, n_days=N_DAYS):
    """
    An adversarial launderer who knows roughly what a detection engine looks
    for and deliberately stays under every individual threshold:

      - amounts kept close to the customer's OWN baseline (1.1-1.9x, well
        within normal noise) instead of a sudden multi-x spike
      - never near the CTR threshold
      - one layering hop every 3-5 days, not a burst -> stays under the
        7-day frequency window and the 24h fan-in/fan-out window
      - reuses a small set of "regular-looking" counterparties instead of
        always paying new/unique ones -> low new-counterparty ratio
      - waits 3-8 days between an inbound deposit and moving it back out
        -> defeats the <6h rapid-passthrough / velocity flag
      - domestic counterparties only, business hours only, normal channels
        (NEFT/UPI) instead of cash/wire -> avoids geo, odd-hour and channel
        flags
      - account is already active (no dormant-reactivation signature)

    The illicit character only shows up in the CUMULATIVE picture over ~6-8
    weeks: an account quietly layering far more total volume through a
    small reused circle of counterparties, and a final destination that is
    unrelated to any of the customer's normal counterparties.
    """
    rows = []
    layering_partners = [f"P{uuid.uuid4().hex[:8]}" for _ in range(3)]
    final_destination = f"P{uuid.uuid4().hex[:8]}"
    start = rng.integers(0, max(1, n_days - 50))
    n_layers = rng.integers(10, 16)
    day_cursor = start

    for i in range(n_layers):
        day_cursor += rng.integers(3, 5)
        if day_cursor >= n_days - 8:
            break
        ts_in = START_DATE + timedelta(days=int(day_cursor), hours=int(rng.integers(10, 16)))
        amount = cust.baseline_avg_amount * rng.uniform(1.1, 1.9)
        in_partner = rng.choice(layering_partners) if rng.random() < 0.7 else f"P{uuid.uuid4().hex[:8]}"
        rows.append({
            "customer_id": cust.customer_id, "timestamp": ts_in, "amount": round(amount, 2),
            "direction": "in", "counterparty_id": in_partner, "counterparty_country": cust.home_country,
            "channel": rng.choice(["NEFT", "UPI", "IMPS"]), "is_laundering": 1, "pattern": "smart_layering",
        })

        # wait several days before moving it onward -> defeats rapid-passthrough window
        delay_days = rng.uniform(3, 8)
        ts_out = ts_in + timedelta(days=float(delay_days), hours=float(rng.uniform(-2, 2)))
        out_day = (ts_out - START_DATE).days
        if out_day < n_days:
            out_partner = final_destination if rng.random() < 0.3 else rng.choice(layering_partners)
            rows.append({
                "customer_id": cust.customer_id, "timestamp": ts_out,
                "amount": round(amount * rng.uniform(0.93, 0.99), 2),
                "direction": "out", "counterparty_id": out_partner, "counterparty_country": cust.home_country,
                "channel": rng.choice(["NEFT", "UPI", "IMPS"]), "is_laundering": 1, "pattern": "smart_layering",
            })
    return rows


INJECTORS = [inject_structuring, inject_smurfing, inject_rapid_movement,
             inject_fan_out, inject_circular, inject_dormant_reactivation]

SMART_INJECTORS = [inject_smart_launderer]


# ============================================================
# HARD NEGATIVES — legitimately busy customers who are NOT
# laundering, but whose raw activity looks superficially similar
# to some of the patterns above (high fan-out, high frequency,
# recurring cross-border inflow, near-threshold cash deposits).
# The real distinguishing signal is that this activity is spread
# evenly across the whole period as an ongoing pattern of life,
# not concentrated into a short burst the way the injected
# laundering episodes are. Without these, the model can cheat by
# learning "high fan-out = laundering" instead of learning the
# actual shape of the behaviour.
# ============================================================

def legit_business_hub(cust, n_days=N_DAYS):
    """Small business owner paying a real, stable set of vendors/staff
    every month — naturally high fan-out, but spread evenly, same
    payees recurring, business-hours, domestic."""
    rows = []
    n_vendors = rng.integers(10, 22)
    vendors = [f"P{uuid.uuid4().hex[:8]}" for _ in range(n_vendors)]
    n_txn = rng.integers(35, 70)
    for _ in range(n_txn):
        day = rng.integers(0, n_days)
        ts = START_DATE + timedelta(days=int(day), hours=int(rng.integers(9, 19)))
        amount = max(2000, rng.lognormal(np.log(max(cust.baseline_avg_amount, 5000)), 0.6))
        rows.append({
            "customer_id": cust.customer_id, "timestamp": ts, "amount": round(amount, 2),
            "direction": "out", "counterparty_id": rng.choice(vendors),
            "counterparty_country": cust.home_country, "channel": rng.choice(["NEFT", "IMPS", "UPI"]),
            "is_laundering": 0, "pattern": "legit_business_hub",
        })
    return rows


def legit_remittance_family(cust, n_days=N_DAYS):
    """Retired/salaried customer receiving recurring support from several
    family members abroad — real cross-border fan-in, but the same
    handful of relatives, roughly monthly, modest amounts."""
    rows = []
    n_relatives = rng.integers(4, 9)
    relatives = [f"P{uuid.uuid4().hex[:8]}" for _ in range(n_relatives)]
    relative_countries = list(rng.choice(COUNTRIES, size=n_relatives))
    n_rounds = n_days // 28
    for r in range(max(1, n_rounds)):
        base_day = r * 28 + rng.integers(0, 5)
        for i, rel in enumerate(relatives):
            if rng.random() < 0.6:  # not every relative sends every month
                day = base_day + rng.integers(0, 4)
                if day >= n_days:
                    continue
                ts = START_DATE + timedelta(days=int(day), hours=int(rng.integers(8, 21)))
                amount = rng.uniform(8000, 60000)
                rows.append({
                    "customer_id": cust.customer_id, "timestamp": ts, "amount": round(amount, 2),
                    "direction": "in", "counterparty_id": rel, "counterparty_country": relative_countries[i],
                    "channel": "wire_intl", "is_laundering": 0, "pattern": "legit_remittance_family",
                })
    return rows


def legit_gig_worker(cust, n_days=N_DAYS):
    """Freelancer paid by many different one-off clients — genuinely
    high new-counterparty ratio, but small amounts, spread thin,
    no rapid outflow afterward."""
    rows = []
    n_txn = rng.integers(20, 45)
    for _ in range(n_txn):
        day = rng.integers(0, n_days)
        ts = START_DATE + timedelta(days=int(day), hours=int(rng.integers(9, 22)))
        amount = rng.uniform(3000, 45000)
        rows.append({
            "customer_id": cust.customer_id, "timestamp": ts, "amount": round(amount, 2),
            "direction": "in", "counterparty_id": f"P{uuid.uuid4().hex[:8]}",
            "counterparty_country": cust.home_country, "channel": rng.choice(["UPI", "IMPS"]),
            "is_laundering": 0, "pattern": "legit_gig_worker",
        })
    return rows


def legit_retail_cash(cust, n_days=N_DAYS):
    """Small retail/shop owner depositing daily takings — genuinely,
    consistently near the reporting threshold because that's the
    real size of the business, every single day, not a one-off burst."""
    rows = []
    daily_amount = rng.uniform(CTR_THRESHOLD * 0.6, CTR_THRESHOLD * 0.95)
    for day in range(n_days):
        if rng.random() < 0.85:  # open most days
            ts = START_DATE + timedelta(days=int(day), hours=int(rng.integers(18, 21)))
            amount = daily_amount * rng.uniform(0.85, 1.05)
            rows.append({
                "customer_id": cust.customer_id, "timestamp": ts, "amount": round(min(amount, CTR_THRESHOLD * 0.98), 2),
                "direction": "in", "counterparty_id": f"P{uuid.uuid4().hex[:8]}",
                "counterparty_country": cust.home_country, "channel": "cash_deposit",
                "is_laundering": 0, "pattern": "legit_retail_cash",
            })
    return rows


LEGIT_HARD_NEGATIVES = [legit_business_hub, legit_remittance_family, legit_gig_worker, legit_retail_cash]


def generate():
    customers = make_customers(N_CUSTOMERS)
    all_rows = []

    # ~3% of customers get 1-2 "naive" laundering episodes (obvious patterns)
    naive_launderers = customers.sample(frac=0.03, random_state=42)
    remaining = customers[~customers["customer_id"].isin(naive_launderers["customer_id"])]
    # a separate, non-overlapping ~1.5% get the adversarial "smart" pattern
    smart_launderers = remaining.sample(frac=0.015, random_state=43)
    remaining2 = remaining[~remaining["customer_id"].isin(smart_launderers["customer_id"])]
    # a separate ~6% get a legitimately-busy hard-negative pattern —
    # deliberately a larger share than the laundering pool, so the model
    # can't just learn "busy = suspicious"
    legit_busy = remaining2.sample(frac=0.06, random_state=44)

    for cust in customers.itertuples():
        all_rows.extend(normal_transactions_for_customer(cust))

    for cust in naive_launderers.itertuples():
        n_patterns = rng.integers(1, 3)
        chosen = rng.choice(len(INJECTORS), size=n_patterns, replace=False)
        for idx in chosen:
            all_rows.extend(INJECTORS[idx](cust))

    for cust in smart_launderers.itertuples():
        all_rows.extend(inject_smart_launderer(cust))

    for cust in legit_busy.itertuples():
        archetype = LEGIT_HARD_NEGATIVES[rng.integers(0, len(LEGIT_HARD_NEGATIVES))]
        all_rows.extend(archetype(cust))

    txns = pd.DataFrame(all_rows)
    txns = txns.sort_values("timestamp").reset_index(drop=True)
    txns.insert(0, "transaction_id", [f"T{i:07d}" for i in range(len(txns))])

    customers.to_csv(BASE_DIR / "customers.csv", index=False)
    txns.to_csv(BASE_DIR / "raw_transactions.csv", index=False)

    print(f"Customers: {len(customers)}  (naive-laundering: {len(naive_launderers)}, "
          f"smart-laundering: {len(smart_launderers)}, legit-busy hard negatives: {len(legit_busy)})")
    print(f"Transactions: {len(txns)}")
    print(f"Laundering-labeled transactions: {txns['is_laundering'].sum()} "
          f"({100*txns['is_laundering'].mean():.2f}%)")
    print(txns['pattern'].value_counts())


if __name__ == "__main__":
    generate()
