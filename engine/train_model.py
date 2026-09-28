"""
Train and evaluate AML detection models.

Key design choices that matter for a real AML deployment, not just a toy
benchmark:
  - TIME-BASED split (train on earlier transactions, test on later ones) —
    a random split would leak future behaviour into training and wildly
    overstate performance, since this is a live streaming problem.
  - Precision-Recall AUC as the primary metric, not accuracy or plain
    ROC-AUC — with ~1.6% positives, a model that predicts "never laundering"
    is 98%+ "accurate" and useless. PR-AUC and recall-at-fixed-alert-budget
    are what an investigations team actually cares about.
  - class_weight='balanced' to counter the imbalance without throwing away
    data via undersampling.
  - An IsolationForest anomaly detector is trained alongside the supervised
    models as an unsupervised sanity check / second opinion layer, since in
    production you rarely have ground-truth labels this clean.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                              roc_curve, confusion_matrix, classification_report, brier_score_loss)
from build_features import FEATURE_COLS

BASE_DIR = Path(__file__).resolve().parent

RANDOM_STATE = 42


def load_features():
    df = pd.read_csv(BASE_DIR / "features.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    return df


def time_split3(df, calib_frac=0.15, test_frac=0.25):
    """Three-way TIME-ORDERED split: train -> calibrate -> test.
    The calibration slice must be data the model never trained on, and it
    must come before the test slice — calibrating on future data (or on
    training data itself) would let the calibrator quietly leak
    information and overstate how trustworthy the resulting scores are."""
    train_frac = 1 - calib_frac - test_frac
    q1 = df["timestamp"].quantile(train_frac)
    q2 = df["timestamp"].quantile(train_frac + calib_frac)
    train = df[df["timestamp"] < q1]
    calib = df[(df["timestamp"] >= q1) & (df["timestamp"] < q2)]
    test = df[df["timestamp"] >= q2]
    return train, calib, test, q1, q2


def prep_xy(df):
    X = df[FEATURE_COLS].copy()
    X["hours_since_last_deposit"] = X["hours_since_last_deposit"].fillna(999)  # "no recent deposit"
    y = df["is_laundering"].values
    return X, y


def evaluate(name, y_true, y_score, y_pred=None):
    roc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    brier = brier_score_loss(y_true, y_score)
    print(f"\n=== {name} ===")
    print(f"ROC-AUC: {roc:.4f}   PR-AUC: {pr_auc:.4f}   Brier score: {brier:.4f}   base rate: {y_true.mean():.4f}")
    if y_pred is not None:
        print(classification_report(y_true, y_pred, digits=3, target_names=["normal", "laundering"]))
        print("Confusion matrix [rows=true, cols=pred]:")
        print(confusion_matrix(y_true, y_pred))
    return roc, pr_auc, brier


def alert_budget_recall(y_true, y_score, budgets=(0.005, 0.01, 0.02, 0.05)):
    """What fraction of true laundering txns do we catch if investigators can
    only review the top X% highest-scored transactions? This is the metric
    that actually matters operationally — analyst time is the bottleneck."""
    order = np.argsort(-y_score)
    n = len(y_score)
    rows = []
    for b in budgets:
        k = max(1, int(n * b))
        top_idx = order[:k]
        caught = y_true[top_idx].sum()
        recall = caught / y_true.sum()
        precision = caught / k
        rows.append((b, k, int(caught), recall, precision))
    out = pd.DataFrame(rows, columns=["alert_budget_%", "n_alerts", "true_positives_caught",
                                       "recall", "precision"])
    out["alert_budget_%"] = (out["alert_budget_%"] * 100).round(2)
    out["recall"] = (out["recall"] * 100).round(1)
    out["precision"] = (out["precision"] * 100).round(1)
    return out


def hard_negative_report(y_score, patterns, threshold, budget_idx=None):
    """The real test of the harder-negatives fix: for each legitimately-busy
    archetype, what fraction get wrongly flagged? This is a false-positive
    rate broken down by *why* the customer looks unusual, which a single
    aggregate precision number hides completely."""
    rows = []
    for pat in sorted(set(patterns)):
        if not pat.startswith("legit_"):
            continue
        mask = patterns == pat
        n = mask.sum()
        flagged = (y_score[mask] >= threshold).sum()
        rows.append((pat, n, flagged, 100 * flagged / n if n else 0))
    return pd.DataFrame(rows, columns=["hard_negative_pattern", "n_txns", "flagged", "false_positive_rate_%"])


def main():
    df = load_features()
    train, calib, test, q1, q2 = time_split3(df)
    print(f"Split cutoffs: train < {q1}  |  calibrate < {q2}  |  test >= {q2}")
    print(f"Train: {len(train)} txns ({train['is_laundering'].sum()} positive)")
    print(f"Calib: {len(calib)} txns ({calib['is_laundering'].sum()} positive)")
    print(f"Test:  {len(test)} txns ({test['is_laundering'].sum()} positive)")

    X_train, y_train = prep_xy(train)
    X_calib, y_calib = prep_xy(calib)
    X_test, y_test = prep_xy(test)
    test_patterns = test["pattern"].values

    # ---- Random Forest (raw, then calibrated) ---------------------------
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_raw_scores = rf.predict_proba(X_test)[:, 1]
    evaluate("Random Forest (raw, uncalibrated)", y_test, rf_raw_scores, (rf_raw_scores >= 0.5).astype(int))

    rf_cal = CalibratedClassifierCV(FrozenEstimator(rf), method="isotonic")
    rf_cal.fit(X_calib, y_calib)
    rf_scores = rf_cal.predict_proba(X_test)[:, 1]
    rf_pred = (rf_scores >= 0.5).astype(int)
    evaluate("Random Forest (isotonic-calibrated)", y_test, rf_scores, rf_pred)

    # ---- Gradient Boosting (raw, then calibrated) ------------------------
    gb = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.08,
        class_weight="balanced", random_state=RANDOM_STATE,
    )
    gb.fit(X_train, y_train)
    gb_cal = CalibratedClassifierCV(FrozenEstimator(gb), method="isotonic")
    gb_cal.fit(X_calib, y_calib)
    gb_scores = gb_cal.predict_proba(X_test)[:, 1]
    gb_pred = (gb_scores >= 0.5).astype(int)
    evaluate("Gradient Boosting (isotonic-calibrated)", y_test, gb_scores, gb_pred)

    # ---- Isolation Forest (unsupervised, no calibration target) ---------
    iso = IsolationForest(n_estimators=300, contamination=0.02, random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_train)
    iso_scores = -iso.score_samples(X_test)
    evaluate("Isolation Forest (unsupervised)", y_test, iso_scores)

    # ---- Operational view: alert-budget recall table --------------------
    print("\n--- Random Forest (calibrated): recall at fixed investigator alert-budget ---")
    rf_budget = alert_budget_recall(y_test, rf_scores)
    print(rf_budget.to_string(index=False))

    print("\n--- Gradient Boosting (calibrated): recall at fixed investigator alert-budget ---")
    gb_budget = alert_budget_recall(y_test, gb_scores)
    print(gb_budget.to_string(index=False))

    # ---- Recall broken down by laundering pattern -------------------------
    print("\n--- Recall by laundering pattern (Random Forest, calibrated, threshold=0.5) ---")
    pattern_rows = []
    for pat in sorted(set(test_patterns)):
        if pat == "normal" or pat.startswith("legit_"):
            continue
        mask = test_patterns == pat
        n = mask.sum()
        caught = ((rf_pred == 1) & mask).sum()
        avg_score = rf_scores[mask].mean()
        pattern_rows.append((pat, n, caught, 100 * caught / n if n else 0, avg_score))
    pat_df = pd.DataFrame(pattern_rows, columns=["pattern", "n_txns", "caught_at_0.5",
                                                  "recall_%", "avg_risk_score"])
    print(pat_df.round(3).to_string(index=False))

    # ---- THE KEY CHECK: false-positive rate on hard negatives -----------
    print("\n--- False-positive rate on legitimately-busy customers (threshold=0.5) ---")
    hn_report = hard_negative_report(rf_scores, test_patterns, threshold=0.5)
    print(hn_report.round(1).to_string(index=False))

    print("\n--- Same, at the top-2%-alert-budget cutoff ---")
    order = np.argsort(-rf_scores)
    k = max(1, int(len(rf_scores) * 0.02))
    budget_threshold = rf_scores[order[k - 1]]
    hn_report_budget = hard_negative_report(rf_scores, test_patterns, threshold=budget_threshold)
    print(hn_report_budget.round(1).to_string(index=False))

    # ---- Feature importance ---------------------------------------------
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\n--- Random Forest feature importances ---")
    print(importances.to_string())

    # ================= PLOTS =================
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))

    # ROC curves
    ax = axes[0, 0]
    for name, scores in [("RF (calibrated)", rf_scores), ("GB (calibrated)", gb_scores),
                          ("Isolation Forest", iso_scores)]:
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc = roc_auc_score(y_test, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend(fontsize=8)

    # PR curves
    ax = axes[0, 1]
    for name, scores in [("RF (calibrated)", rf_scores), ("GB (calibrated)", gb_scores),
                          ("Isolation Forest", iso_scores)]:
        prec, rec, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    ax.axhline(y_test.mean(), color="k", linestyle="--", alpha=0.3, label="Random baseline")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve"); ax.legend(fontsize=8)

    # Reliability diagram: raw vs calibrated
    ax = axes[0, 2]
    for name, scores in [("RF raw", rf_raw_scores), ("RF calibrated", rf_scores)]:
        frac_pos, mean_pred = calibration_curve(y_test, scores, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", markersize=4, label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted risk score"); ax.set_ylabel("Actual fraction laundering")
    ax.set_title("Reliability Diagram — is a 0.7 score really 70%?"); ax.legend(fontsize=7)

    # Feature importance
    ax = axes[1, 0]
    top_imp = importances.head(12)[::-1]
    ax.barh(top_imp.index, top_imp.values, color="#2b6cb0")
    ax.set_title("Top 12 Feature Importances (Random Forest)")
    ax.set_xlabel("Importance")

    # Alert-budget recall
    ax = axes[1, 1]
    ax.plot(rf_budget["alert_budget_%"], rf_budget["recall"], marker="o", label="Random Forest")
    ax.plot(gb_budget["alert_budget_%"], gb_budget["recall"], marker="s", label="Gradient Boosting")
    ax.set_xlabel("Alert budget (% of transactions reviewed)")
    ax.set_ylabel("% of laundering transactions caught (recall)")
    ax.set_title("Investigator Workload vs. Detection Rate")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # False positive rate on hard negatives
    ax = axes[1, 2]
    labels = [p.replace("legit_", "").replace("_", "\n") for p in hn_report["hard_negative_pattern"]]
    ax.bar(labels, hn_report["false_positive_rate_%"], color="#c05621")
    ax.set_title("False-Positive Rate on Legit-Busy Customers")
    ax.set_ylabel("% wrongly flagged (threshold=0.5)")
    ax.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    plt.savefig(BASE_DIR / "model_evaluation.png", dpi=140)
    print("\nSaved model_evaluation.png")

    # Save scored test set for inspection
    test_out = test[["transaction_id", "customer_id", "timestamp", "amount", "pattern",
                      "is_laundering"]].copy()
    test_out["rf_score_raw"] = rf_raw_scores
    test_out["rf_score_calibrated"] = rf_scores
    test_out["gb_score_calibrated"] = gb_scores
    test_out["iso_anomaly_score"] = iso_scores
    test_out.sort_values("rf_score_calibrated", ascending=False).to_csv("scored_transactions.csv", index=False)
    print("Saved scored_transactions.csv (test set, ranked by calibrated RF risk score)")


if __name__ == "__main__":
    main()
