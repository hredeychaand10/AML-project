# Warden: Real-Time AML Transaction Screening (prototype)

## Contents
- `engine/`  Python ML pipeline
  - `generate_data.py`  synthetic customers + transactions, injected laundering patterns
    (structuring, smurfing, rapid pass-through, fan-out, circular, dormant reactivation,
    "smart layering") plus legitimately-busy hard negatives
  - `build_features.py` rolling-window features (no lookahead)
  - `train_model.py`    Random Forest, Gradient Boosting (isotonic-calibrated), Isolation Forest;
    time-based train/calibrate/test split, alert-budget recall, false-positive rate on hard negatives
  - `run_all.sh`        runs all three in order
- `ui/index.html`  the Warden demo UI (single file, open in a browser)



## Notes
- The UI's risk score is a simplified weighted approximation of the model's feature
  importances, not the trained model itself. Wiring the real model in (via a backend
  API) is the next step.
- The Activity tab's database only works when the page is hosted as a Claude artifact.
  Opened locally, the Send flow works fully; the ledger simply stays empty.
- All data is synthetic. Watchlist screening is simulated.
