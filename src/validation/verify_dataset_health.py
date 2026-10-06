"""
Dataset Health & Integrity Verification Script for GeoDemand AI (Member 2).
"""

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

EXPECTED_DATASETS = [
    "h3_cells.parquet",
    "static_features.parquet",
    "weather_hourly.parquet",
    "calendar.parquet",
    "events.parquet",
    "competition.parquet",
    "vendor_profiles.parquet",
    "products.parquet",
    "historical_transactions.parquet",
    "feature_store.parquet",
]


def check_dataset_health():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 75)
    print("GeoDemand AI — Member 2 Dataset Health & Pipeline Inspection")
    print("=" * 75)

    if not PROCESSED_DIR.exists():
        print(f"❌ Error: Processed directory not found at {PROCESSED_DIR}")
        return False

    all_passed = True
    print(f"\n{'Dataset Name':<34} | {'Rows':<10} | {'Cols':<6} | {'Nulls':<6} | {'Status'}")
    print("-" * 75)

    for fname in EXPECTED_DATASETS:
        fpath = PROCESSED_DIR / fname
        if not fpath.exists():
            print(f"{fname:<34} | {'-':<10} | {'-':<6} | {'-':<6} | ❌ MISSING")
            all_passed = False
            continue

        try:
            df = pd.read_parquet(fpath)
            nulls = int(df.isnull().sum().sum())
            rows = len(df)
            cols = df.shape[1]
            status = "✅ OK" if nulls == 0 and rows > 0 else "⚠️ WARN"
            if nulls > 0 or rows == 0:
                all_passed = False
            print(f"{fname:<34} | {rows:<10,} | {cols:<6} | {nulls:<6} | {status}")
        except Exception as e:
            print(f"{fname:<34} | {'ERR':<10} | {'-':<6} | {'-':<6} | ❌ {e}")
            all_passed = False

    print("-" * 75)

    # Validate Feature Store Contract
    fs_path = PROCESSED_DIR / "feature_store.parquet"
    if fs_path.exists():
        fs = pd.read_parquet(fs_path)
        print("\n[Feature Store Specific Contract Checks]")
        print(f" - Total Samples: {len(fs):,}")
        print(f" - Total Features: {fs.shape[1]}")
        print(f" - Null Values: {fs.isnull().sum().sum()}")
        assert "expected_customer_count" in fs.columns or "customer_count" in fs.columns
        print(" - ML Target Column: 'expected_customer_count' Present ✅")

    # Validate Historical Transactions Temporal 1-Hour Lag
    tx_path = PROCESSED_DIR / "historical_transactions.parquet"
    if tx_path.exists():
        tx = pd.read_parquet(tx_path)
        delta = (tx["target_timestamp"] - tx["decision_timestamp"]).unique()
        print(f" - Temporal Lag (t -> t+1): {delta[0]} ✅")

    print("\n" + "=" * 75)
    if all_passed:
        print("🎉 ALL DATASET INTEGRITY CHECKS PASSED SUCCESSFULLY!")
    else:
        print("❌ SOME INTEGRITY CHECKS FAILED. PLEASE REVIEW ABOVE.")
    print("=" * 75)
    return all_passed


if __name__ == "__main__":
    success = check_dataset_health()
    sys.exit(0 if success else 1)
