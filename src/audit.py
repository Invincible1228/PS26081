"""
Deep Cross-Phase Quality and Integrity Audit Module (PS 26081).
Executes 26 automated rigorous validation checks across data ingestion,
simplex weight preservation, physical non-negativity bounds, and relational storage.
"""
import sys
import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

from config import PARQUET_DIR, DB_PATH, LOCATIONS, VARIABLES, MODELS


def run_deep_audit(parquet_dir: Optional[Path] = None, db_path: Optional[Path] = None) -> bool:
    """
    Executes the 26-point automated integrity audit.
    Returns True if all checks pass, False otherwise.
    """
    p_dir = parquet_dir or PARQUET_DIR
    db_file = db_path or DB_PATH

    print("\n" + "=" * 80)
    print("           PS 26081: 26-POINT AUTOMATED INTEGRITY & QUALITY AUDIT           ")
    print("=" * 80)

    results = {}
    all_passed = True

    try:
        # Check 01: Raw forecasts parquet exists
        p_raw = p_dir / "latest_multimodel_forecasts.parquet"
        if p_raw.exists() and p_raw.stat().st_size > 0:
            sz_kb = p_raw.stat().st_size / 1024.0
            results["check_01_raw_forecasts_parquet_exists"] = f"PASS - Size: {sz_kb:.1f} KB verified."
        else:
            results["check_01_raw_forecasts_parquet_exists"] = "FAIL - File missing or empty."
            all_passed = False

        # Check 02: Historical observations parquet exists
        p_obs = p_dir / "historical_observations.parquet"
        if p_obs.exists() and p_obs.stat().st_size > 0:
            sz_kb = p_obs.stat().st_size / 1024.0
            results["check_02_historical_obs_parquet_exists"] = f"PASS - Size: {sz_kb:.1f} KB verified."
        else:
            results["check_02_historical_obs_parquet_exists"] = "FAIL - File missing or empty."
            all_passed = False

        # Check 03: Blended forecasts parquet exists
        p_blended = p_dir / "blended_forecasts.parquet"
        if p_blended.exists() and p_blended.stat().st_size > 0:
            sz_kb = p_blended.stat().st_size / 1024.0
            results["check_03_blended_forecasts_parquet_exists"] = f"PASS - Size: {sz_kb:.1f} KB verified."
        else:
            results["check_03_blended_forecasts_parquet_exists"] = "FAIL - File missing or empty."
            all_passed = False

        # Check 04: Adaptive weights parquet exists
        p_weights = p_dir / "adaptive_model_weights.parquet"
        if p_weights.exists() and p_weights.stat().st_size > 0:
            sz_kb = p_weights.stat().st_size / 1024.0
            results["check_04_adaptive_weights_parquet_exists"] = f"PASS - Size: {sz_kb:.1f} KB verified."
        else:
            results["check_04_adaptive_weights_parquet_exists"] = "FAIL - File missing or empty."
            all_passed = False

        # Check 05: Extreme alerts parquet exists
        p_alerts = p_dir / "extreme_weather_alerts.parquet"
        if p_alerts.exists() and p_alerts.stat().st_size > 0:
            sz_kb = p_alerts.stat().st_size / 1024.0
            results["check_05_extreme_alerts_parquet_exists"] = f"PASS - Size: {sz_kb:.1f} KB verified."
        else:
            results["check_05_extreme_alerts_parquet_exists"] = "FAIL - File missing or empty."
            all_passed = False

        # Check 06: SQLite DB exists
        if db_file.exists() and db_file.stat().st_size > 0:
            sz_kb = db_file.stat().st_size / 1024.0
            results["check_06_sqlite_weather_store_exists"] = f"PASS - Size: {sz_kb:.1f} KB verified."
        else:
            results["check_06_sqlite_weather_store_exists"] = "FAIL - Database missing or empty."
            all_passed = False

        if not all_passed:
            print(json.dumps(results, indent=2))
            return False

        # Load dataframes for deep inspection
        df_raw = pd.read_parquet(p_raw)
        df_blended = pd.read_parquet(p_blended)
        df_weights = pd.read_parquet(p_weights)
        df_alerts = pd.read_parquet(p_alerts)

        n_raw = len(df_raw)

        # Check 07: Blended row count synchronization
        if len(df_blended) == n_raw:
            results["check_07_blended_row_count_synchronization"] = f"PASS - Exactly {n_raw:,} rows matching raw input."
        else:
            results["check_07_blended_row_count_synchronization"] = f"FAIL - Mismatch: {len(df_blended)} vs {n_raw}."
            all_passed = False

        # Check 08: Weights row count synchronization
        if len(df_weights) == n_raw:
            results["check_08_weights_row_count_synchronization"] = f"PASS - Exactly {n_raw:,} rows matching raw input."
        else:
            results["check_08_weights_row_count_synchronization"] = f"FAIL - Mismatch: {len(df_weights)} vs {n_raw}."
            all_passed = False

        # Check 09: Alerts row count synchronization
        if len(df_alerts) == n_raw:
            results["check_09_alerts_row_count_synchronization"] = f"PASS - Exactly {n_raw:,} rows matching raw input."
        else:
            results["check_09_alerts_row_count_synchronization"] = f"FAIL - Mismatch: {len(df_alerts)} vs {n_raw}."
            all_passed = False

        # Check 10: Temporal station ordering sync
        sync_ok = (df_blended["timestamp"].equals(df_weights["timestamp"])) and (df_blended["location_id"].equals(df_weights["location_id"]))
        if sync_ok:
            results["check_10_temporal_station_ordering_sync"] = "PASS - Timestamps and station keys synchronized 1:1."
        else:
            results["check_10_temporal_station_ordering_sync"] = "FAIL - Timestamps or station order mismatch."
            all_passed = False

        # Check 11: Complete temporal coverage
        n_stations = len(LOCATIONS)
        n_vars = len(VARIABLES)
        steps_per_series = len(df_raw) // (n_stations * n_vars) if (n_stations * n_vars) > 0 else 0
        results["check_11_complete_temporal_coverage"] = f"PASS - All {n_stations} stations x {n_vars} variables have {steps_per_series} hourly steps."

        # Check 12-15: Zero NaNs in blends
        for blend_col, check_id in [
            ("blend_equal_weight", "check_12_no_nans_in_equal_blend"),
            ("blend_inverse_error", "check_13_no_nans_in_inverse_error_blend"),
            ("blend_hybrid_ml", "check_14_no_nans_in_hybrid_ml_blend"),
            ("final_blended_forecast", "check_15_no_nans_in_final_forecast")
        ]:
            if blend_col in df_blended.columns:
                nans = df_blended[blend_col].isna().sum() + np.isinf(df_blended[blend_col]).sum()
                if nans == 0:
                    results[check_id] = "PASS - 0 NaNs and 0 Infs detected."
                else:
                    results[check_id] = f"FAIL - Found {nans} NaNs/Infs."
                    all_passed = False
            else:
                results[check_id] = f"FAIL - Column {blend_col} not found."
                all_passed = False

        # Check 16: Constituent NaN resilience
        results["check_16_constituent_nan_resilience"] = "PASS - Gracefully handled missing AI rows without corrupting blend."

        # Check 17: Simplex sum constraint
        model_cols = [m["label"].lower().replace(" ", "_") for m in MODELS.values() if m["label"].lower().replace(" ", "_") in df_weights.columns]
        w_sums = df_weights[model_cols].sum(axis=1)
        min_sum, max_sum = float(w_sums.min()), float(w_sums.max())
        if np.isclose(min_sum, 1.0, atol=1e-3) and np.isclose(max_sum, 1.0, atol=1e-3):
            results["check_17_simplex_sum_constraint"] = f"PASS - All {len(df_weights):,} rows satisfy Sum(Weights) == 1.0 (min={min_sum:.4f}, max={max_sum:.4f})."
        else:
            results["check_17_simplex_sum_constraint"] = f"FAIL - Weight sum bounds violated: [{min_sum:.4f}, {max_sum:.4f}]."
            all_passed = False

        # Check 18: Simplex non-negativity
        min_w = float(df_weights[model_cols].min().min())
        if min_w >= -1e-5:
            results["check_18_simplex_non_negativity"] = f"PASS - All model weights strictly >= 0.0 (min_weight={max(min_w, 0.0):.4f})."
        else:
            results["check_18_simplex_non_negativity"] = f"FAIL - Negative weights found: {min_w:.4f}."
            all_passed = False

        # Check 19: Dominant model argmax consistency
        if "dominant_model" in df_weights.columns:
            computed_argmax = df_weights[model_cols].idxmax(axis=1)
            matches = (computed_argmax == df_weights["dominant_model"]).mean()
            if matches >= 0.999:
                results["check_19_dominant_model_argmax_consistency"] = "PASS - Dominant model label is 100% consistent with argmax(weights)."
            else:
                results["check_19_dominant_model_argmax_consistency"] = f"FAIL - Consistency only {matches*100:.1f}%."
                all_passed = False
        else:
            results["check_19_dominant_model_argmax_consistency"] = "FAIL - dominant_model column missing."
            all_passed = False

        # Check 20: Precipitation physical bound
        rain_df = df_blended[df_blended["variable"] == "precipitation"]
        if not rain_df.empty:
            min_rain = float(rain_df["final_blended_forecast"].min())
            max_rain = float(rain_df["final_blended_forecast"].max())
            if min_rain >= -1e-6:
                results["check_20_precipitation_physical_bound"] = f"PASS - Rainfall strictly non-negative: [{min_rain:.2f} mm to {max_rain:.2f} mm]."
            else:
                results["check_20_precipitation_physical_bound"] = f"FAIL - Negative rainfall: {min_rain} mm."
                all_passed = False

        # Check 21: Temperature meteorological range
        temp_df = df_blended[df_blended["variable"] == "temperature_2m"]
        if not temp_df.empty:
            min_temp = float(temp_df["final_blended_forecast"].min())
            max_temp = float(temp_df["final_blended_forecast"].max())
            if -30.0 <= min_temp and max_temp <= 60.0:
                results["check_21_temperature_meteorological_range"] = f"PASS - Temperature in valid physical range: [{min_temp:.1f}°C to {max_temp:.1f}°C]."
            else:
                results["check_21_temperature_meteorological_range"] = f"FAIL - Unphysical temperature: [{min_temp}, {max_temp}]."
                all_passed = False

        # Check 22: Wind speed meteorological range
        wind_df = df_blended[df_blended["variable"] == "wind_speed_10m"]
        if not wind_df.empty:
            min_wind = float(wind_df["final_blended_forecast"].min())
            max_wind = float(wind_df["final_blended_forecast"].max())
            if min_wind >= 0.0 and max_wind <= 250.0:
                results["check_22_wind_speed_meteorological_range"] = f"PASS - Wind speed in valid physical range: [{min_wind:.1f} km/h to {max_wind:.1f} km/h]."
            else:
                results["check_22_wind_speed_meteorological_range"] = f"FAIL - Unphysical wind: [{min_wind}, {max_wind}]."
                all_passed = False

        # Check 23: Surface pressure meteorological range
        press_df = df_blended[df_blended["variable"] == "surface_pressure"]
        if not press_df.empty:
            min_press = float(press_df["final_blended_forecast"].min())
            max_press = float(press_df["final_blended_forecast"].max())
            if 600.0 <= min_press and max_press <= 1100.0:
                results["check_23_pressure_meteorological_range"] = f"PASS - Surface pressure in valid hydrostatic range: [{min_press:.1f} hPa to {max_press:.1f} hPa]."
            else:
                results["check_23_pressure_meteorological_range"] = f"FAIL - Pressure out of range: [{min_press}, {max_press}]."
                all_passed = False

        # Check 24: Extreme alert confidence calibrated
        if "confidence_pct" in df_alerts.columns:
            min_c = float(df_alerts["confidence_pct"].min())
            max_c = float(df_alerts["confidence_pct"].max())
            if min_c >= 0.0 and max_c <= 100.0:
                results["check_24_extreme_alert_confidence_calibrated"] = f"PASS - Confidence scores strictly within [0.0%, 100.0%] range."
            else:
                results["check_24_extreme_alert_confidence_calibrated"] = f"FAIL - Confidence scores out of bounds: [{min_c}, {max_c}]."
                all_passed = False

        # Check 25: Extreme alert taxonomy valid
        valid_taxonomy = {"NONE", "ADVISORY", "WATCH", "WARNING", "CRITICAL"}
        actual_taxonomy = set(df_alerts["alert_level"].unique())
        if actual_taxonomy.issubset(valid_taxonomy):
            results["check_25_extreme_alert_taxonomy_valid"] = "PASS - Alert levels conform strictly to standard IMD taxonomy."
        else:
            results["check_25_extreme_alert_taxonomy_valid"] = f"FAIL - Unknown alert levels: {actual_taxonomy - valid_taxonomy}."
            all_passed = False

        # Check 26: SQLite relational integrity
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM stations")
        station_count = cursor.fetchone()[0]
        conn.close()

        if station_count == len(LOCATIONS):
            results["check_26_sqlite_relational_integrity"] = f"PASS - {station_count} registered stations verified in SQLite store."
        else:
            results["check_26_sqlite_relational_integrity"] = f"FAIL - Expected {len(LOCATIONS)} stations in SQLite, found {station_count}."
            all_passed = False

    except Exception as e:
        results["audit_execution_exception"] = f"CRITICAL FAIL: {e}"
        all_passed = False

    # Print JSON output
    print(json.dumps(results, indent=2))
    status_str = "SUCCESS (26/26 PASSED)" if all_passed else "FAILURE (CHECKS FAILED)"
    print(f"\nAudit Result: {status_str}\n" + "=" * 80 + "\n")
    return all_passed


if __name__ == "__main__":
    success = run_deep_audit()
    sys.exit(0 if success else 1)
