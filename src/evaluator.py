"""
Comprehensive Meteorological Forecast Evaluator for Hybrid AI-NWP System (PS 26081).
Generates multi-dimensional benchmark scorecards comparing individual constituent models
against baseline and adaptive ML blending across stations, variables, and lead times.
"""
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import logging

from config import VARIABLES, MODELS, PARQUET_DIR, DB_PATH
from storage import WeatherStorage
from metrics import VerificationMetrics
from extreme_detector import THRESHOLDS

logger = logging.getLogger(__name__)


class ForecastEvaluator:
    """Evaluates multi-model forecasting performance and generates verification scorecards."""

    def __init__(self, storage: Optional[WeatherStorage] = None, parquet_dir: Optional[Path] = None):
        self.parquet_dir = parquet_dir or PARQUET_DIR
        self.storage = storage or WeatherStorage(parquet_dir=self.parquet_dir)
        self.metrics = VerificationMetrics()
        self.constituent_models = [m["label"].lower().replace(" ", "_") for m in MODELS.values()]
        self.blending_systems = [
            "blend_equal_weight",
            "blend_inverse_error",
            "blend_hybrid_ml",
            "final_blended_forecast"
        ]

    def build_paired_dataset(
        self,
        df_forecasts: Optional[pd.DataFrame] = None,
        df_observations: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Merges forecasted predictions (including individual models and blended outputs)
        with observed ground-truth records on [timestamp, location_id, variable].
        """
        if df_forecasts is None:
            df_forecasts = pd.read_parquet(self.parquet_dir / "blended_forecasts.parquet")
        if df_observations is None:
            df_observations = pd.read_parquet(self.parquet_dir / "historical_observations.parquet")

        merged = pd.merge(
            df_forecasts,
            df_observations[["timestamp", "location_id", "variable", "observed_value"]],
            on=["timestamp", "location_id", "variable"],
            how="inner"
        )
        logger.info(f"Built paired evaluation dataset: {len(merged):,} samples")
        return merged

    def evaluate_overall_by_variable(self, df_paired: pd.DataFrame) -> pd.DataFrame:
        """
        Generates a summary verification table for each weather variable.
        Compares all models, baseline blends, and the hybrid ML blend against observed truth.
        """
        all_models_to_test = self.constituent_models + self.blending_systems
        records = []

        for var in VARIABLES:
            var_data = df_paired[df_paired["variable"] == var]
            if var_data.empty:
                continue

            y_true = var_data["observed_value"].values

            # First compute best standalone model RMSE as benchmark
            standalone_rmses = {}
            for m in self.constituent_models:
                if m in var_data.columns and var_data[m].notnull().sum() > 5:
                    res = self.metrics.compute_continuous_metrics(y_true, var_data[m].values)
                    standalone_rmses[m] = res["rmse"]

            best_standalone_model = min(standalone_rmses, key=standalone_rmses.get) if standalone_rmses else None
            best_standalone_rmse = standalone_rmses.get(best_standalone_model, np.nan)

            for model_name in all_models_to_test:
                if model_name not in var_data.columns:
                    continue

                y_pred = var_data[model_name].values
                valid_count = (~np.isnan(y_pred) & ~np.isnan(y_true)).sum()
                if valid_count < 5:
                    continue

                res = self.metrics.compute_continuous_metrics(y_true, y_pred)
                
                # Compute Skill Score vs best standalone model
                skill_gain = self.metrics.compute_skill_score(res["rmse"], best_standalone_rmse)

                is_blend = model_name in self.blending_systems
                model_type = "MEGH Operational Blend" if model_name == "final_blended_forecast" else ("Hybrid AI-NWP" if model_name == "blend_hybrid_ml" else ("Baseline Blend" if is_blend else "Standalone NWP/AI"))

                records.append({
                    "variable": var,
                    "model": model_name,
                    "model_type": model_type,
                    "samples": res["n_samples"],
                    "rmse": res["rmse"],
                    "mae": res["mae"],
                    "bias": res["bias"],
                    "correlation": res["correlation"],
                    "skill_score_vs_best_nwp": skill_gain,
                    "best_standalone_benchmark": best_standalone_model
                })

        df_summary = pd.DataFrame(records)
        return df_summary

    def evaluate_by_station(self, df_paired: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluates performance on a per-station basis to demonstrate regional adaptability.
        """
        records = []
        for (loc_id, var), group in df_paired.groupby(["location_id", "variable"]):
            y_true = group["observed_value"].values
            loc_name = group["location_name"].iloc[0]

            # Standalone models
            for m in self.constituent_models + ["blend_hybrid_ml"]:
                if m in group.columns and group[m].notnull().sum() > 5:
                    y_pred = group[m].values
                    res = self.metrics.compute_continuous_metrics(y_true, y_pred)
                    records.append({
                        "location_id": loc_id,
                        "location_name": loc_name,
                        "variable": var,
                        "model": m,
                        "rmse": res["rmse"],
                        "mae": res["mae"],
                        "correlation": res["correlation"]
                    })
        return pd.DataFrame(records)

    def evaluate_by_lead_time(self, df_paired: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluates forecast error degradation across lead time horizons:
        [0-6h, 6-12h, 12-18h, 18-24h+]
        """
        records = []
        lead_time_bins = [
            (0, 6, "0h-6h (Nowcast)"),
            (6, 12, "6h-12h (Short)"),
            (12, 18, "12h-18h (Day 1)"),
            (18, 25, "18h-24h (Day 1 End)")
        ]

        models_to_track = ["gfs", "ecmwf_ifs", "icon", "blend_hybrid_ml"]

        # Calculate relative horizon from beginning of evaluation period
        min_ts = df_paired["timestamp"].min()
        df_eval = df_paired.copy()
        df_eval["eval_horizon_hours"] = ((df_eval["timestamp"] - min_ts).dt.total_seconds() / 3600).astype(int)

        for var, var_group in df_eval.groupby("variable"):
            for low, high, bin_label in lead_time_bins:
                sub = var_group[(var_group["eval_horizon_hours"] >= low) & (var_group["eval_horizon_hours"] < high)]
                if len(sub) < 5:
                    continue

                y_true = sub["observed_value"].values

                for m in models_to_track:
                    if m in sub.columns and sub[m].notnull().sum() > 5:
                        res = self.metrics.compute_continuous_metrics(y_true, sub[m].values)
                        records.append({
                            "variable": var,
                            "lead_time_bin": bin_label,
                            "bin_low_hours": low,
                            "bin_high_hours": high,
                            "model": m,
                            "rmse": res["rmse"],
                            "mae": res["mae"]
                        })
        return pd.DataFrame(records)

    def evaluate_extreme_contingency(self, df_paired: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluates categorical skill for extreme threshold exceedances (POD, FAR, CSI).
        """
        records = []
        test_cases = [
            ("precipitation", THRESHOLDS["precipitation"]["watch"], "Rain > 15mm (Convective Watch)"),
            ("precipitation", THRESHOLDS["precipitation"]["heavy"], "Rain > 64.5mm (IMD Heavy Rain)"),
            ("temperature_2m", THRESHOLDS["temperature_2m"]["hills_heatwave"], "Temp >= 30°C (Hills Heatwave)"),
            ("temperature_2m", THRESHOLDS["temperature_2m"]["coastal_heatwave"], "Temp >= 37°C (Coastal Heatwave)"),
            ("temperature_2m", THRESHOLDS["temperature_2m"]["plains_heatwave"], "Temp >= 40°C (IMD Plains Heatwave)"),
            ("wind_speed_10m", THRESHOLDS["wind_speed_10m"]["squall"], "Wind >= 40 km/h (IMD Squall Warning)")
        ]

        models_to_test = ["gfs", "ecmwf_ifs", "blend_hybrid_ml"]

        for var, thresh, label in test_cases:
            sub = df_paired[df_paired["variable"] == var]
            if sub.empty:
                continue

            y_true = sub["observed_value"].values

            for m in models_to_test:
                if m in sub.columns and sub[m].notnull().sum() > 5:
                    y_pred = sub[m].values
                    res = self.metrics.compute_contingency_table(y_true, y_pred, threshold=thresh)
                    records.append({
                        "variable": var,
                        "hazard_label": label,
                        "model": m,
                        "hits": res["hits"],
                        "false_alarms": res["false_alarms"],
                        "misses": res["misses"],
                        "pod_hit_rate": res["pod"],
                        "far_false_alarm_ratio": res["far"],
                        "csi_threat_score": res["csi"]
                    })
        return pd.DataFrame(records)

    def run_full_evaluation(self) -> Dict[str, pd.DataFrame]:
        """
        Executes end-to-end evaluation and persists output scorecards to Parquet and SQLite.
        """
        logger.info("--- Executing Phase 3 Forecast Evaluation & Benchmarking ---")
        df_paired = self.build_paired_dataset()

        summary_df = self.evaluate_overall_by_variable(df_paired)
        station_df = self.evaluate_by_station(df_paired)
        lead_time_df = self.evaluate_by_lead_time(df_paired)
        contingency_df = self.evaluate_extreme_contingency(df_paired)

        # 1. Save to Parquet
        summary_path = self.parquet_dir / "evaluation_scorecard_summary.parquet"
        station_path = self.parquet_dir / "evaluation_by_station.parquet"
        lead_path = self.parquet_dir / "evaluation_by_lead_time.parquet"
        contingency_path = self.parquet_dir / "evaluation_contingency_extremes.parquet"

        summary_df.to_parquet(summary_path, engine="pyarrow", index=False)
        station_df.to_parquet(station_path, engine="pyarrow", index=False)
        lead_time_df.to_parquet(lead_path, engine="pyarrow", index=False)
        contingency_df.to_parquet(contingency_path, engine="pyarrow", index=False)

        logger.info(f"Saved evaluation summary scorecard -> {summary_path}")
        logger.info(f"Saved station-wise evaluation -> {station_path}")
        logger.info(f"Saved lead-time degradation evaluation -> {lead_path}")
        logger.info(f"Saved contingency extremes evaluation -> {contingency_path}")

        # 2. Save summary to SQLite for dashboard querying and maintain historical ledger
        with self.storage.get_connection() as conn:
            summary_df.to_sql("model_verification_scorecard", conn, if_exists="replace", index=False)
            history_df = summary_df.copy()
            history_df["evaluated_at"] = pd.Timestamp.now()
            history_df.to_sql("model_verification_history", conn, if_exists="append", index=False)

        return {
            "summary": summary_df,
            "station": station_df,
            "lead_time": lead_time_df,
            "contingency": contingency_df
        }


def print_executive_report(eval_results: dict):
    """
    Outputs formatted scorecards, skill gain calculations, and categorical contingency tables.
    """
    summary_df = eval_results["summary"]
    contingency_df = eval_results["contingency"]

    print("\n" + "=" * 80)
    print("      METEOROLOGICAL FORECAST VERIFICATION & BENCHMARK REPORT      ")
    print("      Hybrid AI-NWP Blending System vs Standalone Models (PS 26081) ")
    print("=" * 80)

    # 1. Summary by Variable
    for var in summary_df["variable"].unique():
        var_sub = summary_df[summary_df["variable"] == var].sort_values("rmse")
        best_nwp = var_sub["best_standalone_benchmark"].iloc[0]
        
        print(f"\n--- VARIABLE: {var.upper()} ---")
        print(f"{'Model':22s} | {'Type':16s} | {'RMSE':7s} | {'MAE':7s} | {'Bias':7s} | {'Corr (R)':8s} | {'Skill Gain vs Best':18s}")
        print("-" * 95)

        for _, row in var_sub.iterrows():
            m_name = row["model"]
            m_type = row["model_type"]
            rmse_str = f"{row['rmse']:.3f}"
            mae_str = f"{row['mae']:.3f}"
            bias_str = f"{row['bias']:+.3f}"
            corr_str = f"{row['correlation']:.3f}"
            
            ss = row["skill_score_vs_best_nwp"]
            if row["model"] == "blend_hybrid_ml":
                ss_str = f"[*] {ss:+.1f}% (HYBRID)"
            elif ss > 0:
                ss_str = f"+{ss:.1f}%"
            elif ss == 0 and row["model"] == best_nwp:
                ss_str = "0.0% (Best Standalone)"
            else:
                ss_str = f"{ss:.1f}%"

            marker = ">> " if "blend" in m_name else "   "
            print(f"{marker}{m_name:19s} | {m_type:16s} | {rmse_str:7s} | {mae_str:7s} | {bias_str:7s} | {corr_str:8s} | {ss_str:18s}")

        # Highlight Skill Gain
        hybrid_row = var_sub[var_sub["model"] == "blend_hybrid_ml"]
        if not hybrid_row.empty:
            h_rmse = hybrid_row["rmse"].iloc[0]
            best_nwp_row = var_sub[var_sub["model"] == best_nwp]
            if not best_nwp_row.empty:
                b_rmse = best_nwp_row["rmse"].iloc[0]
                reduction = ((b_rmse - h_rmse) / b_rmse) * 100.0
                print(f"\n  [KEY METRIC] Hybrid AI-NWP Blend achieved {reduction:.1f}% RMSE reduction over best standalone model ({best_nwp})")

    # 2. Extreme Weather Contingency Table
    if not contingency_df.empty:
        print("\n" + "=" * 80)
        print("                 EXTREME WEATHER DETECTION SKILL (CONTINGENCY)               ")
        print("=" * 80)
        print(f"{'Hazard Event':32s} | {'Model':18s} | {'Hits':5s} | {'False Alarms':12s} | {'Hit Rate (POD)':14s} | {'False Alarm (FAR)':17s} | {'Threat (CSI)':12s}")
        print("-" * 115)

        for _, row in contingency_df.iterrows():
            pod_str = f"{row['pod_hit_rate']*100:.1f}%"
            far_str = f"{row['far_false_alarm_ratio']*100:.1f}%"
            csi_str = f"{row['csi_threat_score']*100:.1f}%"
            print(f"{row['hazard_label']:32s} | {row['model']:18s} | {row['hits']:5d} | {row['false_alarms']:12d} | {pod_str:14s} | {far_str:17s} | {csi_str:12s}")

    print("\n" + "=" * 80)
    print("                     END OF EVALUATION REPORT                       ")
    print("=" * 80 + "\n")


ForecastEvaluator.print_executive_report = staticmethod(print_executive_report)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Forecast Evaluation Runner (PS 26081)")
    parser.add_argument("--run", action="store_true", help="Execute complete forecast verification and generate report")
    args = parser.parse_args()

    evaluator = ForecastEvaluator()
    results = evaluator.run_full_evaluation()
    print_executive_report(results)


if __name__ == "__main__":
    main()

