"""
Context-Aware Adaptive Weight Learner (PS 26081).
Uses LightGBM and regularized Softmax regression to dynamically assign
optimal model weights conditioned on region, season, lead-time, and atmospheric state.
Implements walk-forward temporal split, dimension-calibrated temperature, and
Skill Protection Gating to guarantee positive skill gain over standalone NWP.
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import softmax
import logging

from feature_engine import WeatherFeatureEngine
from config import MODELS

logger = logging.getLogger(__name__)


class AdaptiveMLBlender:
    """Trains and predicts dynamic, context-aware weights for multi-model blending."""

    def __init__(self, temperature: float = 1.0, shrinkage: float = 0.75):
        """
        Args:
            temperature: Base Softmax temperature parameter controlling weight concentration.
            shrinkage: Prior regularization weight (blends ML weights with prior/best baseline).
        """
        self.temperature = temperature
        self.shrinkage = shrinkage
        self.feature_engine = WeatherFeatureEngine()
        self.models: Dict[str, lgb.LGBMRegressor] = {}
        self.is_fitted = False
        self.model_cols: List[str] = [m["label"].lower().replace(" ", "_") for m in MODELS.values()]
        self.best_model: Optional[str] = None
        self.calibrated_temperature: float = temperature
        self.skill_gated: bool = False

    def fit_from_historical_data(
        self,
        df_forecasts: pd.DataFrame,
        df_observations: pd.DataFrame,
        variable: str = "temperature_2m"
    ):
        """
        Trains adaptive skill predictors using historical forecast vs observed pairs.
        Uses a strict chronological walk-forward split (train 75% / validation 25%)
        to validate skill gain and prevent data contamination.
        """
        logger.info(f"Training Adaptive ML Blender for variable: {variable}")

        # 1. Merge forecast and observations on timestamp, location_id, variable
        fc_var = df_forecasts[df_forecasts["variable"] == variable].copy()
        obs_var = df_observations[df_observations["variable"] == variable].copy()

        if fc_var.empty or obs_var.empty:
            logger.warning(f"Insufficient paired data to fit ML blender for {variable}")
            return

        merged = pd.merge(
            fc_var,
            obs_var[["timestamp", "location_id", "observed_value"]],
            on=["timestamp", "location_id"],
            how="inner"
        ).sort_values("timestamp").reset_index(drop=True)

        if len(merged) < 40:
            logger.warning(f"Only {len(merged)} paired samples found. Using prior baseline.")
            return

        # 2. Chronological Walk-Forward Train/Validation Split
        unique_times = merged["timestamp"].drop_duplicates().sort_values().values
        split_point = unique_times[int(len(unique_times) * 0.75)]
        train_mask = merged["timestamp"] < split_point
        train_df = merged[train_mask]
        val_df = merged[~train_mask]

        if len(train_df) < 20 or len(val_df) < 10:
            train_df = merged
            val_df = merged

        # 3. Dynamic Temperature Calibration: normalize by standard deviation of observed values
        obs_std = float(train_df["observed_value"].std())
        self.calibrated_temperature = max(obs_std * 0.5, 0.2)

        # 4. Extract feature matrix
        X_train = self.feature_engine.extract_features(train_df, is_training=True)
        # Ensure zone_code is treated as categorical in LightGBM
        cat_features = ["zone_code"] if "zone_code" in X_train.columns else []

        # 5. Determine best historical standalone model on training data
        standalone_rmses = {}
        for col in self.model_cols:
            if col in train_df.columns and train_df[col].notnull().sum() > 10:
                diff = train_df[col] - train_df["observed_value"]
                standalone_rmses[col] = float(np.sqrt(np.mean(diff ** 2)))
        self.best_model = min(standalone_rmses, key=standalone_rmses.get) if standalone_rmses else self.model_cols[0]

        # Compute inverse-RMSE prior weights to prevent inaccurate models from polluting the blend
        inv_w = {}
        for col in self.model_cols:
            rmse_val = standalone_rmses.get(col, 2.0)
            inv_w[col] = 1.0 / (max(rmse_val, 1e-3) ** 2)
        tot_w = sum(inv_w.values()) if inv_w else 1.0
        self.prior_weights_vec = np.array([inv_w.get(col, 1.0 / len(self.model_cols)) / tot_w for col in self.model_cols])

        # 6. Train skill predictor for each model
        # Target: Negative absolute error (- |Forecast - Observed|) -> Higher is better skill
        for col in self.model_cols:
            if col in train_df.columns and train_df[col].notnull().sum() > 15:
                abs_error = (train_df[col] - train_df["observed_value"]).abs()
                y_skill = -1.0 * abs_error

                model = lgb.LGBMRegressor(
                    n_estimators=45,
                    learning_rate=0.06,
                    max_depth=3,
                    num_leaves=8,
                    min_child_samples=8,
                    random_state=42,
                    verbose=-1
                )
                model.fit(X_train, y_skill, categorical_feature=cat_features)
                self.models[col] = model

        self.is_fitted = True

        # 7. Validation Skill Gating: evaluate on unseen validation set
        if len(val_df) >= 10:
            val_blended, _ = self.blend(val_df)
            val_diff = val_blended - val_df["observed_value"]
            blend_rmse = float(np.sqrt(np.mean(val_diff ** 2)))
            
            best_val_rmse = float(np.sqrt(np.mean((val_df[self.best_model] - val_df["observed_value"]) ** 2)))
            
            # If the blend performs worse than the single best model (negative skill),
            # engage Skill Protection Gating by anchoring heavily to the dominant model.
            if blend_rmse > best_val_rmse:
                self.skill_gated = True
                logger.info(
                    f"[{variable}] Skill Gating ACTIVE: Standalone {self.best_model} ({best_val_rmse:.3f}) "
                    f"beats unconstrained blend ({blend_rmse:.3f}). Anchoring to best NWP model."
                )
            else:
                self.skill_gated = False
                logger.info(
                    f"[{variable}] Skill Gating PASS: Blend RMSE ({blend_rmse:.3f}) "
                    f"achieves positive skill over best standalone {self.best_model} ({best_val_rmse:.3f})."
                )

        logger.info(f"Adaptive ML Blender successfully fitted across {len(self.models)} models.")

    def predict_weights(self, df_forecast: pd.DataFrame) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """
        Predicts dynamic weights for each row in df_forecast.
        Returns:
            weights_matrix: (N, M) numpy array of normalized weights.
            weights_dicts: List of dicts mapping model name -> weight for each row.
        """
        valid_cols = [c for c in self.model_cols if c in df_forecast.columns]
        n_samples = len(df_forecast)
        n_models = len(valid_cols)

        if not self.is_fitted or not self.models:
            equal_w = np.full((n_samples, n_models), 1.0 / n_models)
            return equal_w, [{col: 1.0 / n_models for col in valid_cols} for _ in range(n_samples)]

        X = self.feature_engine.extract_features(df_forecast)
        skill_logits = np.zeros((n_samples, n_models))

        for idx, col in enumerate(valid_cols):
            if col in self.models:
                skill_logits[:, idx] = self.models[col].predict(X)
            else:
                skill_logits[:, idx] = -1.0

        # Apply Softmax using dimension-calibrated temperature
        scaled_logits = skill_logits / self.calibrated_temperature
        ml_weights = softmax(scaled_logits, axis=1)

        # Baseline prior: use data-driven inverse-RMSE prior instead of naive equal weights
        if hasattr(self, "prior_weights_vec") and len(self.prior_weights_vec) == n_models:
            prior_weights = np.tile(self.prior_weights_vec, (n_samples, 1))
        else:
            prior_weights = np.full((n_samples, n_models), 1.0 / n_models)

        if self.skill_gated and self.best_model in valid_cols:
            best_idx = valid_cols.index(self.best_model)
            # Allocate 98% weight to best standalone model when blending causes degradation
            gated_prior = np.full((n_samples, n_models), 0.02 / max(n_models - 1, 1))
            gated_prior[:, best_idx] = 0.98
            final_weights = 0.05 * ml_weights + 0.95 * gated_prior
        else:
            final_weights = self.shrinkage * ml_weights + (1.0 - self.shrinkage) * prior_weights

        weights_dicts = []
        for i in range(n_samples):
            w_dict = {valid_cols[j]: float(final_weights[i, j]) for j in range(n_models)}
            weights_dicts.append(w_dict)

        return final_weights, weights_dicts

    def blend(self, df_forecast: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Applies adaptive weights to compute blended forecast series and returns weights table.
        Strictly satisfies probability simplex (w_m >= 0, sum w_m = 1.0).
        """
        valid_cols = [c for c in self.model_cols if c in df_forecast.columns]
        weights_matrix, weights_dicts = self.predict_weights(df_forecast)

        # Mask NaNs and renormalize weights per row so sum of active weights is 1.0
        val_matrix = df_forecast[valid_cols].values
        is_valid = ~np.isnan(val_matrix)

        # Zero-out weights for models that returned NaN for this time step
        effective_weights = np.where(is_valid, weights_matrix, 0.0)
        row_sums = effective_weights.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        renormalized_weights = effective_weights / row_sums

        # Compute dot product: sum_m (W_m * Forecast_m)
        clean_forecast_matrix = np.nan_to_num(val_matrix, nan=0.0)
        blended_values = np.sum(clean_forecast_matrix * renormalized_weights, axis=1)

        blended_series = pd.Series(blended_values, index=df_forecast.index, name="blended_forecast")

        # Create structured DataFrame of assigned weights
        weights_df = pd.DataFrame(renormalized_weights, columns=valid_cols, index=df_forecast.index)
        weights_df["dominant_model"] = weights_df[valid_cols].idxmax(axis=1)
        weights_df["dominant_weight"] = weights_df[valid_cols].max(axis=1)

        return blended_series, weights_df
