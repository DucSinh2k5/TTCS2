from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


CURRENT_YEAR = datetime.now().year
FEATURE_COLUMNS = [
    "brand",
    "car_age_years",
    "engine_power_hp",
    "fuel_consumption_l_per_100km",
    "engine_displacement_cc",
]
TARGET_COLUMN = "price_lakh"
DEFAULT_DATA_PATH = Path(__file__).parent.parent / "Datasets" / "train-data.csv"
DEFAULT_BUNDLE_DIR = Path(__file__).parent.parent / "Models"


@dataclass
class ModelBundle:
    model_name: str
    pipeline: Pipeline
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    numeric_ranges: dict[str, tuple[float, float]]
    brands: list[str]
    reference_df: pd.DataFrame


def _parse_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.extract(r"(\d+(?:\.\d+)?)")[0]
    return pd.to_numeric(cleaned, errors="coerce")


def _extract_brand(name: Any) -> str:
    if pd.isna(name):
        return "Other"
    text = str(name).strip()
    if not text:
        return "Other"
    if text.lower().startswith("land rover"):
        return "Land Rover"
    return text.split()[0]


def prepare_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Name": "car_name",
        "Year": "year",
        "Power": "power_raw",
        "Engine": "engine_raw",
        "Mileage": "mileage_raw",
        "Price": TARGET_COLUMN,
        "New_Price": "new_price_raw",
    }
    df = df_raw.rename(columns=rename_map).copy()

    for col in ["car_name", "year", "power_raw", "engine_raw", "mileage_raw", TARGET_COLUMN]:
        if col not in df.columns:
            df[col] = np.nan

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["engine_power_hp"] = _parse_numeric_series(df["power_raw"])
    df["engine_displacement_cc"] = _parse_numeric_series(df["engine_raw"])
    km_per_l = _parse_numeric_series(df["mileage_raw"])
    df["fuel_consumption_l_per_100km"] = np.where(km_per_l > 0, 100.0 / km_per_l, np.nan)
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    df["brand"] = df["car_name"].apply(_extract_brand)
    df["car_age_years"] = CURRENT_YEAR - df["year"]

    if "new_price_raw" in df.columns:
        df["new_price_lakh"] = _parse_numeric_series(df["new_price_raw"])
    else:
        df["new_price_lakh"] = np.nan

    df = df[df["car_age_years"].between(0, 40, inclusive="both")]

    keep_cols = [
        "car_name",
        "brand",
        "car_age_years",
        "engine_power_hp",
        "fuel_consumption_l_per_100km",
        "engine_displacement_cc",
        TARGET_COLUMN,
        "new_price_lakh",
    ]
    return df[keep_cols].reset_index(drop=True)


def _build_estimator(model_name: str):
    key = model_name.lower().strip()
    if key == "randomforest":
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=14,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

    if key == "linearregression":
        return LinearRegression()

    if key == "xgboost":
        if XGBRegressor is None:
            return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        return XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    raise ValueError(f"Unsupported model_name: {model_name}")


def train_bundle(model_name: str, data_path: Path = DEFAULT_DATA_PATH) -> ModelBundle:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df_raw = pd.read_csv(data_path)
    df = prepare_dataframe(df_raw)

    train_df = df.dropna(subset=[TARGET_COLUMN]).copy()

    numeric_features = [
        "car_age_years",
        "engine_power_hp",
        "fuel_consumption_l_per_100km",
        "engine_displacement_cc",
    ]
    categorical_features = ["brand"]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    estimator = _build_estimator(model_name)
    pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", estimator)])
    pipeline.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])

    ranges = {}
    for col in numeric_features:
        ranges[col] = (
            float(train_df[col].min(skipna=True)),
            float(train_df[col].max(skipna=True)),
        )

    brands = sorted(train_df["brand"].dropna().astype(str).unique().tolist())

    reference_cols = [
        "car_name",
        "brand",
        "car_age_years",
        "engine_power_hp",
        "fuel_consumption_l_per_100km",
        "engine_displacement_cc",
        TARGET_COLUMN,
        "new_price_lakh",
    ]

    return ModelBundle(
        model_name=model_name,
        pipeline=pipeline,
        feature_columns=FEATURE_COLUMNS,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        numeric_ranges=ranges,
        brands=brands,
        reference_df=train_df[reference_cols].copy(),
    )


def save_bundle(bundle: ModelBundle, model_dir: Path = DEFAULT_BUNDLE_DIR) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / f"car_price_bundle_{bundle.model_name.lower()}.pkl"
    joblib.dump(bundle, out_path)
    return out_path


def load_or_train_bundle(
    model_name: str,
    data_path: Path = DEFAULT_DATA_PATH,
    model_dir: Path = DEFAULT_BUNDLE_DIR,
    force_retrain: bool = False,
) -> ModelBundle:
    bundle_path = model_dir / f"car_price_bundle_{model_name.lower()}.pkl"
    if not force_retrain and bundle_path.exists():
        return joblib.load(bundle_path)

    bundle = train_bundle(model_name=model_name, data_path=data_path)
    save_bundle(bundle, model_dir=model_dir)
    return bundle


def predict_price(bundle: ModelBundle, user_input: dict[str, Any]) -> float:
    df = pd.DataFrame([user_input], columns=bundle.feature_columns)
    pred = bundle.pipeline.predict(df)[0]
    return float(pred)


def check_ood(bundle: ModelBundle, user_input: dict[str, Any]) -> tuple[bool, list[str]]:
    messages: list[str] = []
    for col in bundle.numeric_features:
        value = float(user_input[col])
        low, high = bundle.numeric_ranges[col]
        if value < low or value > high:
            messages.append(f"{col}: {value:.2f} is outside training range [{low:.2f}, {high:.2f}]")

    return (len(messages) > 0, messages)


def get_market_comparison(bundle: ModelBundle, user_input: dict[str, Any], k: int = 50) -> dict[str, Any]:
    df = bundle.reference_df.copy()
    if df.empty:
        return {"market_avg": np.nan, "sample_size": 0, "comps": pd.DataFrame()}

    same_brand = df[df["brand"] == str(user_input["brand"])].copy()
    comps = same_brand if len(same_brand) >= 10 else df.copy()

    num_cols = bundle.numeric_features
    arr = comps[num_cols].to_numpy(dtype=float)
    center = np.array([float(user_input[c]) for c in num_cols], dtype=float)

    std = np.nanstd(arr, axis=0)
    std[std == 0] = 1.0
    dist = np.linalg.norm((arr - center) / std, axis=1)
    comps = comps.assign(_dist=dist).sort_values("_dist").head(k)

    return {
        "market_avg": float(comps[TARGET_COLUMN].mean()) if not comps.empty else np.nan,
        "sample_size": int(len(comps)),
        "comps": comps.drop(columns=["_dist"], errors="ignore"),
    }


def evaluate_deal(predicted_price: float, market_avg: float) -> dict[str, Any]:
    if not np.isfinite(market_avg) or market_avg <= 0:
        return {
            "market_label": "Unknown",
            "deal_score": 50,
            "deal_label": "Need more data",
            "explanation": "Not enough market comparables to assess pricing.",
        }

    ratio = predicted_price / market_avg
    if ratio <= 0.93:
        market_label = "Cheaper than market"
    elif ratio >= 1.07:
        market_label = "Over market"
    else:
        market_label = "Fair vs market"

    deal_score = max(0, min(100, int(100 - abs(ratio - 1.0) * 220)))

    if ratio <= 0.93 and deal_score >= 70:
        deal_label = "Good deal"
        explanation = "Predicted price is meaningfully lower than similar market cars."
    elif ratio >= 1.07:
        deal_label = "Overpriced"
        explanation = "Predicted price is above the market level of comparable cars."
    else:
        deal_label = "Fair price"
        explanation = "Predicted price is close to the market average for comparable cars."

    return {
        "market_label": market_label,
        "deal_score": deal_score,
        "deal_label": deal_label,
        "explanation": explanation,
    }


def compute_depreciation_rate(new_price_lakh: float | None, predicted_price_lakh: float) -> float | None:
    if new_price_lakh is None or not np.isfinite(new_price_lakh) or new_price_lakh <= 0:
        return None
    return float((new_price_lakh - predicted_price_lakh) / new_price_lakh)


def estimate_new_price(bundle: ModelBundle, brand: str) -> float | None:
    df = bundle.reference_df
    subset = df[(df["brand"] == brand) & (df["new_price_lakh"].notna())]
    if subset.empty:
        return None
    return float(subset["new_price_lakh"].median())


def find_similar_cars_by_name(bundle: ModelBundle, car_name: str, k: int = 5) -> pd.DataFrame:
    df = bundle.reference_df
    if df.empty or not car_name.strip() or "car_name" not in df.columns:
        return pd.DataFrame()

    names = df["car_name"].dropna().astype(str).unique().tolist()
    query = car_name.strip()
    matched_names = get_close_matches(query, names, n=3, cutoff=0.35)

    if not matched_names:
        q = query.lower()
        matched_names = [n for n in names if q in n.lower()][:3]

    if not matched_names:
        return pd.DataFrame()

    anchor = df[df["car_name"] == matched_names[0]].head(1)
    if anchor.empty:
        return pd.DataFrame()

    anchor_brand = str(anchor.iloc[0]["brand"])
    pool = df[df["brand"] == anchor_brand].copy()
    if pool.empty:
        pool = df.copy()

    num_cols = bundle.numeric_features
    arr = pool[num_cols].to_numpy(dtype=float)
    center = anchor[num_cols].to_numpy(dtype=float)[0]

    std = np.nanstd(arr, axis=0)
    std[std == 0] = 1.0
    dist = np.linalg.norm((arr - center) / std, axis=1)

    out = pool.assign(_dist=dist).sort_values("_dist").head(k)
    show_cols = [
        "car_name",
        "brand",
        "car_age_years",
        "engine_power_hp",
        "fuel_consumption_l_per_100km",
        "engine_displacement_cc",
        TARGET_COLUMN,
    ]
    out = out[show_cols].copy()
    out = out.rename(
        columns={
            "car_name": "Car name",
            "brand": "Brand",
            "car_age_years": "Age (years)",
            "engine_power_hp": "Power (HP)",
            "fuel_consumption_l_per_100km": "Fuel (L/100km)",
            "engine_displacement_cc": "Displacement (cc)",
            TARGET_COLUMN: "Price (lakh)",
        }
    )
    out.reset_index(drop=True, inplace=True)
    out.index = out.index + 1
    return out


def get_feature_importance(bundle: ModelBundle, top_n: int = 12) -> pd.DataFrame:
    model = bundle.pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame()

    preprocess = bundle.pipeline.named_steps["preprocess"]
    names = preprocess.get_feature_names_out(bundle.feature_columns)
    values = model.feature_importances_

    imp = pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False)
    return imp.head(top_n).reset_index(drop=True)


def predict_batch(bundle: ModelBundle, df_input: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Brand": "brand",
        "Car age": "car_age_years",
        "Engine power": "engine_power_hp",
        "Fuel consumption": "fuel_consumption_l_per_100km",
        "Engine displacement": "engine_displacement_cc",
    }
    df = df_input.rename(columns=rename_map).copy()

    for col in bundle.feature_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column in CSV: {col}")

    df_pred = df[bundle.feature_columns].copy()
    preds = bundle.pipeline.predict(df_pred)

    out = df_input.copy()
    out["Predicted_Price_lakh"] = preds
    out["Predicted_Price_VND"] = out["Predicted_Price_lakh"] * 30_000_000
    return out
