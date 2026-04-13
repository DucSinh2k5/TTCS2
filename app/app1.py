from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).parent.parent / "Models" / "model2.pkl"
REF_DATA_PATH = Path(__file__).parent.parent / "Datasets" / "test.csv"
CURRENT_YEAR = datetime.now().year
LAKH_TO_VND = 30_000_000

FUEL_MAP = {
    "Petrol": 0,
    "Diesel": 1,
    "CNG": 2,
    "LPG": 3,
    "Electric": 4,
    "Hybrid": 5,
}
TRANS_MAP = {
    "Manual": 0,
    "Automatic": 1,
}
OWNER_MAP = {
    "First": 1,
    "Second": 2,
    "Third": 3,
    "Fourth & Above": 4,
}


@st.cache_resource(show_spinner="Loading model...")
def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run src/main.py first to generate model2.pkl"
        )
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_reference_data() -> pd.DataFrame:
    if not REF_DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(REF_DATA_PATH)


def infer_hang_xe(car_name: str, known_brands: set[str]) -> str:
    text = (car_name or "").strip()
    if not text:
        return "Other"

    if text.lower().startswith("land rover"):
        brand = "Land Rover"
    else:
        brand = text.split()[0]

    return brand if brand in known_brands else "Other"


def infer_top_xe(car_name: str, known_top_names: set[str]) -> str:
    text = (car_name or "").strip()
    if not text:
        return "Other"
    return text if text in known_top_names else "Other"


def build_feature_row(
    car_name: str,
    year: int,
    km_driven: float,
    fuel_type: str,
    transmission: str,
    owner_type: str,
    mileage_km_l: float,
    engine_cc: float,
    power_bhp: float,
    seats: int,
    known_brands: set[str],
    known_top_names: set[str],
) -> pd.DataFrame:
    tuoi_xe = max(0.0, float(CURRENT_YEAR - int(year)))
    km_val = max(0.0, float(km_driven))

    row = {
        "Loai_nhien_lieu": float(FUEL_MAP[fuel_type]),
        "Hop_so": float(TRANS_MAP[transmission]),
        "Quyen_so_huu": float(OWNER_MAP[owner_type]),
        "Muc_tieu_hao(km/l)": float(mileage_km_l),
        "Dung_tich(cc)": float(engine_cc),
        "Cong_suat_toi_da": float(power_bhp),
        "So_cho_ngoi": float(seats),
        "Tuoi_xe": tuoi_xe,
        "Km_moi_nam": km_val / max(tuoi_xe, 1.0),
        "Chay_nhieu": 1.0 if km_val > 75_000 else 0.0,
        "log_Quang_duong_da_di(km)": float(np.log1p(km_val)),
        "Hang_xe": infer_hang_xe(car_name, known_brands),
        "Top_xe": infer_top_xe(car_name, known_top_names),
    }
    return pd.DataFrame([row])


def predict_with_model(model_data: dict, feature_df: pd.DataFrame, model_name: str) -> float:
    preprocessor = model_data["preprocessor"]
    candidate_features = model_data["candidate_features"]
    selected_indices = model_data["selected_indices"]

    for col in candidate_features:
        if col not in feature_df.columns:
            feature_df[col] = np.nan

    x = preprocessor.transform(feature_df[candidate_features])
    x_sel = x[:, selected_indices]

    if model_name == "Random Forest":
        pred = model_data["model_rf"].predict(x_sel)[0]
    elif model_name == "Linear Regression":
        x_scaled = model_data["scaler_lr"].transform(x_sel)
        pred = model_data["model_lr"].predict(x_scaled)[0]
    else:
        pred = model_data["model_xgb"].predict(x_sel)[0]

    return float(pred)


def batch_from_raw(df_in: pd.DataFrame, known_brands: set[str], known_top_names: set[str]) -> pd.DataFrame:
    required = [
        "Ten_xe",
        "Nam_san_xuat",
        "Quang_duong_da_di(km)",
        "Loai_nhien_lieu",
        "Hop_so",
        "Quyen_so_huu",
        "Muc_tieu_hao(km/l)",
        "Dung_tich(cc)",
        "Cong_suat_toi_da",
        "So_cho_ngoi",
    ]
    missing = [c for c in required if c not in df_in.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    out = pd.DataFrame()
    out["Loai_nhien_lieu"] = df_in["Loai_nhien_lieu"].map(FUEL_MAP)
    out["Hop_so"] = df_in["Hop_so"].map(TRANS_MAP)
    out["Quyen_so_huu"] = df_in["Quyen_so_huu"].map(OWNER_MAP)
    out["Muc_tieu_hao(km/l)"] = pd.to_numeric(df_in["Muc_tieu_hao(km/l)"], errors="coerce")
    out["Dung_tich(cc)"] = pd.to_numeric(df_in["Dung_tich(cc)"], errors="coerce")
    out["Cong_suat_toi_da"] = pd.to_numeric(df_in["Cong_suat_toi_da"], errors="coerce")
    out["So_cho_ngoi"] = pd.to_numeric(df_in["So_cho_ngoi"], errors="coerce")

    years = pd.to_numeric(df_in["Nam_san_xuat"], errors="coerce")
    kms = pd.to_numeric(df_in["Quang_duong_da_di(km)"], errors="coerce").clip(lower=0)

    tuoi_xe = (CURRENT_YEAR - years).clip(lower=0)
    out["Tuoi_xe"] = tuoi_xe
    out["Km_moi_nam"] = kms / tuoi_xe.replace(0, 1)
    out["Chay_nhieu"] = (kms > 75_000).astype(float)
    out["log_Quang_duong_da_di(km)"] = np.log1p(kms)

    car_names = df_in["Ten_xe"].fillna("").astype(str)
    out["Hang_xe"] = car_names.apply(lambda x: infer_hang_xe(x, known_brands))
    out["Top_xe"] = car_names.apply(lambda x: infer_top_xe(x, known_top_names))

    return out


def main():
    st.set_page_config(page_title="Used Car Price App", page_icon="CAR", layout="wide")
    st.title("Used Car Price Prediction")
    st.caption("Streamlit UI for model2.pkl with the same preprocessing flow used in training")

    model_data = load_model_bundle()
    ref_df = load_reference_data()

    known_brands = set(ref_df["Hang_xe"].dropna().astype(str).unique()) if "Hang_xe" in ref_df.columns else {"Other"}
    known_top_names = set(ref_df["Top_xe"].dropna().astype(str).unique()) if "Top_xe" in ref_df.columns else {"Other"}

    if not known_brands:
        known_brands = {"Other"}
    if not known_top_names:
        known_top_names = {"Other"}

    st.sidebar.header("Model settings")
    model_name = st.sidebar.selectbox("Model", ["XGBoost", "Random Forest", "Linear Regression"], index=0)

    tab_single, tab_batch = st.tabs(["Single prediction", "Batch prediction (CSV)"])

    with tab_single:
        c1, c2 = st.columns(2)

        with c1:
            car_name = st.text_input("Car name (optional)", placeholder="e.g. Toyota Fortuner 3.0 Diesel")
            year = st.number_input("Year", min_value=1990, max_value=CURRENT_YEAR, value=2018, step=1)
            km_driven = st.number_input("Kilometers driven", min_value=0.0, max_value=2_000_000.0, value=60_000.0, step=500.0)
            fuel_type = st.selectbox("Fuel type", list(FUEL_MAP.keys()), index=1)
            transmission = st.selectbox("Transmission", list(TRANS_MAP.keys()), index=0)
            owner_type = st.selectbox("Owner type", list(OWNER_MAP.keys()), index=0)

        with c2:
            mileage_km_l = st.number_input("Mileage (km/l)", min_value=2.0, max_value=40.0, value=18.0, step=0.1)
            engine_cc = st.number_input("Engine displacement (cc)", min_value=600.0, max_value=9000.0, value=1500.0, step=10.0)
            power_bhp = st.number_input("Power (bhp)", min_value=20.0, max_value=1200.0, value=110.0, step=1.0)
            seats = st.number_input("Seats", min_value=2, max_value=12, value=5, step=1)

        if st.button("Predict", type="primary", use_container_width=True):
            feature_df = build_feature_row(
                car_name=car_name,
                year=year,
                km_driven=km_driven,
                fuel_type=fuel_type,
                transmission=transmission,
                owner_type=owner_type,
                mileage_km_l=mileage_km_l,
                engine_cc=engine_cc,
                power_bhp=power_bhp,
                seats=seats,
                known_brands=known_brands,
                known_top_names=known_top_names,
            )

            pred_lakh = predict_with_model(model_data, feature_df, model_name=model_name)
            pred_vnd = pred_lakh * LAKH_TO_VND

            m1, m2 = st.columns(2)
            m1.metric("Predicted price (lakh)", f"{pred_lakh:.2f}")
            m2.metric("Predicted price (VND)", f"{pred_vnd:,.0f}")

            with st.expander("Feature row sent to model"):
                st.dataframe(feature_df, use_container_width=True)

    with tab_batch:
        st.caption(
            "CSV can be in raw format with columns: Ten_xe, Nam_san_xuat, Quang_duong_da_di(km), "
            "Loai_nhien_lieu, Hop_so, Quyen_so_huu, Muc_tieu_hao(km/l), Dung_tich(cc), Cong_suat_toi_da, So_cho_ngoi"
        )
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded is not None:
            try:
                df_in = pd.read_csv(uploaded)

                if all(col in df_in.columns for col in model_data["candidate_features"]):
                    feature_df = df_in[model_data["candidate_features"]].copy()
                else:
                    feature_df = batch_from_raw(df_in, known_brands, known_top_names)

                preds = []
                for i in range(len(feature_df)):
                    preds.append(predict_with_model(model_data, feature_df.iloc[[i]].copy(), model_name=model_name))

                df_out = df_in.copy()
                df_out["Predicted_lakh"] = preds
                df_out["Predicted_VND"] = (df_out["Predicted_lakh"] * LAKH_TO_VND).round(0)

                st.success(f"Predicted {len(df_out)} rows")
                st.dataframe(df_out.head(30), use_container_width=True)
                st.download_button(
                    "Download predictions CSV",
                    data=df_out.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Batch prediction error: {exc}")


if __name__ == "__main__":
    main()
