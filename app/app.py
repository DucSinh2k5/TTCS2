import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
from pathlib import Path

try:
    from groq import Groq
except Exception:
    Groq = None

# ─── Đường dẫn model ───────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "Models" / "model2.pkl"
PROCESSED_DATASET_PATH = Path(__file__).parent.parent / "Datasets" / "test.csv"
RAW_DATASET_PATH = Path(__file__).parent.parent / "Datasets" / "train-data.csv"
API_KEY_FILE_PATH = Path(__file__).parent / "api_key.txt"
CURRENT_YEAR = datetime.now().year

# ─── Mapping encode (khớp với chuyen_cot_sang_category) ────────
FUEL_MAP        = {"Petrol": 0, "Diesel": 1, "CNG": 2, "LPG": 3, "Electric": 4}
HOP_SO_MAP      = {"Manual": 0, "Automatic": 1}
OWNER_MAP       = {"First": 1, "Second": 2, "Third": 3, "Fourth & Above": 4}

# 1 lakh INR ≈ 30 triệu VNĐ (tỷ giá tham khảo)
LAKH_TO_VND = 30_000_000


def get_groq_api_key() -> str:
    """Lay API key tu file txt truoc, fallback qua st.secrets va bien moi truong."""
    api_key = ""
    try:
        api_key = API_KEY_FILE_PATH.read_text(encoding="utf-8").strip()
    except Exception:
        api_key = ""

    try:
        if not api_key:
            api_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("GROQ_API_KEY", "")

    api_key = str(api_key).strip()
    if not api_key or api_key.startswith("PASTE_"):
        return ""
    return api_key


@st.cache_resource(show_spinner="Đang tải model...")
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    dataset_path = PROCESSED_DATASET_PATH if PROCESSED_DATASET_PATH.exists() else RAW_DATASET_PATH
    df = pd.read_csv(dataset_path)

    if dataset_path == RAW_DATASET_PATH:
        # Fallback khi chua generate file processed tu main.py.
        df = df.rename(columns=lambda c: str(c).strip())
        df = df.rename(columns={
            "Name": "Ten_xe",
            "Year": "Nam_san_xuat",
            "Fuel_Type": "Loai_nhien_lieu",
            "Seats": "So_cho_ngoi",
            "Kilometers_Driven": "Quang_duong_da_di(km)",
            "Owner_Type": "Quyen_so_huu",
            "Transmission": "Hop_so",
            "Mileage": "Muc_tieu_hao(km/l)",
            "Engine": "Dung_tich(cc)",
            "Power": "Cong_suat_toi_da",
            "Price": "Gia_theo_lakh",
        }, errors="ignore")

    numeric_cols = [
        "Cong_suat_toi_da",
        "Dung_tich(cc)",
        "Muc_tieu_hao(km/l)",
        "Nam_san_xuat",
        "Gia_theo_lakh",
    ]
    for col in numeric_cols:
        if col in df.columns:
            # Tach so khoi chuoi don vi: "58.16 bhp", "998 CC", "26.6 km/kg"...
            cleaned = df[col].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0]
            df[col] = pd.to_numeric(cleaned, errors="coerce")

    required = [
        "Ten_xe",
        "Cong_suat_toi_da",
        "Dung_tich(cc)",
        "Muc_tieu_hao(km/l)",
        "Nam_san_xuat",
        "Gia_theo_lakh",
    ]
    return df.dropna(subset=[c for c in required if c in df.columns])


def tim_xe_tuong_tu(
    df: pd.DataFrame,
    power: float, engine: float, mileage: float, year: int,
    ten_xe: str | None = None,
    n: int = 5,
) -> pd.DataFrame:
    """Tìm n xe có specs gần nhất với xe người dùng nhập (Euclidean sau chuẩn hoá)."""
    feat_cols = ["Cong_suat_toi_da", "Dung_tich(cc)", "Muc_tieu_hao(km/l)", "Nam_san_xuat"]
    query = np.array([power, engine, mileage, float(year)])
    data  = df[feat_cols].to_numpy(dtype=float)
    stds  = data.std(axis=0)
    stds[stds == 0] = 1.0
    dist  = np.linalg.norm((data - query) / stds, axis=1)
    idx   = np.argsort(dist)[:n * 3]
    result = df.iloc[idx].copy()
    result["_dist"] = dist[idx]
    if ten_xe:
        mask = result["Ten_xe"] == ten_xe
        result = pd.concat([result[mask], result[~mask]])
    result = result.drop_duplicates("Ten_xe").head(n).reset_index(drop=True)
    result.index = result.index + 1
    return result


def build_similar_display(
    similar: pd.DataFrame,
    km: float | None,
    fuel_type: str | None,
    transmission: str | None,
    owner_type: str | None,
    seats: int | None,
) -> pd.DataFrame:
    base_cols = [
        "Ten_xe",
        "Nam_san_xuat",
        "Cong_suat_toi_da",
        "Dung_tich(cc)",
        "Muc_tieu_hao(km/l)",
        "Gia_theo_lakh",
    ]

    optional_cols = []
    if km is not None:
        optional_cols.append("Quang_duong_da_di(km)")
    if fuel_type is not None:
        optional_cols.append("Loai_nhien_lieu")
    if transmission is not None:
        optional_cols.append("Hop_so")
    if owner_type is not None:
        optional_cols.append("Quyen_so_huu")
    if seats is not None:
        optional_cols.append("So_cho_ngoi")

    selected_cols = [col for col in base_cols + optional_cols if col in similar.columns]
    similar_display = similar[selected_cols].copy()

    return similar_display.rename(columns={
        "Ten_xe": "Tên xe",
        "Nam_san_xuat": "Năm SX",
        "Cong_suat_toi_da": "Công suất (bhp)",
        "Dung_tich(cc)": "Dung tích (cc)",
        "Muc_tieu_hao(km/l)": "Tiêu hao (km/l)",
        "Gia_theo_lakh": "Giá thực tế (lakh ₹)",
        "Quang_duong_da_di(km)": "Km đã đi",
        "Loai_nhien_lieu": "Loại nhiên liệu",
        "Hop_so": "Hộp số",
        "Quyen_so_huu": "Quyền sở hữu",
        "So_cho_ngoi": "Số chỗ ngồi",
    })


def build_car_comparison(similar: pd.DataFrame, selected_names: list[str]) -> pd.DataFrame:
    """Tạo bảng so sánh 2 xe theo các thông số chính."""
    compare_cols = [
        "Ten_xe",
        "Nam_san_xuat",
        "Cong_suat_toi_da",
        "Dung_tich(cc)",
        "Muc_tieu_hao(km/l)",
        "Gia_theo_lakh",
        "Quang_duong_da_di(km)",
        "Loai_nhien_lieu",
        "Hop_so",
        "Quyen_so_huu",
        "So_cho_ngoi",
    ]
    available_cols = [c for c in compare_cols if c in similar.columns]
    filtered = similar[similar["Ten_xe"].isin(selected_names)][available_cols].copy()

    rename_map = {
        "Ten_xe": "Tên xe",
        "Nam_san_xuat": "Năm SX",
        "Cong_suat_toi_da": "Công suất (bhp)",
        "Dung_tich(cc)": "Dung tích (cc)",
        "Muc_tieu_hao(km/l)": "Tiêu hao (km/l)",
        "Gia_theo_lakh": "Giá (lakh ₹)",
        "Quang_duong_da_di(km)": "Km đã đi",
        "Loai_nhien_lieu": "Nhiên liệu",
        "Hop_so": "Hộp số",
        "Quyen_so_huu": "Quyền sở hữu",
        "So_cho_ngoi": "Số chỗ",
    }
    filtered = filtered.rename(columns=rename_map)
    return filtered.set_index("Tên xe").T


def build_input_df(
    year: int,
    power: float,
    engine: float,
    mileage: float,
    km: float | None,
    fuel_type: str | None,
    transmission: str | None,
    owner_type: str | None,
    seats: int | None,
    ten_xe: str | None = None,
) -> pd.DataFrame:
    """Tạo DataFrame 1 hàng theo đúng tên cột và encoding của pipeline train."""
    tuoi_xe = float(CURRENT_YEAR - year)

    row: dict = {
        "Cong_suat_toi_da":       power,
        "Tuoi_xe":                tuoi_xe,
        "Dung_tich(cc)":          engine,
        "Muc_tieu_hao(km/l)":     mileage,
        "So_cho_ngoi":            float(seats) if seats is not None else np.nan,
        "Loai_nhien_lieu":        float(FUEL_MAP[fuel_type])   if fuel_type    else np.nan,
        "Hop_so":                 float(HOP_SO_MAP[transmission]) if transmission else np.nan,
        "Quyen_so_huu":           float(OWNER_MAP[owner_type])  if owner_type  else np.nan,
        "Top_xe":                 ten_xe if ten_xe else "Other",
    }

    if km is not None and km > 0:
        # km_median ≈ 75 000 km (ngưỡng trung vị trên tập train)
        row["Quang_duong_da_di(km)"]      = float(km)
        row["Km_moi_nam"]                 = float(km) / max(tuoi_xe, 1.0)
        row["Chay_nhieu"]                 = 1.0 if km > 75_000 else 0.0
        row["log_Quang_duong_da_di(km)"]  = np.log1p(float(km))
    else:
        row["Quang_duong_da_di(km)"]      = np.nan
        row["Km_moi_nam"]                 = np.nan
        row["Chay_nhieu"]                 = np.nan
        row["log_Quang_duong_da_di(km)"]  = np.nan

    return pd.DataFrame([row])


def predict_xgb(model_data: dict, df: pd.DataFrame) -> float:
    preprocessor      = model_data["preprocessor"]
    model_xgb         = model_data["model_xgb"]
    candidate_features = model_data["candidate_features"]
    selected_indices  = model_data["selected_indices"]

    # Đảm bảo đủ cột (điền NaN nếu thiếu)
    for col in candidate_features:
        if col not in df.columns:
            df[col] = np.nan

    X     = preprocessor.transform(df[candidate_features])
    X_sel = X[:, selected_indices]
    return float(model_xgb.predict(X_sel)[0])


def render_compare_ai_page() -> None:
    st.title("⚖️ So sánh 2 xe ô tô")
    st.markdown("So sánh chi tiết 2 xe ngay trong app bằng AI.")

    recommended_df = st.session_state.get("recommended_df")
    default_pair = st.session_state.get("selected_two_cars", [])

    st.sidebar.header("⚙️ Chọn xe để so sánh")
    selected_from_top5: list[str] = []

    if recommended_df is not None and not recommended_df.empty and "Ten_xe" in recommended_df.columns:
        options = recommended_df["Ten_xe"].dropna().unique().tolist()
        default_options = default_pair if len(default_pair) == 2 else (options[:2] if len(options) >= 2 else options)
        selected_from_top5 = st.sidebar.multiselect(
            "Chọn đúng 2 xe từ Top 5 recommend",
            options=options,
            default=default_options,
            max_selections=2,
            key="compare_tab_pick",
        )
        if len(selected_from_top5) == 2:
            st.session_state["selected_two_cars"] = selected_from_top5
        else:
            st.sidebar.warning("Vui lòng chọn đúng 2 xe.")
    else:
        st.sidebar.info("Chưa có Top 5 recommend. Hãy sang menu Dự đoán giá để tạo danh sách này.")

    col1, col2 = st.columns(2)
    with col1:
        xe1 = st.text_input(
            "Xe thứ nhất",
            value=selected_from_top5[0] if len(selected_from_top5) == 2 else "",
            placeholder="VD: Toyota Camry 2024",
        )
    with col2:
        xe2 = st.text_input(
            "Xe thứ hai",
            value=selected_from_top5[1] if len(selected_from_top5) == 2 else "",
            placeholder="VD: Honda Accord 2024",
        )

    api_key = get_groq_api_key()

    if api_key:
        st.sidebar.success("Da nap san Groq API key.")
    else:
        st.sidebar.warning("Chua tim thay Groq API key trong secrets/env.")
        st.sidebar.caption("Them GROQ_API_KEY vao .streamlit/secrets.toml hoac bien moi truong.")

    if st.button("So sánh ngay", type="primary"):
        if not xe1 or not xe2:
            st.warning("Vui lòng nhập hoặc chọn đủ 2 xe.")
        elif not api_key:
            st.error("Thiếu Groq API key.")
        elif Groq is None:
            st.error("Chưa cài gói groq trong môi trường hiện tại.")
        else:
            prompt = f"""Hãy so sánh chi tiết 2 xe ô tô sau: **{xe1}** và **{xe2}**.

Vui lòng so sánh theo các tiêu chí:
1. Thông số kỹ thuật
2. Kích thước & trọng lượng
3. Trang bị an toàn
4. Tiện nghi & công nghệ
5. Mức tiêu hao nhiên liệu
6. Giá bán (nếu có)
7. Ưu & nhược điểm
8. Kết luận: Nên chọn xe nào và phù hợp đối tượng nào

Trả lời bằng tiếng Việt, rõ ràng và dễ hiểu."""

            client = Groq(api_key=api_key)

            def stream_response():
                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

            try:
                st.write_stream(stream_response())
            except Exception as exc:
                st.error(f"Lỗi khi gọi AI: {exc}")


# ═══════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Dự đoán giá xe ô tô cũ",
    page_icon="🚗",
    layout="centered",
)

st.markdown(
    """
    <style>
        :root {
            --sb-width: 430px;
            --sb-radius: 24px;
            --sb-shadow: 0 18px 42px rgba(30, 41, 59, 0.22);
            --sb-border: rgba(255, 255, 255, 0.42);
            --sb-bg: linear-gradient(165deg, rgba(255, 255, 255, 0.82), rgba(247, 250, 255, 0.66));
            --sb-label: #3f4a59;
            --sb-input-bg: rgba(255, 255, 255, 0.92);
            --sb-input-border: #d8e2ef;
        }

        /* Nền trái to hơn để sidebar nhìn bề thế */
        section[data-testid="stSidebar"] {
            width: var(--sb-width) !important;
            min-width: var(--sb-width) !important;
            max-width: var(--sb-width) !important;
            background: transparent !important;
        }

        section[data-testid="stSidebar"] > div:first-child {
            width: var(--sb-width) !important;
            margin: 14px;
            border-radius: var(--sb-radius);
            border: 1px solid var(--sb-border);
            background: var(--sb-bg);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            box-shadow: var(--sb-shadow);
            padding: 14px 14px 24px 14px;
        }

        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
            color: var(--sb-label) !important;
            font-weight: 600 !important;
            font-size: 1.03rem !important;
        }

        section[data-testid="stSidebar"] [data-baseweb="select"] > div,
        section[data-testid="stSidebar"] [data-testid="stNumberInput"] input,
        section[data-testid="stSidebar"] [data-testid="stTextInput"] input,
        section[data-testid="stSidebar"] textarea {
            background: var(--sb-input-bg) !important;
            border: 1px solid var(--sb-input-border) !important;
            border-radius: 14px !important;
            min-height: 48px;
            font-size: 1.04rem !important;
            color: #2f3742 !important;
        }

        section[data-testid="stSidebar"] [data-baseweb="select"] > div {
            box-shadow: inset 0 0 0 1px transparent;
        }

        section[data-testid="stSidebar"] [data-baseweb="slider"] {
            padding-top: 4px;
        }

        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
            margin-top: 2px;
        }

        section[data-testid="stSidebar"] hr {
            border-color: rgba(148, 163, 184, 0.22) !important;
        }

        @media (max-width: 1200px) {
            :root {
                --sb-width: 380px;
            }
        }

        @media (max-width: 900px) {
            :root {
                --sb-width: 320px;
            }

            section[data-testid="stSidebar"] > div:first-child {
                margin: 8px;
                border-radius: 18px;
                padding: 10px 10px 18px 10px;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "recommended_df" not in st.session_state:
    st.session_state["recommended_df"] = None
if "selected_two_cars" not in st.session_state:
    st.session_state["selected_two_cars"] = []
if "menu_tab" not in st.session_state:
    st.session_state["menu_tab"] = "Dự đoán giá"

menu = st.sidebar.radio(
    "📂 Menu",
    ["Dự đoán giá", "So sánh 2 xe (AI)"],
    key="menu_tab",
)

if menu == "So sánh 2 xe (AI)":
    render_compare_ai_page()
    st.stop()

st.title("🚗 Dự đoán giá xe ô tô cũ")
st.markdown(
    "Nhập thông tin xe bên dưới để nhận dự đoán giá từ mô hình **XGBoost**.  \n"
    "Các trường đánh dấu **\\*** là **bắt buộc**."
)

model_data = load_model()

df_dataset = load_dataset()
all_car_names = sorted(df_dataset["Ten_xe"].unique().tolist())

# ─── Thông tin bắt buộc ────────────────────────────────────────
st.subheader("📌 Thông tin bắt buộc")
c1, c2 = st.columns(2)

with c1:
    year = st.number_input(
        "Năm sản xuất *",
        min_value=1990, max_value=CURRENT_YEAR, value=2018, step=1,
        help="Năm xe được sản xuất. Tuổi xe = năm hiện tại − năm sản xuất.",
    )
    engine = st.number_input(
        "Dung tích động cơ (cc) *",
        min_value=100.0, max_value=10_000.0, value=1500.0, step=50.0,
        help="Dung tích xi-lanh, đơn vị cc (cm³). Ví dụ: 1197, 1498, 1968.",
    )

with c2:
    power = st.number_input(
        "Công suất tối đa (bhp) *",
        min_value=10.0, max_value=1_000.0, value=100.0, step=1.0,
        help="Công suất cực đại của động cơ, đơn vị bhp. Ví dụ: 74, 118, 170.",
    )
    mileage = st.number_input(
        "Mức tiêu hao nhiên liệu (km/l) *",
        min_value=1.0, max_value=100.0, value=17.0, step=0.1,
        help="Số km đi được trên 1 lít nhiên liệu. Ví dụ: 15.2, 23.1.",
    )

st.caption(f"Tuổi xe tính được: **{CURRENT_YEAR - year} năm** (năm hiện tại {CURRENT_YEAR} − {year})")

# ─── Thông tin tùy chọn ────────────────────────────────────────
st.subheader("📝 Thông tin bổ sung (tùy chọn)")
st.caption("Cung cấp thêm thông tin giúp pipeline xử lý đầy đủ hơn.")

car_name_sel = st.selectbox(
    "🔍 Tên / model xe (tùy chọn)",
    ["(Không chọn)"] + all_car_names,
    help="Chọn tên xe nếu bạn biết. Kết quả sẽ hiển thị các xe tương tự trong dataset.",
)
car_name_val = car_name_sel if not car_name_sel.startswith("(") else None

c3, c4 = st.columns(2)

with c3:
    km_raw = st.number_input(
        "Số km đã đi (km)",
        min_value=0, max_value=1_500_000, value=0, step=1_000,
        help="Tổng quãng đường xe đã đi. Để 0 nếu không biết.",
    )
    fuel_sel = st.selectbox(
        "Loại nhiên liệu",
        ["(Không chọn)", "Petrol", "Diesel", "CNG", "LPG", "Electric"],
    )
    trans_sel = st.selectbox(
        "Hộp số",
        ["(Không chọn)", "Manual", "Automatic"],
    )

with c4:
    owner_sel = st.selectbox(
        "Quyền sở hữu (đời chủ)",
        ["(Không chọn)", "First", "Second", "Third", "Fourth & Above"],
        help="First = chủ đầu tiên, Second = chủ thứ hai, …",
    )
    seats_raw = st.number_input(
        "Số chỗ ngồi",
        min_value=0, max_value=20, value=0, step=1,
        help="Số chỗ ngồi. Để 0 nếu không biết.",
    )

# ─── Parse optional ────────────────────────────────────────────
km_val         = float(km_raw)  if km_raw > 0      else None
fuel_val       = fuel_sel       if not fuel_sel.startswith("(")  else None
trans_val      = trans_sel      if not trans_sel.startswith("(") else None
owner_val      = owner_sel      if not owner_sel.startswith("(") else None
seats_val      = int(seats_raw) if seats_raw > 0   else None

# ─── Chọn 2 xe để so sánh ngay trong tab dự đoán ──────────────
recommended_df = st.session_state.get("recommended_df")
st.subheader("⚖️ Chọn 2 xe để so sánh")

if recommended_df is None or recommended_df.empty:
    st.info("Sau lần dự đoán đầu tiên, bạn có thể chọn 2 xe từ Top 5 recommend tại đây.")
else:
    predict_options = recommended_df["Ten_xe"].dropna().unique().tolist()
    prev_pair = [c for c in st.session_state.get("selected_two_cars", []) if c in predict_options]
    default_pair = prev_pair if len(prev_pair) == 2 else (predict_options[:2] if len(predict_options) >= 2 else predict_options)
    selected_two_cars_predict = st.multiselect(
        "Chọn đúng 2 xe",
        options=predict_options,
        default=default_pair,
        max_selections=2,
        key="predict_tab_pick",
    )
    if len(selected_two_cars_predict) == 2:
        st.session_state["selected_two_cars"] = selected_two_cars_predict
    else:
        st.warning("Vui lòng chọn đúng 2 xe để so sánh.")

# ─── Nút dự đoán ───────────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("🔮 Dự đoán giá xe", use_container_width=True, type="primary")

if predict_btn:
    try:
        df_input    = build_input_df(
            year, power, engine, mileage,
            km_val, fuel_val, trans_val, owner_val, seats_val,
            ten_xe=car_name_val,
        )
        price_lakh  = predict_xgb(model_data, df_input)
        price_vnd   = price_lakh * LAKH_TO_VND

        st.success("✅ Dự đoán thành công!")

        if car_name_val:
            st.markdown(f"### 🚗 Xe: **{car_name_val}**")

        r1, r2, r3 = st.columns(3)
        r1.metric("💰 Giá dự đoán (lakh ₹)",  f"{price_lakh:.2f}")
        r2.metric("🇮🇳 INR",                   f"{price_lakh * 100_000:,.0f} ₹")
        r3.metric("🇻🇳 VNĐ (tham khảo)",       f"{price_vnd / 1_000_000:.1f} triệu")

        # st.info(
        #     f"**{price_lakh:.2f} lakh ₹** "
        #     f"= {price_lakh * 100_000:,.0f} Rupee Ấn Độ  \n"
        #     f"≈ **{price_vnd / 1_000_000:.1f} triệu VNĐ** "
        #     f"(tỷ giá tham khảo 1 lakh ₹ ≈ 30 triệu VNĐ)"
        # )

        with st.expander("🔎 Xe tương tự trong dataset", expanded=True):
            similar = tim_xe_tuong_tu(
                df_dataset, power, engine, mileage, year,
                ten_xe=car_name_val, n=5,
            )
            st.session_state["recommended_df"] = similar
            new_options = similar["Ten_xe"].dropna().unique().tolist()
            old_pair = [c for c in st.session_state.get("selected_two_cars", []) if c in new_options]
            st.session_state["selected_two_cars"] = old_pair if len(old_pair) == 2 else new_options[:2]
            similar_display = build_similar_display(
                similar,
                km=km_val,
                fuel_type=fuel_val,
                transmission=trans_val,
                owner_type=owner_val,
                seats=seats_val,
            )
            st.dataframe(similar_display, use_container_width=True)
            st.caption("Xe được chọn dựa trên độ tương đồng công suất, dung tích, tiêu hao và năm sản xuất.")

        with st.expander("📊 Xem chi tiết đầu vào đã xử lý"):
            label_map = {
                "Cong_suat_toi_da":    "Công suất tối đa (bhp)",
                "Tuoi_xe":             "Tuổi xe (năm)",
                "Dung_tich(cc)":       "Dung tích động cơ (cc)",
                "Muc_tieu_hao(km/l)":  "Mức tiêu hao (km/l)",
                "Quang_duong_da_di(km)":"Số km đã đi",
                "Km_moi_nam":          "Km / năm",
                "Chay_nhieu":          "Chạy nhiều (1=có)",
                "log_Quang_duong_da_di(km)": "log(km)",
                "So_cho_ngoi":         "Số chỗ ngồi",
                "Loai_nhien_lieu":     "Loại nhiên liệu (mã)",
                "Hop_so":              "Hộp số (mã)",
                "Quyen_so_huu":        "Quyền sở hữu (đời chủ)",
                "Top_xe":              "Nhóm tên xe",
            }
            display_df = df_input.rename(columns=label_map)
            st.dataframe(display_df.T.rename(columns={0: "Giá trị"}), use_container_width=True)

        st.session_state["menu_tab"] = "So sánh 2 xe (AI)"
        st.rerun()

    except Exception as exc:
        st.error(f"❌ Lỗi khi dự đoán: {exc}")


