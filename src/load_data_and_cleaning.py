import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import re
def load_data(file):
    df = pd.read_csv(file)
    # df = df.drop(columns=["STT: 0"],errors="ignore")
    return df
def doi_ten_cot(df):
    df = df.rename(columns=lambda i: i.strip())

    mapping = {
        "Name": "Ten_xe",
        "Year": "Nam_san_xuat",
        "Location": "Dia_diem",
        "Fuel_Type": "Loai_nhien_lieu",
        "Seats": "So_cho_ngoi",
        "Kilometers_Driven": "Quang_duong_da_di(km)",
        "Owner_Type": "Quyen_so_huu",
        "Transmission": "Hop_so",
        "Mileage": "Muc_tieu_hao(km/l)",
        "Engine": "Dung_tich(cc)",
        "Power": "Cong_suat_toi_da",
        "Price": "Gia_theo_lakh",
        "New_Price": "Gia_moi_lakh"
    }

    df = df.rename(columns=mapping, errors="ignore")
    return df

def loai_bo_hang_ban(df):
    if "Loai_nhien_lieu" in df.columns:
        valid_fuel = {"Petrol", "Diesel", "CNG", "LPG", "Electric", "Hybrid"}
        df = df[df["Loai_nhien_lieu"].astype(str).isin(valid_fuel)]

    if "Hop_so" in df.columns:
        df = df[df["Hop_so"].astype(str).isin({"Manual", "Automatic"})]

    if "Loai_bao_hiem" in df.columns:
        df["Loai_bao_hiem"] = df["Loai_bao_hiem"].replace("Third Party", "Third Party insurance")
        df = df[df["Loai_bao_hiem"].astype(str).isin({"Comprehensive", "Third Party insurance", "Zero Dep"})]

    if "Nam_san_xuat" in df.columns:
        year = pd.to_numeric(df["Nam_san_xuat"],errors="coerce")
        df = df[year.between(1990,2026)]

    if "So_cho_ngoi" in df.columns:
        seats = pd.to_numeric(df["So_cho_ngoi"],errors="coerce")
        df = df[seats.between(1,20)]

    drop_cols = ["Mo_men_xoan", "Nam_dang_ky", "Loai_bao_hiem", "Dia_diem"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df.reset_index(drop=True)

def chuyen_cot_sang_so(df):
    if "Gia_theo_lakh" in df.columns:
        df = df.dropna(subset=["Gia_theo_lakh"])
    
    cols = ["Quang_duong_da_di(km)","Muc_tieu_hao(km/l)","Dung_tich(cc)", "Cong_suat_toi_da","Gia_theo_lakh", "Gia_moi_lakh"]
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.split(" ").str[0]
            df[col] =pd.to_numeric(
                df[col].astype(str).str.replace(",","").str.extract(r'(\d+(?:\.\d+)?)')[0],
                errors="coerce"
            )
    if "Nam_dang_ky" in df.columns:
        original = df["Nam_dang_ky"].astype(str)
        mask = df["Nam_dang_ky"].isna()
        year = original[mask].str.extract(r'[-/]?(\d{2})$')[0]
        df.loc[mask, "Nam_dang_ky"] = "20" + year
        df["Nam_dang_ky"] = pd.to_numeric(df["Nam_dang_ky"], errors="coerce").astype("Int64")

    if"Nam_san_xuat" in df.columns:
        df["Nam_san_xuat"] = pd.to_numeric(df["Nam_san_xuat"], errors="coerce").astype("Int64")
        
    if "So_cho_ngoi" in df.columns:
        df["So_cho_ngoi"] = pd.to_numeric(df["So_cho_ngoi"],errors='coerce').astype('Int64')
    return df

def chuyen_cot_sang_category(df):
    if "Loai_bao_hiem" in df.columns:
        df["Loai_bao_hiem"] = df["Loai_bao_hiem"].map({
            "Comprehensive": 1,
            "Third Party insurance": 2,
            "Zero Dep": 3
        })
    if "Hop_so" in df.columns:
        df["Hop_so"] = df["Hop_so"].map({
            "Manual": 0,
            "Automatic": 1
        })
    if "Loai_nhien_lieu" in df.columns:
        df["Loai_nhien_lieu"] = df["Loai_nhien_lieu"].map({
            "Petrol": 0,
            "Diesel": 1,
            "CNG": 2,
            "LPG": 3,
            "Electric": 4,
            "Hybrid": 5
        })
    if "Dia_diem" in df.columns:
        df["Dia_diem"] = df["Dia_diem"].astype("category").cat.codes
    if "Quyen_so_huu" in df.columns:
        def parse(i):
            if pd.isna(i):
                return np.nan
            s = str(i).lower()
            match = re.search(r'(\d+)',s)
            if match:
                return int(match.group(1))
            if 'first' in s: return 1
            if 'second' in s: return 2
            if 'third' in s: return 3
            return np.nan
        df["Quyen_so_huu"] = df["Quyen_so_huu"].apply(parse).astype("Int64")
    return df
# if __name__ == "__main__":
#     df = load_data("F:/Documents/CODE/Python/TTCS2/Datasets/merged_data.csv")
#     df = doi_ten_cot(df)
#     df = loai_bo_hang_ban(df)
#     df = chuyen_cot_sang_so(df)
#     df = chuyen_cot_sang_category(df)
#     print(df.info())