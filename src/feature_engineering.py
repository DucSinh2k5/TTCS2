import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.impute import SimpleImputer

def tao_moi_feature(df, km_median=None):
    """
    km_median=None  → chế độ fit (train): tính median từ df, trả về cùng giá trị.
    km_median=<số> → chế độ transform (test): dùng median đã fit từ train.
    Trả về: (df, km_median)
    """
    if "Nam_san_xuat" in df.columns:
        df["Tuoi_xe"] = datetime.now().year - df["Nam_san_xuat"]
    else:
        if "Nam_dang_ky" in df.columns:
            df["Tuoi_xe"] = datetime.now().year - df["Nam_dang_ky"]
    if "Ten_xe" in df.columns:
        df["Hang_xe"]  = df["Ten_xe"].str.split().str[0]
        if "Land Rover" in df["Ten_xe"].values:
            df.loc[df["Ten_xe"].str.contains("Land Rover", case=False, na=False), "Hang_xe"] = "Land Rover"
        # df["Hang_xe"] = df.loc[df["Hang_xe"].isin(["Toyota", "Honda", "Ford", "Mazda", "Hyundai", "Tata", "Mahindra","Land Rover","B"])]["Hang_xe"].fillna("Other")
    if "Quang_duong_da_di(km)" in df.columns:
        if "Nam_dang_ky" in df.columns:
            nam = datetime.now().year - df["Nam_dang_ky"]
            nam = nam.replace({0: 1})
            df["Km_moi_nam"] = df["Quang_duong_da_di(km)"] / nam
        else:
            df["Km_moi_nam"] = df["Quang_duong_da_di(km)"] / df["Tuoi_xe"].replace({0: 1})

    # Chỉ học ngưỡng median từ train; test dùng ngưỡng của train
    if km_median is None:
        km_median = df["Quang_duong_da_di(km)"].median()
    df["Chay_nhieu"] = (df["Quang_duong_da_di(km)"] > km_median).astype(int)

    df["log_Quang_duong_da_di(km)"] = np.log1p(df["Quang_duong_da_di(km)"].clip(lower=0))
    return df, km_median

def xu_ly_gia_tri_thieu(df, imputers=None):
    """
    imputers=None  → chế độ fit (train): fit SimpleImputer, trả về dict imputers.
    imputers=<dict> → chế độ transform (test): chỉ transform, không fit lại.
    Trả về: (df, imputers)
    """
    encode_list = ["Quyen_so_huu", "Hop_so", "Chay_nhieu"]

    num_cols = [col for col in df.columns
                if df[col].dtype.kind in "biufc"
                and col != "Gia_theo_lakh" and col != "Gia_moi_lakh"
                and col not in encode_list]
    cat_cols = [col for col in df.columns
                if df[col].dtype.kind == "object"
                and col != "Ten_xe"]

    if imputers is None:
        imputers = {}
        if num_cols:
            imp_med = SimpleImputer(strategy="median")
            df[num_cols] = imp_med.fit_transform(df[num_cols])
            imputers["median"] = (imp_med, num_cols)
        enc_present = [c for c in encode_list if c in df.columns]
        if enc_present:
            imp_mode = SimpleImputer(strategy="most_frequent")
            df[enc_present] = imp_mode.fit_transform(df[enc_present])
            imputers["mode"] = (imp_mode, enc_present)
    else:
        if "median" in imputers:
            imp_med, num_cols_fit = imputers["median"]
            for c in num_cols_fit:
                if c not in df.columns:
                    df[c] = np.nan
            df[num_cols_fit] = imp_med.transform(df[num_cols_fit])
        if "mode" in imputers:
            imp_mode, enc_fit = imputers["mode"]
            for c in enc_fit:
                if c not in df.columns:
                    df[c] = np.nan
            df[enc_fit] = imp_mode.transform(df[enc_fit])

    for col in cat_cols:
        df[col] = df[col].fillna("Unknown").astype("str")
    return df, imputers

def gioi_han_xe(df, top_names=None):
    """
    top_names=None  → chế độ fit (train): học top 20 từ df.
    top_names=<index> → chế độ transform (test): dùng top 20 của train.
    Trả về: (df, top_names)
    """
    if "Ten_xe" in df.columns:
        if top_names is None:
            top_names = df["Ten_xe"].value_counts().nlargest(20).index
        df["Top_xe"] = df["Ten_xe"].apply(lambda x: x if x in top_names else "Other")
    return df, top_names


def gioi_han_hang_xe(df, top_brands=None, top_n=15):
    """
    top_brands=None  → chế độ fit (train): học top_n hãng từ df.
    top_brands=<index> → chế độ transform (test): dùng top hãng của train.
    Trả về: (df, top_brands)
    """
    if "Hang_xe" in df.columns:
        series = df["Hang_xe"].fillna("Other").astype(str)
        if top_brands is None:
            top_brands = series.value_counts().nlargest(top_n).index
        df["Hang_xe"] = series.apply(lambda x: x if x in top_brands else "Other")
    return df, top_brands


def xu_ly_outlier(df, bounds=None):
    """
    bounds=None   → chế độ fit (train): tính ngưỡng IQR từ df, trả về dict bounds.
    bounds=<dict> → chế độ transform (test): clip theo ngưỡng đã fit từ train.
    Trả về: (df, bounds)
    """
    cols_to_clip = ["Gia_theo_lakh", "Quang_duong_da_di(km)", "Km_moi_nam",
                    "Dung_tich(cc)", "Cong_suat_toi_da",
                    "Muc_tieu_hao(km/l)"]

    if bounds is None:
        bounds = {}
        for col in cols_to_clip:
            if col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
                df[col] = df[col].clip(*bounds[col])
    else:
        for col, (lower, upper) in bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(lower, upper)
    return df, bounds
