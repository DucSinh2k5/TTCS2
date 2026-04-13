import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from load_data_and_cleaning import load_data, chuyen_cot_sang_so, doi_ten_cot, chuyen_cot_sang_category, loai_bo_hang_ban
from EDA_after import eda_after
from EDA_before import eda_before
from preprocessing import tien_xu_ly
from feature_engineering import gioi_han_xe, gioi_han_hang_xe, tao_moi_feature, xu_ly_gia_tri_thieu, xu_ly_outlier
from train_and_evaluate import evaluate_model, save, train_model, tinh_metrics, kiem_tra_overfit


def clean_pipeline(df):
    df = doi_ten_cot(df)
    df = loai_bo_hang_ban(df)
    df = chuyen_cot_sang_so(df)
    df = chuyen_cot_sang_category(df)
    return df


def export_cleaned_dataset(df_source, km_median, imputers, top_names, top_brands, bounds, output_path):
    df_export = df_source.copy()
    df_export, _ = tao_moi_feature(df_export, km_median=km_median)
    df_export, _ = xu_ly_gia_tri_thieu(df_export, imputers=imputers)
    df_export, _ = gioi_han_xe(df_export, top_names=top_names)
    df_export, _ = gioi_han_hang_xe(df_export, top_brands=top_brands)
    df_export, _ = xu_ly_outlier(df_export, bounds=bounds)
    df_export.to_csv(output_path, index=False)
    print(f"Da luu du lieu da cleaning vao {output_path}")


def main():
    print("\n" + "=" * 60)
    print("DỰ ĐOÁN GIÁ XE Ô TÔ CŨ - MACHINE LEARNING")
    print("=" * 60 + "\n")

    # 1. Load
    df_train = load_data("../Datasets/train-data.csv")
    df_test  = load_data("../Datasets/test-data.csv")

    # 2. Làm sạch
    eda_before(df_train)
    df_train = clean_pipeline(df_train)
    df_test  = clean_pipeline(df_test)

    # 3. Tách validation từ train (20%) để đánh giá metrics
    df_tr, df_val = train_test_split(df_train, test_size=0.2, random_state=42)
    df_tr  = df_tr.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    print(f"✓ Train={len(df_tr)}, Validation={len(df_val)}, Test={len(df_test)}\n")

    # 4. Feature engineering — fit median chỉ từ train
    df_tr,  km_median = tao_moi_feature(df_tr)
    df_val, _         = tao_moi_feature(df_val, km_median=km_median)
    df_test, _        = tao_moi_feature(df_test, km_median=km_median)

    # 5. Missing values — fit imputers chỉ từ train
    df_tr,  imputers = xu_ly_gia_tri_thieu(df_tr)
    df_val, _        = xu_ly_gia_tri_thieu(df_val, imputers=imputers)
    df_test, _       = xu_ly_gia_tri_thieu(df_test, imputers=imputers)

    # 6. Giới hạn tên xe — học top 20 chỉ từ train
    df_tr,  top_names = gioi_han_xe(df_tr)
    df_val, _         = gioi_han_xe(df_val, top_names=top_names)
    df_test, _        = gioi_han_xe(df_test, top_names=top_names)

    # 6.5. Giới hạn hãng xe — học top hãng chỉ từ train để giảm overfitting do nhãn hiếm
    df_tr,  top_brands = gioi_han_hang_xe(df_tr, top_n=15)
    df_val, _          = gioi_han_hang_xe(df_val, top_brands=top_brands)
    df_test, _         = gioi_han_hang_xe(df_test, top_brands=top_brands)

    # 7. Outlier clipping — tính ngưỡng IQR chỉ từ train
    df_tr,  bounds = xu_ly_outlier(df_tr)
    df_val, _      = xu_ly_outlier(df_val, bounds=bounds)
    df_test, _     = xu_ly_outlier(df_test, bounds=bounds)
    df_tr.to_csv("../Datasets/train_cleaned.csv", index=False)
    print("✓ Đã lưu bản train đã cleaning: ../Datasets/train_cleaned.csv")
    # 7.5. Lưu bản train đã cleaning/feature engineering để app dùng lại
    export_cleaned_dataset(
        df_train,
        km_median=km_median,
        imputers=imputers,
        top_names=top_names,
        top_brands=top_brands,
        bounds=bounds,
        output_path="../Datasets/test.csv",
    )

    # 8. EDA — chỉ trên train
    eda_after(df_tr)

    # 9. Chuẩn bị target/feature tự động từ train
    target_col = "Gia_theo_lakh"
    excluded_cols = ["Gia_theo_lakh", "Gia_moi_lakh", "Unnamed: 0", "Nam_san_xuat", "Quang_duong_da_di(km)", "Ten_xe"]
    candidate_features = [c for c in df_tr.columns if c not in excluded_cols]
    if not candidate_features:
        raise ValueError("Khong tim thay cot feature trong train")

    # Loại dòng thiếu target để tránh NaN khi huấn luyện
    if df_tr[target_col].isna().any() or df_val[target_col].isna().any():
        before_tr, before_val = len(df_tr), len(df_val)
        df_tr = df_tr[df_tr[target_col].notna()].reset_index(drop=True)
        df_val = df_val[df_val[target_col].notna()].reset_index(drop=True)
        print(f"Da loai bo dong thieu target: train {before_tr}->{len(df_tr)}, val {before_val}->{len(df_val)}")

    numeric_features = [c for c in candidate_features if pd.api.types.is_numeric_dtype(df_tr[c])]
    categorical_features = [c for c in candidate_features if c not in numeric_features]

    print(f"\nNumeric features     ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}\n")

    # 10. Preprocessor — fit trên train, transform cả hai
    preprocessor = tien_xu_ly(numeric_features, categorical_features)
    X_tr  = df_tr[candidate_features]
    y_tr  = df_tr[target_col].values
    X_val = df_val[candidate_features]
    y_val = df_val[target_col].values

    # Đảm bảo test có đủ cột (điền NaN nếu thiếu)
    for col in candidate_features:
        if col not in df_test.columns:
            df_test[col] = np.nan
    X_test = df_test[candidate_features]

    X_tr_trans   = preprocessor.fit_transform(X_tr)
    X_val_trans  = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    # Tên features sau transform (không one-hot): giữ nguyên 11 numeric + 2 categorical
    feature_names = numeric_features + categorical_features

    # 11. Dùng toàn bộ feature sau transform để huấn luyện/đánh giá
    selected_features = feature_names
    selected_indices = list(range(len(feature_names)))
    print(f"Using all features ({len(selected_features)}): {selected_features}")

    # 12. Tìm tham số tối ưu cho RF rồi train
    # best_rf_params = tim_tham_so_rf(X_tr_trans, y_tr, selected_indices, n_iter=30, k=5)
    rf, lr, scaler, xgb = train_model(X_tr_trans, y_tr, selected_indices)

    # 13. Đánh giá trên validation set
    evaluate_model(rf, lr, scaler, xgb, X_val_trans, y_val, selected_indices)

    # 13.5. Cross-Validation — kiểm tra overfitting
    kiem_tra_overfit(X_tr_trans, y_tr, selected_indices, k=5)

    # 14. Dự đoán trên test set và in kết quả
    X_sel    = X_test_trans[:, selected_indices]
    rf_pred  = rf.predict(X_sel)
    lr_pred  = lr.predict(scaler.transform(X_sel))
    xgb_pred = xgb.predict(X_sel)
    df_test["Du_doan_RF"]  = rf_pred
    df_test["Du_doan_LR"]  = lr_pred
    df_test["Du_doan_XGB"] = xgb_pred

    preview_cols = [c for c in ["Ten_xe", "Du_doan_RF", "Du_doan_LR", "Du_doan_XGB"] if c in df_test.columns]
    # print("\nDự đoán trên tập test (10 dòng đầu):")
    # print(df_test[preview_cols].head(10).to_string(index=False))
    # df_test.to_csv("../Quan_sat/test_predictions.csv", index=False)
    # print("\n✓ Đã lưu dự đoán: ../Quan_sat/test_predictions.csv")

    # 15. Lưu model (dùng toàn bộ df_train gốc để lưu quan sát)
    save(df_tr, preprocessor, rf, lr, scaler, xgb, candidate_features,
         selected_features, selected_indices)

    print("\n✅ HOÀN THÀNH!\n")


if __name__ == "__main__":
    main()