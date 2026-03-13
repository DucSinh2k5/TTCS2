import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from xgboost import XGBRegressor
import joblib


# def tim_tham_so_rf(X_train, y_train, selected_indices, n_iter=30, k=5):
#     """
#     Dùng RandomizedSearchCV để tìm bộ tham số tốt nhất cho Random Forest.
#     n_iter : số tổ hợp thử ngẫu nhiên (càng cao càng chính xác, càng chậm)
#     k      : số fold cross-validation
#     """
#     X_sel = X_train[:, selected_indices]

#     param_dist = {
#         "n_estimators":     [100, 200, 300, 500],
#         "max_depth":        [6, 8, 10, 12, 15, None],
#         "min_samples_leaf": [2, 3, 5, 8, 10],
#         "min_samples_split":[5, 10, 15, 20],
#         "max_features":     [0.4, 0.5, 0.6, 0.7, "sqrt"],
#         "max_samples":      [0.7, 0.8, 0.9, None],
#     }

#     search = RandomizedSearchCV(
#         RandomForestRegressor(random_state=42, n_jobs=-1),
#         param_distributions=param_dist,
#         n_iter=n_iter,
#         scoring="r2",
#         cv=KFold(n_splits=k, shuffle=True, random_state=42),
#         random_state=42,
#         n_jobs=-1,
#         verbose=0,
#         refit=True,
#     )
#     print(f"Dang tim tham so RF ({n_iter} to hop x {k}-fold CV)...")
#     search.fit(X_sel, y_train)

#     best = search.best_params_
#     sep = "=" * 60
#     print(sep)
#     print("KET QUA RANDOMIZED SEARCH — THAM SO TOI UU RF")
#     print(sep)
#     for k_name, v in best.items():
#         print(f"  {k_name:<22}: {v}")
#     print(f"  {'R2 CV tot nhat':<22}: {search.best_score_:.4f}")
#     print(sep + "\n")
#     return best


def train_model(X_train_clean, y_train, selected_indices, rf_params=None):
    X_sel = X_train_clean[:, selected_indices]
    base_params = dict(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        min_samples_split=5,
        max_features=0.5,
        max_samples=0.7,
        random_state=42,
        n_jobs=-1,
    )

    if rf_params:
        base_params.update(rf_params)
    #Random Forest
    rf = RandomForestRegressor(**base_params)
    print("Dang train Random Forest...")
    rf.fit(X_sel, y_train)
    print("Da train xong voi Random Forest!")
    #Linear Regression
    print("Dang train Linear Regression...")
    scaler = StandardScaler()
    X_sel_scaled = scaler.fit_transform(X_sel)
    lr = LinearRegression()
    lr.fit(X_sel_scaled, y_train)
    print("Da train xong voi Linear Regression!")
    #XGBoost
    print("Dang train XGBoost...")
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(X_sel, y_train)
    print("Da train xong voi XGBoost!")

    return rf, lr, scaler, xgb


def tinh_metrics(y_true, y_pred, n_features):
    """
    Tính đầy đủ các metrics hồi quy:
      - RMSE   : căn bậc hai sai số bình phương trung bình
      - MAE    : sai số tuyệt đối trung bình
      - R²     : hệ số xác định
      - Adj R² : R² hiệu chỉnh theo số lượng features
      - MAPE   : sai số phần trăm tuyệt đối trung bình (bỏ qua y_true = 0)
      - Tol 10%: % dự đoán nằm trong ngưỡng ±10% so với giá trị thực
      - Tol 20%: % dự đoán nằm trong ngưỡng ±20% so với giá trị thực
    """
    n = len(y_true)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    # adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - n_features - 1, 1)

    # mask = y_true != 0
    # mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    # pct_err  = np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))
    # tol10    = np.mean(pct_err <= 0.10) * 100
    # tol20    = np.mean(pct_err <= 0.20) * 100

    return rmse, mae, r2


# danh gia + tim mo hinh tot hon
def evaluate_model(rf, lr, scaler, xgb, X_test_clean, y_test, selected_indices):
    X_sel      = X_test_clean[:, selected_indices]
    n_features = X_sel.shape[1]

    rf_metrics  = tinh_metrics(y_test, rf.predict(X_sel), n_features)
    lr_metrics  = tinh_metrics(y_test, lr.predict(scaler.transform(X_sel)), n_features)
    xgb_metrics = tinh_metrics(y_test, xgb.predict(X_sel), n_features)

    labels = ["RMSE", "MAE", "R²"]
    descriptions = [
        "Căn bậc hai sai số bình phương TB — nhỏ hơn tốt hơn",
        "Sai số tuyệt đối trung bình — nhỏ hơn tốt hơn",
        "Hệ số xác định (0–1) — lớn hơn tốt hơn"
    ]

    sep = "=" * 90
    print(sep)
    print("SO SÁNH KẾT QUẢ: Random Forest vs Linear Regression vs XGBoost")
    print(sep)
    print(f"{'Metric':<14} {'Random Forest':>14} {'Linear Regression':>18} {'XGBoost':>10}   {'Ý nghĩa'}")
    print("-" * 90)
    for lbl, desc, rf_val, lr_val, xgb_val in zip(labels, descriptions, rf_metrics, lr_metrics, xgb_metrics):
        print(f"{lbl:<14} {rf_val:>14.4f} {lr_val:>18.4f} {xgb_val:>10.4f}   {desc}")
    print(sep)

    best_r2  = max(rf_metrics[2], lr_metrics[2], xgb_metrics[2])
    winner   = {rf_metrics[2]: "Random Forest", lr_metrics[2]: "Linear Regression", xgb_metrics[2]: "XGBoost"}[best_r2]
    print(f"→ Model tốt hơn (theo R²): {winner}\n")
    return rf_metrics[0]  # trả về RMSE của RF


def kiem_tra_overfit(X_train, y_train, selected_indices, k=5):
    """
    Cross-Validation (Overfitting Check) — KFold k lần trên tập train.
    So sánh R² trung bình CV với R² train-full để phát hiện overfitting.

    Tiêu chí:
      - |R²_train - R²_cv_mean| < 0.05  → không overfit
      - 0.05 ≤ khoảng cách < 0.10       → overfit nhẹ
      - khoảng cách ≥ 0.10              → overfit rõ
    """
    X_sel = X_train[:, selected_indices]
    kf    = KFold(n_splits=k, shuffle=True, random_state=42)

    rf_cv_r2, rf_cv_rmse   = [], []
    lr_cv_r2, lr_cv_rmse   = [], []
    xgb_cv_r2, xgb_cv_rmse = [], []

    for tr_idx, val_idx in kf.split(X_sel):
        Xtr, Xval = X_sel[tr_idx], X_sel[val_idx]
        ytr, yval = y_train[tr_idx], y_train[val_idx]

        rf_cv = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        rf_cv.fit(Xtr, ytr)
        rf_pred = rf_cv.predict(Xval)
        rf_cv_r2.append(r2_score(yval, rf_pred))
        rf_cv_rmse.append(np.sqrt(mean_squared_error(yval, rf_pred)))

        sc    = StandardScaler()
        lr_cv = LinearRegression()
        lr_cv.fit(sc.fit_transform(Xtr), ytr)
        lr_pred = lr_cv.predict(sc.transform(Xval))
        lr_cv_r2.append(r2_score(yval, lr_pred))
        lr_cv_rmse.append(np.sqrt(mean_squared_error(yval, lr_pred)))

        xgb_cv = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=42, n_jobs=-1, verbosity=0)
        xgb_cv.fit(Xtr, ytr)
        xgb_pred = xgb_cv.predict(Xval)
        xgb_cv_r2.append(r2_score(yval, xgb_pred))
        xgb_cv_rmse.append(np.sqrt(mean_squared_error(yval, xgb_pred)))

    # R² khi fit toàn bộ train (để so sánh với CV)
    rf_full = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf_full.fit(X_sel, y_train)
    rf_train_r2 = r2_score(y_train, rf_full.predict(X_sel))

    sc_full = StandardScaler()
    lr_full = LinearRegression()
    lr_full.fit(sc_full.fit_transform(X_sel), y_train)
    lr_train_r2 = r2_score(y_train, lr_full.predict(sc_full.transform(X_sel)))

    xgb_full = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=42, n_jobs=-1, verbosity=0)
    xgb_full.fit(X_sel, y_train)
    xgb_train_r2 = r2_score(y_train, xgb_full.predict(X_sel))

    rf_cv_mean  = np.mean(rf_cv_r2)
    lr_cv_mean  = np.mean(lr_cv_r2)
    xgb_cv_mean = np.mean(xgb_cv_r2)
    rf_gap      = rf_train_r2  - rf_cv_mean
    lr_gap      = lr_train_r2  - lr_cv_mean
    xgb_gap     = xgb_train_r2 - xgb_cv_mean

    def ket_luan(gap):
        if gap < 0.05:  return "Khong overfit  ✓"
        if gap < 0.10:  return "Overfit nhe    ⚠"
        return             "Overfit ro     ✗"

    sep = "=" * 90
    print(sep)
    print(f"CROSS-VALIDATION ({k}-Fold) — KIEM TRA OVERFITTING")
    print(sep)
    print(f"{'':32} {'Random Forest':>13} {'Linear Regression':>17} {'XGBoost':>10}")
    print("-" * 90)
    print(f"{'R2 trung binh CV':<32} {rf_cv_mean:>13.4f} {lr_cv_mean:>17.4f} {xgb_cv_mean:>10.4f}")
    print(f"{'R2 do lech chuan CV':<32} {np.std(rf_cv_r2):>13.4f} {np.std(lr_cv_r2):>17.4f} {np.std(xgb_cv_r2):>10.4f}")
    print(f"{'RMSE trung binh CV':<32} {np.mean(rf_cv_rmse):>13.4f} {np.mean(lr_cv_rmse):>17.4f} {np.mean(xgb_cv_rmse):>10.4f}")
    print(f"{'R2 train (full fit)':<32} {rf_train_r2:>13.4f} {lr_train_r2:>17.4f} {xgb_train_r2:>10.4f}")
    print(f"{'Khoang cach (train - CV)':<32} {rf_gap:>13.4f} {lr_gap:>17.4f} {xgb_gap:>10.4f}")
    print("-" * 90)
    print(f"{'Ket luan':<32} {ket_luan(rf_gap):>13} {ket_luan(lr_gap):>17} {ket_luan(xgb_gap):>10}")
    print(sep + "\n")


#luu mo hinh
def save(df, preprocessor, rf, lr, scaler, xgb, candidate_features, selected_features, selected_indices):
    df.to_csv("../Quan_sat/test.csv", index=False)
    print(f"Da luu du lieu vao test.csv")
    with open("../Quan_sat/selected_features.txt","w",encoding="utf-8") as f:
        for feat in selected_features:
            f.write(feat+"\n")
    print("Da luu danh sach feature: selected_features.txt")

    model = {
        "preprocessor": preprocessor,
        "model_rf": rf,
        "model_lr": lr,
        "scaler_lr": scaler,
        "model_xgb": xgb,
        "candidate_features": candidate_features,
        "selected_indices": selected_indices,
        "selected_features": selected_features,
    }
    joblib.dump(model,"../Models/model2.pkl")
    print("Da luu model: model2.pkl")