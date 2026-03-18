import joblib
import numpy as np

print("=" * 60)
print("KIỂM TRA MODEL: model2.pkl")
print("=" * 60)

data = joblib.load("../Models/model2.pkl")

# 1. Kiểm tra các keys
print("\n[1] Các thành phần trong model:")
for key, val in data.items():
    print(f"  {key:<22} → {type(val).__name__}")

# 2. Preprocessor
preprocessor = data["preprocessor"]
print(f"\n[2] Preprocessor: {type(preprocessor).__name__}")
for name, trans, cols in preprocessor.transformers_:
    print(f"  Transformer '{name}': {type(trans).__name__} | Columns ({len(cols)}): {cols}")

# 3. Features
candidate_features = data["candidate_features"]
selected_features  = data["selected_features"]
selected_indices   = data["selected_indices"]
print(f"\n[3] Candidate features ({len(candidate_features)}): {candidate_features}")
print(f"\n[4] Selected features  ({len(selected_features)}): {selected_features}")
print(f"    Selected indices   ({len(selected_indices)}): {selected_indices}")

# 4. Random Forest
rf = data["model_rf"]
print(f"\n[5] Random Forest:")
print(f"  n_estimators  = {rf.n_estimators}")
print(f"  n_features_in = {rf.n_features_in_}")
print(f"  max_depth     = {rf.max_depth}")

importances = rf.feature_importances_
print(f"\n  Top-10 feature importances (theo selected_features):")
imp_sorted = sorted(zip(selected_features, importances), key=lambda x: x[1], reverse=True)
for feat, imp in imp_sorted[:10]:
    print(f"    {feat:<35} {imp:.4f}")

# 5. Linear Regression
lr = data["model_lr"]
print(f"\n[6] Linear Regression:")
print(f"  n_features_in = {lr.n_features_in_}")
print(f"  intercept     = {lr.intercept_:.4f}")
coef_sorted = sorted(zip(selected_features, lr.coef_), key=lambda x: abs(x[1]), reverse=True)
print(f"  Top-10 coefficients (tuyệt đối):")
for feat, coef in coef_sorted[:10]:
    print(f"    {feat:<35} {coef:+.4f}")

# 6. Scaler
scaler = data["scaler_lr"]
print(f"\n[7] StandardScaler (dùng cho LR):")
print(f"  mean (5 đầu): {np.round(scaler.mean_[:5], 4)}")
print(f"  std  (5 đầu): {np.round(scaler.scale_[:5], 4)}")
# 7. XGBoost
xgb = data["model_xgb"]
print(f"\n[8] XGBoost:")
print(f"  n_estimators  = {xgb.n_estimators}")
print(f"  n_features_in = {xgb.n_features_in_}")


print("\n" + "=" * 60)
print("KIỂM TRA HOÀN TẤT")
print("=" * 60)

