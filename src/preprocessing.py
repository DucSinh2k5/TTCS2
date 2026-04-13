import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
def chuan_bi_feature(df):
    feature_khong_chon = []
    features = ["Km_moi_nam","log_Quang_duong_da_di(km)","Muc_tieu_hao(km/l)",
                "Dung_tich(cc)","Cong_suat_toi_da",
                "So_cho_ngoi","Tuoi_xe", "Dia_diem",
                "Quyen_so_huu","Hop_so", "Loai_nhien_lieu","Chay_nhieu", 
                "Top_xe", "Hang_xe"
                ]
    for col in df.columns:
        if col not in features:
            feature_khong_chon.append(col)
    return features

def tien_xu_ly(num,cat):
    transformers = []

    if num:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])
        transformers.append(("num", num_pipeline, num))

    if cat:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        transformers.append(("cat", cat_pipeline, cat))

    return ColumnTransformer(transformers=transformers)

def chon_feature(X_train_clean, y_train, feature_name,imporance_threshold = 0.90):
    rf = RandomForestRegressor(n_estimators=200,random_state=42,n_jobs=-1)
    rf.fit(X_train_clean,y_train)

    feat_imp = pd.Series(rf.feature_importances_,index=feature_name).sort_values(ascending=False)
    
    cumsum = feat_imp.cumsum() / feat_imp.sum()
    final_features = feat_imp[cumsum <= imporance_threshold].index.to_list()

    if not final_features:
        final_features = [feat_imp.index[0]]
    
    selected_indices = [feature_name.index(f) for f in final_features]
    return final_features, selected_indices
