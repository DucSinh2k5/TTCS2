import pandas as pd
import numpy as np
from load_data_and_cleaning import load_data
def eda_before(df, output = "../Quan_sat/eda_before.txt"):
    lines = []
    sep = "=" * 60

    lines+=[sep,"EDA THÔ - TỔNG QUAN DỮ LIỆU GỐC", sep]
    #1 Kich thuoc
    lines.append(f"\nKích thước : {df.shape[0]} dòng * {df.shape[1]} cột")
    #2 Ten cot + Kieu du lieu
    lines += ["\n--- Tên cột & kiểu dữ liệu ---", df.dtypes.to_string()]
    #3 vai dong dau
    lines += ["\n--- 5 dòng đầu ---", df.head().to_string()]
    #4 Hang trung lap
    n_dup = df.duplicated().sum()
    lines.append(f"\nSố hàng trùng lặp : {n_dup} ({n_dup / len(df) * 100:.2f}%)")
    #5 Gia tri thieu
    lines.append("\n--- Giá trị thiếu ---")
    missing = df.isnull().sum()
    miss_df = pd.DataFrame({
        "Missing" : missing,
        "%"       : (missing / len(df) * 100).round(2)
    })
    miss_df = miss_df[miss_df["Missing"] > 0].sort_values("%", ascending=False)
    lines.append(miss_df.to_string() if not miss_df.empty else "  (không có giá trị thiếu)")
    #6 Thong ke mo ta
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        lines += ["\n--- Thống kê mô tả (numeric) ---",
                  df[num_cols].describe().T.round(2).to_string()]
    #7 value_counts
    cat_cols = df.columns.tolist()
    if cat_cols:
        lines.append("\n--- Value counts ---")
        for col in cat_cols:
            vc = df[col].value_counts(dropna = False).head(10)
            lines.append(f"\n  [{col}]  (unique = {df[col].nunique()})")
            for val, cnt in vc.items():
                lines.append(f"    {str(val):<35} {cnt:>6}  ({cnt/len(df)*100:.1f}%)")
    
    lines += ["\n" + sep + "\n"]
    report = "\n".join(lines)
    with open(output, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)

if __name__ == "__main__":
    df = load_data("../Datasets/Used_Car_Dataset.csv")
    df = eda_before(df)
