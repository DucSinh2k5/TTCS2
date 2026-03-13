import os
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
def eda_after(df, target="Gia_theo_lakh", output_dir="../Quan_sat/eda_plots"):
    """
    EDA rút gọn – 5 biểu đồ quan trọng nhất, lưu toàn bộ vào output_dir.
      1. Phân phối target (Histogram+KDE + Boxplot)
      2. Numeric liên tục: Histogram+KDE (trên) + Scatter vs target (dưới)
      3. Features rời rạc & encoded: phân phối + boxplot vs target
      4. Categorical chuỗi: bar count + boxplot vs target
      5. Ma trận tương quan Pearson
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Phân loại cột ────────────────────────────────────────────────────────
    encoded_int_cols = {"Quyen_so_huu", "Hop_so", "Thoi_han_bao_hiem", "Chay_nhieu"}

    binary_labels = {
        "Hop_so":            {0: "Automatic",    1: "Manual"},
        "Thoi_han_bao_hiem": {0: "Khác",         1: "Comprehensive"},
        "Chay_nhieu":        {0: "≤ median km",  1: "> median km"},
    }

    all_num = [c for c in df.columns
               if df[c].dtype.kind in "biufc"
               and c != target
               and c not in encoded_int_cols]

    continuous_cols = [c for c in all_num if df[c].nunique() > 20]
    discrete_cols   = [c for c in all_num if df[c].nunique() <= 20]

    binary_cols  = [c for c in encoded_int_cols
                    if c in df.columns
                    and set(df[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]
    ordinal_cols = [c for c in encoded_int_cols
                    if c in df.columns and c not in binary_cols]

    cat_str_cols = [c for c in df.columns
                    if df[c].dtype.kind not in "biufc"
                    and c not in (target, "Ten_xe")]

    # ─────────────────────────────────────────────────────────
    # 1. PHÂN PHỐI TARGET
    # ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Phân phối biến mục tiêu: {target}", fontsize=14, fontweight="bold")

    sns.histplot(df[target].dropna(), kde=True, bins=40, ax=axes[0], color="steelblue")
    axes[0].set_title(f"Histogram + KDE  (skew = {df[target].skew():.2f})")
    axes[0].set_xlabel(target)
    axes[0].set_ylabel("Tần suất")

    sns.boxplot(y=df[target].dropna(), ax=axes[1], color="coral",
                flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
    axes[1].set_title("Boxplot – kiểm tra outlier")
    axes[1].set_ylabel(target)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_target.png", dpi=100)
    plt.show()

    # ─────────────────────────────────────────────────────────
    # 2. NUMERIC LIÊN TỤC: Histogram+KDE (trên) + Scatter vs Target (dưới)
    # ─────────────────────────────────────────────────────────
    if continuous_cols:
        nc = min(3, len(continuous_cols))
        nr = (len(continuous_cols) + nc - 1) // nc

        fig, axes = plt.subplots(nr * 2, nc, figsize=(6 * nc, 5 * nr))
        axes = np.array(axes).reshape(nr * 2, nc)
        fig.suptitle("Numeric Liên Tục: Phân Phối & Tương Quan với Target",
                     fontsize=14, fontweight="bold")

        for i, col in enumerate(continuous_cols):
            row_top = (i // nc) * 2
            col_idx = i % nc
            data    = df[col].dropna()

            # Hàng trên: Histogram + KDE
            sns.histplot(data, kde=data.std() > 0, bins=30,
                         ax=axes[row_top][col_idx], color="steelblue")
            axes[row_top][col_idx].set_title(f"{col}  (skew={data.skew():.2f})")
            axes[row_top][col_idx].set_ylabel("Tần suất")

            # Hàng dưới: Scatter + đường hồi quy vs target
            r = df[[col, target]].corr().iloc[0, 1]
            sns.regplot(data=df, x=col, y=target, ax=axes[row_top + 1][col_idx],
                        scatter_kws={"alpha": 0.2, "s": 8, "color": "steelblue"},
                        line_kws={"color": "red", "linewidth": 1.5})
            axes[row_top + 1][col_idx].set_title(f"{col} vs {target}  (r={r:.3f})")

        # Ẩn ô thừa
        for i in range(len(continuous_cols), nr * nc):
            row_top = (i // nc) * 2
            col_idx = i % nc
            axes[row_top][col_idx].set_visible(False)
            axes[row_top + 1][col_idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/02_numeric_continuous.png", dpi=100)
        plt.show()

    # ─────────────────────────────────────────────────────────
    # 3. FEATURES RỜI RẠC & ENCODED vs TARGET
    # ─────────────────────────────────────────────────────────
    feat_group = (
        [("discrete", c) for c in discrete_cols]
        + [("binary",   c) for c in binary_cols]
        + [("ordinal",  c) for c in ordinal_cols]
    )

    if feat_group:
        n_f  = len(feat_group)
        fig, axes = plt.subplots(2, n_f, figsize=(6 * n_f, 10))
        if n_f == 1:
            axes = np.array(axes).reshape(2, 1)
        fig.suptitle("Features Rời Rạc & Encoded: Phân Phối & Ảnh Hưởng lên Target",
                     fontsize=14, fontweight="bold")

        for i, (kind, col) in enumerate(feat_group):
            if kind == "binary":
                lbl = binary_labels.get(col, {0: "0", 1: "1"})
                df_b = df[[col, target]].copy()
                df_b["nhãn"] = df_b[col].map(lbl).fillna(df_b[col].astype(str))
                order_lbl = [lbl[k] for k in sorted(lbl.keys())
                             if lbl[k] in df_b["nhãn"].values]

                counts_b = df_b["nhãn"].value_counts()
                sns.barplot(x=order_lbl,
                            y=[counts_b.get(l, 0) for l in order_lbl],
                            ax=axes[0][i], palette="pastel")
                for bar in axes[0][i].patches:
                    h = bar.get_height()
                    axes[0][i].text(bar.get_x() + bar.get_width() / 2, h + 2,
                                    f"{int(h)}", ha="center", va="bottom", fontsize=8)
                axes[0][i].set_title(f"Phân phối: {col}")
                axes[0][i].set_ylabel("Số lượng")

                sns.boxplot(data=df_b, x="nhãn", y=target,
                            order=order_lbl, ax=axes[1][i], palette="pastel")
                axes[1][i].set_title(f"{col} vs {target}")

            elif kind == "ordinal":
                order_o  = sorted(df[col].dropna().unique())
                counts_o = df[col].value_counts().sort_index()
                sns.barplot(x=[str(k) for k in counts_o.index],
                            y=counts_o.values, ax=axes[0][i], palette="Blues")
                axes[0][i].set_title(f"Phân phối: {col}")
                axes[0][i].set_ylabel("Số lượng")

                sns.boxplot(data=df, x=col, y=target,
                            order=order_o, ax=axes[1][i], hue=col, palette="Set3", legend=False)
                axes[1][i].set_title(f"{col} vs {target}")

            else:  # discrete numeric
                order_d = sorted(df[col].dropna().unique())
                step    = 2 if len(order_d) > 8 else 1
                sns.histplot(df[col].dropna(), discrete=True,
                             ax=axes[0][i], color="mediumpurple", stat="count")
                axes[0][i].set_title(f"Phân phối: {col}")
                axes[0][i].set_ylabel("Số lượng")
                axes[0][i].set_xticks(order_d[::step])
                axes[0][i].tick_params(axis="x", rotation=45)

                sns.boxplot(data=df, x=col, y=target,
                            order=order_d, ax=axes[1][i], hue=col, palette="Set2", legend=False)
                axes[1][i].set_title(f"{col} vs {target}")
                visible_pos    = list(range(0, len(order_d), step))
                visible_labels = [str(int(v) if v == int(v) else v)
                                  for v in order_d[::step]]
                axes[1][i].set_xticks(visible_pos)
                axes[1][i].set_xticklabels(visible_labels, rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/03_features_vs_target.png", dpi=100)
        plt.show()

    # ─────────────────────────────────────────────────────────
    # 4. STRING CATEGORICAL vs TARGET
    # ─────────────────────────────────────────────────────────
    if cat_str_cols:
        for col in cat_str_cols:
            fname = re.sub(r'[()/ ]', '_', col)

            if col == "Top_xe" and "Other" in df[col].values:
                df_named = df[df[col] != "Other"].copy()

                order_by_median = (df_named.groupby(col)[target]
                                   .median().sort_values(ascending=False).index.tolist())
                code_map          = {name: f"Xe {j+1}" for j, name in enumerate(order_by_median)}
                order_code_median = [code_map[n] for n in order_by_median]

                counts_named = df_named[col].value_counts().reindex(order_by_median)
                counts_code  = counts_named.rename(index=code_map)

                legend_df = pd.DataFrame({
                    "Mã":            [code_map[n]  for n in order_by_median],
                    "Tên xe":        order_by_median,
                    "n":             [df_named[col].value_counts()[n] for n in order_by_median],
                    "Median giá (L)":[round(df_named[df_named[col]==n][target].median(), 2)
                                      for n in order_by_median],
                })

                df_named["_code"] = df_named[col].map(code_map)

                fig, axes_c = plt.subplots(1, 2, figsize=(18, 7))
                fig.suptitle("Top_xe – Phân phối & Giá (sắp theo median giảm dần)",
                             fontsize=13, fontweight="bold")

                sns.barplot(x=counts_code.values, y=counts_code.index,
                            order=order_code_median,
                            ax=axes_c[0], palette="Blues_r", orient="h")
                for bar in axes_c[0].patches:
                    w = bar.get_width()
                    axes_c[0].text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                                   f"{int(w)}", va="center", ha="left", fontsize=9)
                axes_c[0].set_title("Số lượng (sắp theo median giá giảm dần)")
                axes_c[0].set_xlabel("Số lượng bản ghi")
                axes_c[0].set_ylabel("Mã xe")

                sns.boxplot(data=df_named, x="_code", y=target,
                            order=order_code_median, ax=axes_c[1], hue="_code", palette="Set2", legend=False,
                            flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
                axes_c[1].set_title("Phân phối giá theo mã xe")
                axes_c[1].set_xlabel("Mã xe")
                axes_c[1].tick_params(axis="x", rotation=45)

                # Bảng chú giải nhỏ nhúng trong figure
                legend_str = "\n".join(
                    f"{row['Mã']:5s}: {row['Tên xe']}" for _, row in legend_df.iterrows()
                )
                axes_c[1].text(1.02, 0.98, legend_str,
                               transform=axes_c[1].transAxes,
                               fontsize=7, va="top", ha="left",
                               bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow"))

                plt.tight_layout()
                plt.savefig(f"{output_dir}/04_cat_{fname}.png", dpi=100, bbox_inches="tight")
                plt.show()

            else:
                top_vals = df[col].value_counts().head(15).index
                df_c     = df[df[col].isin(top_vals)]
                order_c  = (df_c.groupby(col)[target]
                            .median().sort_values(ascending=False).index.tolist())

                fig, axes_c = plt.subplots(1, 2, figsize=(16, 6))
                fig.suptitle(f"Categorical: {col}", fontsize=14, fontweight="bold")

                counts_c = df[col].value_counts().head(15)
                sns.barplot(x=counts_c.values, y=counts_c.index,
                            ax=axes_c[0], palette="Blues_r")
                axes_c[0].set_title("Top 15 phổ biến nhất")
                axes_c[0].set_xlabel("Số lượng bản ghi")

                sns.boxplot(data=df_c, x=col, y=target,
                            order=order_c, ax=axes_c[1], hue=col, palette="Set2", legend=False)
                axes_c[1].set_title(f"{col} vs {target} (sắp xếp theo median)")
                axes_c[1].tick_params(axis="x", rotation=45)

                plt.tight_layout()
                plt.savefig(f"{output_dir}/04_cat_{fname}.png", dpi=100)
                plt.show()

    # ─────────────────────────────────────────────────────────
    # 5. MA TRẬN TƯƠNG QUAN PEARSON
    # ─────────────────────────────────────────────────────────
    all_corr_cols = (continuous_cols + discrete_cols
                     + [c for c in encoded_int_cols if c in df.columns]
                     + [target])
    corr_matrix = df[all_corr_cols].corr()
    mask_tri    = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr_matrix, mask=mask_tri,
                annot=True, fmt=".2f",
                cmap="coolwarm", center=0, vmin=-1, vmax=1,
                linewidths=0.4, ax=ax, annot_kws={"size": 7})
    ax.set_title("Ma trận Tương quan Pearson\n(đỏ = dương, xanh = âm)",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/05_correlation.png", dpi=100)
    plt.show()