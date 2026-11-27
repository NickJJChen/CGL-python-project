import argparse
import gc
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = r"C:\Users\User\Desktop\01 python test\01 1CGL\data"
DEFAULT_TITLE_PATH = os.path.join(BASE_DIR, "title.txt")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

ALL_TARGETS = ["Slope", "Crown"]
CAT_FEATURES = ["Coater Number", "Material"]
DROP_COLUMNS = [
    "Time",
    "Length",
    "OP-DR",
    "Z profile-1",
    "Z profile-2",
    "Z profile-3",
    "Z profile-4",
    "Z profile-5",
]


def _read_single_csv(filename: str, col_names: list[str]) -> pd.DataFrame | None:
    """Read, validate column alignment, drop unused columns, and downcast numeric columns."""
    try:
        df = pd.read_csv(filename, header=None)
        if df.empty:
            return None
        if df.shape[1] != len(col_names):
            raise ValueError(
                f"欄位數不符：CSV 有 {df.shape[1]} 欄，但 title 檔有 {len(col_names)} 欄"
            )
        df.columns = col_names
        df = df.drop(columns=DROP_COLUMNS, errors="ignore")
        df = df.dropna(how="all")
        if df.empty:
            return None

        for col in df.columns:
            if col in CAT_FEATURES:
                df[col] = df[col].astype(str).str.strip()
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

        return df if not df.empty else None
    except Exception as exc:
        print(f"略過損壞或無法讀取檔案 {filename}: {exc}")
        return None


def encode_categoricals(
    df: pd.DataFrame, cat_maps: dict[str, dict[str, int]]
) -> pd.DataFrame:
    """Apply consistent incremental integer encoding across all batches."""
    for col in CAT_FEATURES:
        if col not in df.columns:
            continue
        mapping = cat_maps.setdefault(col, {})
        values = df[col].tolist()
        for val in set(values):
            if val not in mapping:
                mapping[val] = len(mapping)
        df[col] = df[col].map(mapping).astype(np.int32)
    return df


def get_monthly_folders(
    path: str, selected_months: list[str] | None = None
) -> dict[str, list[str]]:
    """Group YYYYMMDD subfolders by YYYYMM month prefix in chronological order."""
    month_map: dict[str, list[str]] = {}
    for entry in sorted(os.listdir(path)):
        full_dir = os.path.join(path, entry)
        if os.path.isdir(full_dir) and len(entry) >= 6 and entry[:6].isdigit():
            ym = entry[:6]
            if selected_months and ym not in selected_months:
                continue
            month_map.setdefault(ym, []).append(full_dir)
    return month_map


def load_month_data(
    date_dirs: list[str],
    col_names: list[str],
    max_workers: int = 4,
) -> tuple[pd.DataFrame, int]:
    """Load all CSV files belonging to a single month's YYYYMMDD folders."""
    month_files: list[str] = []
    for d in date_dirs:
        month_files.extend(sorted(glob.glob(os.path.join(d, "*.csv"))))

    if not month_files:
        return pd.DataFrame(), 0

    df_list: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_read_single_csv, f, col_names) for f in month_files]
        for future in as_completed(futures):
            df = future.result()
            if df is not None and not df.empty:
                df_list.append(df)

    if not df_list:
        return pd.DataFrame(), len(month_files)

    month_df = pd.concat(df_list, axis=0, ignore_index=True)
    del df_list
    gc.collect()
    return month_df, len(month_files)


def plot_feature_importance(
    feature_imp_df: pd.DataFrame, target_name: str, output_dir: str, show_plot: bool
) -> str:
    """Plot and save Top 15 XGBoost feature importance horizontal bar chart."""
    import matplotlib.pyplot as plt

    top_15 = feature_imp_df.head(15)
    plt.figure(figsize=(10, 6))
    color = "#2196F3" if target_name == "Slope" else "#00897B"
    plt.barh(top_15["Feature"], top_15["Importance"], color=color)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(f"1CGL XGBoost Feature Importance Top 15: {target_name}")
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{target_name}_top15_feature_importance.png")
    plt.savefig(out_path, dpi=150)
    if show_plot:
        plt.show()
    plt.close()
    return out_path


def plot_correlation_matrix(
    sample_df: pd.DataFrame,
    top_features: list[str],
    target_name: str,
    output_dir: str,
    show_plot: bool,
) -> str:
    """Plot and save correlation heatmap for Top 10 features and target."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(11, 9))
    cols_to_plot = top_features + [target_name]
    corr_matrix = sample_df[cols_to_plot].corr()

    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title(f"1CGL Correlation Heatmap: {target_name} (Top 10 Features)")
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{target_name}_top10_correlation_heatmap.png")
    plt.savefig(out_path, dpi=150)
    if show_plot:
        plt.show()
    plt.close()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="1CGL XGBoost 按月分批增量雙目標特徵重要性分析 (Slope / Crown)"
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="輸入 CSV 資料根目錄")
    parser.add_argument("--title-path", default=DEFAULT_TITLE_PATH, help="欄位名稱檔路徑")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="輸出圖表與報表資料夾")
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=ALL_TARGETS,
        default=ALL_TARGETS,
        help="要分析的目標欄位（預設同時執行 Slope 與 Crown，並自動互相排除）",
    )
    parser.add_argument(
        "--months", nargs="*", default=None, help="指定要載入的月份 (例如 202609 202610)，預設全數月份"
    )
    parser.add_argument(
        "--trees-per-month", type=int, default=20, help="每月增量訓練的樹數量 (預設 20)"
    )
    parser.add_argument(
        "--corr-sample-per-month",
        type=int,
        default=30_000,
        help="每月保留用於相關性熱力圖的抽樣筆數",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="XGBoost 運算裝置 (cpu 或 cuda，預設 cpu)",
    )
    parser.add_argument("--show", action="store_true", help="執行時顯示互動式圖表視窗")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    os.makedirs(args.output_dir, exist_ok=True)
    columns_names = pd.read_csv(args.title_path, sep="\t").columns.tolist()
    print(f"欄位定義檔共 {len(columns_names)} 欄：{args.title_path}")
    print(f"分析目標：{args.targets}（已自動將 {ALL_TARGETS} 排除於輸入特徵之外）")

    month_map = get_monthly_folders(args.data_path, args.months)
    if not month_map:
        raise RuntimeError(f"在 {args.data_path} 找不到符合 YYYYMMDD 格式的月份資料夾。")

    months_list = list(month_map.keys())
    print(f"偵測到 {len(months_list)} 個月份：{months_list[0]} 至 {months_list[-1]}（共 {len(months_list)} 批）")

    cat_maps: dict[str, dict[str, int]] = {}
    models: dict[str, xgb.XGBRegressor | None] = {t: None for t in args.targets}
    target_rows: dict[str, int] = {t: 0 for t in args.targets}
    feature_cols: list[str] | None = None
    corr_samples: list[pd.DataFrame] = []

    for month_idx, ym in enumerate(months_list, start=1):
        date_dirs = month_map[ym]
        batch_df, file_count = load_month_data(date_dirs, columns_names)
        if batch_df.empty:
            print(f"[Month {ym} ({month_idx}/{len(months_list)})] 無有效資料，略過。")
            continue

        batch_df = encode_categoricals(batch_df, cat_maps)

        if feature_cols is None:
            feature_cols = [c for c in batch_df.columns if c not in ALL_TARGETS]

        print(
            f"[Month {ym} ({month_idx}/{len(months_list)})] "
            f"日期資料夾: {len(date_dirs)} 個 | 檔案數: {file_count} | "
            f"原始筆數: {len(batch_df):,}"
        )

        for target in args.targets:
            if target not in batch_df.columns:
                raise RuntimeError(f"找不到目標欄位: {target}")
            valid_df = batch_df.dropna(subset=[target])
            if valid_df.empty:
                print(f"  -> [{target}] 本月無有效資料，略過。")
                continue

            X_batch = valid_df[feature_cols]
            y_batch = valid_df[target]
            target_rows[target] += len(valid_df)
            print(
                f"  -> [{target}] 本月有效筆數: {len(valid_df):,} | "
                f"累計: {target_rows[target]:,} -> 進行增量訓練..."
            )

            prev_model = models[target]
            prev_booster = prev_model.get_booster() if prev_model is not None else None
            model = xgb.XGBRegressor(
                n_estimators=args.trees_per_month,
                learning_rate=0.1,
                max_depth=6,
                tree_method="hist",
                device=args.device,
                n_jobs=-1,
                random_state=42,
            )
            model.fit(X_batch, y_batch, xgb_model=prev_booster)
            models[target] = model

        sample_n = min(args.corr_sample_per_month, len(batch_df))
        corr_samples.append(batch_df.sample(n=sample_n, random_state=42))

        del batch_df
        gc.collect()

    if feature_cols is None or all(m is None for m in models.values()):
        raise RuntimeError(f"在 {args.data_path} 未讀取到任何有效資料。")

    sample_df = pd.concat(corr_samples, axis=0, ignore_index=True)
    del corr_samples
    gc.collect()

    imp_tables: dict[str, pd.DataFrame] = {}
    for target in args.targets:
        model = models[target]
        if model is None:
            continue

        print(f"\n=== [{target}] 訓練完成（總筆數: {target_rows[target]:,}）===")
        feature_imp_df = pd.DataFrame(
            {"Feature": feature_cols, "Importance": model.feature_importances_}
        ).sort_values(by="Importance", ascending=False, ignore_index=True)
        imp_tables[target] = feature_imp_df

        csv_out = os.path.join(args.output_dir, f"{target}_feature_importance.csv")
        feature_imp_df.to_csv(csv_out, index=False, encoding="utf-8-sig")

        imp_png = plot_feature_importance(
            feature_imp_df, target, args.output_dir, args.show
        )
        top_10_features = feature_imp_df["Feature"].head(10).tolist()
        corr_png = plot_correlation_matrix(
            sample_df, top_10_features, target, args.output_dir, args.show
        )

        print(feature_imp_df.head(15).to_string(index=False))
        print(f"已儲存 [{target}] 結果:\n- {csv_out}\n- {imp_png}\n- {corr_png}")

    if "Slope" in imp_tables and "Crown" in imp_tables:
        s_df = imp_tables["Slope"].copy()
        s_df["Slope_Rank"] = s_df.index + 1
        s_df = s_df.rename(columns={"Importance": "Slope_Importance"})

        c_df = imp_tables["Crown"].copy()
        c_df["Crown_Rank"] = c_df.index + 1
        c_df = c_df.rename(columns={"Importance": "Crown_Importance"})

        merged = pd.merge(s_df, c_df, on="Feature", how="outer")
        merged["Avg_Importance"] = (
            merged["Slope_Importance"] + merged["Crown_Importance"]
        ) / 2.0
        merged = merged.sort_values(
            by="Avg_Importance", ascending=False, ignore_index=True
        )
        cmp_csv = os.path.join(
            args.output_dir, "Slope_vs_Crown_feature_importance.csv"
        )
        merged.to_csv(cmp_csv, index=False, encoding="utf-8-sig")
        print(f"\n=== 雙目標綜合對照 (Top 15) ===")
        print(merged.head(15).to_string(index=False))
        print(f"已儲存雙目標對照表: {cmp_csv}")


if __name__ == "__main__":
    main()
