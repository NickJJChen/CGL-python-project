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
DEFAULT_DATA_PATH = r"C:\Users\User\Desktop\01 python test\05 3CGL\data_new"
DEFAULT_TITLE_PATH = r"C:\Users\User\Desktop\01 python test\05 3CGL\title01.txt"
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

TARGET = "Slope"
CAT_FEATURES = ["Coater Number", "AnnCode", "Material", "test"]
DROP_COLUMNS = [
    "time",
    "Length",
    "OP-DR",
    "Z profile-1",
    "Z profile-2",
    "Z profile-3",
    "Z profile-4",
    "Z profile-5",
]


def _read_single_csv(filename: str, col_names: list[str]) -> pd.DataFrame | None:
    """Read, drop unused columns, and downcast numeric columns per file to save RAM."""
    try:
        df = pd.read_csv(filename, header=None, names=col_names)
        if df.empty:
            return None
        df = df.drop(columns=DROP_COLUMNS, errors="ignore")
        # 移除最後一欄 (predict_Bending)
        df = df.iloc[:, :-1]
        df = df.dropna(how="all")
        if df.empty:
            return None

        for col in df.columns:
            if col in CAT_FEATURES:
                df[col] = df[col].astype(str).str.strip()
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

        if TARGET in df.columns:
            df = df.dropna(subset=[TARGET])
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


def get_monthly_folders(path: str, selected_months: list[str] | None = None) -> dict[str, list[str]]:
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
    plt.barh(top_15["Feature"], top_15["Importance"], color="#2196F3")
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(f"XGBoost Feature Importance Top 15: {target_name}")
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
    plt.title(f"Correlation Heatmap: {target_name} (Top 10 Features)")
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{target_name}_top10_correlation_heatmap.png")
    plt.savefig(out_path, dpi=150)
    if show_plot:
        plt.show()
    plt.close()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3CGL XGBoost 按月分批增量特徵重要性分析 (Slope)"
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="輸入 CSV 資料根目錄")
    parser.add_argument("--title-path", default=DEFAULT_TITLE_PATH, help="欄位名稱檔路徑")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="輸出圖表與報表資料夾")
    parser.add_argument(
        "--months", nargs="*", default=None, help="指定要載入的月份 (例如 202601 202602)，預設全數月份"
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

    month_map = get_monthly_folders(args.data_path, args.months)
    if not month_map:
        raise RuntimeError(f"在 {args.data_path} 找不到符合 YYYYMMDD 格式的月份資料夾。")

    months_list = list(month_map.keys())
    print(f"偵測到 {len(months_list)} 個月份：{months_list[0]} 至 {months_list[-1]}（共 {len(months_list)} 批）")

    cat_maps: dict[str, dict[str, int]] = {}
    model: xgb.XGBRegressor | None = None
    feature_cols: list[str] | None = None
    corr_samples: list[pd.DataFrame] = []
    total_rows = 0

    for month_idx, ym in enumerate(months_list, start=1):
        date_dirs = month_map[ym]
        batch_df, file_count = load_month_data(date_dirs, columns_names)
        if batch_df.empty:
            print(f"[Month {ym} ({month_idx}/{len(months_list)})] 無有效資料，略過。")
            continue

        batch_df = encode_categoricals(batch_df, cat_maps)

        if TARGET not in batch_df.columns:
            print(f"找不到目標欄位: {TARGET}")
            return

        if feature_cols is None:
            feature_cols = [c for c in batch_df.columns if c != TARGET]

        X_batch = batch_df[feature_cols]
        y_batch = batch_df[TARGET]
        n_rows = len(batch_df)
        total_rows += n_rows

        print(
            f"[Month {ym} ({month_idx}/{len(months_list)})] "
            f"日期資料夾: {len(date_dirs)} 個 | 檔案數: {file_count} | "
            f"本月筆數: {n_rows:,} | 累計筆數: {total_rows:,} -> 進行增量訓練..."
        )

        prev_booster = model.get_booster() if model is not None else None
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

        # 保留少量跨月抽樣供相關性熱力圖使用，避免佔用過多記憶體
        sample_n = min(args.corr_sample_per_month, n_rows)
        corr_samples.append(batch_df.sample(n=sample_n, random_state=42))

        del batch_df, X_batch, y_batch
        gc.collect()

    if model is None or feature_cols is None:
        raise RuntimeError(f"在 {args.data_path} 未讀取到任何有效資料。")

    print(f"\n全數按月增量訓練完成！總訓練筆數: {total_rows:,}，總月份數: {len(months_list)}")

    feature_imp_df = pd.DataFrame(
        {"Feature": feature_cols, "Importance": model.feature_importances_}
    ).sort_values(by="Importance", ascending=False, ignore_index=True)

    csv_out = os.path.join(args.output_dir, f"{TARGET}_feature_importance.csv")
    feature_imp_df.to_csv(csv_out, index=False, encoding="utf-8-sig")

    imp_png = plot_feature_importance(feature_imp_df, TARGET, args.output_dir, args.show)

    sample_df = pd.concat(corr_samples, axis=0, ignore_index=True)
    del corr_samples
    gc.collect()

    top_10_features = feature_imp_df["Feature"].head(10).tolist()
    corr_png = plot_correlation_matrix(
        sample_df, top_10_features, TARGET, args.output_dir, args.show
    )

    print("\nTop 15 特徵重要性排名:")
    print(feature_imp_df.head(15).to_string(index=False))
    print(f"\n已儲存結果:\n- {csv_out}\n- {imp_png}\n- {corr_png}")


if __name__ == "__main__":
    main()
