import argparse
import gc
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = r"C:\Users\User\Desktop\01 python test\01 1CGL\data"
DEFAULT_TITLE_PATH = os.path.join(BASE_DIR, "title.txt")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_02")

TARGET = "SPM BENDER FORCE"
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

GROUPS = [
    "All_Single_Group",
    "CQ_Thin_Group",
    "CQ_Thick_Coater01",
    "CQ_Thick_Coater2",
    "CF_Group",
    "Other_Group",
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

        if TARGET in df.columns:
            df = df.dropna(subset=[TARGET])
        return df if not df.empty else None
    except Exception as exc:
        print(f"略過損壞或無法讀取檔案 {filename}: {exc}")
        return None


def assign_groups(df: pd.DataFrame) -> pd.Series:
    """Assign 1CGL production groups based on Material, Order Thickness, and Coater Number."""
    mat = df["Material"].astype(str).str.strip()
    thick = df["Order Thickness"]
    coater = pd.to_numeric(df["Coater Number"], errors="coerce")

    group = pd.Series("Other_Group", index=df.index, dtype="object")
    group.loc[mat.str.startswith("CF", na=False)] = "CF_Group"

    cq_mask = mat.str.startswith("CQ", na=False)
    group.loc[cq_mask & (thick < 600)] = "CQ_Thin_Group"
    group.loc[cq_mask & (thick >= 600) & coater.isin([0, 1])] = "CQ_Thick_Coater01"
    group.loc[cq_mask & (thick >= 600) & (coater == 2)] = "CQ_Thick_Coater2"
    return group


def encode_categoricals(
    df: pd.DataFrame, cat_maps: dict[str, dict[str, int]]
) -> pd.DataFrame:
    """Apply consistent integer encoding across the full dataset."""
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
    max_workers: int = 8,
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


def plot_residuals_grid(
    eval_results: dict[str, dict], output_dir: str, show_plot: bool
) -> str:
    """Plot a 2x3 residual distribution grid for all 6 models."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, grp in enumerate(GROUPS):
        ax = axes[idx]
        res = eval_results.get(grp)
        if res is None:
            ax.set_title(f"{grp} (No Data)")
            ax.axis("off")
            continue

        residuals = res["residuals"]
        q_low, q_high = np.percentile(residuals, [0.5, 99.5])
        clipped = residuals[(residuals >= q_low) & (residuals <= q_high)]

        color = "#455A64" if grp == "All_Single_Group" else "#1E88E5"
        ax.hist(clipped, bins=60, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(0.0, color="#D32F2F", linestyle="--", linewidth=1.3)
        ax.set_title(
            f"{grp}\nR²={res['R2']:.4f} | MAE={res['MAE']:.1f} | RMSE={res['RMSE']:.1f}",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.set_xlabel("Residual (Actual - Predicted)")
        ax.set_ylabel("Count")
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle(
        f"1CGL {TARGET} Residual Distributions (6 Models, Full-Data Converged)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    out_path = os.path.join(output_dir, "SPM_BENDER_FORCE_residuals_grid.png")
    plt.savefig(out_path, dpi=150)
    if show_plot:
        plt.show()
    plt.close()
    return out_path


def plot_actual_vs_pred_grid(
    eval_results: dict[str, dict], output_dir: str, show_plot: bool
) -> str:
    """Plot a 2x3 Actual vs Predicted scatter grid for all 6 models."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    rng = np.random.default_rng(42)

    for idx, grp in enumerate(GROUPS):
        ax = axes[idx]
        res = eval_results.get(grp)
        if res is None:
            ax.set_title(f"{grp} (No Data)")
            ax.axis("off")
            continue

        y_true = res["y_test"]
        y_pred = res["y_pred"]
        if len(y_true) > 20_000:
            sel = rng.choice(len(y_true), size=20_000, replace=False)
            y_t_plot, y_p_plot = y_true[sel], y_pred[sel]
        else:
            y_t_plot, y_p_plot = y_true, y_pred

        err = np.abs(y_t_plot - y_p_plot)
        sc = ax.scatter(y_t_plot, y_p_plot, c=err, cmap="viridis", s=4, alpha=0.35)
        lims = [
            min(float(y_t_plot.min()), float(y_p_plot.min())),
            max(float(y_t_plot.max()), float(y_p_plot.max())),
        ]
        ax.plot(lims, lims, "r--", linewidth=1.2)
        ax.set_title(
            f"{grp}\nR²={res['R2']:.4f} | RMSE={res['RMSE']:.1f}",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.set_xlabel("Actual SPM BENDER FORCE")
        ax.set_ylabel("Predicted SPM BENDER FORCE")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"1CGL {TARGET} Actual vs Predicted (6 Models, Full-Data Converged)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    out_path = os.path.join(output_dir, "SPM_BENDER_FORCE_actual_vs_pred_grid.png")
    plt.savefig(out_path, dpi=150)
    if show_plot:
        plt.show()
    plt.close()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="1CGL SPM BENDER FORCE 6 模組全量合併訓練與評估 (R2 / MAE / RMSE / 殘差圖)"
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="輸入 CSV 資料根目錄")
    parser.add_argument("--title-path", default=DEFAULT_TITLE_PATH, help="欄位名稱檔路徑")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="輸出模型、指標與圖表資料夾")
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=GROUPS,
        default=GROUPS,
        help="指定要訓練的模組名稱（預設全數 6 個模組）",
    )
    parser.add_argument(
        "--shuffle-batches",
        type=int,
        default=1,
        help="將全量訓練集打散後分成 N 批進行增量訓練（預設 1 表示單批全量訓練）",
    )
    parser.add_argument(
        "--months", nargs="*", default=None, help="指定要載入的月份 (例如 202609 202610)，預設全數月份"
    )
    parser.add_argument(
        "--n-estimators", type=int, default=2000, help="XGBoost 每批最大樹數量 (預設 2000)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.05, help="XGBoost 學習率 (預設 0.05)"
    )
    parser.add_argument(
        "--max-depth", type=int, default=8, help="XGBoost 最大樹深 (預設 8)"
    )
    parser.add_argument(
        "--early-stopping-rounds", type=int, default=50, help="Early stopping 回合數 (預設 50)"
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
    model_dir = os.path.join(args.output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    columns_names = pd.read_csv(args.title_path, sep="\t").columns.tolist()
    print(f"欄位定義檔共 {len(columns_names)} 欄：{args.title_path}")
    print(
        f"預測目標：{TARGET} | 模式：全量合併 + 80/20 切分 | "
        f"n_estimators={args.n_estimators}, lr={args.learning_rate}, max_depth={args.max_depth}, "
        f"early_stopping={args.early_stopping_rounds}"
    )

    month_map = get_monthly_folders(args.data_path, args.months)
    if not month_map:
        raise RuntimeError(f"在 {args.data_path} 找不到符合 YYYYMMDD 格式的月份資料夾。")

    months_list = list(month_map.keys())
    print(f"偵測到 {len(months_list)} 個月份：{months_list[0]} 至 {months_list[-1]}")

    month_dfs: list[pd.DataFrame] = []
    for month_idx, ym in enumerate(months_list, start=1):
        date_dirs = month_map[ym]
        batch_df, file_count = load_month_data(date_dirs, columns_names)
        if batch_df.empty:
            continue
        print(f"  載入 [Month {ym} ({month_idx}/{len(months_list)})] 檔案: {file_count} | 筆數: {len(batch_df):,}")
        month_dfs.append(batch_df)

    full_df = pd.concat(month_dfs, axis=0, ignore_index=True)
    del month_dfs
    gc.collect()

    print(f"\n全量資料合併完成：共 {len(full_df):,} 筆")
    full_df["_Group"] = assign_groups(full_df)
    cat_maps: dict[str, dict[str, int]] = {}
    full_df = encode_categoricals(full_df, cat_maps)

    feature_cols = [c for c in full_df.columns if c not in (TARGET, "_Group")]

    eval_results: dict[str, dict] = {}
    metrics_rows: list[dict] = []

    print(f"\n=== 開始訓練 {len(args.groups)} 個模組 (shuffle_batches={args.shuffle_batches}) ===")
    for grp in args.groups:
        sub_df = full_df if grp == "All_Single_Group" else full_df[full_df["_Group"] == grp]
        if len(sub_df) < 50:
            print(f"[{grp}] 資料不足 ({len(sub_df)} 筆)，略過。")
            continue

        X = sub_df[feature_cols]
        y = sub_df[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"\n--- 訓練 [{grp}] (Train={len(X_train):,}, Test={len(X_test):,}) ---")
        n_batches = max(1, args.shuffle_batches)
        if n_batches == 1:
            model = xgb.XGBRegressor(
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                max_depth=args.max_depth,
                tree_method="hist",
                device=args.device,
                early_stopping_rounds=args.early_stopping_rounds,
                n_jobs=-1,
                random_state=42,
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=200,
            )
        else:
            perm = np.random.default_rng(42).permutation(len(X_train))
            chunks = np.array_split(perm, n_batches)
            model = None
            for b_idx, idx_chunk in enumerate(chunks, start=1):
                print(f"  -> [Shuffle Batch {b_idx}/{n_batches}] 筆數: {len(idx_chunk):,}")
                prev_booster = model.get_booster() if model is not None else None
                model = xgb.XGBRegressor(
                    n_estimators=args.n_estimators,
                    learning_rate=args.learning_rate,
                    max_depth=args.max_depth,
                    tree_method="hist",
                    device=args.device,
                    early_stopping_rounds=args.early_stopping_rounds,
                    n_jobs=-1,
                    random_state=42 + b_idx,
                )
                model.fit(
                    X_train.iloc[idx_chunk],
                    y_train.iloc[idx_chunk],
                    eval_set=[(X_test, y_test)],
                    verbose=200,
                    xgb_model=prev_booster,
                )

        y_pred = model.predict(X_test)
        y_test_np = y_test.to_numpy()
        r2 = float(r2_score(y_test_np, y_pred))
        mae = float(mean_absolute_error(y_test_np, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test_np, y_pred)))
        residuals = y_test_np - y_pred
        best_iter = getattr(model, "best_iteration", args.n_estimators)

        eval_results[grp] = {
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
            "y_test": y_test_np,
            "y_pred": y_pred,
            "residuals": residuals,
        }

        model_path = os.path.join(model_dir, f"{grp}_spm_bender_force.json")
        model.save_model(model_path)

        metrics_rows.append(
            {
                "Group": grp,
                "Train_Rows": len(X_train),
                "Test_Rows": len(X_test),
                "Best_Iteration": best_iter,
                "R2": round(r2, 6),
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "Model_Path": model_path,
            }
        )
        print(
            f"==> [{grp:18s}] BestIter={best_iter:4d} | R2={r2:.4f} | MAE={mae:.2f} | RMSE={rmse:.2f}"
        )

        del sub_df, X, y, X_train, X_test, y_train, y_test
        gc.collect()

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = os.path.join(args.output_dir, "SPM_BENDER_FORCE_metrics_comparison.csv")
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    res_png = plot_residuals_grid(eval_results, args.output_dir, args.show)
    scatter_png = plot_actual_vs_pred_grid(eval_results, args.output_dir, args.show)

    print("\n=== 6 模組最終對照表 ===")
    print(metrics_df[["Group", "Train_Rows", "Test_Rows", "Best_Iteration", "R2", "MAE", "RMSE"]].to_string(index=False))
    print(f"\n已儲存評估報表與圖表:\n- {metrics_csv}\n- {res_png}\n- {scatter_png}")


if __name__ == "__main__":
    main()
