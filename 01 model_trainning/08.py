import os
import glob
import pandas as pd
import xgboost as xgb
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import Parallel, delayed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# === 1. 設定與工具函數 ===
base_path = r"C:\Users\User\Desktop\python test\05 3CGL\data01"
title_path = r"C:\Users\User\Desktop\python test\05 3CGL\title01.txt"
plot_dir = r"C:\Users\User\Desktop\python test\05 3CGL\plot"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
model_dir = r"C:\Users\User\Desktop\python test\05 3CGL\model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

columns_names = pd.read_csv(title_path, sep='\t').columns.tolist()
targets = ['Bending Force']
cat_features = ['Coater Number', 'Anncode', 'Material', 'test']


def get_data_range(path):
    folders = sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])
    months = sorted(list(set([f[:6] for f in folders])))
    print(f"\n數據範圍為：{months[0]} 至 {months[-1]}")
    return months

def process_data(df):
    # 這裡放入您原有的清洗邏輯 (Drop, Line Speed > 30, Type conversion)
    df = df.drop(columns=['time', 'Length'], errors='ignore')
    df = df.iloc[:, :-1]
    df = df[df['Line Speed'] > 30]
    
    # 數值轉型
    num_cols = [c for c in df.columns if c not in cat_features and c not in targets and c != 'W/R DIAMETER']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
    
    # 分群標記
    df['Material_Group'] = 'Other_Group'
    df.loc[df['Material'].str.startswith('CF', na=False), 'Material_Group'] = 'CF_Group'
    cq_mask = df['Material'].str.startswith('CQ', na=False)
    df.loc[cq_mask & (df['W/R DIAMETER'] < 500), 'Material_Group'] = 'CQ_Small_Group'
    df.loc[cq_mask & (df['W/R DIAMETER'] >= 500), 'Material_Group'] = 'CQ_Large_Group'
    
    # Label Encoding (注意：生產環境建議保存 LabelEncoder 物件)
    for c in cat_features:
        if c in df.columns:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            
    return df.dropna(subset=[t for t in targets if t in df.columns])

# === 2. 核心訓練邏輯 ===
def train_group_model(model, df, group_name, stage_idx, months_batch):
    group_df = df[df['Material_Group'] == group_name]
    
    if group_df.empty:
        print(f"警告: 群組 {group_name} 在此批次無資料。")
        return model

    X = group_df.drop(columns=targets + ['Material_Group', 'W/R DIAMETER'], errors='ignore')
    y = group_df[targets[0]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. 訓練/增量訓練
    if stage_idx == 0:
        model = xgb.XGBRegressor(n_estimators=3000, learning_rate=0.05, max_depth=6, 
                                 tree_method='hist', early_stopping_rounds=50, random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=500)
    else:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=500, xgb_model=model.get_booster())
        
    # 3. 繪圖與評估 (Title 自動帶入群組名稱與月份)
    y_pred = model.predict(X_test)
    batch_label = f"{group_name} | Stage {stage_idx+1}: {months_batch[0]}~{months_batch[-1]}"


    score = model.score(X_test, y_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"階段 {i+1} 結果 - R2: {score:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    # 動態 Title
    batch_label = f"Stage {i+1}: {months_batch[0]}~{months_batch[-1]}"
    
    # 圖表 1: Actual vs Predicted
    plt.figure(figsize=(8, 6))
    abs_errors = np.abs(y_test - y_pred)
    scatter = plt.scatter(y_test, y_pred, c=abs_errors, cmap='coolwarm', alpha=0.5)
    plt.colorbar(scatter, label='Abs Error')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title(f'Actual vs Predicted | {batch_label}')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    save_path1 = os.path.join(plot_dir, f"Actual_vs_Pred_{group_name}_stage{stage_idx+1}.png")
    plt.savefig(save_path1)
    plt.close()
    
    # 圖表 2: Feature Importance
    plt.figure(figsize=(10, 6))
    feature_imp = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    feature_imp = feature_imp.sort_values(by='Importance', ascending=False).head(15)
    plt.barh(feature_imp['Feature'], feature_imp['Importance'])
    plt.gca().invert_yaxis()
    plt.title(f"Feature Importance | {batch_label}")
    save_path2 = os.path.join(plot_dir, f"Importance_{group_name}_stage{stage_idx+1}.png")
    plt.savefig(save_path2)
    plt.close()

    metrics = {
        'Stage': stage_idx + 1,
        'Months': str(months_batch),
        'Group': group_name,
        'R2_Score': score,
        'MAE': mae,
        'RMSE': rmse
    }

    return model, metrics

# === 3. 主流程控制 ===
all_months = get_data_range(base_path)
start_month = input("請輸入啟始月份 (例如 202601): ")
n_months = int(input("請輸入投入數據月數: "))
groups = ['CQ_Small_Group', 'CQ_Large_Group', 'CF_Group', 'Other_Group']

start_idx = all_months.index(start_month)
target_months = all_months[start_idx:]
batches = [target_months[i:i + n_months] for i in range(0, len(target_months), n_months)]
results_list = []

group_models = {g: None for g in groups}

for i, batch in enumerate(batches):
    print(f"\n=== 階段 {i+1}: 處理月份 {batch} ===")
    start_time_data = time.time()
    batch_files = []
    for m in batch:
        batch_files.extend(glob.glob(os.path.join(base_path, f"{m}*", "*.csv")))
        
    print(f"階段 {i+1}：正在載入 {len(batch_files)} 個檔案...")
    df_list = Parallel(n_jobs=-1)(
        delayed(pd.read_csv)(f, header=None, names=columns_names) for f in batch_files
    )
    df = process_data(pd.concat(df_list, axis=0, ignore_index=True))
    print(f"階段 {i+1} 資料準備完畢，分析維度: {df.shape}")
    end_time_data = time.time()
    print(f"資料清理耗時: {end_time_data - start_time_data:.2f} 秒")
    for group in groups:
        print(f"\n--- 正在處理群組: {group} ---")
        start_time_model = time.time()
        result = train_group_model(
            group_models[group], df, group, i, batch
        )
        if result is not None and isinstance(result, tuple):
            model_obj, metrics = result
            group_models[group] = model_obj
            results_list.append(metrics)

            # 確保 model_obj 不為 None 且是一個已經訓練過的模型
            if model_obj is not None:
                save_path = os.path.join(model_dir, f"model_{group}_step_{i}.json")
                # 使用 save_model 前，確保模型已經 fit 過
                model_obj.save_model(save_path)
                print(f"model_{group}_step_{i} 模型已儲存。")
            else:
                print(f"警告: 群組 {group} 未成功訓練模型，跳過儲存。")
        # model_obj, metrics = result
        # group_models[group] = model_obj
        # results_list.append(metrics)

        # save_path = os.path.join(model_dir, f"model_{group}_step_{i}.json")
        # group_models[group].save_model(save_path)
        # print(f"model_{group}_step_{i} 模型已儲存。")
        end_time_model = time.time()
        print(f"訓練模型耗時: {end_time_model - start_time_model:.2f} 秒")

        results_list.append(metrics)

results_df = pd.DataFrame(results_list)
csv_file_path = os.path.join(plot_dir, "training_metrics.csv")
results_df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
print(f"\n所有階段的評估指標已儲存至: {csv_file_path}")    
    