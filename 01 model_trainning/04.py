import os
import glob
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. 設定路徑與目標
base_path = r"C:\Users\User\Desktop\python test\05 3CGL\data"
title_path = r"C:\Users\User\Desktop\python test\05 3CGL\title.txt" 
targets = ['Bending Force']
cat_features = ['Coater Number', 'Anncode', 'Material'] 

# 讀取欄位名稱
columns_names = pd.read_csv(title_path, sep='\t').columns.tolist()

def load_all_data(path, col_names):
    all_files = glob.glob(os.path.join(path, '**', '*.csv'), recursive=True)
    df_list = []
    print(f"找到 {len(all_files)} 個檔案，開始讀取...")
    for filename in all_files:
        try:
            temp_df = pd.read_csv(filename, header=None, names=col_names)
            df_list.append(temp_df)
        except Exception as e:
            print(f"略過損壞檔案 {filename}: {e}")
    if not df_list: return pd.DataFrame()
    print("正在合併所有資料...")
    return pd.concat(df_list, axis=0, ignore_index=True)

# 重新設計分群邏輯
def get_group_label(row):
    material = row['Material']
    diameter = row['W/R DIAMETER']
    
    if material.startswith('CF'):
        return 'CF_Group'
    elif material.startswith('CQ'):
        # 針對 CQ 進行尺寸細分
        return 'CQ_Small_Group' if diameter < 500 else 'CQ_Large_Group'
    else:
        return 'Other_Group'

# --- 執行流程 ---

# 1. 載入與清理
full_df = load_all_data(base_path, columns_names)
full_df = full_df.drop(columns=['time', 'Length'], errors='ignore')
full_df = full_df.iloc[:, :-1]

# 數值轉型與清理
num_cols = [c for c in full_df.columns if c not in cat_features and c not in targets and c != 'W/R DIAMETER']
for col in num_cols:
    full_df[col] = pd.to_numeric(full_df[col], errors='coerce').astype(np.float32)

for c in cat_features:
    if c in full_df.columns:
        full_df[c] = full_df[c].astype(str)

# 建立分群標記 (在這裡將 CQ 拆分)
full_df['Material_Group'] = full_df.apply(get_group_label, axis=1)

# 進行 dummies 轉換
full_df = pd.get_dummies(full_df, columns=[c for c in cat_features if c in full_df.columns])
full_df = full_df.dropna(subset=[t for t in targets if t in full_df.columns])

print(f"資料清理完畢，最終分析維度: {full_df.shape}")

target_name = targets[0]
# 更新群組清單
groups = ['CQ_Small_Group', 'CQ_Large_Group', 'CF_Group', 'Other_Group']

# 2. 建模與分析
for group in groups:
    print(f"\n=== 正在訓練分組模型: {group} ===")
    
    # 篩選資料
    group_df = full_df[full_df['Material_Group'] == group]
    
    # 移除目標、分群標記，以及不需要的數值欄位 (包含原始尺寸)
    drop_cols = targets + ['Material_Group', 'W/R DIAMETER']
    X = group_df.drop(columns=[c for c in drop_cols if c in group_df.columns])
    y = group_df[target_name]
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"測試集準確率 (R2 Score): {score:.4f}")
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"平均絕對誤差 (MAE): {mae:.2f}")
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse) # 額外計算 RMSE，因為 MSE 數值通常很大，RMSE 較容易解讀
    # print(f"平均平方誤差 (MSE): {mse:.2f}")
    print(f"均方根誤差 (RMSE): {rmse:.2f}")
    
    # 繪圖：還原為濃淡點圖形 (密度分布風格)
    abs_errors = np.abs(y_test - y_pred)
    scatter = plt.scatter(y_test, y_pred, c=abs_errors, cmap='coolwarm', alpha=0.6)
    plt.colorbar(scatter, label='Absolute Error Magnitude')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Bending Force')
    plt.ylabel('Predicted Bending Force')
    plt.title(f'Actual vs Predicted: {group}')
    plt.grid(True, alpha=0.3)
    plt.show()

    