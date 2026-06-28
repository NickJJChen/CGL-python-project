import os
import glob
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import Parallel, delayed

# 1. 設定路徑與目標
base_path = r"C:\Users\User\Desktop\python test\05 3CGL\data"
title_path = r"C:\Users\User\Desktop\python test\05 3CGL\title.txt" 
targets = ['Bending Force']
cat_features = ['Coater Number', 'Anncode', 'Material'] 

# 讀取欄位名稱
columns_names = pd.read_csv(title_path, sep='\t').columns.tolist()

def load_all_data(path, col_names):
    all_files = glob.glob(os.path.join(path, '**', '*.csv'), recursive=True)
    print(f"找到 {len(all_files)} 個檔案，開始讀取...")
    
    # 使用多核心並行處理
    df_list = Parallel(n_jobs=-1)(
        delayed(pd.read_csv)(f, header=None, names=col_names) for f in all_files
    )
    
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
start_time = time.time()
full_df = load_all_data(base_path, columns_names)

full_df = full_df.drop(columns=['time', 'Length'], errors='ignore')
full_df = full_df.iloc[:, :-1]

# --- 新增：全域數據品質清洗 ---
# 這些邏輯適用於所有 Material_Group，確保物理意義不合理的資料被先行剔除
print(f"清洗前總筆數: {len(full_df)}")

# 剔除 Rolling Force 非法值
full_df = full_df[full_df['Rolling Force'] > 0]

# 剔除 Leveling Memory 異常負值
full_df = full_df[full_df['Leveling Memory'] >= 0]

# 剔除 Line Speed 過低的啟動數據 (這通常是資料起始階段的零值或無效值)
full_df = full_df[full_df['Line Speed'] > 20]

print(f"清洗後總筆數: {len(full_df)}")
# ------------------------------

# 使用 pd.cut 將數值轉為 20 個區間的類別標籤 (0-19)
# 這能有效保留大小關係，同時將離散規格轉為模型好學的整數
full_df['Width_Bin'] = pd.cut(full_df['Width'], bins=20, labels=False)
full_df['Thickness_Bin'] = pd.cut(full_df['Thickness'], bins=20, labels=False)

# 將原始的 Width/Thickness 移除，避免重複特徵干擾
full_df = full_df.drop(columns=['Width', 'Thickness'])

# 數值轉型與清理
num_cols = [c for c in full_df.columns if c not in cat_features and c not in targets and c != 'W/R DIAMETER']
for col in num_cols:
    full_df[col] = pd.to_numeric(full_df[col], errors='coerce').astype(np.float32)

for c in cat_features:
    if c in full_df.columns:
        full_df[c] = full_df[c].astype(str)

# 建立分群標記 (使用向量化條件判斷，資料清理速度提升)
# full_df['Material_Group'] = full_df.apply(get_group_label, axis=1)

full_df['Material_Group'] = 'Other_Group'
full_df.loc[full_df['Material'].str.startswith('CF', na=False), 'Material_Group'] = 'CF_Group'

# 針對 CQ 進行邏輯篩選
cq_mask = full_df['Material'].str.startswith('CQ', na=False)
full_df.loc[cq_mask & (full_df['W/R DIAMETER'] < 500), 'Material_Group'] = 'CQ_Small_Group'
full_df.loc[cq_mask & (full_df['W/R DIAMETER'] >= 500), 'Material_Group'] = 'CQ_Large_Group'

# 進行 dummies 轉換
full_df = pd.get_dummies(full_df, columns=[c for c in cat_features if c in full_df.columns])
full_df = full_df.dropna(subset=[t for t in targets if t in full_df.columns])

end_time = time.time()
print(f"資料清理耗時: {end_time - start_time:.2f} 秒")

print(f"資料清理完畢，最終分析維度: {full_df.shape}")

target_name = targets[0]
# 更新群組清單
groups = ['CQ_Small_Group', 'CQ_Large_Group', 'CF_Group', 'Other_Group']

# print("\n=== 開始針對 Width 的全域分佈進行統計 ===")

# # 1. 檢查 Width 的數值分佈情形
# print("\n[Width 全域統計描述]")
# print(full_df['Width'].describe())

# # 2. 視覺化：全域分佈圖
# plt.figure(figsize=(10, 5))

# # 使用直方圖觀察全域分佈
# sns.histplot(full_df['Width'], kde=True, bins=50, color='salmon')
# plt.title('Global Distribution of Width')
# plt.xlabel('Width')
# plt.ylabel('Count')
# plt.grid(True, alpha=0.3)
# plt.show()

# # 3. 檢查 Width 與目標值的關係 (全域散佈圖)
# plt.figure(figsize=(10, 5))
# sns.scatterplot(data=full_df, x='Width', y=target_name, alpha=0.1, s=10)
# plt.title(f'Width vs {target_name} (Global Data)')
# plt.xlabel('Width')
# plt.ylabel(target_name)
# plt.grid(True, alpha=0.3)
# plt.show()

# # 4. 檢查是否存在極端寬度值 (異常板寬)
# width_mean = full_df['Width'].mean()
# width_std = full_df['Width'].std()
# width_outliers = full_df[(full_df['Width'] > width_mean + 3 * width_std) | 
#                          (full_df['Width'] < width_mean - 3 * width_std)]
# print(f"\n[診斷] 偵測到 {len(width_outliers)} 筆 Width 極端值 (超過 3 個標準差)")

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
    
    
    y_pred_main = model.predict(X_test)
    if group == 'CQ_Large_Group':
        print("偵測到 CQ_Large_Group，啟動殘差修正模型...")
        # 1. 計算訓練集的殘差
        train_pred = model.predict(X_train)
        residuals_train = y_train - train_pred
        
        # 2. 訓練殘差模型 (學習主模型沒學到的誤差)
        residual_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, n_jobs=-1, random_state=42)
        residual_model.fit(X_train, residuals_train)
        
        # 3. 修正測試集的預測值
        y_pred_res = residual_model.predict(X_test)
        y_pred = y_pred_main + y_pred_res
        # 建立一個包含 X_test 資訊與誤差的 DataFrame
        error_analysis = X_test.copy()
        error_analysis['Actual'] = y_test
        error_analysis['Predicted'] = y_pred
        error_analysis['Residual'] = np.abs(y_test - y_pred) # 使用絕對誤差
    
    #     # 篩選誤差最大的前 10% 資料
    #     top_10_error = error_analysis.sort_values(by='Residual', ascending=False).head(int(len(error_analysis)*0.1))
    
    #     print(f"\n=== {group} 誤差分析：誤差最大的樣本特徵摘要 ===")
    #     # 這裡顯示這些高誤差樣本在各個數值欄位的分佈情況
    #     print(top_10_error.describe())
    else:
        # 其他群組保持原樣
        y_pred = y_pred_main
    # --------------------------

    print(f"測試集準確率 (R2 Score): {score:.4f}")
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

    # 取得重要性數據
    importance = model.feature_importances_
    # 這裡的 feature_names 需要處理一下，因為 get_dummies 後欄位會變多
    feature_names = X.columns
    feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)
    
    # 視覺化重要性
    plt.figure(figsize=(10, 6))
    plt.barh(feature_imp_df['Feature'].head(15), feature_imp_df['Importance'].head(15))
    plt.gca().invert_yaxis()
    plt.title(f"Feature Importance: {group}")
    plt.show()