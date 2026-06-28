import pandas as pd
import xgboost as xgb
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

# === 1. 設定與路徑 ===
model_path = r"C:\Users\User\Desktop\01 python test\05 3CGL\model_test\model_CF_Group_step_1.json"
test_file_path = r"C:\Users\User\Desktop\01 python test\05 3CGL\model_test\Coil Number-5325486.csv"
title_path = r"C:\Users\User\Desktop\01 python test\05 3CGL\title01.txt"

# 定義特徵
targets = ['Bending Force']
cat_features = ['Coater Number', 'Anncode', 'Material', 'test']
columns_names = pd.read_csv(title_path, sep='\t').columns.tolist()

# === 2. 資料清理邏輯 (用於製作輸入模型用的 X) ===
def get_features_for_model(df):
    # 建立副本進行清理，避免修改到原始 df
    df_clean = df.drop(columns=['time', 'Length'], errors='ignore').copy()
    df_clean = df_clean.iloc[:, :-1]
    
    # 數值轉型
    num_cols = [c for c in df_clean.columns if c not in cat_features and c != 'W/R DIAMETER']
    for col in num_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').astype(np.float32)
    
    # 分群標記
    df_clean['Material_Group'] = 'Other_Group'
    df_clean.loc[df_clean['Material'].str.startswith('CF', na=False), 'Material_Group'] = 'CF_Group'
    cq_mask = df_clean['Material'].str.startswith('CQ', na=False)
    df_clean.loc[cq_mask & (df_clean['W/R DIAMETER'] < 500), 'Material_Group'] = 'CQ_Small_Group'
    df_clean.loc[cq_mask & (df_clean['W/R DIAMETER'] >= 500), 'Material_Group'] = 'CQ_Large_Group'
    
    # Label Encoding (若實際生產有儲存 encoder，建議改用 load)
    for c in cat_features:
        if c in df_clean.columns:
            le = LabelEncoder()
            df_clean[c] = le.fit_transform(df_clean[c].astype(str))
            
    return df_clean

# === 3. 載入與推論 ===
# 載入模型
model = xgb.XGBRegressor()
model.load_model(model_path)

# 讀取原始數據 (保留原始格式)
df_original = pd.read_csv(test_file_path, header=None, names=columns_names)

# 產生模型可用的特徵資料
df_for_model = get_features_for_model(df_original)

# 設定目標群組
target_group = 'CF_Group'
mask = df_for_model['Material_Group'] == target_group

# 執行預測
# 初始化欄位，預設為空值
df_original['Predicted_Bending_Force'] = np.nan

if mask.any():
    # --- 關鍵修正：必須顯式刪除掉 'Bending Force' ---
    # 確保 input 的欄位與模型訓練時的特徵完全匹配
    cols_to_drop = targets + ['Material_Group', 'W/R DIAMETER']
    
    # 執行預測前，從 df_for_model 中移除這些欄位
    X_input = df_for_model[mask].drop(columns=cols_to_drop, errors='ignore')
    
    # 再次確保欄位順序與模型訓練時的一致 (重要！)
    # XGBoost 內部會對照訓練時的特徵名稱，但明確指定順序更安全
    # 如果您有存下訓練時的 feature_names，建議使用：
    # X_input = X_input[model.feature_names_in_] 
    
    # 預測並回填
    predictions = model.predict(X_input)
    df_original.loc[mask, 'Predicted_Bending_Force'] = predictions

    # 儲存結果
    output_path = test_file_path.replace('.csv', '_result.csv')
    df_original.to_csv(output_path, index=False, header=False, encoding='utf-8-sig')
    print(f"預測完成，檔案已儲存至: {output_path}")
else:
    print(f"資料中找不到 {target_group} 的樣本。")