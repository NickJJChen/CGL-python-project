import pandas as pd
import xgboost as xgb
import os
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder


# === 1. 設定與路徑 ===
# 請確認模型與資料路徑
model_path = r"C:\Users\User\Desktop\01 python test\05 3CGL\model_test\model_CQ_Large_Group_step_1.pkl"
test_file_path = r"C:\Users\User\Desktop\01 python test\05 3CGL\model_test\Coil Number-1401426-1.csv"
title_path = r"C:\Users\User\Desktop\01 python test\05 3CGL\title01.txt"

# 定義與訓練時相同的特徵與類別欄位
cat_features = ['Coater Number', 'Anncode', 'Material', 'test']
targets = ['Bending Force']
columns_names = pd.read_csv(title_path, sep='\t').columns.tolist()

# === 2. 資料清理函數 (與訓練邏輯一致) ===
def preprocess_for_prediction(df):
    # 刪除與訓練時相同的無用欄位
    df = df.drop(columns=['time', 'Length'], errors='ignore')
    df = df.iloc[:, :-1]
    
    # 數值轉型
    num_cols = [c for c in df.columns if c not in cat_features and c not in targets and c != 'W/R DIAMETER']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
    
    # 分群標記邏輯
    df['Material_Group'] = 'Other_Group'
    df.loc[df['Material'].str.startswith('CF', na=False), 'Material_Group'] = 'CF_Group'
    cq_mask = df['Material'].str.startswith('CQ', na=False)
    df.loc[cq_mask & (df['W/R DIAMETER'] < 500), 'Material_Group'] = 'CQ_Small_Group'
    df.loc[cq_mask & (df['W/R DIAMETER'] >= 500), 'Material_Group'] = 'CQ_Large_Group'
    
    # 注意：這裡 LabelEncoder 的處理方式與訓練時必須一致
    # 建議在生產環境中儲存並載入訓練時的 LabelEncoder 物件，若簡單處理則如下：
    for c in cat_features:
        if c in df.columns:
            le = LabelEncoder()
            # 若資料量較小，這裡可能會因為未包含所有類別而編碼不一致
            # 實際生產環境強烈建議使用已儲存的 Encoder
            df[c] = le.fit_transform(df[c].astype(str))
            
    return df

# === 3. 載入模型與執行預測 ===
# 載入模型
# 使用 joblib 載入 .pkl 模型
print(f"正在載入模型: {model_path} ...")
model = joblib.load(model_path)

# 讀取測試資料
df_test = pd.read_csv(test_file_path, header=None, names=columns_names)
df_processed = preprocess_for_prediction(df_test)

# 篩選對應群組
target_group = 'CQ_Large_Group' # 對應您的群組名稱
df_target = df_processed[df_processed['Material_Group'] == target_group]

if not df_target.empty:
    X_input = df_target.drop(columns=targets + ['Material_Group', 'W/R DIAMETER'], errors='ignore')
    
    # === 關鍵修改：針對 Stacking 模型進行兩階段預測 ===
    if isinstance(model, dict):
        # 1. 第一階段：取得 XGB 與 CatBoost 的預測結果
        p_xgb = model['xgb'].predict(X_input).reshape(-1, 1)
        p_cat = model['cat'].predict(X_input).reshape(-1, 1)
        
        # 2. 合併作為 Meta-Model 的輸入
        meta_features = np.hstack([p_xgb, p_cat])
        
        # 3. 第二階段：使用線性回歸模型進行最終預測
        predictions = model['meta'].predict(meta_features)
    else:
        # 如果不是 dict，則為一般 XGBoost 模型，直接預測
        predictions = model.predict(X_input)
    
    # 輸出結果
    df_target = df_target.copy()
    df_target['Predicted_Bending_Force'] = predictions
    print(df_target[['Predicted_Bending_Force']])
    
else:
    print(f"資料中沒有符合 {target_group} 的樣本。")