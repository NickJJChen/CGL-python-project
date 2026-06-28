import pandas as pd
import xgboost as xgb
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

# === 1. 設定與路徑 ===
# 請確認模型與資料路徑
model_path = r"C:\Users\User\Desktop\01 python test\05 3CGL\model_test\model_CF_Group_step_1.json"
test_file_path = r"C:\Users\User\Desktop\01 python test\05 3CGL\model_test\Coil Number-5325486.csv"
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
model = xgb.XGBRegressor()
model.load_model(model_path)

# 讀取測試資料
df_test = pd.read_csv(test_file_path, header=None, names=columns_names)

# 處理資料
df_processed = preprocess_for_prediction(df_test)

# 篩選對應群組 (範例為 CF_Group)
target_group = 'CF_Group'
df_target = df_processed[df_processed['Material_Group'] == target_group]

if not df_target.empty:
    # 移除預測目標欄位與不需要的欄位
    X_input = df_target.drop(columns=targets + ['Material_Group', 'W/R DIAMETER'], errors='ignore')
    
    # 預測
    predictions = model.predict(X_input)
    
    # 將結果加入 DataFrame 並輸出
    df_target = df_target.copy()
    df_target['Predicted_Bending_Force'] = predictions
#     print(df_target[['Material', 'Predicted_Bending_Force']])
    print(df_target[['Predicted_Bending_Force']])
    
    # 儲存結果
#     output_path = test_file_path.replace('.csv', '_result.csv')
#     df_target.to_csv(output_path, index=False, encoding='utf-8-sig')
#     print(f"\n預測結果已儲存至: {output_path}")
else:
    print(f"資料中沒有符合 {target_group} 的樣本。")