import pandas as pd
import xgboost as xgb
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

# === 1. 設定與路徑 ===
# 指向存放所有模型檔案的資料夾
model_folder = r"C:\Users\User\Desktop\01 python test\05 3CGL\model_new"
test_file_path = r"C:\Users\User\Desktop\01 python test\05 3CGL\model_test\Coil Number-5325486.csv"
title_path = r"C:\Users\User\Desktop\01 python test\05 3CGL\title01.txt"

# 定義特徵與類別
cat_features = ['Coater Number', 'Anncode', 'Material', 'test']
targets = ['Bending Force']
columns_names = pd.read_csv(title_path, sep='\t').columns.tolist()

# === 2. 資料清理函數 ===
def preprocess_for_prediction(df):
    df = df.drop(columns=['time', 'Length'], errors='ignore')
    df = df.iloc[:, :-1]
    
    num_cols = [c for c in df.columns if c not in cat_features and c not in targets and c != 'W/R DIAMETER']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
    
    # 分群邏輯
    df['Material_Group'] = 'Other_Group'
    df.loc[df['Material'].str.startswith('CF', na=False), 'Material_Group'] = 'CF_Group'
    cq_mask = df['Material'].str.startswith('CQ', na=False)
    df.loc[cq_mask & (df['W/R DIAMETER'] < 500), 'Material_Group'] = 'CQ_Small_Group'
    df.loc[cq_mask & (df['W/R DIAMETER'] >= 500), 'Material_Group'] = 'CQ_Large_Group'
        
    for c in cat_features:
        if c in df.columns:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
    return df

# === 3. 載入模型與自動預測 ===
df_original = pd.read_csv(test_file_path, header=None, names=columns_names)
df_processed = preprocess_for_prediction(df_original)
df_original['Predicted_Bending_Force'] = np.nan # 初始化結果欄位
is_empty_row = df_original.isnull().all(axis=1)

# 取得資料中存在的所有群組
groups = df_processed['Material_Group'].unique()

for group in groups:
    # 動態產生模型檔名
    model_filename = f"model_{group}_step_1.json"
    model_path = os.path.join(model_folder, model_filename)
    
    if not os.path.exists(model_path):
        print(f"警告: 找不到模型檔案 {model_filename}，跳過群組 {group}")
        continue
    
    # 載入該群組對應的模型
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    # 篩選該群組的資料
    mask = (df_processed['Material_Group'] == group)
    final_mask = mask & ~is_empty_row
    X_input = df_processed[final_mask].drop(columns=targets + ['Material_Group', 'W/R DIAMETER'], errors='ignore')
    
    # 執行預測並填入結果
    if not X_input.empty:
        df_original.loc[final_mask, 'Predicted_Bending_Force'] = model.predict(X_input)
        print(df_original[['Predicted_Bending_Force']])
        print(f"群組 {group} 預測完成。")

# === 4. 輸出結果 ===
output_path = test_file_path.replace('.csv', '_result.csv')
df_original.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n全部預測完成，結果已儲存至: {output_path}")