import os
import glob
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 設定路徑與目標
base_path = r"C:\Users\User\Desktop\01 python test\05 3CGL\data"
title_path = r"C:\Users\User\Desktop\01 python test\05 3CGL\title.txt" 
# targets = ['Rolling Force', 'Bending Force', 'Leveling Memory']
targets = ['Slope']
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

def plot_correlation_matrix(df, top_features, target_name):
    """繪製相關性熱力圖"""
    plt.figure(figsize=(10, 8))
    # 選取 Top 10 特徵與目標變數
    cols_to_plot = top_features + [target_name]
    corr_matrix = df[cols_to_plot].corr()
    
    # 繪製熱力圖
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title(f"Correlation Heatmap: {target_name} (Top 10 Features)")
    plt.tight_layout()
    plt.show()

# --- 執行流程 ---

# 1. 載入與清理
full_df = load_all_data(base_path, columns_names)
full_df = full_df.drop(columns=['time', 'Length', 'OP-DR', 'Z profile-1', 'Z profile-2', 'Z profile-3', 'Z profile-4', 'Z profile-5'], errors='ignore')
full_df = full_df.iloc[:, :-1]

# 數值轉型與清理
num_cols = [c for c in full_df.columns if c not in cat_features and c not in targets]
full_df[num_cols] = full_df[num_cols].apply(pd.to_numeric, errors='coerce').astype(np.float32)

for c in cat_features:
    if c in full_df.columns:
        full_df[c] = full_df[c].astype(str)

full_df = full_df.dropna(how='all')
full_df = pd.get_dummies(full_df, columns=[c for c in cat_features if c in full_df.columns])
full_df = full_df.dropna(subset=[t for t in targets if t in full_df.columns])
print(f"資料清理完畢，最終分析維度: {full_df.shape}")

# 2. 建模與分析
all_features = [col for col in full_df.columns if col not in targets]

for target_name in targets:
    if target_name not in full_df.columns: continue
    
    print(f"\n--- 正在分析目標: {target_name} ---")
    X = full_df[all_features]
    y = full_df[target_name]
    
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, n_jobs=-1, random_state=42)
    model.fit(X, y)
        
    # 取得重要性數據
    importance = model.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': all_features, 'Importance': importance})
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)
    
    # 視覺化重要性
    plt.figure(figsize=(10, 6))
    plt.barh(feature_imp_df['Feature'].head(15), feature_imp_df['Importance'].head(15))
    plt.gca().invert_yaxis()
    plt.title(f"XGBoost Feature Importance: {target_name}")
    plt.show()
    
    # 繪製相關性熱力圖 (取 Top 10)
    top_10_features = feature_imp_df['Feature'].head(10).tolist()
    plot_correlation_matrix(full_df, top_10_features, target_name)
    
    print(feature_imp_df.head(10))
