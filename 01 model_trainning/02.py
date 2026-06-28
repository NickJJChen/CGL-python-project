import os
import glob
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1. 設定路徑與目標
base_path = r"C:\Users\User\Desktop\01 python test\05 3CGL\data01"
title_path = r"C:\Users\User\Desktop\01 python test\05 3CGL\title.txt" 
# targets = ['Rolling Force', 'Bending Force', 'Leveling Memory']
targets = ['Bending Force']
cat_features = ['Coater Number', 'Anncode', 'Material', 'W/R DIAMETER']

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

def categorize_material(material_name):
    if material_name.startswith('CQ'):
        return 'CQ_Group'
    elif material_name.startswith('CF'):
        return 'CF_Group'
    else:
        return 'Other_Group'

# --- 執行流程 ---

# 1. 載入與清理
full_df = load_all_data(base_path, columns_names)
full_df = full_df.drop(columns=['time', 'Length'], errors='ignore')
full_df = full_df.iloc[:, :-1]

# print("\n=== 各材質資料量統計 ===")
# material_counts = full_df['Material'].value_counts()
# print(material_counts)
# print(f"共有 {len(material_counts)} 種不同材料。")
# print("========================\n")

# 數值轉型與清理
num_cols = [c for c in full_df.columns if c not in cat_features and c not in targets]
# full_df[num_cols] = full_df[num_cols].apply(pd.to_numeric, errors='coerce').astype(np.float32)

for col in num_cols:
    # 逐一處理每個欄位，處理完一個就釋放該欄位的轉換記憶體
    full_df[col] = pd.to_numeric(full_df[col], errors='coerce').astype(np.float32)

for c in cat_features:
    if c in full_df.columns:
        full_df[c] = full_df[c].astype(str)

full_df = full_df.dropna(how='all')
# print("=== 工輥尺寸分佈統計 ===")
# print(f"工輥尺寸分類數: {full_df['W/R DIAMETER'].nunique()}")
# print(full_df['W/R DIAMETER'].value_counts().sort_index())

full_df['Material_Group'] = full_df['Material'].apply(categorize_material)
full_df = pd.get_dummies(full_df, columns=[c for c in cat_features if c in full_df.columns])
full_df = full_df.dropna(subset=[t for t in targets if t in full_df.columns])
print(f"資料清理完畢，最終分析維度: {full_df.shape}")

target_name = targets[0]
groups = ['CQ_Group', 'CF_Group', 'Other_Group']

# 2. 建模與分析
# all_features = [col for col in full_df.columns if col not in targets]
# 
# for target_name in targets:
#     if target_name not in full_df.columns: continue
#     
#     print(f"\n--- 正在分析目標: {target_name} ---")
#     X = full_df[all_features]
#     y = full_df[target_name]

for group in groups:
    print(f"\n=== 正在訓練分組模型: {group} ===")
    
    # 篩選資料
    group_df = full_df[full_df['Material_Group'] == group]
    X = group_df.drop(columns=targets + ['Material_Group'])
    y = group_df[target_name]
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"測試集準確率 (R2 Score): {score:.4f}")
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"平均絕對誤差 (MAE): {mae:.2f}")
    
    abs_errors = np.abs(y_test - y_pred)
    scatter = plt.scatter(y_test, y_pred, c=abs_errors, cmap='coolwarm', alpha=0.6)
    plt.colorbar(scatter, label='Absolute Error Magnitude')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')

#     plt.xlabel('Actual Bending Force')
#     plt.ylabel('Predicted Bending Force')
#     plt.title('Actual vs Predicted (Color indicates Error Magnitude)')
    plt.legend(loc="upper right")
#     plt.grid(True, alpha=0.3)
#     plt.show()
    
#     plt.figure(figsize=(10, 8))
#     plt.scatter(y_test, y_pred, alpha=0.3)
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
#     plt.xlabel('Actual Bending Force')
#     plt.ylabel('Predicted Bending Force')
#     plt.title('Actual vs Predicted')
    plt.show()
    
#     residuals = y_test - y_pred
# 
#     # 繪製殘差分佈圖
#     plt.figure(figsize=(8, 6))
#     plt.scatter(y_pred, residuals, alpha=0.3, color='blue')
#     plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # 基準線
#     plt.xlabel(f'Predicted {target_name}')
#     plt.ylabel('Residuals (Actual - Predicted)')
#     plt.title(f'Residual Plot: {target_name}')
#     plt.grid(True, alpha=0.3)
#     plt.show()
        
    # 取得重要性數據
#     importance = model.feature_importances_
#     feature_imp_df = pd.DataFrame({'Feature': all_features, 'Importance': importance})
#     feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)
#     
#     # 視覺化重要性
#     plt.figure(figsize=(10, 6))
#     plt.barh(feature_imp_df['Feature'].head(15), feature_imp_df['Importance'].head(15))
#     plt.gca().invert_yaxis()
#     plt.title(f"XGBoost Feature Importance: {target_name}")
#     plt.show()
#     
#     # 繪製相關性熱力圖 (取 Top 10)
#     top_10_features = feature_imp_df['Feature'].head(10).tolist()
#     plot_correlation_matrix(full_df, top_10_features, target_name)
#     
#     print(feature_imp_df.head(10))
    
#     print("\n" + "="*30)
#     print("開始測試：針對主要材質進行分群訓練")
#     print("="*30)
# 
#     # 請根據您實際的欄位名稱調整（因為 pd.get_dummies 後，欄位變成了 'Material_CHDJ' 等格式）
#     target_materials = ['Material_CHDJ', 'Material_CHBJ', 'Material_CSGJ'] 
# 
#     for mat_col in target_materials:
#         # 檢查欄位是否存在
#         if mat_col not in full_df.columns:
#             print(f"找不到欄位 {mat_col}，跳過...")
#             continue
#     
#         # 篩選特定材質的資料 (該材質為 1 的資料)
#         mat_df = full_df[full_df[mat_col] == 1]
#     
#         if len(mat_df) < 500:
#             print(f"材質 {mat_col} 資料量不足 (僅 {len(mat_df)} 筆)，跳過訓練。")
#             continue
#         
#         print(f"\n--- 正在為 {mat_col} 獨立訓練模型 ---")
#     
#         # 切分資料
#         X_mat = mat_df[all_features]
#         y_mat = mat_df[target_name]
#     
#         X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
#         # 注意：這裡直接沿用前面的 train_test_split 邏輯可能需要微調，
#         # 建議重新對 mat_df 進行 split，確保訓練集與測試集分開
#         X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_mat, y_mat, test_size=0.2, random_state=42)
#     
#         # 訓練模型
#         model_mat = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, n_jobs=-1, random_state=42)
#         model_mat.fit(X_train_m, y_train_m)
#     
#         # 評估準確度
#         y_pred_m = model_mat.predict(X_test_m)
#         mae_m = np.abs(y_test_m - y_pred_m).mean()
#         print(f"【結果】{mat_col} 獨立模型的 MAE: {mae_m:.2f}")