import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

class ModelPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("鋼捲彎輥力預測模型測試系統")
        self.root.geometry("650x550")

        # --- 上半部：設定區 ---
        self.frame_top = tk.LabelFrame(root, text="設定區", padx=10, pady=10)
        self.frame_top.pack(fill="x", padx=10, pady=5)

        self.test_path = self.create_path_input(self.frame_top, "測試資料檔 (.csv):", is_file=True)
        self.model_folder = self.create_path_input(self.frame_top, "模型資料夾:", is_file=False)
        self.title_path = self.create_path_input(self.frame_top, "欄位名稱檔 (.txt):", is_file=True)

        self.btn_run = tk.Button(self.frame_top, text="執行數據預測", command=self.run_prediction, 
                                 bg="#2196F3", fg="white", font=("Arial", 10, "bold"), height=2)
        self.btn_run.pack(fill="x", pady=10)

        # --- 下半部：結果區 ---
        self.frame_bottom = tk.LabelFrame(root, text="執行結果", padx=10, pady=10)
        self.frame_bottom.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.result_label = tk.Label(self.frame_bottom, text="預測結果將顯示於此", justify="left", fg="gray")
        self.result_label.pack(anchor="nw")

    def create_path_input(self, parent, label_text, is_file=True):
        frame = tk.Frame(parent)
        frame.pack(fill="x", pady=3)
        tk.Label(frame, text=label_text, width=16, anchor="w").pack(side="left")
        entry = tk.Entry(frame)
        entry.pack(side="left", fill="x", expand=True, padx=5)
        btn = tk.Button(frame, text="瀏覽", command=lambda: self.browse_path(entry, is_file))
        btn.pack(side="right")
        return entry

    def browse_path(self, entry, is_file):
        path = filedialog.askopenfilename() if is_file else filedialog.askdirectory()
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)

    def preprocess_for_prediction(self, df, title_path):
        # 讀取欄位名稱並應用
        columns_names = pd.read_csv(title_path, sep='\t').columns.tolist()
        df.columns = columns_names
        
        # 清理欄位
        df = df.drop(columns=['time', 'Length'], errors='ignore')
        df = df.iloc[:, :-1]
        
        # 數值轉換
        num_cols = [c for c in df.columns if c not in ['Coater Number', 'Anncode', 'Material', 'test', 'Bending Force', 'W/R DIAMETER']]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
        
        # 分群邏輯
        df['Material_Group'] = 'Other_Group'
        df.loc[df['Material'].str.startswith('CF', na=False), 'Material_Group'] = 'CF_Group'
        cq_mask = df['Material'].str.startswith('CQ', na=False)
        df.loc[cq_mask & (df['W/R DIAMETER'] < 500), 'Material_Group'] = 'CQ_Small_Group'
        df.loc[cq_mask & (df['W/R DIAMETER'] >= 500), 'Material_Group'] = 'CQ_Large_Group'
        
        # 編碼 (注意：生產環境建議載入訓練時的 LabelEncoder)
        for c in ['Coater Number', 'Anncode', 'Material', 'test']:
            if c in df.columns:
                le = LabelEncoder()
                df[c] = le.fit_transform(df[c].astype(str))
        return df

    def run_prediction(self):
        try:
            m_folder = self.model_folder.get()
            t_path = self.test_path.get()
            title_p = self.title_path.get()

            if not all([m_folder, t_path, title_p]):
                messagebox.showwarning("提醒", "請完整選擇路徑")
                return

            df_test = pd.read_csv(t_path, header=None)
            df_processed = self.preprocess_for_prediction(df_test, title_p)
            
            results = []
            # 依據分群進行迴圈預測
            groups = df_processed['Material_Group'].unique()
            
            for group in groups:
                model_filename = f"model_{group}_step_1.json"
                model_path = os.path.join(m_folder, model_filename)
                
                if not os.path.exists(model_path):
                    print(f"警告: 找不到模型 {model_filename}，跳過此群組。")
                    continue
                
                model = xgb.XGBRegressor()
                model.load_model(model_path)
                
                df_subset = df_processed[df_processed['Material_Group'] == group]
                X_input = df_subset.drop(columns=['Bending Force', 'Material_Group', 'W/R DIAMETER'], errors='ignore')
                
                preds = model.predict(X_input)
                df_subset = df_subset.copy()
                df_subset['Predicted_Force'] = preds
                results.append(df_subset[['Material_Group', 'Predicted_Force']])

            # 顯示結果
            final_df = pd.concat(results)
            self.result_label.config(text=f"預測完成！共預測 {len(final_df)} 筆資料。\n\n結果預覽：\n{final_df.head(10).to_string()}")
            messagebox.showinfo("成功", "所有群組已自動匹配並完成預測。")

        except Exception as e:
            messagebox.showerror("執行錯誤", f"發生錯誤:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelPredictionApp(root)
    root.mainloop()