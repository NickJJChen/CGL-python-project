import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ModelPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("鋼捲彎輥力預測模型測試系統")
        self.root.geometry("650x750")

        # --- 上半部：設定區 ---
        self.frame_top = tk.LabelFrame(root, text="設定區", padx=10, pady=10)
        self.frame_top.pack(fill="x", padx=10, pady=5)

        self.test_path = self.create_path_input(self.frame_top, "測試資料檔 (.csv):", is_file=True)
        self.model_folder = self.create_path_input(self.frame_top, "模型資料夾:", is_file=False)
        self.title_path = self.create_path_input(self.frame_top, "欄位名稱檔 (.txt):", is_file=True)

        # 預設路徑
        self.test_path.insert(0, r"C:\Users\User\Desktop\01 python test\05 3CGL\model_test\Coil Number-5325486.csv")
        self.model_folder.insert(0, r"C:\Users\User\Desktop\01 python test\05 3CGL\model_new")
        self.title_path.insert(0, r"C:\Users\User\Desktop\01 python test\05 3CGL\model_test\title01.txt")

        self.btn_run = tk.Button(self.frame_top, text="執行數據預測並繪圖", command=self.run_prediction, 
                                 bg="#2196F3", fg="white", font=("Arial", 10, "bold"), height=2)
        self.btn_run.pack(fill="x", pady=10)

        # --- 下半部：執行結果與圖表區 ---
        self.frame_bottom = tk.LabelFrame(root, text="執行結果與趨勢圖", padx=10, pady=10)
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
        columns_names = pd.read_csv(title_path, sep='\t').columns.tolist()
        df.columns = columns_names
        
        df = df.drop(columns=['time', 'Length'], errors='ignore')
        df = df.iloc[:, :-1]
        
        num_cols = [c for c in df.columns if c not in ['Coater Number', 'Anncode', 'Material', 'test', 'Bending Force', 'W/R DIAMETER']]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
        
        df['Material_Group'] = 'Other_Group'
        df.loc[df['Material'].str.startswith('CF', na=False), 'Material_Group'] = 'CF_Group'
        cq_mask = df['Material'].str.startswith('CQ', na=False)
        df.loc[cq_mask & (df['W/R DIAMETER'] < 500), 'Material_Group'] = 'CQ_Small_Group'
        df.loc[cq_mask & (df['W/R DIAMETER'] >= 500), 'Material_Group'] = 'CQ_Large_Group'
        
        for c in ['Coater Number', 'Anncode', 'Material', 'test']:
            if c in df.columns:
                le = LabelEncoder()
                df[c] = le.fit_transform(df[c].astype(str))
        return df

    def run_prediction(self):
        try:
            m_folder, t_path, title_p = self.model_folder.get(), self.test_path.get(), self.title_path.get()
            if not all([m_folder, t_path, title_p]):
                messagebox.showwarning("提醒", "請完整選擇路徑")
                return

            df_original = pd.read_csv(t_path, header=None)
            
            df_processed = self.preprocess_for_prediction(df_original.copy(), title_p)
            
            # 設定欄位：假設原始 CSV 第2欄(index 1)為 Length，第56欄(index 55)為舊預測
            df_original.columns = [f"Col_{i}" for i in range(df_original.shape[1])]
            df_original = df_original.rename(columns={"Col_1": "Length", "Col_4": "Actural_Force", "Col_55": "Predicted_Force_Old"})
            
            df_original['Predicted_Force_New'] = np.nan
            is_empty_row = df_original.isnull().all(axis=1)
            
            found_models = []
            missing_models = []
            groups = df_processed['Material_Group'].unique()
            for group in groups:
                model_path = os.path.join(m_folder, f"model_{group}_step_1.json")
                if not os.path.exists(model_path): 
                    if group not in missing_models:
                        missing_models.append(group)
                    continue
                else:
                    found_models.append(group)
                
                model = xgb.XGBRegressor()
                model.load_model(model_path)
                
                mask = (df_processed['Material_Group'] == group)
                final_mask = mask & ~is_empty_row
                
                X_input = df_processed[final_mask].drop(columns=['Bending Force', 'Material_Group', 'W/R DIAMETER'], errors='ignore')
                
                # 修正型別問題：強制轉換數值以避免 object 型態報錯
                X_input = X_input.apply(pd.to_numeric, errors='coerce').fillna(0)
                
                if not X_input.empty:
                    df_original.loc[final_mask, 'Predicted_Force_New'] = model.predict(X_input)

            # 繪圖
            valid_data = df_original.dropna(subset=['Length', 'Predicted_Force_Old', 'Predicted_Force_New'], how='all')
            if valid_data.empty:
                messagebox.showwarning("警告", "資料中未找到有效數值，無法繪圖")
                return
            
            window = 7  # 可根據需求調整為 7
            pad = window // 2
            df_original['Predicted_Force_MA'] = df_original['Predicted_Force_New'].rolling(window=window, center=True).mean()

            plot_data = df_original.dropna(subset=['Length', 'Predicted_Force_Old', 'Predicted_Force_New', 'Actural_Force']).copy()
            if len(plot_data) > (2 * pad):
                plot_data = plot_data.iloc[pad:-pad]

            self.plot_results(
                plot_data['Length'], 
                plot_data['Predicted_Force_Old'], 
                plot_data['Predicted_Force_New'], 
                plot_data['Actural_Force'], 
                plot_data['Predicted_Force_MA']
                )            
            output_path = t_path.replace('.csv', '_result.csv')
            df_original.to_csv(output_path, index=False, header=False, encoding='utf-8-sig')
            # self.result_label.config(text=f"預測完成！圖表已更新。\n儲存至：{os.path.basename(output_path)}")
            msg_found = f"\n對應模型: {', '.join(found_models)}" if found_models else ""
            missing_str = f"\n⚠️ 無對應模型: {', '.join(missing_models)}" if missing_models else ""
            self.result_label.config(text=f"預測完成！圖表已更新。\n儲存至：{os.path.basename(output_path)}{msg_found}{missing_str}")

        except Exception as e:
            messagebox.showerror("執行錯誤", f"發生錯誤:\n{str(e)}")

    def plot_results(self, x, y1, y2, y3, y4):
        for widget in self.frame_bottom.winfo_children():
            if isinstance(widget, tk.Canvas) or str(widget).startswith(".!labelframe.!frame"):
                # 如果是圖表區塊直接銷毀
                widget.destroy()
            elif isinstance(widget, FigureCanvasTkAgg().get_tk_widget().__class__):
                widget.destroy()

        plt.clf()     # 清除當前圖表
        plt.cla()
        plt.close('all')

        fig, ax = plt.subplots(figsize=(6, 3))
        
        ax.plot(x, y1, label='Old Model (0.1T)', color='red', alpha=0.5)
        ax.plot(x, y2, label='New Model (0.1T)', color='blue', linestyle='--')
        ax.plot(x, y3, label='Actual Force', color='green', alpha=0.5)
        ax.plot(x, y4, label='New Model (MA)', color='orange', linewidth=2)
        ax.set_xlabel('Length (M)')
        ax.set_ylabel('Force')
        ax.legend()
        
        canvas = FigureCanvasTkAgg(fig, master=self.frame_bottom)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelPredictionApp(root)
    root.mainloop()