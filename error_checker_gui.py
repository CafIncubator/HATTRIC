import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import pandas as pd
import cv2
import numpy as np

BASE_DIR = "output"

class OCRCheckerGUI:
    def __init__(self, master):
        self.master = master
        master.title("HATTRIC - OCR Table Validator")

        self.csv_path = ""
        self.image_folder = ""
        self.table = ""
        self.current_csv = None
        self.row_idx = 0
        self.col_idx = 0
        self.outlier_indices = set()
        self.checking_outliers = False

        # Default values for min/max/std
        self.use_min_max = tk.BooleanVar(value=False)
        self.use_std = tk.BooleanVar(value=False)
        self.min_val = tk.StringVar(value="-50")
        self.max_val = tk.StringVar(value="99")
        self.std_thresh = tk.StringVar(value="2")

        self.create_widgets()
        self.master.bind('<Return>', self.handle_enter_key)

    def create_widgets(self):
        # Top input fields and buttons centered
        top_frame = tk.Frame(self.master)
        top_frame.grid(row=0, column=0, columnspan=2, pady=(5, 0))
        top_inner = tk.Frame(top_frame)
        top_inner.pack()

        tk.Button(top_inner, text="Choose CSV", command=self.select_csv_file).grid(row=0, column=0, columnspan=2, pady=5)
        self.csv_label = tk.Label(top_inner, text="No CSV selected")
        self.csv_label.grid(row=1, column=0, columnspan=2)
        tk.Button(top_inner, text="Load CSV", command=self.load_csv).grid(row=2, column=0, columnspan=2, pady=5)
        tk.Button(top_inner, text="Add Decimal Prefix", command=self.add_decimal_prefix).grid(row=3, column=0, columnspan=2, pady=5)

        # Main content with image and data display
        content_frame = tk.Frame(self.master)
        content_frame.grid(row=1, column=0, columnspan=2)

        # Left column: image and controls
        left_column = tk.Frame(content_frame)
        left_column.pack(side="left", padx=(10, 0))

        self.image_panel = tk.Label(left_column)
        self.image_panel.pack()

        control_frame = tk.Frame(left_column)
        control_frame.pack(pady=(15, 0))

        self.current_text = tk.StringVar()
        self.entry = tk.Entry(control_frame, textvariable=self.current_text)
        self.entry.pack(pady=(0, 5))

        action_btns = tk.Frame(control_frame)
        action_btns.pack()
        tk.Button(action_btns, text="Confirm", command=self.confirm_cell, width=10).pack(side="left", padx=5)
        tk.Button(action_btns, text="Empty", command=self.clear_cell, width=10).pack(side="left", padx=5)

        self.ignore_nan_var = tk.BooleanVar()
        tk.Checkbutton(control_frame, text="Ignore 'NaN' Values", variable=self.ignore_nan_var).pack(pady=5)
        tk.Button(control_frame, text="Save Now", command=self.save_csv, width=25).pack(pady=(0, 10))

        # Add min/max/std controls below Save Now
        validation_frame = tk.Frame(control_frame)
        validation_frame.pack(pady=(5, 0), anchor="w")

        tk.Checkbutton(validation_frame, text="Enable Min/Max Check", variable=self.use_min_max).grid(row=0, column=0, sticky="w")
        tk.Label(validation_frame, text="Min:").grid(row=0, column=1)
        tk.Entry(validation_frame, textvariable=self.min_val, width=6).grid(row=0, column=2)
        tk.Label(validation_frame, text="Max:").grid(row=0, column=3)
        tk.Entry(validation_frame, textvariable=self.max_val, width=6).grid(row=0, column=4)

        tk.Checkbutton(validation_frame, text="Enable Std Dev Outlier Check", variable=self.use_std).grid(row=1, column=0, sticky="w")
        tk.Label(validation_frame, text="Std Threshold:").grid(row=1, column=1)
        tk.Entry(validation_frame, textvariable=self.std_thresh, width=6).grid(row=1, column=2)

        # Right column: text frame with scrollbars
        right_column = tk.Frame(content_frame)
        right_column.pack(side="left", padx=10, fill="both", expand=True)

        self.text_display = tk.Text(right_column, height=18, width=100, wrap="none")
        self.text_display.grid(row=0, column=0, sticky="nsew")
        self.text_display.bind("<Button-1>", self.on_single_click_text)

        y_scrollbar = tk.Scrollbar(right_column, orient="vertical", command=self.text_display.yview)
        y_scrollbar.grid(row=0, column=1, sticky="ns")
        self.text_display.config(yscrollcommand=y_scrollbar.set)

        x_scrollbar = tk.Scrollbar(right_column, orient="horizontal", command=self.text_display.xview)
        x_scrollbar.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.text_display.config(xscrollcommand=x_scrollbar.set)

        right_column.grid_rowconfigure(0, weight=1)
        right_column.grid_columnconfigure(0, weight=1)

        # Search Row/Col below Step Mode
        search_frame = tk.Frame(right_column)
        search_frame.grid(row=3, column=0, columnspan=2, pady=(10, 20), sticky="n")
        tk.Label(search_frame, text="Row:").pack(side="left")
        self.search_row = tk.Entry(search_frame, width=5)
        self.search_row.pack(side="left")
        tk.Label(search_frame, text="Col:").pack(side="left")
        self.search_col = tk.Entry(search_frame, width=5)
        self.search_col.pack(side="left")
        tk.Button(search_frame, text="Go to Cell", command=self.goto_cell).pack(side="left", padx=5)

    def select_csv_file(self):
        filepath = filedialog.askopenfilename(
            initialdir=BASE_DIR,
            title="Select CSV File",
            filetypes=(("CSV Files", "*.csv"),)
        )
        if filepath:
            self.csv_path = filepath
            self.csv_label.config(text=os.path.basename(filepath))
            # Infer image_folder and table from path: output/image_folder/csv_outputs/table.csv
            rel_path = os.path.relpath(filepath, BASE_DIR)
            parts = rel_path.split(os.sep)
            if len(parts) >= 3:
                self.image_folder = parts[0]
                self.table = parts[2].replace(".csv", "")
            else:
                self.image_folder = ""
                self.table = ""

    def load_csv(self):
        """
        Loads the selected CSV file into memory and sets up the display.
        Initializes tracking indices and triggers outlier finding.
        """
        if not self.csv_path or not os.path.exists(self.csv_path):
            messagebox.showerror("Error", "No CSV file selected or file does not exist.")
            return

        self.current_csv = pd.read_csv(self.csv_path, header=None, dtype=str)
        self.current_csv = self.current_csv.applymap(lambda x: "" if str(x).strip().lower() in {"x"} else x)

        # Set table_path for images: output/image_folder/table
        self.table_path = os.path.join(BASE_DIR, self.image_folder, self.table)
        self.row_idx = 0
        self.col_idx = 0
        self.checking_outliers = False

        self.find_outliers()
        self.update_csv_display()
        self.load_next_invalid_cell()

    def update_csv_display(self):
        self.text_display.delete(1.0, tk.END)
        for row_idx, row in self.current_csv.iterrows():
            prefix = "➡ " if row_idx == self.row_idx else "   "
            self.text_display.insert(tk.END, f"{prefix}Row {row_idx + 1}: ")
            for col_idx, value in enumerate(row):
                cell_str = str(value)
                self.text_display.insert(tk.END, cell_str)
                if col_idx < len(row) - 1:
                    self.text_display.insert(tk.END, "\t")
            self.text_display.insert(tk.END, "\n")

    def is_invalid(self, value, is_first_col):
        value = value.strip()
        if value.lower() == "x" or value == "":
            return True
        if value.lower() == "nan" and self.ignore_nan_var.get():
            return False
        try:
            num = float(value)
            if self.use_min_max.get():
                min_v = float(self.min_val.get())
                max_v = float(self.max_val.get())
                return not (min_v <= num <= max_v)
            else:
                return not (-50 <= num <= 99)
        except ValueError:
            return True

    def find_outliers(self):
        self.outlier_indices.clear()
        if not self.use_std.get():
            return  # Skip outlier detection if not enabled
        try:
            std_thresh = float(self.std_thresh.get())
        except Exception:
            std_thresh = 2
        for col in range(1, self.current_csv.shape[1]):
            try:
                col_values = pd.to_numeric(self.current_csv.iloc[:, col], errors='coerce')
                valid_vals = col_values.dropna()
                if not valid_vals.empty:
                    mean = valid_vals.mean()
                    std = valid_vals.std()
                    for row in range(len(col_values)):
                        val = col_values.iat[row]
                        if pd.notna(val) and abs(val - mean) > std_thresh * std:
                            self.outlier_indices.add((row, col))
            except Exception:
                continue

    def load_next_invalid_cell(self):
        while self.row_idx < len(self.current_csv):
            while self.col_idx < self.current_csv.shape[1]:
                value = self.current_csv.iat[self.row_idx, self.col_idx]
                if not self.checking_outliers:
                    if self.is_invalid(str(value), self.col_idx == 0):
                        self.load_cell(value)
                        return
                elif self.col_idx > 0 and (self.row_idx, self.col_idx) in self.outlier_indices:
                    self.load_cell(value)
                    return
                self.col_idx += 1
            self.col_idx = 0
            self.row_idx += 1

        if not self.checking_outliers:
            self.checking_outliers = True
            self.row_idx = 0
            self.col_idx = 0
            self.find_outliers()
            self.load_next_invalid_cell()
        else:
            self.save_csv()
            messagebox.showinfo("Done", "No more invalid or outlier cells! CSV has been saved.")

    def load_cell(self, cell_value):
        self.current_text.set(cell_value)
        self.entry.select_range(0, tk.END)
        self.entry.focus_set()

        self.search_row.delete(0, tk.END)
        self.search_row.insert(0, str(self.row_idx + 1))
        self.search_col.delete(0, tk.END)
        self.search_col.insert(0, str(self.col_idx + 1))

        self.update_csv_display()

        img_path = os.path.join(self.table_path, f"row_{self.row_idx+1}", f"col_{self.col_idx+1}.png")
        if (os.path.exists(img_path)):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (300, 300))
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.image_panel.configure(image=imgtk)
            self.image_panel.image = imgtk
        else:
            self.image_panel.configure(image=None)
            self.image_panel.image = None

    def confirm_cell(self):
        value = self.current_text.get()
        self.current_csv.iat[self.row_idx, self.col_idx] = "" if value.strip().lower() in {"x", "nan"} else value
        self.col_idx += 1
        self.load_next_invalid_cell()

    def clear_cell(self):
        self.current_csv.iat[self.row_idx, self.col_idx] = ""
        self.col_idx += 1
        self.load_next_invalid_cell()

    def goto_cell(self):
        try:
            row = int(self.search_row.get()) - 1
            col = int(self.search_col.get()) - 1
            if row < 0 or col < 0 or row >= len(self.current_csv) or col >= self.current_csv.shape[1]:
                raise ValueError("Out of bounds")
            self.row_idx = row
            self.col_idx = col
            self.load_cell(self.current_csv.iat[row, col])
        except Exception:
            messagebox.showerror("Invalid Input", "Please enter valid row and column numbers.")

    def save_csv(self):
        self.current_csv.to_csv(self.csv_path, index=False, header=False)
        messagebox.showinfo("Saved", f"CSV saved to: {self.csv_path}")

    def add_decimal_prefix(self):
        confirm = messagebox.askyesno("Confirm Action", "Are you sure you want to add a decimal to the start of all values (excluding the first column)?")
        if not confirm:
            return

        for col in range(1, self.current_csv.shape[1]):
            for row in range(len(self.current_csv)):
                val = str(self.current_csv.iat[row, col]).strip()
                if val and val.lower() not in {"x", "nan"}:
                    try:
                        float(val)
                        if "." not in val:
                            self.current_csv.iat[row, col] = f".{val}"
                    except ValueError:
                        continue

        self.update_csv_display()
        messagebox.showinfo("Success", "Decimal prefixes added.")

    def handle_enter_key(self, event):
        self.confirm_cell()

    def on_single_click_text(self, event):
        try:
            index = self.text_display.index(f"@{event.x},{event.y}")
            line_num, char_index = map(int, index.split('.'))

            line_text = self.text_display.get(f"{line_num}.0", f"{line_num}.end")
            if "Row" not in line_text:
                return

            row_part, data_part = line_text.split(":", 1)
            row_idx = int(row_part.strip().replace("➡", "").replace("Row", "").strip()) - 1

            tab_parts = data_part.strip().split('\t')
            acc_len = 0
            for i, part in enumerate(tab_parts):
                acc_len += len(part) + 1  # +1 for tab
                if acc_len > char_index - len(row_part) - 2:  # -2 for ": "
                    col_idx = i
                    break
            else:
                col_idx = len(tab_parts) - 1  # fallback to last col

            self.row_idx = row_idx
            self.col_idx = col_idx
            self.load_cell(self.current_csv.iat[self.row_idx, self.col_idx])
        except Exception as e:
            print("Click parse error:", e)

    def validate_value(self, value, values_list):
        """Validate value against min, max, and std thresholds."""
        try:
            val = float(value)
        except ValueError:
            return False
        if self.use_min_max.get():
            min_v = float(self.min_val.get())
            max_v = float(self.max_val.get())
            if val < min_v or val > max_v:
                return False
        if self.std_thresh.get():
            arr = [float(v) for v in values_list if v not in ("", "NaN")]
            if arr:
                mean = sum(arr) / len(arr)
                std = (sum((x - mean) ** 2 for x in arr) / len(arr)) ** 0.5
                thresh = float(self.std_thresh.get())
                if abs(val - mean) > thresh * std:
                    return False
        return True

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRCheckerGUI(root)
    root.mainloop()