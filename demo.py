import tkinter as tk
from app import OCRAppGUI

class DemoOCRAppGUI(OCRAppGUI):
    def run_ocr(self):
        # Override to show a message instead of running OCR
        tk.messagebox.showinfo("Demo", "OCR would run here. This is a demo.")

if __name__ == "__main__":
    root = tk.Tk()
    app = DemoOCRAppGUI(root)
    root.mainloop()