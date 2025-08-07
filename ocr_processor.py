import os
import pandas as pd
from google.cloud import vision

client = vision.ImageAnnotatorClient()

def process_image(image_path):
    with open(image_path, "rb") as f:
        content = f.read()
    image = vision.Image(content=content)
    image_context = vision.ImageContext(language_hints=["en"])
    response = client.text_detection(image=image, image_context=image_context)
    texts = response.text_annotations
    return texts[0].description.strip() if texts else ""

def run_ocr_on_table(table_path, csv_output_folder, image_folder, table):
    data = []
    # Sort row folders numerically
    row_folders = sorted(
        [d for d in os.listdir(table_path) if d.startswith("row_") and os.path.isdir(os.path.join(table_path, d))],
        key=lambda x: int(x.split("_")[1])
    )
    for row_folder in row_folders:
        row_path = os.path.join(table_path, row_folder)
        row_data = []
        # Sort col images numerically
        col_files = sorted(
            [f for f in os.listdir(row_path) if f.startswith("col_") and f.lower().endswith(".png")],
            key=lambda x: int(x.split("_")[1].split(".")[0])
        )
        for col_file in col_files:
            col_path = os.path.join(row_path, col_file)
            text = process_image(col_path)
            row_data.append(text)
        data.append(row_data)
    os.makedirs(csv_output_folder, exist_ok=True)
    csv_filename = f"{table}.csv"
    csv_path = os.path.join(csv_output_folder, csv_filename)
    pd.DataFrame(data).to_csv(csv_path, index=False, header=False)
    print(f"✅ OCR finished and saved: {csv_path}")