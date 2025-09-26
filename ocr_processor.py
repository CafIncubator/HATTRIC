import os
import pandas as pd
from google.cloud import vision
from google.auth.exceptions import DefaultCredentialsError

# Lazily create the Vision client so importing this module doesn't require credentials

def _find_service_account_json():
    """Return path to a service account JSON if found via env or key/ folder."""
    env_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if env_path and os.path.isfile(env_path):
        return env_path
    # Look for a JSON key inside the local key/ folder
    here = os.path.dirname(os.path.abspath(__file__))
    key_dir = os.path.join(here, "key")
    if os.path.isdir(key_dir):
        for fname in os.listdir(key_dir):
            if fname.lower().endswith(".json"):
                return os.path.join(key_dir, fname)
    return None


def _get_vision_client():
    """Initialize and return a Vision API client if credentials are available, else None."""
    cred_path = _find_service_account_json()
    if cred_path and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
    try:
        return vision.ImageAnnotatorClient()
    except DefaultCredentialsError:
        return None


def process_image(image_path, client):
    with open(image_path, "rb") as f:
        content = f.read()
    image = vision.Image(content=content)
    image_context = vision.ImageContext(language_hints=["en"])
    response = client.text_detection(image=image, image_context=image_context)
    texts = response.text_annotations
    return texts[0].description.strip() if texts else ""


def run_ocr_on_table(table_path, csv_output_folder, image_folder, table):
    client = _get_vision_client()
    if client is None:
        raise RuntimeError(
            "Google Cloud Vision credentials not found. Place your service account .json in the 'key' folder or set the GOOGLE_APPLICATION_CREDENTIALS environment variable."
        )

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
            text = process_image(col_path, client)
            row_data.append(text)
        data.append(row_data)
    os.makedirs(csv_output_folder, exist_ok=True)
    csv_filename = f"{table}.csv"
    csv_path = os.path.join(csv_output_folder, csv_filename)
    pd.DataFrame(data).to_csv(csv_path, index=False, header=False)
    print(f"✅ OCR finished and saved: {csv_path}")