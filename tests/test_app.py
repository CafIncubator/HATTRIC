import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil
import numpy as np
import cv2
import pytest

from segmentation import start_segmentation
from app import get_output_folder, get_csv_output_folder, sharpen_image, sharpen_segmented_images, OCRAppGUI
from error_checker_gui import OCRCheckerGUI


from unittest import mock
import pandas as pd

# Reusable dummy GUI class for testing
class DummyGUI:
    def __init__(self, tmp_path):
        # Only set up the minimal attributes needed for tests
        self.current_text = mock.Mock()
        self.entry = mock.Mock()
        self.search_row = mock.Mock()
        self.search_col = mock.Mock()
        self.text_display = mock.Mock()
        self.update_csv_display = mock.Mock()
        self.table_path = str(tmp_path)
        self.row_idx = 0
        self.col_idx = 0
        self.image_panel = mock.Mock()
        self.current_csv = pd.DataFrame([["a", "b"], ["c", "d"]])



# Use a dedicated output folder for tests
TEST_OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "output")

def test_get_output_folder(tmp_path, monkeypatch):
    """
    Test that get_output_folder creates the correct directory structure.
    """
    # Patch the OUTPUT_ROOT to use the test output directory
    monkeypatch.setattr("app.OUTPUT_ROOT", TEST_OUTPUT_ROOT)
    folder = get_output_folder("test_folder", "test_table")
    # Check that the folder exists and has the correct path
    assert os.path.exists(folder)
    assert folder.endswith(os.path.join("test_folder", "test_table"))
    # Cleanup after test
    shutil.rmtree(os.path.join(TEST_OUTPUT_ROOT, "test_folder"))

def test_get_csv_output_folder(tmp_path, monkeypatch):
    """
    Test that get_csv_output_folder creates the correct CSV output directory.
    """
    monkeypatch.setattr("app.OUTPUT_ROOT", TEST_OUTPUT_ROOT)
    folder = get_csv_output_folder("test_folder")
    assert os.path.exists(folder)
    assert folder.endswith(os.path.join("test_folder", "csv_outputs"))
    # Cleanup after test
    shutil.rmtree(os.path.join(TEST_OUTPUT_ROOT, "test_folder"))

def test_sharpen_image():
    """
    Test that sharpen_image returns an image of the same shape and dtype.
    """
    img = np.ones((10, 10, 3), dtype=np.uint8) * 127
    sharpened = sharpen_image(img)
    assert sharpened.shape == img.shape
    assert sharpened.dtype == img.dtype

def test_sharpen_segmented_images(tmp_path):
    """
    Test that sharpen_segmented_images processes images in a folder.
    """
    # Create a fake image in a temp folder
    folder = tmp_path / "seg"
    folder.mkdir()
    img_path = folder / "test.png"
    img = np.ones((10, 10, 3), dtype=np.uint8) * 127
    cv2.imwrite(str(img_path), img)
    # Run the sharpening function
    sharpen_segmented_images(str(folder))
    assert os.path.exists(img_path)
    loaded = cv2.imread(str(img_path))
    assert loaded is not None

def test_start_segmentation_saves_cells(monkeypatch):
    """
    Test that start_segmentation saves at least one segmented cell image.
    Skips if no test images are available.
    """
    # Locate the test images folder
    april_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "input_tables", "april"))
    if not os.path.exists(april_folder):
        pytest.skip("input_tables/april does not exist, skipping segmentation test.")
    # Find PNG images in the folder
    pngs = [f for f in os.listdir(april_folder) if f.lower().endswith(".png")]
    if not pngs:
        pytest.skip("No PNG images found in input_tables/april/, skipping segmentation test.")
    img_file = pngs[0]
    img_path = os.path.join(april_folder, img_file)

    # Output to tests/output
    test_output_folder = os.path.join(os.path.dirname(__file__), "output", "april", os.path.splitext(img_file)[0])
    os.makedirs(test_output_folder, exist_ok=True)

    with mock.patch("cv2.imshow"), mock.patch("cv2.waitKey", return_value=13):
        from segmentation import start_segmentation
        start_segmentation(img_path, test_output_folder)

    # Check that at least one segmented cell image was saved
    found = False
    for root, dirs, files in os.walk(test_output_folder):
        for file in files:
            if file.lower().endswith(".png"):
                found = True
                break
    assert found, f"No segmented cell images saved in {test_output_folder}"

def test_google_vision_api_connection():
    """
    Test that a connection to the Google Vision API can be established.
    This test sets the GOOGLE_APPLICATION_CREDENTIALS environment variable
    to the key found in the 'key' folder at the project root.
    """
    import pathlib

    # Find the key file in the 'key' folder at the project root
    root_dir = pathlib.Path(__file__).parent.parent
    key_folder = root_dir / "key"
    key_files = list(key_folder.glob("*.json"))
    if not key_files:
        pytest.skip("No Google Vision API key file found in the 'key' folder.")
    key_path = str(key_files[0])

    # Set the environment variable for authentication
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

    try:
        from google.cloud import vision
        from google.api_core.exceptions import InvalidArgument
        client = vision.ImageAnnotatorClient()
        # Intentionally send an incomplete request to test connection/auth
        try:
            client.annotate_image({'image': {'content': b''}})
        except InvalidArgument:
            # 400 error means connection/auth is OK, just invalid request
            return
        except Exception as e:
            pytest.fail(f"Unexpected error type: {e}")
        else:
            # If no error at all, that's also fine (but unlikely)
            return
    except ImportError:
        pytest.skip("google-cloud-vision is not installed.")
    except Exception as e:
        pytest.fail(f"Could not connect to Google Vision API: {e}")


def test_load_cell(monkeypatch, tmp_path):
    """
    Test the load_cell method using the real OCRCheckerGUI class,
    with all GUI and file/image operations mocked.
    """
    # Create an instance without calling __init__ to avoid Tkinter setup
    gui = OCRCheckerGUI.__new__(OCRCheckerGUI)
    # Manually set required attributes
    gui.current_text = mock.Mock()
    gui.entry = mock.Mock()
    gui.search_row = mock.Mock()
    gui.search_col = mock.Mock()
    gui.text_display = mock.Mock()
    gui.update_csv_display = mock.Mock()
    gui.table_path = str(tmp_path)
    gui.row_idx = 0
    gui.col_idx = 0
    gui.image_panel = mock.Mock()
    gui.current_csv = pd.DataFrame([["a", "b"], ["c", "d"]])

    # Test when image does not exist
    monkeypatch.setattr("os.path.exists", lambda path: False)
    gui.load_cell("test_value")
    gui.current_text.set.assert_called_with("test_value")
    gui.image_panel.configure.assert_called_with(image=None)
    assert gui.image_panel.image is None

    # Test when image exists (mock image processing)
    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr("cv2.imread", lambda path: mock.Mock())
    monkeypatch.setattr("cv2.cvtColor", lambda img, code: mock.Mock())
    monkeypatch.setattr("cv2.resize", lambda img, size: mock.Mock())
    monkeypatch.setattr("PIL.Image.fromarray", lambda arr: mock.Mock())
    monkeypatch.setattr("PIL.ImageTk.PhotoImage", lambda image: "imgtk")
    gui.load_cell("another_value")
    gui.image_panel.configure.assert_called_with(image="imgtk")
    assert gui.image_panel.image == "imgtk"