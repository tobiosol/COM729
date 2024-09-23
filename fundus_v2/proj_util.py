import os
import cv2
import numpy as np
from PIL import Image
import torch



ROOT_DIR = os.path.join("dataset", "v2")
LABEL_DIR = os.path.join(ROOT_DIR, "label")
TRAIN_LABEL_PATH = os.path.join(LABEL_DIR, "train_label.csv")
VALIDATION_LABEL_PATH = os.path.join(LABEL_DIR, "validation_label.csv")
TEST_LABEL_PATH = os.path.join(LABEL_DIR, "test_label.csv")

IMAGES_DIR = os.path.join(ROOT_DIR, "images")
TRAINING_DIR = os.path.join(IMAGES_DIR, "training")
VALIDATION_DIR = os.path.join(IMAGES_DIR, "validation")
TESTING_DIR = os.path.join(IMAGES_DIR, "testing")

ORIGINAL_DIR = os.path.join(ROOT_DIR, "original")
ORIGINAL_TRAINING_DIR = os.path.join(ORIGINAL_DIR, "training")
ORIGINAL_TESTING_DIR = os.path.join(ORIGINAL_DIR, "testing")
ORIGINAL_VALIDATION_DIR = os.path.join(ORIGINAL_DIR, "validation")


def get_filename_from_path(filepath):
    """
    Extracts the filename from a given file path.

    Args:
        filepath (str): The full path to the file.

    Returns:
        str: The filename without the extension.
    """
    return os.path.splitext(os.path.basename(filepath))[0]

def load_images_from_folder(image_folder):
    return [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.lower().endswith((".jpg", ".png"))]

def save_processed_image(image, dest_image_dir, original_filename):
    new_filename = f"{os.path.splitext(original_filename)[0]}{os.path.splitext(original_filename)[1]}"
    save_path = os.path.join(dest_image_dir, new_filename)
    
    if isinstance(image, Image.Image):
        image = np.array(image)
    elif isinstance(image, np.ndarray):
        pass
    else:
        raise TypeError("Input must be a PIL image or a NumPy array.")
    
    cv2.imwrite(save_path, image)

def extract_filename(file_path):
    """
    Extracts the base filename without extension and underscore suffix.

    Args:
        file_path (str): The path of the file.

    Returns:
        str: The base filename without the underscore suffix.
    """
    basename = os.path.basename(file_path)
    name, _ = os.path.splitext(basename)
    name_only = name.split('_')[0]
    return name_only

def get_trained_model(model_name):
    folder_path = 'trained_model'
    os.makedirs(folder_path, exist_ok=True)
    return os.path.join(folder_path, model_name)

def save_model(model, model_path):
    model_file = get_trained_model(model_name=model_path)
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file}")

def load_model(model, path):
    model_file = get_trained_model(model_name=path)
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))
        model.eval()  # Set the model to evaluation mode
        print(f"Model loaded from {model_file}")
    else:
        print(f"Model file {model_file} does not exist. Train the model first.")
    return model

def model_file_exist(model_path):
    model_file = get_trained_model(model_name=model_path)
    return os.path.exists(model_file)