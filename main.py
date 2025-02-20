import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import os
import time
from PIL import Image
import numpy as np
import shutil


# Step 1: Load the Teachable Machine Model
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


# check if the network path is accessible
def check_network_path(network_path):
    try:
        os.listdir(network_path)
        return True
    except OSError:
        return False


# Wait for the network path to be accessible
def wait_for_network_path(network_path, delay=10):
    while not check_network_path(network_path):
        print("Waiting for network path to be accessible")
        time.sleep(delay)


# Step 2: Create a User Interface for Directory Selection
def select_directory():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_selected = filedialog.askdirectory()
    root.destroy()
    return folder_selected


# Step 3: Process Images in the Directory in Batches
def process_directory_in_batches(directory, model, batch_size=1000):
    batch_images = []
    batch_paths = []
    total_files = sum([len(files) for r, d, files in os.walk(directory)])
    processed = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                image = Image.open(image_path)
                image = image.resize((224, 224))  # Resize according to your model's requirement
                image_array = np.array(image) / 255.0  # Normalize if needed
                batch_images.append(image_array)
                batch_paths.append(image_path)

                if len(batch_images) == batch_size:
                    predictions = model.predict(np.array(batch_images))
                    for path, prediction in zip(batch_paths, predictions):
                        yield path, prediction, processed, total_files
                        processed += 1
                    batch_images = []
                    batch_paths = []

    # Process the last batch
    if batch_images:
        predictions = model.predict(np.array(batch_images))
        for path, prediction in zip(batch_paths, predictions):
            yield path, prediction, processed, total_files
            processed += 1


# Step 4: Move Images to the Appropriate Directories
def move_image(image_path, prediction, root_directory, input_directory):
    # Network path check
    wait_for_network_path(root_directory)

    # Determine the class with the highest probability
    predicted_class = np.argmax(prediction)

    if predicted_class == 0:  # Illustrations
        dest_dir = os.path.join(root_directory, 'Illustration')
    elif predicted_class == 1:  # Text
        dest_dir = os.path.join(root_directory, 'Text')
    elif predicted_class == 2:  # Empty Pages
        dest_dir = os.path.join(root_directory, 'EmptyPages')
    elif predicted_class == 3:  # Title Pages
        dest_dir = os.path.join(root_directory, 'TitlePages')
    elif predicted_class == 4:  # Tables
        dest_dir = os.path.join(root_directory, 'Tables')
    else:
        raise ValueError(f'Unknown class index: {predicted_class}')

    move_file(image_path, dest_dir, input_directory)


def move_file(image_path, dest_dir, input_directory):
    # Network path check
    wait_for_network_path(input_directory)

    sub_path = os.path.relpath(image_path, start=input_directory)
    dest_path = os.path.join(dest_dir, sub_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.move(image_path, dest_path)


# Main Function
def main():
    model = load_model('keras_model.h5')  # Replace with your model path
    input_directory = select_directory()
    root_directory = os.path.dirname(input_directory)

    for image_path, prediction, processed, total in process_directory_in_batches(input_directory, model):
        move_image(image_path, prediction, root_directory, input_directory)
        print(f'Processed {processed} of {total} images')


if __name__ == "__main__":
    main()
