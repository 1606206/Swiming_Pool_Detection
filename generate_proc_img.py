import NDVI2
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob


def save_image(image_data, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find the number of existing files in the directory
    existing_files = len([name for name in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, name))])

    # Start numbering files from the next number after the existing files
    file_number = existing_files

    # Create a PIL Image object from the image data
    image = Image.fromarray(image_data)

    # Save the image with a sequential filename
    filename = str(file_number).zfill(9) + '.jpg'  # Ensure 9 digits with leading zeros
    filepath = os.path.join(output_dir, filename)
    image.save(filepath)


# Directory containing the images
directory = 'DATABASE/test_data_images'

# Find all files ending with '.jpg' in the directory
image_files = sorted(glob.glob(os.path.join(directory, '*.jpg')))

# Iterate through the image files
for filename in image_files:
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = NDVI2.preprocess_image(image)
    save_image(image, 'DATABASE/test_data_images_modif_bis')

