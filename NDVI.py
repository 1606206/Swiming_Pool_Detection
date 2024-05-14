import numpy as np
import cv2
import pandas as pd
import matplotlib as plt
import time
import PIL
import os
import skimage as ski
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


def calculate_GLI(image):
    # Convert image to float32
    img_float = image.astype(np.float32)

    # Split the image into its RGB channels
    red_channel = img_float[:, :, 0]
    green_channel = img_float[:, :, 1]
    blue_channel = img_float[:, :, 2]

    # Calculate the GLI
    gli = (green_channel - red_channel - (10 * blue_channel)) / (green_channel + red_channel + (10 * blue_channel))

    # Scale GLI to [0, 255]
    gli_scaled = ((gli + 1) / 2) * 255

    return gli_scaled.astype(np.uint8)


def generate_synthetic_NIR(gray_img):
    # Apply contrast stretching to enhance the contrast of the grayscale image
    min_intensity = np.min(gray_img)
    max_intensity = np.max(gray_img)
    stretched_img = cv2.convertScaleAbs(gray_img, alpha=255.0 / (max_intensity - min_intensity),
                                        beta=-min_intensity * (255.0 / (max_intensity - min_intensity)))

    # Apply thresholding to identify regions corresponding to vegetation
    _, vegetation_mask = cv2.threshold(stretched_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert the mask to a binary image
    vegetation_binary = cv2.threshold(vegetation_mask, 1, 255, cv2.THRESH_BINARY)[1]

    # Dilate the binary image to enhance vegetation regions
    kernel = np.ones((5, 5), np.uint8)
    dilated_vegetation = cv2.dilate(vegetation_binary, kernel, iterations=1)

    # Create a synthetic NIR band by multiplying the grayscale image with the dilated vegetation mask
    synthetic_NIR = cv2.bitwise_and(gray_img, gray_img, mask=dilated_vegetation)

    return synthetic_NIR


def generate_synthetic_red(rgb_img):
    # Split the RGB image into its channels
    red_channel = rgb_img[:, :, 2]  # Red channel is the third channel (index 2)

    # Calculate the difference between max and min intensities
    intensity_diff = np.max(red_channel) - np.min(red_channel)
    # Check if the difference is zero
    if intensity_diff == 0:
        # You can choose to return the original red channel or some default value
        # For example, you can return a black image of the same size:
        return np.zeros_like(red_channel, dtype=np.uint8)

    # Apply contrast stretching to enhance the contrast of the red channel
    stretched_red_channel = cv2.convertScaleAbs(red_channel, alpha=255.0 / intensity_diff, beta=0)
    """stretched_red_channel = cv2.convertScaleAbs(red_channel, alpha=255.0 / intensity_diff, beta=-np.min(
    red_channel) * (255.0 / intensity_diff))"""

    return stretched_red_channel


def preprocess_image(image):
    """
    Preprocesses the image for use in neural network
    image: RGB image
    :param image:
    :return:
    """

    #Calculate GLI(Green-Leaf Index), out put as gray scale
    image_GLI = calculate_GLI(image)

    # Apply histogram equalization
    img_equalized = cv2.equalizeHist(image_GLI)

    #Simmulate a NIR(near-infrared) on img
    img_NIR = generate_synthetic_NIR(img_equalized)

    """test_img = -(img_equalized - img_NIR)
    plt.imshow(test_img)
    plt.axis("off")
    plt.show()
    
    # Apply the threshold
    thresholded_img = cv2.bitwise_and(image, image, mask=test_img)
    plt.imshow(thresholded_img)
    plt.axis("off")
    plt.show()"""

    #Synthetic red band
    synthetic_red = generate_synthetic_red(image)

    # Merge the synthetic NIR, red, and original green channels into a single image
    false_color_composite = cv2.merge((img_NIR, image[:, :, 1], synthetic_red))

    # Convert the data type to float32 for numerical stability in calculations
    synthetic_NIR = img_NIR.astype(np.float32)
    synthetic_red = synthetic_red.astype(np.float32)

    # Calculate the NDVI-like index
    ndvi_like_index = (synthetic_NIR - synthetic_red) / (synthetic_NIR + synthetic_NIR)

    """# Scale the index to the range [0, 255] for visualization (optional)
    ndvi_like_index_scaled = ((ndvi_like_index + 1) / 2) * 255"""

    # Threshold the NDVI-like index to separate water bodies and vegetation
    _, water_mask = cv2.threshold(ndvi_like_index, 0, 255, cv2.THRESH_BINARY)
    water_mask = cv2.bitwise_not(water_mask)

    # Apply the water mask to the false-color composite image
    false_color_composite[np.where((water_mask == 255))] = [0, 0, 255]

    # Invert the water mask to represent vegetation as white
    vegetation_mask = cv2.bitwise_not(water_mask)

    # Apply the vegetation mask to the false-color composite image
    image[np.where((vegetation_mask == 255))] = [255, 0, 0]
    return image
