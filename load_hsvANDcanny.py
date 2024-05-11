import os
import skimage.io as io
import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_images_from_files(file_list, directory):
    images = []
    for file in file_list:
        path = os.path.join(directory, file)
        image = io.imread(path)
        images.append(image)
    return images

def filter_contours_by_area(contours, min_area):
    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            filtered_contours.append(contour)
    return filtered_contours

def segment_yellow_and_detect_pools(img, min_pool_area=500):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    result = cv2.bitwise_and(img, img, mask=mask)
    edges = cv2.Canny(result, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = filter_contours_by_area(contours, min_pool_area)
    return contours

def draw_detected_pools(image, contours):
    pool_data = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        pool_data.append((x, y, x + w, y + h))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image, pool_data

def visualize_results(images, file_names):
    pool_data_all_images = {}
    for img, file_name in zip(images, file_names):
        contours = segment_yellow_and_detect_pools(img)
        img_with_boxes, pool_data = draw_detected_pools(img.copy(), contours)
        pool_data_all_images[file_name] = pool_data
        
        #plt.figure(figsize=(12, 6))
        #plt.subplot(1, 2, 1)
        #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #plt.title(file_name)
        #plt.axis('off')

        #plt.subplot(1, 2, 2)
        #plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
        #plt.title("Yellow area segmentation with edge detection (filtered by area)")
        #plt.axis('off')

        #plt.show()
        
    return pool_data_all_images


def hsvCanny():
    train_dir = "./DATABASE/training_data/images/"
    train_files = os.listdir(train_dir)
    train_imgs = load_images_from_files(train_files, train_dir)

    pool_data_all_images = visualize_results(train_imgs, train_files)
    # Imprimir los datos de las bounding boxes
    for image_name, pool_data in pool_data_all_images.items():
        print(f"{image_name}: {pool_data}")
    return pool_data_all_images



