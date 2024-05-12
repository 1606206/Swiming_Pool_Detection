import os
import skimage.io as io
import numpy as np
import cv2

def load_images_from_files(file_list, directory):
    images = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')  # Extensiones de archivo de imagen válidas
    
    for file in file_list:
        if file.lower().endswith(valid_extensions):
            path = os.path.join(directory, file)
            image = io.imread(path)
            images.append(image)
    
    return images

def segment_yellow_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

def draw_detected_pools(image, rectangles):
    pool_data = []
    for i, (x, y, w, h) in enumerate(rectangles, 1):
        pool_data.append((x, y, x + w, y + h))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image, pool_data

def visualize_results(images, file_names):
    pool_data_all_images = {}
    for img, file_name in zip(images, file_names):
        yellow_segmented_img_hsv = segment_yellow_hsv(img)
        contours, _ = cv2.findContours(cv2.cvtColor(yellow_segmented_img_hsv, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = [cv2.boundingRect(contour) for contour in contours]
        image_with_boxes, pool_data = draw_detected_pools(yellow_segmented_img_hsv.copy(), rectangles)
        pool_data_all_images[file_name] = pool_data
    return pool_data_all_images

def hsvMask():
    train_dir = "./DATABASE/training_data/images/"
    train_files = os.listdir(train_dir)
    train_imgs = load_images_from_files(train_files, train_dir)

    pool_data_all_images = visualize_results(train_imgs, train_files)
    # Imprimir los datos de las bounding boxes
    for image_name, pool_data in pool_data_all_images.items():
        print(f"{image_name}: {pool_data}")
    return pool_data_all_images


