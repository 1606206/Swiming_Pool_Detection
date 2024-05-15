import os
import skimage.io as io
import numpy as np
import cv2

def load_images_from_files(directory):
    images = []
    for file in os.listdir(directory):
        if file.endswith('.jpg'):
            path = os.path.join(directory, file)
            image = io.imread(path)
            images.append((file, image))
    return images

# Función para segmentar las áreas amarillas en el espacio de color HSV
def segment_yellow_hsv(img):
    # Convertir la imagen a espacio de color HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Definir los rangos de color para el amarillo en HSV
    lower_yellow = np.array([15, 80, 80])   # Umbral inferior para tono, saturación y valor
    upper_yellow = np.array([40, 255, 255]) # Umbral superior

    # Crear una máscara que identifique los píxeles que están dentro del rango de color amarillo
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Aplicar operaciones morfológicas para mejorar la segmentación
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Aplicar la máscara a la imagen original
    result = cv2.bitwise_and(img, img, mask=mask)

    return result

# Función para dibujar los cuadrados delimitadores alrededor de las piscinas, eliminando las piscinas muy pequeñas
def draw_detected_pools(image, rectangles, min_area=500):
    filtered_rectangles = [rect for rect in rectangles if rect[2] * rect[3] > min_area]
    bounding_boxes = []
    for i, (x, y, w, h) in enumerate(filtered_rectangles, 1):
        xmax = x + w
        ymax = y + h
        bounding_boxes.append((x, y, xmax, ymax))
        cv2.rectangle(image, (x, y), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return image, bounding_boxes

def visualize_results(images):
    pool_boxes = {}  # Diccionario para almacenar las bounding boxes de las piscinas detectadas en cada imagen
    for file, img in images:
        # Segmentación de áreas amarillas en el espacio de color HSV
        yellow_segmented_img_hsv = segment_yellow_hsv(img)

        # Obtener los contornos de las piscinas en la imagen segmentada
        contours, _ = cv2.findContours(cv2.cvtColor(yellow_segmented_img_hsv, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = [cv2.boundingRect(contour) for contour in contours]

        # Dibujar los cuadrados verdes alrededor de las piscinas y obtener las bounding boxes
        _, bounding_boxes = draw_detected_pools(yellow_segmented_img_hsv.copy(), rectangles)

        # Actualizar el diccionario de bounding boxes de piscinas
        pool_boxes[file] = bounding_boxes

    return pool_boxes

def hsvMask(preprocesado):

    if preprocesado == 0:
        train_dir = "./DATABASE/training_data/images/"
    elif preprocesado == 1:
        train_dir = "./DATABASE/training_data_modif/"
        
    train_imgs = load_images_from_files(train_dir)
    hsv_and_mask_labels = visualize_results(train_imgs)

    return hsv_and_mask_labels



