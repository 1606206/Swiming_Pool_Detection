import os
import xml.etree.ElementTree as ET
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

# Función para leer los archivos XML y obtener las etiquetas de las imágenes
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    filename = root.find('filename').text
    objects = root.findall('object')
    labels = []
    for obj in objects:
        name = obj.find('name').text
        if name == '2':  # Si el objeto es una piscina
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            labels.append((xmin, ymin, xmax, ymax))
    return filename, labels

# Función para cargar las imágenes y sus etiquetas juntas
def load_images_and_labels(img_dir, xml_dir):
    images_with_pools = {}
    images_without_pools = []
    for img_file in os.listdir(img_dir):
        if img_file.endswith('.jpg'):
            img_path = os.path.join(img_dir, img_file)
            xml_file = os.path.join(xml_dir, os.path.splitext(img_file)[0] + '.xml')
            if os.path.exists(xml_file):
                filename, img_labels = parse_xml(xml_file)
                # Verificar si hay etiquetas de piscinas en este archivo XML
                if img_labels:
                    img = load_img(img_path, target_size=(224, 224))
                    img_array = img_to_array(img)
                    images_with_pools[img_file] = img_labels
                else:
                    images_without_pools.append(img_file)
    return images_with_pools, images_without_pools



def clasification():
    img_dir = "DATABASE/training_data/images"
    xml_dir = "DATABASE/training_data/labels"
    images_with_pools, images_without_pools = load_images_and_labels(img_dir, xml_dir)
    print("Imágenes con piscinas:")
    for img_file, labels in images_with_pools.items():
        print(img_file, ":", labels)

    print("Imágenes sin piscinas:", images_without_pools)
    return images_with_pools, images_without_pools

