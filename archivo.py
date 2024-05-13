import os
import random
import shutil

# Directorios de entrada y salida
input_folder = "./DATABASE/ultralytics_modif/todas"

TRAIN_img = "./DATABASE/ultralytics_modif/images/train"
VAL_img = "./DATABASE/ultralytics_modif/images/val"
TEST_img = "./DATABASE/ultralytics_modif/images/test"

TRAIN_labels = "./DATABASE/ultralytics_modif/labels/train"
VAL_labels = "./DATABASE/ultralytics_modif/labels/val"
TEST_labels = "./DATABASE/ultralytics_modif/labels/test"

# Crear directorios de salida si no existen
for directory in [TRAIN_img, VAL_img, TEST_img, TRAIN_labels, VAL_labels, TEST_labels]:
    os.makedirs(directory, exist_ok=True)

# Obtener lista de archivos de imágenes en el directorio de entrada
image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

# Determinar el tamaño de cada conjunto (train, val, test)
total_files = len(image_files)
train_size = int(0.7 * total_files)
val_size = int(0.15 * total_files)
test_size = total_files - train_size - val_size

# Dividir los archivos en conjuntos
random.shuffle(image_files)
train_files = image_files[:train_size]
val_files = image_files[train_size:train_size + val_size]
test_files = image_files[train_size + val_size:]

# Función para copiar imágenes y etiquetas al conjunto correspondiente
def copy_images_and_labels(file_list, source_dir, dest_img_dir, dest_label_dir):
    for filename in file_list:
        img_path = os.path.join(source_dir, filename)
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(source_dir, label_filename)
        
        if os.path.exists(label_path):
            # Copiar la imagen al directorio de imágenes del conjunto
            shutil.copy(img_path, os.path.join(dest_img_dir, filename))
            # Copiar el archivo de etiqueta al directorio de etiquetas del conjunto
            shutil.copy(label_path, os.path.join(dest_label_dir, label_filename))
        else:
            print(f"Advertencia: No se encontró el archivo de etiqueta asociado para {filename}.")

# Copiar archivos al directorio de salida según los conjuntos
copy_images_and_labels(train_files, input_folder, TRAIN_img, TRAIN_labels)
copy_images_and_labels(val_files, input_folder, VAL_img, VAL_labels)
copy_images_and_labels(test_files, input_folder, TEST_img, TEST_labels)

print("Archivos copiados exitosamente en los directorios de train, val y test.")
