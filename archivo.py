import os
import random
import shutil

# Directorios de entrada y salida
input_folder = "./DATABASE/ultralytics/todas"

TRAIN_img = "./DATABASE/ultralytics/images/train"
VAL_img = "./DATABASE/ultralytics/images/val"
TEST_img = "./DATABASE/ultralytics/images/test"

TRAIN_labels = "./DATABASE/ultralytics/labels/train"
VAL_labels = "./DATABASE/ultralytics/labels/val"
TEST_labels = "./DATABASE/ultralytics/labels/test"

# Crear directorios de salida si no existen
for directory in [TRAIN_img, VAL_img, TEST_img, TRAIN_labels, VAL_labels, TEST_labels]:
    os.makedirs(directory, exist_ok=True)

# Obtener lista de archivos en el directorio de entrada
files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

# Determinar el tamaño de cada conjunto (train, val, test)
total_files = len(files)
train_size = int(0.7 * total_files)
val_size = int(0.15 * total_files)
test_size = total_files - train_size - val_size

# Escoger aleatoriamente los archivos para cada conjunto
random.shuffle(files)
train_files = files[:train_size]
val_files = files[train_size:train_size+val_size]
test_files = files[train_size+val_size:]

# Función para copiar archivos
def copy_files(file_list, source_dir, dest_img_dir, dest_label_dir):
    for filename in file_list:
        if filename.endswith('.jpg'):
            # Es un archivo de imagen (.png), copiar a directorio de imágenes
            shutil.copy(os.path.join(source_dir, filename), os.path.join(dest_img_dir, filename))
        elif filename.endswith('.txt'):
            # Es un archivo de etiqueta (.txt), copiar a directorio de etiquetas
            shutil.copy(os.path.join(source_dir, filename), os.path.join(dest_label_dir, filename))
        else:
            print(f"Advertencia: Archivo {filename} no reconocido.")

# Copiar archivos al directorio de salida según los conjuntos
copy_files(train_files, input_folder, TRAIN_img, TRAIN_labels)
copy_files(val_files, input_folder, VAL_img, VAL_labels)
copy_files(test_files, input_folder, TEST_img, TEST_labels)

print("Archivos copiados exitosamente en los directorios de train, val y test.")
