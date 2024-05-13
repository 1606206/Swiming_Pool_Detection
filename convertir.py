import os

def convertir_archivo(input_file, scale_factor):
    # Leer el contenido actual del archivo
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    
    # Abrir el archivo para escritura, sobrescribiendo su contenido
    with open(input_file, 'w') as outfile:
        # Procesar cada línea del archivo original
        for line in lines:
            # Dividir la línea en partes separadas por espacios
            parts = line.split()
            
            # Extraer las coordenadas del centro y tamaño
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Calcular las coordenadas de la esquina superior izquierda y la esquina inferior derecha
            xmin = int((x_center - width/2) * scale_factor)
            ymin = int((y_center - height/2) * scale_factor)
            xmax = int((x_center + width/2) * scale_factor)
            ymax = int((y_center + height/2) * scale_factor)
            
            # Escribir las coordenadas convertidas en el archivo de salida
            outfile.write(f"{xmin} {ymin} {xmax} {ymax}\n")

# Directorio de archivos de entrada
input_directory = 'DATABASE/ultralytics/labels/test'

# Factor de escala para la normalización (multiplicar por 224)
scale_factor = 224

# Listar todos los archivos en el directorio de entrada
file_list = os.listdir(input_directory)

# Procesar cada archivo en la lista
for filename in file_list:
    # Construir la ruta completa del archivo de entrada
    input_path = os.path.join(input_directory, filename)
    
    # Convertir el archivo directamente
    convertir_archivo(input_path, scale_factor)

print("Conversion completada. Archivos modificados directamente.")
