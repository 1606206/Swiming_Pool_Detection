import os

def convertir_formato_inplace(input_folder):
    # Obtener la lista de archivos .txt en el carpeta de entrada
    txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    
    # Iterar sobre cada archivo .txt
    for txt_file in txt_files:
        file_path = os.path.join(input_folder, txt_file)
        
        # Leer el contenido del archivo original
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Abrir el mismo archivo para escribir (sobrescribir el contenido)
        with open(file_path, 'w') as f:
            for line in lines:
                # Dividir la línea en partes usando espacio como separador
                parts = line.strip().split()
                
                if len(parts) == 5:  # Asegurarse de que la línea tiene 5 partes
                    clase = parts[0]
                    xmin = float(parts[1])
                    ymin = float(parts[2])
                    xmax = float(parts[3])
                    ymax = float(parts[4])
                    
                    # Calcular el centro (x_center, y_center) y el tamaño (width, height)
                    x_center = (xmin + xmax) / 2.0
                    y_center = (ymin + ymax) / 2.0
                    width = xmax - xmin
                    height = ymax - ymin
                    
                    # Construir la línea en el nuevo formato
                    new_line = f"{clase} {x_center} {y_center} {width} {height}\n"
                    
                    # Escribir la línea en el archivo (sobrescribir el contenido)
                    f.write(new_line)

# Directorio de entrada (misma carpeta que contiene los archivos .txt)
input_directory = "DATABASE/ultralytics_modif/labels/val"

# Llamar a la función para convertir el formato de los archivos .txt inplace
convertir_formato_inplace(input_directory)