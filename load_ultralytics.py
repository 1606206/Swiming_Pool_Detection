from ultralytics import YOLO
from pathlib import Path

def load_ultralytics_dicc(epochs, modif = False):
    if (modif == True):
        input_dir = Path('DATABASE/ultralytics_modif/images/test')
    else:
        input_dir = Path('DATABASE/ultralytics/images/test')
    # Rutas de directorios
    if (epochs == 10):
        if (modif == True):
            output_dir_epoch = Path('DATABASE/ultralytics_modif/images/10epoch_accuracy')
        else:
            output_dir_epoch = Path('DATABASE/ultralytics/images/10epoch_accuracy')

    elif (epochs == 100):
        if (modif == True):
            output_dir_epoch = Path('DATABASE/ultralytics_modif/images/100epoch_accuracy')
        else:
            output_dir_epoch = Path('DATABASE/ultralytics/images/100epoch_accuracy')

    # Verificar si los directorios de salida existen, si no, crearlos
    output_dir_epoch.mkdir(parents=True, exist_ok=True)

    if (epochs == 10):
        if (modif == True):
            model_epoch = YOLO("ultralytics_trainings/10epoch_accuracy_modif.pt")
        else:
            model_epoch = YOLO("ultralytics_trainings/10epoch_accuracy.pt")
    elif (epochs == 100):
        if (modif == True):
            model_epoch = YOLO("ultralytics_trainings/100epoch_accuracy_modif.pt")
        else:
            model_epoch = YOLO("ultralytics_trainings/100epoch_accuracy.pt")
    
    
    image_files = list(input_dir.glob('*.*'))

    dict_epoch = {}
    for image_file in image_files:
        coords_epoch = []
        filename = image_file.stem

        output_path_epoch = output_dir_epoch / f"{filename}.jpg"

        results_epoch = model_epoch([str(image_file)])

        
        for result in results_epoch:
            boxes = result.boxes         
            if len(boxes) > 1:
                for box in boxes:
                    coord = box.xyxy.tolist()
                    coords_epoch.append(coord)
            elif len(boxes) == 1:
                coord = boxes.xyxy.tolist()
                coords_epoch.append(coord)
            result.save(filename=str(output_path_epoch))

        dict_epoch[f"{filename}.jpg"] = coords_epoch

    return dict_epoch