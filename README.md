# Swiming_Pool_Detection
Grupo formado por:
- Eric Rodríguez Merichal
- Wenpeng Ji
- Guillermo Vivancos Alonso

# Estructura de los Archivos python
Cuando hablamos de imagenes **modificadas**, es que hemos usado las iamgenes procesadas por **NDVI**

**Swiming_pool_Detection.ipynb** --> Archivo principal del proyecto, aqui cargamos todos los archivos que mencionaremos a continuación, se encarga de comparar las bbox de cada piscina para devolver un accuracy midiendo el iou (intersection over union).

**load_clasification** --> Carga los labels de la BBDD

**load_ hsvANDcanny** --> Carga los labels obtenidos de las bbox generadas por el algoritmo hsv mezclado con Canny
    - esta funcion tiene 1 input (preprocesado)
        - Preprocesado puede ser *0* o *1*, en caso de que sea *1* entonces preprocesamos

**load_hsvMask** -->  Carga los labels obtenidos de las bbox generadas por el algoritmo hsv mezclado con Mask
    - esta funcion tiene 1 input (preprocesado)
        - Preprocesado puede ser *0* o *1*, en caso de que sea *1* entonces preprocesamos

**load_ultralytics** --> Carga los labels obtenidos de las bbox generadas en las predicciones de imagenes de test de ultralytics con Yolov8
    - esta funcion tiene 2 inputs (epochs, modificador)
        - Epochs puede ser *10* o *100*, ya que son los entrenamientos realizados en la red neuronal
        - Modificador puede ser *True* o *False* para dictaminar si estamos usando las imagenes modificadas con NDVI o las normales

**NDVI.py** --> Archivo donde se preprocesan las imagenes, se eliminan obstaculos innecesarios y se filtra por color para obtener imagenes MODIFICADAS

**generate_proc_img.py** --> Genera las imagenes preprocesadas por el algoritmo NDVI

**old_school** --> Algoritmo Old svhool con HSV Canny y Mask, *load_hsvANDcanny* y *load_hsvMask* vienen de este archivo

**yolo_ultralytics.ipynb** --> Archivo donde se entrena el modelo con yolov8 de las imagenes normales para poder obtener el accuracy
    - Este archivo utiliza la BBDD ubicada en *"DATABASE/ultralytics/images"* donde dividimos las imagenes en *train/test/val*
    - Las imagenes de test predecidas se pueden ver accediendo 1 carpeta mas en 10epoch_accuracy o 100_epoch_accuracy

**yolo_ultralytics_modif.ipynb** --> Archivo donde se entrena el modelo con yolov8 de las imagenes modificadas para poder obtener el accuracy
    - Este archivo utiliza la BBDD ubicada en *"DATABASE/ultralytics_modif/images"* donde dividimos las imagenes en train/test/val
    - Las imagenes de test predecidas se pueden ver accediendo 1 carpeta mas en 10epoch_accuracy o 100_epoch_accuracy

**yolo_ultralytics_pred.ipynb**  --> Archivo que hace lo mismo que su anterior pero predice sobre todas las imagenes de test sin labels, en el mismo archivo se hace un promedio de la probabilidad con la que el algoritmo piensa que es correcto, pero no hay una forma de verificarlo
    - Este archivo utiliza la BBDD ubicada en *"DATABASE/ultralytics_sin_accuracy/images"* donde dividimos las imagenes en *train/val*
    - Las imagenes de test predecidas se pueden encontrar en *"DATABASE/predictions_ultralytics"*, puesto que teniamos muchas mas imagenes para entrenar, hemos añadido 1 epoch también

**yolo_ultralytics_pred_modif.ipynb**  --> Archivo que hace lo mismo que su anterior pero para las imagenes modificadas
    - Este archivo utiliza la BBDD ubicada en *"DATABASE/ultralytics_sin_accuracy_modif/images"* donde dividimos las imagenes en *train/val*
    - Las imagenes de test predecidas se pueden encontrar en *"DATABASE/predictions_ultralytics_modif"*, puesto que teniamos muchas mas imagenes para entrenar, hemos añadido 1 epoch también


**Verificar_DATASET.ipynb** --> Archivo que se encarga desde un inicio de clasificar todos los labels xml en txt, asignar carpetas donde hay piscinas y donde no las hay, modificar la estructura de los archivos... Trabajar con la BBDD proporcionada para facilitarnos su uso

**Detect_video.ipynb** --> Archivo para predecir frames de un video y hacer el output con el nombre *test_video_out.mp4*
**make_video.ipynb** --> Convierte 300 imagenes del directorio test_data_images en un video para poder predecir sobre el

# Estructura Carpetas

**Carpeta runs** --> En la carpeta runs enconctramos todos los resultados de cada entrenamiento, los archivos finalizados con *Detect* son los aplicados a imagenes de test sin labels, los finalizados en *Accuracy* los que si que hemos podido ver el accuracy
    - En esta carpeta encontramos todo tipo de información destacable del entrenamiento, la cual se puede ver en la imagen *results.jpg*.
    - También encontramos resumenes de algunas imagenes durante el entrenamiento *train_batch.jpg*, *val_batch.jpg* y *pred_batch.jpg*
    - Además tenemos información de los propios labels en *labels.jpg* y *labels_correlogram.jpg*
    - Finalmente encontramos información sobre las métricas *precision-recall.jpg*, *Recall-confidence.jpg* y *Precision-confidence.jpg*

**Carpeta DATABASE** --> En esta carpeta encontramos todas las BBDD utilizadas a lo largo del proyecto

**Carpeta ultralytics_trainings** --> En esta carpeta encontraremos todos los entrenamientos realizados hasta la fecha con sus respectivos nombres