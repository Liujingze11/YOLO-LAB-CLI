# YOLO Lab CLI

[English](../README.md) | [中文](README_zh.md) | [Français](README_fr.md)

Herramienta de línea de comandos para entrenamiento de segmentación YOLO, basada en Ultralytics.

## Funcionalidades

- Tres modos de entrenamiento: Nuevo / Reanudar / Ajustar
- Flujo de confirmación paso a paso: YAML del dataset → hiperparámetros → aumento → Mixup → guardar todos los pesos de épocas
- Aumento de datos activable (mosaic, mixup, copy-paste, borrado aleatorio, volteos, HSV ...)
- Congelar backbone, LR coseno, planificadores de LR intercambiables (adaptive / restart / cosine)
- Opción «guardar todos los pesos de épocas», organizados en `weights/epochs/`
- Validación automática con registro CSV (métricas globales y por clase)
- Aislamiento de experimentos: cada ejecución crea directorios y registros independientes
- Parámetros CLI (`--epochs`, `--imgsz`, `--batch`, `--device`, `--name`)
- Detección automática del idioma del sistema (zh/en/fr/es)

## Inicio Rápido

```bash
git clone https://github.com/Liujingze11/YOLO-LAB-CLI.git
cd YOLO-LAB-CLI
pip install -r requirements.txt
python main.py
```

## Requisitos

- Python 3.8+
- ultralytics, PyYAML, numpy

```bash
pip install ultralytics pyyaml numpy
```

## Estructura del Proyecto

```
YOLO-LAB-CLI/
├── main.py                 # Punto de entrada CLI (i18n, análisis de argumentos, menú)
├── training_flows.py       # Flujos de entrenamiento (nuevo / reanudar / ajustar)
├── cli_config.py           # Rutas CLI predeterminadas + reexportaciones TrainConfig
├── yaml/                   # Archivos de configuración de datasets
│   └── data.yaml.example   # Ejemplo de configuración
├── core/                   # Biblioteca compartida (idéntica CLI/GUI/LAB)
│   ├── train_config.py     # Clase TrainConfig + persistencia de configuración
│   ├── training.py         # Utilidades de entrenamiento
│   ├── train_logger.py     # Registro CSV
│   ├── lr_schedulers.py    # Callbacks de planificadores de LR
│   ├── device.py           # Detección de GPU
│   ├── i18n.py             # Ayuda i18n
│   └── paths.py            # Registro de modelos
├── tools/                  # Scripts de utilidad
│   ├── yolo_task_predict.py  # Inferencia (parámetros por tarea)
│   ├── yolo_batch_predict.py # Inferencia por lotes, imágenes anotadas
│   ├── yolo_split_dataset.py # División de datasets
│   └── yolo_create_empty_labels.py # Crear etiquetas vacías
├── docs/                   # README traducidos (zh / fr / es)
├── outputs/                # Salidas de entrenamiento (git-ignorado)
│   ├── result/             # Pesos de modelos y gráficos
│   └── logs/               # Registros CSV de entrenamiento
├── locales/                # Archivos de traducción i18n
└── pretrained_models/      # Pesos de modelos pre-entrenados
```

## Modos de Entrenamiento

Ejecute `python main.py` y elija:

- **1** — Nuevo entrenamiento desde pesos iniciales
- **2** — Reanudar desde last.pt
- **3** — Ajustar desde best.pt histórico

Cada modo pasa por un flujo de confirmación — YAML del dataset → hiperparámetros → aumento de datos → valor de Mixup → guardar todos los pesos de épocas — para revisar y ajustar cada configuración antes de comenzar el entrenamiento.

## Opciones CLI

```bash
python main.py --epochs 200 --imgsz 1280 --batch 8 --device 0 --name mi_experimento
```

El idioma se detecta automáticamente.

## Formato de Configuración de Datos

Coloque sus archivos YAML de configuración en el directorio `yaml/`. Ver `yaml/data.yaml.example`:

```yaml
path: ./data/datasets
train: images/train
val: images/val
test: images/test
names:
  0: background
  1: class_a
  2: class_b
```

## Resultados

- Pesos y gráficos: `outputs/result/<experiment_name>/`
- Pesos por época (opcional): `outputs/result/<experiment_name>/weights/epochs/`
- Registros CSV: `outputs/logs/`

## Licencia

MIT
