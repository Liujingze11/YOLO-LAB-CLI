# YOLO Lab CLI

[English](../README.md) | [中文](README_zh.md) | [Français](README_fr.md)

Herramienta de línea de comandos para entrenamiento de segmentación YOLO, basada en Ultralytics.

## Funcionalidades

- Tres modos de entrenamiento: Nuevo / Reanudar / Ajustar
- Aumento de datos activable
- Validación automática con registro CSV (métricas globales y por clase)
- Aislamiento de experimentos: cada ejecución crea directorios y registros independientes
- Parámetros CLI (`--epochs`, `--imgsz`, `--batch`, `--device`, `--name`)
- Detección automática del idioma del sistema (zh/en/fr/es), con `--lang` para forzar

## Inicio Rápido

```bash
git clone https://github.com/Liujingze11/YOLO-LAB-CLI.git
cd YOLO-LAB-CLI
pip install -r requirements.txt
python main.py
```

## Requisitos

- Python 3.8+
- ultralytics, PyYAML

```bash
pip install ultralytics pyyaml
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
│   ├── device.py           # Detección de GPU
│   ├── i18n.py             # Ayuda i18n
│   └── paths.py            # Registro de modelos
├── tools/                  # Scripts de utilidad
│   ├── predict_tools/      # Inferencia (predict.py + parámetros de tarea)
│   └── dataset_tools/      # División de datasets y herramientas de etiquetas
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

## Opciones CLI

```bash
python main.py --epochs 200 --imgsz 1280 --batch 8 --device 0 --name mi_experimento
```

El idioma se detecta automáticamente. Forzar con `--lang`:

```bash
python main.py --lang es   # Español
python main.py --lang en   # English
python main.py --lang fr   # Français
python main.py --lang zh   # 中文
```

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
- Registros CSV: `outputs/logs/`

## Licencia

MIT
