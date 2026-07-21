# YOLO Lab CLI

[English](../README.md) | [中文](README_zh.md) | [Español](README_es.md)

Outil de formation en ligne de commande pour la segmentation YOLO, basé sur Ultralytics.

## Fonctionnalités

- Trois modes d'entraînement : Nouveau / Reprendre / Ajuster
- Augmentation de données activable
- Validation automatique avec journalisation CSV (métriques globales et par classe)
- Isolation des expériences : chaque exécution crée des répertoires et journaux indépendants
- Paramètres CLI (`--epochs`, `--imgsz`, `--batch`, `--device`, `--name`)
- Détection automatique de la langue système (zh/en/fr/es), avec `--lang` pour forcer

## Démarrage Rapide

```bash
git clone https://github.com/Liujingze11/YOLO-LAB-CLI.git
cd YOLO-LAB-CLI
pip install -r requirements.txt
python main.py
```

## Prérequis

- Python 3.8+
- ultralytics, PyYAML

```bash
pip install ultralytics pyyaml
```

## Structure du Projet

```
YOLO-LAB-CLI/
├── main.py                 # Point d'entrée CLI (i18n, analyse des arguments, menu)
├── work_flows.py                # Flux d'entraînement (nouveau / reprendre / ajuster)
├── config.py               # Chemins par défaut + réexportations core
├── yaml/                   # Fichiers de configuration des jeux de données
│   └── data.yaml.example   # Exemple de configuration
├── core/                   # Bibliothèque partagée (identique CLI/GUI/LAB)
│   ├── config.py           # Classe de données TrainConfig
│   ├── training.py         # Utilitaires d'entraînement
│   ├── train_logger.py     # Journalisation CSV
│   ├── device.py           # Détection GPU
│   ├── i18n.py             # Aide à l'i18n
│   └── paths.py            # Registre des modèles
├── tools/                  # Scripts utilitaires
│   ├── predict_tools/      # Inférence (predict.py + paramètres de tâche)
│   └── dataset_tools/      # Division des jeux de données & outils d'étiquettes
├── outputs/                # Résultats d'entraînement (git-ignoré)
│   ├── result/             # Poids des modèles et graphiques
│   └── logs/               # Journaux d'entraînement CSV
├── locales/                # Fichiers de traduction i18n
└── pretrained_models/      # Poids des modèles pré-entraînés
```

## Modes d'Entraînement

Lancez `python main.py` et choisissez :

- **1** — Nouvel entraînement depuis les poids initiaux
- **2** — Reprendre depuis last.pt
- **3** — Ajuster depuis le best.pt historique

## Options CLI

```bash
python main.py --epochs 200 --imgsz 1280 --batch 8 --device 0 --name mon_experience
```

La langue est détectée automatiquement. Forcer avec `--lang` :

```bash
python main.py --lang fr   # Français
python main.py --lang en   # English
python main.py --lang zh   # 中文
python main.py --lang es   # Español
```

## Format de Configuration des Données

Placez vos fichiers YAML de configuration dans le répertoire `yaml/`. Voir `yaml/data.yaml.example` :

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

## Résultats

- Poids et graphiques : `outputs/result/<experiment_name>/`
- Journaux CSV : `outputs/logs/`

## License

MIT
