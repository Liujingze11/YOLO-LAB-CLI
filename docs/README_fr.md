# YOLO Lab CLI

[English](../README.md) | [中文](README_zh.md) | [Español](README_es.md)

Outil de formation en ligne de commande pour la segmentation YOLO, basé sur Ultralytics.

## Fonctionnalités

- Trois modes d'entraînement : Nouveau / Reprendre / Ajuster
- Flux de confirmation étape par étape : YAML du jeu de données → hyperparamètres → augmentation → Mixup → sauvegarde de tous les poids d'époques
- Augmentation de données activable (mosaic, mixup, copy-paste, effacement aléatoire, retournements, HSV ...)
- Gel du backbone, LR cosinus, ordonnanceurs de LR interchangeables (adaptive / restart / cosine)
- Option « sauvegarder tous les poids d'époques », organisés sous `weights/epochs/`
- Validation automatique avec journalisation CSV (métriques globales et par classe)
- Isolation des expériences : chaque exécution crée des répertoires et journaux indépendants
- Paramètres CLI (`--epochs`, `--imgsz`, `--batch`, `--device`, `--name`)
- Détection automatique de la langue système (zh/en/fr/es)

## Démarrage Rapide

```bash
git clone https://github.com/Liujingze11/YOLO-LAB-CLI.git
cd YOLO-LAB-CLI
pip install -r requirements.txt
python main.py
```

## Prérequis

- Python 3.8+
- ultralytics, PyYAML, numpy

```bash
pip install ultralytics pyyaml numpy
```

## Structure du Projet

```
YOLO-LAB-CLI/
├── main.py                 # Point d'entrée CLI (i18n, analyse des arguments, menu)
├── training_flows.py       # Flux d'entraînement (nouveau / reprendre / ajuster)
├── cli_config.py           # Chemins CLI par défaut + réexportations TrainConfig
├── yaml/                   # Fichiers de configuration des jeux de données
│   └── data.yaml.example   # Exemple de configuration
├── core/                   # Bibliothèque partagée (identique CLI/GUI/LAB)
│   ├── train_config.py     # Classe TrainConfig + persistance de configuration
│   ├── training.py         # Utilitaires d'entraînement
│   ├── train_logger.py     # Journalisation CSV
│   ├── lr_schedulers.py    # Callbacks d'ordonnanceurs de LR
│   ├── device.py           # Détection GPU
│   ├── i18n.py             # Aide à l'i18n
│   └── paths.py            # Registre des modèles
├── tools/                  # Scripts utilitaires
│   ├── yolo_task_predict.py  # Inférence (paramètres par tâche)
│   ├── yolo_batch_predict.py # Inférence par lot, images annotées
│   ├── yolo_split_dataset.py # Division des jeux de données
│   └── yolo_create_empty_labels.py # Créer des étiquettes vides
├── docs/                   # README traduits (zh / fr / es)
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

Chaque mode passe par un flux de confirmation — YAML du jeu de données → hyperparamètres → augmentation de données → valeur de Mixup → sauvegarde de tous les poids d'époques — afin de vérifier et ajuster chaque réglage avant de lancer l'entraînement.

## Options CLI

```bash
python main.py --epochs 200 --imgsz 1280 --batch 8 --device 0 --name mon_experience
```

La langue est détectée automatiquement.

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
- Poids par époque (optionnel) : `outputs/result/<experiment_name>/weights/epochs/`
- Journaux CSV : `outputs/logs/`

## License

MIT
