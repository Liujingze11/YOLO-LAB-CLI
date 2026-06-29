# Flatten Project Structure — Design Spec

**Date:** 2026-06-29
**Status:** approved

## Motivation

`main.py` at the project root is a thin passthrough — it only imports `main` from `scripts/train_segment.py` and calls it. The real CLI entry point lives deep inside `scripts/`. The user wants `main.py` to be the genuine entry point with its own logic, and the `scripts/` directory removed.

## Target Structure

```
YOLO-LAB-CLI/
├── main.py           ← Real CLI entry point: i18n, arg parsing, mode menu
├── config.py         ← TrainConfig + paths merged
├── train.py          ← Pure training library (train_segment minus CLI)
├── train_logger.py   ← Logging (unchanged)
├── predict.py        ← Prediction (renamed from predict_test.py)
├── data.yaml
├── locales/
├── dataset_tools/
├── pretrained_models/
```

The `scripts/` directory is deleted.

## File Changes

### 1. `config.py` — merge paths.py, fix root

- Merge `scripts/paths.py` constants into `config.py`
- Change `PROJECT_ROOT` from `Path(__file__).resolve().parent.parent` to `Path(__file__).resolve().parent` (since it lives at root now)
- Delete `scripts/paths.py`

### 2. `train.py` — rename + strip CLI

Formerly `scripts/train_segment.py`.

**Removed:**
- `main()` function
- `parse_args()`
- `override_config_from_args()`
- i18n functions: `_detect_lang()`, `_load_locale()`, `_t()`
- `if __name__ == "__main__"` block
- `LOCALE_DIR` constant

**Added:**
- `set_locale(loc)` — allows `main.py` to inject the locale dict into the module

**Kept:**
- All training functions (`start_new_training`, `resume_training`, `train_from_previous_best`)
- All helper functions (`ask_confirm_train`, `list_experiments`, `find_latest_experiment_dir`, `build_train_kwargs`, `ask_use_augment`, `get_class_names_from_data_yaml`, `get_val_labels_dir`, `count_val_label_stats`, `get_val_metrics`, `log_validation_result`)
- Module-level `_loc` variable

### 3. `main.py` — become the real entry point

**Added:**
- i18n functions: `_detect_lang()`, `_load_locale()`, `_t()`
- `LOCALE_DIR` pointing to `parent / "locales"`
- `parse_args()`
- `override_config_from_args()`
- Real `main()` with mode selection menu (1=new, 2=resume, 3=finetune)
- Calls `train.set_locale(_loc)` before entering the menu

**Imports from train.py:**
- `set_locale`, `start_new_training`, `resume_training`, `train_from_previous_best`
- Imports from config.py: `TrainConfig`

### 4. `predict.py` — rename

Rename `scripts/predict_test.py` → `predict.py`. Fix imports from `paths` → `config`.

### 5. `train_logger.py` — unchanged

Move from `scripts/` to root. No code changes.

## Path Changes

All existing constants in `paths.py` use `PROJECT_ROOT` combined with relative paths. Since `PROJECT_ROOT` changes from `parent.parent` to `parent`, the actual resolved paths remain identical — the file just lives one level higher now.

`LOCALE_DIR` moves from `train_segment.py` to `main.py`:
- Old: `Path(__file__).resolve().parent.parent / "locales"`  (in scripts/)
- New: `Path(__file__).resolve().parent / "locales"`  (in root)

## i18n Injection Pattern

`train.py` exposes `set_locale()` so `main.py` can inject the locale after parsing `--lang`:

```python
# main.py
_loc = _load_locale(lang)
train.set_locale(_loc)
```

`train.py` functions continue to reference the module-level `_loc` variable — no function signatures need to change.

## What Stays Intact

- `data.yaml` — untouched
- `locales/` — untouched
- `dataset_tools/` — untouched
- `pretrained_models/` — untouched
- `train_logger.py` — no code changes
- All training logic, validation logic — no behavior changes
