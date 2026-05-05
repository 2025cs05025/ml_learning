"""
Model training and batch inference **scripts** (**File:** ``src/model_training/__init__.py``).

**Contents**

- ``train.py`` — Builds sklearn pipelines, ``GridSearchCV``, evaluation metrics,
  MLflow experiment logging under ``mlruns/``, and writes the winning pipeline plus
  metadata to ``models/``.
- ``predict.py`` — Loads ``models/best_model.pkl``, reapplies the same preprocessing
  as training on a raw CSV, writes optional batch predictions.

**Distinction.** The directory ``models/`` (plural) holds **artifacts**
(``best_model.pkl``, ``feature_names.pkl``, ``training_metadata.json``). This package
``model_training/`` holds **code** only.

**Upstream.** Requires ``data/heart_clean.csv`` from ``eda/eda.py`` for training and
``data_preprocessing`` for prediction feature parity.

This package intentionally exports no runtime API beyond submodule imports; run scripts
with ``python src/model_training/train.py`` from the repo root (see README).
"""

__all__: list[str] = []
