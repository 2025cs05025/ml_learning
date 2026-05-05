This folder holds trained artifacts produced by: python src/model_training/train.py

Files (do not edit by hand):

  best_model.pkl
      Joblib dump of the winning sklearn Pipeline (imputer + scaler or RF + classifier).
      Loaded by src/model_training/predict.py (batch) and src/api/api.py (Docker / REST).

  feature_names.pkl
      Ordered list of training column names. Prediction aligns X columns to this list.

  training_metadata.json
      Human-readable summary: which model won, test metrics, sklearn/pandas versions,
      paths to the pickles above, and the clean CSV path used for training.

These files are regenerated each time you train; keep backups if you need an older model.
