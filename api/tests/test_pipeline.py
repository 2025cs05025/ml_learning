"""
Task 5: Unit Tests for Data Processing and Model Code
Run with:  pytest tests/ -v --cov=src --cov-report=term-missing
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.eda import load_data, clean_data
from src.train import (
    build_logistic_regression,
    build_random_forest,
    prepare_features,
)


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def raw_df():
    return load_data("data/cleve.csv")


@pytest.fixture
def clean_df(raw_df):
    return clean_data(raw_df)


@pytest.fixture
def splits(clean_df):
    return prepare_features(clean_df)


# ─────────────────────────────────────────────
# TASK 1: DATA LOADING TESTS
# ─────────────────────────────────────────────

def test_load_data_shape(raw_df):
    """Dataset must have more than 200 rows and 14 columns."""
    assert raw_df.shape[0] > 200, "Too few rows"
    assert raw_df.shape[1] == 14, f"Expected 14 cols, got {raw_df.shape[1]}"


def test_load_data_columns(raw_df):
    """Required column names must be present."""
    required = ['age', 'sex', 'cp', 'trestbps', 'chol',
                'fbs', 'restecg', 'thalach', 'exang',
                'oldpeak', 'slope', 'ca', 'thal', 'target']
    for col in required:
        assert col in raw_df.columns, f"Missing column: {col}"


# ─────────────────────────────────────────────
# TASK 2: PREPROCESSING TESTS
# ─────────────────────────────────────────────

def test_clean_no_missing_values(clean_df):
    """Cleaned dataset must have zero NaN values."""
    assert clean_df.isna().sum().sum() == 0, "NaN values found after cleaning"


def test_clean_no_question_marks(clean_df):
    """No '?' strings should remain after cleaning."""
    for col in clean_df.select_dtypes(include='object').columns:
        assert '?' not in clean_df[col].values, f"'?' found in column {col}"


def test_clean_target_is_binary(clean_df):
    """Target must only contain 0 and 1."""
    unique_targets = set(clean_df['target'].unique())
    assert unique_targets <= {0, 1}, f"Non-binary target values: {unique_targets}"


def test_clean_all_numeric(clean_df):
    """After cleaning every column should be numeric."""
    non_numeric = [c for c in clean_df.columns
                   if not pd.api.types.is_numeric_dtype(clean_df[c])]
    assert len(non_numeric) == 0, f"Non-numeric columns: {non_numeric}"


def test_clean_sex_encoding(clean_df):
    """Sex must be encoded as 0 or 1."""
    assert set(clean_df['sex'].unique()) <= {0, 1}


def test_clean_target_both_classes(clean_df):
    """Both classes (0 and 1) must be present in the target."""
    assert 0 in clean_df['target'].values
    assert 1 in clean_df['target'].values


# ─────────────────────────────────────────────
# TASK 3: FEATURE / SPLIT TESTS
# ─────────────────────────────────────────────

def test_train_test_split_sizes(splits):
    X_train, X_test, y_train, y_test = splits
    total = len(y_train) + len(y_test)
    assert abs(len(y_test) / total - 0.2) < 0.05, "Test split not ~20%"


def test_no_target_in_features(splits):
    X_train, _, _, _ = splits
    assert 'target' not in X_train.columns


def test_feature_count(splits):
    X_train, _, _, _ = splits
    assert X_train.shape[1] == 13, f"Expected 13 features, got {X_train.shape[1]}"


# ─────────────────────────────────────────────
# TASK 4: MODEL PIPELINE TESTS
# ─────────────────────────────────────────────

def test_logistic_regression_pipeline(splits):
    """LR pipeline must fit and predict without errors."""
    X_train, X_test, y_train, y_test = splits
    model = build_logistic_regression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(preds) <= {0, 1}


def test_random_forest_pipeline(splits):
    """RF pipeline must fit and predict without errors."""
    X_train, X_test, y_train, y_test = splits
    model = build_random_forest()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(preds) <= {0, 1}


def test_model_predict_proba(splits):
    """Model must return valid probabilities (sum to 1)."""
    X_train, X_test, y_train, y_test = splits
    model = build_random_forest()
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    assert probs.shape == (len(y_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_model_accuracy_reasonable(splits):
    """Models should achieve >70% accuracy on test set."""
    X_train, X_test, y_train, y_test = splits
    for build_fn in [build_logistic_regression, build_random_forest]:
        model = build_fn()
        model.fit(X_train, y_train)
        acc = (model.predict(X_test) == y_test).mean()
        assert acc > 0.70, f"{build_fn.__name__} accuracy {acc:.2f} < 0.70"


# ─────────────────────────────────────────────
# TASK 5: API INPUT VALIDATION TESTS
# ─────────────────────────────────────────────

def test_valid_patient_data():
    """PatientData schema should accept valid input."""
    from src.app import PatientData
    p = PatientData(
        age=63, sex=1, cp=3, trestbps=145, chol=233,
        fbs=1, restecg=2, thalach=150, exang=0,
        oldpeak=2.3, slope=2, ca=0, thal=1
    )
    assert p.age == 63
    assert p.sex == 1


def test_patient_data_requires_all_fields():
    """PatientData must raise ValidationError if a field is missing."""
    from pydantic import ValidationError
    from src.app import PatientData
    with pytest.raises(ValidationError):
        PatientData(age=63, sex=1)  # missing many required fields
