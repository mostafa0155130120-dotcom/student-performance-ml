"""Shared schema and preprocessing for the student performance project."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Class"
CLASS_ORDER = ["L", "M", "H"]

# Demographic fields are retained only for a descriptive audit. They are not
# passed to the deployed classifier.
AUDIT_ONLY_COLUMNS = ["gender", "NationalITy", "PlaceofBirth"]
NUMERIC_FEATURES = [
    "raisedhands",
    "VisITedResources",
    "AnnouncementsView",
    "Discussion",
]
CATEGORICAL_FEATURES = [
    "StageID",
    "GradeID",
    "SectionID",
    "Topic",
    "Semester",
    "Relation",
    "ParentAnsweringSurvey",
    "ParentschoolSatisfaction",
    "StudentAbsenceDays",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim strings and normalize target labels without changing feature values."""
    result = frame.copy()
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].astype("string").str.strip()
    if TARGET in result.columns:
        result[TARGET] = result[TARGET].str.upper()
    return result


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
