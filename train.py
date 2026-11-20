"""Reproducible model comparison, holdout evaluation, and artifact creation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from model import (
    AUDIT_ONLY_COLUMNS,
    CLASS_ORDER,
    FEATURE_COLUMNS,
    TARGET,
    build_preprocessor,
    normalize_frame,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "xAPI-Edu-Data.csv"
DEFAULT_ARTIFACTS = ROOT / "artifacts"
RANDOM_STATE = 42


def candidate_models() -> dict[str, Pipeline]:
    """Small, diverse model set suitable for a dataset of this size."""
    models = {
        "Logistic regression": LogisticRegression(
            max_iter=2500, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Extra trees": ExtraTreesClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "RBF support vector machine": SVC(
            C=1.0,
            kernel="rbf",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }
    return {
        name: Pipeline(
            steps=[("preprocess", build_preprocessor()), ("classifier", estimator)]
        )
        for name, estimator in models.items()
    }


def _save_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        display_labels=["Low (L)", "Medium (M)", "High (H)"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Holdout confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _fairness_audit(
    gender: pd.Series, y_true: pd.Series, y_pred: pd.Series
) -> pd.DataFrame:
    rows = []
    aligned_gender = gender.reset_index(drop=True)
    actual = y_true.reset_index(drop=True)
    predicted = y_pred.reset_index(drop=True)
    for group in sorted(aligned_gender.dropna().unique()):
        mask = aligned_gender.eq(group)
        group_true, group_pred = actual[mask], predicted[mask]
        low_mask = group_true.eq("L")
        rows.append(
            {
                "gender_group": str(group),
                "n": int(mask.sum()),
                "accuracy": float(accuracy_score(group_true, group_pred)),
                "macro_f1": float(
                    f1_score(
                        group_true,
                        group_pred,
                        labels=CLASS_ORDER,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "low_class_recall": (
                    float(recall_score(group_true, group_pred, labels=["L"], average="macro", zero_division=0))
                    if low_mask.any()
                    else None
                ),
                "low_class_count": int(low_mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def train_and_evaluate(
    data_path: Path = DEFAULT_DATA, artifact_dir: Path = DEFAULT_ARTIFACTS
) -> dict:
    """Fit and save the best cross-validated model, then assess on a holdout set."""
    data_path, artifact_dir = Path(data_path), Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(data_path)
    frame = normalize_frame(raw)
    required = set(FEATURE_COLUMNS + [TARGET])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if frame[TARGET].isna().any() or not set(frame[TARGET].dropna()).issubset(CLASS_ORDER):
        raise ValueError("Target column 'Class' must contain only L, M, and H values.")

    y = frame[TARGET].astype(str)
    X = frame[FEATURE_COLUMNS]
    indices = frame.index.to_numpy()
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    smallest_class = int(y_train.value_counts().min())
    folds = min(5, smallest_class)
    if folds < 2:
        raise ValueError("Each class needs at least two training examples for stratified CV.")
    cv = RepeatedStratifiedKFold(
        n_splits=folds, n_repeats=3, random_state=RANDOM_STATE
    )
    scoring = {"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"}

    leaderboard_rows = []
    models = candidate_models()
    for name, pipeline in models.items():
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
            error_score="raise",
        )
        leaderboard_rows.append(
            {
                "model": name,
                "cv_macro_f1_mean": float(scores["test_macro_f1"].mean()),
                "cv_macro_f1_std": float(scores["test_macro_f1"].std()),
                "cv_balanced_accuracy_mean": float(
                    scores["test_balanced_accuracy"].mean()
                ),
                "cv_balanced_accuracy_std": float(
                    scores["test_balanced_accuracy"].std()
                ),
            }
        )

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["cv_macro_f1_mean", "cv_balanced_accuracy_mean"], ascending=False
    )
    leaderboard.to_csv(artifact_dir / "cv_leaderboard.csv", index=False)
    best_name = str(leaderboard.iloc[0]["model"])
    calibration_folds = min(3, int(y_train.value_counts().min()))
    best_model = CalibratedClassifierCV(
        estimator=clone(models[best_name]),
        method="sigmoid",
        cv=calibration_folds,
        ensemble=False,
    ).fit(X_train, y_train)
    y_pred = pd.Series(best_model.predict(X_test), index=y_test.index, name="predicted_class")
    model_scores = pd.DataFrame(
        best_model.predict_proba(X_test), columns=best_model.classes_, index=y_test.index
    ).reindex(columns=CLASS_ORDER)
    one_hot = np.column_stack([(y_test == label).astype(int) for label in CLASS_ORDER])
    multiclass_brier = float(np.mean(np.sum((model_scores.to_numpy() - one_hot) ** 2, axis=1)))

    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    metrics = {
        "selected_model": best_name,
        "selection_metric": "mean macro F1 over repeated stratified cross-validation",
        "random_state": RANDOM_STATE,
        "rows": int(len(frame)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "features_used": FEATURE_COLUMNS,
        "audit_only_fields": [c for c in AUDIT_ONLY_COLUMNS if c in frame.columns],
        "class_order": CLASS_ORDER,
        "holdout_accuracy": float(accuracy_score(y_test, y_pred)),
        "holdout_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "holdout_macro_f1": float(
            f1_score(y_test, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)
        ),
        "holdout_weighted_f1": float(
            f1_score(y_test, y_pred, labels=CLASS_ORDER, average="weighted", zero_division=0)
        ),
        "holdout_multiclass_brier_score": multiclass_brier,
        "majority_baseline_accuracy": float(accuracy_score(y_test, dummy_pred)),
        "probabilities_calibrated": True,
        "calibration_method": "sigmoid calibration using training folds",
        "data_source_rows": int(len(raw)),
    }
    (artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=CLASS_ORDER,
        target_names=["Low (L)", "Medium (M)", "High (H)"],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(artifact_dir / "classification_report.csv")
    _save_confusion_matrix(y_test, y_pred, artifact_dir / "confusion_matrix.png")

    if "gender" in frame.columns:
        audit = _fairness_audit(frame.loc[test_idx, "gender"], y_test, y_pred)
        audit.to_csv(artifact_dir / "gender_audit.csv", index=False)
    else:
        pd.DataFrame(columns=["gender_group", "n", "accuracy", "macro_f1", "low_class_recall", "low_class_count"]).to_csv(
            artifact_dir / "gender_audit.csv", index=False
        )

    importance = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=6,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance_frame = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "macro_f1_importance_mean": importance.importances_mean,
            "macro_f1_importance_std": importance.importances_std,
        }
    ).sort_values("macro_f1_importance_mean", ascending=False)
    importance_frame.to_csv(artifact_dir / "feature_importance.csv", index=False)

    predictions = pd.DataFrame(
        {
            "actual_class": y_test,
            "predicted_class": y_pred,
            "model_score_for_predicted_class": model_scores.max(axis=1),
        }
    )
    if "gender" in frame.columns:
        predictions["gender_audit_only"] = frame.loc[test_idx, "gender"]
    predictions.to_csv(artifact_dir / "holdout_predictions.csv", index_label="row_index")

    joblib.dump(best_model, artifact_dir / "model.joblib", compress=3)
    metadata = {
        **metrics,
        "pipeline_file": "model.joblib",
        "model_score_note": "Sigmoid-calibrated scores; small holdout means calibration remains uncertain.",
    }
    (artifact_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "metrics": metrics,
        "leaderboard": leaderboard.reset_index(drop=True),
        "classification_report": pd.DataFrame(report).T,
        "gender_audit": pd.read_csv(artifact_dir / "gender_audit.csv"),
        "feature_importance": importance_frame,
        "model": best_model,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Input CSV path")
    parser.add_argument(
        "--artifacts", type=Path, default=DEFAULT_ARTIFACTS, help="Output model/metrics folder"
    )
    args = parser.parse_args()
    result = train_and_evaluate(args.data, args.artifacts)
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
    print("\nCross-validation leaderboard:")
    print(result["leaderboard"].to_string(index=False))


if __name__ == "__main__":
    main()
