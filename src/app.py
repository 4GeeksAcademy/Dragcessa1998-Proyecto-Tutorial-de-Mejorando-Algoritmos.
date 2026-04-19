from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
MPLCONFIGDIR = Path(tempfile.gettempdir()) / "dragcessa_boosting_matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

TARGET_COLUMN = "Outcome"
RANDOM_STATE = 42
DATA_URL = "https://breathecode.herokuapp.com/asset/internal-link?id=930&path=diabetes.csv"

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
INTERIM_DIR = BASE_DIR / "data" / "interim"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_PATH = RAW_DIR / "diabetes.csv"
PROCESSED_TRAIN_PATH = PROCESSED_DIR / "clean_train.csv"
PROCESSED_TEST_PATH = PROCESSED_DIR / "clean_test.csv"
MODEL_COMPARISON_PATH = PROCESSED_DIR / "model_comparison.csv"
CLASS_PRECISION_PATH = PROCESSED_DIR / "class_precision_comparison.csv"
BOOSTING_RESULTS_PATH = INTERIM_DIR / "boosting_grid_results.csv"
BOOSTING_PARAMS_PATH = INTERIM_DIR / "best_boosting_params.json"
BOOSTING_PLOT_PATH = FIGURES_DIR / "boosting_hyperparameter_impact.png"

OFFICIAL_MODEL_PATH = MODELS_DIR / "boosting_classifier_nestimators-20_learnrate-0.001_42.sav"
BEST_MODEL_PATH = MODELS_DIR / "xgboost_diabetes_model.sav"


def ensure_directories() -> None:
    for directory in (RAW_DIR, PROCESSED_DIR, INTERIM_DIR, MODELS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_or_create_processed_splits() -> dict[str, pd.DataFrame | pd.Series]:
    if PROCESSED_TRAIN_PATH.exists() and PROCESSED_TEST_PATH.exists():
        train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
        test_df = pd.read_csv(PROCESSED_TEST_PATH)
    else:
        if RAW_DATA_PATH.exists():
            df = pd.read_csv(RAW_DATA_PATH)
        else:
            df = pd.read_csv(DATA_URL)
            df.to_csv(RAW_DATA_PATH, index=False)

        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=df[TARGET_COLUMN],
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        train_df.to_csv(PROCESSED_TRAIN_PATH, index=False)
        test_df.to_csv(PROCESSED_TEST_PATH, index=False)

    X_train = train_df.drop(columns=TARGET_COLUMN)
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=TARGET_COLUMN)
    y_test = test_df[TARGET_COLUMN]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_df": train_df,
        "test_df": test_df,
    }


def build_official_boosting_model() -> Any:
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=200,
            learning_rate=0.001,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        )
    except Exception as error:
        print(
            "Aviso: XGBoost no pudo cargarse en este entorno local. "
            "En Codespaces deberia funcionar con la dependencia xgboost. "
            f"Se usa GradientBoostingClassifier como fallback local. Detalle: {error}"
        )
        return GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.001,
            random_state=RANDOM_STATE,
        )


def tune_decision_tree(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid={
            "criterion": ["gini", "entropy"],
            "max_depth": [3, 4, 5, 6, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        scoring="accuracy",
        cv=5,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        param_grid={
            "n_estimators": [100, 200],
            "max_depth": [4, 6, None],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        },
        scoring="accuracy",
        cv=5,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def tune_boosting(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    search = GridSearchCV(
        estimator=GradientBoostingClassifier(random_state=RANDOM_STATE),
        param_grid={
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [2, 3, 4],
            "subsample": [0.8, 1.0],
        },
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def evaluate_model(model_name: str, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict[str, Any], dict[str, Any]]:
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    class_precisions = {
        "class_0_precision": report["0"]["precision"],
        "class_1_precision": report["1"]["precision"],
    }
    best_class = max(class_precisions, key=class_precisions.get).split("_")[1]
    worst_class = min(class_precisions, key=class_precisions.get).split("_")[1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision_weighted": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "class_0_precision": class_precisions["class_0_precision"],
        "class_1_precision": class_precisions["class_1_precision"],
        "best_precision_class": best_class,
        "worst_precision_class": worst_class,
    }
    return metrics, report


def save_boosting_artifacts(search: GridSearchCV, official_model: Any) -> pd.DataFrame:
    results_df = pd.DataFrame(search.cv_results_).sort_values("mean_test_score", ascending=False)
    results_df.to_csv(BOOSTING_RESULTS_PATH, index=False)

    with BOOSTING_PARAMS_PATH.open("w", encoding="utf-8") as handler:
        json.dump(search.best_params_, handler, indent=2)

    with BEST_MODEL_PATH.open("wb") as handler:
        pickle.dump(official_model, handler)

    with OFFICIAL_MODEL_PATH.open("wb") as handler:
        pickle.dump(official_model, handler)

    return results_df


def plot_boosting_results(results_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    heatmap_df = (
        results_df.groupby(["param_learning_rate", "param_n_estimators"])["mean_test_score"]
        .max()
        .unstack()
        .sort_index()
    )

    depth_df = (
        results_df.groupby("param_max_depth")["mean_test_score"]
        .max()
        .reset_index()
        .sort_values("param_max_depth")
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[0])
    axes[0].set_title("Mejor accuracy CV por learning rate y n_estimators")
    axes[0].set_xlabel("n_estimators")
    axes[0].set_ylabel("learning_rate")

    sns.barplot(
        data=depth_df,
        x="param_max_depth",
        y="mean_test_score",
        hue="param_max_depth",
        palette="crest",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Mejor accuracy CV por max_depth")
    axes[1].set_xlabel("max_depth")
    axes[1].set_ylabel("accuracy media en validacion")

    fig.tight_layout()
    fig.savefig(BOOSTING_PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compare_models(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict[str, Any]:
    official_boosting = build_official_boosting_model()
    official_boosting.fit(X_train, y_train)

    tree_search = tune_decision_tree(X_train, y_train)
    forest_search = tune_random_forest(X_train, y_train)
    boosting_search = tune_boosting(X_train, y_train)

    boosting_results = save_boosting_artifacts(boosting_search, official_boosting)
    plot_boosting_results(boosting_results)

    models = {
        "decision_tree": tree_search.best_estimator_,
        "random_forest": forest_search.best_estimator_,
        "official_xgboost_solution": official_boosting,
        "tuned_gradient_boosting": boosting_search.best_estimator_,
    }

    metrics_rows: list[dict[str, Any]] = []
    precision_rows: list[dict[str, Any]] = []
    reports: dict[str, dict[str, Any]] = {}

    for model_name, model in models.items():
        metrics, report = evaluate_model(model_name, model, X_test, y_test)
        metrics_rows.append(metrics)
        reports[model_name] = report
        precision_rows.append(
            {
                "model": model_name,
                "precision_class_0": report["0"]["precision"],
                "precision_class_1": report["1"]["precision"],
            }
        )

    comparison_df = pd.DataFrame(metrics_rows).sort_values("accuracy", ascending=False)
    precision_df = pd.DataFrame(precision_rows).sort_values("model")
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    precision_df.to_csv(CLASS_PRECISION_PATH, index=False)

    return {
        "comparison_df": comparison_df,
        "precision_df": precision_df,
        "reports": reports,
        "tree_search": tree_search,
        "forest_search": forest_search,
        "boosting_search": boosting_search,
        "official_boosting": official_boosting,
        "boosting_results_df": boosting_results,
    }


def run_full_pipeline(verbose: bool = True) -> dict[str, Any]:
    ensure_directories()
    split_output = load_or_create_processed_splits()
    comparison_output = compare_models(
        split_output["X_train"],
        split_output["X_test"],
        split_output["y_train"],
        split_output["y_test"],
    )

    output = {**split_output, **comparison_output}

    if verbose:
        best_row = output["comparison_df"].iloc[0]
        official_row = output["comparison_df"].query("model == 'official_xgboost_solution'").iloc[0]
        print("Processed train/test datasets loaded from data/processed.")
        print("Official solution model: XGBClassifier(n_estimators=200, learning_rate=0.001, random_state=42)")
        print(f"Official solution accuracy: {official_row['accuracy']:.4f}")
        print(f"Best tuned boosting params: {output['boosting_search'].best_params_}")
        print("\nModel comparison:")
        print(output["comparison_df"].to_string(index=False))
        print(
            "\nSelected best model by accuracy:",
            best_row["model"],
            f"with accuracy={best_row['accuracy']:.4f}",
        )
        print(f"\nOfficial-compatible model saved to: {OFFICIAL_MODEL_PATH}")
        print(f"Boosting plot saved to: {BOOSTING_PLOT_PATH}")

    return output


def main() -> None:
    run_full_pipeline(verbose=True)


if __name__ == "__main__":
    main()
