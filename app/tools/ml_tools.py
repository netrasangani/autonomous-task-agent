import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def _build_preprocessor(X):
    numeric_columns = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ]
    )

    return preprocessor


def train_classifier(
    file_path: str,
    target_column: str,
) -> dict:
    """
    Train and compare several classification models
    using a robust preprocessing pipeline.
    """

    df = pd.read_csv(file_path)

    if target_column not in df.columns:
        return {
            "error": (
                f"Target column '{target_column}' "
                "does not exist."
            )
        }

    if len(df) < 10:
        return {
            "error": (
                "Dataset is too small for reliable "
                "classification evaluation."
            )
        }

    df = df.dropna(subset=[target_column])

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if y.nunique() < 2:
        return {
            "error": (
                "Classification requires at least "
                "two target classes."
            )
        }

    class_counts = y.value_counts()

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "DecisionTree": DecisionTreeClassifier(
            random_state=42,
            max_depth=5,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
    }

    results = {}

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                (
                    "preprocessing",
                    _build_preprocessor(X_train),
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

        try:
            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)

            results[model_name] = {
                "accuracy": round(
                    float(
                        accuracy_score(
                            y_test,
                            predictions,
                        )
                    ),
                    4,
                ),
                "balanced_accuracy": round(
                    float(
                        balanced_accuracy_score(
                            y_test,
                            predictions,
                        )
                    ),
                    4,
                ),
                "precision": round(
                    float(
                        precision_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0,
                        )
                    ),
                    4,
                ),
                "recall": round(
                    float(
                        recall_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0,
                        )
                    ),
                    4,
                ),
                "f1": round(
                    float(
                        f1_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0,
                        )
                    ),
                    4,
                ),
            }

        except Exception as exc:
            results[model_name] = {
                "error": str(exc)
            }

    valid_results = {
        name: result
        for name, result in results.items()
        if "accuracy" in result
    }

    if valid_results:
        best_model = max(
            valid_results,
            key=lambda name: valid_results[name][
                "balanced_accuracy"
            ],
        )
    else:
        best_model = None

    majority_class = class_counts.index[0]
    majority_accuracy = (
        class_counts.iloc[0] / len(y)
    )

    return {
        "task": "classification",
        "target": target_column,
        "rows": len(df),
        "features": X.columns.tolist(),
        "class_counts": {
            str(key): int(value)
            for key, value in class_counts.items()
        },
        "number_of_classes": int(y.nunique()),
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "majority_class": str(majority_class),
        "majority_baseline_accuracy": round(
            float(majority_accuracy),
            4,
        ),
        "models": results,
        "best_model": best_model,
        "evaluation_note": (
            "Results are based on a small holdout set "
            "and should not be treated as reliable "
            "generalization estimates."
        ),
    }


def train_regressor(
    file_path: str,
    target_column: str,
) -> dict:
    """
    Train and compare regression models.
    """

    df = pd.read_csv(file_path)

    if target_column not in df.columns:
        return {
            "error": (
                f"Target column '{target_column}' "
                "does not exist."
            )
        }

    df = df.dropna(subset=[target_column])

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if not pd.api.types.is_numeric_dtype(y):
        return {
            "error": (
                "Regression requires a numeric "
                "target column."
            )
        }

    if len(df) < 10:
        return {
            "error": (
                "Dataset is too small for reliable "
                "regression evaluation."
            )
        }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    models = {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
        ),
    }

    results = {}

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                (
                    "preprocessing",
                    _build_preprocessor(X_train),
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

        try:
            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)

            mse = mean_squared_error(
                y_test,
                predictions,
            )

            results[model_name] = {
                "mae": round(
                    float(
                        mean_absolute_error(
                            y_test,
                            predictions,
                        )
                    ),
                    4,
                ),
                "rmse": round(
                    float(mse ** 0.5),
                    4,
                ),
                "r2": round(
                    float(
                        r2_score(
                            y_test,
                            predictions,
                        )
                    ),
                    4,
                ),
            }

        except Exception as exc:
            results[model_name] = {
                "error": str(exc)
            }

    return {
        "task": "regression",
        "target": target_column,
        "rows": len(df),
        "features": X.columns.tolist(),
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "models": results,
        "evaluation_note": (
            "Results are based on a small holdout set "
            "and should not be treated as reliable "
            "generalization estimates."
        ),
    }