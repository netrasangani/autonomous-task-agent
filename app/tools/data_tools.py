import pandas as pd


def inspect_dataset(file_path: str) -> dict:
    """
    Perform a comprehensive structural inspection of a CSV dataset.
    """

    df = pd.read_csv(file_path)

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(
        exclude="number"
    ).columns.tolist()

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),

        "data_types": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },

        "missing_values": {
            column: int(count)
            for column, count in df.isnull().sum().items()
        },

        "unique_counts": {
            column: int(count)
            for column, count in df.nunique(dropna=True).items()
        },

        "duplicate_rows": int(df.duplicated().sum()),

        "numeric_columns": numeric_columns,

        "categorical_columns": categorical_columns,

        "numeric_summary": (
            df[numeric_columns]
            .describe()
            .round(4)
            .to_dict()
            if numeric_columns
            else {}
        ),

        "categorical_values": {
            column: {
                str(value): int(count)
                for value, count in df[column]
                .value_counts(dropna=False)
                .items()
            }
            for column in categorical_columns
        },

        "sample_rows": df.head(5).fillna("").to_dict(
            orient="records"
        ),
    }


def analyze_target(
    file_path: str,
    target_column: str
) -> dict:
    """
    Analyze a candidate target column and determine its
    classification characteristics.
    """

    df = pd.read_csv(file_path)

    if target_column not in df.columns:
        return {
            "error": (
                f"Target column '{target_column}' "
                "does not exist."
            )
        }

    target = df[target_column]

    class_counts = target.value_counts(
        dropna=False
    ).to_dict()

    total = len(target)

    class_percentages = {
        str(value): round(
            (count / total) * 100, 2
        )
        for value, count in class_counts.items()
    }

    return {
        "target_column": target_column,
        "data_type": str(target.dtype),

        "unique_values": [
            str(value)
            for value in target.dropna().unique()
        ],

        "number_of_classes": int(
            target.nunique(dropna=True)
        ),

        "class_counts": {
            str(value): int(count)
            for value, count in class_counts.items()
        },

        "class_percentages": class_percentages,

        "missing_values": int(
            target.isnull().sum()
        ),

        "is_categorical": (
            target.dtype == "object"
            or str(target.dtype).startswith("category")
        ),

        "is_numeric": pd.api.types.is_numeric_dtype(
            target
        ),
    }


def compare_target_candidates(
    file_path: str
) -> dict:
    """
    Identify potentially useful target columns based on
    column type and cardinality.
    """

    df = pd.read_csv(file_path)

    candidates = []

    for column in df.columns:
        unique_count = int(
            df[column].nunique(dropna=True)
        )

        missing_count = int(
            df[column].isnull().sum()
        )

        dtype = str(df[column].dtype)

        if (
            df[column].dtype == "object"
            and 2 <= unique_count <= 20
        ):
            task_type = "classification"
        elif (
            pd.api.types.is_numeric_dtype(df[column])
            and unique_count > 5
        ):
            task_type = "regression"
        else:
            task_type = "possible_feature"

        candidates.append(
            {
                "column": column,
                "data_type": dtype,
                "unique_values": unique_count,
                "missing_values": missing_count,
                "possible_task": task_type,
            }
        )

    return {
        "target_candidates": candidates
    }