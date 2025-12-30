import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_FEATURES = [
    "weight_kg",
    "height_cm",
    "age",
    "sex",
    "waist_cm",
    "hip_cm",
    "steps",
    "training_minutes",
    "calories",
]

DEFAULT_TARGETS = [
    "body_fat_pct",
    "lean_mass_kg",
    "bmr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV with data")
    parser.add_argument("--out", required=True, help="Path to save model joblib")
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        help="List of feature columns",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_TARGETS,
        help="List of target columns",
    )
    return parser.parse_args()


def build_pipeline(cat_features: list[str], num_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", "passthrough", num_features),
        ]
    )

    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=300,
            random_state=42,
        )
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if "date" in df.columns:
        df = df.sort_values("date")

    features = args.features
    targets = args.targets

    missing_features = [c for c in features if c not in df.columns]
    missing_targets = [c for c in targets if c not in df.columns]
    if missing_features or missing_targets:
        raise ValueError(
            f"Missing columns. Features: {missing_features}. Targets: {missing_targets}."
        )

    X = df[features]
    y = df[targets]

    cat_features = [c for c in features if X[c].dtype == "object"]
    num_features = [c for c in features if c not in cat_features]

    if len(df) >= 10:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
    else:
        X_train, y_train = X, y
        X_val, y_val = X, y

    pipeline = build_pipeline(cat_features, num_features)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_val)
    mae = mean_absolute_error(y_val, preds, multioutput="raw_values")
    mae_report = dict(zip(targets, mae))

    artifact = {
        "model": pipeline,
        "features": features,
        "targets": targets,
        "mae": mae_report,
    }

    joblib.dump(artifact, out_path)
    print(f"Saved model to {out_path}")
    print(f"Validation MAE: {mae_report}")


if __name__ == "__main__":
    main()
