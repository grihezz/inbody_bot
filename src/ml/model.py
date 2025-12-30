from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class InBodyModel:
    def __init__(self, artifact: dict[str, Any]) -> None:
        self.model = artifact["model"]
        self.features = artifact["features"]
        self.targets = artifact["targets"]
        self.mae = artifact.get("mae", {})

    @classmethod
    def load(cls, path: str | Path) -> "InBodyModel":
        artifact = joblib.load(path)
        return cls(artifact)

    def predict(self, features: dict[str, Any]) -> dict[str, float]:
        missing = [c for c in self.features if c not in features]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        row = {k: features[k] for k in self.features}
        df = pd.DataFrame([row])
        preds = self.model.predict(df)[0]
        return dict(zip(self.targets, [float(x) for x in preds]))
