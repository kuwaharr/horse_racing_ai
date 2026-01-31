from pathlib import Path

import lightgbm as lgb
import pandas as pd


def show_importance(model_path: Path) -> None:
    model = lgb.Booster(model_file=model_path)

    imp = pd.DataFrame({
        "feature": model.feature_name(),
        "importance": model.feature_importance(importance_type="gain")
    })

    imp = imp.sort_values("importance", ascending=False)

    print(imp.head(30))