import lightgbm as lgb
import pandas as pd

from src.common.config import DATA_ROOT


def main():
    model_path = DATA_ROOT / "model" / "lgbm_place_rule.txt"
    model = lgb.Booster(model_file=model_path)

    imp = pd.DataFrame({
        "feature": model.feature_name(),
        "importance": model.feature_importance(importance_type="gain")
    })

    imp = imp.sort_values("importance", ascending=False)

    print(imp.head(30))


if __name__ == "__main__":
    main()