from lightgbm import Booster

from ..data.paths import MODEL_DIR

def export_model(model: Booster, file_name: str) -> None:
    text_path = MODEL_DIR / file_name
    model.save_model(text_path)