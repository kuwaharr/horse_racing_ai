import json

from ..data.data_path import RAW_DIR


def combine_race_dict(race_info: dict, race_meta: dict) -> dict:
    race = {
        "race_id": race_info["race_id"],
        "date": race_meta["date"],
        "track": race_info["track"],
        "race_number": race_info["race_number"],
        "post_time": race_meta["post_time"],
        "surface": race_meta["surface"],
        "distance": race_meta["distance"],
        "course_direction": race_meta["course_direction"],
        "course_layout": race_meta["course_layout"],
        "course_variant": race_meta["course_variant"],
        "weather": race_meta["weather"],
        "track_condition": race_meta["track_condition"],
        "race_size": race_meta["race_size"],
    }
    return race


def export_json(race: dict, runners: list, odds: list) -> None:
    data = {
        "race": race,
        "runners": runners,
        "odds": odds,
    }

    data_path = RAW_DIR / f"{data['race']['race_id']}.json"

    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)