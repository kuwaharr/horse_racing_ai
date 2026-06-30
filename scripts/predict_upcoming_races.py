import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.maps import SEX_MAP, STABLE_MAP
from src.data.paths import DB_PATH
from src.features.place_top3 import (
    HISTORY_FEATURE_COLUMNS,
    PEDIGREE_FEATURE_COLUMNS,
    PEDIGREE_HISTORY_FEATURE_COLUMNS,
    PLACE_TOP3_BASE_QUERY,
    TRAINING_FEATURE_COLUMNS,
    _append_history_features,
)
from src.scrape.extracters import extract_race_ids, parse_url
from src.scrape.fetchers import make_soup


LOCAL_MODEL_DIR = ROOT_DIR / "local_models"
DEFAULT_PLACE_MODEL = LOCAL_MODEL_DIR / "catboost_place_top3_model.cbm"
DEFAULT_PLACE_METADATA = LOCAL_MODEL_DIR / "catboost_place_top3_model_metadata.json"
DEFAULT_WIN_MODEL = LOCAL_MODEL_DIR / "catboost_win_top1_model.cbm"
DEFAULT_WIN_METADATA = LOCAL_MODEL_DIR / "catboost_win_top1_model_metadata.json"
DEFAULT_OUTPUT = LOCAL_MODEL_DIR / "live_predictions.csv"
OUTPUT_COLUMNS = [
    "race_id",
    "date",
    "track_id",
    "race_number",
    "post_time_min",
    "surface_id",
    "distance",
    "race_size",
    "horse_number",
    "horse_name",
    "horse_id",
    "jockey_id",
    "trainer_id",
    "pred",
    "odds_min",
    "odds_max",
    "odds_mid",
    "time_bucket",
    "snapshot_at",
    "pred_rank",
    "ev_mid",
]


PLACE_RULE = {
    "pred_min": 0.34,
    "odds_min": 3.2,
    "odds_max": 6.0,
    "distance_min": 1200,
    "distance_max": None,
    "exclude_track_ids": [3, 7, 10],
    "pred_rank_max": 5,
    "ev_mid_min": 1.4,
}

WIN_RULE = {
    "pred_min": 0.15,
    "odds_min": 1.2,
    "odds_max": 3.5,
    "distance_min": 1600,
    "distance_max": None,
    "include_track_ids": [4, 5, 6, 8, 9],
    "surface_id": 0,
    "pred_rank_max": 3,
}

RULE_KEYS = {
    "pred_min",
    "odds_min",
    "odds_max",
    "distance_min",
    "distance_max",
    "include_track_ids",
    "exclude_track_ids",
    "surface_id",
    "pred_rank_max",
    "ev_mid_min",
}


def race_list_url_for_date(value: date) -> str:
    return f"https://race.netkeiba.com/top/race_list.html?kaisai_date={value:%Y%m%d}"


def shutuba_url(race_id: str) -> str:
    return f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}&rf=race_submenu"


def normalize_race_date(race_id: str, raw_date: str | None) -> str | None:
    if raw_date is None:
        return None
    m = re.match(r"(\d{1,2})月(\d{1,2})日.*", raw_date)
    if not m:
        return None
    return f"{race_id[:4]}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"


def extract_shutuba_race_meta(soup) -> dict:
    date_tag = soup.select_one("#RaceList_DateList dd.Active")
    if date_tag is None:
        raise RuntimeError("Could not find active race date")
    race_data_01 = soup.select_one(".RaceColumn01 .RaceData01")
    if race_data_01 is None:
        raise RuntimeError("Could not find RaceData01")

    parts = [part.strip() for part in race_data_01.get_text(" ", strip=True).split("/") if part.strip()]
    post_time = None
    surface = None
    distance = None
    course_direction = None
    course_layout = None
    course_variant = None
    weather = None
    track_condition = None
    for part in parts:
        text = "".join(part.split())
        if "発走" in text:
            m = re.search(r"(\d{1,2}:\d{2})発走", text)
            post_time = None if m is None else m.group(1)
        elif "m" in text:
            m = re.search(r"(芝|ダ)(\d{3,4})m(?:\(([^)]*)\))?", text)
            if m is None:
                raise RuntimeError(f"Unsupported race course text: {text}")
            surface = m.group(1)
            distance = m.group(2)
            course_info = m.group(3) or ""
            if course_info.startswith("直線"):
                course_direction = "直線"
                course_info = course_info[2:]
            elif course_info:
                course_direction = course_info[0]
                course_info = course_info[1:]
            for letter in course_info:
                if letter in ["外", "内"]:
                    course_layout = letter
                elif letter in ["A", "B", "C", "D"]:
                    course_variant = letter
        elif text.startswith("天候"):
            weather = text.split(":", 1)[-1]
        elif text.startswith("馬場") and track_condition is None:
            track_condition = text.split(":", 1)[-1]

    race_size = None
    race_data_02 = soup.select_one(".RaceColumn01 .RaceData02")
    if race_data_02 is not None:
        m = re.search(r"(\d+)頭", race_data_02.get_text("", strip=True))
        race_size = None if m is None else m.group(1)

    return {
        "date": date_tag.get_text(strip=True),
        "post_time": post_time,
        "surface": surface,
        "distance": distance,
        "course_direction": course_direction,
        "course_layout": course_layout,
        "course_variant": course_variant,
        "weather": weather,
        "track_condition": track_condition,
        "race_size": race_size,
    }


def _clean_int(value: str | None) -> int | None:
    if value is None:
        return None
    m = re.search(r"\d+", value.replace(",", ""))
    return None if m is None else int(m.group(0))


def _clean_float(value: str | None) -> float | None:
    if value is None:
        return None
    m = re.search(r"\d+(?:\.\d+)?", value.replace(",", ""))
    return None if m is None else float(m.group(0))


def _link_id(href: str | None, pattern: str) -> str | None:
    if href is None:
        return None
    m = re.search(pattern, href)
    return None if m is None else m.group(1)


def _cell_text(cells, idx: int | None) -> str | None:
    if idx is None or idx >= len(cells):
        return None
    text = cells[idx].get_text(strip=True)
    return text if text else None


def extract_shutuba_runners(soup) -> list[dict]:
    table = soup.find("table", class_="RaceTable01")
    if table is None:
        raise RuntimeError("Could not find RaceTable01 on shutuba page")
    headers = [th.get_text(strip=True).replace(" ", "") for th in table.find("tr").find_all("th")]
    col_idx = {name: idx for idx, name in enumerate(headers)}

    def idx(*names: str) -> int | None:
        for name in names:
            if name in col_idx:
                return col_idx[name]
        return None

    rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if not cells:
            continue
        horse_number = _clean_int(_cell_text(cells, idx("馬番")))
        horse_name = _cell_text(cells, idx("馬名"))
        if horse_number is None or horse_name is None:
            continue
        sex_age = _cell_text(cells, idx("性齢"))
        sex_id = None
        age = None
        if sex_age is not None:
            m = re.match(r"(\D+)(\d{1,2})", sex_age)
            if m:
                sex_id = SEX_MAP.get(m.group(1))
                age = int(m.group(2))

        stable_id = None
        trainer_id = None
        trainer_raw = _cell_text(cells, idx("厩舎"))
        if trainer_raw:
            m = re.match(r"(美浦|栗東|地方)(.+)", trainer_raw)
            if m:
                stable_id = STABLE_MAP.get(m.group(1))

        horse_a = cells[idx("馬名")].find("a", href=True) if idx("馬名") is not None else None
        jockey_cell_idx = idx("騎手")
        jockey_a = cells[jockey_cell_idx].find("a", href=True) if jockey_cell_idx is not None else None
        trainer_cell_idx = idx("厩舎")
        trainer_a = cells[trainer_cell_idx].find("a", href=True) if trainer_cell_idx is not None else None
        if trainer_a is not None:
            trainer_id = _link_id(trainer_a.get("href"), r"trainer/(?:result/recent/)?(\d+)")

        rows.append(
            {
                "gate": _clean_int(_cell_text(cells, idx("枠"))),
                "horse_number": horse_number,
                "horse_name": horse_name,
                "horse_id": None if horse_a is None else _link_id(horse_a.get("href"), r"horse/(\d+)"),
                "sex_id": sex_id,
                "age": age,
                "jockey_id": None if jockey_a is None else _link_id(jockey_a.get("href"), r"jockey/(?:result/recent/)?(\d+)"),
                "weight": _clean_float(_cell_text(cells, idx("斤量"))),
                "stable_id": stable_id,
                "trainer_id": trainer_id,
            }
        )
    return sorted(rows, key=lambda row: row["horse_number"])


def normalize_race_meta(race_id: str, soup) -> dict:
    race_url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}&rf=race_submenu"
    parsed = parse_url(race_url)
    if not parsed.success:
        raise RuntimeError(parsed.error)
    raw = {**parsed.value, **extract_shutuba_race_meta(soup)}
    race_date = normalize_race_date(race_id, raw["date"])
    if raw["post_time"] is not None:
        hour, minute = raw["post_time"].split(":")
        post_time_min = int(hour) * 60 + int(minute)
    else:
        post_time_min = None
    from src.common.maps import (
        CRS_DIRECTION_MAP,
        CRS_LAYOUT_MAP,
        CRS_VARIANT_MAP,
        SURFACE_MAP,
        TRACK_COND_MAP,
        TRACK_MAP,
        WEATHER_MAP,
    )

    return {
        "race_id": race_id,
        "date": race_date,
        "track_id": TRACK_MAP[raw["track"]],
        "race_number": int(raw["race_number"]),
        "post_time_min": post_time_min,
        "surface_id": None if raw["surface"] is None else SURFACE_MAP[raw["surface"]],
        "distance": None if raw["distance"] is None else int(raw["distance"]),
        "course_direction_id": None if raw["course_direction"] is None else CRS_DIRECTION_MAP.get(raw["course_direction"]),
        "course_layout_id": None if raw["course_layout"] is None else CRS_LAYOUT_MAP.get(raw["course_layout"]),
        "course_variant_id": None if raw["course_variant"] is None else CRS_VARIANT_MAP.get(raw["course_variant"]),
        "weather_id": None if raw["weather"] is None else WEATHER_MAP.get(raw["weather"]),
        "track_condition_id": None if raw["track_condition"] is None else TRACK_COND_MAP.get(raw["track_condition"]),
        "race_size": None if raw["race_size"] is None else int(raw["race_size"]),
    }


def fetch_race_ids_for_date(value: date) -> list[str]:
    result = make_soup(race_list_url_for_date(value))
    if not result.success:
        raise RuntimeError(result.error)
    return sorted(race_id for race_id in extract_race_ids(result.value) if len(race_id) == 12)


def fetch_live_base_rows(race_ids: list[str], db_path: Path) -> list[dict]:
    import sqlite3

    live_rows = []
    with sqlite3.connect(db_path) as conn:
        horse_table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'horse'"
        ).fetchone()
        if horse_table_exists:
            horse_pedigrees = {
                row[0]: row
                for row in conn.execute(
                    """
                    SELECT horse_id, sire_id, dam_id, broodmare_sire_id, pedigree_fetch_status
                    FROM horse
                    """
                ).fetchall()
            }
        else:
            horse_pedigrees = {}

    for race_id in race_ids:
        result = make_soup(shutuba_url(race_id))
        if not result.success:
            raise RuntimeError(f"race_id={race_id} fetch failed: {result.error}")
        soup = result.value
        try:
            race = normalize_race_meta(race_id, soup)
        except Exception as e:
            print(f"Skip race_id={race_id}: {e}", file=sys.stderr)
            continue
        runners = extract_shutuba_runners(soup)
        for runner in runners:
            pedigree = horse_pedigrees.get(runner.get("horse_id"))
            row = {
                **race,
                **runner,
                "sire_id": None if pedigree is None else pedigree[1],
                "dam_id": None if pedigree is None else pedigree[2],
                "broodmare_sire_id": None if pedigree is None else pedigree[3],
                "pedigree_available": 1 if pedigree is not None and pedigree[4] == "fetched" else 0,
                "time_sec": None,
                "finish_diff": None,
                "popularity": None,
                "finish_3f": None,
                "corner_4": None,
                "horse_weight": None,
                "horse_weight_diff": None,
                "finish_position": 99,
                "win_odds": None,
                "place_odds_min": None,
                "place_odds_max": None,
                "target_top3": 0,
            }
            live_rows.append(row)
    return live_rows


def build_live_features(db_path: Path, live_rows: list[dict], profile: str):
    import pandas as pd
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        historical = pd.read_sql_query(PLACE_TOP3_BASE_QUERY, conn)
    live_df = pd.DataFrame(live_rows)
    base_columns = historical.columns.tolist()
    historical = historical.copy()
    historical["_live_row"] = False
    live_for_concat = live_df[base_columns].copy()
    live_for_concat["_live_row"] = True
    combined = pd.concat([historical, live_for_concat], ignore_index=True)
    featured = _append_history_features(combined)
    live_featured = featured[featured["_live_row"]].copy()
    if profile == "win":
        live_featured["target_top3"] = 0

    columns = TRAINING_FEATURE_COLUMNS
    horse_name_index = columns.index("horse_name")
    columns = (
        columns[: horse_name_index + 1]
        + PEDIGREE_FEATURE_COLUMNS
        + columns[horse_name_index + 1 :]
    )
    history_columns = HISTORY_FEATURE_COLUMNS + PEDIGREE_HISTORY_FEATURE_COLUMNS
    columns = columns[:-1] + history_columns + columns[-1:]
    return live_featured[columns]


def latest_pre_race_odds(db_path: Path, race_ids: list[str], profile: str):
    import pandas as pd
    import sqlite3

    bet_type = "win" if profile == "win" else "place"
    placeholders = ",".join(["?"] * len(race_ids))
    with sqlite3.connect(db_path) as conn:
        snapshot_table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'pre_race_odds_snapshot'"
        ).fetchone()
        if not snapshot_table_exists:
            return pd.DataFrame(
                columns=["race_id", "horse_number", "odds_min", "odds_max", "odds_mid", "time_bucket", "snapshot_at"]
            )
        snapshots = pd.read_sql_query(
            f"""
            SELECT *
            FROM pre_race_odds_snapshot
            WHERE race_id IN ({placeholders})
              AND bet_type = ?
              AND status = 'fetched'
            ORDER BY snapshot_at
            """,
            conn,
            params=[*race_ids, bet_type],
        )
        if snapshots.empty:
            return pd.DataFrame(
                columns=["race_id", "horse_number", "odds_min", "odds_max", "odds_mid", "time_bucket", "snapshot_at"]
            )
        latest = snapshots.sort_values("snapshot_at").drop_duplicates(["race_id", "bet_type"], keep="last")
        snapshot_ids = latest["snapshot_id"].tolist()
        id_placeholders = ",".join(["?"] * len(snapshot_ids))
        odds_table = "pre_race_win_odds" if profile == "win" else "pre_race_place_odds"
        odds_table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (odds_table,),
        ).fetchone()
        if not odds_table_exists:
            return pd.DataFrame(
                columns=["race_id", "horse_number", "odds_min", "odds_max", "odds_mid", "time_bucket", "snapshot_at"]
            )
        if profile == "win":
            odds = pd.read_sql_query(
                f"SELECT snapshot_id, horse_number, odds FROM pre_race_win_odds WHERE snapshot_id IN ({id_placeholders})",
                conn,
                params=snapshot_ids,
            )
            odds["odds_min"] = odds["odds"]
            odds["odds_max"] = odds["odds"]
        else:
            odds = pd.read_sql_query(
                f"SELECT snapshot_id, horse_number, odds_min, odds_max FROM pre_race_place_odds WHERE snapshot_id IN ({id_placeholders})",
                conn,
                params=snapshot_ids,
            )
        merged = odds.merge(latest[["snapshot_id", "race_id", "time_bucket", "snapshot_at"]], on="snapshot_id")
        merged["odds_mid"] = (merged["odds_min"] + merged["odds_max"]) / 2
        return merged


def prepare_model_input(features, metadata: dict):
    feature_cols = metadata["feature_cols"]
    missing = [col for col in feature_cols if col not in features.columns]
    if missing:
        raise RuntimeError(f"Live features missing model columns: {missing[:20]}")
    x = features[feature_cols].copy()
    categorical_cols = [col for col in metadata["categorical_cols"] if col in x.columns]
    for col in categorical_cols:
        x[col] = x[col].fillna("__MISSING__").astype("string")
    return x


def _optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null", ""}:
        return None
    return float(value)


def _optional_int(value: str) -> int | None:
    if value.lower() in {"none", "null", ""}:
        return None
    return int(value)


def _optional_int_list(value: str) -> list[int] | None:
    if value.lower() in {"none", "null", ""}:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def load_buy_rule(profile: str, rule_json: Path | None, overrides: dict) -> dict:
    rule = dict(WIN_RULE if profile == "win" else PLACE_RULE)
    if rule_json is not None:
        loaded = json.loads(rule_json.read_text(encoding="utf-8"))
        unknown = sorted(set(loaded) - RULE_KEYS)
        if unknown:
            raise RuntimeError(f"Unknown rule keys in {rule_json}: {unknown}")
        rule.update(loaded)
    rule.update({key: value for key, value in overrides.items() if value is not None})
    return rule


def apply_buy_rule(predictions, rule: dict):
    selected = predictions.copy()
    selected["pred_rank"] = selected.groupby("race_id")["pred"].rank(method="first", ascending=False)
    if "odds_mid" in selected.columns:
        selected["ev_mid"] = selected["pred"] * selected["odds_mid"]
    else:
        selected["ev_mid"] = None
    mask = selected["pred"] >= rule["pred_min"]
    if rule.get("distance_min") is not None:
        mask &= selected["distance"] >= rule["distance_min"]
    if rule.get("distance_max") is not None:
        mask &= selected["distance"] < rule["distance_max"]
    if rule.get("include_track_ids") is not None:
        mask &= selected["track_id"].isin(rule["include_track_ids"])
    if rule.get("exclude_track_ids") is not None:
        mask &= ~selected["track_id"].isin(rule["exclude_track_ids"])
    if rule.get("surface_id") is not None:
        mask &= selected["surface_id"] == rule["surface_id"]
    if rule.get("pred_rank_max") is not None:
        mask &= selected["pred_rank"] <= rule["pred_rank_max"]
    if "odds_mid" in selected.columns:
        mask &= selected["odds_mid"] >= rule["odds_min"]
        mask &= selected["odds_mid"] < rule["odds_max"]
    else:
        mask &= False
    if rule.get("ev_mid_min") is not None:
        mask &= selected["ev_mid"] >= rule["ev_mid_min"]
    return selected[mask].copy()


def expand_race_ids(values: list[str] | None) -> list[str]:
    if not values:
        return []
    race_ids = []
    for value in values:
        race_ids.extend(part.strip() for part in value.split(",") if part.strip())
    return race_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict upcoming races from netkeiba shutuba pages.")
    parser.add_argument("--profile", choices=["place", "win"], default="place")
    parser.add_argument("--race-id", action="append")
    parser.add_argument("--date", type=lambda value: date.fromisoformat(value), default=None)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rule-json", type=Path, default=None)
    parser.add_argument("--rule-pred-min", type=_optional_float, default=None)
    parser.add_argument("--rule-odds-min", type=_optional_float, default=None)
    parser.add_argument("--rule-odds-max", type=_optional_float, default=None)
    parser.add_argument("--rule-distance-min", type=_optional_int, default=None)
    parser.add_argument("--rule-distance-max", type=_optional_int, default=None)
    parser.add_argument("--rule-include-track-ids", type=_optional_int_list, default=None)
    parser.add_argument("--rule-exclude-track-ids", type=_optional_int_list, default=None)
    parser.add_argument("--rule-surface-id", type=_optional_int, default=None)
    parser.add_argument("--rule-pred-rank-max", type=_optional_int, default=None)
    parser.add_argument("--rule-ev-mid-min", type=_optional_float, default=None)
    args = parser.parse_args()
    if not args.race_id and args.date is None:
        parser.error("--race-id or --date is required")

    try:
        from catboost import CatBoostClassifier
    except ModuleNotFoundError as e:
        raise RuntimeError("予測には catboost が必要です。") from e

    race_ids = expand_race_ids(args.race_id)
    if args.date is not None:
        race_ids.extend(fetch_race_ids_for_date(args.date))
    race_ids = sorted(set(race_ids))
    if not race_ids:
        raise RuntimeError("No race_ids found")

    model_path = args.model or (DEFAULT_WIN_MODEL if args.profile == "win" else DEFAULT_PLACE_MODEL)
    metadata_path = args.metadata or (DEFAULT_WIN_METADATA if args.profile == "win" else DEFAULT_PLACE_METADATA)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("profile") != args.profile:
        raise RuntimeError(f"Model metadata profile mismatch: {metadata.get('profile')} != {args.profile}")
    buy_rule = load_buy_rule(
        args.profile,
        args.rule_json,
        {
            "pred_min": args.rule_pred_min,
            "odds_min": args.rule_odds_min,
            "odds_max": args.rule_odds_max,
            "distance_min": args.rule_distance_min,
            "distance_max": args.rule_distance_max,
            "include_track_ids": args.rule_include_track_ids,
            "exclude_track_ids": args.rule_exclude_track_ids,
            "surface_id": args.rule_surface_id,
            "pred_rank_max": args.rule_pred_rank_max,
            "ev_mid_min": args.rule_ev_mid_min,
        },
    )

    live_rows = fetch_live_base_rows(race_ids, args.db)
    if not live_rows:
        raise RuntimeError("No supported runners found. Unsupported races, such as obstacle races, are skipped.")
    features = build_live_features(args.db, live_rows, args.profile)
    x = prepare_model_input(features, metadata)
    model = CatBoostClassifier()
    model.load_model(model_path)
    features = features.copy()
    features["pred"] = model.predict_proba(x)[:, 1]
    odds = latest_pre_race_odds(args.db, race_ids, args.profile)
    all_predictions = features.merge(odds, on=["race_id", "horse_number"], how="left")
    all_predictions["pred_rank"] = all_predictions.groupby("race_id")["pred"].rank(method="first", ascending=False)
    all_predictions["ev_mid"] = all_predictions["pred"] * all_predictions["odds_mid"]
    all_predictions = all_predictions.sort_values(["race_id", "pred"], ascending=[True, False])
    buys = apply_buy_rule(all_predictions, buy_rule)
    output_cols = [col for col in OUTPUT_COLUMNS if col in all_predictions.columns]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_predictions[output_cols].to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Races: {len(race_ids):,}")
    print(f"Rows: {len(all_predictions):,}")
    print(f"CSV: {args.output}")
    print("")
    print("Buy candidates")
    if buys.empty:
        print("  none")
    else:
        for row in buys.itertuples(index=False):
            odds_text = "n/a" if getattr(row, "odds_mid", None) != getattr(row, "odds_mid", None) else f"{row.odds_mid:.2f}"
            print(
                f"  race_id={row.race_id} horse={row.horse_number} {row.horse_name} "
                f"pred={row.pred:.4f} odds_mid={odds_text}"
            )


if __name__ == "__main__":
    main()
