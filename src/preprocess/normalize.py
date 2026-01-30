from __future__ import annotations

import re

from ..common.maps import (
    TRACK_MAP,
    SURFACE_MAP,
    CRS_DIRECTION_MAP,
    CRS_LAYOUT_MAP,
    CRS_VARIANT_MAP,
    WEATHER_MAP,
    TRACK_COND_MAP,
    STATUS_MAP,
    SEX_MAP,
    STABLE_MAP,
    MARGIN_MAP,
)
from ..common.result import Result


def normalize_race(race: dict) -> Result[dict]:
    race_required_keys = [
        "race_id", # value required
        "date", # value required
        "track", # value required
        "race_number", # value required
        "post_time",
        "surface",
        "distance",
        "course_direction",
        "course_layout",
        "course_variant",
        "weather",
        "track_condition",
        "race_size",
    ]
    race_missing_keys = [k for k in race_required_keys if k not in race.keys()]
    if len(race_missing_keys) > 0:
        return Result(success=False, error=f"Missing keys in race: {race_missing_keys}")

    normalized_race = {}

    if race["race_id"] is not None:
        normalized_race["race_id"] = race["race_id"]
    else:
        return Result(success=False, error="Missing race_id value")

    if race["date"] is not None:
        try:
            m_date = re.match(r"(\d{1,2})月(\d{1,2})日.*", race["date"])
            month = int(m_date.group(1))
            day = int(m_date.group(2))
            date = race["race_id"][:4] + "-" + f"{month:02d}" + "-" + f"{day:02d}"
            normalized_race["date"] = date
        except Exception as e:
            return Result(success=False, error=f"Invalid date format: {race['date']}\n{e}")
    else:
        return Result(success=False, error="Missing date value")

    if race["track"] is not None:
        try:
            normalized_race["track_id"] = TRACK_MAP[race["track"]]
        except Exception as e:
            return Result(success=False, error=f"Unknown track value: {race['track']}\n{e}")
    else:
        return Result(success=False, error="Missing track value")

    if race["race_number"] is not None:
        try:
            normalized_race["race_number"] = int(race["race_number"])
        except Exception as e:
            return Result(
                success=False,
                error=f"Invalid race_number value: {race['race_number']}\n{e}"
            )
    else:
        return Result(success=False, error="Missing race_number value")

    if race["post_time"] is not None:
        try:
            h, m = race["post_time"].split(":")
            normalized_race["post_time_min"] = int(h) * 60 + int(m)
        except Exception as e:
            return Result(
                success=False,
                error=f"Invalid post_time format: {race['post_time']}\n{e}"
            )
    else:
        normalized_race["post_time_min"] = None

    if race["surface"] is not None:
        try:
            normalized_race["surface_id"] = SURFACE_MAP[race["surface"]]
        except Exception as e:
            return Result(
                success=False,
                error=f"Unknown surface value: {race['surface']}\n{e}"
            )
    else:
        normalized_race["surface_id"] = None

    if race["distance"] is not None:
        try:
            normalized_race["distance"] = int(race["distance"])
        except Exception as e:
            return Result(
                success=False,
                error=f"Invalid distance value: {race['distance']}\n{e}"
            )
    else:
        normalized_race["distance"] = None

    if race["course_direction"] is not None:
        try:
            normalized_race["course_direction_id"] = CRS_DIRECTION_MAP[race["course_direction"]]
        except Exception as e:
            return Result(
                success=False,
                error=f"Unknown course_direction value: {race['course_direction']}\n{e}"
            )
    else:
        normalized_race["course_direction_id"] = None

    if race["course_layout"] is not None:
        try:
            normalized_race["course_layout_id"] = CRS_LAYOUT_MAP[race["course_layout"]]
        except Exception as e:
            return Result(
                success=False,
                error=f"Unknown course_layout value: {race['course_layout']}\n{e}"
            )
    else:
        normalized_race["course_layout_id"] = None

    if race["course_variant"] is not None:
        try:
            normalized_race["course_variant_id"] = CRS_VARIANT_MAP[race["course_variant"]]
        except Exception as e:
            return Result(
                success=False,
                error=f"Unknown course_variant value: {race['course_variant']}\n{e}"
            )
    else:
        normalized_race["course_variant_id"] = None

    if race["weather"] is not None:
        try:
            normalized_race["weather_id"] = WEATHER_MAP[race["weather"]]
        except Exception as e:
            return Result(
                success=False,
                error=f"Unknown weather value: {race['weather']}\n{e}"
            )
    else:
        normalized_race["weather_id"] = None

    if race["track_condition"] is not None:
        try:
            normalized_race["track_condition_id"] = TRACK_COND_MAP[race["track_condition"]]
        except Exception as e:
            return Result(
                success=False,
                error=f"Unknown track_condition value: {race['track_condition']}\n{e}"
            )
    else:
        normalized_race["track_condition_id"] = None

    if race["race_size"] is not None:
        try:
            normalized_race["race_size"] = int(race["race_size"])
        except Exception as e:
            return Result(
                success=False,
                error=f"Invalid race_size value: {race['race_size']}\n{e}"
            )
    else:
        normalized_race["race_size"] = None

    return Result(success=True, value=normalized_race)


def normalize_runners(race_id: str, runners: list) -> Result[list]:
    runners_required_keys = [
        "gate",
        "horse_number",
        "finish", # value required
        "horse_name", # value required
        "horse_id",
        "sex_age",
        "jockey_raw",
        "jockey_id",
        "weight",
        "time",
        "finish_diff",
        "popularity",
        "finish_3f",
        "corner",
        "trainer_raw",
        "trainer_id",
        "horse_weight",
    ]

    normalized_runners = []

    for row in runners:
        row_missing_keys = [k for k in runners_required_keys if k not in row.keys()]
        if len(row_missing_keys) > 0:
            return Result(success=False, error=f"Missing keys in runners row: {row_missing_keys}")

        normalized_row = {}

        normalized_row["race_id"] = race_id

        if row["gate"] is not None:
            try:
                normalized_row["gate"] = int(row["gate"])
            except Exception as e:
                return Result(
                    success=False,
                    error=f"Invalid gate value: {row['gate']}\n{e}"
                )
        else:
            normalized_row["gate"] = None

        if row["horse_number"] is not None:
            try:
                normalized_row["horse_number"] = int(row["horse_number"])
            except Exception as e:
                return Result(
                    success=False,
                    error=f"Invalid horse_number value: {row['horse_number']}\n{e}"
                )
        else:
            normalized_row["horse_number"] = None

        if row["finish"] is not None:
            if row["finish"].isdigit():
                normalized_row["finish"] = int(row["finish"])
                normalized_row["status_id"] = 0 # ok
            elif row["finish"] in STATUS_MAP.keys():
                normalized_row["finish"] = None
                normalized_row["status_id"] = STATUS_MAP[row["finish"]]
            else:
                return Result(
                    success=False,
                    error=f"Unknown finish value: {row['finish']}"
                )
        else:
            return Result(success=False, error="Missing finish value")

        if row["horse_name"] is not None:
            normalized_row["horse_name"] = row["horse_name"]
        else:
            return Result(success=False, error="Missing horse_name value")

        normalized_row["horse_id"] = row["horse_id"] if row["horse_id"] is not None else None

        if row["sex_age"] is not None:
            m_sex_age = re.match(r"(\D+)(\d{1,2})", row["sex_age"])
            if m_sex_age:
                try:
                    normalized_row["sex_id"] = SEX_MAP[m_sex_age.group(1)]
                except Exception as e:
                    return Result(
                        success=False,
                        error=f"Unknown sex value: {m_sex_age.group(1)}\n{e}"
                    )
                try:
                    normalized_row["age"] = int(m_sex_age.group(2))
                except Exception as e:
                    return Result(
                        success=False,
                        error=f"Invalid age value: {m_sex_age.group(2)}\n{e}"
                    )
            else:
                return Result(
                    success=False,
                    error=f"Invalid sex_age format: {row['sex_age']}"
                )
        else:
            normalized_row["sex_id"] = None
            normalized_row["age"] = None

        normalized_row["jockey_raw"] = row["jockey_raw"] if row["jockey_raw"] is not None else None

        normalized_row["jockey_id"] = row["jockey_id"] if row["jockey_id"] is not None else None

        if row["weight"] is not None:
            try:
                normalized_row["weight"] = float(row["weight"])
            except Exception as e:
                return Result(
                    success=False,
                    error=f"Invalid weight value: {row['weight']}\n{e}"
                )
        else:
            normalized_row["weight"] = None

        if row["time"] is not None:
            try:
                m, s = row["time"].split(":")
                normalized_row["time_sec"] = int(m) * 60 + float(s)
            except Exception as e:
                return Result(
                    success=False,
                    error=f"Invalid time format: {row['time']}\n{e}"
                )
        else:
            normalized_row["time_sec"] = None

        if row["finish_diff"] is not None:
            if row["finish_diff"] in MARGIN_MAP.keys():
                normalized_row["finish_diff"] = MARGIN_MAP[row["finish_diff"]]

            elif "/" in row["finish_diff"]:
                if "." in row["finish_diff"]:
                    integ, frac = row["finish_diff"].split(".")
                    numer, denom = frac.split("/")
                else:
                    integ = 0
                    numer, denom = row["finish_diff"].split("/")
                normalized_row["finish_diff"] = int(integ) + int(numer) / int(denom)

            else:
                try:
                    normalized_row["finish_diff"] = float(row["finish_diff"])
                except Exception as e:
                    return Result(
                        success=False,
                        error=f"Invalid finish_diff value: {row['finish_diff']}\n{e}"
                    )
        else:
            normalized_row["finish_diff"] = None

        if row["popularity"] is not None:
            try:
                normalized_row["popularity"] = int(row["popularity"])
            except Exception as e:
                return Result(
                    success=False,
                    error=f"Invalid popularity value: {row['popularity']}\n{e}"
                )
        else:
            normalized_row["popularity"] = None

        if row["finish_3f"] is not None:
            try:
                normalized_row["finish_3f"] = float(row["finish_3f"])
            except Exception as e:
                return Result(
                    success=False,
                    error=f"Invalid finish_3f value: {row['finish_3f']}\n{e}"
                )
        else:
            normalized_row["finish_3f"] = None

        corners = ["corner_1", "corner_2", "corner_3", "corner_4"]
        if row["corner"] is not None:
            nums = [int(n) for n in row["corner"].split("-") if n.strip().isdigit()]
            corner_count = len(nums)
            nums += [None] * 4
            for c, n in zip(corners, nums[:4]):
                normalized_row[c] = n
            normalized_row["corner_count"] = corner_count
        else:
            for c in corners:
                normalized_row[c] = None
            normalized_row["corner_count"] = None

        if row["trainer_raw"] is not None:
            m_trainer_raw = re.match(r"(美浦|栗東|地方)(.+)", row["trainer_raw"])
            if m_trainer_raw:
                try:
                    normalized_row["stable_id"] = STABLE_MAP[m_trainer_raw.group(1)]
                except Exception as e:
                    return Result(
                        success=False,
                        error=f"Unknown stable value: {m_trainer_raw.group(1)}\n{e}"
                    )
                normalized_row["trainer_raw"] = m_trainer_raw.group(2)
            else:
                normalized_row["stable_id"] = 3
                normalized_row["trainer_raw"] = row["trainer_raw"]
        else:
            normalized_row["stable_id"] = None
            normalized_row["trainer_raw"] = None

        normalized_row["trainer_id"] = row["trainer_id"] if row["trainer_id"] is not None else None

        if row["horse_weight"] is not None:
            m_horse_weight = re.match(r"(\d+)(?:\(([+-]?\d+)\))?", row["horse_weight"])
            if m_horse_weight:
                try:
                    normalized_row["horse_weight"] = int(m_horse_weight.group(1))
                except Exception as e:
                    return Result(
                        success=False,
                        error=f"Invalid horse_weight value: {m_horse_weight.group(1)}\n{e}"
                    )
                horse_weight_diff = m_horse_weight.group(2)
                if horse_weight_diff is not None:
                    try:
                        normalized_row["horse_weight_diff"] = int(horse_weight_diff)
                    except Exception as e:
                        return Result(
                            success=False,
                            error=f"Invalid horse_weight_diff value: {horse_weight_diff}\n{e}"
                        )
                else:
                    normalized_row["horse_weight_diff"] = horse_weight_diff
            else:
                return Result(
                    success=False,
                    error=f"Invalid horse_weight formalt: {row['horse_weight']}"
                )
        else:
            normalized_row["horse_weight"] = None
            normalized_row["horse_weight_diff"] = None

        normalized_runners.append(normalized_row)

    return Result(success=True, value=normalized_runners)


def normalize_place(race_id: str, odds: dict) -> Result[list]:
    normalized_odds = []
    for num, odds in odds.items():
        normalized_row = {}

        normalized_row["race_id"] = race_id

        try:
            normalized_row["horse_number"] = int(num)
        except Exception as e:
            return Result(
                success=False,
                error=f"Invalid horse_number value: {num}\n{e}"
            )
        try:
            if not odds.startswith("-"):
                min_raw, max_raw = odds.split("-")
                normalized_row["odds_min"] = float(min_raw.replace(",", ""))
                normalized_row["odds_max"] = float(max_raw.replace(",", ""))
            else:
                normalized_row["odds_max"] = None
                normalized_row["odds_min"] = None
        except Exception as e:
            return Result(
                success=False,
                error=f"Invalid odds value: {odds}\n{e}"
            )
        normalized_odds.append(normalized_row)

    return Result(success=True, value=normalized_odds)


def normalize_wide(race_id: str, odds: dict) -> Result[list]:
    normalized_odds = []
    for nums, odds in odds.items():
        normalized_row = {}

        normalized_row["race_id"] = race_id

        try:
            horse_nums = [int(num) for num in nums.split("-")]

            if len(horse_nums) != 2:
                return Result(
                    success=False,
                    error=f"Invalid horse_numbers value: {nums}"
                )

            horse_nums = sorted(horse_nums)
            normalized_row["horse_number_1"] = horse_nums[0]
            normalized_row["horse_number_2"] = horse_nums[1]
        except Exception as e:
            return Result(
                success=False,
                error=f"Invalid horse_numbers value: {nums}\n{e}"
            )
        try:
            if not odds.startswith("-"):
                min_raw, max_raw = odds.split("-")
                normalized_row["odds_min"] = float(min_raw.replace(",", ""))
                normalized_row["odds_max"] = float(max_raw.replace(",", ""))
            else:
                normalized_row["odds_max"] = None
                normalized_row["odds_min"] = None
        except Exception as e:
            return Result(
                success=False,
                error=f"Invalid odds value: {odds}\n{e}"
            )
        normalized_odds.append(normalized_row)

    return Result(success=True, value=normalized_odds)


def normalize_trio(race_id: str, odds: dict) -> Result[list]:
    normalized_odds = []
    for nums, odds in odds.items():
        normalized_row = {}

        normalized_row["race_id"] = race_id

        try:
            horse_nums = [int(num) for num in nums.split("-")]

            if len(horse_nums) != 3:
                return Result(
                    success=False,
                    error=f"Invalid horse_numbers value: {nums}"
                )

            horse_nums = sorted(horse_nums)
            normalized_row["horse_number_1"] = horse_nums[0]
            normalized_row["horse_number_2"] = horse_nums[1]
            normalized_row["horse_number_3"] = horse_nums[2]
        except Exception as e:
            return Result(
                success=False,
                error=f"Invalid horse_numbers value: {nums}\n{e}"
            )
        try:
            normalized_row["odds"] = float(odds.replace(",", "")) if not odds.startswith("-") else None
        except Exception as e:
            return Result(
                success=False,
                error=f"Invalid odds value: {odds}\n{e}"
            )
        normalized_odds.append(normalized_row)

    return Result(success=True, value=normalized_odds)