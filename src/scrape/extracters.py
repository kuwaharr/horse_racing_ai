import json
import re

from bs4 import BeautifulSoup

from ..common.logger import get_logger
from ..common.maps import TRACK_MAP_INV
from ..common.result import Result

logger = get_logger("src.scrape.extracters")

def parse_url(url: str) -> Result[dict]:
    try:
        race_id_m = re.search(r"race_id=(\d{12})", url)
        race_id = race_id_m.group(1)
    except Exception as e:
        return Result(success=False, error=f"Could not get race_id: {e}")

    track = TRACK_MAP_INV[int(race_id[4:6])]

    race_number = race_id[-2:]

    race_info = {
        "race_id": race_id,
        "track": track,
        "race_number": race_number,
    }
    return Result(success=True, value=race_info)


def extract_race_meta(soup: BeautifulSoup) -> Result[dict]:
    try:
        date_list_tag = soup.find(id="RaceList_DateList")
        date_tag = date_list_tag.find("dd", class_="Active")
        date = date_tag.get_text(strip=True)
    except Exception as e:
        return Result(success=False, error=f"Could not get RaceList_DateList: {e}")

    try:
        race_column_01_tag = soup.find("div", class_="RaceColumn01")
        race_data_01_tag = race_column_01_tag.find("div", class_="RaceData01")
    except Exception as e:
        return Result(success=False, error=f"Could not find RaceColumn01: {e}")

    try:
        race_data_01_raw = race_data_01_tag.get_text(strip=True)
    except Exception as e:
        return Result(success=False, error=f"Could not find RaceData01: {e}")

    race_data_01_texts = "".join(race_data_01_raw.split())
    race_data_01_texts = race_data_01_texts.split("/")

    weather = None
    track_condition = None
    for text in race_data_01_texts:
        if "発走" in text:
            post_time_m = re.match(r"(.+)発走", text)
            if post_time_m:
                post_time = post_time_m.group(1)
            else:
                logger.warning("extract_race_meta Could not get post_time")
                post_time = None

        elif "m" in text:
            track_info_m = re.match(r"(芝|ダ)(\d{3,4})m\((.+)\)", text)
            if track_info_m:
                surface = track_info_m.group(1)

                distance = track_info_m.group(2)

                course_info = track_info_m.group(3)
                course_layout = None
                course_variant = None
                if course_info.startswith("直線"):
                    course_direction = "直線"
                    course_info = course_info[2:]
                else:
                    course_direction = course_info[0]
                    course_info = course_info[1:]
                for letter in course_info:
                    if letter in ["外", "内"]:
                        course_layout = letter
                    elif letter in ["A", "B", "C", "D"]:
                        course_variant = letter
                    else:
                        logger.warning("extract_race_meta Unknown letter in course info: %s", letter)
            else:
                logger.warning("extract_race_meta Could not get surface, distance, course info")
                surface = None
                distance = None
                course_direction = None

        elif "天候" in text:
            weather = text[3:]

        elif "馬場" in text:
            track_condition = text[3:]

        else:
            logger.warning("extract_race_meta Unknown text in race_data_01_texts: %s", text)

    race_size = None
    race_data_02_tag = race_column_01_tag.find("div", class_="RaceData02")
    if race_data_02_tag:
        race_data_02_tag_text = ""
        spans = race_data_02_tag.find_all("span")
        for span in spans:
            race_data_02_tag_text += span.text
        race_data_02_tag_text = race_data_02_tag_text.replace(" ", "")

        race_data_02_tag_m = re.search(r"(\d+)頭", race_data_02_tag_text)
        if race_data_02_tag_m:
            race_size = race_data_02_tag_m.group(1)
        else:
            logger.warning("extract_race_meta Could not get race_size")
    else:
        logger.warning("extract_race_meta Could not get race_size")

    race_meta = {
        "date": date,
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
    return Result(success=True, value=race_meta)


def extract_runners(soup: BeautifulSoup) -> Result[list[dict]]:
    race_table_tag = soup.find("table", class_="RaceTable01")
    if race_table_tag is None:
        return Result(success=False, error="Could not find RaceTable01")

    header_tags = race_table_tag.find("tr").find_all("th")
    headers = []
    for cell in header_tags:
        cell_text = cell.get_text(strip=True).replace(" ", "")
        headers.append(cell_text)

    required_cols = [
        "着順",
        "枠",
        "馬番",
        "馬名",
        "性齢",
        "斤量",
        "騎手",
        "タイム",
        "着差",
        "人気",
        "後3F",
        "コーナー通過順",
        "厩舎",
        "馬体重(増減)",
    ]
    missing_cols = [c for c in required_cols if c not in headers]
    if len(missing_cols) > 0:
        return Result(success=False, error=f"Missing columns in table: {missing_cols}")

    col_idx = {name: idx for idx, name in enumerate(headers)}

    def idx(name: str) -> int:
        return col_idx[name]

    runners = []
    for tr in race_table_tag.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue

        row = {}
        cells = [td.get_text(strip=True) for td in tds]

        finish = cells[idx("着順")]
        row["finish"] = finish if len(finish) > 0 else None
        if row["finish"] is None:
            return Result(success=False, error="Could not get finish")

        gate = cells[idx("枠")]
        row["gate"] = gate if len(gate) > 0 else None

        horse_number = cells[idx("馬番")]
        row["horse_number"] = horse_number if len(horse_number) > 0 else None

        horse_name = cells[idx("馬名")]
        row["horse_name"] = horse_name if len(horse_name) > 0 else None
        if row["horse_name"] is None:
            return Result(success=False, error="Could not get horse_name")

        sex_age = cells[idx("性齢")]
        row["sex_age"] = sex_age if len(sex_age) > 0 else None

        weight = cells[idx("斤量")]
        row["weight"] = weight if len(weight) > 0 else None

        jockey_raw = cells[idx("騎手")]
        row["jockey_raw"] = jockey_raw if len(jockey_raw) > 0 else None

        time = cells[idx("タイム")]
        row["time"] = time if len(time) > 0 else None

        finish_diff = cells[idx("着差")]
        row["finish_diff"] = finish_diff if len(finish_diff) > 0 else None

        popularity = cells[idx("人気")]
        row["popularity"] = popularity if len(popularity) > 0 else None

        finish_3f = cells[idx("後3F")]
        row["finish_3f"] = finish_3f if len(finish_3f) > 0 else None

        corner = cells[idx("コーナー通過順")]
        row["corner"] = corner if len(corner) > 0 else None

        trainer_raw = cells[idx("厩舎")]
        row["trainer_raw"] = trainer_raw if len(trainer_raw) > 0 else None

        horse_weight = cells[idx("馬体重(増減)")]
        row["horse_weight"] = horse_weight if len(horse_weight) > 0 else None

        horse_id = None
        horse_info_tag = tr.find("td", class_="Horse_Info")
        if (
            horse_info_tag
            and (horse_a_tag := horse_info_tag.find("a"))
            and horse_a_tag.has_attr("href")
            and (m := re.search(r"horse/(\d+)", horse_a_tag["href"]))
        ):
            horse_id = m.group(1)
        else:
            logger.warning("extract_runners Could not get horse_id")

        row["horse_id"] = horse_id

        jockey_id = None
        if row["jockey_raw"] is not None:
            jockey_tag = tr.find("td", class_="Jockey")
            if (
                jockey_tag
                and (jockey_a_tag := jockey_tag.find("a"))
                and jockey_a_tag.has_attr("href")
                and (m := re.search(r"recent/(\d+)", jockey_a_tag["href"]))
            ):
                jockey_id = m.group(1)
            else:
                logger.warning("extract_runners Could not get jockey_id")

        row["jockey_id"] = jockey_id

        trainer_id = None
        if row["trainer_raw"] is not None:
            trainer_tag = tr.find("td", class_="Trainer")
            if (
                trainer_tag
                and (trainer_a_tag := trainer_tag.find("a"))
                and trainer_a_tag.has_attr("href")
                and (m := re.search(r"recent/(\d+)", trainer_a_tag["href"]))
            ):
                trainer_id = m.group(1)
            else:
                logger.warning("extract_runners Could not get trainer_id")

            row["trainer_id"] = trainer_id

        key_order = [
            "gate",
            "horse_number",
            "finish",
            "horse_name",
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
        row_sorted = {}
        for key in key_order:
            row_sorted[key] = row.get(key)
        runners.append(row_sorted)

    runners_sorted = sorted(
        runners,
        key=lambda x: int(x["horse_number"]) if (x["horse_number"].isdigit()) and (x["horse_number"]) is not None else 10**9
    )
    return Result(success=True, value=runners_sorted)


def parse_jsonp(jsonp: str, odds_type: int) -> Result[dict]:
    json_part_m = re.match(r".*\((.+)\)", jsonp)
    try:
        json_part = json_part_m.group(1)

        data_all = json.loads(json_part)
        odds_block = data_all["data"]["odds"][str(odds_type)]
        return Result(success=True, value=odds_block)
    except Exception as e:
        return Result(success=False, error=str(e))


def parse_place(odds_block: dict) -> dict:
    place = {}
    for key, value in odds_block.items():
        place[key] = f"{value[0]}-{value[1]}"

    return place


def parse_win(odds_block: dict) -> dict:
    win = {}
    for key, value in odds_block.items():
        win[key] = value[0]

    return win


def parse_wide(odds_block: dict) -> dict:
    wide = {}
    for key, value in odds_block.items():
        new_key = key[:2] + "-" + key[2:]
        wide[new_key] = f"{value[0]}-{value[1]}"

    return wide


def parse_trio(odds_block: dict) -> dict:
    trio = {}
    for key, value in odds_block.items():
        new_key = key[:2] + "-" + key[2:4] + "-" + key[4:]
        trio[new_key] = value[0]

    return trio


def extract_race_ids(soup: BeautifulSoup) -> set:
    ids = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if m := re.match(r"/race/(\d{12})", href):
            ids.add(m.group(1))

    sorted_ids = sorted(ids)
    return sorted_ids
