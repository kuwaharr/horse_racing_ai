import requests
from bs4 import BeautifulSoup

from ..common.result import Result


headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebkit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


def make_soup(url: str) -> Result[BeautifulSoup]:
    try:
        resp = requests.get(url, headers=headers)
        soup = BeautifulSoup(resp.content, "lxml")
        return Result(success=True, value=soup)
    except Exception as e:
        return Result(success=False, error=str(e))


def fetch_odds_jsonp(race_id: str, odds_type: int, compress=0) -> Result[str]:
    url = (
        "https://race.netkeiba.com/api/api_get_jra_odds.html"
        "?pid=api_get_jra_odds"
        "&input=UTF-8"
        "&output=jsonp"
        f"&race_id={race_id}"
        f"&type={odds_type}"
        "&action=init"
        "&sort=odds"
        f"&compress={compress}"
    )
    try:
        resp = requests.get(url, headers=headers)
        return Result(success=True, value=resp.text)

    except Exception as e:
        return Result(success=False, error=str(e))



def make_race_url(race_id: str) -> str:
    race_url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}&rf=race_submenu"
    return race_url


def make_horse_url(horse_id: str) -> str:
    return f"https://db.netkeiba.com/horse/{horse_id}/"


def make_horse_pedigree_url(horse_id: str) -> str:
    return f"https://db.netkeiba.com/horse/ped/{horse_id}/"
