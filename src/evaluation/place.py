import numpy as np
import pandas as pd


def eval_ev(df: pd.DataFrame, ev_th: float, max_k: int = 3) -> dict:
    df_high_ev = df[df["odds_mid"].notna() & (df["ev"] >= ev_th)].copy()

    picked = (
        df_high_ev.sort_values(["race_id", "ev"], ascending=[True, False])
        .groupby("race_id")
        .head(max_k)
    )

    bets = len(picked)
    hits = int(picked["is_place"].sum())
    hit_rate = hits / bets if bets else 0.0

    ret = float((picked["is_place"] * picked["odds_mid"]).sum())
    roi = ret / bets if bets else float("nan")

    return {
        "ev_th": ev_th,
        "bets": bets,
        "hits": hits,
        "hit_rate": hit_rate,
        "roi": roi,
    }