from pathlib import Path
from typing import Any


def _load_dataset(path: Path, engine: str):
    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        raise RuntimeError("評価には pandas が必要です。") from e

    return pd.read_parquet(path, engine=engine)


def _split_by_date(df, test_ratio: float):
    dates = sorted(df["date"].dropna().unique())
    if not dates:
        raise ValueError("Dataset has no date values")

    split_idx = int(len(dates) * (1 - test_ratio))
    split_idx = min(max(split_idx, 1), len(dates) - 1)
    split_date = dates[split_idx]

    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()
    return train_df, test_df, split_date


def _evaluate_topk(df, score_col: str, ascending: bool, top_k_values: list[int]) -> list[dict[str, Any]]:
    results = []
    scored = df.dropna(subset=[score_col]).copy()
    if scored.empty:
        return [
            {
                "score": score_col,
                "top_k": top_k,
                "races": 0,
                "selections": 0,
                "hits": 0,
                "hit_rate_pct": None,
            }
            for top_k in top_k_values
        ]

    for top_k in top_k_values:
        selected = (
            scored.sort_values(["race_id", score_col, "horse_number"], ascending=[True, ascending, True])
            .groupby("race_id", as_index=False)
            .head(top_k)
        )
        selections = len(selected)
        hits = int(selected["target_top3"].sum())
        results.append(
            {
                "score": score_col,
                "top_k": top_k,
                "races": int(selected["race_id"].nunique()),
                "selections": selections,
                "hits": hits,
                "hit_rate_pct": None if selections == 0 else hits / selections * 100,
            }
        )

    return results


def evaluate_baselines(dataset_path: Path, engine: str = "auto", test_ratio: float = 0.2) -> dict[str, Any]:
    df = _load_dataset(dataset_path, engine)
    train_df, test_df, split_date = _split_by_date(df, test_ratio)

    test_df = test_df.copy()
    test_df["place_odds_mid"] = (test_df["place_odds_min"] + test_df["place_odds_max"]) / 2

    results = []
    results.extend(_evaluate_topk(test_df, "popularity", ascending=True, top_k_values=[1, 2, 3]))
    results.extend(_evaluate_topk(test_df, "place_odds_min", ascending=True, top_k_values=[1, 2, 3]))
    results.extend(_evaluate_topk(test_df, "place_odds_mid", ascending=True, top_k_values=[1, 2, 3]))

    return {
        "dataset_path": str(dataset_path),
        "rows": len(df),
        "races": int(df["race_id"].nunique()),
        "train_rows": len(train_df),
        "train_races": int(train_df["race_id"].nunique()),
        "test_rows": len(test_df),
        "test_races": int(test_df["race_id"].nunique()),
        "split_date": split_date,
        "results": results,
    }


def format_baseline_report(report: dict[str, Any]) -> str:
    lines = [
        f"Dataset: {report['dataset_path']}",
        f"Rows: {report['rows']:,}",
        f"Races: {report['races']:,}",
        f"Split date: {report['split_date']}",
        f"Train: {report['train_rows']:,} rows / {report['train_races']:,} races",
        f"Test: {report['test_rows']:,} rows / {report['test_races']:,} races",
        "",
        "Baseline results on test period",
        "score           top_k  races  selections  hits  hit_rate",
    ]

    for row in report["results"]:
        hit_rate = row["hit_rate_pct"]
        hit_rate_text = "n/a" if hit_rate is None else f"{hit_rate:.2f}%"
        lines.append(
            f"{row['score']:<15} {row['top_k']:>5}  {row['races']:>5,}  "
            f"{row['selections']:>10,}  {row['hits']:>4,}  {hit_rate_text:>8}"
        )

    return "\n".join(lines)
