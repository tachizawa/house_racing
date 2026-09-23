"""3着以内の採点（report3 §7）。

採点規則はこのプランの Task 8 に列挙したもので、config.TOP3_EVAL_RULES_VERSION
で版を持つ。集計はレース単位（馬の行を独立標本として扱わない）。差の不確かさは
開催日クラスタで再標本化したブートストラップで出す。
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from keiba import config
from keiba.models import ResultRow

__all__ = ["Top3Outcome", "aggregate", "outcome_from_results", "paired_bootstrap",
           "race_date_cluster", "score_race"]

_EPS = 1e-12


@dataclass(frozen=True)
class Top3Outcome:
    race_id: str
    top3: tuple[str, ...]
    starters: tuple[str, ...]
    excluded: tuple[str, ...]


def outcome_from_results(race_id: str, results: Sequence[ResultRow]) -> Top3Outcome | None:
    starters = tuple(r.entry.umaban for r in results if r.counts_as_starter)
    excluded = tuple(r.entry.umaban for r in results if not r.counts_as_starter)
    top3 = tuple(r.entry.umaban for r in results
                 if r.can_win and r.kakutei_jyuni is not None
                 and 1 <= r.kakutei_jyuni <= config.TOP3_PLACE)
    if not top3:
        return None
    return Top3Outcome(race_id=race_id, top3=top3, starters=starters, excluded=excluded)


def score_race(selected: Sequence[str], probabilities: Mapping[str, float] | None,
               outcome: Top3Outcome) -> dict:
    actual = set(outcome.top3)
    chosen = list(selected)
    result = {
        "hits": sum(1 for u in chosen if u in actual),
        "all_top3_selected": actual <= set(chosen),
        "top3_exceeds_n": len(actual) > len(chosen),
        "selected_scratched": sum(1 for u in chosen if u in outcome.excluded),
        "brier": None, "log_loss": None, "prob_sum": None, "prob_sum_minus_actual": None,
    }
    if probabilities is None:
        return result
    missing = [u for u in outcome.starters if u not in probabilities]
    if missing:
        raise ValueError(f"{outcome.race_id}: 出走馬の3着内確率がありません: {missing}")
    ps = [(probabilities[u], 1.0 if u in actual else 0.0) for u in outcome.starters]
    result["brier"] = sum((p - y) ** 2 for p, y in ps) / len(ps)
    result["log_loss"] = -sum(
        y * math.log(max(p, _EPS)) + (1 - y) * math.log(max(1 - p, _EPS)) for p, y in ps
    ) / len(ps)
    result["prob_sum"] = sum(p for p, _ in ps)
    result["prob_sum_minus_actual"] = result["prob_sum"] - len(actual)
    return result


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def aggregate(scored: Sequence[dict], *, n: int) -> dict:
    with_prob = [s for s in scored if s["brier"] is not None]
    distribution = {str(k): 0 for k in range(n + 1)}
    for s in scored:
        distribution[str(min(s["hits"], n))] += 1
    return {
        "n": n,
        "rules_version": config.TOP3_EVAL_RULES_VERSION,
        "n_races": len(scored),
        "mean_hits": _mean([s["hits"] for s in scored]),
        "hit_distribution": distribution,
        "all_top3_selected_rate": _mean([1.0 if s["all_top3_selected"] else 0.0 for s in scored]),
        "n_top3_exceeds_n": sum(1 for s in scored if s["top3_exceeds_n"]),
        "n_selected_scratched": sum(s["selected_scratched"] for s in scored),
        "n_with_probabilities": len(with_prob),
        "brier": _mean([s["brier"] for s in with_prob]),
        "log_loss": _mean([s["log_loss"] for s in with_prob]),
        "prob_sum_minus_actual_mean": _mean([s["prob_sum_minus_actual"] for s in with_prob]),
    }


def race_date_cluster(race_id: str) -> str:
    """race_id の先頭8桁（年4+月日4）= 開催日。同じ日のレースは馬場・天候を共有する。"""
    return race_id[:8]


def paired_bootstrap(a: Mapping[str, float], b: Mapping[str, float], *,
                     cluster_of: Callable[[str], str], n_resamples: int = 2000,
                     seed: int = 0) -> dict:
    """レースごとの値 a - b の平均と、クラスタ再標本化の95%区間。両方にあるレースだけ使う。"""
    common = sorted(set(a) & set(b))
    diffs = {race_id: a[race_id] - b[race_id] for race_id in common}
    clusters: dict[str, list[float]] = {}
    for race_id, d in diffs.items():
        clusters.setdefault(cluster_of(race_id), []).append(d)
    keys = sorted(clusters)
    mean_diff = _mean(list(diffs.values()))
    if not keys:
        return {"mean_diff": mean_diff, "lo": float("nan"), "hi": float("nan"),
                "n_races": 0, "n_clusters": 0}
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        sample = [d for key in (rng.choice(keys) for _ in keys) for d in clusters[key]]
        means.append(sum(sample) / len(sample))
    means.sort()
    return {"mean_diff": mean_diff,
            "lo": means[int(0.025 * (n_resamples - 1))],
            "hi": means[int(0.975 * (n_resamples - 1))],
            "n_races": len(common), "n_clusters": len(keys)}
