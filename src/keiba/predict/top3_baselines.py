"""Jev を使わない基準（report3 §7-3）。Jev による改善を測る物差し。

能力順位は資料の ability.field_rank（直近走の加重平均着差のレース内順位、
小さいほど良い）だけで並べる。欠けた馬は後ろにまとめるが、全頭欠けたときの
馬番順は「順位」ではないので all_tied で報告する（Review Focus 2）。
"""

from __future__ import annotations

from dataclasses import dataclass

from keiba import config

__all__ = ["BaselineRanking", "ability_ranking", "uniform_probabilities"]


@dataclass(frozen=True)
class BaselineRanking:
    ranking: tuple[str, ...]
    all_tied: bool
    n_unranked: int


def ability_ranking(dossier: dict) -> BaselineRanking:
    # 馬番は数値として並べる（"10" を "2" より前にしない。ゼロ埋めの無い馬番の入力もある）。
    horses = dossier["horses"]
    ranked = sorted((h["ability"]["field_rank"], int(u), u) for u, h in horses.items()
                    if h["ability"]["field_rank"] is not None)
    unranked = sorted((u for u, h in horses.items() if h["ability"]["field_rank"] is None),
                      key=int)
    return BaselineRanking(ranking=tuple(u for _, _, u in ranked) + tuple(unranked),
                           all_tied=not ranked, n_unranked=len(unranked))


def uniform_probabilities(dossier: dict) -> dict[str, float]:
    n = len(dossier["horses"])
    p = min(config.TOP3_PLACE, n) / n if n else 0.0
    return dict.fromkeys(dossier["horses"], p)
