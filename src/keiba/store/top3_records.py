"""3着内予想の記録。勝率系の PredictionRecord とは別の型・別の木に置く。

勝率と3着内確率は意味が違うので、同じ記録形式・同じファイルに混ぜない
（report3 §9）。月ごとの jsonl に答えを、送った state と質問は1予想1ファイルで
別の木に置く（store/prompts.py と同じ理由: 18頭で数十万文字あり、答えを読む
たびに引きずらないため）。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from keiba import config
from keiba.predict.top3_questions import TOP3_PROMPT_VERSION

__all__ = ["Top3Record", "load_month", "save", "versions_for"]


@dataclass(frozen=True)
class Top3Record:
    record_id: str
    race_id: str | None
    #: "python" / "plan_a" / "plan_b"。同じ Jev 応答から作る "winner_same_state" は
    #: 記録を分けず、plan_a / plan_b の記録の winner_probabilities から評価側が作る。
    method: str
    predicted_at: datetime
    prediction_cutoff: datetime
    versions: dict
    top3_probabilities: dict | None
    winner_probabilities: dict | None
    winner_missing: list
    pace_probabilities: dict | None
    ranking: list
    usage: dict
    state_chars: int | None
    seconds: float
    data_quality: dict


def versions_for(*, data_version: str, model: str | None) -> dict:
    return {
        "data_version": data_version,
        "model": model,
        "dossier_version": config.DOSSIER_VERSION,
        "top3_prompt_version": TOP3_PROMPT_VERSION,
        "eval_rules_version": config.TOP3_EVAL_RULES_VERSION,
    }


def _root(data_dir: Path) -> Path:
    return Path(data_dir) / "top3"


def _to_json(record: Top3Record) -> dict:
    payload = asdict(record)
    payload["predicted_at"] = record.predicted_at.isoformat()
    payload["prediction_cutoff"] = record.prediction_cutoff.isoformat()
    return payload


def _from_json(payload: dict) -> Top3Record:
    payload = dict(payload)
    payload["predicted_at"] = datetime.fromisoformat(payload["predicted_at"])
    payload["prediction_cutoff"] = datetime.fromisoformat(payload["prediction_cutoff"])
    return Top3Record(**payload)


def save(record: Top3Record, *, sent: dict | None, data_dir: Path) -> Path:
    month = record.predicted_at.strftime("%Y%m")
    path = _root(data_dir) / "predictions" / f"{month}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_json(record), ensure_ascii=False) + "\n")
    if sent is not None:
        prompt_path = _root(data_dir) / "prompts" / month / f"{record.record_id}.json"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        # 記録自身のキーを後に置く。sent に同名のキーがあっても上書きされない
        # （final review M-3）。
        prompt_path.write_text(json.dumps(
            {**sent, "record_id": record.record_id, "race_id": record.race_id,
             "versions": record.versions},
            ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return path


def load_month(year_month: str, *, data_dir: Path) -> list[Top3Record]:
    path = _root(data_dir) / "predictions" / f"{year_month}.jsonl"
    if not path.exists():
        return []
    return [_from_json(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
