"""資料（features/dossier.py の dict）を人間が確認するための Markdown にする。

値を作らない・丸めない・補わない。欠損は「—」で出し、0 と見分けられるようにする。
"""

from __future__ import annotations

__all__ = ["render"]

_DASH = "—"


def _v(value) -> str:
    if value is None:
        return _DASH
    if isinstance(value, list):
        return "-".join(_v(x) for x in value)
    return str(value)


def _label(labels: dict[str, str], key: str) -> str:
    """キーの日本語名。表に無いキーは黙って消さず、そのまま出す。"""
    return labels.get(key, key)


# course_facts.json の facts のキー -> (日本語名, 単位)。
_FACT_LABELS: dict[str, tuple[str, str]] = {
    "straight_m": ("直線距離", "m"),
    "circumference_m": ("1周の距離", "m"),
    "width_m": ("幅員", "m"),
    "elevation_note": ("高低差", ""),
    "layout": ("コースの形", ""),
}


def _fact_line(key: str, value) -> str:
    name, unit = _FACT_LABELS.get(key, (key, ""))
    return f"- {name}: {_v(value)}{unit if value is not None else ''}"


def _course(c: dict) -> list[str]:
    lines = ["## コース", ""]
    if not c["identified"]:
        lines.append(f"コースを特定していない: {c['reason']}")
        return lines + [""]
    lines.append(f"- コース: {c['track_label']} {c['kyori_m']}m"
                 f"（仮柵: {_v(c['course_kubun_cd'])}）")
    if c["facts"] is None:
        lines.append("- 基本情報: 表に無い（推測で補っていない）")
    else:
        for key, value in c["facts"].items():
            lines.append(_fact_line(key, value))
        lines.append(f"- スタート: {_v(c['start_note'])}")
        src = c["facts_source"]
        if src.get("rail_basis") is not None:
            lines.append(f"- 出典: {src['url']}（取得 {src['retrieved_at']}、"
                         f"数値は{src['rail_basis']}のもの）")
        else:
            lines.append(f"- 出典: {src['url']}（取得 {src['retrieved_at']}）")
    return lines + [""]


def _race(r: dict, cutoff: str) -> list[str]:
    return ["## 今回の条件", "",
            "| 項目 | 値 |", "|---|---|",
            f"| 予想締切 | {cutoff} |",
            f"| 競馬場 | {_v(r.get('jyo'))} |",
            f"| 芝・ダート | {_v(r.get('track_kind'))} |",
            f"| 距離 | {_v(r.get('kyori_m'))} |",
            f"| 馬場 | {_v(r.get('baba'))} |",
            f"| クラス | {_v(r.get('class_name'))}（順序値 {_v(r.get('class_level'))}） |",
            f"| 頭数 | {_v(r.get('entry_count'))} |", ""]


_RUN_COLS = ("date", "jyo", "track_label", "kyori_m", "baba", "class_level",
             "syusso_tosu", "kakutei_jyuni", "margin_sec", "corner_positions", "last_3f_sec")
_RUN_COL_LABELS = {
    "date": "日付", "jyo": "競馬場", "track_label": "コース", "kyori_m": "距離(m)",
    "baba": "馬場", "class_level": "クラス順序値", "syusso_tosu": "頭数",
    "kakutei_jyuni": "着順", "margin_sec": "着差(秒)", "corner_positions": "通過順(1-4角)",
    "last_3f_sec": "上がり3F(秒)",
}

# vs_last_run / ability / running_style のキー -> 日本語名。Jev に送った値を
# そのまま出す（丸めない）。表に無いキーもキー名のまま出す（黙って落とさない）。
_VS_LAST_RUN_LABELS = {
    "days_since_last_run": "前走からの日数",
    "distance_change_m": "距離の変化(m)",
    "futan_change_kg": "斤量の変化(kg)",
    "class_level_change": "クラス順序値の変化",
}
_ABILITY_LABELS = {
    "recent_margin_sec_wavg": "近走の着差の加重平均(秒)",
    "runs_used": "使った走数",
    "recent_class_level_wavg": "近走のクラス順序値の加重平均",
    "best_class_level": "最高クラス順序値",
    "field_rank": "レース内順位（着差）",
    "margin_period": "着差の集計期間",
}
_RUNNING_STYLE_LABELS = {
    "corner_pos_mean": "通過順の相対位置の平均",
    "corner_pos_stdev": "通過順の相対位置の標準偏差",
    "corner_pos_starts": "使った走数",
    "corner_source": "使ったコーナー",
}


def _period_text(period) -> str:
    if isinstance(period, dict):
        return f"{_v(period.get('from'))}〜{_v(period.get('to'))}"
    return _v(period)


def _key_values(title: str, values: dict | None, labels: dict[str, str]) -> str:
    if not values:
        return f"- {title}: {_DASH}"
    parts = []
    for key, value in values.items():
        text = _period_text(value) if key.endswith("period") else _v(value)
        parts.append(f"{_label(labels, key)} {text}")
    return f"- {title}: " + " / ".join(parts)


def _counts_text(counts: dict) -> str:
    return "、".join(f"{key} {value}" for key, value in counts.items()) or _DASH


def _condition_definitions_table(definitions: dict) -> list[str]:
    lines = ["### 条件別実績の比較条件", "", "| 区分 | 比較条件 |", "|---|---|"]
    for key, definition in definitions.items():
        lines.append(f"| {key} | {definition} |")
    return lines + [""]


def _horse(umaban: str, h: dict, status_notes: dict, missing: list[str] | None) -> list[str]:
    note = status_notes[h["history"]["status"]]
    lines = [f"### {umaban} {h['bamei']}（{_v(h['sex_age'])} / {_v(h['futan_kg'])}kg / 枠{_v(h['waku'])}）",
             "", f"履歴: {h['history']['status']} — {note}"
             f"（{h['history']['runs_found']}走）", ""]
    lines.append(_key_values("前走との比較", h.get("vs_last_run"), _VS_LAST_RUN_LABELS))
    lines.append(_key_values("能力", h.get("ability"), _ABILITY_LABELS))
    lines.append(_key_values("脚質", h.get("running_style"), _RUNNING_STYLE_LABELS))
    if missing:
        lines.append(f"- 計算できなかった項目: {'、'.join(missing)}")
    lines.append("")
    if h["recent_runs"]:
        lines += ["| " + " | ".join(_RUN_COL_LABELS[k] for k in _RUN_COLS) + " |",
                  "|" + "---|" * len(_RUN_COLS)]
        for run in h["recent_runs"]:
            lines.append("| " + " | ".join(_v(run[k]) for k in _RUN_COLS) + " |")
        lines.append("")
    lines += ["| 条件 | 出走 | 勝 | 3着内 | 平均着差 | 備考 |", "|---|---|---|---|---|---|"]
    for key, c in h["conditions"].items():
        lines.append(f"| {key} | {_v(c['starts'])} | {_v(c['wins'])} | {_v(c['top3'])} | "
                     f"{_v(c['margin_sec_mean'])} | {_v(c['inapplicable_reason'])} |")
    return lines + [""]


def _quality(dq: dict, fs: dict) -> list[str]:
    lines = ["## データの充足状況", "",
             f"- データ版: {dq.get('data_version')}（収録開始 {_v(dq.get('coverage_from'))}）",
             f"- 過去走がある馬: {fs['n_with_history']} / {fs['n_horses']}",
             f"- 履歴の内訳: {_counts_text(fs['history_status_counts'])}"]
    for note in dq.get("notes", []):
        lines.append(f"- 注記: {note}")
    for umaban, notes in dq.get("horse_notes", {}).items():
        for note in notes:
            lines.append(f"- {umaban}番: {note}")
    return lines + [""]


def render(dossier: dict) -> str:
    lines = [f"# 予想用資料（資料の版 {dossier['dossier_version']}）", ""]
    lines += _course(dossier["course"])
    lines += _race(dossier["race"], dossier["prediction_cutoff"])
    lines += ["## 出走馬", ""]
    lines += _condition_definitions_table(dossier["condition_definitions"])
    missing = dossier["data_quality"].get("missing", {})
    for umaban, h in dossier["horses"].items():
        lines += _horse(umaban, h, dossier["history_status_notes"], missing.get(umaban))
    lines += _quality(dossier["data_quality"], dossier["field_summary"])
    return "\n".join(lines)
