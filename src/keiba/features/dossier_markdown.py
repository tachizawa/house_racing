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
            lines.append(f"- {key}: {value}")
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


def _horse(umaban: str, h: dict) -> list[str]:
    lines = [f"### {umaban} {h['bamei']}（{_v(h['sex_age'])} / {_v(h['futan_kg'])}kg / 枠{_v(h['waku'])}）",
             "", f"履歴: {h['history']['status']} — {h['history']['note']}"
             f"（{h['history']['runs_found']}走）", ""]
    if h["recent_runs"]:
        lines += ["| " + " | ".join(_RUN_COLS) + " |",
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
             f"- 履歴の内訳: {fs['history_status_counts']}"]
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
    for umaban, h in dossier["horses"].items():
        lines += _horse(umaban, h)
    lines += _quality(dossier["data_quality"], dossier["field_summary"])
    return "\n".join(lines)
