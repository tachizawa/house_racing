"""features 層のうち、Repository と入力照合を橋渡しする部分。

第2層以降が読む ``PredictionContext`` は、``ability.py`` / ``derive.py`` が返す
個々の指標を、レース1件・出走全頭ぶんに束ねるだけ。ここでは合成も重み付けも
しない（それは第3層の仕事）。

ここが最も気をつけるべきなのは、対象レース自身の結果が混ざらないこと。
``Repository.past_runs`` は ``as_of`` ゲートで守られているが、それを安全に
使い切る（別の時刻を渡さない、結果側の行から値を借りない）のはこのモジュールの
責任で、次の防波堤は ``summary.py`` の文字列検索テストになる。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from keiba.data.repository import Repository
from keiba.features.ability import AbilitySummary, ability_summary, field_ranks
from keiba.features.derive import (
    BABA_UNKNOWN_REASON,
    ConditionRecord,
    RecentForm,
    RunningStyle,
    Transport,
    condition_records,
    recent_form,
    running_style,
    transport,
)
from keiba.input.match import match_horses
from keiba.models import EntryRow, HorseEntry, PastRun, RaceInput

# 過去走（ketto_num で引く集計）に依存する特徴量群。引けた過去走が0本の馬
# （未登録で ketto_num が無い馬も、照合できたがデータ版に先行出走が無い馬も）
# では、これらすべてが「計算できない」のであって「値が0だった」のではない。
# 区別を missing に残す。transport はここには含めない——引けた走の数では
# 決まらず、transport_.base が解決できたかどうかで別途 missing に足すため
# （build_context 参照）。
_HISTORY_GROUPS: tuple[str, ...] = ("ability", "by_condition", "recent", "running_style")


@dataclass(frozen=True)
class HorseFeatures:
    """1頭分の特徴量一式。値は ability.py / derive.py が計算したものをそのまま
    持つ。ここでは並べるだけで、合成はしない。

    transport_base_source は transport.base をどこから得たか（"entry": 今回の
    出馬表 / "last_run": 直近走からの推定 / None: どちらも無い）。推定値と
    確認済みの値を state 上でも区別できるようにするため（spec §4.2）。
    """

    umaban: str
    bamei: str
    ketto_num: str | None
    entry: HorseEntry
    ability: AbilitySummary
    field_rank: float | None
    recent: RecentForm
    by_condition: dict[str, ConditionRecord]
    running_style: RunningStyle
    transport: Transport
    transport_base_source: str | None
    missing: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class PredictionContext:
    """1レース分の予想材料。``summary.py`` がこれを dict（state）に変換する。"""

    race: RaceInput
    horses: tuple[HorseFeatures, ...]
    data_version: str
    data_fetched_at: datetime
    notes: tuple[str, ...]
    pace_definition: dict | None = None

    def horse(self, umaban: str) -> HorseFeatures:
        for features in self.horses:
            if features.umaban == umaban:
                return features
        raise KeyError(f"umaban {umaban!r} is not in this race")


def _current_tozai_cd(entry_row: EntryRow | None,
                       runs: Sequence[PastRun]) -> tuple[str, str | None]:
    """輸送の起点として使う「今回の所属」コードと、その出どころ。

    利用者入力の ``HorseEntry`` は tozai_cd を持たない（出走条件として申告を
    求めていない）ので、他から補うしかない。対象レースの公式な出馬表/出走馬名表
    （``Repository.entries()``）が既に存在すればそれを最優先で使う（出どころ
    ``"entry"``）。まだ存在しない（対象レースが未発表）場合は、直近の過去走
    時点の所属を現在の所属の推定として使う（出どころ ``"last_run"``）——
    ``derive.transport`` が過去走ごとに「その走時点の所属」を見るのと同じ
    考え方（転厩後も直近走には新しい所属が反映されているため）。どちらも
    無ければ空文字列・出どころ ``None`` を返す（``transport()`` 内で
    base=None に落ちる。未知のトレセンコードとして扱われるだけで、例外には
    ならない）。

    出どころを返り値に含めるのは、``"last_run"`` が確認済みの情報ではなく
    推定であることを state 上でも区別するため（spec §4.2。決定2の「輸送距離の
    概数」と同じ理由——推定を確認済みの情報のふりで渡さない）。
    """
    if entry_row is not None:
        return entry_row.tozai_cd, "entry"
    if not runs:
        return "", None
    latest = max(runs, key=lambda run: run.race.hasso_at)
    return latest.result.entry.tozai_cd, "last_run"


def build_context(race_input: RaceInput, repo: Repository, *,
                   pace_definition: dict | None = None) -> PredictionContext:
    """1レース分の ``PredictionContext`` を組み立てる。

    ``as_of`` は常に ``race_input.prediction_cutoff``。呼び出し側が別の時刻を
    渡す経路を作らない（CLAUDE.md の制約。この関数の外に as_of を引き回す
    パラメータを持たないのはそのため）。

    対象レースの条件（距離・トラック種別・開催場・馬場）は ``race_input`` から
    そのまま使い、``Repository.race()`` は呼ばない——利用者が申告した値が条件
    そのものであり、対象レースの行は ``as_of`` でゲートされ ``None`` のことも
    あるため。例外は2つだけで、どちらも利用者の入力に無い値を補う場合に限る。

    1. ``class_level``（今回のクラス順序値。``recent_form`` が前走との差分を
       出すのに必要）。``race_input.class_name`` は表示用の文字列で、順序値には
       ならない。対象レースの ``race_id`` が分かっていれば
       ``repo.race_conditions()`` から読む——``repo.race()`` ではない。
       ``class_level`` は発走前に確定するので結果行から読んでも漏洩にならず、
       ``race_conditions()`` はその列だけを返す型になっている。``race()`` を
       使っていたときは、結果行しか無いレース（実データでは結果行を持つ
       480レースすべてがこれで、出馬表行を併せ持つものは0件）で常に ``None``
       になり、``class_level_change`` が全馬 None に潰れていた（fix5 決定3）。
    2. 対象馬ごとの「登録後の変更」と「今回の所属トレセン」。``EntryRow`` にしか
       無い情報なので ``repo.entries()`` から読む（同じ理由で結果は含まれない
       ——``EntryRow`` は構造的に結果の項目を持てない）。
    """
    as_of = race_input.prediction_cutoff
    match_results = match_horses(race_input, repo, as_of=as_of)

    class_level: int | None = None
    entries_by_umaban: dict[str, EntryRow] = {}
    if race_input.race_id is not None:
        conditions = repo.race_conditions(race_input.race_id, as_of=as_of)
        if conditions is not None:
            class_level = conditions.class_level
        entries_by_umaban = {
            entry_row.umaban: entry_row
            for entry_row in repo.entries(race_input.race_id, as_of=as_of)
        }

    prepared: list[dict] = []
    for result in match_results:
        entry = result.entry
        runs = repo.past_runs(result.ketto_num, as_of=as_of) if result.ketto_num else []
        entry_row = entries_by_umaban.get(result.umaban)

        ability = ability_summary(runs)
        by_condition = condition_records(runs, race=race_input)
        recent = recent_form(runs, as_of=as_of, race=race_input,
                              futan_kg=entry.futan_kg, class_level=class_level)
        style = running_style(runs)
        tozai_cd, base_source = _current_tozai_cd(entry_row, runs)
        transport_features = transport(
            runs, tozai_cd=tozai_cd, jyo_cd=race_input.jyo_cd or "")

        # missing: **引けた過去走が0本なら**、過去走に依存する4群はまるごと
        # 計算不能。照合できたか（ketto_num の有無）ではなく実際に引けた走の
        # 数で立てる——馬マスタには居るが取得済みのデータ版に先行出走が
        # 1本も無い馬は、照合には成功するのに材料がゼロで、それでも欠損を
        # 申告しないまま ability は空・by_condition は 0戦 として state に
        # 載っていた。Jev からは「計算できなかった」ではなく「0戦0勝の馬」に
        # 読める。取得窓を遡れば同じ馬の 0本 は 3本 になりうるので、
        # 「何本引けたか」こそが欠損の実体（report2 §6 の #6）。
        #
        # transport はそれとは独立に、base が解決できなかった場合だけ足す
        # （地方・外国招待や、所属を推定する手がかりすら無い場合。stay_status
        # が既定の "unknown" であること自体はどの馬にも起きる通常の状態なので
        # missing の理由にしない——決定3）。
        missing_list: list[str] = list(_HISTORY_GROUPS) if not runs else []
        if transport_features.base is None:
            missing_list.append("transport")
        missing = tuple(missing_list)

        notes: list[str] = []
        # 照合側が具体的な理由を持っていればそれを使う（申告の馬齢に合う馬が
        # 居なかった／性別が食い違ったなど）。これを取り込まないと、照合が
        # 静かに unregistered を返す経路で理由が利用者にも state にも届かない。
        notes.extend(result.notes)
        if result.status == "unregistered" and len(result.notes) == 0:
            # 索引は馬マスタ（UM）と出走記録（SE）の両方から作られるので、
            # 「マスタ未登録」だけでは実態を表さない。どちらにも無かったのが理由。
            notes.append(
                f"{result.umaban}番は馬名が馬マスタにも取得済みの出走記録にも"
                "見つからなかった。過去実績なしとして扱った。")
        if entry_row is not None and entry_row.changed_after_entry:
            notes.append(f"{result.umaban}番は登録後に斤量・騎手が変更されている。")
        if base_source == "last_run":
            notes.append(
                f"{result.umaban}番は今回の出馬表がまだ無いため、"
                "所属トレセン（輸送起点）を直近走のデータから推定した。"
            )

        prepared.append({
            "umaban": result.umaban,
            "bamei": entry.bamei,
            "ketto_num": result.ketto_num,
            "entry": entry,
            "ability": ability,
            "recent": recent,
            "by_condition": by_condition,
            "running_style": style,
            "transport": transport_features,
            "transport_base_source": base_source,
            "missing": missing,
            "notes": tuple(notes),
        })

    # field_rank はレース内の全頭を横並びで見る必要があるので、各馬の ability
    # を出し終えたあとにまとめて計算する（ability.py の field_ranks は
    # 「計算済みの値の上でだけ働く純粋関数」として作られている）。
    ranks = field_ranks({
        item["umaban"]: item["ability"].recent_margin_sec_wavg for item in prepared
    })

    horses = tuple(
        HorseFeatures(
            umaban=item["umaban"],
            bamei=item["bamei"],
            ketto_num=item["ketto_num"],
            entry=item["entry"],
            ability=item["ability"],
            field_rank=ranks[item["umaban"]],
            recent=item["recent"],
            by_condition=item["by_condition"],
            running_style=item["running_style"],
            transport=item["transport"],
            transport_base_source=item["transport_base_source"],
            missing=item["missing"],
            notes=item["notes"],
        )
        for item in prepared
    )

    race_notes: list[str] = []
    if race_input.baba_cd is None:
        # 理由の文言は derive.BABA_UNKNOWN_REASON の1箇所だけから組む。
        # ここで別の文字列を書くと、理由を変えても注記が追随しない
        # （fix3 決定2: 同じ事実についての日本語文を2箇所に置かない）。
        race_notes.append(f"同馬場状態の集計は行っていない（{BABA_UNKNOWN_REASON}）。")

    return PredictionContext(
        race=race_input,
        horses=horses,
        data_version=repo.data_version,
        data_fetched_at=repo.data_fetched_at,
        notes=tuple(race_notes),
        pace_definition=pace_definition,
    )
