"""利用者が入力した馬名を血統登録番号（ketto_num）に解決する。

**この層のルール: 選べないときは選ばない。** 馬名が2頭以上の候補に一致し、
手元の情報では1頭に絞れないなら、それはエラーとしてユーザーへ返す。
違う馬の履歴で予想すると、結果はもっともらしい見た目のまま静かに壊れる
（別の馬の実績の上に予想が建つ）。下流のどこにも異常が出ないので、
ここで止めるしかない（仕様 14.3 節）。

未登録（この馬はデータの中に居ない）と、決めきれない（曖昧・矛盾）は別物として
扱う。未登録は「過去実績なし」として続行してよい（過去走が無い馬として扱われる
だけで、それ自体はデータの欠落ではない）。決めきれない場合は必ず例外にする。

未登録になるのは2通り。名前が0件ヒットのときと、**名前は一致したが申告された
馬齢に合う候補が1頭も居ないとき**である。後者も「居ない」と言い切れるのは、
馬齢 = レースの年 - 生年 が実データで2度検証されているから（950/950 の直接照合と
204/204 の突き合わせ）。生年が合わない馬はこの馬ではない。ここを「決めきれない」
として止めていたときは、実測で 466 レース中 31 件がそれだけで中止していた。
利用者の打ち間違いに備えて正しい入力のレースを捨てるのは不均衡なので、
備えは注記（同名馬の血統登録番号と生年を並べる）で行う。

決めきれないには2種類ある。名前が2頭以上に一致し馬齢・性別でも絞れない
「曖昧（ambiguous）」と、候補が複数あるのに申告された性別がそのどれの
sex_cd とも一致しない「矛盾（conflict）」。後者を unregistered にすると、
実際には履歴がある馬を「過去実績なし」という嘘の状態で予想へ進めてしまう
（data_quality がデータについて虚偽を語ることになる）。入力とマスタの
どちらが誤っているかコードには判断できないので、同名馬を勝手に選ばない
のと同じ理由で、conflict も選ばずに止める（仕様 14.3.1 節）。

**ただし候補が1頭なら、性別だけの食い違いでは止めない。** 名前（と馬齢）で
1頭に特定できているなら識別は成功しており、性別は識別の裏付けであって
識別子ではない。しかもデータ側の性別は古びうる（去勢された馬の UM 行が
更新されない。実測で出走馬の 1.6%）。レース全体を捨てるのは不均衡なので、
その馬として扱い、食い違いを notes に残す（黙って続行はしない）。

絞り込みの順序は **名前 -> 馬齢 -> 性別**。馬齢のほうが識別子として強い
（同名かつ同じ生年月日の馬は実データで0件）ため、性別より先に当てる。

このモジュールはファイル I/O をしない。読むのは Repository 経由のみ。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

from keiba import config
from keiba.data.repository import Repository
from keiba.models import HorseEntry, HorseMaster, RaceInput
from keiba.names import normalize_name

__all__ = [
    "AmbiguousMatchError",
    "MatchResult",
    "match_horses",
    "normalize_name",
]


# 性別名 -> コード の逆引き。config.SEX_CD_TO_NAME を単一の正とする
# （config.py には手を入れない。逆引きは使う側で都度組み立てる）。
_SEX_NAMES: tuple[str, ...] = tuple(config.SEX_CD_TO_NAME.values())

# 申告された馬齢に合う候補が1頭も居らず、未登録として扱ったときの注記。
# 打ち間違いを見つけられるよう、申告馬齢と同名馬の生年を並べる
# （血統登録番号は出さない。_describe_candidates 参照）。
_AGE_UNREGISTERED_NOTE = ("{umaban}番は申告の馬齢（{barei}歳）に合う馬がデータに"
                          "見つからなかった。過去実績なしとして扱った。"
                          "同名の馬の生年: {candidates}")

# 注記に並べる同名候補の上限。18頭立てで全馬に長い注記が付くと読めなくなるので、
# 超えたぶんは頭数だけ示す。
_NOTE_CANDIDATE_LIMIT = 3

# 申告された性別がデータと食い違うが、1頭に特定できたときの注記。
# どちらの出どころと食い違ったか（race = SE 行 / master = UM 行）まで示す。
_SEX_NOTE = ("{umaban}番は申告の性別（{declared}）とデータの性別"
             "（{actual}、出どころ={source}）が食い違っている。"
             "名前で1頭に特定できたのでその馬として扱った。")


@dataclass(frozen=True)
class _SexAgeHint:
    """sex_age（"牡4" のような自由入力）を1回だけ解析した結果。

    性別と馬齢を1箇所で解析して両方の絞り込みに渡す。同じ文字列を2箇所で
    解析すると、いつか片方だけが直される。読み取れなかった側は None
    （「その手がかりでは絞らない」の合図）。0 は「読み取れなかった」ではない
    ので、判定は必ず is None で行う。
    """

    sex_name: str | None
    barei: int | None


def _parse_sex_age(sex_age: str | None) -> _SexAgeHint:
    """sex_age から性別名と馬齢を読み取る。

    性別は先頭の 牡/牝/セン。いずれでもなければ None（解釈をあきらめる）。
    馬齢は最初に現れる数字。数字が無ければ None。どちらも「絞り込みに
    使わない」の合図として呼び出し側が扱う（解釈できない手がかりで正しい馬を
    落とさないため）。
    """
    if sex_age is None:
        return _SexAgeHint(sex_name=None, barei=None)
    sex_name = next((name for name in _SEX_NAMES if sex_age.startswith(name)), None)
    digits = re.search(r"\d+", sex_age)
    return _SexAgeHint(sex_name=sex_name,
                       barei=int(digits.group()) if digits is not None else None)


def _narrow_by_age(candidates: list[HorseMaster], hint: _SexAgeHint,
                    race_date: date) -> list[HorseMaster] | None:
    """申告された馬齢に合う生年の候補に絞る。

    馬齢の規則は実データで検証済み（950行中950行、100.00%）:

        馬齢 = レースの年 - 生年

    基準はレース自身の日付（RaceInput.date）。hasso_at は None になりうるので
    使わない。

    見るのは HorseMaster.birth_year。マスタの birth_date があればその年、
    無ければ出走記録から計算された年が入っている（repository 側で解決済み）。
    birth_date を直接見てはいけない——日まで分かるのはマスタだけで、マスタには
    出走馬の 13.2% しか居ないため、birth_date を条件にすると現役馬（出走馬の
    大半）で年齢が使えなくなる。

    馬齢が読めなければ None を返す（「絞り込みを行わなかった」の合図）。
    birth_year が None の候補は落とさない。生年が分からないだけで、その馬で
    ないとは言えないからである。

    **空のリストを返すことは「どの候補もこの馬ではない」を意味する**（生年が
    合わない馬は別の馬であり、この規則は実データで2度検証されている）。
    呼び出し側はそれを未登録として扱う。None（絞らなかった）と 0 件（絞った
    結果どれも違った）は別物なので、`is None` と `len(...) == 0` を必ず分けて
    判定すること。

    JRA は抹消後に馬名を再使用し、マスタは 1936 年まで遡るので同名の馬は
    普通に複数いる（実データで正規化名の 3.9%）。同名グループ 2,000 件を
    調べて生年月日が重なるものは 0 件で、この絞り込みで完全に解ける。
    """
    if hint.barei is None:
        return None
    birth_year = race_date.year - hint.barei
    return [c for c in candidates
            if c.birth_year is None or c.birth_year == birth_year]


def _narrow_by_sex(candidates: list[HorseMaster],
                    hint: _SexAgeHint) -> list[HorseMaster] | None:
    """sex_age が性別として解釈できたときだけ、その性別の候補に絞る。

    sex_age が未入力、または解釈できない書式なら None を返す（「絞り込みを
    行わなかった」の合図）。呼び出し側はこれを見て元の候補をそのまま使う。

    解釈できた場合はフィルタした結果をそのまま返す。これは 0 件になりうる
    （名前は一致したのに、申告された性別がどの候補とも一致しない場合）。
    その 0 件を黙って候補一覧へ戻すフォールバックはここではしない。
    0 件という結果自体を呼び出し側にそのまま伝える（候補が1頭なら注記つきの
    matched、複数なら conflict の材料になる）。
    """
    if hint.sex_name is None:
        return None
    return [c for c in candidates if config.SEX_CD_TO_NAME.get(c.sex_cd) == hint.sex_name]


def _describe_candidates(candidates: list[HorseMaster]) -> str:
    """注記に並べる同名候補の説明。**生年だけ**を、最大3頭まで。

    並べる候補はすべて birth_year を持っている（生年不明の候補が1頭でもあれば
    _narrow_by_age の結果は空にならず、この注記は作られない）。

    血統登録番号は出さない。この注記は ``to_user_view`` 経由で利用者に届くので、
    入力した馬とは別の馬の識別子まで見せることになる。注記の目的は「申告した
    馬齢の打ち間違いに気づける」ことで、それには生年だけで足りる
    （fix5 決定10）。
    """
    shown = ", ".join(f"{c.birth_year}年生"
                      for c in candidates[:_NOTE_CANDIDATE_LIMIT])
    rest = len(candidates) - _NOTE_CANDIDATE_LIMIT
    return shown if rest <= 0 else f"{shown}, 他 {rest} 頭"


@dataclass(frozen=True)
class MatchResult:
    """馬名照合の結果1頭分。

    - matched: ketto_num が入り、candidates は空。
    - unregistered: ketto_num は None、candidates も空。過去実績なしとして続行できる。
      名前が0件だった場合と、申告の馬齢に合う候補が0頭だった場合がある。
      後者は notes にその旨（申告馬齢と同名馬の素性）が入る。
    - ambiguous: ketto_num は None（選ばないので入れない）。candidates に
      絞りきれなかった候補全部が入る。ユーザーが選ぶための材料。
    - conflict: ketto_num は None。candidates には性別で絞り込む前の候補
      （名前と馬齢で絞った一覧）が入る。名前は一致したが、申告された性別が
      どの候補の sex_cd とも一致せず、しかも候補が複数ある状態。入力と
      データのどちらが誤っているかコードには判断できないので選ばない。

    notes は利用者に伝える注記。matched でも空とは限らない（申告と食い違う
    手がかりがあっても1頭に特定できた場合、黙って続行せず注記を残す）。
    features/context.py が馬単位の注記に混ぜる。
    """

    umaban: str
    entry: HorseEntry
    ketto_num: str | None
    status: str
    candidates: tuple[HorseMaster, ...]
    notes: tuple[str, ...] = ()


class AmbiguousMatchError(ValueError):
    """1頭でも「1頭に決めきれない」馬（ambiguous または conflict）があれば送出する。

    決めきれないには2種類あり、どちらもこの例外にまとめる。
    - ambiguous: 名前が2頭以上に一致し、馬齢・性別でも絞れない。
    - conflict: 候補が複数あるのに、申告された性別がそのどれの sex_cd とも
      一致しない。入力とデータのどちらが誤っているかコードには判断できない。
      候補が1頭なら注記つきの matched になるので、ここには来ない。
    両者を分けず同じ例外にしているのは、原則が同じだから
    （同名馬を勝手に選ばないのと同じ理由で、矛盾した手がかりでも選ばない）。
    unregistered（名前が0件、または申告の馬齢に合う候補が0頭）はここに含めない。
    それは異常ではなく、過去実績なしとして続行してよいため。

    results にはそのレース全頭分の MatchResult を持たせる（解決できた馬も
    含む）。ユーザーが一度で全部の問題を把握できるようにするため。1頭ずつ
    投げ返すとラウンドトリップのたびに次の問題が発覚する。

    クラス名は据え置く。後続タスクでこれを予想失敗として扱うときの型に使う。
    """

    def __init__(self, results: tuple[MatchResult, ...]) -> None:
        unresolved = [r for r in results if r.status in ("ambiguous", "conflict")]
        names = "、".join(f"{r.umaban}番 {r.entry.bamei}（{r.status}）" for r in unresolved)
        super().__init__(f"馬名が一意に決まりません: {names}")
        self.results = results


def match_horses(race_input: RaceInput, repo: Repository, *, as_of: datetime) -> list[MatchResult]:
    """レース1件分の HorseEntry を馬マスタ（ketto_num）に照合する。

    1頭に決めきれない馬（ambiguous または conflict）が1頭でもあれば、
    レース全頭分の結果を添えて AmbiguousMatchError を送出する（先に全頭
    処理してから判定する。1頭目が決めきれなくても残りの処理を続け、
    ユーザーが一度で全問題を見られるようにする）。名前が0件の馬と、申告の
    馬齢に合う候補が0頭の馬は unregistered として続行する。それは異常ではない
    （仕様 14.3.1 節: 「決めきれない（曖昧・矛盾）」と「未登録で過去実績なし」
    は別物）。
    """
    results: list[MatchResult] = []
    for entry in race_input.horses:
        normalized = normalize_name(entry.bamei)
        candidates = repo.find_horses_by_name(normalized, as_of=as_of)

        if len(candidates) == 0:
            # 名前が1件もマスタに無い。これは矛盾ではなく未登録。
            results.append(MatchResult(
                umaban=entry.umaban, entry=entry, ketto_num=None,
                status="unregistered", candidates=(),
            ))
            continue

        # 性別と馬齢は1箇所で解析し、同じ結果を両方の絞り込みに渡す。
        hint = _parse_sex_age(entry.sex_age)
        notes: list[str] = []

        # 絞り込みは 名前 -> 馬齢 -> 性別 の順。馬齢のほうが識別子として強い
        # （同名かつ同じ生年月日の馬は実データで0件）。性別を先に当てると、
        # 馬齢の合わない同名馬（性別だけが申告と一致）を選んでしまう。
        by_age = _narrow_by_age(candidates, hint, race_input.date)
        if by_age is None:
            # 馬齢が読めないので絞らない。「馬齢が分からない」は「馬齢が
            # 合わない」とは違うので、未登録にはしない。
            effective = candidates
        elif len(by_age) == 0:
            # 申告された馬齢に合う候補が1頭も居ない。生年が合わない馬は
            # この馬ではない（馬齢 = レースの年 - 生年 は実データで2度検証
            # 済み）ので、これは「この馬はデータの中に居ない」を意味する。
            # 決めきれない（ambiguous）ではなく未登録として続行する。候補を
            # 残して止めていた頃は、これだけで実測 31 件のレースが中止して
            # いた。打ち間違いへの備えは注記で足りる。
            results.append(MatchResult(
                umaban=entry.umaban, entry=entry, ketto_num=None,
                status="unregistered", candidates=(),
                notes=(_AGE_UNREGISTERED_NOTE.format(
                    umaban=entry.umaban, barei=hint.barei,
                    candidates=_describe_candidates(candidates)),),
            ))
            continue
        else:
            effective = by_age

        by_sex = _narrow_by_sex(effective, hint)
        if by_sex is not None and len(by_sex) == 0 and len(effective) == 1:
            # 申告された性別がデータと食い違うが、名前（と馬齢）で1頭に
            # 特定できている。識別は成功しており、性別は識別の裏付けで
            # あって識別子ではない。しかもデータ側の性別は古びうる
            # （去勢された馬の UM 行が更新されない。実測で出走馬の 1.6%）。
            # レース全体を捨てるのは不均衡なので、注記を残して続行する
            # （黙って続行はしない）。
            horse = effective[0]
            notes.append(_SEX_NOTE.format(
                umaban=entry.umaban, declared=hint.sex_name,
                actual=config.SEX_CD_TO_NAME.get(horse.sex_cd, horse.sex_cd),
                source=horse.sex_cd_source,
            ))
            results.append(MatchResult(
                umaban=entry.umaban, entry=entry, ketto_num=horse.ketto_num,
                status="matched", candidates=(), notes=tuple(notes),
            ))
            continue

        if by_sex is not None and len(by_sex) == 0:
            # 性別がどの候補とも一致せず、候補も複数。名前でも1頭に
            # 特定できていないので選ばない。candidates には性別で絞る前の
            # 候補を入れ、ユーザーがデータ側の記録を見て判断できるようにする。
            results.append(MatchResult(
                umaban=entry.umaban, entry=entry, ketto_num=None,
                status="conflict", candidates=tuple(effective), notes=tuple(notes),
            ))
            continue

        # by_sex が None なら絞り込みを行っていない（sex_age 未指定/解釈不能）
        # ので、直前までの候補をそのまま使う。
        final = effective if by_sex is None else by_sex
        if len(final) == 1:
            results.append(MatchResult(
                umaban=entry.umaban, entry=entry, ketto_num=final[0].ketto_num,
                status="matched", candidates=(), notes=tuple(notes),
            ))
        else:
            results.append(MatchResult(
                umaban=entry.umaban, entry=entry, ketto_num=None,
                status="ambiguous", candidates=tuple(final), notes=tuple(notes),
            ))

    results_tuple = tuple(results)
    if any(r.status in ("ambiguous", "conflict") for r in results_tuple):
        raise AmbiguousMatchError(results_tuple)
    return results
