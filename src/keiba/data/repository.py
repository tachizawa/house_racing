"""予想側が読む Repository と、評価側が読む ResultRepository。

このモジュールの存在理由はひとつ。**未来の情報を予想に混ぜないこと。**

- `Repository` の公開メソッドはすべて `as_of` をキーワード専用・既定値なしで取る。
  既定値を与えれば「うっかり as_of なしで呼ぶ」経路ができてしまうので、それを
  型のレベルで塞ぐ。過去走は `hasso_at < as_of` のものだけを返す（等号は含めない。
  同時刻に発走したレースは「その時点でまだ結果を知り得ない」ため）。
- `ResultRepository` は逆に `as_of` を一切取らない。評価（当たったかどうかを
  あとから確認する）専用であることをシグネチャそのもので示す。

もうひとつの核となるルールが DataKubun の優先順位（`_pick_by_data_kubun`）。
JV-Data は同じレースを 出走馬名表(木) → 出馬表(金土) → 速報成績 → 成績(月) の順に
何度も配信するため、CSV には同じキーの行が複数入っている。選び方を間違えると
「本当は未確定の値」を確定値として使ってしまう。この選択ロジックは1か所
（`_pick_by_data_kubun`）にまとめ、レース単位・馬単位のどちらからも同じ関数を通す。
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from keiba import config
from keiba.data.store import DataStore
from keiba.models import EntryRow, HorseMaster, PastRun, RaceConditions, RaceRow, ResultRow

# race() が「まだ結果を知り得ないレース」に使ってよい data_kubun。
# 出走馬名表(1)・出馬表(2) は発走前に配信されるため、常に安全。
_ENTRY_ONLY = config.DATA_KUBUN_ENTRY_PRIORITY
# 「手に入る中で最も情報の多い行」を選ぶときの優先順位。結果側を優先し、
# 結果がまだ無ければ出走馬名表/出馬表にフォールバックする。
_BEST_AVAILABLE = config.DATA_KUBUN_RESULT_PRIORITY + config.DATA_KUBUN_ENTRY_PRIORITY


def _pick_by_data_kubun(rows: pd.DataFrame, priority: tuple[str, ...]) -> pd.DataFrame:
    """同じ (race_id, umaban) の複数レコードから1行を選ぶ。

    JV-Data は木曜→金土→速報→月曜と同じレースを何度も配信するので、
    CSV には同じキーの行が複数入っている。優先順位で1行に絞る。
    """
    rows = rows[~rows["data_kubun"].isin(config.DATA_KUBUN_DROP)]
    if rows.empty:
        return rows
    rank = {kubun: i for i, kubun in enumerate(priority)}
    rows = rows[rows["data_kubun"].isin(rank)].copy()
    rows["_rank"] = rows["data_kubun"].map(rank)
    rows = rows.sort_values(["_rank", "make_date"], ascending=[True, False])
    keys = ["race_id", "umaban"] if "umaban" in rows.columns else ["race_id"]
    return rows.drop_duplicates(subset=keys, keep="first").drop(columns="_rank")


def _int_or_none(value: str) -> int | None:
    """空文字列は None。0 という値そのものは None にしない。"""
    value = (value or "").strip()
    return int(value) if value else None


def _float_or_none(value: str) -> float | None:
    value = (value or "").strip()
    return float(value) if value else None


def _date_or_none(value: str) -> date | None:
    value = (value or "").strip()
    return date.fromisoformat(value) if value else None


def _to_race_row(row: pd.Series) -> RaceRow:
    return RaceRow(
        race_id=row["race_id"],
        hasso_at=datetime.fromisoformat(row["hasso_at"]),
        jyo_cd=row["jyo_cd"],
        kyori_m=int(row["kyori_m"]),
        track_cd=row["track_cd"],
        track_kind=row["track_kind"] or None,
        baba_cd=row["baba_cd"],
        class_level=_int_or_none(row["class_level"]),
        grade_cd=row["grade_cd"],
        jyoken_cd_youngest=row["jyoken_cd_youngest"],
        toroku_tosu=_int_or_none(row["toroku_tosu"]),
        syusso_tosu=_int_or_none(row["syusso_tosu"]),
        nyusen_tosu=_int_or_none(row["nyusen_tosu"]),
        first_3f_sec=_float_or_none(row["first_3f_sec"]),
        race_name=row["race_name"],
    )


def _to_entry_row(row: pd.Series) -> EntryRow:
    """EntryRow に必要な列だけを取り出す。結果側の列には一切触れない。"""
    return EntryRow(
        umaban=row["umaban"],
        wakuban=row["wakuban"],
        ketto_num=row["ketto_num"],
        bamei=row["bamei"],
        sex_cd=row["sex_cd"],
        barei=row["barei"],
        futan_kg=_float_or_none(row["futan_kg"]),
        kisyu_code=row["kisyu_code"],
        tozai_cd=row["tozai_cd"],
        chokyosi_code=row["chokyosi_code"],
        futan_before_kg=_float_or_none(row["futan_before_kg"]),
        kisyu_code_before=row["kisyu_code_before"],
    )


def _to_result_row(row: pd.Series) -> ResultRow:
    return ResultRow(
        entry=_to_entry_row(row),
        ijyo_cd=row["result_ijyo_cd"],
        kakutei_jyuni=_int_or_none(row["result_kakutei_jyuni"]),
        dochaku_kubun=row["result_dochaku_kubun"],
        dochaku_tosu=_int_or_none(row["result_dochaku_tosu"]),
        time_sec=_float_or_none(row["result_time_sec"]),
        margin_sec=_float_or_none(row["result_margin_sec"]),
        jyuni_1c=_int_or_none(row["result_jyuni_1c"]),
        jyuni_2c=_int_or_none(row["result_jyuni_2c"]),
        jyuni_3c=_int_or_none(row["result_jyuni_3c"]),
        jyuni_4c=_int_or_none(row["result_jyuni_4c"]),
        last_3f_sec=_float_or_none(row["result_last_3f_sec"]),
        ba_taijyu_kg=_int_or_none(row["result_ba_taijyu_kg"]),
        zogen_sa_kg=_int_or_none(row["result_zogen_sa_kg"]),
        odds=_float_or_none(row["result_odds"]),
        ninki=_int_or_none(row["result_ninki"]),
    )


def _birth_year_from_race_row(row: pd.Series) -> int | None:
    """出走記録（SE）の1行から生年を計算する。

        生年 = レースの年 - 馬齢

    馬齢 = レースの年 - 生年 は実データ 950 行中 950 行（100.00%）で成立する。
    レースの年は race_id の先頭4桁（例 2026090601020612 -> 2026）。

    barei が正の整数として読めなければ計算せず None を返す。**特に barei="00"
    を「今年生まれ」にしない。** 0 は年齢としてありえない値で「不明」を意味し、
    実在の馬齢として扱うとその馬だけ生年が当年になり、年齢の絞り込みで静かに
    落ちる。race_id の先頭4桁が数字でない場合も同じく None（例外にして
    レース全体を止めるより、その馬の生年を「不明」にするほうが害が小さい）。
    """
    barei = (row["barei"] or "").strip()
    race_year = (row["race_id"] or "")[:4]
    if not barei.isdigit() or not race_year.isdigit():
        return None
    years = int(barei)
    if years <= 0:
        return None
    return int(race_year) - years


def _to_horse_master(row: pd.Series, race_row: pd.Series | None = None) -> HorseMaster:
    """馬マスタ（UM）の行から HorseMaster を組み立てる。

    race_row（同じ馬の SE 行。as_of 以前の最新行）があれば sex_cd はそちらから
    採る。sex_cd は去勢で変わる時点依存の値で、UM 行は最後に更新された時点の値
    しか持たない。実データでは出走馬 5,177 頭のうち 82 頭（1.6%）が食い違い、
    向きはすべて SE=3（セン）/ UM=1（牡）だった（UM 行が古い）。SE はその
    レース時点の値を持つので、SE 行があるならそちらが正しい。

    birth_year はマスタの birth_date があればその年（"master"）、無ければ
    同じ race_row から計算する（"race"）。どちらも取れなければ None / None。
    birth_date 自体はマスタのみで、SE から日付をこしらえることはしない。
    """
    if race_row is None:
        sex_cd, sex_cd_source = row["sex_cd"], "master"
    else:
        sex_cd, sex_cd_source = race_row["sex_cd"], "race"
    birth_date = _date_or_none(row["birth_date"])
    if birth_date is not None:
        birth_year, birth_year_source = birth_date.year, "master"
    elif race_row is not None:
        birth_year = _birth_year_from_race_row(race_row)
        birth_year_source = "race" if birth_year is not None else None
    else:
        birth_year, birth_year_source = None, None
    return HorseMaster(
        ketto_num=row["ketto_num"],
        bamei=row["bamei"],
        bamei_kana=row["bamei_kana"],
        birth_date=birth_date,
        sex_cd=sex_cd,
        sex_cd_source=sex_cd_source,
        birth_year=birth_year,
        birth_year_source=birth_year_source,
    )


def _to_se_horse_master(row: pd.Series) -> HorseMaster:
    """出走記録（SE）から馬マスタ相当を組み立てる。

    SE には bamei_kana も birth_date も無い列なので None のまま返す
    （半角カナでの照合ができなくなる劣化。マスタが空・不完全でも出走記録
    さえあれば名前で引けることを優先する）。

    生年は「年だけ」なら barei と race_id から計算できるので入れる。
    マスタに居ない馬（実測で出走馬の 86.8%）でも年齢で同名を見分けられる
    ようにするため、ここが効く。
    """
    birth_year = _birth_year_from_race_row(row)
    return HorseMaster(
        ketto_num=row["ketto_num"],
        bamei=row["bamei"],
        bamei_kana=None,
        birth_date=None,
        sex_cd=row["sex_cd"],
        sex_cd_source="race",
        birth_year=birth_year,
        birth_year_source="race" if birth_year is not None else None,
    )


def _newest_by_ketto_num(rows: pd.DataFrame, ketto_nums: list[str],
                          drop: frozenset[str], *,
                          as_of: datetime) -> dict[str, pd.Series]:
    """候補の ketto_num に絞り、drop に該当せず make_date が as_of 以前の行から、
    ketto_num ごとに make_date が最も新しい行を1つ選ぶ。

    make_date は日付だけで時刻を持たないので、as_of と同日に作られた行は含める。
    前日18:00 の締切に対して同日に作られる SE 行は出馬表の行なので、含めるのが
    正しい。1日ぶん寛容な側に倒すのは、出馬表を取りこぼすより安全だからである。

    これで対象レース自身の結果行（make_date はレース後）が候補から外れ、
    そのレース時点の値（特に sex_cd）が採られる。絞った結果その馬の行が1つも
    残らなければ、その馬は候補にならない（as_of の規律どおり）。

    UM・SE の両方に使う共通処理だが、drop（除外集合）は呼び出し側が渡す。
    UM と SE ではデータ区分の意味が違う（同じ "9" が UM では抹消、SE では
    中止）ため、ここで固定の集合を使ってしまうと取り違えが起きる。

    make_date が同値のときは安定ソート（kind="mergesort"）で CSV 順の先頭が
    残る。データ版は不変なので、同じ版なら常に同じ答えになる（pandas 既定の
    quicksort は安定性を保証しないため、明示する）。同値どうしのどちらが
    望ましいかという序列は付けない。必要なのは決定性だけ。
    """
    rows = rows[rows["ketto_num"].isin(ketto_nums)]
    rows = rows[~rows["data_kubun"].isin(drop)]
    rows = rows[rows["make_date"] <= as_of.strftime("%Y%m%d")]
    if rows.empty:
        return {}
    rows = rows.sort_values("make_date", ascending=False, kind="mergesort")
    rows = rows.drop_duplicates(subset=["ketto_num"], keep="first")
    return {row["ketto_num"]: row for _, row in rows.iterrows()}


class Repository:
    """予想用のデータ参照。

    全公開メソッドが as_of を必須のキーワード引数に取る。既定値を与えないのは、
    うっかり as_of なしで呼べる経路を作らないため。テストで signature を検査する。
    評価用（結果を見る参照）は ResultRepository に分けてある。
    """

    def __init__(self, store: DataStore) -> None:
        self._store = store

    @property
    def data_version(self) -> str:
        return self._store.version_id

    @property
    def data_fetched_at(self) -> datetime:
        return self._store.fetched_at

    @property
    def coverage_from(self) -> datetime | None:
        """データ版に入っている最も早い発走（manifest 由来）。不明なら None。"""
        return self._store.coverage_from

    def past_runs(self, ketto_num: str, *, as_of: datetime,
                  limit: int | None = None) -> list[PastRun]:
        """as_of より前に発走した平地の走を、新しい順に返す。

        limit は返す走数の上限。limit=None が「全部」で、limit=0 は「0走」
        （空リスト）。偽値と欠損を同じ扱いにしない。

        採否の判定は _latest_known_hasso_at（そのレースについて知っている
        すべての版の発走時刻の最大値）で行う。race() と同じ規則を通すことで、
        同じデータに対して両者が食い違う答えを返さないようにしている
        （単一の版の hasso_at だけを信じると、その版だけがまだ古い時刻を
        持っている場合に取りこぼす／混ぜてしまう）。返す RaceRow 自体は
        従来どおり _race_row（手に入る中で最も情報の多い単一の行）から作る。

        既知の限界: RA 側がまだ出走馬名表/出馬表（1/2）のままで、成績側の
        SE だけが速報・確定（3〜7）に進んでいる場合、_latest_known_hasso_at
        は RA 側の古い時刻しか見つけられず、真の発走時刻を見逃しうる
        （このモジュールの元レポートで「RA と SE が歩調を合わせて進む」と
        仮定した箇所に相当する）。本システムの prediction_cutoff は前日夜
        （仕様 §1.2）なので、この穴が as_of を跨ぐには数時間単位の遅延が
        必要で、当面は実害にならない。当日ギリギリの予想を扱うようになる
        前に仕様 §3.2.2 を解決すること。
        """
        rows = self._store.race_horses
        rows = rows[rows["ketto_num"] == ketto_num]
        rows = _pick_by_data_kubun(rows, config.DATA_KUBUN_RESULT_PRIORITY)
        runs: list[PastRun] = []
        for _, row in rows.iterrows():
            race_id = row["race_id"]
            latest_hasso_at = self._latest_known_hasso_at(race_id)
            if latest_hasso_at is None or latest_hasso_at >= as_of:   # 同時刻も含めない
                continue
            race = self._race_row(race_id)
            if race is None or not race.is_flat:
                continue
            runs.append(PastRun(race=race, result=_to_result_row(row)))
        runs.sort(key=lambda run: run.race.hasso_at, reverse=True)
        # limit=0（「0走使う」）と limit=None（「全部」）は別物。`if limit` だと
        # 0 が None と同じ扱いになり、0走と言ったのに全走返ってしまう。
        return runs[:limit] if limit is not None else runs

    def find_horses_by_name(self, normalized_name: str, *,
                             as_of: datetime) -> list[HorseMaster]:
        """馬名（呼び出し側で正規化済み）から馬マスタを探す。

        索引（DataStore.name_index）は馬マスタ（UM, horses.csv）と出走記録
        （SE, race_horses.csv）の両方から作られている。JVOpen("RACE") は
        UM を返さないことがあり、UM を運ぶ JV-Data の差分配信（DIFF）も
        取得期間中に更新の無かった馬を含まないため、マスタは空・不完全で
        ありうる。一方 SE には、**このデータ版の取得期間に走った**全頭の
        bamei・ketto_num が入っている（取得期間の外で休養していた馬は
        入らない）。「この版に履歴がある」ことと「SE に出てくる」ことは
        同値なので、SE だけからでも名前で引けなければ、照合の成否が運用者が
        どう更新を走らせたかという運用の偶然に依存してしまう。

        同じ ketto_num が両方にあれば、欄ごとに出どころを分ける。**時点に依存
        する sex_cd だけは SE 行から採り**、bamei_kana・birth_date のような
        SE に無い欄はマスタから採る（sex_cd_source にどちらから採ったかを残す）。
        sex_cd は去勢で変わるのに UM 行は最後の更新時点の値しか持たず、実測で
        出走馬の 1.6% が食い違っていた（向きはすべて SE=セン / UM=牡）。
        SE にしか無い馬は bamei_kana と birth_date が None になる（半角カナでの
        照合ができなくなる劣化。マスタの完全性に予想の成立そのものが依存する
        よりはよい）。

        索引側も同じ normalize_name で正規化してある。bamei・bamei_kana
        （マスタ側のみ）のどちらから正規化しても同じキーになった馬は同じ
        候補として扱う。生の bamei とただ比較するだけでは、NFKC で畳めない
        長音符「ー」を含む馬名（JRA の
        馬名では珍しくない）が正規化後の名前と一致せず、すべて未登録扱いに
        なってしまう。

        除外集合は出どころで別に扱う。UM 行には DATA_KUBUN_DROP_HORSE、
        SE 行には DATA_KUBUN_DROP を使う。同じ "9" が UM では「抹消」
        （引退した実在の馬。戦績は完全に残る）、SE では「中止」を意味する、
        別のコード表だからである（config.py 参照）。

        as_of は行の選択に効く。_newest_by_ketto_num が make_date の as_of 以前
        の行しか見ないので、対象レース自身の結果行（make_date はレース後）は
        候補にならず、そのレース時点の sex_cd が採られる。索引
        （DataStore.name_index）は cached_property で as_of を持たないが、
        索引の広さ（後から知られる名前も含む）はそのままでよい。索引に名前が
        あっても、as_of 以前の行が1つも無ければ候補にならない。
        """
        candidate_ketto_nums = self._store.name_index.get(normalized_name, [])
        if not candidate_ketto_nums:
            return []
        um_rows = _newest_by_ketto_num(self._store.horses, candidate_ketto_nums,
                                        config.DATA_KUBUN_DROP_HORSE, as_of=as_of)
        se_rows = _newest_by_ketto_num(self._store.race_horses, candidate_ketto_nums,
                                        config.DATA_KUBUN_DROP, as_of=as_of)
        result: list[HorseMaster] = []
        # bamei と bamei_kana の両方が同じキーに正規化されて索引の候補リストに
        # 同じ ketto_num が重複して入っていても、dict.fromkeys で一度にまとめる。
        for ketto_num in dict.fromkeys(candidate_ketto_nums):
            um_row = um_rows.get(ketto_num)
            se_row = se_rows.get(ketto_num)
            if um_row is not None:
                # 両方にあれば sex_cd は SE から採り、生年はマスタの
                # birth_date が無いときだけ SE から計算する（se_row が None
                # ならどちらもマスタの値がそのまま使われる）。
                result.append(_to_horse_master(um_row, se_row))
            elif se_row is not None:
                result.append(_to_se_horse_master(se_row))
        return result

    def entries(self, race_id: str, *, as_of: datetime) -> list[EntryRow]:
        """出走予定馬を EntryRow で返す。

        優先順位は出馬表(2) → 出走馬名表(1)。どちらも無ければ結果側の行から
        EntryRow に該当する列だけを取り出す（ResultRow は作らない。EntryRow に
        結果の項目が存在しないことが、対象レースの結果が予想側へ混ざらないための
        型レベルの防波堤になっている）。

        as_of はここでのフィルタには使わない。race() と違い、どの data_kubun
        から読んでも EntryRow は構造的に結果の項目を持てない（上記の防波堤）ため、
        どちらの版を選ぶかで未来情報が漏れることがない。他の公開メソッドと同じ
        「as_of なしでは呼べない」規律を保つために必須にしているだけ。
        """
        rows = self._store.race_horses
        rows = rows[rows["race_id"] == race_id]
        entry_rows = _pick_by_data_kubun(rows, config.DATA_KUBUN_ENTRY_PRIORITY)
        if entry_rows.empty:
            entry_rows = _pick_by_data_kubun(rows, config.DATA_KUBUN_RESULT_PRIORITY)
        entry_rows = entry_rows.sort_values("umaban")
        return [_to_entry_row(row) for _, row in entry_rows.iterrows()]

    def race_conditions(self, race_id: str, *, as_of: datetime) -> RaceConditions | None:
        """レース前に確定する条件だけを返す。行が無ければ None。

        **返すのは ``class_level`` の1列だけ。この列は発走前に確定するので、
        結果行から読んでも漏洩にならない。返す列を増やすときは、その列が発走前に
        確定することを確かめ、``RaceConditions`` の allow-list テストも更新する
        こと。** 確かめられない列は ``RaceRow``（``race()`` が as_of でゲートする）
        側に置いたままにする。

        ``race()`` との違いはここだけ。``race()`` は「レースが終わらないと
        決まらない項目」も同じ型に持つため、発走が as_of 以降なら結果を含みうる
        行（data_kubun 3-7）を一切使えず、出馬表行が無いレースでは ``None`` に
        なる。実データの races.csv は data_kubun が {"2": 84, "7": 480} で、
        結果行を持つ480レースのうち出馬表行を併せ持つものは0件だった——つまり
        バックテストの全レースで ``class_level`` が取れず、``class_level_change``
        と ``class_level_change_missing`` の2列が常に無効になっていた（学習時と
        推論時で特徴分布が系統的に違う。fix5 決定3）。

        行の選択は ``entries()`` と同じで、``_pick_by_data_kubun`` の優先順位
        （出馬表(2) → 出走馬名表(1)、無ければ結果側）を使う。``as_of`` は
        フィルタには使わない——どの版から読んでも返す列は発走前に確定して
        いるので、版の選択で未来情報が漏れることがない。他の公開メソッドと
        同じ「as_of なしでは呼べない」規律を保つために必須にしてある。
        """
        rows = self._store.races[self._store.races["race_id"] == race_id]
        picked = _pick_by_data_kubun(rows, config.DATA_KUBUN_ENTRY_PRIORITY)
        if picked.empty:
            picked = _pick_by_data_kubun(rows, config.DATA_KUBUN_RESULT_PRIORITY)
        if picked.empty:
            return None
        return RaceConditions(class_level=_int_or_none(picked.iloc[0]["class_level"]))

    def race(self, race_id: str, *, as_of: datetime) -> RaceRow | None:
        """race_id のレース条件を返す。

        RaceRow は toroku_tosu/syusso_tosu/nyusen_tosu/last_3f_sec など、
        レースが終わらないと決まらない項目も同じ型に持つ（EntryRow と違い、
        型では守られていない）。データ版そのものは as_of とは無関係に「現在
        取得できる全件」を含んでいるため、対象レースの発走が as_of 以降なら、
        結果を含みうる data_kubun（3-7）の行は一切使わない。出走馬名表/出馬表
        （1/2）だけから組み立て、それも無ければ安全に答えられないので None。

        採否の判定は _latest_known_hasso_at を使う。past_runs と同じ規則を
        通すことで、同じデータに対して両者が食い違う答えを返さないように
        している。
        """
        rows = self._store.races[self._store.races["race_id"] == race_id]
        entry_rows = _pick_by_data_kubun(rows, _ENTRY_ONLY)
        entry_race = _to_race_row(entry_rows.iloc[0]) if not entry_rows.empty else None

        full_race = self._race_row(race_id)
        if full_race is None:
            return None
        latest_hasso_at = self._latest_known_hasso_at(race_id)
        if latest_hasso_at is not None and latest_hasso_at >= as_of:
            return entry_race
        return full_race

    def _race_row(self, race_id: str) -> RaceRow | None:
        """race_id について、手に入る中で最も情報の多い行から RaceRow を作る。

        as_of の判断はここでは行わない。past_runs・race() のどちらも、
        含めてよいかどうかを _latest_known_hasso_at で判断したあとにこれを
        呼び、返す RaceRow の中身だけをここから作る。
        """
        rows = self._store.races[self._store.races["race_id"] == race_id]
        rows = _pick_by_data_kubun(rows, _BEST_AVAILABLE)
        if rows.empty:
            return None
        return _to_race_row(rows.iloc[0])

    def _latest_known_hasso_at(self, race_id: str) -> datetime | None:
        """そのレースについて分かっているすべての版の発走時刻の最大値。

        RA.発走時刻（位置874）は版ごとに書き換わり、変更前の値は
        変更前発走時刻（位置878、区分3〜7で設定）に移る（JV-Data 仕様書
        RA レコード）。past_runs と race() の両方が「as_of 以降に発走した
        （＝知り得ないはずの）レースかどうか」を、単一の版の hasso_at ではなく
        必ずこれを通して判定する。一枚の版だけを信じると、その版だけが
        古い時刻のままの場合に未来の情報が漏れる（過去の修正で見つかった
        不具合）。どの版が「正しい」かは版だけからは決められないので、
        最も保守的な読み（最大値）を採る。

        RA 側がまだ更新されておらず、結果側（SE）だけが先に進んでいる場合は
        この関数からは判定できない。past_runs のドキュメントに、その限界と
        実害が無視できる理由（prediction_cutoff が前日夜であること）を書いた。
        """
        rows = self._store.races[self._store.races["race_id"] == race_id]
        rows = rows[~rows["data_kubun"].isin(config.DATA_KUBUN_DROP)]
        if rows.empty:
            return None
        return max(datetime.fromisoformat(value) for value in rows["hasso_at"])


class ResultRepository:
    """評価専用のデータ参照。as_of を取らない。

    予想が当たったかどうかを後から確認するためだけに使う。Repository と
    クラスを分けているのは、「結果を見てよい場面」と「見てはいけない場面」を
    呼び出す型そのものから区別できるようにするため。
    """

    def __init__(self, store: DataStore) -> None:
        self._store = store

    def race(self, race_id: str) -> RaceRow | None:
        rows = self._store.races[self._store.races["race_id"] == race_id]
        rows = _pick_by_data_kubun(rows, config.DATA_KUBUN_RESULT_PRIORITY)
        if rows.empty:
            return None
        return _to_race_row(rows.iloc[0])

    def results(self, race_id: str) -> list[ResultRow]:
        """出走したすべての馬の結果を返す。除外・失格などをどう扱うかは
        呼び出し側（評価ロジック）の判断に任せ、ここでは絞らない。"""
        rows = self._store.race_horses[self._store.race_horses["race_id"] == race_id]
        rows = _pick_by_data_kubun(rows, config.DATA_KUBUN_RESULT_PRIORITY)
        rows = rows.sort_values("umaban")
        return [_to_result_row(row) for _, row in rows.iterrows()]

    def winners(self, race_id: str) -> list[str]:
        """1着馬の umaban。同着があれば複数返る。

        ResultRow.is_winner（can_win かつ kakutei_jyuni == 1）を使う。
        失格・競走中止は着順の数字が残っていても can_win が False になるため、
        ここで kakutei_jyuni だけを見ると誤って勝ち馬に含めてしまう。
        """
        return [result.entry.umaban for result in self.results(race_id) if result.is_winner]

    def races_between(self, start: datetime, end: datetime) -> list[RaceRow]:
        """[start, end] に発走した平地レースを、発走が早い順に返す。両端を含む。"""
        rows = _pick_by_data_kubun(self._store.races, config.DATA_KUBUN_RESULT_PRIORITY)
        races = [_to_race_row(row) for _, row in rows.iterrows()]
        races = [race for race in races if race.is_flat and start <= race.hasso_at <= end]
        races.sort(key=lambda race: race.hasso_at)
        return races
