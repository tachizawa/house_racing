"""予想パイプラインが扱う型。

もっとも重要なのは EntryRow と ResultRow の分離。
予想側（features / predict）は EntryRow しか受け取れない。ResultRow を触るのは
評価側（evaluate）と、過去走を組み立てる data/repository.py だけ。
この分離が、対象レースの結果が state に混ざることを型で防ぐ。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from keiba import config


@dataclass(frozen=True)
class EntryRow:
    """出馬表相当。予想パイプラインが受け取れるのはこれだけ。

    futan_before_kg は変更があった場合のみ設定される（未設定は None）。
    **kisyu_code_before はそうではない**——JV-Data は「変更なし」を空白では
    なくゼロ埋め（``"00000"``）で入れてくるので、値の有無では変更の有無を
    判定できない。実データの race_horses.csv では 7,455 行中 7,389 行が
    ``"00000"``（= 変更なし）だった。判定は changed_after_entry に閉じ込めて
    あるので、この2列を直接読んで「入っていれば変更あり」と扱わないこと。
    """

    umaban: str
    wakuban: str
    ketto_num: str
    bamei: str
    sex_cd: str
    barei: str
    futan_kg: float | None
    kisyu_code: str
    tozai_cd: str
    chokyosi_code: str
    futan_before_kg: float | None = None
    kisyu_code_before: str = ""

    @property
    def changed_after_entry(self) -> bool:
        """斤量か騎手が変更されているか。

        ``kisyu_code_before`` の「変更なし」はゼロ埋め（``"00000"``）で来る。
        ``bool("00000".strip())`` は True なので、素直に真偽で見ると**ほぼ
        全頭が「変更あり」になる**——実際そうなっていて、1,358頭全頭に注記が
        付き、そのうち 99.1% が偽だった。ingest/records.py でも正規化して
        いるが、**すでに保存済みのデータ版には生の "00000" が残っている**ので、
        読む側のここでも「ゼロだけのコードは未設定」と読む。
        """
        kisyu_before = self.kisyu_code_before.strip()
        return self.futan_before_kg is not None or bool(kisyu_before.strip("0"))

    @property
    def sex_age(self) -> str:
        """「牡4」のような性齢の表示。馬齢が読めなければ**性別だけ**を返す。

        ``barei`` が正の整数として読めないとき（``"00"`` や空文字列）に
        「0歳」を作らない。0 は年齢としてありえない値で「不明」を意味し、
        ``repository._birth_year_from_race_row`` は同じ理由で ``years <= 0`` を
        意図的に不明にしている。ここだけ「0歳」を作ると、その値が入力の申告
        として照合に渡り、``_narrow_by_age`` が生年 = 当年 - 0 で絞って0件
        （その馬の過去走がすべて失われる）になり、しかも注記が「申告の馬齢
        （0歳）」と**入力について虚偽を述べる**——誰も 0 を申告していない
        （fix5 決定8）。
        """
        sex = config.SEX_CD_TO_NAME.get(self.sex_cd, "?")
        barei = self.barei.strip()
        if not barei.isdigit() or int(barei) <= 0:
            return sex
        return f"{sex}{int(barei)}"

    @property
    def base(self) -> str | None:
        """所属トレセン。美浦 / 栗東 / None（地方・外国招待など）。"""
        return config.TOZAI_CD_TO_BASE.get(self.tozai_cd)


@dataclass(frozen=True)
class ResultRow:
    """結果込み。評価側と過去走の組み立てだけが触る。"""

    entry: EntryRow
    ijyo_cd: str
    kakutei_jyuni: int | None
    dochaku_kubun: str
    dochaku_tosu: int | None
    time_sec: float | None
    margin_sec: float | None
    jyuni_1c: int | None
    jyuni_2c: int | None
    jyuni_3c: int | None
    jyuni_4c: int | None
    last_3f_sec: float | None
    ba_taijyu_kg: int | None
    zogen_sa_kg: int | None
    odds: float | None
    ninki: int | None

    @property
    def counts_as_starter(self) -> bool:
        """出走集合に含めるか。取消・発走除外・競走除外は走っていないので除く。"""
        return self.ijyo_cd not in config.IJYO_EXCLUDED

    @property
    def can_win(self) -> bool:
        """勝ち馬になり得るか。競走中止・失格は集合に残すが勝ち馬ではない。"""
        return self.counts_as_starter and self.ijyo_cd not in config.IJYO_NO_TIME

    @property
    def is_winner(self) -> bool:
        return self.can_win and self.kakutei_jyuni == 1


@dataclass(frozen=True)
class RaceRow:
    race_id: str
    hasso_at: datetime
    jyo_cd: str
    kyori_m: int
    track_cd: str
    track_kind: str | None
    baba_cd: str
    class_level: int | None
    grade_cd: str
    jyoken_cd_youngest: str
    toroku_tosu: int | None
    syusso_tosu: int | None
    nyusen_tosu: int | None
    first_3f_sec: float | None
    race_name: str

    @property
    def is_flat(self) -> bool:
        """平地か。初期版は障害を対象外にする。"""
        return self.track_kind in ("turf", "dirt")

    @property
    def jyo(self) -> str:
        return config.JYO_CD_TO_NAME.get(self.jyo_cd, self.jyo_cd)

    @property
    def baba(self) -> str | None:
        return config.BABA_CD_TO_NAME.get(self.baba_cd)


@dataclass(frozen=True)
class RaceConditions:
    """レースの条件のうち、**発走前に確定する列だけ**。発走前に確定する3列。

    ``class_level`` は ``grade_cd`` / ``jyoken_cd`` 由来でレース成立時に決まって
    いるので、結果行（data_kubun 3-7）から読んでも未来情報にはならない。
    ``EntryRow`` が出走馬について同じことをしているのと同じ構図で、こちらは
    レース側を担う。``grade_cd`` / ``jyoken_cd_youngest`` 自体も発走前に確定するが、
    どこからも読まれていないので置かない（fix6 ②）。

    **この型にフィールドを足すときは、その列が発走前に確定することを確かめ、
    ``tests/test_repository.py`` の allow-list
    （``test_race_conditions_returns_exactly_one_column``）も更新すること。**
    禁止名の一覧（deny-list）では足りない——リストに無い結果由来の列が素通り
    する（レビュアーが実測: ``lap_times_sec`` と ``race_name`` を足しても
    1069件のテストが全部通った）。``syusso_tosu`` / ``nyusen_tosu`` /
    ``first_3f_sec`` / ``lap_times_sec`` は結果由来、``baba_cd`` は当日確定。
    レース全体の行が必要なときは ``RaceRow``（``Repository.race()`` が as_of で
    ゲートする）を使う。
    """

    class_level: int | None
    #: 2009 トラックコード。コース（芝・右外など）は番組で決まり発走前に確定する。
    track_cd: str | None = None
    #: コース区分（仮柵の位置 A/B/C…）。開催前に発表される。空白は None。
    course_kubun_cd: str | None = None


@dataclass(frozen=True)
class PastRun:
    """過去走1件。レース条件と、その馬のその走の結果。"""

    race: RaceRow
    result: ResultRow

    @property
    def margin_sec(self) -> float | None:
        return self.result.margin_sec

    @property
    def has_time(self) -> bool:
        return self.result.ijyo_cd not in config.IJYO_NO_TIME and self.margin_sec is not None


@dataclass(frozen=True)
class HorseMaster:
    """馬マスタ（UM）のうち時点に依存しない項目だけ。

    UM の累計成績（Ruikei* / ChakuKaisu* / Kyakusitu）は「現在までの累計」で
    過去時点の値を復元できないため、この型には含めない。
    """

    ketto_num: str
    bamei: str
    # 出走記録（SE）から組み立てた HorseMaster は半角カナを持たない（SE に列が
    # 無い）。birth_date も同じ理由で None になりうる。
    bamei_kana: str | None
    birth_date: date | None
    sex_cd: str
    #: sex_cd をどこから採ったか。"race"（SE 行）または "master"（UM 行）。
    #: 去勢で変わる値なので、マスタ側は最後の更新時点で古びている（実測で
    #: 出走馬の 1.6% が食い違う）。食い違いを利用者に伝える注記でこの値を使う。
    #: 既定値は持たない。出どころを記録する欄が既定値を持つと、渡し忘れた
    #: 組み立てが黙って「マスタ由来」を主張し、注記が嘘をつく。
    sex_cd_source: str
    #: 生年。マスタの birth_date があればその年、無ければ出走記録から
    #: 「レースの年 - 馬齢」で計算する（実データ 950/950 で検証済みの規則）。
    #: 同名の馬を年齢で見分けるのに使う。どちらも取れなければ None。
    #: birth_date と別の欄にしているのは、日まで分かるのはマスタだけで、
    #: 年だけなら出走記録からでも分かるため（マスタには出走馬の 13.2% しか
    #: 居ないので、年齢の絞り込みをマスタに依存させられない）。
    birth_year: int | None
    #: birth_year をどこから採ったか。"master" / "race" / None（取れなかった）。
    #: sex_cd_source と同じ理由で既定値は持たない。
    birth_year_source: str | None


@dataclass(frozen=True)
class HorseEntry:
    """利用者が入力した1頭分。"""

    umaban: str
    bamei: str
    wakuban: str | None = None
    sex_age: str | None = None
    futan_kg: float | None = None
    kisyu: str | None = None


@dataclass(frozen=True)
class RaceInput:
    """利用者が入力したレース1件。

    entry_count は「出走予定頭数」。RaceRow.syusso_tosu（結果から算出される
    出走頭数）とは別物なので名前を分けている。
    """

    date: date
    jyo: str                     # 表示名。「中山」など
    race_num: int
    track_kind: str              # "turf" / "dirt"
    kyori_m: int
    entry_count: int
    prediction_cutoff: datetime
    horses: tuple[HorseEntry, ...]
    # コードに解決した値。過去走（jyo_cd / baba_cd を持つ）との突き合わせに使う。
    # 表示名のままでは条件別成績を集計できないので、入力を読む時点で解決する。
    jyo_cd: str | None = None
    baba_cd: str | None = None
    class_name: str | None = None
    baba: str | None = None      # 表示名。「良」など
    hasso_at: datetime | None = None
    top_n: int = config.DEFAULT_TOP_N
    race_id: str | None = None
