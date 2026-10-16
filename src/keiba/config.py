"""設定値とコード表。

コード表の値は vendor/jravan-sdk-500-64bit/ドキュメント/JV-Data仕様書_4.9.0.1.xlsx
の「コード表」シートから写したもの。vendor/ は再配布不可なので import せず、
必要な値だけをここに持つ。括弧内の番号は仕様書のコード表番号。
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# --- 版。いずれかが変わった予想は別の記録として扱う -------------------------

# CSV（データ版）の列構成の版。ingest/store.py が manifest に書き、data/store.py が
# 読む側の期待と突き合わせる。RECORD_SCHEMA_VERSION とは変わる理由が別なので
# 分けてある（列を1つ足しただけで保存済みの予想記録が読めなくなる、あるいは
# その逆が起きないようにするため。fix5 決定9-1）。
CSV_SCHEMA_VERSION = "1"

# 予想記録（store/records.py の JSONL）の形式の版。記録のフィールドが増減した
# ときだけ上げる。CSV の列構成とは無関係。
#
# "2": usage に3フィールド増やした（layer1_requests / layer2_requests / requests。
#      仕様 §13 の「requests」）。**"1" の記録は実在する**——data/predictions に
#      usage が3キーだけの記録が保存済みで、6キーの記録と同じファイルに並んでいる。
#      版を分けることで、どちらの形式かを記録自身から判別できる（"1" の記録に
#      対して usage["requests"] を引くと KeyError になるが、それが正しい——
#      無い値を 0 として扱うと「1回も投げなかった」という偽の主張になる）。
# "3": data_quality に calibration（較正を適用したか／しない理由・再正規化前の
#      合計・1.0 からのずれ・較正マップの定義域・端に張り付いた馬番）が増え、
#      calibrated_win_prob が null とは限らなくなった（report2 §6 の #8。
#      仕様 §11.2 が要求していた記録が、これまで実装されていなかった）。
#      読む側の期待が変わるので版を分ける。
RECORD_SCHEMA_VERSION = "3"

# 特徴量の定義の版。
# "2": by_condition の各区分に inapplicable_reason を追加し、条件が定義できないとき
#      starts/wins/top3 を 0 ではなく null にした（spec §6.3.1）。モデルが見る入力が
#      変わるので、"1" の記録と答えを直接比べてはいけない。
#      (fix4) state の中身自体は変えず、data_quality.notes をレース単位だけに絞り
#      horse_notes（馬番ごとの注記）を新設しただけなので、"2" のまま据え置く
#      ——モデルが見る入力（state）は変わっておらず、記録側の注記の置き場所が
#      変わっただけなので、"2" の記録どうしの比較可能性は保たれている。
# "3": state の中身が2箇所変わった（report2 §6 の #3 / #6）。
#      (a) EntryRow.changed_after_entry がゼロ埋めの "00000"（変更なし）を
#          「変更あり」と読まなくなり、偽の注記が data_quality.horse_notes から
#          消える。実データでは 1,358頭全頭に注記が付き、1,346頭（99.1%）が偽
#          だった。(b) data_quality.missing を「照合できたか」ではなく「引けた
#          過去走が0本か」で立てるようにしたので、欠損を申告する馬が増える。
#      どちらも**モデルが見る入力そのもの**なので、"2" の記録と答えを直接
#      比べてはいけない。
FEATURE_VERSION = "3"
INTEGRATION_VERSION = "1"
EXCEPTION_RULES_VERSION = "1"

# 第1層の勝率は同じ入力でも run ごとに揺れる。2頭の勝率の差がこの幅より小さいとき、
# その2頭の前後関係は再現しない（順位の形をしているが情報ではない）。
#
# 出典: 実レース8件・各5回・101頭ぶんの run 間の幅を測った実測値の95%点
#       （2026-09-22、中央値 0.0100 / 90%点 0.0200 / 95%点 0.0300 / 最大 0.0900）。
#       同じ測定で8レース中3レースの上位3頭が run ごとに入れ替わった。
#
# 注意: この値は第1層の勝率について測ったもの。**この定数を model 尺度
#       （model_score / model_win_prob）に当ててはいけない。当てるなら model 尺度で
#       測り直すこと。** 当てていたときの実害: model_score の幅が ±0.2 程度に
#       なる run（λ が大きい／学習レースが少ないとき十分ありうる）では18頭全部が
#       1つの組になり、「予想順位」を持つ馬が0頭＝製品の最終出力（上位n頭）が
#       消えたうえ、測っていない尺度のばらつきを主張することになる（fix5 決定1）。
TIE_BAND = 0.03

# --- 既定値 -----------------------------------------------------------------

# 資料（features/dossier.py）の形の版。キーや中身の定義が変わったら上げる。
# "2": 条件別実績の definition と履歴の note を各馬から top-level
#      （condition_definitions / history_status_notes）へ集約し、直近走の既定を
#      5走から4走に減らした（state size fix。実測 100レースのドライランで
#      17/100 が旧上限 48,000 文字を超えていた）。
# "3": data_quality.coverage_from（と履歴の found_may_be_truncated の判定）を
#      fromdate と first_hasso_at の遅いほうにした。option 4 の版では訂正再配信の
#      1989年などが入り、打ち切りの可能性が一度も付かなかった。形は "2" と同じ。
DOSSIER_VERSION = "3"
# 「3着以内」の 3。表示頭数（DEFAULT_TOP_N）とは別物なので分けて持つ。
TOP3_PLACE = 3
# 3着内の採点規則（evaluate/top3_metrics.py の docstring）の版。同着・取消・
# 失格の扱いを変えたら上げる。版が違う報告どうしは比べない。
TOP3_EVAL_RULES_VERSION = "1"
# 資料に載せる直近の過去走の件数。多いほど state が大きくなる（STATE_MAX_CHARS）。
# 5 -> 4: 定義文・注記の集約（DOSSIER_VERSION "2"）と合わせて実施。4走・
# STATE_MAX_CHARS=52,000 での実測（100レース、2026-01..09-22、--dry-run）は
# max 51,397 文字・0/100 が上限超過（旧: 5走・48,000 文字上限で 17/100 超過）。
DEFAULT_DOSSIER_RECENT_RUNS = 4
# state の上限（json.dumps(state, ensure_ascii=False) の文字数）。
# 実測: input_tokens / (state_chars + question_chars) = 0.583〜0.590
# （Task 14、9リクエスト、2026-09-23）。Jev の制約は「state + 最長の質問 ≤ 32k
# トークン」で、3着内の最長の質問は1,000文字未満。
# (32,000 - 1,000) / 0.59 ≈ 52,500 なので 52,000 に切り下げる。
STATE_MAX_CHARS = 52_000
# field_summary で「先行型」と数える4角の相対位置（corner_pos_mean）の上限。
FRONT_RUNNER_MAX_CORNER_POS = 1 / 3
# 見つかった最古の過去走がデータ版の収録開始からこの日数以内なら、それより前の
# 出走が版の外にある可能性がある。中央の出走間隔はおおむね3〜8週なので、
# 90日以内なら「さらに前にも走っていた」ことを否定できない。
HISTORY_TRUNCATION_MARGIN_DAYS = 90

DEFAULT_TOP_N = 3
DEFAULT_RECENT_RUNS = 5          # 直近何走を集計に使うか。仕様の初期案は 5〜10
SAME_DISTANCE_TOLERANCE_M = 200  # 「同距離」とみなす幅
LONG_TRANSPORT_KM = 300          # 「長距離輸送」とみなす閾値
PACE_MIN_RACES = 30              # ペース三分位を層別に計算するのに必要な最小レース数

# JVOpen の dataspec。RACE はレース系（RA/SE など）を運ぶが、競走馬マスタ UM は
# 運ばない。UM を運ぶのは蓄積系差分の DIFF（SDK の蓄積系提供データ一覧で、
# UM の差分ファイル名が UMFW…（3文字目 F = DIFF）であることから確認）。
# DIFF は差分なので、取得期間中に更新が無かった馬のマスタ行は含まれない。
# 照合はマスタの完全性に依存しない作りになっているので（data/store.py の
# name_index が race_horses.csv も見る）、DIFF で取れる分だけで十分。
DEFAULT_DATASPECS: tuple[str, ...] = ("RACE", "DIFF")

# 血統（SK 産駒マスタ / HN 繁殖馬マスタ / BT 系統情報）。BLOD ではなく BLDN を使う。
# BLOD は JV-Data 4.9.0（2023-08）より前の形式（繁殖登録番号8桁）のまま 2023-07 作成分で
# 止まっている（2026-09-25 実測。2023年生まれ以降の馬が入らない）。新形式（10桁）を
# 1986年以降ぶん運ぶのが BLDN（DIFF と DIFN の関係と同じ）。
# レースの版とは別の系統に保存する（ingest/store.py の BLOOD_FILES を参照）。
DEFAULT_BLOOD_DATASPECS: tuple[str, ...] = ("BLDN",)
BLOOD_DATA_DIR = DATA_DIR / "blood"

# 払戻（HR）とオッズ（O1〜O6）。RACE が運ぶが、レースの版とは別の系統に保存する
# （ingest/store.py の MARKET_FILES）。3連単のオッズは5年で約4,000万行あり、
# レースの版に混ぜると予想の読み込みが重くなるうえ、CURRENT を共有すると
# 片方の取り直しがもう片方を巻き込む。
DEFAULT_MARKET_DATASPECS: tuple[str, ...] = ("RACE",)
MARKET_DATA_DIR = DATA_DIR / "market"

# 入力の距離が現実的か判定するための幅。中央の平地はおよそ 1000m〜3600m だが、
# 障害・特殊条件の余地を残して広めに取る。
MIN_DISTANCE_M = 800
MAX_DISTANCE_M = 5000

# --- 2001. 競馬場コード（中央のみ。30番台以降は地方、A0 以降は海外） --------
JYO_CD_TO_NAME: dict[str, str] = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}

# --- 2301. 東西所属コード ---------------------------------------------------
# 0:その他 1:関東(美浦) 2:関西(栗東) 3:地方招待 4:外国招待
# 輸送の起点になるのはトレセン所属の 1 と 2 だけ。
TOZAI_CD_TO_BASE: dict[str, str] = {"1": "美浦", "2": "栗東"}

# --- 2202. 性別コード -------------------------------------------------------
SEX_CD_TO_NAME: dict[str, str] = {"1": "牡", "2": "牝", "3": "セン"}

# --- 2010. 馬場状態コード ---------------------------------------------------
BABA_CD_TO_NAME: dict[str, str] = {"1": "良", "2": "稍重", "3": "重", "4": "不良"}

# --- 2101. 異常区分コード ---------------------------------------------------
# 0:正常 1:出走取消 2:発走除外 3:競走除外 4:競走中止 5:失格 6:落馬再騎乗 7:降着
#
# 1/2/3 は走っていないので出走集合から抜く。
# 4/5 は走ったが着順がないので集合には残し、勝ち馬になり得ないものとして扱う。
# 6/7 は確定着順があるので通常扱いにする（ここで除外すると降着馬の着順を捨てる）。
IJYO_EXCLUDED = frozenset({"1", "2", "3"})
IJYO_NO_TIME = frozenset({"4", "5"})

# --- データ区分（レコードヘッダ）-------------------------------------------
# 1:出走馬名表(木) 2:出馬表(金土) 3-6:速報成績 7:成績(月) 9:中止 0:削除
# A:地方競馬 B:海外国際レース
DATA_KUBUN_RESULT_PRIORITY: tuple[str, ...] = ("7", "6", "5", "4", "3")
DATA_KUBUN_ENTRY_PRIORITY: tuple[str, ...] = ("2", "1")
DATA_KUBUN_DROP = frozenset({"0", "9", "A", "B"})

# 払戻（HR）とオッズ（O1〜O6）のデータ区分は RA/SE とも互いとも意味が違う。
#   HR   : 1:速報成績(払戻金確定) 2:成績(月曜) 9:レース中止 0:削除
#   O1〜O6: 1:中間 2:前日売最終 3:最終 4:確定 5:確定(月曜) 9:レース中止 0:削除
# 払戻金が確定しているのは HR の 1/2、確定オッズは O の 4/5。O の 1〜3 は確定前の値で、
# 確定オッズとは一致しない。同じレースに複数あれば先頭を優先する。0 と 9 は使わない。
# 注意: 確定オッズは締切後にしか分からない。「期待値で買い目を選ぶ」検証に確定オッズを
# 使うと、買う時点では知りえない値で選んでいることになる（締切直前との差の分だけ甘くなる）。
DATA_KUBUN_PAYOUT_PRIORITY: tuple[str, ...] = ("2", "1")
DATA_KUBUN_ODDS_PRIORITY: tuple[str, ...] = ("5", "4")

# UM（競走馬マスタ）のデータ区分は RA/SE と意味が違う。
#   1:新規馬名登録 2:馬名変更 3:再登録(抹消後の再登録) 4:その他更新
#   9:抹消        0:該当レコード削除(提供ミスなどの理由による)
# 9(抹消) は「実在の馬が引退した」という意味で、戦績は完全に残っている。
# バックテストで予想する過去レースの出走馬は現在の master ではほぼ全頭が抹消済みなので、
# これを落とすと過去レースの馬名照合がほぼ全滅する。落とすのは 0 だけ。
DATA_KUBUN_DROP_HORSE = frozenset({"0"})

# --- 輸送距離（推定値）------------------------------------------------------
# トレセンから各競馬場までの代表的な陸路距離の概数。単位 km。
# 実際の出発地・経路・輸送日を示すものではない。state に渡すときも推定値と明示する。
TRANSPORT_KM: dict[tuple[str, str], int] = {
    ("美浦", "01"): 1100, ("栗東", "01"): 1500,   # 札幌
    ("美浦", "02"): 950,  ("栗東", "02"): 1350,   # 函館
    ("美浦", "03"): 250,  ("栗東", "03"): 650,    # 福島
    ("美浦", "04"): 330,  ("栗東", "04"): 550,    # 新潟
    ("美浦", "05"): 110,  ("栗東", "05"): 480,    # 東京
    ("美浦", "06"): 90,   ("栗東", "06"): 500,    # 中山
    ("美浦", "07"): 400,  ("栗東", "07"): 130,    # 中京
    ("美浦", "08"): 500,  ("栗東", "08"): 20,     # 京都
    ("美浦", "09"): 540,  ("栗東", "09"): 60,     # 阪神
    ("美浦", "10"): 1100, ("栗東", "10"): 620,    # 小倉
}


def track_kind(track_cd: str) -> str | None:
    """2009. トラックコード → turf / dirt / jump。

    10-22: 平地 芝 / 23-29: 平地 ダート・サンド / 51-59: 障害
    """
    if not track_cd.strip().isdigit():
        return None
    value = int(track_cd)
    if 10 <= value <= 22:
        return "turf"
    if 23 <= value <= 29:
        return "dirt"
    if 51 <= value <= 59:
        return "jump"
    return None


# 2003. グレードコード。D はグレードのない重賞なので OP 相当の 5。
# E（重賞以外の特別競走）と空白（一般競走）はクラスを表さないので競走条件へ送る。
_GRADE_TO_CLASS_LEVEL: dict[str, int] = {"A": 8, "B": 7, "C": 6, "L": 5, "D": 5}


def class_level(grade_cd: str, jyoken_cd_youngest: str) -> int | None:
    """クラスを 0-8 の順序値にする。決まらなければ None（0 にしない）。

    ``jyoken_cd_youngest`` は RA の競走条件コード5要素のうち「最若年条件」
    （出走可能な最も馬齢が若い馬に対する条件）。レースのクラスを代表する。

    0 新馬・未出走 / 1 未勝利 / 2 1勝 / 3 2勝 / 4 3勝 / 5 OP・L・重賞
    6 G3 / 7 G2 / 8 G1
    """
    if grade_cd in _GRADE_TO_CLASS_LEVEL:
        return _GRADE_TO_CLASS_LEVEL[grade_cd]

    code = (jyoken_cd_youngest or "").strip()
    if code in ("701", "702"):
        return 0
    if code == "703":
        return 1
    if code == "999":
        return 5
    if code.isdigit() and code != "000":
        # 001-100 は収得賞金による条件。005/010/016 が 1勝/2勝/3勝クラス。
        value = int(code)
        if value <= 5:
            return 2
        if value <= 10:
            return 3
        if value <= 16:
            return 4
        return 5
    return None
