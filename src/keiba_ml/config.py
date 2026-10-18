"""機械学習系の設定。

設定は TOML（既定 ``configs/ml/pedigree.toml``）から読む。学習期間・テスト期間・
血統の世代数・CatBoost の目的関数など、**比較実験で切り替える値はすべてここを
通す**（設計原則 3「学習期間を設定で切り替えられるようにする」）。

コードに直書きしてよいのは「決めた値」だけ。未決定事項（開発方針 §9）に
当たる値は必ず設定に出し、既定値は仮置きであることをコメントで示す。
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, replace
from datetime import date
from pathlib import Path
from typing import Any, get_args, get_type_hints

from keiba import config as keiba_config

PROJECT_ROOT = keiba_config.PROJECT_ROOT
#: 既存モデルのデータ版（races.csv / race_horses.csv）の置き場所。読むだけで書かない。
RACE_DATA_DIR = keiba_config.DATA_DIR
#: 既存の血統の版（pedigrees.csv / breeding_horses.csv。dataspec BLDN）の置き場所。読むだけ。
BLOOD_DATA_DIR = keiba_config.BLOOD_DATA_DIR
#: 機械学習系の出力（血統の版・データセット・モデル・予測・評価）。/data/ 配下なので
#: .gitignore 済み（JRA-VAN のデータは再配布不可）。
ML_DATA_DIR = keiba_config.DATA_DIR / "ml"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "ml" / "pedigree.toml"

#: 3代血統の14枠の並び（JV-Data の「3代血統情報」の配列順）。
#: S=父（sire）、D=母（dam）。左から辿る。例: "DS" = 母の父（母父）。
KETTO3_PATHS: tuple[str, ...] = (
    "S", "D",
    "SS", "SD", "DS", "DD",
    "SSS", "SSD", "SDS", "SDD", "DSS", "DSD", "DDS", "DDD",
)

#: 枠の日本語名。レポートの見出し用。
PATH_LABELS: dict[str, str] = {
    "S": "父", "D": "母", "SS": "父父", "SD": "父母", "DS": "母父", "DD": "母母",
    "SSS": "父父父", "SSD": "父父母", "SDS": "父母父", "SDD": "父母母",
    "DSS": "母父父", "DSD": "母父母", "DDS": "母母父", "DDD": "母母母",
}

#: 距離区分（SMILE に近い）。境界は「その値以上」。
DISTANCE_BANDS: tuple[tuple[int, str], ...] = (
    (0, "S"),        # Sprint      ～1399m
    (1400, "M"),     # Mile        1400～1799m
    (1800, "I"),     # Intermediate 1800～2199m
    (2200, "L"),     # Long        2200～2799m
    (2800, "E"),     # Extended    2800m～
)

#: 学習・評価に使う列のうち、**特徴量にしてはいけない**列の接頭辞と名前。
#: integrity.check_feature_columns() がこれを使って特徴量の列名を検査する。
FORBIDDEN_FEATURE_PREFIXES: tuple[str, ...] = ("result_", "target_")
FORBIDDEN_FEATURE_NAMES: frozenset[str] = frozenset({
    "kakutei_jyuni", "odds", "ninki", "top3", "win", "time_sec", "margin_sec",
})


def all_paths(generations: int) -> tuple[str, ...]:
    """generations 世代ぶんの祖先の位置を、世代の浅い順・父系優先で返す。

    1世代 = (S, D)、2世代 = (SS, SD, DS, DD) ... 。3世代までの並びは
    KETTO3_PATHS と一致する（テストで固定）。
    """
    if generations < 1:
        raise ValueError("generations は 1 以上にしてください。")
    paths: list[str] = []
    current = [""]
    for _ in range(generations):
        current = [p + c for p in current for c in ("S", "D")]
        paths.extend(current)
    return tuple(paths)


def path_label(path: str) -> str:
    if path in PATH_LABELS:
        return PATH_LABELS[path]
    return "".join("父" if c == "S" else "母" for c in path)


@dataclass(frozen=True)
class DataConfig:
    #: 既存データ版のうち読むもの。空なら RACE_DATA_DIR/CURRENT の1版だけ。
    #: 期間を分けて取得した版を束ねたいときに並べる（同じ行は data_kubun の
    #: 優先順位と make_date で1行に絞る）。
    race_versions: tuple[str, ...] = ()
    #: 血統をどこから読むか。
    #:   "blood" … 既存の血統の版（data/blood/versions/。update_data.py --kind blood、
    #:              dataspec BLDN）。実データで照合済みなので既定
    #:   "ml"    … この層の版（data/ml/pedigree/versions/。scripts/ml/update_pedigree.py）。
    #:              UM の3代血統で補えるのはこちらだけ
    pedigree_source: str = "blood"
    #: 血統の版。空なら "blood" は data/blood/CURRENT の1版、"ml" は
    #: data/ml/pedigree/versions/ の完成済みの版を全部、古い順に重ねる
    #: （セットアップ取得 → 差分 の順に積む想定。新しい行が勝つ）。
    pedigree_versions: tuple[str, ...] = ()
    #: 中央（01〜10）だけを使う。地方・海外は条件が揃わないため既定で除く。
    jra_only: bool = True
    #: 障害を含めるか。血統モデルは平地の適性を見るので既定で除く。
    include_jump: bool = False


@dataclass(frozen=True)
class CodeConfig:
    """年代で異なるコード体系の変換設定。中身の意味は dataset/codes.py を参照。"""

    #: この年（を含む）以前のレースの「馬齢」は数え年。2001年に満年齢表記へ
    #: 移行した。データで確認した年に合わせて変える（integrity の
    #: age_convention 検査が実データとの一致率を出す）。
    kazoe_age_until_year: int = 2000


@dataclass(frozen=True)
class PedigreeConfig:
    #: 何世代さかのぼるか。3 は SK/UM の3代血統だけで埋まる。4 以上は繁殖馬
    #: マスタ HN の父母リンクを再帰的に辿って埋める（HN が無ければ欠損）。
    #: 既定は configs/ml/pedigree.toml と同じ5世代（inbreeding_generations 以上に
    #: しておかないと、設定ファイルで省いたときに validate() が通らない）。
    generations: int = 5
    #: カテゴリ特徴量として CatBoost に渡す祖先の位置の世代上限。0 なら渡さない（既定）。
    #:
    #: CatBoost の ordered target statistics は「同じカテゴリの、その行より前の行」の
    #: 結果を数えるので本人や近親の過去走が混ざり（exclude_self が効かない）、しかも
    #: 数値は学習期間の結果で固定される。実データ（2014〜2023年で学習、2024年で検証、
    #: 2025年でテスト。2026-10-19）では、カテゴリを入れると母・母母が重要度の 30〜40% を
    #: 占めて 35〜37 反復で過学習し、検証からテストへの落ち方が 0.16〜0.17 頭と大きかった
    #: （入れない場合は 0.04 頭。テストの的中数は 0.889 → 1.001）。祖先の情報は本人を除いた
    #: as-of の産駒成績（st_<位置>_*）で入れる。docs/ml/README.md §4.2。
    categorical_generations: int = 0
    #: categorical_generations > 0 のとき、カテゴリから外す祖先の位置（例: ("D", "DD")）。
    categorical_exclude: tuple[str, ...] = ()
    #: 近交係数（Wright の近似）とクロス数を数える世代数。
    inbreeding_generations: int = 5
    #: 産駒成績（as-of の A/E）を集計する祖先の位置。
    #: D（母）の産駒成績 = 兄弟姉妹の成績。
    stat_paths: tuple[str, ...] = ("S", "D", "DS", "SS")
    #: A/E の縮小推定の強さ（期待値で何頭ぶんの事前を足すか）。
    #: 階層ベイズを採るかは未決定（開発方針 §9）。これは最も軽い経験ベイズ。
    shrinkage: float = 10.0
    #: 産駒成績から「その馬自身の過去走」を除く。除かないと、父や母の産駒成績に
    #: 本人の戦績が混ざり（母は産駒が少ないので特に）、血統モデルが実質的に
    #: 「本人の過去成績モデル」になる。過去成績は別の独立モデルの担当。
    exclude_self: bool = True
    #: 産駒成績の履歴を「学習期間の開始日以降」に限るか（"window"）、
    #: 手元の全期間を使うか（"all"）。学習期間比較で「データ量の効果」を
    #: 正しく切り分けたいなら "window"。
    stats_history: str = "window"


@dataclass(frozen=True)
class FeatureConfig:
    #: 条件付けに使う文脈（レースの条件と馬の性齢）。血統の効果は距離・馬場・
    #: 年齢に依存して現れるため入れる。馬自身の過去成績・騎手・オッズは入れない
    #: （それぞれ別の独立モデルの担当。オッズは設計原則 5 で分離する）。
    context: tuple[str, ...] = (
        "track_kind", "kyori_m", "distance_band", "baba_cd", "jyo_cd",
        "age", "sex_cd", "field_size", "month",
    )


@dataclass(frozen=True)
class SplitConfig:
    #: テスト期間（両端含む）。学習期間を変えてもここは固定する。
    test_start: date = date(2025, 1, 1)
    test_end: date = date(2025, 12, 31)
    #: テスト直前の検証期間（early stopping 用）の長さ（年）。
    valid_years: int = 1
    #: 比較する学習期間（年）。検証期間の直前から遡る。
    train_years: tuple[int, ...] = (10, 20, 30, 40)


@dataclass(frozen=True)
class ModelConfig:
    #: 目的関数は未決定（開発方針 §9）。比較できるように設定に出す。
    #:   "Logloss"            … 3着以内か（2値）。確率として較正を測れる
    #:   "YetiRank" など       … レース単位の順位学習（group_id = race_id）
    loss_function: str = "Logloss"
    iterations: int = 3000
    learning_rate: float = 0.05
    depth: int = 6
    l2_leaf_reg: float = 3.0
    early_stopping_rounds: int = 200
    random_seed: int = 42
    thread_count: int = -1
    #: CatBoost に渡すその他の引数（設定ファイルからそのまま渡す）。
    #: 上の項目や early stopping・has_time と同じ意味の引数は書けない
    #: （RESERVED_MODEL_PARAMS。validate() が拒否する）。
    extra: dict[str, Any] = field(default_factory=dict)


#: model.extra に書いてはいけない CatBoost の引数（別名を含む）。モデルの種類・
#: 目的変数・group_id・予測の経路は loss_function から決めるので、extra で
#: 目的関数を上書きされると学習した目的と周りの処理が食い違う。has_time を
#: 外されると祖先カテゴリの数値化に後のレースの結果が入る。
RESERVED_MODEL_PARAMS: frozenset[str] = frozenset({
    "loss_function", "objective", "iterations", "num_boost_round", "n_estimators",
    "num_trees", "learning_rate", "eta", "depth", "max_depth", "l2_leaf_reg",
    "reg_lambda", "random_seed", "random_state", "thread_count", "od_type", "od_wait",
    "early_stopping_rounds", "use_best_model", "has_time",
})


@dataclass(frozen=True)
class EvalConfig:
    top_n: int = 3
    calibration_bins: int = 10
    bootstrap: int = 1000
    bootstrap_seed: int = 0


@dataclass(frozen=True)
class MLConfig:
    data: DataConfig = field(default_factory=DataConfig)
    codes: CodeConfig = field(default_factory=CodeConfig)
    pedigree: PedigreeConfig = field(default_factory=PedigreeConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluate: EvalConfig = field(default_factory=EvalConfig)

    def with_overrides(self, **sections: Any) -> "MLConfig":
        return replace(self, **sections)


def _is_type(expected: type, value: Any) -> bool:
    # bool は int の派生なので、true を整数の設定に・1 を真偽値の設定に通さない。
    if expected is bool or isinstance(value, bool):
        return expected is bool and isinstance(value, bool)
    return isinstance(value, expected)


def _same_type(expected: Any, value: Any) -> bool:
    return _is_type(type(expected), value)


def _coerce(cls: type, values: dict[str, Any]) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"[{cls.__name__}] に知らない設定があります: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    defaults = cls()
    hints = get_type_hints(cls)
    for name, value in values.items():
        current = getattr(defaults, name)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        elif isinstance(current, date) and isinstance(value, str):
            value = date.fromisoformat(value)
        elif isinstance(current, float) and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        # 型の違う値は黙って通さない（"false" という文字列は真として働いてしまう）。
        if not _same_type(current, value):
            raise ValueError(
                f"[{cls.__name__}] {name} の型が違います: {value!r}"
                f"（{type(current).__name__} で書いてください）")
        if isinstance(current, tuple):
            # 要素の型は注釈（tuple[str, ...] など）から取る。既定が空の版のリストも見る。
            element = get_args(hints[name])[0]
            bad = [v for v in value if not _is_type(element, v)]
            if bad:
                raise ValueError(
                    f"[{cls.__name__}] {name} の要素の型が違います: {bad}"
                    f"（{element.__name__} で書いてください）")
        kwargs[name] = value
    return cls(**kwargs)


_SECTIONS = {
    "data": DataConfig, "codes": CodeConfig, "pedigree": PedigreeConfig,
    "features": FeatureConfig, "split": SplitConfig, "model": ModelConfig,
    "evaluate": EvalConfig,
}


def load_config(path: Path | None = None) -> MLConfig:
    """TOML を読む。知らないキーは誤記なので拒否する。

    path を省いたときだけ、既定のファイル（DEFAULT_CONFIG_PATH）が無ければ既定値で
    進める。明示したファイルが無いのは打ち間違いなので FileNotFoundError にする
    （黙って既定値で学習し、config.json にその既定値が残るのを防ぐ）。
    """
    if path is None:
        if not DEFAULT_CONFIG_PATH.exists():
            return MLConfig()
        path = DEFAULT_CONFIG_PATH
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルがありません: {path}")
    return config_from_dict(tomllib.loads(path.read_text(encoding="utf-8")), source=str(path))


def config_from_dict(raw: dict[str, Any], *, source: str = "設定") -> MLConfig:
    """TOML か、実行時に保存した config.json（asdict の結果）から組み立てる。"""
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"{source}: 知らないセクションがあります: {sorted(unknown)}")
    sections = {name: _coerce(cls, raw.get(name, {})) for name, cls in _SECTIONS.items()}
    config = MLConfig(**sections)
    validate(config)
    return config


def validate(config: MLConfig) -> None:
    if config.data.pedigree_source not in ("blood", "ml"):
        raise ValueError('data.pedigree_source は "blood" か "ml" です。')
    p = config.pedigree
    if p.generations < 1:
        raise ValueError("pedigree.generations は 1 以上にしてください。")
    if not 0 <= p.categorical_generations <= p.generations:
        raise ValueError("pedigree.categorical_generations は 0〜generations にしてください。")
    categorical = all_paths(p.categorical_generations) if p.categorical_generations else ()
    bad = [x for x in p.categorical_exclude if x not in categorical]
    if bad:
        raise ValueError(
            f"pedigree.categorical_exclude に categorical_generations を超える位置があります: {bad}")
    if not 1 <= p.inbreeding_generations <= p.generations:
        raise ValueError("pedigree.inbreeding_generations は 1〜generations にしてください。")
    if not p.stat_paths:
        # experiment.run_window が stat_paths を検算に使う。空だと学習の途中で落ちる。
        raise ValueError("pedigree.stat_paths は1つ以上指定してください。")
    valid_paths = set(all_paths(p.generations))
    bad = [x for x in p.stat_paths if x not in valid_paths]
    if bad:
        raise ValueError(f"pedigree.stat_paths に generations を超える位置があります: {bad}")
    if p.shrinkage <= 0:
        raise ValueError("pedigree.shrinkage は正の値にしてください（0 だと出走0の祖先で割り算が壊れる）。")
    if p.stats_history not in ("window", "all"):
        raise ValueError('pedigree.stats_history は "window" か "all" です。')
    s = config.split
    if s.test_end < s.test_start:
        raise ValueError("split.test_end が test_start より前です。")
    if s.valid_years < 1:
        raise ValueError("split.valid_years は 1 以上にしてください。")
    if not s.train_years or any(y < 1 for y in s.train_years):
        raise ValueError("split.train_years は 1 以上の年数を1つ以上並べてください。")
    reserved = sorted(set(config.model.extra) & RESERVED_MODEL_PARAMS)
    if reserved:
        raise ValueError(f"model.extra には書けない引数があります: {reserved}"
                         "（[model] の項目として書いてください）")
    e = config.evaluate
    if e.top_n < 1:
        raise ValueError("evaluate.top_n は 1 以上にしてください。")
    if e.calibration_bins < 1:
        raise ValueError("evaluate.calibration_bins は 1 以上にしてください。")
    if e.bootstrap < 0:
        raise ValueError("evaluate.bootstrap は 0 以上にしてください（0 なら区間を出さない）。")
