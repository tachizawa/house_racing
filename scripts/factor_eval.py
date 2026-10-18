"""要素別の予想の過去での評価（仕様 §5）。Jev は呼ばない（課金なし）。

    .venv/Scripts/python.exe scripts/factor_eval.py --data-dir data/_long

要素の点数の計算は重い（年ごとの物差しで30分以上）ので、点数の表を
<cache-dir>/factors_<データ版>_<血統版>_<要素の版の印>.pkl に保存して再利用する（--rebuild で作り直し）。
要素の版の印は registry.versions_tag()。版を上げるとファイル名が変わるので、古い版の点数を取り違えない。

キャッシュの名前に評価の年は入らないので、読んだ表に評価の年（の前年から）の年ごとの要素が欠けていれば止める。
保存は一時ファイルに書いてから置き換える（途中で止まっても壊れた表がキャッシュとして残らない）。

--recompute-table-from PATH: 古い版の点数の表 PATH を読み、表の要素（縮小の要素7つ）だけを今の版で作り直して
新しい名前で保存する（同じ名前のキャッシュがあっても置き換える）。表の要素は走の表と e3・ae だけから決まるので、
年ごとの要素を作り直す必要がない。PATH は factors_<データ版>_<血統版>_<印>.pkl の名前で、データ版・血統版が今と同じ、
かつ印が「表の要素以外は今の版」の組のどれかと一致するものだけを受け付ける（引き継ぐ列の版を名前で確かめる。
表の中身からは版を確かめられないので、名前を付け替えた表は見抜けない）。

--compare-to DIR: 前の実行の summary.csv と判定を比べる節をレポートに足す。
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from keiba import config  # noqa: E402
from keiba.evaluate.factor_eval import EVAL_YEARS, N_BOOT, SEED, by_year, evaluate, summarize  # noqa: E402
from keiba.evaluate.factor_eval_report import render  # noqa: E402
from keiba.factors import table_factors as tf  # noqa: E402
from keiba.factors.data import load_runs  # noqa: E402
from keiba.factors.registry import (FACTOR_NAMES, VERSIONS, YEARLY_FACTOR_NAMES, compute_all,  # noqa: E402
                                    recompute_table_factors, versions_tag)
from keiba.features.speed_figure import prepare_runs  # noqa: E402
from keiba.ingest.store import current_version, read_manifest, version_dir  # noqa: E402

JST = timezone(timedelta(hours=9))
SUBSETS = {"op": ("OP・重賞", "op"), "mixed": ("3歳以上の古馬混合戦", "mixed")}
LAMBDA_REPORT_YEARS = (2016, 2017, 2018, 2019, 2025)   # レポートに出す年（全年は lambda.csv）


# factors_<データ版>_<血統版>_<印>.pkl。血統版・要素の版を名前に入れる前の古い名前（factors_<データ版>.pkl）は
# 血統版を確かめられないので受け付けない
_CACHE_NAME = re.compile(r"factors_(?P<data>[^_]+)_(?P<blood>[^_]+)_(?P<tag>[0-9a-f]{8})\.pkl")


def _current(d: Path) -> str:
    version = current_version(d)
    if version is None:
        raise SystemExit(f"CURRENT がありません: {d}")
    return version


def _tags_with_current_non_table_versions() -> set[str]:
    """表の要素以外は今の版（VERSIONS）、表の要素は既知の版（table_factors.ESTIMATORS）のどれか、の組の印すべて。

    印は版の組のハッシュで、表から版を読み戻せない。表の要素の版は作り直すので何でもよいが、ほかの要素の列は
    そのまま引き継ぐので今の版と同じでなければならない。表の要素の版の組み合わせは 2^7 = 128 通りなので
    総当たりで照らし合わせる（ほかの要素の版が違えば、どの組み合わせの印とも一致しない）。
    """
    table = list(tf.TABLE_FACTORS)
    tags = set()
    for combo in itertools.product(sorted(tf.ESTIMATORS), repeat=len(table)):
        tags.add(versions_tag({**VERSIONS, **dict(zip(table, combo))}))
    return tags


def recompute_source_problem(path: Path, data_version: str, blood_version: str) -> str | None:
    """--recompute-table-from の表が今のデータ版・血統版、今の版の（表以外の）要素から作ったものか。違えば理由を返す。

    作り直した表は今の版の名前で保存するので、別の版の表を渡すと中身と名前が食い違ったまま
    以後の実行で再利用される。版はファイル名からしか分からない（表の中には入っていない）。
    """
    m = _CACHE_NAME.fullmatch(path.name)
    if m is None:
        return (f"{path.name} から版を読み取れません（血統版の無い古い名前 factors_<データ版>.pkl も、血統版を"
                "確かめられないので使いません）。factors_<データ版>_<血統版>_<印>.pkl の名前の表を渡すか、"
                "--recompute-table-from を外し、--rebuild を付けて作り直してください。")
    if m["data"] != data_version:
        return f"{path.name} はデータ版 {m['data']} の表です。今のデータ版は {data_version} です。"
    if m["blood"] != blood_version:
        return f"{path.name} は血統版 {m['blood']} の表です。今の血統版は {blood_version} です。"
    if m["tag"] not in _tags_with_current_non_table_versions():
        return (f"{path.name} の要素の版の印 {m['tag']} は、表の要素以外を今の版にした組のどれとも合いません"
                "（表の要素以外の版が違う表は、その列を引き継げません）。"
                "--recompute-table-from を外し、--rebuild を付けて作り直してください。")
    return None


def missing_years(R: pd.DataFrame, years: list[int]) -> list[int]:
    """走があるのに年ごとの要素（物差しから作る要素）が1つも入っていない年。

    年ごとの要素は compute_all に渡した年だけ計算し、ほかの年は NaN のまま。キャッシュの名前には評価の年が
    入っていないので、評価の年を変えたときに古い年の範囲の表を黙って使わないよう、中身で確かめる。
    走が1つも無い年（データ版がその年を含まない）は、作り直しても入らないので数えない。
    """
    cols = [c for c in YEARLY_FACTOR_NAMES if c in R.columns]
    has = (R[cols].notna().any(axis=1) if cols else pd.Series(False, index=R.index)).groupby(R["year"]).any()
    return [y for y in years if y in has.index and not bool(has[y])]


def save_atomically(R: pd.DataFrame, path: Path) -> None:
    """一時ファイルに書いてから置き換える。途中で止まっても、壊れた表がキャッシュの名前で残らない。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        R.to_pickle(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def cache_path(cache_dir: Path, data_version: str, blood_version: str, versions: dict[str, str] | None = None) -> Path:
    """点数の表のキャッシュの場所。データ版・血統版・要素の版のどれかが違えば別のファイルになる。"""
    return cache_dir / f"factors_{data_version}_{blood_version}_{versions_tag(versions)}.pkl"


def git_head(repo: Path = ROOT) -> str | None:
    """実行したコードの版（git の HEAD）。git が無い・リポジトリでないときは None（記録に残せないだけで止めない）。"""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return (out.stdout.strip() or None) if out.returncode == 0 else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR / "_long")
    ap.add_argument("--blood-dir", type=Path, default=config.DATA_DIR / "blood")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=config.DATA_DIR / "factor_eval",
                    help="点数の表のキャッシュを置く場所（--out-dir とは別。既定 data/factor_eval/）")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--recompute-table-from", type=Path, default=None)
    ap.add_argument("--compare-to", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)

    version = _current(args.data_dir)
    vdir = version_dir(args.data_dir, version)
    blood = _current(args.blood_dir)
    bdir = version_dir(args.blood_dir, blood)
    stamp = datetime.now(JST).strftime("%Y%m%dT%H%M%S")
    out_dir = args.out_dir or (config.DATA_DIR / "factor_eval" / stamp)
    cache = cache_path(args.cache_dir, version, blood)
    years = [EVAL_YEARS[0] - 1] + EVAL_YEARS
    log = lambda m: print(m, flush=True)  # noqa: E731

    reused, source = False, None
    if args.recompute_table_from is not None:
        # 明示した作り直しは、同じ名前のキャッシュがあってもそちらを優先する（キャッシュは作り直した表で置き換える）
        problem = recompute_source_problem(args.recompute_table_from, version, blood)
        if problem:
            print(problem)
            return 2
        source = str(args.recompute_table_from)
        print(f"古い点数の表から表の要素だけ作り直す: {source}", flush=True)
        old = pd.read_pickle(args.recompute_table_from)
        if miss := missing_years(old, years):
            print(f"{source} には {miss} 年の年ごとの要素がありません（評価の年 {years[0]}〜{years[-1]}）。"
                  "--recompute-table-from を外し、--rebuild を付けて作り直してください。")
            return 2
        R = recompute_table_factors(old, log=log)
    elif cache.exists() and not args.rebuild:
        R = pd.read_pickle(cache)
        reused = True
        print(f"要素の点数をキャッシュから読む: {cache}")
        if miss := missing_years(R, years):
            print(f"キャッシュには {miss} 年の年ごとの要素がありません（評価の年 {years[0]}〜{years[-1]}。"
                  "別の評価の年で作った表です）。--rebuild で作り直してください。")
            return 2
    else:
        print("走の表を作る …", flush=True)
        read_manifest(vdir)          # 書きかけ・壊れた版はここで止める（DataVersionError）
        read_manifest(bdir)
        R0 = load_runs(vdir, bdir)
        races = pd.read_csv(vdir / "races.csv", dtype=str, na_filter=False)
        horses = pd.read_csv(vdir / "race_horses.csv", dtype=str, na_filter=False)
        runs_all = prepare_runs(races, horses, as_of=datetime(EVAL_YEARS[-1] + 1, 1, 1, tzinfo=JST))
        R = compute_all(R0, runs_all, years=years, log=log)
    if not reused:
        save_atomically(R, cache)
        print(f"点数の表を保存: {cache}")

    res = evaluate(R, FACTOR_NAMES, years=EVAL_YEARS, log=log)
    pr = res["per_race"]
    summary = summarize(pr, FACTOR_NAMES, n_boot=args.n_boot, seed=SEED)
    yearly = by_year(pr, FACTOR_NAMES)
    subsets, subset_years = {}, {}
    for key, (label, col) in SUBSETS.items():
        part = pr[pr[col]]
        if part.empty:          # 作り物などで該当レースが無いときは出さない
            continue
        subsets[key] = (label, summarize(part, FACTOR_NAMES, n_boot=args.n_boot, seed=SEED))
        subset_years[key] = by_year(part, FACTOR_NAMES)
    lambdas = tf.lambda_table(R, [y for y in years if y >= tf.FIRST_YEAR])
    previous = pd.read_csv(args.compare_to / "summary.csv", index_col="factor") if args.compare_to else None

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(render(
        summary, yearly, {label: s for label, s in subsets.values()}, res["excluded"], data_version=version,
        years=EVAL_YEARS, lambdas=lambdas[lambdas["year"].isin(LAMBDA_REPORT_YEARS)], versions=VERSIONS,
        previous=previous, previous_label=str(args.compare_to) if args.compare_to else "", n_boot=args.n_boot),
        encoding="utf-8")
    summary.to_csv(out_dir / "summary.csv")
    yearly.to_csv(out_dir / "by_year.csv")
    for key, (_, s) in subsets.items():
        s.to_csv(out_dir / f"summary_{key}.csv")
        subset_years[key].to_csv(out_dir / f"by_year_{key}.csv")
    lambdas.to_csv(out_dir / "lambda.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps({
        "data_version": version, "blood_version": blood, "eval_years": EVAL_YEARS, "n_boot": args.n_boot,
        "seed": SEED, "factors": VERSIONS, "versions_tag": versions_tag(), "excluded": res["excluded"],
        "git_head": git_head(), "cache": str(cache), "cache_reused": reused, "cache_source": source,
        "compared_to": str(args.compare_to) if args.compare_to else None,
        "created_at": datetime.now(JST).isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"書き出し: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
