"""要素別の予想の過去での評価（仕様 §5）。Jev は呼ばない（課金なし）。

    .venv/Scripts/python.exe scripts/factor_eval.py --data-dir data/_long

要素の点数の計算は重いので、data/factor_eval/factors_<版>.pkl に保存して再利用する（--rebuild で作り直し）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba import config  # noqa: E402
from keiba.evaluate.factor_eval import EVAL_YEARS, N_BOOT, SEED, by_year, evaluate, summarize  # noqa: E402
from keiba.evaluate.factor_eval_report import render  # noqa: E402
from keiba.factors.data import load_runs  # noqa: E402
from keiba.factors.registry import FACTOR_NAMES, FACTORS, compute_all  # noqa: E402
from keiba.features.speed_figure import prepare_runs  # noqa: E402

JST = timezone(timedelta(hours=9))


def _current(d: Path) -> str:
    return (d / "CURRENT").read_text(encoding="utf-8").strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR / "_long")
    ap.add_argument("--blood-dir", type=Path, default=config.DATA_DIR / "blood")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)

    version = _current(args.data_dir)
    vdir = args.data_dir / "versions" / version
    bdir = args.blood_dir / "versions" / _current(args.blood_dir)
    stamp = datetime.now(JST).strftime("%Y%m%dT%H%M%S")
    out_dir = args.out_dir or (config.DATA_DIR / "factor_eval" / stamp)
    cache = (args.out_dir or config.DATA_DIR / "factor_eval") / f"factors_{version}.pkl"
    years = [EVAL_YEARS[0] - 1] + EVAL_YEARS

    if cache.exists() and not args.rebuild:
        R = pd.read_pickle(cache)
        print(f"要素の点数をキャッシュから読む: {cache}")
    else:
        print("走の表を作る …", flush=True)
        R0 = load_runs(vdir, bdir)
        races = pd.read_csv(vdir / "races.csv", dtype=str, na_filter=False)
        horses = pd.read_csv(vdir / "race_horses.csv", dtype=str, na_filter=False)
        runs_all = prepare_runs(races, horses, as_of=datetime(EVAL_YEARS[-1] + 1, 1, 1, tzinfo=JST))
        R = compute_all(R0, runs_all, years=years, log=lambda m: print(m, flush=True))
        cache.parent.mkdir(parents=True, exist_ok=True)
        R.to_pickle(cache)
    res = evaluate(R, FACTOR_NAMES, years=EVAL_YEARS, log=lambda m: print(m, flush=True))
    pr = res["per_race"]
    summary = summarize(pr, FACTOR_NAMES, n_boot=args.n_boot, seed=SEED)
    yearly = by_year(pr, FACTOR_NAMES)
    subsets = {"OP・重賞": summarize(pr[pr["op"]], FACTOR_NAMES, n_boot=args.n_boot, seed=SEED),
               "3歳以上の古馬混合戦": summarize(pr[pr["mixed"]], FACTOR_NAMES, n_boot=args.n_boot, seed=SEED)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(render(summary, yearly, subsets, res["excluded"], data_version=version,
                                              years=EVAL_YEARS), encoding="utf-8")
    summary.to_csv(out_dir / "summary.csv")
    yearly.to_csv(out_dir / "by_year.csv")
    (out_dir / "manifest.json").write_text(json.dumps({
        "data_version": version, "blood_version": bdir.name, "eval_years": EVAL_YEARS, "n_boot": args.n_boot,
        "seed": SEED, "factors": {f.name: f.version for f in FACTORS}, "excluded": res["excluded"],
        "created_at": datetime.now(JST).isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"書き出し: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
