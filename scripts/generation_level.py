"""世代レベル（タイム指数による世代の強さ）のレポートを書く。Jev は呼ばない（課金なし）。

    .venv/Scripts/python.exe scripts/generation_level.py --as-of 20261004 \
        --data-dir tmp/keiba-report/oct04_2026/data --example-race 2026100405040211

仕様: docs/superpowers/specs/2026-10-04-generation-level-design.md
出力: data/generation/<as_of>/（report.md / cohorts.csv / age_curve.csv / figures.csv / manifest.json）
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba import config  # noqa: E402
from keiba.data.store import DataStore  # noqa: E402
from keiba.evaluate.age_curve import age_curve, curve_levels  # noqa: E402
from keiba.evaluate.generation import (METRICS, bootstrap_intervals, cohort_table,  # noqa: E402
                                       comparable_years, replicate_stats)
from keiba.evaluate.generation_report import ExampleHorse, ReportInputs, render  # noqa: E402
from keiba.features.speed_figure import apply_figures, fit_model, prepare_runs, repeatability  # noqa: E402
from keiba.ingest.store import read_manifest, version_dir  # noqa: E402

JST = timezone(timedelta(hours=9))
SURFACES = (("全体", None), ("芝", "turf"), ("ダート", "dirt"))


def coverage_start(manifest: dict, runs: pd.DataFrame) -> date:
    """比べてよい年を決める取得開始日。manifest の fromdate、無ければ対象の走の最初の日。

    coverage.first_hasso_at は訂正再配信の古いレース（1989年など）が混じるので使わない。
    """
    fromdate = manifest.get("fromdate")
    if fromdate:
        return datetime.strptime(fromdate[:8], "%Y%m%d").date()
    return runs["date"].min().date()


def _examples(store: DataStore, race_ids: list[str]) -> dict[str, list[ExampleHorse]]:
    out = {}
    for race_id in race_ids:
        rows = store.race_horses[store.race_horses["race_id"] == race_id]
        rows = rows.sort_values("make_date").drop_duplicates("umaban", keep="last").sort_values("umaban")
        year = int(race_id[:4])
        out[race_id] = [ExampleHorse(umaban=r.umaban, bamei=r.bamei, age=int(r.barei),
                                     cohort=year - int(r.barei)) for r in rows.itertuples()]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as-of", required=True, help="時点 YYYYMMDD（その日の 0:00 より前のレースだけ使う）")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=None, help="既定 data/generation/<as_of>")
    parser.add_argument("--ages", default="2,3,4,5")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--example-race", action="append", default=[], help="例として載せるレース（繰り返し可）")
    args = parser.parse_args(argv)

    as_of = datetime.strptime(args.as_of, "%Y%m%d").date()
    as_of_dt = datetime(as_of.year, as_of.month, as_of.day, tzinfo=JST)
    requested = sorted({int(a) for a in args.ages.split(",")})
    ages = list(range(2, max(requested) + 1))          # 世代の歩み（§5.4）は2歳時から並べる
    out_dir = args.out_dir or (config.DATA_DIR / "generation" / args.as_of)

    store = DataStore.open_current(args.data_dir)
    manifest = read_manifest(version_dir(args.data_dir, store.version_id))
    runs = prepare_runs(store.races, store.race_horses, as_of=as_of_dt)
    model = fit_model(runs)
    fig = apply_figures(runs, model)
    years = comparable_years(coverage_start(manifest, runs), as_of.year)
    print(f"対象の走 {len(runs):,} 件、古馬戦 {model.n_races:,} レース、比べる年 {years}", flush=True)

    curves = {name: age_curve(fig, years=years, before=as_of, surface=sf, n_boot=args.bootstrap, seed=args.seed)
              for name, sf in SURFACES}
    levels = curve_levels(curves["全体"])
    point = replicate_stats(fig, as_of=as_of, years=years, ages=ages, curve_levels=levels)
    print(f"ブートストラップ {args.bootstrap} 回……", flush=True)
    ci = bootstrap_intervals(runs, as_of=as_of, years=years, ages=ages, curve_levels=levels,
                             n_boot=args.bootstrap, seed=args.seed)
    table = cohort_table(fig, as_of=as_of, ages=ages, years=years)

    out_dir.mkdir(parents=True, exist_ok=True)
    report = render(ReportInputs(as_of=as_of, version_id=store.version_id, years=years, ages=ages, model=model,
                                 fig=fig, table=table, point=point, ci=ci, curves=curves,
                                 repeat=repeatability(fig), n_boot=args.bootstrap, seed=args.seed,
                                 examples=_examples(store, args.example_race)))
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    cohorts = table.copy()
    for m in METRICS:
        keys = [f"value|{r.year}|{r.age}|{m}" for r in table.itertuples()]
        cohorts[f"{m}_lo"] = [ci.at[k, "lo"] if k in ci.index else float("nan") for k in keys]
        cohorts[f"{m}_hi"] = [ci.at[k, "hi"] if k in ci.index else float("nan") for k in keys]
    cohorts.to_csv(out_dir / "cohorts.csv", index=False)
    pd.concat([c.assign(surface=name) for name, c in curves.items()]).to_csv(out_dir / "age_curve.csv", index=False)
    fig[["race_id", "umaban", "ketto_num", "cohort", "barei", "figure", "jyo_cd", "track_cd", "kyori_m",
         "day", "track_kind", "has_standard", "has_variant"]].to_csv(out_dir / "figures.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps({
        "data_version": store.version_id, "as_of": as_of.isoformat(), "years": years, "ages": ages,
        "bootstrap": args.bootstrap, "seed": args.seed, "sigma2": model.sigma2, "tau2": model.tau2,
        "n_races": model.n_races, "examples": args.example_race,
        "created_at": datetime.now(JST).isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"書き出し: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
