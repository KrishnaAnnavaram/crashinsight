"""Command-line interface: `crashinsight <command>`."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from . import synthetic
from .analysis import age_effect
from .codebook import load_codebook
from .config import Settings
from .dataset import build_dataset, target_values
from .evaluate import classification_metrics
from .explain import permutation_table
from .models import MODEL_NAMES
from .report import describe_tables, render
from .schema import SchemaError, load_tables
from .split import grouped_holdout
from .train import SUMMARY_KEYS, load, save, train


def _build(data_dir: Path, settings: Settings):
    tables = load_tables(data_dir)
    codebook = load_codebook()
    build = build_dataset(tables["vehicle"], tables.get("accident"), tables.get("person"), codebook, settings.crash_year)
    return tables, codebook, build


def cmd_synth(args, settings: Settings) -> int:
    paths = synthetic.write(args.out, n_crashes=args.crashes, seed=args.seed)
    for name, path in paths.items():
        print(f"wrote {name}: {path}")
    return 0


def cmd_validate(args, settings: Settings) -> int:
    tables = load_tables(args.data)
    codebook = load_codebook()
    problems = 0
    for name, frame in tables.items():
        undecoded = codebook.undecoded_codes(frame)
        print(f"{name}: {len(frame)} rows, schema OK")
        for var, codes in undecoded.items():
            problems += 1
            print(f"  {var}: codes without a label: {codes}")
    share = codebook.sentinel_share(tables["vehicle"])
    for var, value in sorted(share.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  sentinel share {var}: {value:.1%}")
    print("codebook coverage: OK" if problems == 0 else f"codebook coverage: {problems} variable(s) with unknown codes")
    return 0 if problems == 0 else 2


def cmd_describe(args, settings: Settings) -> int:
    _, _, build = _build(args.data, settings)
    for by, table in describe_tables(build.frame).items():
        print(f"\nKA rate by {by} (survey-weighted)")
        print(table[[by, "n", "rate", "ci_low", "ci_high", "unweighted_rate"]].to_string(index=False, float_format="%.4f"))
    return 0


def cmd_age(args, settings: Settings) -> int:
    _, _, build = _build(args.data, settings)
    result = age_effect(build.frame, bootstrap=args.bootstrap, seed=settings.seed)
    print(f"newer KA rate {result.rate_newer.value:.4f} (SE {result.rate_newer.se:.4f})")
    print(f"older KA rate {result.rate_older.value:.4f} (SE {result.rate_older.se:.4f})")
    print(result.summary())
    return 0


def cmd_train(args, settings: Settings) -> int:
    _, codebook, build = _build(args.data, settings)
    result = train(build.frame, build.spec, target=args.target, model=args.model, seed=settings.seed,
                   n_splits=settings.n_splits, n_boot=args.bootstrap)
    out = Path(args.out or settings.output_dir / f"{args.model}_{args.target}")
    paths = save(result, out, codebook.version, str(args.data))
    print(f"model {args.model}, target {args.target}: {result.n_train} train rows, {result.n_test} test rows")
    for key in SUMMARY_KEYS:
        if key in result.test:
            print(f"  {key:18s} cv {result.cv.mean(key):.3f}  test {result.test[key]:.3f}  "
                  f"baseline {result.baseline_test.get(key, float('nan')):.3f}")
    for name, path in paths.items():
        print(f"wrote {name}: {path}")
    return 0


def cmd_evaluate(args, settings: Settings) -> int:
    bundle = load(args.model)
    _, _, build = _build(args.data, settings)
    y = target_values(build.frame, bundle["target"])
    pipe = bundle["pipeline"]
    metrics = classification_metrics(y, pipe.predict_proba(build.frame[bundle["spec"].all]), pipe.classes_)
    print(json.dumps({k: metrics[k] for k in (*SUMMARY_KEYS, "recall_by_class") if k in metrics}, indent=2))
    return 0


def cmd_report(args, settings: Settings) -> int:
    tables, codebook, build = _build(args.data, settings)
    age = age_effect(build.frame, bootstrap=args.bootstrap, seed=settings.seed)
    results = [train(build.frame, build.spec, target="ka", model=m, seed=settings.seed,
                     n_splits=settings.n_splits, n_boot=0) for m in ("logreg", "boosting")]
    best = max(results, key=lambda r: r.test.get("ka_pr_auc", 0.0))
    y = target_values(build.frame, "ka")
    _, te = grouped_holdout(build.frame, y, seed=settings.seed)
    test = build.frame.iloc[te]
    importance = permutation_table(best.pipeline, test[build.spec.all], y.iloc[te], seed=settings.seed)
    text = render(build, tables["vehicle"], codebook, str(args.data), age, results, importance)
    out = Path(args.out or settings.output_dir / "report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote report: {out}")
    return 0


def cmd_demo(args, settings: Settings) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "synthetic"
        synthetic.write(data, n_crashes=args.crashes, seed=args.seed)
        print(f"synthetic data: {args.crashes} crashes (seed {args.seed})")
        ns = argparse.Namespace(data=data, bootstrap=args.bootstrap,
                                out=Path(args.out) if args.out else settings.output_dir / "demo_report.md")
        cmd_validate(ns, settings)
        cmd_age(ns, settings)
        for model in ("majority", "logreg", "boosting"):
            _, _, build = _build(data, settings)
            r = train(build.frame, build.spec, target="ka", model=model, seed=settings.seed,
                      n_splits=settings.n_splits, n_boot=args.bootstrap_metrics)
            ci = r.test_ci.get("ka_pr_auc", (float("nan"), float("nan")))
            print(f"{model:9s} test macro_f1 {r.test['macro_f1']:.3f}  KA PR-AUC {r.test['ka_pr_auc']:.3f} "
                  f"(95% CI {ci[0]:.3f}-{ci[1]:.3f})  KA recall {r.test['recall_by_class'].get('1', 0):.3f}")
        cmd_report(ns, settings)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="crashinsight", description="Survey-aware injury-severity analysis on NHTSA GES data")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="write synthetic VEHICLE, ACCIDENT and PERSON files")
    s.add_argument("--out", type=Path, default=Path("data/synthetic"))
    s.add_argument("--crashes", type=int, default=3000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_synth)

    def with_data(name, func, help_text):
        q = sub.add_parser(name, help=help_text)
        q.add_argument("--data", type=Path, required=True, help="folder with VEHICLE.CSV (ACCIDENT/PERSON optional)")
        q.set_defaults(func=func)
        return q

    with_data("validate", cmd_validate, "check the schema and the codebook coverage")
    with_data("describe", cmd_describe, "survey-weighted KA rates by group")
    q = with_data("age-effect", cmd_age, "adjusted vehicle-age odds ratio with a PSU bootstrap CI")
    q.add_argument("--bootstrap", type=int, default=50)
    q = with_data("train", cmd_train, "grouped CV, hold-out test, save the model and the model card")
    q.add_argument("--target", choices=["ka", "kabco"], default="ka")
    q.add_argument("--model", choices=MODEL_NAMES, default="boosting")
    q.add_argument("--bootstrap", type=int, default=200, help="crash-level bootstrap resamples for metric CIs")
    q.add_argument("--out", type=Path)
    q = with_data("evaluate", cmd_evaluate, "evaluate a saved model on another file (for example another year)")
    q.add_argument("--model", type=Path, required=True)
    q = with_data("report", cmd_report, "write the full Markdown report")
    q.add_argument("--bootstrap", type=int, default=50)
    q.add_argument("--out", type=Path)

    d = sub.add_parser("demo", help="run the full pipeline on synthetic data (no download)")
    d.add_argument("--crashes", type=int, default=3000)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--bootstrap", type=int, default=30)
    d.add_argument("--bootstrap-metrics", type=int, default=200)
    d.add_argument("--out", type=Path)
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    try:
        return args.func(args, settings)
    except (SchemaError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
