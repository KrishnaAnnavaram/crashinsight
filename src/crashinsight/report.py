"""Markdown report: data quality, weighted descriptives, the age analysis and the model results."""
from __future__ import annotations

import pandas as pd

from .analysis import AgeEffect
from .codebook import Codebook
from .dataset import BuildResult
from .survey import weighted_rate_table
from .train import SUMMARY_KEYS, TrainResult

DESCRIBE_BY = ("body_group", "collision_manner", "region", "light", "time_of_day")
AGE_BANDS = [-1, 3, 6, 10, 15, 200]
AGE_LABELS = ["00-03", "04-06", "07-10", "11-15", "16+"]


def _table(frame: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in frame.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(floatfmt.format(v) if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def describe_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    data = frame.copy()
    data["vehicle_age_band"] = pd.cut(data["vehicle_age"], AGE_BANDS, labels=AGE_LABELS).astype(object)
    tables = {}
    for by in (*DESCRIBE_BY, "vehicle_age_band"):
        if by in data.columns:
            tables[by] = weighted_rate_table(data, by)
    return tables


def data_quality(raw_vehicle: pd.DataFrame, codebook: Codebook) -> pd.DataFrame:
    share = codebook.sentinel_share(raw_vehicle)
    undecoded = codebook.undecoded_codes(raw_vehicle)
    rows = [{"variable": k, "sentinel_share": v, "codes_without_label": ", ".join(map(str, undecoded.get(k, []))) or "-"}
            for k, v in sorted(share.items())]
    return pd.DataFrame(rows)


def render(
    build: BuildResult,
    raw_vehicle: pd.DataFrame,
    codebook: Codebook,
    data_label: str,
    age: AgeEffect | None = None,
    results: list[TrainResult] | None = None,
    importance: pd.DataFrame | None = None,
) -> str:
    frame = build.frame
    out = [
        "# crashinsight report",
        "",
        f"Data: {data_label}. Codebook: {codebook.version}.",
        "",
        "## Data",
        "",
        f"- Vehicle rows in the file: {len(raw_vehicle)}.",
        f"- Rows with a KABCO target: {len(frame)}. Removed: {build.dropped_unknown_target} unknown severity, "
        f"{build.dropped_not_applicable_target} not applicable (no occupants or died before the crash).",
        "- No outlier filter is applied. Every K and A row stays in the data.",
        "",
        "KABCO counts: " + ", ".join(f"{k} = {v}" for k, v in frame["severity"].value_counts().items()) + ".",
        "",
        "### Sentinel codes and codebook coverage",
        "",
        _table(data_quality(raw_vehicle, codebook)),
        "",
        "## Survey-weighted KA rate",
        "",
        "Rate = weighted share of vehicles with a serious or fatal injury. SE uses Taylor linearization "
        "with PSUSTRAT strata and PSU clusters. `small_cell` marks fewer than 30 rows.",
    ]
    for by, table in describe_tables(frame).items():
        keep = table[[by, "n", "weighted_n", "rate", "ci_low", "ci_high", "unweighted_rate", "small_cell"]]
        out += ["", f"### By {by}", "", _table(keep)]
    if age is not None:
        out += [
            "", "## Vehicle age and KA (adjusted)", "",
            f"- Weighted KA rate, newer (<= {age.newer_max_age} years): {age.rate_newer.value:.4f} "
            f"(SE {age.rate_newer.se:.4f}, n = {age.rate_newer.n}).",
            f"- Weighted KA rate, older: {age.rate_older.value:.4f} (SE {age.rate_older.se:.4f}, n = {age.rate_older.n}).",
            f"- Controls: {', '.join(age.controls)}.",
            f"- {age.summary()}",
        ]
    if results:
        out += ["", "## Models (crash-grouped CV and hold-out test)", "",
                "| model | target | " + " | ".join(f"test {k}" for k in SUMMARY_KEYS) + " |",
                "|" + "---|" * (2 + len(SUMMARY_KEYS))]
        for r in results:
            vals = [f"{r.test[k]:.3f}" if k in r.test else "-" for k in SUMMARY_KEYS]
            out.append(f"| {r.model} | {r.target} | " + " | ".join(vals) + " |")
        base = results[0].baseline_test
        out.append("| majority baseline | " + results[0].target + " | "
                   + " | ".join(f"{base[k]:.3f}" if k in base else "-" for k in SUMMARY_KEYS) + " |")
    if importance is not None:
        out += ["", "## Permutation importance (drop in KA PR-AUC on the test rows)", "", _table(importance.head(12))]
    return "\n".join(out) + "\n"
