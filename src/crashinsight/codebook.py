"""The GES codebook: code labels, sentinel codes and the decode rules for each variable.

The labels come from a JSON file that is transcribed from the official NHTSA manual. They are not
written in the code. Sentinel codes (unknown, not reported, not applicable) never become numbers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

UNKNOWN_LABEL = "Unknown"
NOT_APPLICABLE_LABEL = "Not applicable"
KINDS = {"nominal", "ordinal", "binary", "numeric"}


@dataclass(frozen=True)
class Variable:
    name: str
    file: str
    kind: str
    label: str
    codes: dict[int, str] = field(default_factory=dict)
    ranges: tuple[tuple[int, int, str], ...] = ()
    unknown: frozenset[int] = frozenset()
    not_applicable: frozenset[int] = frozenset()
    positive: frozenset[int] = frozenset()
    valid: tuple[float, float] | None = None
    topcode: dict[int, float] = field(default_factory=dict)

    @property
    def sentinels(self) -> frozenset[int]:
        return self.unknown | self.not_applicable

    def label_of(self, code: int) -> str | None:
        """Return the label of one code, or None if the codebook does not know the code."""
        if code in self.codes:
            return self.codes[code]
        for low, high, text in self.ranges:
            if low <= code <= high:
                return text
        return None

    def is_known(self, code: int) -> bool:
        if code in self.sentinels or code in self.topcode:
            return True
        if self.kind == "numeric":
            return self.valid is not None and self.valid[0] <= code <= self.valid[1]
        return self.label_of(code) is not None

    def decode(self, values: pd.Series) -> pd.Series:
        """Decode raw integer codes. Sentinels become NaN (numeric/binary) or their own category."""
        raw = pd.to_numeric(values, errors="coerce")
        if self.kind == "numeric":
            out = raw.astype(float).copy()
            for code, cap in self.topcode.items():
                out[raw == code] = cap
            out[raw.isin(list(self.sentinels))] = np.nan
            if self.valid is not None:
                low, high = self.valid
                capped = raw.isin(list(self.topcode))
                out[~capped & ((out < low) | (out > high))] = np.nan
            return out
        if self.kind == "binary":
            known = set(self.codes) - set(self.sentinels)
            out = pd.Series(np.nan, index=values.index, dtype=float)
            out[raw.isin(list(self.positive))] = 1.0
            out[raw.isin(list(known - set(self.positive)))] = 0.0
            return out
        labels = []
        for code in raw:
            if pd.isna(code):
                labels.append(np.nan)
                continue
            code = int(code)
            if code in self.unknown:
                labels.append(UNKNOWN_LABEL)
            elif code in self.not_applicable:
                labels.append(NOT_APPLICABLE_LABEL)
            else:
                text = self.label_of(code)
                labels.append(text if text is not None else np.nan)
        return pd.Series(labels, index=values.index, dtype=object)


class Codebook:
    def __init__(self, variables: dict[str, Variable], version: str, source: str):
        self.variables = variables
        self.version = version
        self.source = source

    def __getitem__(self, name: str) -> Variable:
        return self.variables[name]

    def __contains__(self, name: str) -> bool:
        return name in self.variables

    def for_file(self, file: str) -> list[Variable]:
        return [v for v in self.variables.values() if v.file == file]

    def undecoded_codes(self, frame: pd.DataFrame) -> dict[str, list[int]]:
        """Return, for each codebook variable in the frame, the codes that have no label."""
        report: dict[str, list[int]] = {}
        for name, var in self.variables.items():
            if name not in frame.columns:
                continue
            codes = pd.to_numeric(frame[name], errors="coerce").dropna().astype(int).unique()
            missing = sorted(int(c) for c in codes if not var.is_known(int(c)))
            if missing:
                report[name] = missing
        return report

    def sentinel_share(self, frame: pd.DataFrame) -> dict[str, float]:
        """Return the share of rows that carry a sentinel code, for each variable in the frame."""
        out = {}
        for name, var in self.variables.items():
            if name in frame.columns and len(frame):
                raw = pd.to_numeric(frame[name], errors="coerce")
                out[name] = float(raw.isin(list(var.sentinels)).mean())
        return out


def _parse(name: str, spec: dict) -> Variable:
    kind = spec["kind"]
    if kind not in KINDS:
        raise ValueError(f"{name}: unknown kind {kind!r}")
    codes = {int(k): str(v) for k, v in spec.get("codes", {}).items()}
    ranges = tuple((int(a), int(b), str(t)) for a, b, t in spec.get("ranges", []))
    if kind != "numeric" and not codes and not ranges:
        raise ValueError(f"{name}: a {kind} variable needs codes or ranges")
    valid = tuple(spec["valid"]) if "valid" in spec else None
    if kind == "numeric" and valid is None:
        raise ValueError(f"{name}: a numeric variable needs a valid range")
    positive = frozenset(int(c) for c in spec.get("positive", []))
    if kind == "binary" and not positive:
        raise ValueError(f"{name}: a binary variable needs positive codes")
    return Variable(
        name=name,
        file=spec.get("file", "VEHICLE"),
        kind=kind,
        label=spec.get("label", name),
        codes=codes,
        ranges=ranges,
        unknown=frozenset(int(c) for c in spec.get("unknown", [])),
        not_applicable=frozenset(int(c) for c in spec.get("not_applicable", [])),
        positive=positive,
        valid=valid,
        topcode={int(k): float(v) for k, v in spec.get("topcode", {}).items()},
    )


def load_codebook(path: str | Path | None = None) -> Codebook:
    """Load the bundled GES 2015 codebook, or a codebook JSON file at `path`."""
    if path is None:
        text = resources.files("crashinsight").joinpath("data/codebook_ges2015.json").read_text("utf-8")
    else:
        text = Path(path).read_text("utf-8")
    doc = json.loads(text)
    variables = {name: _parse(name, spec) for name, spec in doc["variables"].items()}
    return Codebook(variables, version=doc.get("version", "unknown"), source=doc.get("source", ""))
