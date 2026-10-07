"""Synthetic GES-like files for the offline demo and the tests.

The generator writes VEHICLE, ACCIDENT and PERSON tables with the same column names and code
conventions as GES 2015: sentinel codes, several vehicles in one crash, a shared crash effect,
case weights, variance strata and PSUs. The values are random. They are not NHTSA data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BODY_CODES = {  # code: (probability, gvwr class, effect on severity)
    4: (0.38, 1, 0.0), 2: (0.05, 1, 0.0), 14: (0.15, 1, -0.15), 15: (0.05, 1, -0.15),
    20: (0.04, 1, -0.1), 31: (0.10, 1, -0.05), 34: (0.04, 1, -0.05), 49: (0.03, 1, 0.0),
    50: (0.01, 2, -0.6), 64: (0.02, 2, -0.5), 66: (0.03, 3, -0.8), 80: (0.03, 0, 1.4),
    98: (0.07, 9, 0.0),
}
SPEED_LIMITS = np.array([25, 30, 35, 40, 45, 55, 65, 70])
THRESHOLDS = np.array([1.6, 2.45, 3.55, 5.3])  # latent cut points for O|C, C|B, B|A, A|K


def _choice(rng, values, probs, size):
    probs = np.asarray(probs, dtype=float)
    return rng.choice(np.asarray(values), size=size, p=probs / probs.sum())


def generate(n_crashes: int = 3000, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_strata, psu_per_stratum = 8, 4
    psustrat = rng.integers(1, n_strata + 1, n_crashes)
    psu = (psustrat - 1) * 10 + rng.integers(1, psu_per_stratum + 1, n_crashes)
    region = ((psustrat - 1) % 4) + 1
    stratum = rng.integers(1, 7, n_crashes)
    weight = np.round(rng.lognormal(mean=np.log(40 + 25 * stratum), sigma=0.5), 4)
    n_veh = _choice(rng, [1, 2, 3], [0.32, 0.55, 0.13], n_crashes)
    limit = _choice(rng, SPEED_LIMITS, [0.1, 0.1, 0.22, 0.12, 0.18, 0.15, 0.08, 0.05], n_crashes)
    hour = rng.integers(0, 24, n_crashes)
    hour[rng.random(n_crashes) < 0.01] = 99
    lgt = np.where((hour >= 7) & (hour <= 18), 1, _choice(rng, [2, 3, 4, 5], [0.45, 0.45, 0.05, 0.05], n_crashes))
    lgt[rng.random(n_crashes) < 0.01] = 9
    weather = _choice(rng, [1, 2, 4, 10, 5, 98, 99], [0.70, 0.12, 0.03, 0.10, 0.01, 0.02, 0.02], n_crashes)
    crash_effect = rng.normal(0, 0.6, n_crashes)
    accident = pd.DataFrame({
        "CASENUM": 201500000000 + np.arange(n_crashes), "PSU": psu, "PSUSTRAT": psustrat, "STRATUM": stratum,
        "REGION": region, "WEIGHT": weight, "LGT_COND": lgt, "WEATHER": weather, "VE_TOTAL": n_veh,
    })

    rows = []
    body_codes = list(BODY_CODES)
    body_p = [BODY_CODES[c][0] for c in body_codes]
    for i in range(n_crashes):
        k = int(n_veh[i])
        if k == 1:
            manner = 0
        else:
            manner = int(_choice(rng, [1, 2, 6, 7, 8, 9, 99], [0.38, 0.05, 0.35, 0.12, 0.03, 0.02, 0.05], 1)[0])
        surface = int(_choice(rng, [1, 2, 3, 4, 98], [0.78, 0.15, 0.02, 0.02, 0.03], 1)[0])
        dark = lgt[i] in (2, 6)
        for v in range(1, k + 1):
            body = int(_choice(rng, body_codes, body_p, 1)[0])
            year = int(rng.integers(1995, 2017))
            age = 2015 - year
            older = age > 10
            alcohol_p = 0.03 + (0.04 if older else 0.0) + (0.05 if dark else 0.0)
            alcohol = 1 if rng.random() < alcohol_p else 2
            if rng.random() < 0.07:
                alcohol = 9
            speeding = 0 if rng.random() > 0.06 + 0.04 * older else int(rng.choice([3, 4, 5]))
            rollover_p = 0.01 + (0.06 if k == 1 else 0.0) + (0.02 if body in (14, 15, 31, 34) else 0.0)
            rollover = int(rng.choice([1, 2])) if rng.random() < rollover_p else 0
            hit_run = 1 if (v == k and k > 1 and rng.random() < 0.08) else 0
            occupants = int(_choice(rng, [1, 2, 3, 4, 0], [0.73, 0.17, 0.05, 0.03, 0.02], 1)[0])
            true_speed = max(0.0, rng.normal(limit[i] * (0.85 if manner else 1.0), 9))
            travel = int(round(true_speed)) if rng.random() > 0.5 else int(rng.choice([998, 999], p=[0.8, 0.2]))
            lp = (
                0.045 * (limit[i] - 35) + 1.7 * (rollover > 0) + 0.9 * (alcohol == 1) + 0.6 * (speeding > 0)
                + BODY_CODES[body][2] + 0.35 * older + 0.45 * dark + 0.8 * (manner == 2) + 0.3 * (k == 1)
                + 0.15 * (surface == 1) + crash_effect[i]
            )
            latent = lp + rng.logistic()
            sev = int(np.searchsorted(THRESHOLDS, latent))
            if occupants == 0:
                sev = 8
            elif rng.random() < 0.04:
                sev = 9
            elif rng.random() < 0.006:
                sev = 5
            if rng.random() < 0.03:
                year = int(rng.choice([9998, 9999]))
            rows.append({
                "CASENUM": accident.CASENUM[i], "VEH_NO": v, "REGION": region[i], "STRATUM": stratum[i],
                "PSU": psu[i], "PSUSTRAT": psustrat[i], "NUMOCCS": occupants, "HOUR": hour[i],
                "MAN_COLL": manner, "HIT_RUN": hit_run, "BODY_TYP": body, "MOD_YEAR": year,
                "GVWR": BODY_CODES[body][1], "TRAV_SP": travel, "ROLLOVER": rollover,
                "VTRAFWAY": int(rng.choice([1, 2, 3, 4, 8])), "VALIGN": int(rng.choice([1, 1, 1, 2, 3])),
                "VPROFILE": int(rng.choice([1, 1, 2, 5, 6, 8])), "VSPD_LIM": limit[i], "VSURCOND": surface,
                "VEH_ALCH": alcohol, "SPEEDREL": speeding, "FIRE_EXP": int(rng.random() < 0.002 + 0.01 * (sev == 4)),
                "MAX_VSEV": sev, "NUM_INJV": 0 if sev in (0, 8, 9) else 1, "DEFORMED": int(rng.choice([2, 4, 6])),
                "WEIGHT": weight[i],
            })
    vehicle = pd.DataFrame(rows)

    people = []
    for row in vehicle.itertuples(index=False):
        driver_age = int(np.clip(rng.normal(42, 17), 16, 92))
        if row.HIT_RUN == 1 or rng.random() < 0.02:
            driver_age = 999
        people.append({"CASENUM": row.CASENUM, "VEH_NO": row.VEH_NO, "PER_NO": 1, "PER_TYP": 1,
                       "AGE": driver_age, "SEX": int(rng.choice([1, 2, 9], p=[0.54, 0.44, 0.02]))})
        for p in range(2, max(2, row.NUMOCCS + 1)):
            people.append({"CASENUM": row.CASENUM, "VEH_NO": row.VEH_NO, "PER_NO": p, "PER_TYP": 2,
                           "AGE": int(rng.integers(1, 80)), "SEX": int(rng.choice([1, 2]))})
    person = pd.DataFrame(people)
    return {"vehicle": vehicle, "accident": accident, "person": person}


def write(out_dir: str | Path, n_crashes: int = 3000, seed: int = 0) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = generate(n_crashes=n_crashes, seed=seed)
    paths = {}
    for name, frame in tables.items():
        path = out_dir / f"{name.upper()}.CSV"
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths
