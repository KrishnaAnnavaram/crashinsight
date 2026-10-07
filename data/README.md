# Data

crashinsight does not include NHTSA data. Git ignores every file in this folder except this README.

## Source

| Item | Value |
|---|---|
| Name | NHTSA National Automotive Sampling System (NASS) General Estimates System (GES), 2015 |
| Publisher | National Highway Traffic Safety Administration (NHTSA), U.S. Department of Transportation |
| Download | NHTSA file downloads site: <https://www.nhtsa.gov/file-downloads?p=nhtsa/downloads/GES/> (folder `GES15`) |
| Codebook | *1988-2015 NASS GES Analytical User Manual*, report DOT HS 812 320 |
| Licence / terms | U.S. government work, public domain. Cite NHTSA as the source. |
| Later years | CRSS (Crash Report Sampling System), 2016 and later, on the same site. Column names are similar but not identical. Check the CRSS manual before you use it. |

## Expected files

Put the CSV files of one year in one folder, for example `data/ges2015/`.

| File | Needed? | Columns that crashinsight reads |
|---|---|---|
| `VEHICLE.CSV` | Yes | `CASENUM`, `VEH_NO`, `WEIGHT`, `PSU`, `PSUSTRAT`, `MAX_VSEV`, `REGION`, `BODY_TYP`, `MOD_YEAR`, `TRAV_SP`, `VSPD_LIM`, `MAN_COLL`, `HIT_RUN`, `ROLLOVER`, `VEH_ALCH`, `SPEEDREL`, `VSURCOND`, `NUMOCCS`, `HOUR`. Optional: `VTRAFWAY`, `VALIGN`, `VPROFILE`, `GVWR`, `FIRE_EXP`, `STRATUM` |
| `ACCIDENT.CSV` | No | `CASENUM`, `LGT_COND`, `WEATHER` |
| `PERSON.CSV` | No | `CASENUM`, `VEH_NO`, `PER_TYP`, `AGE`, `SEX` (the driver row has `PER_TYP` = 1) |

## Download procedure

1. Open the NHTSA file downloads site and go to the `GES` folder, then `GES15`.
2. Download the CSV archive of the year and extract it.
3. Copy `VEHICLE.CSV`, `ACCIDENT.CSV` and `PERSON.CSV` into `data/ges2015/`.
4. Run `crashinsight validate --data data/ges2015`.

## Synthetic data (no download)

`crashinsight synth --out data/synthetic` writes three files with the same column names and code
conventions. The values are random. They are not NHTSA data, and results on them say nothing about
real crashes.
