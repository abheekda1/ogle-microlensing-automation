#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OGLE-IV 2025 Bulge — anomalies DB builder + joiner

What this script does
---------------------
1) Reads your 2025 features CSV that has a column named `event` like 'blg-0001'.
2) Normalizes to canonical OGLE IDs: 'OGLE-2025-BLG-0001' and a short 'BLG-0001'.
3) Reads one or more label CSVs you curate (binary/planetary/parallax/etc.).
4) Collapses duplicates with a priority rule into a single 'known_anomalies' table.
5) Left-joins the anomalies DB to your features and writes an output CSV (and Parquet if available).
6) Prints a coverage summary and a small “why-not-matched” hint report.

Label CSV expectations
----------------------
Each label CSV can be minimal; the script is forgiving. Recommended columns:
- event_id_raw (string, any form like 'OGLE-2019-BLG-12', 'BLG-0012', 'blg12')
- lens_type (e.g., 2L1S_binary, 2L1S_planetary, 1L2S_binary_source, parallax, xallarap, repeating, finite_source/caustic, other/exotic)
- confidence (confirmed | candidate | suspected)
- source_ref (e.g., Paper2019, SeasonReport2021)
Optional: doi_or_arxiv, q, s, tE, u0, ra, dec, notes

Usage examples
--------------
# Single file of labels:
python ogle_anomalies_join.py \
    --features features_ogleiv_2025.csv \
    --labels labels_binary.csv \
    --out-features features_2025_with_labels.csv \
    --out-db known_anomalies.csv

# Multiple label files (glob OK):
python ogle_anomalies_join.py \
    --features features_ogleiv_2025.csv \
    --labels labels_binary.csv labels_planetary.csv labels_parallax_xallarap.csv \
    --season 2025

# Labels directory (use shell glob):
python ogle_anomalies_join.py --features features_ogleiv_2025.csv --labels labels_*.csv
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Optional, List

import pandas as pd


# ----------------------------
# Regex + Normalization
# ----------------------------
FULL_RE  = re.compile(r'ogle[\s\-_]*(\d{4})[\s\-_]*blg[\s\-_]*0*(\d+)', re.I)
SHORT_RE = re.compile(r'blg[\s\-_]*0*(\d+)', re.I)
CANON_RE = re.compile(r'OGLE-(\d{4})-BLG-(\d{4})', re.I)

def to_canonical_event_id(s: str, default_year: int) -> Optional[str]:
    """Normalize variants to 'OGLE-YYYY-BLG-NNNN' (NNNN = zero-padded 4 digits)."""
    if not isinstance(s, str):
        return None
    m = FULL_RE.search(s)
    if m:
        year = int(m.group(1))
        num  = int(m.group(2))
        return f"OGLE-{year}-BLG-{num:04d}"
    m = SHORT_RE.search(s)
    if m:
        num = int(m.group(1))
        return f"OGLE-{default_year}-BLG-{num:04d}"
    m = CANON_RE.search(s)
    if m:
        # already canonical
        return f"OGLE-{m.group(1)}-BLG-{m.group(2)}"
    return None

def to_short_id(canonical: str) -> Optional[str]:
    """Return 'BLG-NNNN' from 'OGLE-YYYY-BLG-NNNN'."""
    if not isinstance(canonical, str):
        return None
    m = CANON_RE.search(canonical)
    return f"BLG-{m.group(2)}" if m else None


# ----------------------------
# Label collapsing priority
# ----------------------------
LABEL_PRIORITY = [
    '2L1S_planetary',
    '2L1S_binary',
    '1L2S_binary_source',
    '3L1S_triple',
    'parallax',
    'xallarap',
    'repeating',
    'finite_source/caustic',
    'other/exotic'
]
CONF_RANK = {'confirmed': 2, 'candidate': 1, 'suspected': 0}

def pick_label(group: pd.DataFrame) -> pd.Series:
    """Collapse multiple rows for one event_id into one representative row."""
    # choose highest-priority lens_type present
    chosen_lt = None
    for lt in LABEL_PRIORITY:
        if (group['lens_type'] == lt).any():
            chosen_lt = lt
            break
    g = group.copy()
    g['_conf_rank'] = g['confidence'].str.lower().map(CONF_RANK).fillna(0)
    if chosen_lt is not None:
        g = g[g['lens_type'] == chosen_lt]
    g = g.sort_values('_conf_rank', ascending=False)
    row = g.iloc[0].copy()
    for c in list(row.index):
        if c.startswith('_'):
            del row[c]
    return row


# ----------------------------
# I/O helpers
# ----------------------------
MIN_LABEL_COLS = ['event_id_raw', 'lens_type', 'confidence', 'source_ref']
KEEP_COLS = ['event_id','lens_type','confidence','source_ref','doi_or_arxiv','q','s','tE','u0','ra','dec','notes']

def load_labels(label_paths: List[Path], default_year: int) -> pd.DataFrame:
    """Load and normalize all label CSVs into a single dataframe."""
    frames = []
    for p in label_paths:
        if not p.exists():
            print(f"[warn] label file not found: {p}")
            continue
        df = pd.read_csv(p)
        for req in MIN_LABEL_COLS:
            if req not in df.columns:
                df[req] = None
        # normalize to canonical
        df['event_id'] = df['event_id_raw'].map(lambda x: to_canonical_event_id(x, default_year))
        # ensure all expected columns exist
        for k in KEEP_COLS:
            if k not in df.columns:
                df[k] = None
        frames.append(df[KEEP_COLS])
    if not frames:
        return pd.DataFrame(columns=KEEP_COLS)
    labels_all = pd.concat(frames, ignore_index=True)
    labels_all = labels_all.dropna(subset=['event_id']).drop_duplicates()
    # collapse duplicates by priority
    labels_best = (labels_all
                   .groupby('event_id', as_index=False, sort=False)
                   .apply(pick_label)
                   .reset_index(drop=True))
    # add short_id for secondary matching
    labels_best['short_id'] = labels_best['event_id'].map(to_short_id)
    return labels_best


def load_features(features_path: Path, default_year: int) -> pd.DataFrame:
    """Load features and normalize IDs."""
    df = pd.read_csv(features_path)
    if 'event' not in df.columns:
        # try to find a reasonable alternative
        candidates = [c for c in df.columns if 'event' in c.lower() or 'id' in c.lower()]
        if not candidates:
            raise ValueError("Features file must have an `event` column (e.g., 'blg-0001').")
        print(f"[info] Using '{candidates[0]}' as event column.")
        df.rename(columns={candidates[0]: 'event'}, inplace=True)
    df['event_id'] = df['event'].map(lambda x: to_canonical_event_id(x, default_year))
    df['short_id'] = df['event_id'].map(to_short_id)
    return df


def left_join_labels(features: pd.DataFrame, labels_best: pd.DataFrame) -> pd.DataFrame:
    """Join on canonical event_id; fall back to short_id when needed."""
    out = features.merge(labels_best, on='event_id', how='left', suffixes=('','_lbl'))
    need = out['lens_type'].isna()
    if need.any():
        # fallback to short_id
        fallback = features.loc[need, ['short_id']].merge(
            labels_best.drop_duplicates('short_id'),
            on='short_id', how='left'
        )
        for col in ['lens_type','confidence','source_ref','doi_or_arxiv','q','s','tE','u0','ra','dec','notes']:
            if col in out.columns and col in fallback.columns:
                out.loc[need, col] = out.loc[need, col].fillna(fallback[col].values)
    return out


def write_outputs(labeled: pd.DataFrame, labels_best: pd.DataFrame,
                  out_features: Path, out_db: Path) -> None:
    labeled.to_csv(out_features, index=False)
    print(f"[ok] wrote labeled features: {out_features}")
    labels_best.to_csv(out_db, index=False)
    print(f"[ok] wrote anomalies DB: {out_db}")
    # optional Parquet
    try:
        labeled.to_parquet(out_features.with_suffix('.parquet'), index=False)
        labels_best.to_parquet(out_db.with_suffix('.parquet'), index=False)
        print(f"[ok] also wrote Parquet versions.")
    except Exception:
        pass


def coverage_report(labeled: pd.DataFrame) -> None:
    n_total = len(labeled)
    n_bad_id = labeled['event_id'].isna().sum()
    n_labeled = labeled['lens_type'].notna().sum()
    print("\n=== Coverage Report =================================")
    print(f"Total events                 : {n_total}")
    print(f"Unparsed event IDs           : {n_bad_id} ({n_bad_id/max(n_total,1):.1%})")
    print(f"Labeled by anomalies DB      : {n_labeled} ({n_labeled/max(n_total,1):.1%})")
    if n_labeled:
        print("\nCounts by lens_type:")
        print(labeled['lens_type'].value_counts(dropna=True))
    # small hint report for misses
    misses = labeled[labeled['lens_type'].isna()]
    if len(misses) > 0:
        # show a few unmapped short_ids to help fill your label CSVs
        sample = misses['short_id'].dropna().value_counts().head(10).index.tolist()
        if sample:
            print("\nExamples of unmatched short_ids you might want to label next:")
            for sid in sample:
                print(f"  - {sid}")
    print("=====================================================\n")


# ----------------------------
# Main
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Build a Known Anomalies DB and join to OGLE-IV 2025 features.")
    ap.add_argument("--features", required=True, type=Path,
                    help="Path to your features CSV (must contain column 'event' like 'blg-0001').")
    ap.add_argument("--labels", nargs="+", type=Path, default=[],
                    help="One or more label CSV files you curate (glob in shell is fine).")
    ap.add_argument("--season", type=int, default=2025,
                    help="Default season/year to use for short IDs like 'blg-0001' (default: 2025).")
    ap.add_argument("--out-features", type=Path, default=Path("features_2025_with_labels.csv"),
                    help="Output CSV for labeled features (default: features_2025_with_labels.csv).")
    ap.add_argument("--out-db", type=Path, default=Path("known_anomalies.csv"),
                    help="Output CSV for the collapsed anomalies DB (default: known_anomalies.csv).")
    ap.add_argument("--show-map", action="store_true",
                    help="Print the first 10 event→canonical mappings for sanity.")
    return ap.parse_args()


def main():
    args = parse_args()

    # 1) Load features and normalize ids
    feats = load_features(args.features, args.season)
    if args.show_map:
        demo = feats[['event','event_id','short_id']].head(10)
        print("\nFirst 10 ID mappings:")
        for _, r in demo.iterrows():
            print(f"  {r['event']}  -->  {r['event_id']}  (short: {r['short_id']})")
        print()

    # 2) Load labels and build anomalies DB
    labels_best = load_labels(args.labels, args.season)
    if labels_best.empty:
        print("[warn] No labels loaded. You can still run the join; all lens_type will be NaN.")

    # 3) Join labels into features
    labeled = left_join_labels(feats, labels_best)

    # 4) Write outputs
    write_outputs(labeled, labels_best, args.out_features, args.out_db)

    # 5) Coverage report
    coverage_report(labeled)


if __name__ == "__main__":
    main()
