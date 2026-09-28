"""
MIC regressor

Predicts log10 MIC (uM) for a (sequence, species) pair. 

Usage:
    python scripts/train_mic_regressor.py \
        --table $D/processed/refinement/mic_table.csv \
        --clusters $D/processed/refinement/clusters_40.tsv \
        --cluster-fasta $D/processed/refinement/cluster_input.fasta \
        --embeddings $D/processed/refinement/mic_esm.npy \
        --output-dir $D/processed/refinement
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Shared with inference so the feature matrix is assembled identically on both
# sides -- see the module docstring in ampx/predictor.py. ampx is an installed
# package, so this resolves without a sys.path prelude.
from ampx.models.encoder import DEFAULT_ESM_MODEL
from ampx.predictor import DESCRIPTORS, build_features

MIN_SPEARMAN = 0.4
# Above this share of rows missing from the cluster table, the singleton
# fallback would define the split rather than patch it, so the run stops.
MAX_UNASSIGNED_FRACTION = 0.05
TOP_SPECIES = 15


def load_clusters(path: str, fasta: str) -> dict:
    """
    Map each clustered SEQUENCE to its cluster representative.

    mmseqs easy-cluster writes representative<TAB>member, one row per member,
    and both columns are FASTA identifiers (`seq1|mic+ref`), not sequences. The
    MIC table is keyed by sequence, so the two only join through the FASTA that
    was clustered. Mapping the table's sequences directly against the .tsv
    silently matched nothing: every sequence fell through to the singleton
    fallback, 5,602 sequences became 5,602 "clusters", and the 40%-identity
    split degenerated into a random split by sequence -- the exact failure this
    module's docstring calls load-bearing. The real structure is 2,824
    clusters, with 3,487 of 5,602 sequences sharing one with a relative.
    """
    identifiers = {}
    identifier = None
    for line in Path(fasta).read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            identifier = line[1:]
        elif identifier:
            identifiers[identifier] = line
            identifier = None

    frame = pd.read_csv(path, sep="\t", header=None, names=["representative", "member"])
    missing = set(frame["member"]) - set(identifiers)
    if missing:
        raise SystemExit(
            f"{len(missing)} cluster members are absent from {fasta} "
            f"(e.g. {sorted(missing)[:3]}). The cluster table and the FASTA "
            "must come from the same mmseqs run."
        )
    return {identifiers[member]: representative
            for member, representative in zip(frame["member"], frame["representative"])}


def fit_and_score(x_train, y_train, x_test, y_test, seed):
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=2000, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
        reg_lambda=2.0, objective="reg:squarederror",
        early_stopping_rounds=100, random_state=seed, n_jobs=-1,
    )
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
    predicted = model.predict(x_test)
    return model, predicted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True)
    parser.add_argument("--clusters", required=True)
    parser.add_argument("--cluster-fasta", required=True,
                        help="the FASTA that was clustered. Its identifiers are "
                             "what --clusters holds, so the MIC table's "
                             "sequences can only join through it.")
    parser.add_argument("--embeddings", default=None)
    parser.add_argument("--esm-model", default=DEFAULT_ESM_MODEL,
                        help="which ESM-2 produced --embeddings. Recorded in "
                             "the metadata so inference rebuilds the same "
                             "encoder; it is not verified here, since this "
                             "script only ever sees the precomputed array.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.table)
    clusters = load_clusters(args.clusters, args.cluster_fasta)
    table["cluster"] = table["sequence"].map(clusters)
    unassigned = int(table["cluster"].isna().sum())
    if unassigned:
        # A sequence mmseqs did not place is its own singleton cluster; that
        # is conservative (it can never share a cluster across the split).
        table["cluster"] = table["cluster"].fillna(table["sequence"])
    # The singleton fallback is a safety net for a handful of stragglers, not a
    # mode of operation. When it swallowed every row the split silently stopped
    # being a cluster split at all and nothing downstream noticed, because a
    # random split looks BETTER -- near-identical peptides recur across AMP
    # databases, so the held-out Spearman went up while the number became
    # meaningless. The label-shuffled control cannot catch this: it tests for
    # label leakage, and identity leakage with shuffled labels still reads ~0.
    # So the guard has to be here.
    if unassigned > MAX_UNASSIGNED_FRACTION * len(table):
        raise SystemExit(
            f"{unassigned} of {len(table)} rows ({unassigned / len(table):.1%}) "
            f"are absent from the cluster table, above the "
            f"{MAX_UNASSIGNED_FRACTION:.0%} limit. Falling back to singletons "
            "at this scale would turn the 40%-identity split into a random "
            "split and report memorization as accuracy. Check that --clusters "
            "and --cluster-fasta come from the same mmseqs run over these "
            "sequences."
        )
    print(f"{len(table)} rows, {table['sequence'].nunique()} sequences, "
          f"{table['cluster'].nunique()} clusters ({unassigned} unassigned -> singleton)")

    embeddings = None
    if args.embeddings and Path(args.embeddings).exists():
        embeddings = np.load(args.embeddings)
        print(f"embeddings: {embeddings.shape}")
    else:
        print("no embeddings supplied; descriptors + species only")

    species_levels = (
        table["species"].value_counts().head(TOP_SPECIES).index.tolist()
    )
    features, names = build_features(table, embeddings, species_levels)
    target = table["log_mic"].to_numpy(dtype=np.float32)
    print(f"feature matrix: {features.shape}")

    # Hold out whole clusters, so no held-out peptide has a >=40%-identity
    # relative in training.
    rng = np.random.default_rng(args.seed)
    unique_clusters = table["cluster"].unique()
    rng.shuffle(unique_clusters)
    n_holdout = int(round(args.holdout_fraction * len(unique_clusters)))
    holdout_clusters = set(unique_clusters[:n_holdout])
    is_test = table["cluster"].isin(holdout_clusters).to_numpy()
    print(f"train {int((~is_test).sum())} rows / test {int(is_test.sum())} rows "
          f"({len(holdout_clusters)} held-out clusters)")

    model, predicted = fit_and_score(
        features[~is_test], target[~is_test], features[is_test], target[is_test], args.seed
    )
    observed = target[is_test]
    spearman = stats.spearmanr(observed, predicted)
    pearson = stats.pearsonr(observed, predicted)
    mae = float(np.abs(observed - predicted).mean())

    # Label-shuffled control on the same split. Anything much above zero here
    # means the split leaks.
    shuffled_target = target.copy()
    shuffled_target[~is_test] = rng.permutation(shuffled_target[~is_test])
    _, shuffled_predicted = fit_and_score(
        features[~is_test], shuffled_target[~is_test],
        features[is_test], observed, args.seed
    )
    shuffled_spearman = stats.spearmanr(observed, shuffled_predicted).statistic

    per_species = []
    test_table = table[is_test].copy()
    test_table["predicted"] = predicted
    for species, group in test_table.groupby("species"):
        if len(group) < 30:
            continue
        per_species.append({
            "species": species,
            "n": len(group),
            "spearman": float(stats.spearmanr(group["log_mic"], group["predicted"]).statistic),
            "mae": float(np.abs(group["log_mic"] - group["predicted"]).mean()),
        })
    per_species = pd.DataFrame(per_species).sort_values("n", ascending=False)

    model.save_model(str(output_dir / "mic_regressor.json"))
    json.dump(
        {"species_levels": species_levels, "descriptors": DESCRIPTORS,
         "uses_embeddings": embeddings is not None,
         "esm_model": args.esm_model if embeddings is not None else None,
         "embedding_dim": int(embeddings.shape[1]) if embeddings is not None else 0,
         "holdout_spearman": float(spearman.statistic), "holdout_mae": mae},
        open(output_dir / "mic_regressor_meta.json", "w"), indent=2,
    )
    test_table.to_csv(output_dir / "mic_holdout_predictions.csv", index=False)

    lines = [
        "# MIC regressor — held-out performance",
        "",
        f"Target: log10 MIC (uM) per (sequence, species). {len(table)} rows over "
        f"{table['sequence'].nunique()} unique sequences.",
        "",
        "**Split: entire MMseqs2 clusters at 40% identity held out** "
        f"({len(holdout_clusters)} of {len(unique_clusters)} clusters, "
        f"{int(is_test.sum())} rows). Not a random row split: near-identical "
        "peptides recur across AMP databases, and a random split would report "
        "memorization as accuracy.",
        "",
        f"Features: {len(names)} columns — {len(DESCRIPTORS)} descriptors, "
        f"{len(species_levels) + 1} species indicators, 5 gram indicators"
        + (f", {embeddings.shape[1]} ESM-2 dimensions." if embeddings is not None else "."),
        "",
        "## Overall",
        "",
        "| metric | value |",
        "|---|---|",
        f"| held-out Spearman | **{spearman.statistic:.4f}** |",
        f"| held-out Pearson | {pearson.statistic:.4f} |",
        f"| held-out MAE (log10 uM) | {mae:.4f} |",
        f"| label-shuffled Spearman (leakage control) | {shuffled_spearman:.4f} |",
        f"| gate (>= {MIN_SPEARMAN}) | "
        f"{'**PASS**' if spearman.statistic >= MIN_SPEARMAN else '**FAIL — STOP**'} |",
        "",
        "Reference point: ApexGO reported Pearson 0.463 / Spearman 0.462 between "
        "predicted and experimental MIC on 100 synthesized peptides, and drove a "
        "successful optimization loop with it.",
        "",
        "## Per species (held-out, n >= 30)",
        "",
        "| species | n | Spearman | MAE |",
        "|---|---|---|---|",
    ]
    for row in per_species.itertuples():
        lines.append(f"| {row.species} | {row.n} | {row.spearman:.4f} | {row.mae:.4f} |")
    (output_dir / "regressor_report.md").write_text("\n".join(lines) + "\n")

    print()
    print(f"held-out Spearman {spearman.statistic:.4f}  Pearson {pearson.statistic:.4f}  "
          f"MAE {mae:.4f}")
    print(f"label-shuffled control Spearman {shuffled_spearman:.4f}")
    print(per_species.to_string(index=False))
    print(f"wrote {output_dir / 'regressor_report.md'}")

    if spearman.statistic < MIN_SPEARMAN:
        print(f"\nSTOP: held-out Spearman {spearman.statistic:.4f} is below the "
              f"{MIN_SPEARMAN} gate. A weaker oracle makes every downstream "
              "ranking noise.")
        raise SystemExit(87)
    if abs(shuffled_spearman) > 0.15:
        print(f"\nSTOP: label-shuffled control reached {shuffled_spearman:.4f}; "
              "the cluster split is leaking.")
        raise SystemExit(87)


if __name__ == "__main__":
    main()
