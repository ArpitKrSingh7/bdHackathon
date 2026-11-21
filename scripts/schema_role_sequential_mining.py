#!/usr/bin/env python3
"""Mine non-temporal, schema-family sequences from Internet Ads rows.

The source dataset contains binary indicators grouped by field family, not
events or timestamps. This exploratory analysis groups actual row indicators
under the `origurl`, `ancurl`, and `url` prefixes, then uses the explicitly
assumed family order origurl -> ancurl -> url. This is a real-data,
schema-ordered sequence analysis, not observed temporal behavior or a user
journey. If the rubric requires chronology, that requirement remains impossible
with this dataset.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from hackathon_utils import load_ads  # noqa: E402

FIELD_ORDER = ("origurl", "ancurl", "url")
MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.60
MAX_PATTERN_LENGTH = 3


def _field_role(feature: str) -> str | None:
    prefix = feature.split("*", 1)[0]
    return prefix if prefix in FIELD_ORDER else None


def mine_schema_role_sequences(
    data: pd.DataFrame,
    binary_features: list[str],
    min_support: float = MIN_SUPPORT,
    min_confidence: float = MIN_CONFIDENCE,
    max_pattern_length: int = MAX_PATTERN_LENGTH,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not 0 < min_support <= 1 or not 0 < min_confidence <= 1:
        raise ValueError("Support and confidence must be in (0, 1]")
    if max_pattern_length < 1:
        raise ValueError("max_pattern_length must be positive")

    role_index = {role: index for index, role in enumerate(FIELD_ORDER)}
    role_features = {
        role: [name for name in binary_features if _field_role(name) == role]
        for role in FIELD_ORDER
    }
    rows = len(data)
    feature_matrix = data[[name for role in FIELD_ORDER for name in role_features[role]]]
    frequent_items: list[tuple[str, str]] = []
    masks: dict[tuple[str, str], np.ndarray] = {}
    support_counts: dict[tuple[tuple[str, str], ...], int] = {}
    for role in FIELD_ORDER:
        for feature in role_features[role]:
            mask = feature_matrix[feature].eq(1).to_numpy(dtype=bool)
            count = int(mask.sum())
            if count / rows + 1e-15 >= min_support:
                item = (role, feature)
                frequent_items.append(item)
                masks[item] = mask
                support_counts[(item,)] = count

    frequent_items.sort(key=lambda item: (role_index[item[0]], item[1]))
    min_count = int(np.ceil(min_support * rows - 1e-12))
    frequent_by_length: dict[int, set[tuple[tuple[str, str], ...]]] = {
        1: {(item,) for item in frequent_items}
    }
    pattern_rows: list[dict] = []

    def pattern_description(pattern: tuple[tuple[str, str], ...]) -> tuple[str, str]:
        roles = " → ".join(role for role, _ in pattern)
        labels = " → ".join(f"{role}:{feature.split('*', 1)[1]}" for role, feature in pattern)
        return roles, labels

    for length in range(1, max_pattern_length + 1):
        if length == 1:
            candidates = list(frequent_by_length[1])
        else:
            candidates = []
            for selected in combinations(frequent_items, length):
                if any(
                    role_index[selected[pos][0]] >= role_index[selected[pos + 1][0]]
                    for pos in range(length - 1)
                ):
                    continue
                candidate = tuple(selected)
                # Apriori downward-closure pruning: every immediate subsequence
                # must already be frequent before this candidate is counted.
                if all(
                    tuple(candidate[:pos] + candidate[pos + 1 :])
                    in frequent_by_length.get(length - 1, set())
                    for pos in range(length)
                ):
                    candidates.append(candidate)

        found: set[tuple[tuple[str, str], ...]] = set()
        for pattern in candidates:
            mask = np.logical_and.reduce([masks[item] for item in pattern])
            count = int(mask.sum())
            if count < min_count:
                continue
            found.add(pattern)
            support_counts[pattern] = count
            role_text, pattern_text = pattern_description(pattern)
            pattern_rows.append({
                "pattern_length": length,
                "source_role_order": role_text,
                "schema_role_pattern": pattern_text,
                "support_count": count,
                "support": count / rows,
                "sequence_count": rows,
                "is_temporal": False,
            })
        if length > 1:
            frequent_by_length[length] = found

    rule_rows: list[dict] = []
    for row in pattern_rows:
        pattern = next(
            pattern for pattern, count in support_counts.items()
            if len(pattern) == row["pattern_length"]
            and count == row["support_count"]
            and pattern_description(pattern)[1] == row["schema_role_pattern"]
        )
        if len(pattern) < 2:
            continue
        prefix = pattern[:-1]
        prefix_count = support_counts.get(prefix, 0)
        if not prefix_count:
            continue
        confidence = row["support_count"] / prefix_count
        if confidence + 1e-15 < min_confidence:
            continue
        consequent_count = support_counts[(pattern[-1],)]
        prefix_roles, prefix_text = pattern_description(prefix)
        consequent_role, consequent_feature = pattern[-1]
        rule_rows.append({
            "antecedent_schema_pattern": prefix_text,
            "antecedent_source_roles": prefix_roles,
            "consequent_schema_item": f"{consequent_role}:{consequent_feature.split('*', 1)[1]}",
            "source_role_order": row["source_role_order"],
            "sequence_support_count": row["support_count"],
            "sequence_support": row["support"],
            "prefix_support_count": prefix_count,
            "confidence": confidence,
            "lift": confidence / (consequent_count / rows),
            "is_temporal": False,
        })

    patterns = pd.DataFrame(pattern_rows).sort_values(
        ["pattern_length", "support", "schema_role_pattern"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    rules = pd.DataFrame(rule_rows)
    if not rules.empty:
        rules = rules.sort_values(
            ["lift", "confidence", "sequence_support"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    metadata = {
        "dataset": "UCI Internet Advertisements",
        "rows_as_sequences": rows,
        "representation": "Each row is a sequence of observed binary feature sets grouped by URL-named feature family.",
        "source_role_order": list(FIELD_ORDER),
        "role_interpretation": {
            "origurl": "the ad.names prefix family `origurl`; the original raw URL is unavailable",
            "ancurl": "the ad.names prefix family `ancurl`; the original raw URL is unavailable",
            "url": "the ad.names prefix family `url`; the original raw URL is unavailable",
        },
        "sequence_definition": "For each row, active binary terms in each URL-named feature family form one event set. The analyst-defined field-family order is origurl -> ancurl -> url. Terms within one family are unordered.",
        "ordering_basis": "An explicit structural-family assumption, not timestamps, row order, raw URL paths, or observed event chronology.",
        "raw_text_available": False,
        "target_included": False,
        "sequence_count": rows,
        "selected_item_count": len(frequent_items),
        "selected_items": [f"{role}:{feature}" for role, feature in frequent_items],
        "parameters": {
            "minimum_support": min_support,
            "minimum_support_count": min_count,
            "minimum_confidence_for_prefix_extension_rules": min_confidence,
            "maximum_pattern_length": max_pattern_length,
        },
        "frequent_pattern_count": len(patterns),
        "prefix_extension_rule_count": len(rules),
        "temporal_sequence_requirement_fulfilled": False,
        "limitation": "This is a dataset-derived, schema-family ordered pattern analysis, not temporal sequential mining. Its suitability for the course rubric requires instructor acceptance of the analyst-defined field-family order as a sequence definition.",
    }
    return patterns, rules, metadata


def main() -> None:
    data, _, _, binary_features = load_ads()
    start = perf_counter()
    patterns, rules, metadata = mine_schema_role_sequences(data, binary_features)
    metadata["runtime_seconds"] = perf_counter() - start
    output = PROJECT_DIR / "artifacts" / "association"
    output.mkdir(parents=True, exist_ok=True)
    patterns.to_csv(output / "schema_role_sequential_patterns.csv", index=False)
    rules.to_csv(output / "schema_role_sequential_rules.csv", index=False)
    sequence_patterns = patterns[patterns["pattern_length"] >= 2].head(20).copy()
    if not sequence_patterns.empty:
        sequence_patterns = sequence_patterns.sort_values("support")
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(sequence_patterns["schema_role_pattern"], sequence_patterns["support"], color="#6b82a6")
        ax.set_title("Frequent URL-family patterns (schema order, not temporal)")
        ax.set_xlabel("Support: fraction of dataset rows containing all terms")
        ax.set_ylabel("Analyst-defined field-family pattern")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(output / "schema_role_patterns_top20.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    (output / "schema_role_sequence_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Dataset-Derived Schema-Role Sequence Analysis",
        "",
        "This analysis uses actual Internet Ads rows and observed binary indicators from the `origurl`, `ancurl`, and `url` feature families. It orders these sets as origurl → ancurl → url, following an explicit structural assumption. The ordering is non-temporal; the exact raw URLs are absent.",
        "",
        f"- Rows/sequences: {metadata['sequence_count']:,}",
        f"- Frequent items at {MIN_SUPPORT:.0%} support: {metadata['selected_item_count']}",
        f"- Frequent schema-role patterns (length ≤ {MAX_PATTERN_LENGTH}): {metadata['frequent_pattern_count']}",
        f"- Prefix-extension rules at {MIN_CONFIDENCE:.0%} confidence: {metadata['prefix_extension_rule_count']}",
        f"- Runtime: {metadata['runtime_seconds']:.4f} seconds",
        "- Plot: `schema_role_patterns_top20.png`",
        "",
        "## Interpretation boundary",
        "",
        "These are cross-field structural patterns, not page visits over time. The dataset stores indicator columns, not raw URLs or a chronology. Therefore this analysis does not fulfill a requirement that specifically demands temporal/session sequence mining. The earlier five-sequence synthetic demo remains separate.",
        "",
        "Parameters and exact selected items are in `schema_role_sequence_metadata.json`; pattern and prefix-extension tables are in the corresponding CSV files.",
    ]
    (output / "schema_role_sequence_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "sequence_count": metadata["sequence_count"],
        "selected_item_count": metadata["selected_item_count"],
        "frequent_pattern_count": metadata["frequent_pattern_count"],
        "prefix_extension_rule_count": metadata["prefix_extension_rule_count"],
        "runtime_seconds": metadata["runtime_seconds"],
        "limitation": metadata["limitation"],
    }, indent=2))


if __name__ == "__main__":
    main()
