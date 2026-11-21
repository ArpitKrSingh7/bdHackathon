"""Apriori, CHARM, and a separate synthetic sequential-pattern demonstration.

Internet Ads is cross-sectional: each observation is a web-page/image example,
not a timestamped event. The dataset miners below treat each row as one ordinary
transaction. Sequential patterns are demonstrated only on an explicit toy event
sequence and are never attributed to the Internet Ads data.
"""
from __future__ import annotations

from itertools import combinations
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json
import platform

import numpy as np
import pandas as pd

try:  # Package import (notebook) and direct-script execution are both supported.
    from .hackathon_utils import RAW_DATA_DIR, load_ads
except ImportError:  # pragma: no cover - exercised by the direct-script entrypoint
    from hackathon_utils import RAW_DATA_DIR, load_ads

TARGET = "class"
DEFAULT_MIN_SUPPORT = 0.05
DEFAULT_MIN_CONFIDENCE = 0.60
DEFAULT_UNKNOWN_LOCAL_ITEM = "local=unknown"


def build_transactions(
    data: pd.DataFrame,
    binary_features: Sequence[str],
    include_class: bool = True,
    min_item_support: Optional[float] = DEFAULT_MIN_SUPPORT,
    target_col: str = TARGET,
) -> dict:
    """Build row-preserving transactions from observed binary indicators.

    Only observed values equal to 1 become feature items. Missing local values
    receive an explicit unknown token; missing values are not converted into
    observed zeroes. A target class token is appended only when include_class is
    true, which is the justified class-association setting used for class rules.
    Numeric geometry is excluded because no bins have been defined for it.

    Feature items below min_item_support are omitted as a search-space filter.
    The denominator is all rows, including rows where a feature is unknown.
    Repeated input rows remain repeated transaction entries and therefore retain
    their empirical multiplicity in support counts.
    """
    if data.empty:
        raise ValueError("Cannot build transactions from an empty table")
    if min_item_support is not None and not (0.0 < min_item_support <= 1.0):
        raise ValueError("min_item_support must be in (0, 1] or None")
    missing_cols = [c for c in binary_features if c not in data.columns]
    if missing_cols:
        raise KeyError(f"Binary feature columns are missing: {missing_cols[:10]}")
    if include_class and target_col not in data.columns:
        raise KeyError(f"Target column {target_col!r} is missing")

    n_rows = len(data)
    support_counts: Dict[str, int] = {}
    selected_features: List[str] = []
    for feature in binary_features:
        values = data[feature]
        unexpected = set(values.dropna().unique()) - {0, 1}
        if unexpected:
            raise ValueError(f"{feature} is not binary; observed {sorted(unexpected)}")
        count = int(values.eq(1).sum())
        item = f"{feature}=1"
        support_counts[item] = count
        rate = count / n_rows
        if min_item_support is None or rate >= min_item_support:
            selected_features.append(feature)

    transactions: List[List[str]] = []
    local_is_binary = "local" in binary_features
    for _, row in data.iterrows():
        items = [f"{feature}=1" for feature in selected_features if row[feature] == 1]
        if local_is_binary and pd.isna(row["local"]):
            items.append(DEFAULT_UNKNOWN_LOCAL_ITEM)
            support_counts[DEFAULT_UNKNOWN_LOCAL_ITEM] = support_counts.get(DEFAULT_UNKNOWN_LOCAL_ITEM, 0) + 1
        if include_class:
            label = str(row[target_col]).strip().rstrip(".")
            if label not in {"ad", "nonad"}:
                raise ValueError(f"Unexpected class label {label!r}")
            class_item = f"class={label}"
            items.append(class_item)
            support_counts[class_item] = support_counts.get(class_item, 0) + 1
        transactions.append(sorted(items))

    item_support = {item: count / n_rows for item, count in support_counts.items()}
    return {
        "transactions": transactions,
        "selected_features": selected_features,
        "item_support": item_support,
        "transaction_count": n_rows,
        "include_class": include_class,
        "target_col": target_col if include_class else None,
        "min_item_support": min_item_support,
    }


def _normalize_transactions(transactions_or_build: Iterable[Iterable[str]] | Mapping) -> List[frozenset]:
    if isinstance(transactions_or_build, Mapping):
        transactions_or_build = transactions_or_build["transactions"]
    transactions = [frozenset(str(item) for item in row) for row in transactions_or_build]
    if not transactions:
        raise ValueError("At least one transaction is required")
    return transactions


def _tid_masks(transactions: Sequence[frozenset]) -> Dict[str, int]:
    """Vertical item-to-TID bitmasks; repeated rows get distinct bit positions."""
    masks: Dict[str, int] = {}
    for tid, transaction in enumerate(transactions):
        bit = 1 << tid
        for item in transaction:
            masks[item] = masks.get(item, 0) | bit
    return masks


def _minimum_count(min_support: float, n_transactions: int) -> int:
    if not (0.0 < min_support <= 1.0):
        raise ValueError("min_support must be in (0, 1]")
    return ceil(min_support * n_transactions - 1e-12)


def _itemset_frame(itemset_tid_masks: Mapping[frozenset, int], n_transactions: int) -> pd.DataFrame:
    rows = []
    for itemset, tids in itemset_tid_masks.items():
        ordered = tuple(sorted(itemset))
        count = tids.bit_count()
        rows.append({
            "itemset": ordered,
            "itemset_text": " AND ".join(ordered),
            "length": len(ordered),
            "support_count": count,
            "support": count / n_transactions,
        })
    columns = ["itemset", "itemset_text", "length", "support_count", "support"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["length", "support_count", "itemset_text"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def _rule_frame(
    frequent_itemset_tids: Mapping[frozenset, int],
    singleton_tids: Mapping[str, int],
    n_transactions: int,
    min_confidence: float,
) -> pd.DataFrame:
    """Generate all nonempty antecedent/consequent partitions of frequent sets."""
    rows = []
    full_mask = (1 << n_transactions) - 1
    for union, union_tids in frequent_itemset_tids.items():
        if len(union) < 2:
            continue
        ordered = tuple(sorted(union))
        union_count = union_tids.bit_count()
        for antecedent_size in range(1, len(ordered)):
            for antecedent_values in combinations(ordered, antecedent_size):
                antecedent = frozenset(antecedent_values)
                consequent = union - antecedent
                ante_tids = full_mask
                for item in antecedent:
                    ante_tids &= singleton_tids[item]
                cons_tids = full_mask
                for item in consequent:
                    cons_tids &= singleton_tids[item]
                ante_count = ante_tids.bit_count()
                cons_count = cons_tids.bit_count()
                if not ante_count or not cons_count:
                    continue
                confidence = union_count / ante_count
                if confidence + 1e-15 < min_confidence:
                    continue
                support = union_count / n_transactions
                consequent_support = cons_count / n_transactions
                lift = confidence / consequent_support
                rows.append({
                    "antecedent": tuple(sorted(antecedent)),
                    "consequent": tuple(sorted(consequent)),
                    "antecedent_text": " AND ".join(sorted(antecedent)),
                    "consequent_text": " AND ".join(sorted(consequent)),
                    "union_itemset": ordered,
                    "support_count": union_count,
                    "support": support,
                    "confidence": confidence,
                    "lift": lift,
                    "leverage": support - (ante_count / n_transactions) * consequent_support,
                })
    columns = [
        "antecedent", "consequent", "antecedent_text", "consequent_text",
        "union_itemset", "support_count", "support", "confidence", "lift", "leverage",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["lift", "confidence", "support"], ascending=[False, False, False]
    ).reset_index(drop=True)


def run_apriori(
    transactions_or_build: Iterable[Iterable[str]] | Mapping,
    min_support: float = DEFAULT_MIN_SUPPORT,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_len: Optional[int] = None,
) -> dict:
    """Run horizontal Apriori candidate generation with vertical support counts.

    Candidate k-itemsets are joined from frequent (k-1)-itemsets sharing a
    prefix, then pruned when any (k-1)-subset is infrequent. TID bitmasks make
    support counting efficient while each original row remains a distinct TID.
    """
    if not (0.0 <= min_confidence <= 1.0):
        raise ValueError("min_confidence must be in [0, 1]")
    transactions = _normalize_transactions(transactions_or_build)
    n_transactions = len(transactions)
    min_count = _minimum_count(min_support, n_transactions)
    masks = _tid_masks(transactions)
    started = perf_counter()

    current: Dict[frozenset, int] = {
        frozenset([item]): tids for item, tids in masks.items()
        if tids.bit_count() >= min_count
    }
    frequent = dict(current)
    level = 2
    while current and (max_len is None or level <= max_len):
        previous_sets = set(current)
        previous_tuples = sorted(tuple(sorted(itemset)) for itemset in previous_sets)
        candidates = set()
        for left, right in combinations(previous_tuples, 2):
            if left[:-1] != right[:-1]:
                continue
            candidate = tuple(sorted(set(left) | set(right)))
            if len(candidate) != level:
                continue
            if all(frozenset(subset) in previous_sets
                   for subset in combinations(candidate, level - 1)):
                candidates.add(candidate)

        next_level: Dict[frozenset, int] = {}
        for candidate in candidates:
            tids = masks[candidate[0]]
            for item in candidate[1:]:
                tids &= masks[item]
            if tids.bit_count() >= min_count:
                next_level[frozenset(candidate)] = tids
        frequent.update(next_level)
        current = next_level
        level += 1

    elapsed = perf_counter() - started
    itemsets = _itemset_frame(frequent, n_transactions)
    rules = _rule_frame(frequent, masks, n_transactions, min_confidence)
    class_rules = rules[
        rules["consequent"].map(
            lambda consequent: len(consequent) == 1 and consequent[0] in {"class=ad", "class=nonad"}
        )
    ].reset_index(drop=True)
    return {
        "itemsets": itemsets,
        "rules": rules,
        "class_rules": class_rules,
        "elapsed_seconds": elapsed,
        "transaction_count": n_transactions,
        "parameters": {
            "min_support": min_support,
            "min_support_count": min_count,
            "min_confidence": min_confidence,
            "max_len": max_len,
        },
    }


def run_charm(
    transactions_or_build: Iterable[Iterable[str]] | Mapping,
    min_support: float = DEFAULT_MIN_SUPPORT,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    verify_exhaustively: bool = True,
) -> dict:
    """Mine closed frequent itemsets with CHARM's TID-set properties.

    This depth-first implementation keeps itemset/TID-set pairs, intersects
    vertical TID bitmasks, and applies CHARM Properties 1–4 (equal, subset,
    superset, and incomparable TID sets). It returns closure itemsets keyed by
    their exact transaction sets, plus a bounded exhaustive verification on the
    same frequent singleton alphabet when requested.
    """
    transactions = _normalize_transactions(transactions_or_build)
    n_transactions = len(transactions)
    min_count = _minimum_count(min_support, n_transactions)
    masks = _tid_masks(transactions)
    frequent_singletons = {
        item: tids for item, tids in masks.items() if tids.bit_count() >= min_count
    }
    started = perf_counter()
    counters = {"tidset_intersections": 0, "generated_children": 0, "recursive_calls": 0}
    initial_nodes = [(frozenset([item]), tids) for item, tids in frequent_singletons.items()]

    def extend(nodes: List[Tuple[frozenset, int]]) -> Dict[int, set]:
        counters["recursive_calls"] += 1
        nodes = sorted(nodes, key=lambda node: (node[1].bit_count(), tuple(sorted(node[0]))))
        closed_by_tidset: Dict[int, set] = {}
        i = 0
        while i < len(nodes):
            xi, ti = nodes[i]
            children: List[Tuple[frozenset, int]] = []
            j = i + 1
            while j < len(nodes):
                xj, tj = nodes[j]
                intersection = ti & tj
                counters["tidset_intersections"] += 1
                if intersection.bit_count() < min_count:
                    j += 1
                    continue
                old_xi = xi
                joined = xi | xj
                if ti == tj:
                    # Property 1: same TID set, so merge into one closure branch.
                    nodes.pop(j)
                    xi = joined
                    nodes[i] = (xi, ti)
                    children = [
                        (child_items | xj if old_xi.issubset(child_items) else child_items, child_tids)
                        for child_items, child_tids in children
                    ]
                elif intersection == ti:
                    # Property 2: T(Xi) subset T(Xj); Xj belongs to Xi's closure.
                    xi = joined
                    nodes[i] = (xi, ti)
                    children = [
                        (child_items | xj if old_xi.issubset(child_items) else child_items, child_tids)
                        for child_items, child_tids in children
                    ]
                    j += 1
                elif intersection == tj:
                    # Property 3: T(Xj) subset T(Xi); carry the subsumed branch down.
                    nodes.pop(j)
                    children.append((joined, tj))
                    counters["generated_children"] += 1
                else:
                    # Property 4: incomparable parents; recurse on their intersection.
                    children.append((joined, intersection))
                    counters["generated_children"] += 1
                    j += 1

            if children:
                for tids, child_items in extend(children).items():
                    closed_by_tidset.setdefault(tids, set()).update(child_items)
            closed_by_tidset.setdefault(ti, set()).update(xi)
            i += 1
        return closed_by_tidset

    raw_closed = extend(initial_nodes)
    closed: Dict[frozenset, int] = {}
    for tids in raw_closed:
        # Closure: every frequent singleton present in every transaction in T(X).
        closure = frozenset(
            item for item, item_tids in frequent_singletons.items()
            if (item_tids & tids) == tids
        )
        if closure and tids.bit_count() >= min_count:
            closed[closure] = tids
    charm_elapsed = perf_counter() - started

    verification = {"performed": False, "matches_exhaustive": None}
    verify_started = perf_counter()
    if verify_exhaustively:
        expected: Dict[frozenset, int] = {}
        ordered_items = sorted(frequent_singletons)
        full_mask = (1 << n_transactions) - 1

        def enumerate_frequent(prefix: Tuple[str, ...], start: int, tids: int):
            for position in range(start, len(ordered_items)):
                item = ordered_items[position]
                next_tids = tids & frequent_singletons[item]
                if next_tids.bit_count() < min_count:
                    continue
                next_prefix = prefix + (item,)
                expected.setdefault(next_tids, next_tids)
                enumerate_frequent(next_prefix, position + 1, next_tids)

        enumerate_frequent(tuple(), 0, full_mask)
        exhaustive_closures = {
            frozenset(
                item for item, item_tids in frequent_singletons.items()
                if (item_tids & tids) == tids
            ): tids
            for tids in expected
        }
        verification = {
            "performed": True,
            "matches_exhaustive": set(exhaustive_closures.items()) == set(closed.items()),
            "exhaustive_frequent_tidsets": len(expected),
            "exhaustive_closed_itemsets": len(exhaustive_closures),
            "verification_seconds": perf_counter() - verify_started,
        }
        if not verification["matches_exhaustive"]:
            missing = set(exhaustive_closures.items()) - set(closed.items())
            extra = set(closed.items()) - set(exhaustive_closures.items())
            raise RuntimeError(
                "CHARM verification against exhaustive TID intersections failed: "
                f"missing={len(missing)}, extra={len(extra)}"
            )

    itemsets = _itemset_frame(closed, n_transactions)
    full_mask = (1 << n_transactions) - 1
    rule_rows = []
    for itemset, tids in closed.items():
        class_items = [item for item in itemset if item in {"class=ad", "class=nonad"}]
        if len(class_items) != 1:
            continue
        consequent = class_items[0]
        antecedent = itemset - {consequent}
        if not antecedent:
            continue
        ante_tids = full_mask
        for item in antecedent:
            ante_tids &= masks[item]
        antecedent_count = ante_tids.bit_count()
        support_count = tids.bit_count()
        if not antecedent_count:
            continue
        confidence = support_count / antecedent_count
        if confidence + 1e-15 < min_confidence:
            continue
        consequent_support = masks[consequent].bit_count() / n_transactions
        support = support_count / n_transactions
        rule_rows.append({
            "antecedent": tuple(sorted(antecedent)),
            "consequent": (consequent,),
            "antecedent_text": " AND ".join(sorted(antecedent)),
            "consequent_text": consequent,
            "closed_itemset": tuple(sorted(itemset)),
            "support_count": support_count,
            "support": support,
            "confidence": confidence,
            "lift": confidence / consequent_support,
        })
    closed_rules = pd.DataFrame(rule_rows)
    if not closed_rules.empty:
        closed_rules = closed_rules.sort_values(
            ["lift", "confidence", "support"], ascending=[False, False, False]
        ).reset_index(drop=True)
    return {
        "itemsets": itemsets,
        "class_rules": closed_rules,
        "elapsed_seconds": charm_elapsed,
        "transaction_count": n_transactions,
        "parameters": {
            "min_support": min_support,
            "min_support_count": min_count,
            "min_confidence": min_confidence,
            "search": "CHARM TID-set properties 1-4 using integer bitmasks",
        },
        "search_counters": counters,
        "verification": verification,
    }


def _is_subsequence(pattern: Tuple[str, ...], sequence: Sequence[str]) -> bool:
    position = 0
    for event in sequence:
        if position < len(pattern) and pattern[position] == event:
            position += 1
    return position == len(pattern)


def mine_toy_sequential_apriori(
    sequences: Optional[Sequence[Sequence[str]]] = None,
    min_support: float = 0.40,
    max_len: int = 4,
) -> dict:
    """GSP-style sequential Apriori demonstration on explicitly synthetic data.

    This function is intentionally independent from Internet Ads rows. Each toy
    event is a single abstract item, and sequence order is supplied by the toy
    fixture rather than inferred from attributes or input row order.
    """
    if sequences is None:
        sequences = [
            ["view", "search", "click", "purchase"],
            ["view", "search", "click"],
            ["search", "click", "purchase"],
            ["view", "click", "purchase"],
            ["view", "search", "purchase"],
        ]
    sequences = [tuple(str(event) for event in seq) for seq in sequences]
    if not sequences or not (0.0 < min_support <= 1.0):
        raise ValueError("Provide toy sequences and min_support in (0, 1]")
    minimum_count = _minimum_count(min_support, len(sequences))
    sequence_counts: Dict[Tuple[str, ...], int] = {}
    level = {((item,),) for seq in sequences for item in seq}
    # Normalize a pattern representation as tuple[str,...], one event per position.
    frequent: Dict[Tuple[str, ...], int] = {}
    current = set(tuple(pattern[0]) for pattern in level)
    for pattern in current:
        count = sum(_is_subsequence(pattern, seq) for seq in sequences)
        if count >= minimum_count:
            frequent[pattern] = count
    k = 2
    while current and k <= max_len:
        candidates = set()
        ordered = sorted(current)
        for left in ordered:
            for right in ordered:
                if k == 2:
                    # Every ordered pair of frequent 1-sequences is a candidate.
                    candidate = left + right
                else:
                    if left[1:] != right[:-1]:
                        continue
                    candidate = left + (right[-1],)
                if len(candidate) != k:
                    continue
                # GSP pruning: every length-(k-1) subsequence must be frequent.
                if all(candidate[:i] + candidate[i + 1:] in current for i in range(k)):
                    candidates.add(candidate)
        next_level = {}
        for candidate in candidates:
            count = sum(_is_subsequence(candidate, seq) for seq in sequences)
            if count >= minimum_count:
                next_level[candidate] = count
        frequent.update(next_level)
        current = set(next_level)
        k += 1

    rows = [{
        "pattern": " -> ".join(pattern),
        "length": len(pattern),
        "support_count": count,
        "support": count / len(sequences),
        "source": "synthetic toy only; not Internet Ads data",
    } for pattern, count in frequent.items()]
    frame = pd.DataFrame(rows).sort_values(
        ["length", "support_count", "pattern"], ascending=[True, False, True]
    ).reset_index(drop=True)
    return {
        "patterns": frame,
        "sequences": [list(seq) for seq in sequences],
        "parameters": {"min_support": min_support, "min_support_count": minimum_count, "max_len": max_len},
    }


def _csv_safe(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if out[column].map(lambda value: isinstance(value, tuple)).any():
            out[column] = out[column].map(lambda value: " | ".join(value) if isinstance(value, tuple) else value)
    return out


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without optional tabulate dependency."""
    safe = _csv_safe(frame)
    if safe.empty:
        return ""
    headers = [str(column).replace("|", "\\|") for column in safe.columns]
    rows = [[str(value).replace("|", "\\|") for value in row]
            for row in safe.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _run_sensitivity(
    feature_build: dict,
    class_build: dict,
    supports=(0.02, 0.03, 0.05, 0.08),
) -> pd.DataFrame:
    rows = []
    for support in supports:
        # A common depth cap keeps the threshold comparison finite and comparable.
        feature_result = run_apriori(
            feature_build["transactions"], min_support=support,
            min_confidence=0.60, max_len=3,
        )
        class_result = run_apriori(
            class_build["transactions"], min_support=support,
            min_confidence=0.60, max_len=3,
        )
        counts = feature_result["itemsets"]["length"].value_counts().to_dict()
        rows.append({
            "min_support": support,
            "minimum_count": feature_result["parameters"]["min_support_count"],
            "feature_frequent_singletons": int(counts.get(1, 0)),
            "feature_frequent_pairs": int(counts.get(2, 0)),
            "feature_frequent_triples": int(counts.get(3, 0)),
            "feature_itemsets_through_length_3": len(feature_result["itemsets"]),
            "class_rules_at_confidence_0_60_max_len_3": len(class_result["class_rules"]),
            "feature_only_runtime_seconds": feature_result["elapsed_seconds"],
            "class_augmented_runtime_seconds": class_result["elapsed_seconds"],
            "max_len": 3,
        })
    return pd.DataFrame(rows)


def run_dataset_association(
    output_dir: Path | str,
    min_support: float = DEFAULT_MIN_SUPPORT,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_item_support: Optional[float] = None,
    sensitivity_supports=(0.02, 0.03, 0.05, 0.08),
) -> dict:
    """Persist feature-only mining, separate class rules, and the toy sequence demo."""
    data, feature_names, numeric_features, binary_features = load_ads()
    if min_item_support is None:
        min_item_support = min_support
    feature_build = build_transactions(
        data, binary_features, include_class=False,
        min_item_support=min_item_support, target_col=TARGET,
    )
    class_build = build_transactions(
        data, binary_features, include_class=True,
        min_item_support=min_item_support, target_col=TARGET,
    )
    apriori = run_apriori(
        feature_build, min_support=min_support, min_confidence=min_confidence
    )
    class_apriori = run_apriori(
        class_build, min_support=min_support, min_confidence=min_confidence
    )
    charm = run_charm(
        feature_build, min_support=min_support, min_confidence=min_confidence,
        verify_exhaustively=True,
    )
    class_charm = run_charm(
        class_build, min_support=min_support, min_confidence=min_confidence,
        verify_exhaustively=True,
    )
    # Use the lowest support under study to choose one common item alphabet for
    # the sensitivity sweep; each run then prunes by its own minimum support.
    sensitivity_support_floor = min(sensitivity_supports)
    sensitivity_feature_build = build_transactions(
        data, binary_features, include_class=False,
        min_item_support=sensitivity_support_floor, target_col=TARGET,
    )
    sensitivity_class_build = build_transactions(
        data, binary_features, include_class=True,
        min_item_support=sensitivity_support_floor, target_col=TARGET,
    )
    sensitivity = _run_sensitivity(
        sensitivity_feature_build, sensitivity_class_build, supports=sensitivity_supports
    )
    toy_sequence = mine_toy_sequential_apriori()

    # Count repeated raw binary signatures independently of the selected item
    # alphabet, so support filtering cannot hide the multiplicities we preserve.
    raw_feature_unique_count = int(data[binary_features].drop_duplicates().shape[0])
    raw_feature_repeated_rows = int(data[binary_features].duplicated(keep=False).sum())
    raw_feature_extra_multiplicity = int(data[binary_features].duplicated().sum())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _csv_safe(apriori["itemsets"]).to_csv(output_dir / "apriori_itemsets.csv", index=False)
    _csv_safe(apriori["rules"]).to_csv(output_dir / "apriori_rules.csv", index=False)
    _csv_safe(class_apriori["class_rules"]).to_csv(output_dir / "apriori_class_rules.csv", index=False)
    _csv_safe(charm["itemsets"]).to_csv(output_dir / "charm_closed_itemsets.csv", index=False)
    _csv_safe(class_charm["itemsets"]).to_csv(
        output_dir / "charm_class_augmented_closed_itemsets.csv", index=False
    )
    _csv_safe(class_charm["class_rules"]).to_csv(output_dir / "charm_class_rules.csv", index=False)
    sensitivity.to_csv(output_dir / "apriori_support_sensitivity.csv", index=False)
    toy_sequence["patterns"].to_csv(output_dir / "toy_sequential_patterns.csv", index=False)

    metadata = {
        "dataset": "UCI Internet Advertisements",
        "dataset_file": "bdHackathon/data/raw/ad.data",
        "source_url": "https://archive.ics.uci.edu/static/public/51/internet+advertisements.zip",
        "rows_as_transactions": feature_build["transaction_count"],
        "transaction_summary": {
            "transaction_count": feature_build["transaction_count"],
            "unique_raw_binary_feature_signatures": raw_feature_unique_count,
            "rows_belonging_to_repeated_raw_feature_signatures": raw_feature_repeated_rows,
            "repeated_raw_feature_rows_beyond_first_occurrence": raw_feature_extra_multiplicity,
            "repeated_rows_retained_as_separate_transactions": True,
        },
        "raw_rows_preserved_with_multiplicity": True,
        "input_features": len(feature_names),
        "numeric_features_excluded_from_transactions": numeric_features,
        "binary_features_available": len(binary_features),
        "selected_binary_features": feature_build["selected_features"],
        "selected_binary_feature_count": len(feature_build["selected_features"]),
        "missing_local_item": DEFAULT_UNKNOWN_LOCAL_ITEM,
        "min_item_support": min_item_support,
        "sensitivity_item_support_floor": sensitivity_support_floor,
        "sensitivity_selected_binary_feature_count": len(
            sensitivity_feature_build["selected_features"]
        ),
        "feature_only_mining": {
            "target_included": False,
            "selected_items": [f"{feature}=1" for feature in feature_build["selected_features"]],
            "item_support": feature_build["item_support"],
            "apriori_parameters": apriori["parameters"],
            "apriori_runtime_seconds_excluding_transaction_build": apriori["elapsed_seconds"],
            "apriori_frequent_itemset_count": len(apriori["itemsets"]),
            "apriori_rule_count_at_min_confidence": len(apriori["rules"]),
            "charm_parameters": charm["parameters"],
            "charm_runtime_seconds_excluding_transaction_build": charm["elapsed_seconds"],
            "charm_closed_itemset_count": len(charm["itemsets"]),
            "charm_search_counters": charm["search_counters"],
            "charm_exhaustive_verification": charm["verification"],
        },
        "class_association_rule_mining": {
            "target_col": TARGET,
            "target_tokens": ["class=ad", "class=nonad"],
            "transactions_include_target_tokens": True,
            "selected_items": [f"{feature}=1" for feature in class_build["selected_features"]]
            + ["class=ad", "class=nonad"],
            "item_support": class_build["item_support"],
            "apriori_parameters": class_apriori["parameters"],
            "apriori_runtime_seconds_excluding_transaction_build": class_apriori["elapsed_seconds"],
            "apriori_itemset_count": len(class_apriori["itemsets"]),
            "apriori_class_rule_count_at_min_confidence": len(class_apriori["class_rules"]),
            "charm_parameters": class_charm["parameters"],
            "charm_runtime_seconds_excluding_transaction_build": class_charm["elapsed_seconds"],
            "charm_class_augmented_closed_itemset_count": len(class_charm["itemsets"]),
            "charm_class_rule_count_at_min_confidence": len(class_charm["class_rules"]),
            "charm_search_counters": class_charm["search_counters"],
            "charm_exhaustive_verification": class_charm["verification"],
        },
        "toy_sequential_demo": {
            "separate_from_dataset_findings": True,
            "sequence_count": len(toy_sequence["sequences"]),
            "sequences": toy_sequence["sequences"],
            "parameters": toy_sequence["parameters"],
            "pattern_count": len(toy_sequence["patterns"]),
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    (output_dir / "association_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary_path = output_dir / "association_results.md"
    top_apriori = apriori["rules"].head(10)
    top_apriori_class = class_apriori["class_rules"].head(10)
    top_charm_class = class_charm["class_rules"].head(10)
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# Association Mining Results\n\n")
        f.write("## Dataset representation\n\n")
        f.write(
            f"The raw file contributes {feature_build['transaction_count']:,} transactions, "
            f"one per row. Repeated feature signatures are retained as separate "
            f"transactions: {raw_feature_repeated_rows:,} rows belong to repeated raw "
            f"binary feature signatures ({raw_feature_extra_multiplicity:,} occurrences "
            f"beyond the first). Transactions use observed binary indicators equal to 1; "
            f"the three continuous geometry fields are excluded rather than binned. "
            f"Primary Apriori and CHARM itemsets/rules use feature-only transactions. "
            f"Class-association rules use a separate copy with `class=ad` or "
            f"`class=nonad` appended. The 15 missing `local` values use an explicit "
            f"`local=unknown` token.\n\n"
        )
        f.write("## Ordinary Apriori: feature-only results\n\n")
        f.write(
            f"Parameters: min_support={min_support:.1%} "
            f"(at least {apriori['parameters']['min_support_count']} transactions), "
            f"min_confidence={min_confidence:.0%}. The binary-item alphabet was "
            f"filtered at min_item_support={min_item_support:.1%}; this removes items "
            f"that cannot appear in any frequent itemset under the chosen support. "
            f"No target token is present in this mining run.\n\n"
        )
        f.write(
            f"Observed: {len(apriori['itemsets']):,} frequent feature itemsets and "
            f"{len(apriori['rules']):,} feature-only rules. Mining time excluding "
            f"transaction construction: {apriori['elapsed_seconds']:.6f} s.\n\n"
        )
        f.write("Top feature-only Apriori rules (sorted by lift, then confidence):\n\n")
        f.write(_markdown_table(top_apriori) if not top_apriori.empty else "No rules met the threshold.")
        f.write("\n\n## Apriori class-association rules\n\n")
        f.write(
            f"A separate Apriori run appends target tokens to the same row-level "
            f"transactions. It found {len(class_apriori['class_rules']):,} rules with "
            f"a single class consequent from {len(class_apriori['itemsets']):,} "
            f"class-augmented frequent itemsets in "
            f"{class_apriori['elapsed_seconds']:.6f} s.\n\n"
        )
        f.write("Top class-consequent Apriori rules (sorted by lift, then confidence):\n\n")
        f.write(
            _markdown_table(top_apriori_class)
            if not top_apriori_class.empty else "No class rules met the threshold."
        )
        f.write("\n\n## CHARM feature-only closed frequent itemsets\n\n")
        f.write(
            f"Observed: {len(charm['itemsets']):,} closed feature itemsets from "
            f"{len(apriori['itemsets']):,} feature-only Apriori frequent itemsets. The "
            f"CHARM TID-set search used {charm['search_counters']['tidset_intersections']:,} "
            f"intersections and took {charm['elapsed_seconds']:.6f} s. Exhaustive "
            f"TID-intersection verification matched: "
            f"{charm['verification']['matches_exhaustive']}.\n\n"
        )
        f.write("## CHARM class-association rules\n\n")
        f.write(
            f"Class rules come from a separate CHARM run over target-augmented "
            f"transactions. It found {len(class_charm['itemsets']):,} class-augmented "
            f"closed itemsets and {len(class_charm['class_rules']):,} closed-itemset "
            f"class rules in {class_charm['elapsed_seconds']:.6f} s. Exhaustive "
            f"TID-intersection verification matched: "
            f"{class_charm['verification']['matches_exhaustive']}.\n\n"
        )
        f.write("Top CHARM closed-itemset class rules:\n\n")
        f.write(
            _markdown_table(top_charm_class)
            if not top_charm_class.empty else "No closed class rules met the threshold."
        )
        f.write("\n\n## Support sensitivity\n\n")
        f.write(
            "Feature-only itemset counts are mined without the target. The class-rule "
            "count at each threshold comes from a separate target-augmented Apriori "
            "run. Both runs use min_confidence=60% and max_len=3; the class-rule count "
            "therefore covers only rules whose union itemset has at most three items. "
            "The primary runs have no maximum length. The feature alphabet is fixed at "
            "the lowest swept support (2%) for all rows in this table.\n\n"
        )
        f.write(_markdown_table(sensitivity))
        f.write("\n\n## Sequential-pattern requirement\n\n")
        f.write(
            "The Internet Ads rows contain no timestamp, session identifier, or event "
            "order. Sequential-pattern mining is therefore not run on this dataset. "
            "The separate `toy_sequential_patterns.csv` is a GSP-style sequential "
            "Apriori demonstration on the explicit synthetic event sequences listed "
            "in `association_metadata.json`; it is not a dataset result and is not "
            "evidence about Internet Ads behavior.\n"
        )

    return {
        "transactions": class_build["transactions"],
        "build": class_build,
        "feature_build": feature_build,
        "class_build": class_build,
        "apriori": apriori,
        "class_apriori": class_apriori,
        "charm": charm,
        "class_charm": class_charm,
        "sensitivity": sensitivity,
        "toy_sequence": toy_sequence,
        "output_dir": output_dir,
        "metadata": metadata,
    }


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "artifacts" / "association"
    result = run_dataset_association(output_dir)
    print(f"Transactions: {result['build']['transaction_count']:,}")
    print(f"Selected binary features: {len(result['build']['selected_features']):,}")
    print(f"Feature-only Apriori frequent itemsets: {len(result['apriori']['itemsets']):,}")
    print(f"Separate Apriori class rules: {len(result['class_apriori']['class_rules']):,}")
    print(f"Feature-only CHARM closed itemsets: {len(result['charm']['itemsets']):,}")
    print(f"CHARM exhaustive verification: {result['charm']['verification']}")
    print(f"Synthetic toy sequential patterns: {len(result['toy_sequence']['patterns']):,}")
    print(f"Outputs: {result['output_dir']}")


if __name__ == "__main__":
    main()
