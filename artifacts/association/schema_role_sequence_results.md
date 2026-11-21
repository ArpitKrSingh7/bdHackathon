# Dataset-Derived Schema-Role Sequence Analysis

This analysis uses actual Internet Ads rows and observed binary indicators from the `origurl`, `ancurl`, and `url` feature families. It orders these sets as origurl → ancurl → url, following an explicit structural assumption. The ordering is non-temporal; the exact raw URLs are absent.

- Rows/sequences: 3,279
- Frequent items at 5% support: 16
- Frequent schema-role patterns (length ≤ 3): 71
- Prefix-extension rules at 60% confidence: 29
- Runtime: 0.0821 seconds
- Plot: `schema_role_patterns_top20.png`

## Interpretation boundary

These are cross-field structural patterns, not page visits over time. The dataset stores indicator columns, not raw URLs or a chronology. Therefore this analysis does not fulfill a requirement that specifically demands temporal/session sequence mining. The earlier five-sequence synthetic demo remains separate.

Parameters and exact selected items are in `schema_role_sequence_metadata.json`; pattern and prefix-extension tables are in the corresponding CSV files.
