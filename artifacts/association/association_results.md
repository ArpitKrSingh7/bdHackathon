# Association Mining Results

## Dataset representation

The raw file contributes 3,279 transactions, one per row. Repeated feature signatures are retained as separate transactions: 1,659 rows belong to repeated raw binary feature signatures (1,303 occurrences beyond the first). Transactions use observed binary indicators equal to 1; the three continuous geometry fields are excluded rather than binned. Primary Apriori and CHARM itemsets/rules use feature-only transactions. Class-association rules use a separate copy with `class=ad` or `class=nonad` appended. The 15 missing `local` values use an explicit `local=unknown` token.

## Ordinary Apriori: feature-only results

Parameters: min_support=5.0% (at least 164 transactions), min_confidence=60%. The binary-item alphabet was filtered at min_item_support=5.0%; this removes items that cannot appear in any frequent itemset under the chosen support. No target token is present in this mining run.

Observed: 1,194 frequent feature itemsets and 44,122 feature-only rules. Mining time excluding transaction construction: 0.011651 s.

Top feature-only Apriori rules (sorted by lift, then confidence):

| antecedent | consequent | antecedent_text | consequent_text | union_itemset | support_count | support | confidence | lift | leverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| url*geoguideii=1 | url*images+geoguideii=1 | url*geoguideii=1 | url*images+geoguideii=1 | url*geoguideii=1 \| url*images+geoguideii=1 | 189 | 0.05763952424519671 | 1.0 | 17.349206349206348 | 0.054317209489984086 |
| url*images+geoguideii=1 | url*geoguideii=1 | url*images+geoguideii=1 | url*geoguideii=1 | url*geoguideii=1 \| url*images+geoguideii=1 | 189 | 0.05763952424519671 | 1.0 | 17.349206349206348 | 0.054317209489984086 |
| url*geoguideii=1 | url*images+geoguideii=1 \| url*www.geocities.com=1 | url*geoguideii=1 | url*images+geoguideii=1 AND url*www.geocities.com=1 | url*geoguideii=1 \| url*images+geoguideii=1 \| url*www.geocities.com=1 | 189 | 0.05763952424519671 | 1.0 | 17.349206349206348 | 0.054317209489984086 |
| url*images+geoguideii=1 | url*geoguideii=1 \| url*www.geocities.com=1 | url*images+geoguideii=1 | url*geoguideii=1 AND url*www.geocities.com=1 | url*geoguideii=1 \| url*images+geoguideii=1 \| url*www.geocities.com=1 | 189 | 0.05763952424519671 | 1.0 | 17.349206349206348 | 0.054317209489984086 |
| url*geoguideii=1 \| url*www.geocities.com=1 | url*images+geoguideii=1 | url*geoguideii=1 AND url*www.geocities.com=1 | url*images+geoguideii=1 | url*geoguideii=1 \| url*images+geoguideii=1 \| url*www.geocities.com=1 | 189 | 0.05763952424519671 | 1.0 | 17.349206349206348 | 0.054317209489984086 |
| url*images+geoguideii=1 \| url*www.geocities.com=1 | url*geoguideii=1 | url*images+geoguideii=1 AND url*www.geocities.com=1 | url*geoguideii=1 | url*geoguideii=1 \| url*images+geoguideii=1 \| url*www.geocities.com=1 | 189 | 0.05763952424519671 | 1.0 | 17.349206349206348 | 0.054317209489984086 |
| url*geoguideii=1 | origurl*geocities.com=1 \| url*images+geoguideii=1 | url*geoguideii=1 | origurl*geocities.com=1 AND url*images+geoguideii=1 | origurl*geocities.com=1 \| url*geoguideii=1 \| url*images+geoguideii=1 | 189 | 0.05763952424519671 | 1.0 | 17.349206349206348 | 0.054317209489984086 |
| url*images+geoguideii=1 | origurl*geocities.com=1 \| url*geoguideii=1 | url*images+geoguideii=1 | origurl*geocities.com=1 AND url*geoguideii=1 | origurl*geocities.com=1 \| url*geoguideii=1 \| url*images+geoguideii=1 | 189 | 0.05763952424519671 | 1.0 | 17.349206349206348 | 0.054317209489984086 |
| origurl*geocities.com=1 \| url*geoguideii=1 | url*images+geoguideii=1 | origurl*geocities.com=1 AND url*geoguideii=1 | url*images+geoguideii=1 | origurl*geocities.com=1 \| url*geoguideii=1 \| url*images+geoguideii=1 | 189 | 0.05763952424519671 | 1.0 | 17.349206349206348 | 0.054317209489984086 |
| origurl*geocities.com=1 \| url*images+geoguideii=1 | url*geoguideii=1 | origurl*geocities.com=1 AND url*images+geoguideii=1 | url*geoguideii=1 | origurl*geocities.com=1 \| url*geoguideii=1 \| url*images+geoguideii=1 | 189 | 0.05763952424519671 | 1.0 | 17.349206349206348 | 0.054317209489984086 |

## Apriori class-association rules

A separate Apriori run appends target tokens to the same row-level transactions. It found 1,190 rules with a single class consequent from 2,388 class-augmented frequent itemsets in 0.037412 s.

Top class-consequent Apriori rules (sorted by lift, then confidence):

| antecedent | consequent | antecedent_text | consequent_text | union_itemset | support_count | support | confidence | lift | leverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ancurl*com=1 | class=ad | ancurl*com=1 | class=ad | ancurl*com=1 \| class=ad | 189 | 0.05763952424519671 | 0.9130434782608695 | 6.522591645353794 | 0.0488026190119441 |
| url*images+geoguideii=1 | class=nonad | url*images+geoguideii=1 | class=nonad | class=nonad \| url*images+geoguideii=1 | 189 | 0.05763952424519671 | 1.0 | 1.1627659574468086 | 0.008068478691230646 |
| url*geoguideii=1 | class=nonad | url*geoguideii=1 | class=nonad | class=nonad \| url*geoguideii=1 | 189 | 0.05763952424519671 | 1.0 | 1.1627659574468086 | 0.008068478691230646 |
| url*geocities.com=1 \| url*geoguideii=1 | class=nonad | url*geocities.com=1 AND url*geoguideii=1 | class=nonad | class=nonad \| url*geocities.com=1 \| url*geoguideii=1 | 189 | 0.05763952424519671 | 1.0 | 1.1627659574468086 | 0.008068478691230646 |
| url*geoguideii=1 \| url*images+geoguideii=1 | class=nonad | url*geoguideii=1 AND url*images+geoguideii=1 | class=nonad | class=nonad \| url*geoguideii=1 \| url*images+geoguideii=1 | 189 | 0.05763952424519671 | 1.0 | 1.1627659574468086 | 0.008068478691230646 |
| origurl*www.geocities.com=1 \| url*images+geoguideii=1 | class=nonad | origurl*www.geocities.com=1 AND url*images+geoguideii=1 | class=nonad | class=nonad \| origurl*www.geocities.com=1 \| url*images+geoguideii=1 | 189 | 0.05763952424519671 | 1.0 | 1.1627659574468086 | 0.008068478691230646 |
| url*geoguideii=1 \| url*www.geocities.com=1 | class=nonad | url*geoguideii=1 AND url*www.geocities.com=1 | class=nonad | class=nonad \| url*geoguideii=1 \| url*www.geocities.com=1 | 189 | 0.05763952424519671 | 1.0 | 1.1627659574468086 | 0.008068478691230646 |
| url*geoguideii=1 \| url*images=1 | class=nonad | url*geoguideii=1 AND url*images=1 | class=nonad | class=nonad \| url*geoguideii=1 \| url*images=1 | 189 | 0.05763952424519671 | 1.0 | 1.1627659574468086 | 0.008068478691230646 |
| ancurl*geocities.com=1 \| url*images+geoguideii=1 | class=nonad | ancurl*geocities.com=1 AND url*images+geoguideii=1 | class=nonad | ancurl*geocities.com=1 \| class=nonad \| url*images+geoguideii=1 | 189 | 0.05763952424519671 | 1.0 | 1.1627659574468086 | 0.008068478691230646 |
| local=1 \| url*images+geoguideii=1 | class=nonad | local=1 AND url*images+geoguideii=1 | class=nonad | class=nonad \| local=1 \| url*images+geoguideii=1 | 189 | 0.05763952424519671 | 1.0 | 1.1627659574468086 | 0.008068478691230646 |

## CHARM feature-only closed frequent itemsets

Observed: 41 closed feature itemsets from 1,194 feature-only Apriori frequent itemsets. The CHARM TID-set search used 190 intersections and took 0.000264 s. Exhaustive TID-intersection verification matched: True.

## CHARM class-association rules

Class rules come from a separate CHARM run over target-augmented transactions. It found 80 class-augmented closed itemsets and 37 closed-itemset class rules in 0.000492 s. Exhaustive TID-intersection verification matched: True.

Top CHARM closed-itemset class rules:

| antecedent | consequent | antecedent_text | consequent_text | closed_itemset | support_count | support | confidence | lift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ancurl*com=1 | class=ad | ancurl*com=1 | class=ad | ancurl*com=1 \| class=ad | 189 | 0.05763952424519671 | 0.9130434782608695 | 6.522591645353794 |
| ancurl*geocities.com=1 \| local=1 \| origurl*geocities.com=1 \| origurl*www.geocities.com=1 \| url*geocities.com=1 \| url*geoguideii=1 \| url*images+geoguideii=1 \| url*images=1 \| url*www.geocities.com=1 | class=nonad | ancurl*geocities.com=1 AND local=1 AND origurl*geocities.com=1 AND origurl*www.geocities.com=1 AND url*geocities.com=1 AND url*geoguideii=1 AND url*images+geoguideii=1 AND url*images=1 AND url*www.geocities.com=1 | class=nonad | ancurl*geocities.com=1 \| class=nonad \| local=1 \| origurl*geocities.com=1 \| origurl*www.geocities.com=1 \| url*geocities.com=1 \| url*geoguideii=1 \| url*images+geoguideii=1 \| url*images=1 \| url*www.geocities.com=1 | 189 | 0.05763952424519671 | 1.0 | 1.1627659574468086 |
| ancurl*geocities.com=1 \| ancurl*www.geocities.com=1 \| local=1 \| origurl*geocities.com=1 \| origurl*www.geocities.com=1 \| url*geocities.com=1 \| url*geoguideii=1 \| url*images+geoguideii=1 \| url*images=1 \| url*www.geocities.com=1 | class=nonad | ancurl*geocities.com=1 AND ancurl*www.geocities.com=1 AND local=1 AND origurl*geocities.com=1 AND origurl*www.geocities.com=1 AND url*geocities.com=1 AND url*geoguideii=1 AND url*images+geoguideii=1 AND url*images=1 AND url*www.geocities.com=1 | class=nonad | ancurl*geocities.com=1 \| ancurl*www.geocities.com=1 \| class=nonad \| local=1 \| origurl*geocities.com=1 \| origurl*www.geocities.com=1 \| url*geocities.com=1 \| url*geoguideii=1 \| url*images+geoguideii=1 \| url*images=1 \| url*www.geocities.com=1 | 170 | 0.0518450747179018 | 1.0 | 1.1627659574468086 |
| origurl*geocities.com=1 \| origurl*www.geocities.com=1 \| url*images=1 | class=nonad | origurl*geocities.com=1 AND origurl*www.geocities.com=1 AND url*images=1 | class=nonad | class=nonad \| origurl*geocities.com=1 \| origurl*www.geocities.com=1 \| url*images=1 | 196 | 0.05977432143946325 | 0.98989898989899 | 1.1510208467655276 |
| local=1 \| origurl*geocities.com=1 \| origurl*www.geocities.com=1 \| url*images=1 | class=nonad | local=1 AND origurl*geocities.com=1 AND origurl*www.geocities.com=1 AND url*images=1 | class=nonad | class=nonad \| local=1 \| origurl*geocities.com=1 \| origurl*www.geocities.com=1 \| url*images=1 | 194 | 0.05916437938395852 | 0.9897959183673469 | 1.1509009986973513 |
| ancurl*geocities.com=1 \| url*images=1 | class=nonad | ancurl*geocities.com=1 AND url*images=1 | class=nonad | ancurl*geocities.com=1 \| class=nonad \| url*images=1 | 193 | 0.058859408356206164 | 0.9897435897435898 | 1.1508401527550465 |
| origurl*geocities.com=1 \| origurl*www.geocities.com=1 \| url*geocities.com=1 \| url*images=1 \| url*www.geocities.com=1 | class=nonad | origurl*geocities.com=1 AND origurl*www.geocities.com=1 AND url*geocities.com=1 AND url*images=1 AND url*www.geocities.com=1 | class=nonad | class=nonad \| origurl*geocities.com=1 \| origurl*www.geocities.com=1 \| url*geocities.com=1 \| url*images=1 \| url*www.geocities.com=1 | 191 | 0.058249466300701436 | 0.9896373056994818 | 1.150716569286738 |
| ancurl*geocities.com=1 \| local=1 \| origurl*geocities.com=1 \| origurl*www.geocities.com=1 \| url*geocities.com=1 \| url*images=1 \| url*www.geocities.com=1 | class=nonad | ancurl*geocities.com=1 AND local=1 AND origurl*geocities.com=1 AND origurl*www.geocities.com=1 AND url*geocities.com=1 AND url*images=1 AND url*www.geocities.com=1 | class=nonad | ancurl*geocities.com=1 \| class=nonad \| local=1 \| origurl*geocities.com=1 \| origurl*www.geocities.com=1 \| url*geocities.com=1 \| url*images=1 \| url*www.geocities.com=1 | 190 | 0.05794449527294907 | 0.9895833333333334 | 1.1506538120567378 |
| ancurl*geocities.com=1 \| ancurl*www.geocities.com=1 \| url*images=1 | class=nonad | ancurl*geocities.com=1 AND ancurl*www.geocities.com=1 AND url*images=1 | class=nonad | ancurl*geocities.com=1 \| ancurl*www.geocities.com=1 \| class=nonad \| url*images=1 | 174 | 0.053064958828911254 | 0.9886363636363636 | 1.1495527079303676 |
| ancurl*geocities.com=1 \| ancurl*www.geocities.com=1 \| local=1 \| origurl*geocities.com=1 \| origurl*www.geocities.com=1 \| url*geocities.com=1 \| url*images=1 \| url*www.geocities.com=1 | class=nonad | ancurl*geocities.com=1 AND ancurl*www.geocities.com=1 AND local=1 AND origurl*geocities.com=1 AND origurl*www.geocities.com=1 AND url*geocities.com=1 AND url*images=1 AND url*www.geocities.com=1 | class=nonad | ancurl*geocities.com=1 \| ancurl*www.geocities.com=1 \| class=nonad \| local=1 \| origurl*geocities.com=1 \| origurl*www.geocities.com=1 \| url*geocities.com=1 \| url*images=1 \| url*www.geocities.com=1 | 171 | 0.05215004574565416 | 0.9884393063583815 | 1.149323576435863 |

## Support sensitivity

Feature-only itemset counts are mined without the target. The class-rule count at each threshold comes from a separate target-augmented Apriori run. Both runs use min_confidence=60% and max_len=3; the class-rule count therefore covers only rules whose union itemset has at most three items. The primary runs have no maximum length. The feature alphabet is fixed at the lowest swept support (2%) for all rows in this table.

| min_support | minimum_count | feature_frequent_singletons | feature_frequent_pairs | feature_frequent_triples | feature_itemsets_through_length_3 | class_rules_at_confidence_0_60_max_len_3 | feature_only_runtime_seconds | class_augmented_runtime_seconds | max_len |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.02 | 66 | 87 | 456 | 1568 | 2111 | 531 | 0.024616935000267404 | 0.01721194200035825 | 3 |
| 0.03 | 99 | 31 | 127 | 343 | 501 | 143 | 0.0017876999991131015 | 0.002362746000471816 | 3 |
| 0.05 | 164 | 18 | 61 | 152 | 231 | 75 | 0.0006834969999545137 | 0.0009390010000061011 | 3 |
| 0.08 | 263 | 10 | 26 | 41 | 77 | 33 | 0.00019793299998127623 | 0.00030487699950754177 | 3 |

## Sequential-pattern requirement

The Internet Ads rows contain no timestamp, session identifier, or event order, so temporal sequential mining is not available from this dataset. The separate `toy_sequential_patterns.csv` is a GSP-style sequential Apriori demonstration on explicit synthetic event sequences listed in `association_metadata.json`; it is not a dataset result. A later, separate analysis in `schema_role_sequence_results.md` mines actual URL-family indicator rows under the analyst-defined `origurl → ancurl → url` order. Those are non-temporal structural patterns, not temporal event sequences; instructor acceptance is still needed before treating them as satisfying the sequential-pattern rubric.
