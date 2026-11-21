# Association Mining Results

## Dataset representation

The raw file contributes 3,279 transactions, one per row. Repeated feature signatures are retained as separate transactions. Transactions use observed binary indicators equal to 1; the three continuous geometry fields are excluded rather than binned. For class-association rules only, `class=ad` or `class=nonad` is appended. The 15 missing `local` values use an explicit `local=unknown` token.

## Ordinary Apriori (dataset results)

Parameters: min_support=5.0% (at least 164 transactions), min_confidence=60%. The binary-item alphabet was filtered at min_item_support=5.0%; this removes items that cannot appear in any frequent itemset under the chosen support. The support-sensitivity sweep uses a common feature alphabet selected at 2.0%.

Observed: 2,388 frequent itemsets and 133,556 rules at the confidence threshold; 1,190 have a single class consequent. Mining time excluding transaction construction: 0.041159 s.

Top Apriori class rules (sorted by lift, then confidence):

| antecedent | consequent | antecedent_text | consequent_text | union_itemset | support_count | support | confidence | lift | leverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ancurl*com=1 | class=ad | ancurl*com=1 | class=ad | ancurl*com=1 \| class=ad | 189 | 0.05763952424519671 | 0.9130434782608695 | 6.522591645353794 | 0.0488026190119441 |
| url*images+geoguideii=1 | class=nonad | url*images+geoguideii=1 | class=nonad | class=nonad \| url*images+geoguideii=1 | 189 | 0.05763952424519671 | 1.0 | 1.1627659574468086 | 0.008068478691230646 |
| url*geoguideii=1 | class=nonad | url*geoguideii=1 | class=nonad | class=nonad \| url*geoguideii=1 | 189 | 0.05763952424519671 | 1.0 | 1.1627659574468086 | 0.008068478691230646 |
| url*images+geoguideii=1 \| url*images=1 | class=nonad | url*images+geoguideii=1 AND url*images=1 | class=nonad | class=nonad \| url*images+geoguideii=1 \| url*images=1 | 189 | 0.05763952424519671 | 1.0 | 1.1627659574468086 | 0.008068478691230646 |
| url*images+geoguideii=1 \| url*www.geocities.com=1 | class=nonad | url*images+geoguideii=1 AND url*www.geocities.com=1 | class=nonad | class=nonad \| url*images+geoguideii=1 \| url*www.geocities.com=1 | 189 | 0.05763952424519671 | 1.0 | 1.1627659574468086 | 0.008068478691230646 |
| url*geoguideii=1 \| url*images+geoguideii=1 | class=nonad | url*geoguideii=1 AND url*images+geoguideii=1 | class=nonad | class=nonad \| url*geoguideii=1 \| url*images+geoguideii=1 | 189 | 0.05763952424519671 | 1.0 | 1.1627659574468086 | 0.008068478691230646 |
| url*geoguideii=1 \| url*images=1 | class=nonad | url*geoguideii=1 AND url*images=1 | class=nonad | class=nonad \| url*geoguideii=1 \| url*images=1 | 189 | 0.05763952424519671 | 1.0 | 1.1627659574468086 | 0.008068478691230646 |
| origurl*geocities.com=1 \| url*images+geoguideii=1 | class=nonad | origurl*geocities.com=1 AND url*images+geoguideii=1 | class=nonad | class=nonad \| origurl*geocities.com=1 \| url*images+geoguideii=1 | 189 | 0.05763952424519671 | 1.0 | 1.1627659574468086 | 0.008068478691230646 |
| url*geocities.com=1 \| url*images+geoguideii=1 | class=nonad | url*geocities.com=1 AND url*images+geoguideii=1 | class=nonad | class=nonad \| url*geocities.com=1 \| url*images+geoguideii=1 | 189 | 0.05763952424519671 | 1.0 | 1.1627659574468086 | 0.008068478691230646 |
| origurl*geocities.com=1 \| url*geoguideii=1 | class=nonad | origurl*geocities.com=1 AND url*geoguideii=1 | class=nonad | class=nonad \| origurl*geocities.com=1 \| url*geoguideii=1 | 189 | 0.05763952424519671 | 1.0 | 1.1627659574468086 | 0.008068478691230646 |

## CHARM closed frequent itemsets (dataset results)

Observed: 80 closed frequent itemsets from 2,388 Apriori frequent itemsets. The CHARM TID-set search used 268 intersections and took 0.000511 s. Exhaustive TID-intersection verification matched: True.

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

Each sensitivity run uses min_confidence=60% and max_len=3; its class-rule count therefore covers only rules whose union itemset has at most three items. The primary 5% Apriori run has no maximum length, so its larger class-rule count is not directly comparable to this capped sensitivity column. The item alphabet is fixed at the lowest swept support (2%) for all rows in this table.

| min_support | minimum_count | frequent_singletons | frequent_pairs | frequent_triples | frequent_itemsets_through_length_3 | class_rules_at_confidence_0_60_max_len_3 | runtime_seconds | max_len |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.02 | 66 | 89 | 541 | 2021 | 2651 | 531 | 0.018235206000099424 | 3 |
| 0.03 | 99 | 33 | 160 | 460 | 653 | 143 | 0.0024685850003152154 | 3 |
| 0.05 | 164 | 20 | 78 | 212 | 310 | 75 | 0.0009366650001538801 | 3 |
| 0.08 | 263 | 12 | 36 | 65 | 113 | 33 | 0.00028302300052018836 | 3 |

## Sequential-pattern requirement

The Internet Ads rows contain no timestamp, session identifier, or event order. Sequential-pattern mining is therefore not run on this dataset. The separate `toy_sequential_patterns.csv` is a GSP-style sequential Apriori demonstration on the explicit synthetic event sequences listed in `association_metadata.json`; it is not a dataset result and is not evidence about Internet Ads behavior.
