# Day 13 Graph Interpretation

## What this visualization shows

This image shows the top citation graph nodes selected from graph metrics.
Nodes represent cases, constitutional articles, or statutory sections.
Directed arrows represent citation or mention relationships.

## Important authority nodes

- `ARTICLE_21` | article 21 | type=ARTICLE | in_degree=3 | out_degree=0
- `CASE_REF_ks_puttaswamy_v_union_of_india` | ks puttaswamy v union of india | type=CASE | in_degree=1 | out_degree=0
- `CASE_REF_maneka_gandhi_v_union_of_india` | maneka gandhi v union of india | type=CASE | in_degree=1 | out_degree=0
- `CASE_REF_ak_gopalan_v_state_of_madras` | ak gopalan v state of madras | type=CASE | in_degree=1 | out_degree=0
- `CASE_001` | K.S. Puttaswamy v. Union of India | type=CASE | in_degree=0 | out_degree=2
- `CASE_002` | Maneka Gandhi v. Union of India | type=CASE | in_degree=0 | out_degree=2
- `CASE_003` | A.K. Gopalan v. State of Madras | type=CASE | in_degree=0 | out_degree=2

## Interpretation

Nodes with high in-degree are important because many cases point to them.
In legal research, these are likely to be influential precedents or frequently cited statutes.

## Limitation

This visualization only shows the top 30 nodes, so it is a simplified view of the full citation graph.