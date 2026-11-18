# Candidate generation

`build_segments(nodes, arcs)` joins node coordinates by endpoint ID and preserves each `arc_id`, `from_node_id`, and `to_node_id`. Geometry is a straight segment in the trip's supplied metre CRS.

`generate_candidates(observations, segments, CandidateConfig(...))` indexes segment midpoints with length-stratified SciPy KD-trees. Each tree query radius is expanded by half the longest segment in that stratum, so nearby long segments are considered without letting an outlier inflate every query; exact point-to-segment distances then enforce the requested radius. If the primary radius finds no segment for an observation, it retries at `expanded_search_radius_m` and marks rows `search_expanded=True`. Candidate count is capped after nearest-first sorting. If neither radius finds a segment, that observation has no candidate rows and remains countable in coverage summaries. No reference route or label is accepted by this API.

Default radii are 100 m and 300 m, with at most 8 candidates. Tune on permitted training trips based on candidate coverage and computation cost. The midpoint index can be memory-heavy for exceptionally long segments because its conservative query bound uses the longest segment in the network.
