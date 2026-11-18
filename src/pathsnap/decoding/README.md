# HMM and Viterbi

Candidate states preserve road IDs. Emissions use classifier scores; transitions use partial-arc and directed shortest-path distances. Backpointers recover the sequence. Missing candidates, large/invalid gaps and impossible transitions reset the chain. Valid graph connectors form separate route runs. Settings are in configs/decoder.json; actual search/results are under results/final. This is separate from Part 1 reference alignment.
