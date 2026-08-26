# Greedy validation

Standard greedy repeatedly selects the largest deterministic feasible marginal
gain and never chooses more than one candidate per search agent. Lazy greedy
uses upper-bound heap reevaluation with the same candidate-key tie break.
Finite standby enumeration is identical for both.

Across 180 formal E1 instances, standard and lazy solutions were identical in
selected IDs, standby point and objective. All fixed-y and joint-y 1/2 checks
passed. The observed joint greedy/exact ratio had minimum 0.983549219825 and
mean 0.99877892802445. These empirical values validate the implementation on
E1; they do not strengthen the theoretical 1/2 guarantee.
