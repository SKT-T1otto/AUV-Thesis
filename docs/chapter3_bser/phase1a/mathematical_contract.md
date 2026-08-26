# BSER mathematical contract

Let the current planner grid be `C`, with normalized target belief `b(c)`. A
search element `e=(i,k,x,path)` binds one search agent to one reachable finite
candidate. Its fixed detection probability is

`p_e(c)=clip(p_max_i exp(-d(c,path)^2/(2 sigma_i^2)) (1-occ(c)),0,1)`.

The role-specific `p_max` values are explicitly declared algorithm-model
parameters; `sigma` is the public sensor radius times the frozen multiplier.
They are not presented as calibrated physical sensor truth and were not tuned
from E1 outcomes.

For executor standby candidate `y`, the read-only path service gives travel
time `T_y(c)`. The response weight is `w_y(c)=exp(-T_y(c)/tau)` when reachable
and zero otherwise. For fixed `y`,

`F(A|y)=sum_c b(c) w_y(c) [1-product_{e in A}(1-p_e(c))]`.

Search-only uses exactly the same terms with `w_y(c)=1`. Travel cost is a
feasibility constraint and is not subtracted from the objective. There are no
negative penalties or linear reward supplements.

Candidates are partitioned as `E=E_0 union E_1 union E_2`; feasible sets obey
`|A intersect E_i| <= 1`. Thus feasibility is a partition matroid and an agent
may remain unassigned.

For one cell, `f_c(A)=1-product(1-p_e(c))` is nonnegative and monotone. Its
marginal gain is `p_e(c) product_{a in A}(1-p_a(c))`, which cannot increase as
`A` grows, proving diminishing returns. Multiplication by nonnegative
`b(c)w_y(c)` and summation preserve nonnegativity, monotonicity and
submodularity. Consequently standard greedy under the partition matroid has the
classical 1/2 guarantee for each fixed `y`. Enumerating the finite standby set
and choosing its best greedy result preserves the 1/2 guarantee relative to the
best finite `(A,y)` pair. These are standard submodular-optimization results;
the contribution is their search–execution coupling in this heterogeneous task.

The conditional response diagnostic is

`T_bar(A,y)=sum_c b(c)P_A(c)T_y(c)/(sum_c b(c)P_A(c)+epsilon)`.

It is a reported metric, not the objective. The guarantees apply only to the
frozen finite candidate set and fixed nonnegative kernels. They do not establish
continuous-space global optimality, learned-policy optimality, online task
success, or robustness after the state changes.
