# Cross-repository environment contract diff

These values are frozen from the same CPU smoke protocol (seed 1729, reset once, then three zero-action steps). The hashes cover the live state fields listed in each JSON contract.

| Contract item | CH3 | CH4 | CH5 |
|---|---|---|---|
| Agents / role order | `4 / search_fast, search_balanced, search_precise, executor` | `4 / identical` | `4 / identical` |
| Observation dims | `[28, 28, 28, 28]` | `[139, 139, 139, 139]` | `[139, 139, 139, 139]` |
| Observation contract | `28-D local block` | `28-D local + 3×37-D neighbor blocks` | `28-D local + 3×37-D acoustic/semantic neighbor blocks` |
| Action dims | `[3, 3, 3, 3]` | `[3, 3, 3, 3]` | `[3, 3, 3, 3]` |
| Reward vector shape | `[4]` | `[4]` | `[4]` |
| Target motion default | `static` | `static` | `static` |
| Obstacle default | `False` | `False` | `False` |
| Disturbance default | `nominal/no CH4 protocol` | `False` | `False` |
| Communication | `ch3_fixed_reliable` | `Chapter-4 basic dynamic communication state` | `acoustic/dynamic communication state` |
| Reset hash | `b25690c82e47af54d8c1c3618e3306f2d3e5dc0ac3bcf1be5d49a937750bde86` | `ce85f251aefa747adfea42eb3d92e706d2ce583ed0a3f0747742eedad8f42e20` | `041889d496d7c59d5cd07a1f330fb82960fd9000b52b1b32e973b02bc2495a0d` |
| Three-step hash | `b96cf688ee10b3508d8272e53e452bcf1dde6afe9fc50f37a491163c432ee8e7` | `39b8bccc739a5d860851d5d19342e658fe8136774c94b87b175c72df97a0de80` | `adc14d136d9143f38a202f6edd4dc9be30cd132a468a139689261a96a6a8fa02` |

## Shared mission semantics

All repositories preserve four ordered roles, a three-dimensional residual-acceleration action, per-agent rewards, searcher detection (`task_found`), executor knowledge handoff, and executor capture/hold completion. Exact definitions are recorded in the JSON contracts.

## Material differences

CH3 observes only the 28-D local mission block and uses fixed reliable one-step handoff. CH4/CH5 append three 37-D neighbor communication blocks. CH4 retains plant-disturbance machinery but rejects removed Chapter-5 semantic/robust switches; CH5 contains the acoustic loss/delay/bandwidth/TTL queues and semantic communication state. Default construction is nominal; formal experiment configuration remains separately locked by its retained protocol/evidence.

A hash difference across repositories is expected because the state schemas and communication state differ. A future change within the same repository is a contract-review trigger, not an automatic approval to update the fixture.
