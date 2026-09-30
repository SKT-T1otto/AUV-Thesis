# D2 integration source review — 2026-09-30

Authorization: the user explicitly requested freezing D2, retiring original BSER from the future mainline, and basing B2/B3/HGR on D2. Formal experiments remain manual.

Scope is Chapter 3 assembly, planner selection, default config routing and additive provenance. No core, model, replay, reward, action/observation, dynamics, target, obstacle, success or handoff contract changes. Original D2 and V4 algorithm modules remain byte-identical to the sealed F0 reference.

The wrapper delegates the original search controller. Its only override avoids recursive attribute lookup when HGR reconstructs empty objects during snapshot restore. Owners, allocators, planning views and bridge remain in the snapshot graph. Learned runtimes retain their own policy action calls; no zero-action experimental runtime is used.

Ten changed paths are explicitly enumerated in the D2 provenance module and sealed with parent hashes. Ten additions comprise six Chapter 3 modules and four configs. Every other file in all three complete parent profiles remains pinned. Migration, collision, HGR, baseline, V, R, D and F manifests are not rewritten. Changed dependencies, review/reference files, extra source files or mixed profiles fail verification.

B0/B1 keep historical configs and identities. Current B2/B3 defaults require D2 and reject planner-mismatched checkpoints; current HGR defaults use D2. Explicit historical configs retain legacy behavior. Full config/source hashes continue to bind checkpoints. New output roots preserve prior experiments.

Historical D2 evidence establishes only the recorded 20-scene development result. Integration smoke does not establish B2/B3 or HGR improvement, generalization or thesis-level novelty. No formal run is launched. Latest recorded local test outcomes are written separately in local_verification.json, never as historical evidence or CI.

Historical result JSON references have two exact complete byte profiles: original Windows and Git LF. Runtime hashes raw bytes and rejects mixed references; it never normalizes source or evidence. New D2 seal metadata itself is written as LF.

Git attributes additionally preserve the exact two historical SafeSearch v1/v2 CRLF manifest files. Their parent hashes are not changed. This closes the previously untested metadata conversion gap when cloning to Linux.

The attributes file itself explicitly uses LF, preventing Windows autocrlf from changing this source-inventory input on checkout.
