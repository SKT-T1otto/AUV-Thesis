# Full D2 pipeline source review

Authorization: 2026-09-30 explicit user request for one script completing all
nine training jobs before all ten evaluation jobs. D2 requires no checkpoint.
Formal experiment execution remains manual; only bounded smoke is authorized.

Three closed-listed additions are pipeline.py, pipeline_provenance.py and
run_d2_full_suite.sh. The sole existing protected change is the exact
successor dispatch in safe_search_v1/provenance.py. All earlier manifests,
review bytes, native algorithms and 27 historical records remain unchanged.
All complete byte profiles and exact transitions are verified without exemptions.

Preview performs the existing read-only source, path, dependency and receipt
checks. Explicit execution holds the existing shared Linux lock throughout
training, evaluation and summary, with one unique log/report directory. It
uses native execute(stage=train) for all configured seeds, validates every
training receipt and checkpoint, then calls execute(stage=evaluate) for all
four arms. Even D2 evaluation waits for the training barrier. D2 continues to
pass checkpoint=None to its zero-residual native evaluator; learning models
use their own final checkpoints. No legacy job-interleaved stage=all call is
used. No algorithm, budget, scene, policy-mode or metric changes are made.

Any phase error stops later phases, retains logs/results, records failure and
propagates a nonzero exit. Evaluation receipts are checked before summary;
only a complete four-arm summary can mark the pipeline complete. Repeated
launches revalidate and skip completed jobs, create a new report, and never
overwrite unfinished jobs or automatically resume a checkpoint. Status is
updated between phases; abrupt termination may leave the lock and running
status requiring manual inspection. This is not performance evidence.
