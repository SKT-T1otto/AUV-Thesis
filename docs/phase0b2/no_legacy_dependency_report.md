# No-legacy dependency report

Final runtime dependency count is zero. `legacy_adapters` contains one Markdown
file and zero Python files.

Three independent validation methods passed:

1. Worktree method: the entire directory was renamed to
   `legacy_adapters.__disabled__`; core-only E0 passed 60/60 and the complete
   test suite passed 17/17 before the directory name was restored.
2. Tracked-file method: 150 Git-tracked files were copied to an independent
   `C:\tmp` repository while excluding `legacy_adapters`; the complete suite
   passed 17/17 with PYTHONPATH limited to the temporary repository.
3. Archive method: `git archive HEAD` was extracted without `.git`; the
   documentation-only legacy directory was removed. The complete suite passed
   17/17, explicit reset/step and scenario generation passed, bounded training
   smoke passed, and E0 passed 60/60 against the frozen golden manifest.

No test or experiment consulted WORKSPACE/CH3, CH4, CH5, DATA_TO_KEEP,
historical checkpoints, or external business data.
