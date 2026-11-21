# Blast Radius entry

Picks, for each patch to sqlglot at `eb11aec`, the tests that can fail, by running the affected tests against
the patched code inside one warm Python process.

## How it decides

Build time (`br/build_maps.py`, run by the Dockerfile) runs every test of the pinned commit once and records
facts about the pinned commit only:

- which source lines each test executes, and which each pool call inside a test executes;
- which sqlglot classes each test instantiates and which files it opens;
- which functions run while sqlglot is imported;
- the result of every pool call and the DuckDB answer to every fixture query, keyed by their exact input.

Nothing is keyed by a patch or by a kind of edit.

Run time (`br/main.py`), per patch:

1. Compile the patched file and compare code objects with the pinned ones. No executable difference and no
   test that reads the file: nothing can fail.
2. A change inside function bodies: tests that execute the changed statement are suspects. The new code
   objects are swapped onto the live functions in a forked child and the suspects are run.
3. A change at class or module level, in code that runs at import, or in a test file: the child re-imports
   sqlglot from an overlay tree, compares the imported state with the pinned state (class attributes, module
   globals), and runs the tests that use what changed.
4. A change to a data file: tests that open it are run against the overlay tree.
5. Anything not understood selects every test.

Suspects are run cheapest first with the project's own `unittest` classes (no pytest); process pools are
replaced by an inline pool that pickles arguments and results like a spawned worker. A test is selected when
it fails, times out, or was not verified. Verification stops where including the remaining tests costs less
than running them.

A suspect test is not always run in full:

- A pool call inside a test (one fixture case, one TPC query) reuses the result recorded on the pinned commit
  when its pickled arguments are byte-identical and it executes no changed statement and uses no changed class.
- A top-level `self.validate...(...)` or `self.assert...(...)` statement of a test method is skipped under
  the same condition (`BR_STMT=0` turns this off). Tests whose statements depend on each other are detected at
  build time and never have statements skipped.
- A DuckDB query whose exact text was answered on the pinned commit is not executed again.

One rule is statistical rather than exact: when at least five cheaper tests execute the changed statement and
all pass, `test_execute_tpcds` is taken to pass (`BR_RISK=0` turns this off). On 44 synthetic changes of that
kind the test passed 44 times.

## Layout

- `Dockerfile` — pinned base image, locked packages, sqlglot at the pinned commit, map build.
- `br/common.py` — patch parsing, overlay trees, inline pool, test runner, code-object diff, packing.
- `br/statediff.py` — fingerprints of imported classes and module globals.
- `br/stmts.py` — rewrites test methods into skippable statement units.
- `br/build_maps.py` — build-time tracer.
- `br/main.py` — run-time selector.

## Local run

`make sample` builds the image and scores it on the starter kit's sample commits (`SAMPLE=<dir>` to point at
them).
