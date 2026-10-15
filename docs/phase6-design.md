# Phase 6 Design Note — Parallel Runner, Result Reproduction, CI

Status: **approved 2026-10-14 (user: "go"); implemented**

## 0. What Phase 6 adds

Turning the model into a tool that keeps its own results honest:

1. **A parallel runner** (`python/runner.py`) used by every sweep and check
   script. Its output is **byte-identical whatever the worker count**.
2. **Result reproduction** (`python/reproduce.py`). It regenerates every
   committed results CSV of Phases 2–5 at HEAD and byte-compares each one
   with the committed file. A change that silently moves any published
   number then fails the regression.
3. **CI** (`.github/workflows/regression.yml`). On every push it runs both
   builds, the whole test suite, the RESULTS.md number check and the
   reproduction. Nightly it runs everything under ASan, including the RTL,
   plus all five mutation checks.
4. **Measured regression runtime**, reported two ways: the GitHub job and
   the Codespace. They are never mixed.

No model changes. Phase 6 must not move a single published number, and item
2 is the proof.

## 1. Parallel runner: `python/runner.py` (standard library only)

```python
run_tasks(tasks, fn, jobs) -> list of (task, result | exception)   # in task order
```

- **Pool:** `concurrent.futures.ProcessPoolExecutor` with `jobs` workers
  (default `os.cpu_count()`). `jobs = 1` runs in-process, sequentially, with
  no pool, which is the reference behaviour.
- **Order:** results are returned **in task order, never completion order**.
  Every script builds its CSV from that list, so worker count and scheduling
  cannot change a byte.
- **Isolation:** each task gets its own temporary directory. Today
  `gemm_check.py` shares one directory across cases; that becomes per task.
- **Failures:** a task's exception is captured and reported in task order.
  The script fails at the end and lists every failed task, as the sweeps do
  today. A failure never hides the others.
- **Adopted by:**
  - `gemm_check.py`, `attn_check.py` and `correlate.py` (cases);
  - `sweep_p3.py` (configs), `study_p4.py` (programs), `sweep_p5.py`
    (programs).

  Each gains `--jobs N`. The committed exit scripts keep calling them
  without `--jobs`, so they now run in parallel. Their outputs are proven
  unchanged by §2.
- **Self-test** (`runner.py --selftest`, a CTest): 8 tasks that sleep in
  *reverse* order of their index (task 0 the longest). Completion order is
  therefore reversed, and the result order must still be 0…7. A runner that
  returned completion order would fail deterministically, not by chance.

## 2. Result reproduction: `python/reproduce.py`

Regenerates, in a temporary directory, each committed results CSV with the
exact command its phase exit script used, then byte-compares:

| Committed file | Command (as in the exit script) |
|---|---|
| `results/phase2/functional.csv` | `gemm_check.py --cases 200` (PE) |
| `results/phase3/functional_accel.csv` | `gemm_check.py --target accel --cases 100` |
| `results/phase3/sweep.csv` | `sweep_p3.py` |
| `results/phase4/functional_gemm.csv` | `gemm_check.py --target accel --prec all --cases 200` |
| `results/phase4/functional_attn.csv` | `attn_check.py --cases 80` |
| `results/phase4/study.csv` | `study_p4.py` |
| `results/phase5/correlation.csv` | `correlate.py` |
| `results/phase5/sweep.csv` | `sweep_p5.py` |

- **Not compared:** text reports (they hold provenance, dates and wall
  times) and PNGs (rendering may differ between machines; the CSV is the
  result, the PNG is a picture of it).
- **Output:** one line per file: `IDENTICAL`, or `DIFFERS` with the first
  differing line, both versions and a count of differing lines. Exit status
  0 only if all 8 are identical.
- **Why this is the strongest regression check available:**
  - The Phase 2/3 CSVs were generated at older commits, before Phases 4–5
    generalized the PE, the layout, the scheduler and the DRAM map.
  - Every hand-derived test already passes unchanged. Reproduction extends
    that guarantee from a few dozen directed cases to all 1,000+ published
    rows.
- **If a committed CSV does not reproduce,** I stop and report it. Either a
  later phase changed a published number without noticing (a model bug or
  an undocumented behaviour change), or a script changed its output format.
  The cause gets root-caused and logged in BUGS.md. Nothing is regenerated
  to make the diff go away without your decision.

## 3. CI: `.github/workflows/regression.yml`

**Runner:** `ubuntu-24.04`, the same distro as the Codespace. The toolchain
is installed by the same `.devcontainer/setup.sh`, so the versions are
identical; `env_report.sh` prints them in the log. The repository is
public, so Actions minutes are free and the standard runner has 4 vCPUs.

| Trigger | Job | Steps |
|---|---|---|
| push, pull request | `regression` | 1. install toolchain (SystemC cached between runs when possible, see below) 2. **release** build with RTL; full CTest (unit, sim, functional, determinism, RTL lint, hand traces, quick correlation) 3. **ASan** build **without RTL** (`-DAPM_RTL=OFF`); full CTest 4. `scripts/check_results.sh` (every number in RESULTS.md) 5. `python/reproduce.py` (all 8 CSVs) |
| nightly (cron), manual | `full` | as above, but ASan **with** RTL, then `scripts/phase{1..5}_mutation_check.sh`. Their reports go to the job artefacts. They write into `results/` in the CI checkout only and are never committed. |

- **Why the RTL is excluded from the ASan build on push:** the 14 Verilated
  variants under ASan took about 25 min to compile in the Codespace (Phase 5
  exit run). The release build still runs every RTL test on every push, and
  the nightly job covers the RTL under ASan.
- **SystemC cache:** `actions/cache` keyed on the pinned SystemC commit,
  restoring `/opt/systemc-3.0.2`. `setup.sh` already skips the build when
  the installed commit matches. If restoring into `/opt` is not permitted on
  the runner, the job builds SystemC every time, and the measured runtime
  says which happened.
- **Artefacts on failure:** CTest logs, the reproduction diff, and the
  correlation output.
- **Timing summary:** every step is timed by the script it calls and
  written to the job summary.

## 4. Measuring "total regression runtime"

Two numbers, labelled separately, never mixed:

1. **GitHub Actions:** wall time of the `regression` job on the Phase 6
   push, and of one manually dispatched `full` job, with per-step times and
   the runner type. `scripts/phase6_ci_times.sh RUN_ID` reads them from the
   GitHub API (`gh run view --json`) and writes them, with provenance, to
   `results/phase6/ci_runtime.txt`.
2. **Codespace (2 cores):** `scripts/phase6_regression.sh` runs exactly the
   push-job steps locally, with `date +%s` around each step. It writes a
   step-by-step table to `results/phase6/regression_runtime.txt`.

Also measured: the parallel runner's speedup. In the Codespace,
`study_p4.py` and `sweep_p5.py` run with `--jobs 1` and `--jobs 2`; the
wall times and the byte-identity of the outputs go to
`results/phase6/runner_speedup.txt`.

## 5. Tests written before the tools

- **`runner.py --selftest`:** the reverse-sleep ordering case (§1), an
  exception in one task reported in order without hiding the others, and
  `jobs = 1` vs `jobs = 3` giving identical results.
- **Determinism through real scripts (CTest):** `attn_check.py --cases 6`
  and `gemm_check.py --target accel --prec all --cases 8`, each run with
  `--jobs 1` and `--jobs 3`. The CSVs must be byte-identical.
- **`reproduce.py --selftest`:** given a committed CSV and a copy with one
  digit changed, it must report `IDENTICAL` and `DIFFERS` respectively,
  naming the right line.

## 6. Exit criterion

1. All tests pass in both presets, including the new runner and
   determinism tests.
2. **`reproduce.py`: all 8 committed CSVs are byte-identical at HEAD,** in
   the Codespace and in CI. If one is not, §2 applies.
3. **CI is green** on the Phase 6 push, and a manually dispatched `full`
   run is green, with all five mutation checks reporting "all mutants
   caught".
4. **Runtimes measured both ways (§4),** plus the runner speedup, all in
   `results/phase6/` with provenance.
5. The README gets a final "how to reproduce everything" section and a
   short summary of what each phase found, every number from RESULTS.md.

## 7. Mutation checks (scripts/phase6_mutation_check.sh)

| Mutant | Change | Must be caught by |
|---|---|---|
| P1 | The runner returns results in completion order | `runner.py --selftest` (reverse-sleep case) |
| P2 | The runner drops a failed task's exception (returns None) | `runner.py --selftest` |
| P3 | `reproduce.py` compares only line counts | `reproduce.py --selftest` |
| P4 | `gemm_check.py` shares one temp dir between parallel cases again | determinism CTest (`--jobs 3`) |

## 8. Hypotheses (before measuring)

- **H1:** all 8 committed CSVs reproduce byte-identically at HEAD, with
  `--jobs 1` and with the default worker count. I am fairly but not fully
  sure about `results/phase2/functional.csv`: it was produced before the
  line-ending fix (BUGS #6), and its committed copy was normalized to LF by
  git, which should equal today's output.
- **H2:** in the Codespace (2 cores), `--jobs 2` makes `study_p4.py` and
  `sweep_p5.py` each **≥ 1.6×** faster than `--jobs 1`. Their programs are
  independent processes. The limit is the longest single program, a few
  seconds each.
- **H3:** the push job takes **≤ 20 min** wall time on the GitHub runner.
  My estimate is about 12–15 min with an uncached SystemC build (about
  3–5 min).
- **H4:** the nightly `full` job (ASan with RTL plus five mutation checks)
  takes **≤ 120 min**.

## 9. Decisions to confirm

1. One process-pool runner shared by all scripts, with task-ordered output
   (§1).
2. Reproduction of the 8 committed CSVs as a regression check, on every
   push (§2). On a mismatch, stop and report; never regenerate silently.
3. The CI split: the push job runs release with RTL plus ASan without RTL;
   the nightly job runs ASan with RTL plus all mutation checks (§3).
4. Runtime reported as two separate measurements, plus runner speedup (§4).
5. Phase 6 closes the project: a final README summary, with no new model
   features.

## 10. Code layout

```
python/runner.py                 process-pool runner + --selftest
python/reproduce.py              regenerate and byte-compare the 8 committed CSVs + --selftest
python/{gemm_check,attn_check,correlate,sweep_p3,study_p4,sweep_p5}.py    --jobs N
.github/workflows/regression.yml push and nightly jobs
scripts/phase6_regression.sh     the push-job steps with per-step timing (CI and Codespace)
scripts/phase6_ci_times.sh       GitHub run/job times -> results/phase6/ci_runtime.txt
scripts/phase6_exit.sh, phase6_mutation_check.sh, phase6_results.sh
```
