# Accelerator Performance Model

A SystemC TLM-2.0 transaction-level performance model of a generic AI
inference accelerator: off-chip DRAM, a banked on-chip SRAM, DMA, and (in
later phases) a PE grid with systolic compute units, a mesh network,
attention workloads, numerics analysis and an RTL correlation of the compute
unit.

**This models no real chip.** Every latency, bandwidth and size is a
configurable modelling parameter, not a datasheet value. See
[docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Status

| Phase | Content | Design note | Status |
|---|---|---|---|
| 1 | Memory system: DRAM channel, banked SRAM, DMA | [docs/phase1-design.md](docs/phase1-design.md) | done: [exit criterion](results/phase1/exit_criterion.txt), [mutation check](results/phase1/mutation_check.txt) |
| 2 | Compute unit, double buffering, functional reference | [docs/phase2-design.md](docs/phase2-design.md) | done: [exit criterion](results/phase2/exit_criterion.txt), [functional cases](results/phase2/functional.csv), [mutation check](results/phase2/mutation_check.txt) |
| 3 | Accelerator top, stall stack, roofline | [docs/phase3-design.md](docs/phase3-design.md) | done: [exit criterion](results/phase3/exit_criterion.txt), [sweep](results/phase3/sweep.csv), [roofline](results/phase3/roofline.png), [mutation check](results/phase3/mutation_check.txt), [hypotheses](RESULTS.md) |
| 4 | Attention / GenAI workloads, precision modes | — | not started |
| 5 | Microarchitecture sweeps, SystemVerilog correlation | — | not started |
| 6 | Automation and CI | — | not started |

## Architecture (through Phase 3)

```
                 ┌──────────────────────── PE ───────────────────────────┐
  DramChannel ◄──┤ DmaEngine ──► LMEM (banked Sram):  A0 B0 A1 B1 C0 C1  │
  (banks, rows,  │                 │ ideal port            ▲ drain       │
   shared bus,   │                 ▼                       │             │
   queue)        │      systolic array R×C (output-stationary, INT8→INT32)│
                 │      controller: double-buffered GEMM schedule         │
                 └────────────────────────────────────────────────────────┘
```

- **Memory system (Phase 1):** all components talk through the TLM-2.0 base
  protocol in the approximately-timed style. Targets sample requests once
  per cycle and arbitrate round-robin. Timing inside each target comes from
  integer formulas derived by hand. See
  [docs/phase1-design.md](docs/phase1-design.md) §2–§3.
- **PE (Phase 2):** an output-stationary systolic array with tile latency
  `K + 2R + C − 2`, a local memory with capacity checks, and a controller
  that overlaps DMA with compute (double buffering). For every GEMM the
  model computes real INT8→INT32 results, which are checked bit-exactly
  against NumPy and a standalone C++ reference. See
  [docs/phase2-design.md](docs/phase2-design.md).
- **Accelerator (Phase 3):** a Px×Py grid of these PEs on a mesh (XY
  routing, fixed hop latency, per-link bandwidth), a shared banked SRAM, and
  a scheduler that stages operand blocks DRAM→SRAM once per wave and lets
  every PE that needs them fetch them over the mesh.
  - **Stall stack:** every PE-cycle is charged to exactly one of compute,
    dram, sram_conflict, network, dma or idle. Every run asserts that the
    stack sums to P·T_end, that compute matches the tile formula, and that
    DRAM bytes match the staging formula.
  - **Roofline:** a three-ceiling roofline (compute, DRAM, NoC) computed
    from formulas only. See [docs/phase3-design.md](docs/phase3-design.md).

```
 DRAM ─ global DMA ─ shared SRAM @ router sram_node ─┬─ mesh ─ PE(0,0) … PE(Px-1,Py-1)
                                                      (each PE as above, operands via the mesh)
```

## Build and test

The toolchain is installed by `.devcontainer/setup.sh` (Ubuntu 24.04,
SystemC 3.0.2 built from source at a pinned commit, GoogleTest via CMake
FetchContent). Open the repo in a GitHub Codespace, or run that script on
Ubuntu 24.04.

```
cmake --preset release && cmake --build --preset release && ctest --preset release   # -O3, used for results
cmake --preset asan    && cmake --build --preset asan    && ctest --preset asan      # ASan + UBSan, correctness
build/release/apm_scenario --list        # directed scenarios
build/release/apm_scenario t2            # one scenario, JSON cycles and counters
# one GEMM: A.bin (M*K int8) and B.bin (K*N int8) in, C.bin (M*N int32) out, cycles as JSON
build/release/apm_gemm M=8 N=32 K=128 R=4 C=16 Kt=64 a=A.bin b=B.bin out=C.bin
python3 python/gemm_check.py --apm-gemm build/release/apm_gemm --gemm-ref build/release/gemm_ref --cases 20
# one GEMM on a 2x3 grid with the SRAM in the middle: cycles and the stall stack as JSON
build/release/apm_accel M=16 N=96 K=256 R=4 C=16 Kt=64 px=2 py=3 sram_x=1 sram_y=1 seed=3
python3 python/roofline.py                                       # formula-only ceilings, design note §4.1
python3 python/sweep_p3.py --apm-accel build/release/apm_accel --out-dir /tmp/sweep
```

## Reproducing results

| Result | Command | Output |
|---|---|---|
| All Phase 1 results, each on a clean tree | `scripts/phase1_results.sh` | both files below |
| Phase 1 exit criterion (hand-derived cycles, both presets) | `scripts/phase1_exit.sh` | `results/phase1/exit_criterion.txt` |
| Phase 1 mutation check | `scripts/phase1_mutation_check.sh` | `results/phase1/mutation_check.txt` |
| All Phase 2 results, each on a clean tree | `scripts/phase2_results.sh` | the three files below |
| Phase 2 exit criterion (hand-derived timing, both presets, 200-case functional match) | `scripts/phase2_exit.sh` | `results/phase2/exit_criterion.txt`, `results/phase2/functional.csv` |
| Phase 2 mutation check | `scripts/phase2_mutation_check.sh` | `results/phase2/mutation_check.txt` |
| All Phase 3 results, each on a clean tree | `scripts/phase3_results.sh` | the files below |
| Phase 3 exit criterion: hand-derived stall stacks, 100-case grid functional match, 115-config sweep | `scripts/phase3_exit.sh` | `results/phase3/exit_criterion.txt`, `functional_accel.csv`, `sweep.csv`, `roofline.png`, `stall_stacks.png` |
| Phase 3 mutation check | `scripts/phase3_mutation_check.sh` | `results/phase3/mutation_check.txt` |
| Every number in RESULTS.md | `scripts/check_results.sh` | pass/fail |

Every results file starts with a provenance block (`scripts/env_report.sh`):
commit, clean/dirty tree, machine, and tool versions.

## Documentation

- [docs/phase1-design.md](docs/phase1-design.md): Phase 1 design note
- [docs/phase2-design.md](docs/phase2-design.md): Phase 2 design note
- [docs/phase3-design.md](docs/phase3-design.md): Phase 3 design note
- [RESULTS.md](RESULTS.md): measured results and hypothesis verdicts
- [docs/BUGS.md](docs/BUGS.md): every bug found, and how it was detected
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md): what is not modelled
