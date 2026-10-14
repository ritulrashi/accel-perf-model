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
| 1 | Memory system: DRAM channel, banked SRAM, DMA | [docs/phase1-design.md](docs/phase1-design.md) | in progress |
| 2 | Compute unit, double buffering, functional reference | — | not started |
| 3 | Accelerator top, stall stack, roofline | — | not started |
| 4 | Attention / GenAI workloads, precision modes | — | not started |
| 5 | Microarchitecture sweeps, SystemVerilog correlation | — | not started |
| 6 | Automation and CI | — | not started |

## Architecture (Phase 1)

```
  DmaEngine 0..n ──┬──────────────► DramChannel  (banks, open rows, shared data bus, request queue)
  (initiators)     │
                   └──────────────► Sram         (banks, one line per bank per cycle)
```

All components talk through the TLM-2.0 base protocol in the
approximately-timed style. Targets sample requests once per cycle and
arbitrate round-robin. Timing inside each target is computed by integer
formulas that are derived by hand in the design note. See
[docs/phase1-design.md](docs/phase1-design.md) §2–§3.

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
```

## Reproducing results

| Result | Command | Output |
|---|---|---|
| Phase 1 exit criterion (hand-derived cycles, both presets) | `scripts/phase1_exit.sh` | `results/phase1/exit_criterion.txt` |
| Phase 1 mutation check | `scripts/phase1_mutation_check.sh` | `results/phase1/mutation_check.txt` |

Every results file starts with a provenance block (`scripts/env_report.sh`):
commit, clean/dirty tree, machine, and tool versions.

## Documentation

- [docs/phase1-design.md](docs/phase1-design.md): Phase 1 design note
- [docs/BUGS.md](docs/BUGS.md): every bug found, and how it was detected
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md): what is not modelled
