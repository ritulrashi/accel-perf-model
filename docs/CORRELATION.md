# RTL ↔ Model Correlation

How the SystemVerilog compute array (`rtl/os_array.sv`) is compared with the
model, and every deviation found. Method: [phase5-design.md](phase5-design.md)
§1–§5. **The contract (Phase 2 §2.2) is the referee, not the model.**

## What is compared

The model exports, for every PE, the feed schedule its controller actually
ran (`schedule=FILE`): feed cycles, the operand blocks read and the values
drained. The Verilator testbench (`tb/os_array_tb.cpp`) replays each schedule
into the RTL variant of that (R, C). It requires three things:

1. **Tile timing:** the RTL's first drain cycle equals the contract's
   `drain_start`, and its done cycle equals the model's recorded tile done.
2. **Values:** the R drained rows equal the model's accumulators, bit for bit.
3. **Every cycle:** the RTL's per-cell MAC trace (which cell performs which k
   in each cycle, and which row drains) equals `contract_trace` of the
   schedule, byte for byte.

**Independent of both sides:** the hand-written traces HT1 and HT2
(`tests/rtl/`) are written from the contract text. The RTL and
`contract_trace` must each reproduce them. The Phase 2 directed schedules are
also checked against the Phase 2 hand-derived tile-done cycles.

## Measured (results/phase5, commit 052a61f)

`python3 python/correlate.py --tools build/release`: **312 of 312 schedules
match at all three levels.** That covers 455 PE schedules, 2131 tiles,
774826 cycles and 26336028 MACs, compared over 301747 trace lines.

| Set | Schedules | Match |
|---|---|---|
| hand (HT1, HT2) | 2 | 2, both hand traces reproduced by RTL and contract |
| Phase 2 directed P2-T1–T5b | 6 | 6, tile-done cycles equal the Phase 2 hand derivations |
| Phase 2 random PE cases | 200 | 200 |
| Phase 3 random grid cases (every PE) | 100 | 100 |
| A4 INT8 attention GEMM ops | 4 | 4 |

The comparison can fail: `results/phase5/mutation_check.txt` injects six
RTL-side mutants (row skew, drain order, clear-on-first, drain timing,
contract expansion, testbench operand alignment), and all six are caught.

## Deviation log

| # | Program | Cycle / cell | Symptom | Root cause | Side at fault | Fix | Commit |
|---|---|---|---|---|---|---|---|
| — | none | — | No deviation was found in any development or official run. | — | — | — | — |

One RTL bug was found before any comparison ran: an instance name that is a
Verilog-2001 reserved word ([BUGS.md](BUGS.md) #13). It was a build error,
not a behavioural deviation.
