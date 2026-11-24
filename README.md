![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg) ![](../../workflows/fpga/badge.svg)

# MANTIS: FP8 matrix multiplier for Tiny Tapeout

MANTIS (`tt_um_lucas_rigsby_mantis`) multiplies a $2 \times K$ matrix by a $K \times 2$ matrix
of FP8 values (E4M3FN or E5M2) and accumulates the result in BF16. It occupies a
$1 \times 2$ tile slot on Sky130 and runs at 50 MHz.

- [Datasheet: how it works and how to test it](docs/info.md)
- [Development repository](https://github.com/lucas-rigsby/mantis-fp8-matmul):
  specification, design notes, unit tests, formal proofs and synthesis scripts

This repository is the Tiny Tapeout submission. The files in `src/` are copied
from `rtl/` in the development repository, and `test/test.py` is the pin-level
test suite from that repository adapted to the Tiny Tapeout testbench.

## Layout

| Path | Contents |
|---|---|
| `src/` | Verilog sources, top module `tt_um_lucas_rigsby_mantis` |
| `info.yaml` | project metadata, pinout and source list |
| `docs/info.md` | datasheet page |
| `test/test.py` | cocotb tests through `test/tb.v` |
| `test/mantis_ref.py` | bit-accurate Python reference model |

## Running the tests

The tests need Icarus Verilog and the packages in `test/requirements.txt`.

```bash
cd test && make
```

After the GDS workflow produces a gate-level netlist, the same tests run against
it with `make GATES=yes`. At gate level the checks that probe internal signals
are skipped and the random workloads are shorter.

## Tiny Tapeout

Tiny Tapeout is an educational project that makes it easier and cheaper to get
digital and analog designs manufactured on a real chip. See
https://tinytapeout.com, the [FAQ](https://tinytapeout.com/faq/) and the guide to
[building a design locally](https://www.tinytapeout.com/guides/local-hardening/).
