<!---
This file is used to generate the project datasheet.
-->

## How it works

MANTIS multiplies a $2 \times K$ matrix $A$ by a $K \times 2$ matrix $B$ of 8-bit
floating-point values and accumulates the four results in BF16:

$$C_{ij} = \sum_{k=0}^{K-1} A_{ik} B_{kj}, \quad i, j \in \{0, 1\}$$

Each operand matrix can be OCP FP8 E4M3FN or E5M2, selected independently. Every
FP8 product is exact in BF16, so the only rounding is round-to-nearest-even after
each accumulation step. NaN operands, $\infty \times 0$ and
$(+\infty) + (-\infty)$ give the canonical NaN `0x7FC0`, which stays in that
accumulator until the next CLEAR. Subnormal BF16 results are flushed to zero.

The host writes one byte per clock. Four bytes form one reduction step $k$, in the
order $A_{0k}$, $A_{1k}$, $B_{k0}$, $B_{k1}$. The fourth write issues four
multiply-accumulates to a single shared multiplier and two-stage adder, which
serve the accumulators $C_{00}$, $C_{01}$, $C_{10}$ and $C_{11}$ in turn. A new
step can be issued every 4 clocks, which matches the input rate, so the host never
waits. `busy` is high while steps are in flight and falls at most 8 clocks after
the last write.

Commands are given on `uio[2:1]` and accepted on a rising clock edge when
`in_valid` (`uio[0]`) is high:

| `cmd` | Name | Effect |
|---|---|---|
| `00` | WRITE | store `ui_in` as the next operand byte |
| `01` | CLEAR | zero the accumulators, set the formats from `ui_in[0]` (A) and `ui_in[1]` (B), reset the read pointer |
| `10` | READ | advance the read pointer $p$ to $(p + 1) \bmod 8$ |
| `11` | NOP | no effect |

A format bit of 0 selects E4M3FN and 1 selects E5M2. CLEAR also discards any
partial group and any steps still in flight.

`uo_out` always shows the result byte selected by the read pointer $p$, low byte
first:

| $p$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| `uo_out` | $C_{00}$ low | $C_{00}$ high | $C_{01}$ low | $C_{01}$ high | $C_{10}$ low | $C_{10}$ high | $C_{11}$ low | $C_{11}$ high |

The pins `uio[3]`, `uio[5]`, `uio[6]` and `uio[7]` are reserved for a later JTAG
test port. In this design the inputs are ignored and `uio[7]` drives 0.

The full specification, design notes and verification are in the development
repository: <https://github.com/lucas-rigsby/mantis-fp8-matmul>.

## How to test

Drive the pins from the RP2040 on the demo board, or any microcontroller, with
the project clock at up to 50 MHz.

1. Hold `rst_n` low for at least one clock, then release it.
2. CLEAR with `ui_in = 0x00` to select E4M3FN for both matrices.
3. WRITE these eight bytes, one per clock: `38 44 4A 4C 40 48 4E 50`.
   In E4M3FN, $A$ has rows $(1, 2)$ and $(3, 4)$, and $B$ has rows $(5, 6)$ and
   $(7, 8)$.
4. Wait until `busy` (`uio[4]`) is low.
5. Read `uo_out`, then issue READ. Repeat until eight bytes have been read. They
   are `98 41 B0 41 2C 42 48 42`, the BF16 values $19$, $22$, $43$ and $50$.
   After the eighth READ the pointer is back at 0.

`uio_oe` is fixed at `0x90`, so `uio[4]` and `uio[7]` are outputs and the other
`uio` pins are inputs.

## External hardware

None. The demo board's RP2040 can drive the inputs and read the outputs.
