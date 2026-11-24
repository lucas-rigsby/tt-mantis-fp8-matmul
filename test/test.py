# SPDX-License-Identifier: Apache-2.0
"""Pin-level cocotb tests for tt_um_mantis, run through tb.v.

Ported from tb/test_top.py in the development repository
(https://github.com/lucas-rigsby/mantis-fp8-matmul), which also holds the
exhaustive multiplier and adder tests and the formal proofs. Each test names
the requirement IDs from docs/spec.md that it checks.

  readout_example  R1, R12: spec bytes 98 41 B0 41 2C 42 48 42, p wraps 7 -> 0
  pointer_model    R12, R13, R14, P5, P6: p and uo_out against a model
  read_while_busy  R12: READs during computation don't disturb results
  static_pins      R15, P1: uio_oe = 0x90, unused uio_out bits stay 0
  nan_and_busy     P2, P3, R11: busy on uio_out[4], drain <= 8, NaN reads C0 7F
  special_results_at_pins  R8, R9: +inf, -inf and exact zero at the pins
  flush_at_pins    R13, R14, P10: CLEAR or reset 1..7 edges after an issue,
                   checked through uo_out
  jtag_inert       R25, P8: asynchronous toggling of the reserved JTAG pins
                   leaves every output identical to a quiet run
  stream_readout   R1, R10: back-to-back streams in all four format pairs

Run from test/:  make            (RTL)
                 make GATES=yes  (gate level, after the GDS flow)

At gate level the internal signals (p, acc, busy) do not exist, so checks
that probe them are skipped and the random workloads are shortened.
"""
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge, Timer

from mantis_ref import matmul, rand_matmul, readout_bytes, write_stream

GL = "GL_TEST" in cocotb.plusargs


def n(rtl, gl):
    """Workload size: full at RTL, reduced at gate level."""
    return gl if GL else rtl


WRITE, CLEAR, READ, NOP = 0, 1, 2, 3
JTAG_MASK = 0b1111_1000            # uio_in[7:3]: tck, (4), tms, tdi, (7)
_QUIET_RUN = {}                    # jtag_inert_quiet -> jtag_inert_noisy
SETTLE_NS = 5                      # output sampling point after the rising edge


class Top:
    """Pin driver. Low uio_in bits come from the command driver, high bits
    from the optional JTAG noise task; both are merged on every write."""

    def __init__(self, dut):
        self.dut = dut
        self.p = 0                    # model of the read pointer
        self.low = 0
        self.high = 0
        self.trace = []

    def _drive_uio(self):
        self.dut.uio_in.value = (self.high & JTAG_MASK) | (self.low & 0b111)

    async def cycle(self, valid=0, cmd=NOP, din=0, rst_n=1):
        await FallingEdge(self.dut.clk)
        self.dut.rst_n.value = rst_n
        self.dut.ui_in.value = din
        self.low = (cmd << 1) | valid
        self._drive_uio()
        await RisingEdge(self.dut.clk)
        # Sample after the edge has settled: in the gate-level netlist the
        # flip-flop outputs change UNIT_DELAY (1 ns) after the edge.
        await Timer(SETTLE_NS, unit="ns")
        await ReadOnly()
        uo, uio, oe = int(self.dut.uo_out.value), int(self.dut.uio_out.value), int(self.dut.uio_oe.value)
        assert oe == 0x90, f"P1: uio_oe = {oe:#04x}"
        assert uio & 0b1110_1111 == 0, f"P1, R25: uio_out = {uio:#04x}"
        self.trace.append((uo, uio))
        if not rst_n or (valid and cmd == CLEAR):
            self.p = 0
        elif valid and cmd == READ:
            self.p = (self.p + 1) % 8

    @property
    def busy(self):
        return (int(self.dut.uio_out.value) >> 4) & 1

    @property
    def uo(self):
        return int(self.dut.uo_out.value)

    async def clear(self, fa=0, fb=0):
        await self.cycle(1, CLEAR, (fb << 1) | fa)

    async def write(self, b):
        await self.cycle(1, WRITE, b)

    async def read(self):
        await self.cycle(1, READ, 0)

    async def drain(self, limit=64):
        for _ in range(limit):
            if not self.busy:
                return
            await self.cycle()
        raise AssertionError("R11: busy never fell")

    async def read_all(self):
        """Read p = 0..7 starting from p = 0; ends back at p = 0."""
        out = []
        for _ in range(8):
            out.append(self.uo)
            await self.read()
        return out


async def jtag_noise(top, rng):
    """Toggle uio_in[7:3] at random asynchronous times (not clock-aligned)."""
    while True:
        await Timer(rng.randint(1_000, 37_000), unit="ps")
        top.high = rng.getrandbits(8)
        top._drive_uio()


async def start(dut, noise_seed=None):
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())   # 50 MHz
    dut.ena.value = 1
    top = Top(dut)
    if noise_seed is not None:
        cocotb.start_soon(jtag_noise(top, random.Random(noise_seed)))
    await top.cycle(rst_n=0)
    await top.cycle(rst_n=0)
    await top.cycle()
    top.trace.clear()
    return top


async def matmul_on_pins(top, A, B, fa, fb):
    await top.clear(fa, fb)
    for b in write_stream(A, B):
        await top.write(b)
    await top.drain()
    return await top.read_all()


@cocotb.test()
async def readout_example(dut):
    """R1, R12: the spec's 8 result bytes, in order, and p wraps 7 -> 0."""
    top = await start(dut)
    got = await matmul_on_pins(top, [[0x38, 0x40], [0x44, 0x48]], [[0x4A, 0x4C], [0x4E, 0x50]], 0, 0)
    assert got == [0x98, 0x41, 0xB0, 0x41, 0x2C, 0x42, 0x48, 0x42], \
        "R1: " + " ".join("%02X" % b for b in got)
    assert top.uo == 0x98, "R12: p must wrap from 7 to 0"
    await top.clear()
    assert top.uo == 0x00, "R13: CLEAR must zero accumulators and p"


@cocotb.test()
async def pointer_model(dut):
    """R12, R13, R14, P5, P6: p and uo_out against a model on every cycle."""
    top = await start(dut)
    rng = random.Random(81)
    p = 0
    for cycle in range(n(6000, 1000)):
        r = rng.random()
        if r < 0.01:
            await top.cycle(rng.getrandbits(1), rng.randrange(4), rng.getrandbits(8), rst_n=0)
            p = 0                                           # R14
        else:
            valid, cmd = rng.getrandbits(1), rng.randrange(4)
            await top.cycle(valid, cmd, rng.getrandbits(8))
            if valid and cmd == CLEAR:
                p = 0                                       # R13
            elif valid and cmd == READ:
                p = (p + 1) % 8                             # R12
        if GL:
            continue
        core = dut.user_project
        assert int(core.p.value) == p, f"P5/P6: cycle {cycle} p={int(core.p.value)} model={p}"
        acc = int(core.acc.value)
        exp = (acc >> (8 * p)) & 0xFF                       # byte map == little-endian {C11..C00}
        assert top.uo == exp, f"R12: cycle {cycle} p={p} uo_out={top.uo:#04x} expected {exp:#04x}"
    dut._log.info(f"R12: {cycle + 1} random cycles" + ("" if GL else ", p and uo_out match the model"))


@cocotb.test()
async def read_while_busy(dut):
    """R12: READ during computation doesn't disturb results or ordering."""
    top = await start(dut)
    rng = random.Random(82)
    for trial in range(n(60, 10)):
        A, B, fa, fb = rand_matmul(rng, rng.randint(1, 24))
        await top.clear(fa, fb)
        reads = 0
        for b in write_stream(A, B):
            while rng.random() < 0.4:
                await top.read()
                reads += 1
            await top.write(b)
        while top.busy:
            await top.read()
            reads += 1
        for _ in range((8 - reads % 8) % 8):                # back to p = 0
            await top.read()
        got = await top.read_all()
        exp = readout_bytes(matmul(A, B, fa, fb))
        assert got == exp, f"trial {trial}: got {got} expected {exp}"


@cocotb.test()
async def static_pins(dut):
    """R15, P1: static outputs hold through reset, traffic and idle cycles
    (checked on every cycle inside Top.cycle)."""
    top = await start(dut)
    rng = random.Random(83)
    for _ in range(n(3000, 500)):
        await top.cycle(rng.getrandbits(1), rng.randrange(4), rng.getrandbits(8),
                        rst_n=0 if rng.random() < 0.01 else 1)
        if not GL:
            assert top.busy == int(dut.user_project.busy.value), "busy must drive uio_out[4]"


@cocotb.test()
async def nan_and_busy(dut):
    """P2, P3, R11: busy pin and drain; NaN accumulator reads C0 7F."""
    top = await start(dut)
    await top.clear(0, 0)
    for b in (0x7F, 0x38, 0x38):                            # A00 = NaN
        await top.write(b)
        assert top.busy == 0, "R11: partial group must not set busy"
    await top.write(0x38)
    assert top.busy == 1, "P2: busy must be 1 after the issuing edge"
    drain = 1
    while True:
        await top.cycle()
        if not top.busy:
            break
        drain += 1
        assert drain <= 8, "R11: drain exceeds 8 cycles"
    dut._log.info(f"R11: drain = {drain} cycles at the pins")
    # NaN persists through a later finite step (P3)
    for b in (0x38, 0x38, 0x38, 0x38):
        await top.write(b)
    await top.drain()
    got = await top.read_all()
    assert got[0:2] == [0xC0, 0x7F] and got[2:4] == [0xC0, 0x7F], f"P3: {got}"
    assert got[4:8] == [0x00, 0x40, 0x00, 0x40], f"finite C10/C11 = 2.0: {got}"


async def _traffic(top, seed):
    rng = random.Random(seed)
    results = []
    for _ in range(n(40, 8)):
        A, B, fa, fb = rand_matmul(rng, rng.randint(1, 16), finite=rng.random() < 0.8)
        await top.clear(fa, fb)
        for b in write_stream(A, B):
            while rng.random() < 0.25:
                await top.cycle(rng.getrandbits(1), rng.choice([NOP, READ]), rng.getrandbits(8))
            await top.write(b)
        await top.drain()
        while top.p:
            await top.read()
        got = await top.read_all()
        assert got == readout_bytes(matmul(A, B, fa, fb)), f"R1: traffic result {got}"
        results.append(got)
    return results


@cocotb.test()
async def special_results_at_pins(dut):
    """R8, R9: +inf, -inf and an exact zero read back through uo_out (E5M2)."""
    top = await start(dut)
    one, m_one, inf, m_inf = 0x3C, 0xBC, 0x7C, 0xFC
    cases = (
        ([[inf, one], [m_inf, one]], [[one, one], [one, m_one]]),   # C0x = +inf, C1x = -inf
        ([[one, one], [one, one]], [[one, one], [m_one, one]]),     # C00 = C10 = 1 - 1 = +0
    )
    for A, B in cases:
        got = await matmul_on_pins(top, A, B, 1, 1)
        exp = readout_bytes(matmul(A, B, 1, 1))
        assert got == exp, f"R8/R9: got {got} expected {exp}"


@cocotb.test()
async def flush_at_pins(dut):
    """R13, R14, P10: CLEAR or reset at each edge 1..7 after an issue reads back
    all zeros, and nothing issued before it commits later."""
    top = await start(dut)
    for action in ("clear", "reset"):
        for k in range(1, 8):
            await top.clear(0, 0)
            for b in write_stream([[0x38, 0x38], [0x38, 0x38]], [[0x38, 0x38], [0x38, 0x38]]):
                await top.write(b)                        # last WRITE is the issuing edge
            for _ in range(k - 1):
                await top.cycle()
            if action == "clear":
                await top.clear(0, 0)
            else:
                await top.cycle(rst_n=0)
            assert await top.read_all() == [0] * 8, f"{action} {k} edges after issue: not zero"
            for _ in range(12):
                await top.cycle()
            assert await top.read_all() == [0] * 8, f"P10: {action} {k}: an old step committed"


@cocotb.test()
async def jtag_inert_quiet(dut):
    """R25 reference run: JTAG pins held at 0."""
    top = await start(dut)
    res = await _traffic(top, 84)
    _QUIET_RUN['res'], _QUIET_RUN['trace'] = res, list(top.trace)


@cocotb.test()
async def jtag_inert_noisy(dut):
    """R25, P8: asynchronous noise on tck/tms/tdi gives identical outputs."""
    top = await start(dut, noise_seed=85)
    res = await _traffic(top, 84)
    q_res, q_trace = _QUIET_RUN['res'], _QUIET_RUN['trace']
    assert res == q_res, "R25: results differ with JTAG pin activity"
    assert top.trace == q_trace, "R25: an output differed on some cycle with JTAG pin activity"
    assert all(uio >> 7 == 0 for _, uio in top.trace), "R25: tdo must stay 0"
    dut._log.info(f"R25: {len(top.trace)} cycles identical with JTAG noise")


@cocotb.test()
async def stream_readout(dut):
    """R1, R10: 4 x 200 back-to-back groups, results read from the pins."""
    top = await start(dut)
    rng = random.Random(86)
    for fa, fb in ((0, 0), (0, 1), (1, 0), (1, 1)):
        A, B, _, _ = rand_matmul(rng, n(200, 50), fa, fb)
        got = await matmul_on_pins(top, A, B, fa, fb)
        assert got == readout_bytes(matmul(A, B, fa, fb)), f"R10 fmt=({fa},{fb})"
