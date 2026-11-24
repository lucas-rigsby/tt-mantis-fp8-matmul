"""MANTIS bring-up test: the same checks in simulation and on the demo board.

The test logic (smoke check, every vector in vectors.py, optional at-speed
drain) is written once against a small pin driver:

  CocotbPins  simulation: RTL (make bringup-sim) or the gate-level netlist
              (Tiny Tapeout repo, make GATES=yes)
  BoardPins   Tiny Tapeout demo board, MicroPython on the RP2040, using the
              tt-micropython-firmware SDK (ttboard)

On the board, copy this file and vectors.py to the RP2040 and run

  import test_bringup
  test_bringup.main("tt_um_lucas_rigsby_mantis")                 # manual clock
  test_bringup.main("tt_um_lucas_rigsby_mantis", [1_000_000, 10_000_000, 50_000_000])

The second form also runs every vector with the 6-cycle drain of each step
clocked at each listed frequency (see plan.md, step 5).

Protocol (docs/spec.md): uio_in[0] = in_valid, uio_in[2:1] = cmd
(WRITE 0, CLEAR 1, READ 2, NOP 3), ui_in = data, uo_out = result byte at the
read pointer, uio_out[4] = busy, uio_out[7] = 0.
"""
from binascii import unhexlify

from vectors import VECTORS

WRITE, CLEAR, READ, NOP = 0, 1, 2, 3
MAX_DRAIN = 8                     # R11: busy falls within 8 clocks of the issuing edge
TOP = "tt_um_lucas_rigsby_mantis"


def _uio(valid, cmd):
    return (cmd << 1) | valid


# ------------------------------------------------------------------ test logic
async def command(p, cmd, din=0):
    await p.cycle(din, _uio(1, cmd))


async def idle(p):
    await p.cycle(0, 0)


def busy(p):
    return (p.uio_out() >> 4) & 1


async def read_all(p):
    """Read the 8 result bytes from p = 0; leaves p back at 0."""
    out = []
    for _ in range(8):
        out.append(p.uo_out())
        await command(p, READ)
    return bytes(out)


async def drain(p):
    n = 0
    while busy(p):
        await idle(p)
        n += 1
        if n > MAX_DRAIN:
            raise AssertionError("R11: busy still high %d clocks after the last WRITE" % n)


async def smoke(p):
    """Reset state at the pins: all results +0, busy = 0, tdo = 0, p wraps."""
    await p.reset()
    assert p.uio_out() & 0x90 == 0, "after reset uio_out = 0x%02X: busy or tdo set" % p.uio_out()
    got = await read_all(p)
    assert got == bytes(8), "after reset the results read %s, expected all 00" % got.hex()
    await command(p, CLEAR, 0)
    for _ in range(3):
        await idle(p)
    assert p.uio_out() & 0x90 == 0, "uio_out = 0x%02X while idle" % p.uio_out()


async def run_vector(p, fa, fb, writes, at_speed_hz=None):
    await command(p, CLEAR, (fb << 1) | fa)
    for i, b in enumerate(writes):
        await command(p, WRITE, b)
        if at_speed_hz and i % 4 == 3:            # this WRITE issued a step
            await p.burst(at_speed_hz)            # its pipeline runs at speed
    await drain(p)
    return await read_all(p)


async def run_vectors(p, at_speed_hz=None, log=print):
    failures = []
    for name, fa, fb, w, r in VECTORS:
        got = await run_vector(p, fa, fb, unhexlify(w), at_speed_hz)
        if got != unhexlify(r):
            failures.append(name)
            log("FAIL %s: got %s expected %s" % (name, got.hex(), r))
    return failures


# ------------------------------------------------------------------ simulation
try:
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge, Timer
except ImportError:              # MicroPython on the demo board
    cocotb = None


class CocotbPins:
    """Drive the Tiny Tapeout wrapper (or the top module itself) in cocotb.

    Inputs change at the falling edge; outputs are sampled 5 ns after the
    rising edge, after the gate-level flip-flops have updated.
    """
    SETTLE_NS = 5

    def __init__(self, dut):
        self.dut = dut

    async def cycle(self, ui, uio, rst_n=1):
        await FallingEdge(self.dut.clk)
        self.dut.rst_n.value = rst_n
        self.dut.ui_in.value = ui
        self.dut.uio_in.value = uio
        await RisingEdge(self.dut.clk)
        await Timer(self.SETTLE_NS, unit="ns")
        await ReadOnly()

    async def reset(self):
        for _ in range(2):
            await self.cycle(0, 0, rst_n=0)
        await self.cycle(0, 0)

    async def burst(self, hz):
        for _ in range(MAX_DRAIN):                # the simulation clock is already 50 MHz
            await self.cycle(0, 0)

    def uo_out(self):
        return int(self.dut.uo_out.value)

    def uio_out(self):
        return int(self.dut.uio_out.value)


if cocotb is not None:
    async def _start(dut):
        cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())   # 50 MHz
        dut.ena.value = 1
        p = CocotbPins(dut)
        await p.reset()
        return p

    @cocotb.test()
    async def bringup_smoke(dut):
        """Board step 3a: reset state and read pointer at the pins."""
        await smoke(await _start(dut))

    @cocotb.test()
    async def bringup_vectors(dut):
        """Board step 3b: every exported vector, read back through uo_out."""
        p = await _start(dut)
        failures = await run_vectors(p, log=dut._log.error)
        assert not failures, "%d of %d vectors failed" % (len(failures), len(VECTORS))
        dut._log.info("%d bring-up vectors match the reference model" % len(VECTORS))

    @cocotb.test()
    async def bringup_at_speed(dut):
        """Board step 5 in simulation: vectors with an idle burst after each issue."""
        p = await _start(dut)
        failures = await run_vectors(p, at_speed_hz=50_000_000, log=dut._log.error)
        assert not failures, "%d of %d vectors failed" % (len(failures), len(VECTORS))


def test_bringup_sim():
    """pytest entry for `make bringup-sim`: RTL of the top module, Icarus."""
    import os
    from pathlib import Path
    from cocotb_tools.runner import get_runner
    here = Path(__file__).resolve().parent
    rtl = here.parent / "rtl"
    sim = os.getenv("SIM", "icarus")
    runner = get_runner(sim)
    runner.build(
        sources=[rtl / f for f in ("fp8_unpack.v", "fp8_mul.v", "bf16_add_s1.v", "bf16_add_s2.v",
                                   "mantis_stage.v", "mantis_dp_shared.v", TOP + ".v")],
        hdl_toplevel=TOP,
        build_dir=here.parent / "tb" / "sim_build" / ("bringup_" + sim),
        always=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(hdl_toplevel=TOP, test_module="test_bringup", test_dir=here)


# ------------------------------------------------------------------ demo board
class BoardPins:
    """Tiny Tapeout demo board through the ttboard SDK (MicroPython, RP2040).

    The project clock is stepped by the RP2040 (clock_project_once), so every
    command is applied with the inputs stable around the clock edge. Not yet
    run on hardware: check the SDK names against the firmware on the board.
    """

    def __init__(self, top):
        from ttboard.demoboard import DemoBoard
        from ttboard.mode import RPMode
        self.tt = tt = DemoBoard.get()
        tt.mode = RPMode.ASIC_RP_CONTROL
        getattr(tt.shuttle, top).enable()
        tt.clock_project_stop()
        # RP2040 drives uio[3:0] and uio[6:5] (in_valid, cmd and the reserved
        # JTAG inputs, held at 0); uio[4] (busy) and uio[7] (tdo) are chip outputs.
        tt.uio_oe_pico.value = 0b0110_1111

    async def cycle(self, ui, uio, rst_n=1):
        self.tt.ui_in.value = ui
        self.tt.uio_in.value = uio
        self.tt.clock_project_once()

    async def reset(self):
        self.tt.ui_in.value = 0
        self.tt.uio_in.value = 0
        self.tt.reset_project(True)
        for _ in range(2):
            self.tt.clock_project_once()
        self.tt.reset_project(False)
        self.tt.clock_project_once()

    async def burst(self, hz):
        import time
        self.tt.ui_in.value = 0
        self.tt.uio_in.value = 0                  # idle: in_valid = 0
        self.tt.clock_project_PWM(hz)
        time.sleep_ms(2)                          # thousands of cycles; the drain needs 6
        self.tt.clock_project_stop()

    def uo_out(self):
        return int(self.tt.uo_out.value)

    def uio_out(self):
        return int(self.tt.uio_out.value)


def main(top=TOP, at_speed_hz=()):
    """Run on the demo board. Returns True when every check passes."""
    try:
        import asyncio
    except ImportError:
        import uasyncio as asyncio

    async def go():
        p = BoardPins(top)
        await smoke(p)
        print("smoke: pass")
        ok = True
        failures = await run_vectors(p)
        print("vectors, manual clock: %d of %d pass" % (len(VECTORS) - len(failures), len(VECTORS)))
        ok = ok and not failures
        for hz in at_speed_hz:
            failures = await run_vectors(p, at_speed_hz=hz)
            print("vectors, drain at %d Hz: %d of %d pass" % (hz, len(VECTORS) - len(failures), len(VECTORS)))
            ok = ok and not failures
        return ok

    return asyncio.run(go())
