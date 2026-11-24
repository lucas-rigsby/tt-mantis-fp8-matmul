"""Reference model for MANTIS FP8 -> BF16 arithmetic (see docs/spec.md).

Independence: every FP8 and BF16 bit pattern is decoded to an exact rational
value with ``fractions.Fraction``; products and sums are computed exactly in
the rationals; the result is re-encoded by an explicit BF16 round-to-nearest-
even encoder (``encode_bf16``) that works on the magnitude as a rational
number. The RTL instead manipulates integer significands and exponents. No
bit-level decode/normalize/round logic is shared.

Exactness: Fraction arithmetic is exact and unbounded, so nothing rounds
except in ``encode_bf16``, which is the single place the spec's RNE + overflow
+ FTZ rules are applied. ``encode_bf16`` uses only integer floor division and
comparisons on Fractions (no floats). A secondary, deliberately different
check (``_float_path_mul``) recomputes products through Python binary64; that
is exact because |product| has <= 8 significant bits and its exponent lies in
[-32, 32], well inside binary64.

Bit-accurate interface: all inputs/outputs are Python ints holding port bit
patterns. fmt: 0 = E4M3FN, 1 = E5M2 (spec, Number formats).

Doctests (run: python3 -m doctest -v tb/mantis_ref.py):

>>> hex(mul(0x7E, 0x7E, 0, 0))         # R2: E4M3FN 448*448 = 200704
'0x4844'
>>> hex(mul(0x01, 0x38, 0, 0))         # R4: min E4M3FN subnormal * 1
'0x3b00'
>>> hex(mul(0x3C, 0x7B, 1, 1))         # R5: E5M2 1 * 57344
'0x4760'
>>> hex(mul(0x38, 0x3C, 0, 1))         # R6: A=E4M3FN 1, B=E5M2 1
'0x3f80'
>>> hex(mul(0x7F, 0x38, 0, 0))         # R4, R7: E4M3FN NaN operand
'0x7fc0'
>>> hex(mul(0x7C, 0x00, 1, 1))         # R7: inf * 0 = NaN
'0x7fc0'
>>> hex(mul(0x7C, 0xBC, 1, 1))         # R8: +inf * -1 = -inf
'0xff80'
>>> hex(mul(0x80, 0x38, 0, 0))         # R9: -0 * 1 = -0
'0x8000'
>>> acc = 0
>>> for a, b in [(0x58, 0x58), (0x38, 0x38), (0x38, 0x38)]:   # R3: 16*16, 1*1, 1*1
...     acc = mac(acc, a, b, 0, 0)
>>> hex(acc)
'0x4380'
>>> hex(mac(mac(0, 0x38, 0x38, 0, 0), 0xB8, 0x38, 0, 0))     # R9: +1 then -1
'0x0'
>>> hex(mac(0, 0x80, 0x38, 0, 0))      # R9: +0 + (-0) = +0
'0x0'
>>> hex(mac(0x8000, 0x80, 0x38, 0, 0)) # R9 (unit level): -0 + -0 = -0
'0x8000'
>>> hex(mac(mac(0, 0x7C, 0x3C, 1, 1), 0x38, 0x38, 1, 1))     # R8: inf + 0.25
'0x7f80'
>>> hex(mac(0x7F80, 0xFC, 0x3C, 1, 1)) # R7: +inf + -inf = NaN
'0x7fc0'
>>> hex(mac(mac(0, 0x7F, 0x38, 0, 0), 0x38, 0x38, 0, 0))     # R7: NaN persists
'0x7fc0'
>>> hex(add_bf16(0x7F7F, encode_bf16(0, Fraction(2**119))))   # R8 overflow tie -> inf
'0x7f80'
>>> hex(add_bf16(0x7F7F, encode_bf16(0, Fraction(2**118))))   # below tie -> max finite
'0x7f7f'
>>> encode_bf16(0, Fraction(1, 2**127)) == 0                   # R9: FTZ
True
>>> C = matmul([[0x38, 0x40], [0x44, 0x48]], [[0x4A, 0x4C], [0x4E, 0x50]], 0, 0)
>>> ' '.join('%02X' % b for b in readout_bytes(C))           # R1 worked example
'98 41 B0 41 2C 42 48 42'
>>> ' '.join('%02X' % b for b in write_stream([[0x38, 0x40], [0x44, 0x48]], [[0x4A, 0x4C], [0x4E, 0x50]]))
'38 44 4A 4C 40 48 4E 50'
"""
import random
import struct
from fractions import Fraction

E4M3FN, E5M2 = 0, 1
BF16_NAN = 0x7FC0              # canonical NaN (R7)
BF16_PINF, BF16_NINF = 0x7F80, 0xFF80

# Value classes returned by the decoders
NUM, INF, NAN = "num", "inf", "nan"


# ---------------------------------------------------------------- decode
def decode_fp8(x: int, fmt: int):
    """Return (cls, sign, magnitude: Fraction) for FP8 bit pattern x."""
    assert 0 <= x <= 0xFF, "FP8 operand must be a byte"
    assert fmt in (E4M3FN, E5M2), "fmt must be 0 (E4M3FN) or 1 (E5M2)"
    s = x >> 7
    if fmt == E4M3FN:                              # R4: s.eeee.mmm, bias 7
        e, m = (x >> 3) & 0xF, x & 0x7
        if e == 0xF and m == 0x7:                  # R4: 0x7F/0xFF are NaN, no inf
            return NAN, s, None
        if e == 0:                                 # R4: subnormal m/8 * 2^-6
            return NUM, s, Fraction(m, 8) * Fraction(1, 2**6)
        return NUM, s, Fraction(8 + m, 8) * _pow2(e - 7)
    e, m = (x >> 2) & 0x1F, x & 0x3                # R5: s.eeeee.mm, bias 15
    if e == 0x1F:
        return (INF, s, None) if m == 0 else (NAN, s, None)   # R5
    if e == 0:                                     # R5: subnormal m/4 * 2^-14
        return NUM, s, Fraction(m, 4) * Fraction(1, 2**14)
    return NUM, s, Fraction(4 + m, 4) * _pow2(e - 15)


def decode_bf16(x: int):
    """Return (cls, sign, magnitude) for a BF16 bit pattern (accumulator)."""
    assert 0 <= x <= 0xFFFF, "BF16 value must be 16 bits"
    s, e, m = x >> 15, (x >> 7) & 0xFF, x & 0x7F
    if e == 0xFF:
        return (INF, s, None) if m == 0 else (NAN, s, None)
    if e == 0:
        # Subnormal accumulators are never produced (FTZ); treat as zero if fed.
        return NUM, s, Fraction(0)
    return NUM, s, Fraction(128 + m, 128) * _pow2(e - 127)


def _pow2(n: int) -> Fraction:
    return Fraction(2**n) if n >= 0 else Fraction(1, 2**-n)


# ---------------------------------------------------------------- encode
def encode_bf16(sign: int, mag: Fraction) -> int:
    """Round |value| = mag to BF16 with RNE, unbounded exponent first.

    Spec: overflow when the RNE result (unbounded exponent) exceeds the max
    finite exponent, i.e. exact magnitude >= 2^128 - 2^119 (R8). Results whose
    rounded magnitude is < 2^-126 flush to zero with the given sign (R9).
    """
    assert mag >= 0
    if mag == 0:
        return sign << 15
    # exponent e with 2^e <= mag < 2^(e+1), using exact integer bit lengths
    e = mag.numerator.bit_length() - mag.denominator.bit_length()
    if _pow2(e) > mag:
        e -= 1
    scaled = mag / _pow2(e - 7)                    # in [128, 256)
    q = scaled.numerator // scaled.denominator
    r = scaled - q
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and q & 1):   # R3: RNE
        q += 1
    if q == 256:                                   # carry into next binade
        q, e = 128, e + 1
    if e > 127:                                    # R8: overflow -> inf
        return (sign << 15) | 0x7F80
    if e < -126:                                   # R9: FTZ
        return sign << 15
    return (sign << 15) | ((e + 127) << 7) | (q - 128)


# ---------------------------------------------------------------- arithmetic
def mul(a: int, b: int, fmt_a: int, fmt_b: int) -> int:
    """Exact FP8 x FP8 product as a BF16 bit pattern (R2, R4-R9).

    The product is exactly representable (<= 8 significant bits, exponent in
    [-32, 32]); the assert below enforces that claim (R2).
    """
    ca, sa, va = decode_fp8(a, fmt_a)
    cb, sb, vb = decode_fp8(b, fmt_b)
    s = sa ^ sb
    if ca == NAN or cb == NAN:                     # R7
        return BF16_NAN
    if ca == INF or cb == INF:
        if (ca == NUM and va == 0) or (cb == NUM and vb == 0):
            return BF16_NAN                        # R7: inf * 0
        return (s << 15) | 0x7F80                  # R8: signed inf
    p = va * vb
    out = encode_bf16(s, p)                        # R9: zero product keeps XOR sign
    assert decode_bf16(out)[2] == p, \
        "R2: product must be exactly representable in BF16"
    return out


def add_bf16(acc: int, prod: int) -> int:
    """BF16_RNE(acc + prod) with the spec's special-value rules (R3, R7-R9)."""
    c1, s1, v1 = decode_bf16(acc)
    c2, s2, v2 = decode_bf16(prod)
    if c1 == NAN or c2 == NAN:                     # R7: NaN in, NaN persists
        return BF16_NAN
    if c1 == INF and c2 == INF:
        return BF16_NAN if s1 != s2 else acc       # R7: +inf + -inf; R8 same-sign
    if c1 == INF:
        return acc                                 # R8: inf + finite
    if c2 == INF:
        return prod
    total = (-v1 if s1 else v1) + (-v2 if s2 else v2)
    if total == 0:
        # R9: exact zero sum is +0 unless both addends are -0
        if v1 == 0 and v2 == 0 and s1 and s2:
            return 0x8000
        return 0x0000
    return encode_bf16(1 if total < 0 else 0, abs(total))


def mac(acc: int, a: int, b: int, fmt_a: int, fmt_b: int) -> int:
    """One accumulation step: acc <- BF16_RNE(acc + a*b) (R1, R3)."""
    return add_bf16(acc, mul(a, b, fmt_a, fmt_b))


def matmul(A, B, fmt_a: int, fmt_b: int):
    """A: 2xK FP8 bytes, B: Kx2 FP8 bytes -> [[C00, C01], [C10, C11]] BF16 (R1)."""
    K = len(A[0])
    assert len(A) == 2 and len(B) == K and K >= 1, "R1: A is 2xK, B is Kx2, K>=1"
    C = [[0, 0], [0, 0]]                           # R13: start from +0
    for k in range(K):                             # P11: ascending k
        for i in range(2):
            for j in range(2):
                C[i][j] = mac(C[i][j], A[i][k], B[k][j], fmt_a, fmt_b)
    return C


def write_stream(A, B):
    """WRITE byte order: per k, A[0][k], A[1][k], B[k][0], B[k][1]."""
    out = []
    for k in range(len(A[0])):
        out += [A[0][k], A[1][k], B[k][0], B[k][1]]
    return out


def readout_bytes(C):
    """Bytes for p = 0..7: C00, C01, C10, C11, each little-endian (R12)."""
    out = []
    for w in (C[0][0], C[0][1], C[1][0], C[1][1]):
        out += [w & 0xFF, w >> 8]
    return out


# ---------------------------------------------------------------- stimulus
SPECIAL_E4M3 = [0x00, 0x80, 0x01, 0x81, 0x07, 0x08, 0x38, 0xB8, 0x7E, 0xFE, 0x7F, 0xFF]
SPECIAL_E5M2 = [0x00, 0x80, 0x01, 0x81, 0x03, 0x04, 0x3C, 0xBC, 0x7B, 0xFB,
                0x7C, 0xFC, 0x7D, 0x7E, 0x7F, 0xFD, 0xFF]


def rand_operands(rng: random.Random):
    """Uniform legal (a, b, fmt_a, fmt_b)."""
    return rng.getrandbits(8), rng.getrandbits(8), rng.getrandbits(1), rng.getrandbits(1)


def boundary_operands(rng: random.Random):
    """Operands at format boundaries: zeros, min/max subnormal, min normal,
    max finite, inf, NaN (R4, R5, R7, R8, R9)."""
    fa, fb = rng.getrandbits(1), rng.getrandbits(1)
    pick = lambda f: rng.choice(SPECIAL_E5M2 if f else SPECIAL_E4M3)
    a = pick(fa) if rng.random() < 0.7 else rng.getrandbits(8)
    b = pick(fb) if rng.random() < 0.7 else rng.getrandbits(8)
    return a, b, fa, fb


def rand_finite(rng: random.Random, fmt: int) -> int:
    """Random finite (non-NaN, non-inf) FP8 byte."""
    while True:
        x = rng.getrandbits(8)
        if decode_fp8(x, fmt)[0] == NUM:
            return x


def rand_matmul(rng: random.Random, K: int, fmt_a=None, fmt_b=None, finite=True):
    """Random (A, B, fmt_a, fmt_b) for a K-step matrix multiply (R1, R10)."""
    fa = rng.getrandbits(1) if fmt_a is None else fmt_a
    fb = rng.getrandbits(1) if fmt_b is None else fmt_b
    ga = (lambda: rand_finite(rng, fa)) if finite else (lambda: rng.getrandbits(8))
    gb = (lambda: rand_finite(rng, fb)) if finite else (lambda: rng.getrandbits(8))
    A = [[ga() for _ in range(K)] for _ in range(2)]
    B = [[gb() for _ in range(2)] for _ in range(K)]
    return A, B, fa, fb


# Multiplier worked examples and boundaries: (a, b, fmt_a, fmt_b, expected, why)
DIRECTED_MUL = [
    (0x7E, 0x7E, 0, 0, 0x4844, "R2: E4M3FN 448*448 exact"),
    (0x01, 0x38, 0, 0, 0x3B00, "R4: min E4M3FN subnormal * 1"),
    (0x01, 0x01, 0, 0, 0x3680, "R4: subnormal * subnormal = 2^-18"),
    (0x07, 0x07, 0, 0, 0x3944, "R4: max subnormal squared, 49*2^-18"),
    (0x08, 0x38, 0, 0, 0x3C80, "R4: min E4M3FN normal * 1"),
    (0x3C, 0x7B, 1, 1, 0x4760, "R5: E5M2 1 * 57344"),
    (0x01, 0x01, 1, 1, 0x2F80, "R5: min E5M2 subnormals, 2^-32"),
    (0x7B, 0x7B, 1, 1, 0x4F44, "R5: E5M2 max finite squared"),
    (0x38, 0x3C, 0, 1, 0x3F80, "R6: mixed formats, 1*1"),
    (0x7E, 0x7B, 0, 1, 0x4BC4, "R6: E4M3FN max * E5M2 max"),
    (0x7F, 0x38, 0, 0, 0x7FC0, "R4, R7: E4M3FN NaN 0x7F"),
    (0xFF, 0x00, 0, 0, 0x7FC0, "R4, R7: E4M3FN NaN 0xFF * 0"),
    (0x7D, 0x3C, 1, 1, 0x7FC0, "R5, R7: E5M2 NaN"),
    (0x7C, 0x00, 1, 1, 0x7FC0, "R7: inf * +0"),
    (0x80, 0xFC, 0, 1, 0x7FC0, "R7: -0 * -inf"),
    (0x7C, 0x3C, 1, 1, 0x7F80, "R8: +inf * 1"),
    (0x7C, 0xBC, 1, 1, 0xFF80, "R8: +inf * -1"),
    (0xFC, 0x01, 1, 1, 0xFF80, "R8: -inf * min subnormal"),
    (0x80, 0x38, 0, 0, 0x8000, "R9: -0 * 1 = -0"),
    (0x80, 0xB8, 0, 0, 0x0000, "R9: -0 * -1 = +0"),
    (0x00, 0x7E, 0, 0, 0x0000, "R9: +0 * max"),
]


# Adder (accumulate) worked examples and boundaries: (acc, prod, expected, why)
DIRECTED_ADD = [
    (0x4380, 0x3F80, 0x4380, "R3: 256 + 1 tie -> even 256"),
    (0x4380, 0x4000, 0x4381, "R3: 256 + 2 exact 258"),
    (0x4381, 0x3F80, 0x4382, "R3: 258 + 1 tie -> even 260"),
    (0x4380, 0x3F81, 0x4381, "R3: 256 + (1+2^-7): sticky breaks the tie, up to 258"),
    (0x4F80, 0x3F80, 0x4F80, "R3: 2^32 + 1, far alignment, all sticky"),
    (0x3F81, 0xBF80, 0x3C00, "R3: catastrophic cancellation -> 2^-7"),
    (0x3F80, 0xBF80, 0x0000, "R9: 1 + -1 = +0"),
    (0x0000, 0x8000, 0x0000, "R9: +0 + -0 = +0"),
    (0x8000, 0x0000, 0x0000, "R9: -0 + +0 = +0"),
    (0x8000, 0x8000, 0x8000, "R9 (unit): -0 + -0 = -0"),
    (0x0000, 0xBF80, 0xBF80, "R9: +0 + -1 = -1"),
    (0x0081, 0x8080, 0x0000, "R9 (unit): FTZ positive tiny difference"),
    (0x8081, 0x0080, 0x8000, "R9 (unit): FTZ keeps negative sign"),
    (0x0001, 0x3F80, 0x3F80, "subnormal accumulator input reads as zero"),
    (0x7F80, 0x3F80, 0x7F80, "R8: +inf + finite"),
    (0x3F80, 0xFF80, 0xFF80, "R8: finite + -inf"),
    (0x7F80, 0x7F80, 0x7F80, "R8: +inf + +inf"),
    (0x7F80, 0xFF80, 0x7FC0, "R7: +inf + -inf = NaN"),
    (0x7F81, 0x3F80, 0x7FC0, "R7: NaN payload canonicalized"),
    (0xFFC0, 0x0000, 0x7FC0, "R7: negative NaN canonicalized"),
    (0x3F80, 0x7FC0, 0x7FC0, "R7: NaN product"),
    (0x7F7F, 0x7B00, 0x7F80, "R8 (unit): max + 2^119 tie -> inf"),
    (0x7F7F, 0x7A80, 0x7F7F, "R8 (unit): max + 2^118 stays finite"),
    (0xFF7F, 0xFB00, 0xFF80, "R8 (unit): negative overflow -> -inf"),
]


def rand_bf16(rng: random.Random) -> int:
    """Uniform BF16 bit pattern (any class, unit-level adder testing)."""
    return rng.getrandbits(16)


def rand_product(rng: random.Random) -> int:
    """A BF16 value the multiplier can actually produce (port-reachable)."""
    return mul(*rand_operands(rng))


def boundary_add(rng: random.Random):
    """(acc, prod) aimed at adder thresholds: near-cancellation, alignment
    distances 0..12 around the guard/round/sticky edges, rounding ties,
    overflow and FTZ edges (R3, R8, R9)."""
    kind = rng.randrange(5)
    if kind == 0:                                  # near cancellation
        a = rng.getrandbits(16) & 0x7FFF
        a = max(0x0080, min(a, 0x7F7F))
        b = (a ^ 0x8000) + rng.choice([-2, -1, 0, 1, 2])
        return a, b & 0xFFFF
    if kind == 1:                                  # alignment distance d
        a = rng.randrange(0x0A00, 0x7500)
        d = rng.randrange(0, 13)
        b = (a - (d << 7) + rng.randrange(-64, 64)) & 0x7FFF
        return a ^ (rng.getrandbits(1) << 15), b ^ (rng.getrandbits(1) << 15)
    if kind == 2:                                  # rounding ties: d = 8
        a = rng.randrange(0x0A00, 0x7500)
        b = (a - (8 << 7)) & 0x7F80                # half-ulp-sized addend
        return a, b ^ (rng.getrandbits(1) << 15)
    if kind == 3:                                  # overflow edge
        a = 0x7F00 | rng.randrange(0x60, 0x80)
        b = rng.randrange(0x7900, 0x7F80)
        s = rng.getrandbits(1) << 15
        return a | s, b | s
    a = rng.randrange(0x0080, 0x0200)              # FTZ edge
    b = (a + rng.randrange(-8, 8)) ^ 0x8000
    return a, b & 0xFFFF


# ---------------------------------------------------------------- self-checks
def _float_path_mul(a, b, fa, fb):
    """Independent second path for finite products: binary64 multiply, then
    BF16 via float32 bit pattern (exact: <= 8 significant bits)."""
    ca, sa, va = decode_fp8(a, fa)
    cb, sb, vb = decode_fp8(b, fb)
    p = float(va) * float(vb) * (-1 if sa ^ sb else 1)
    if p == 0:
        return (sa ^ sb) << 15
    bits = struct.unpack(">I", struct.pack(">f", p))[0]
    assert bits & 0xFFFF == 0, "product needs more than BF16 precision"
    return bits >> 16


if __name__ == "__main__":
    import collections
    import doctest
    doctest.testmod(verbose=False)
    for a, b, fa, fb, exp, why in DIRECTED_MUL:
        assert mul(a, b, fa, fb) == exp, (hex(a), hex(b), fa, fb, why, hex(mul(a, b, fa, fb)))
    # Exhaustive cross-check of the finite product path against binary64.
    n = 0
    for fa in (0, 1):
        for fb in (0, 1):
            for a in range(256):
                for b in range(256):
                    if decode_fp8(a, fa)[0] == NUM and decode_fp8(b, fb)[0] == NUM:
                        assert mul(a, b, fa, fb) == _float_path_mul(a, b, fa, fb)
                        n += 1
    print(f"float-path cross-check: {n} finite pairs agree")
    # Outcome-class distribution of the generators.
    rng = random.Random(1)

    def cls(x):
        if x == BF16_NAN: return "nan"
        if x & 0x7FFF == 0x7F80: return "inf"
        if x & 0x7FFF == 0: return "-zero" if x >> 15 else "+zero"
        return "neg" if x >> 15 else "pos"
    for name, gen in [("rand", rand_operands), ("boundary", boundary_operands)]:
        c = collections.Counter(cls(mul(*gen(rng))) for _ in range(5000))
        print(name, dict(c))
    rng = random.Random(2)
    c = collections.Counter()
    for _ in range(300):
        A, B, fa, fb = rand_matmul(rng, rng.randint(1, 64))
        for row in matmul(A, B, fa, fb):
            c.update(cls(x) for x in row)
    print("matmul (finite)", dict(c))
