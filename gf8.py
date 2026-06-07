"""
GF(8) arithmetic module.
Field: GF(8) = GF(2)[x] / (x^3 + x + 1), polynomial 0b1011 = 11
Elements are integers 0..7 representing polynomial coefficients in binary.
Addition: XOR
Multiplication: carry-less multiply mod x^3 + x + 1
"""

POLY = 0b1011  # x^3 + x + 1

def gf8_mul(a: int, b: int) -> int:
    """Multiply two GF(8) elements."""
    result = 0
    for _ in range(3):
        if b & 1:
            result ^= a
        a <<= 1
        if a & 0b1000:
            a ^= POLY
        b >>= 1
    return result


def build_mul_table() -> list[list[int]]:
    """Build the 8x8 GF(8) multiplication lookup table."""
    return [[gf8_mul(a, b) for b in range(8)] for a in range(8)]


def gf8_pow(base: int, exp: int) -> int:
    """Compute base^exp in GF(8)."""
    result = 1
    for _ in range(exp):
        result = gf8_mul(result, base)
    return result


# Precompute
MUL_TABLE = build_mul_table()
