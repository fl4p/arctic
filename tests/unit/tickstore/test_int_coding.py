"""Ported/adapted from jnb test/coding (which held only benchmark scripts, no pytest tests).

Covers arctic.tickstore.int_coding directly -- the varint + zigzag primitives the index/codec paths
build on. varint_decode_unsigned's 2nd arg is a bitmask (always (1<<64)-1 in real use), and it returns
bit-correct values; we compare via .astype(uint64). Values stay below 2**62, the realistic range for
cumulative ms/ns timestamps and zigzag-encoded deltas (zigzag needs one spare bit).
"""
import numpy as np

from arctic.tickstore.int_coding import (
    varint_encode_unsigned, varint_decode_unsigned,
    zigzag_encode_inplace, zigzag_decode_inplace,
)

MASK64 = (1 << 64) - 1


def _encode(arr):
    out = bytearray()
    varint_encode_unsigned(arr, out.extend)
    return bytes(out)


def test_varint_roundtrip_unsigned():
    arr = np.array([0, 1, 127, 128, 255, 256, 16383, 16384,
                    2 ** 32, 2 ** 53, 2 ** 62 - 1], dtype=np.uint64)
    out = varint_decode_unsigned(_encode(arr), MASK64)
    assert (out.astype(np.uint64) == arr).all()


def test_varint_roundtrip_random():
    rng = np.random.default_rng(0)
    for bits in (8, 16, 32, 62):
        arr = rng.integers(0, 2 ** bits, size=10_000, dtype=np.uint64, endpoint=False)
        out = varint_decode_unsigned(_encode(arr), MASK64)
        assert (out.astype(np.uint64) == arr).all()


def test_varint_empty():
    out = varint_decode_unsigned(_encode(np.array([], dtype=np.uint64)), MASK64)
    assert len(out) == 0


def test_zigzag_roundtrip():
    arr = np.array([0, -1, 1, -2, 2, -1000, 1000, 2 ** 40, -(2 ** 40)], dtype=np.int64)
    orig = arr.copy()
    zigzag_encode_inplace(arr)
    assert (arr >= 0).all()
    zigzag_decode_inplace(arr)
    assert (arr == orig).all()


def test_zigzag_known_values():
    arr = np.array([0, -1, 1, -2, 2], dtype=np.int64)
    zigzag_encode_inplace(arr)
    assert (arr == np.array([0, 1, 2, 3, 4], dtype=np.int64)).all()


def test_zigzag_then_varint_roundtrip():
    # the delta-coding pipeline: zigzag signed deltas -> varint encode -> decode -> un-zigzag
    deltas = np.array([0, 5, -3, 1000, -1, 42, -100000], dtype=np.int64)
    z = deltas.copy()
    zigzag_encode_inplace(z)
    out = varint_decode_unsigned(_encode(z.astype(np.uint64)), MASK64).astype(np.int64)
    zigzag_decode_inplace(out)
    assert (out == deltas).all()
