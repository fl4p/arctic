import io
import struct

import numpy as np

#import pyximport  # cython
#pyximport.install(
#    setup_args={"include_dirs": np.get_include()},
#    reload_support=True,
#    language_level=3
#)

from .int_coding import varint_encode, varint_decode_unsigned

def nparray_varint_encode(arr:np.array):
    iob = io.BytesIO()
    varint_encode(arr, iob.write)
    iob.seek(0)
    return iob.read(-1)

def nparray_varint_decode(buf):
    return varint_decode_unsigned(buf, (1 << 64) - 1, int)

VARINT_MAX_BITS = 64


def ZigZagEncode(value: int):
    """
    Encode signed integers so that they can be stored more efficiently using varint coding.
    :param value:
    :return:
    """
    if value >= 0:
        return value << 1
    return (value << 1) ^ -1


def ZigZagDecode(value: int):
    if not value & 0x1:
        return value >> 1
    return (value >> 1) ^ -1


def _VarintEncoder(signed=True):
    pack_int2byte = struct.Struct('>B').pack

    def EncodeSignedVarint(write, value):
        if value < 0:
            if signed:
                value += (1 << VARINT_MAX_BITS)
            else:
                raise ValueError('positive integers only')
        bits = value & 0x7f
        value >>= 7
        while value:
            write(pack_int2byte(0x80 | bits))
            bits = value & 0x7f
            value >>= 7
        return write(pack_int2byte(bits))

    return EncodeSignedVarint

def _UnsignedVarintEncoder(signed=True):
    pack_int2byte = struct.Struct('>B').pack

    def EncodeSignedVarint(write, value):
        bits = value & 0x7f
        value >>= 7
        while value:
            write(pack_int2byte(0x80 | bits))
            bits = value & 0x7f
            value >>= 7
        return write(pack_int2byte(bits))

    return EncodeSignedVarint


def _UnsignedVarintDecoder(mask, result_type):
    """Return an encoder for a basic varint value (does not include tag).

    Decoded values will be bitwise-anded with the given mask before being
    returned, e.g. to limit them to 32 bits.  The returned decoder does not
    take the usual "end" parameter -- the caller is expected to do bounds checking
    after the fact (often the caller can defer such checking until later).  The
    decoder returns a (value, new_pos) pair.
    """

    def DecodeVarint(buffer, pos: int = None):
        result = 0
        shift = 0
        while 1:
            if pos is None:
                # Read from BytesIO
                try:
                    b = buffer.read(1)[0]
                except IndexError as e:
                    if shift == 0:
                        # End of BytesIO.
                        return None
                    else:
                        raise ValueError('Fail to read varint %s' % str(e))
            else:
                b = buffer[pos]
                pos += 1
            result |= ((b & 0x7f) << shift)
            if not (b & 0x80):
                result &= mask
                result = result_type(result)
                return result if pos is None else (result, pos)
            shift += 7
            if shift >= VARINT_MAX_BITS:
                raise ValueError('Too many bytes when decoding varint.')

    return DecodeVarint


def _SignedVarintDecoder(bits, result_type):
    signbit = 1 << (bits - 1)
    mask = (1 << bits) - 1

    def DecodeVarint(buffer, pos):
        result = 0
        shift = 0
        while 1:
            b = buffer[pos]
            result |= ((b & 0x7f) << shift)
            pos += 1
            if not (b & 0x80):
                result &= mask
                result = (result ^ signbit) - signbit
                result = result_type(result)
                return (result, pos)
            shift += 7
            if shift >= VARINT_MAX_BITS:
                raise ValueError('Too many bytes when decoding varint.')

    return DecodeVarint


UnsignedVarintEncode = _VarintEncoder(signed=False)
SignedVarintEncode = _VarintEncoder(signed=True)

UnsignedVarintDecode = _UnsignedVarintDecoder((1 << VARINT_MAX_BITS) - 1, int)
SignedVarintDecode = _SignedVarintDecoder(VARINT_MAX_BITS, int)
