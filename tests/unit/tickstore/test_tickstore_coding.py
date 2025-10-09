import time

import numpy as np

def _VarintDecoder(mask, result_type):
  """Return an encoder for a basic varint value (does not include tag).

  Decoded values will be bitwise-anded with the given mask before being
  returned, e.g. to limit them to 32 bits.  The returned decoder does not
  take the usual "end" parameter -- the caller is expected to do bounds checking
  after the fact (often the caller can defer such checking until later).  The
  decoder returns a (value, new_pos) pair.
  """

  def DecodeVarint(buffer, pos: int=None):
    result = 0
    shift = 0
    while 1:
      b = buffer[pos]
      pos += 1
      result |= ((b & 0x7f) << shift)
      if not (b & 0x80):
        result &= mask
        result = result_type(result)
        return result if pos is None else (result, pos)
      shift += 7
      if shift >= 64:
        raise ValueError('Too many bytes when decoding varint.')

  return DecodeVarint

def test_varint_unsigned():
    from arctic.tickstore.coding import nparray_varint_encode, nparray_varint_decode

    refDec =  _VarintDecoder((1 << 64) - 1, int)

    i = np.cumsum([int(time.time()), 1,1,1,1]).astype(np.uint64)
    buf = nparray_varint_encode(i)
    dec = nparray_varint_decode(buf)
    assert np.all(i == dec)
    assert np.all(i[0] == refDec(buf, 0)[0])

    i = np.cumsum([int(time.time()*1000), 1,1,1,1]).astype(np.uint64)
    assert np.all(i == i.astype(np.int64))
    buf = nparray_varint_encode(i)
    dec = nparray_varint_decode(buf)
    assert np.all(i == dec)

    i = np.cumsum([int(time.time()*1e9), 1,1,1,1]).astype(np.uint64)
    assert np.all(i == i.astype(np.int64))
    buf = nparray_varint_encode(i)
    dec = nparray_varint_decode(buf)
    assert np.all(i == dec)


def test_LnQ25VQLgz():
    #a = np.array([1.5824204e+28, 2.4443570e+27], dtype=np.float32)
    a = np.array([2.2e+27, 1e-12,  1.4563e-12, 1e-13, 1.1e-14], dtype=np.float32)
    from arctic.tickstore.tickstore import codec_registry
    code = codec_registry['LnQ25VQLgz']
    b = code.encode(a)
    d = code.decode(b)
    rtol = max(abs(d - a) / abs(a))
    assert rtol < code.rtol_max

    from arctic.tickstore.coding import LnQ16_VQL
    code2 = LnQ16_VQL(loq_loss=25, comp='zlib', log_prescale=28)
    a = np.array([1.2e+30, 1e-12], dtype=np.float32)
    b = code2.encode(a)
    d = code2.decode(b)
    rtol = max(abs(d - a) / abs(a))
    assert rtol < code2.rtol_max, rtol


    from arctic.tickstore.coding import LnQ16_VQL
    code2 = LnQ16_VQL(loq_loss=25, comp='zlib', log_prescale=24, loq_preadd=1e-8)
    a = np.array([2e+31, 1e-15, 1e-10, 0.00453, 1234.54], dtype=np.float32)
    b = code2.encode(a)
    d = code2.decode(b)
    rtol = max(abs(d - a) / abs(a))
    assert rtol < code2.rtol_max, rtol