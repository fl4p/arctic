"""
Benchmark: is pcodec (`pco`) worth using for TickStore *rowmasks*?

TickStore stores, per column per chunk, a boolean ROWMASK marking which of the
bucket's ticks carry that column. It is currently persisted as

    lz4_compressHC(np.packbits(mask).tobytes())          # the baseline here

and the code comment in tickstore.py claims pco "cannot compress the packed-bit
bytes anyway" (true -- packed bits look like high-entropy noise to a numeric
codec). The interesting question is whether a *different, pco-friendly*
representation of the same mask (gap deltas, run lengths, raw 0/1 bytes) beats
the lz4(packbits) baseline on size and/or CPU.

This script compares, for each mask representation:
  - size (bytes, and bits per row)
  - encode / decode time
  - round-trip correctness (asserted)

across mask distributions that mirror real L2 tick data:
  - all-ones (the overwhelmingly common case: a column present on every tick)
  - dense-with-rare-gaps
  - sparse random (Bernoulli) at several densities
  - bursty / clustered (present in runs -- e.g. trade fields during activity)
  - masks derived from a synthetic wide L2 frame (NaN -> absent, the real
    `_to_bucket_pandas` rule: rm = ~np.isnan(val))

Run with a venv that has pcodec + lz4 (+ optional brotli/zstd):
  /Users/fab/dev/venvs/jnb314/bin/python3 benchmarks/pco_rowmask/bench_rowmask_pco.py
"""
import time

import numpy as np

# --- codecs -----------------------------------------------------------------
from lz4.block import compress as lz4_compress, decompress as lz4_decompress
from pcodec import standalone as pco, ChunkConfig

try:
    import brotli
    HAVE_BROTLI = True
except ImportError:
    HAVE_BROTLI = False

try:
    import zstandard as _zstd
    _ZC = _zstd.ZstdCompressor(level=19)
    _ZD = _zstd.ZstdDecompressor()
    HAVE_ZSTD = True
except ImportError:
    HAVE_ZSTD = False


def lz4_hc(b):
    return lz4_compress(b, mode='high_compression')


PCO_LEVEL = 8  # same level TickStore's pco_encode uses


def pco_c(arr, enable_8_bit=False):
    # pco refuses 8-bit dtypes unless enable_8_bit is set ("often a mistake").
    return pco.simple_compress(np.ascontiguousarray(arr),
                               ChunkConfig(compression_level=PCO_LEVEL, enable_8_bit=enable_8_bit))


def pco_d(buf):
    return pco.simple_decompress(buf)


# --- mask representations ---------------------------------------------------
# Each is (encode(mask)->bytes, decode(bytes, n)->mask). decode takes n because
# some layouts (gaps, the packbits tail) need the original length, exactly as
# _decode_bucket already knows doc_length from the index.

def enc_lz4_packbits(mask):              # *** the current TickStore baseline ***
    return lz4_hc(np.packbits(mask).tobytes())

def dec_lz4_packbits(buf, n):
    return np.unpackbits(np.frombuffer(lz4_decompress(buf), dtype='uint8'))[:n].astype(bool)


def enc_lz4_fast_packbits(mask):         # lz4 default level (cheaper encode)
    return lz4_compress(np.packbits(mask).tobytes())

def dec_lz4_fast_packbits(buf, n):
    return np.unpackbits(np.frombuffer(lz4_decompress(buf), dtype='uint8'))[:n].astype(bool)


def enc_brotli_packbits(mask):
    return brotli.compress(np.packbits(mask).tobytes(), quality=11, lgwin=24)

def dec_brotli_packbits(buf, n):
    return np.unpackbits(np.frombuffer(brotli.decompress(buf), dtype='uint8'))[:n].astype(bool)


def enc_zstd_packbits(mask):
    return _ZC.compress(np.packbits(mask).tobytes())

def dec_zstd_packbits(buf, n):
    return np.unpackbits(np.frombuffer(_ZD.decompress(buf), dtype='uint8'))[:n].astype(bool)


def enc_pco_packbits(mask):              # feed the packed *bytes* to pco (the comment's claim)
    return pco_c(np.packbits(mask), enable_8_bit=True)      # uint8 array

def dec_pco_packbits(buf, n):
    return np.unpackbits(pco_d(buf))[:n].astype(bool)


def enc_pco_bits_u8(mask):               # raw 0/1 bytes, one per row, through pco
    return pco_c(mask.astype(np.uint8), enable_8_bit=True)

def dec_pco_bits_u8(buf, n):
    return pco_d(buf).astype(bool)


def enc_pco_gaps(mask):
    # store set-bit positions as gaps (delta between consecutive set indices),
    # then pco the gap stream. pco's delta+bin-packing shines when gaps are
    # small/regular (dense) or low-entropy (clustered). uint32 -> chunk<=100k fits.
    pos = np.flatnonzero(mask).astype(np.int64)
    if len(pos) == 0:
        return b''                       # empty -> all-zeros mask
    gaps = np.empty(len(pos), dtype=np.uint32)
    gaps[0] = pos[0]
    gaps[1:] = np.diff(pos)
    return pco_c(gaps)

def dec_pco_gaps(buf, n):
    out = np.zeros(n, dtype=bool)
    if buf == b'':
        return out
    gaps = pco_d(buf)
    pos = np.cumsum(gaps.astype(np.int64))
    out[pos] = True
    return out


def enc_pco_runs(mask):
    # run-length: lengths of alternating runs. First run is the run of the
    # mask's first value. pco the run lengths (uint32). Great for bursty/dense.
    if len(mask) == 0:
        return np.uint8(0).tobytes()
    changes = np.flatnonzero(np.diff(mask.astype(np.int8)))
    bounds = np.concatenate(([0], changes + 1, [len(mask)]))
    runs = np.diff(bounds).astype(np.uint32)
    first = np.uint8(mask[0])
    return first.tobytes() + pco_c(runs)

def dec_pco_runs(buf, n):
    first = bool(np.frombuffer(buf[:1], dtype=np.uint8)[0])
    runs = pco_d(buf[1:])
    out = np.empty(n, dtype=bool)
    val = first
    i = 0
    for r in runs:
        out[i:i + r] = val
        i += int(r)
        val = not val
    return out


REPRS = [
    ("lz4HC(packbits)   [BASELINE]", enc_lz4_packbits, dec_lz4_packbits, True),
    ("lz4(packbits)",                enc_lz4_fast_packbits, dec_lz4_fast_packbits, True),
    ("pco(packbits u8)",            enc_pco_packbits, dec_pco_packbits, True),
    ("pco(bits u8)",                enc_pco_bits_u8, dec_pco_bits_u8, True),
    ("pco(gaps u32)",               enc_pco_gaps, dec_pco_gaps, True),
    ("pco(runlen u32)",             enc_pco_runs, dec_pco_runs, True),
]
if HAVE_BROTLI:
    REPRS.append(("brotli11(packbits)", enc_brotli_packbits, dec_brotli_packbits, False))
if HAVE_ZSTD:
    REPRS.append(("zstd19(packbits)", enc_zstd_packbits, dec_zstd_packbits, False))


# --- timing -----------------------------------------------------------------
def timeit(fn, *args, repeat=5):
    best = float('inf')
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return out, best


def bench_mask(mask):
    n = len(mask)
    rows = []
    for name, enc, dec, _core in REPRS:
        buf, te = timeit(enc, mask)
        back, td = timeit(dec, buf, n)
        ok = np.array_equal(back, mask)
        if not ok:
            raise AssertionError("round-trip FAILED for %s" % name)
        rows.append((name, len(buf), 8.0 * len(buf) / n, te * 1e6, td * 1e6))
    return rows


def fmt_scenario(title, mask):
    n = len(mask)
    dens = mask.mean() if n else 0.0
    print("\n" + "=" * 92)
    print("%s   (n=%d, density=%.4f, set=%d)" % (title, n, dens, mask.sum()))
    print("-" * 92)
    print("%-30s %10s %10s %12s %12s" % ("representation", "bytes", "bits/row", "enc us", "dec us"))
    rows = bench_mask(mask)
    base_bytes = rows[0][1]
    for name, b, bpr, te, td in rows:
        rel = "" if b == base_bytes else ("  (%+.0f%% vs base)" % (100.0 * (b - base_bytes) / max(base_bytes, 1)))
        print("%-30s %10d %10.4f %12.1f %12.1f%s" % (name, b, bpr, te, td, rel))
    return rows


# --- mask generators (mirror L2 tick data) ---------------------------------
def gen_scenarios(n=100_000, seed=0):
    rng = np.random.default_rng(seed)
    sc = []

    sc.append(("all-ones (dense column, the common case)", np.ones(n, dtype=bool)))

    m = np.ones(n, dtype=bool)
    m[rng.choice(n, size=n // 100, replace=False)] = False
    sc.append(("dense, 1% random gaps", m))

    for p in (0.30, 0.10, 0.05, 0.01):
        sc.append(("sparse random, density=%.2f" % p, rng.random(n) < p))

    # bursty / clustered: present in contiguous runs (trade fields during bursts).
    # Markov chain with sticky states -> long runs, ~given on-fraction.
    def bursty(p_on, mean_run):
        out = np.empty(n, dtype=bool)
        state = rng.random() < p_on
        i = 0
        # on/off run lengths geometric; tune off-runs to hit p_on
        mean_off = mean_run * (1 - p_on) / p_on
        while i < n:
            if state:
                r = max(1, int(rng.exponential(mean_run)))
            else:
                r = max(1, int(rng.exponential(mean_off)))
            out[i:i + r] = state
            i += r
            state = not state
        return out

    sc.append(("bursty, ~30% on, mean run 50", bursty(0.30, 50)))
    sc.append(("bursty, ~10% on, mean run 20", bursty(0.10, 20)))
    sc.append(("bursty, ~5% on, mean run 200", bursty(0.05, 200)))

    return sc


def gen_l2_frame_masks(n=100_000, seed=1):
    """Synthetic wide L2 frame; masks = ~isnan(col), the real _to_bucket_pandas rule.

    Models a depth-10 book + trade fields: top levels almost always present,
    deeper levels increasingly sparse, trade px/qty/side present only on trade
    ticks (bursty), an act-flag present on a subset.
    """
    rng = np.random.default_rng(seed)
    masks = []

    # book levels 0..9 for bid+ask: presence decays with depth, mildly bursty
    for side in ("bid", "ask"):
        for lvl in range(10):
            base = max(0.04, 1.0 - lvl * 0.11)           # L0 ~all present, deep ~sparse
            present = rng.random(n) < base
            masks.append(("%s_px_%d" % (side, lvl), present))
            # size mask tracks the price mask almost exactly (same update)
            masks.append(("%s_sz_%d" % (side, lvl), present & (rng.random(n) < 0.999)))

    # trades: bursty ~8% of ticks
    trade = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if rng.random() < 0.08:
            r = max(1, int(rng.exponential(8)))
            trade[i:i + r] = True
            i += r
        else:
            i += max(1, int(rng.exponential(40)))
    masks.append(("trade_px", trade.copy()))
    masks.append(("trade_qty", trade & (rng.random(n) < 0.995)))
    masks.append(("trade_side", trade.copy()))
    masks.append(("act_flag", rng.random(n) < 0.02))
    return masks


def summarize(all_rows, title):
    """Aggregate total bytes + total enc/dec time per representation across masks."""
    agg = {}
    for rows in all_rows:
        for name, b, _bpr, te, td in rows:
            a = agg.setdefault(name, [0, 0.0, 0.0])
            a[0] += b
            a[1] += te
            a[2] += td
    print("\n" + "#" * 92)
    print("AGGREGATE over %s (%d masks)" % (title, len(all_rows)))
    print("-" * 92)
    print("%-30s %12s %10s %12s %12s" % ("representation", "tot bytes", "vs base", "tot enc us", "tot dec us"))
    base = agg[REPRS[0][0]][0]
    for name, _e, _d, _c in REPRS:
        b, te, td = agg[name]
        print("%-30s %12d %9.1f%% %12.1f %12.1f" % (name, b, 100.0 * b / base, te, td))


def main():
    print("pcodec rowmask benchmark | pco level=%d | brotli=%s zstd=%s | numpy=%s"
          % (PCO_LEVEL, HAVE_BROTLI, HAVE_ZSTD, np.__version__))

    n = 100_000  # TickStore default chunk_size
    syn_rows = []
    for title, mask in gen_scenarios(n):
        syn_rows.append(fmt_scenario(title, mask))
    summarize(syn_rows, "synthetic distribution scenarios")

    print("\n\n" + "*" * 92)
    print("SYNTHETIC L2 FRAME (per-column masks, rule: present = ~isnan)")
    print("*" * 92)
    l2_rows = []
    for col, mask in gen_l2_frame_masks(n):
        l2_rows.append(fmt_scenario("col %-12s" % col, mask))
    summarize(l2_rows, "synthetic L2 frame columns")


if __name__ == "__main__":
    main()
