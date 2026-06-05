"""
Rowmask pco benchmark on REAL L2 book data.

Loads a disk_cached arctic L2 read (a wide book DataFrame: bid/ask px+size per
level, plus flags), derives the real per-column rowmask exactly as
TickStore._to_bucket_pandas does (present = ~isnan(col)), slices it into
chunk_size-row buckets, and runs the same representations as
bench_rowmask_pco.py -- so we compare pco vs the lz4(packbits) baseline on the
actual mask distributions, including their real autocorrelation.

Usage:
  /Users/fab/dev/venvs/jnb314/bin/python3 benchmarks/pco_rowmask/bench_rowmask_real.py <pickle> [max_rows]

The pickle is the disk_cache payload: (DataFrame, expiry, meta) or a bare
DataFrame. max_rows caps how many leading rows are used (default 5_000_000 ->
50 buckets of 100k; enough for stable aggregates without loading masks for the
whole multi-year history into RAM).
"""
import pickle  # trusted: this is the user's own disk_cache payload, self-generated
import sys

import bench_rowmask_pco as B   # reuse REPRS + bench_mask from the synthetic script

CHUNK = 100_000  # TickStore default chunk_size


def load_df(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    # disk_cache stores (ret, expiry, meta); ret here is the DataFrame
    if isinstance(obj, tuple):
        obj = obj[0]
    return obj


def main():
    path = sys.argv[1]
    max_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 5_000_000

    print("loading %s ..." % path)
    df = load_df(path)
    n_all = len(df)
    n = min(n_all, max_rows)
    print("DataFrame: %d rows x %d cols (using first %d rows = %d full buckets)"
          % (n_all, df.shape[1], n, n // CHUNK))
    print("pco level=%d | brotli=%s zstd=%s" % (B.PCO_LEVEL, B.HAVE_BROTLI, B.HAVE_ZSTD))

    # Derive masks once per column (present = notna, == ~isnan for float cols),
    # then drop the values to keep memory bounded.
    cols = list(df.columns)
    masks = {}
    densities = {}
    for c in cols:
        m = df[c].iloc[:n].notna().to_numpy()
        masks[c] = m
        densities[c] = float(m.mean())
    del df

    n_buckets = n // CHUNK
    if n_buckets == 0:
        print("fewer than one full bucket; aborting")
        return

    # Aggregate totals per representation across (column x bucket).
    agg = {name: [0, 0.0, 0.0] for name, *_ in B.REPRS}
    # Also bucket masks by density band to show where pco wins.
    bands = [(0.0, 0.02), (0.02, 0.2), (0.2, 0.8), (0.8, 0.999), (0.999, 1.001)]
    band_agg = {b: {name: [0, 0] for name, *_ in B.REPRS} for b in bands}  # [bytes, count]

    print("\nper-column density (fraction of rows present):")
    for c in cols:
        print("  %-16s %.4f" % (str(c)[:16], densities[c]))

    n_masks = 0
    for c in cols:
        full = masks[c]
        for bi in range(n_buckets):
            mask = full[bi * CHUNK:(bi + 1) * CHUNK]
            if len(mask) < CHUNK:
                continue
            rows = B.bench_mask(mask)          # round-trip asserted inside
            n_masks += 1
            d = mask.mean()
            band = next(b for b in bands if b[0] <= d < b[1])
            for name, b, _bpr, te, td in rows:
                a = agg[name]
                a[0] += b; a[1] += te; a[2] += td
                ba = band_agg[band][name]
                ba[0] += b; ba[1] += 1

    base_name = B.REPRS[0][0]
    base = agg[base_name][0]
    print("\n" + "#" * 84)
    print("AGGREGATE over REAL L2 masks  (%d masks = %d cols x %d buckets, n=%d each)"
          % (n_masks, len(cols), n_buckets, CHUNK))
    print("-" * 84)
    print("%-30s %14s %9s %12s %12s" % ("representation", "tot bytes", "vs base", "enc us/mask", "dec us/mask"))
    for name, *_ in B.REPRS:
        b, te, td = agg[name]
        print("%-30s %14d %8.1f%% %12.1f %12.1f"
              % (name, b, 100.0 * b / base, te / n_masks, td / n_masks))

    print("\nby density band (total bytes; vs baseline):")
    hdr = "%-18s" % "band"
    for name, *_ in B.REPRS:
        hdr += " %14s" % name.split()[0][:14]
    print(hdr)
    for lo, hi in bands:
        ba = band_agg[(lo, hi)]
        cnt = ba[base_name][1]
        if cnt == 0:
            continue
        line = "%-18s" % ("[%.3f,%.3f) n=%d" % (lo, hi, cnt))
        bb = ba[base_name][0]
        for name, *_ in B.REPRS:
            v = ba[name][0]
            line += " %13s%%" % ("%.0f" % (100.0 * v / bb) if bb else "-")
        print(line)


if __name__ == "__main__":
    main()
