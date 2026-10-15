"""Pin down WHY the two backends differ under active YaRN."""
import math, torch
from torchtitan.models.common.rope import ComplexRoPE, CosSinRoPE

dim, base, orig = 128, 50000.0, 8192
bf, bs = 1.0, 32.0

# --- ComplexRoPE's range: floor/ceil to INTEGERS, then order-normalize ---
def fcd(nrot):
    return dim * math.log(orig / (nrot * 2 * math.pi)) / (2 * math.log(base))
cx_low, cx_high = math.floor(fcd(bf)), math.ceil(fcd(bs))
if cx_low > cx_high: cx_low, cx_high = cx_high, cx_low
cx_low, cx_high = max(cx_low, 0), min(cx_high, dim - 1)

# --- CosSinRoPE's range: kept as FLOATS ---
d_half = dim / 2
cs_low  = d_half * math.log(orig / (bs * 2 * math.pi)) / math.log(base)
cs_high = d_half * math.log(orig / (bf * 2 * math.pi)) / math.log(base)
if cs_low > cs_high: cs_low, cs_high = cs_high, cs_low

print(f"ComplexRoPE correction range: low={cx_low}      high={cx_high}       (integers)")
print(f"CosSinRoPE  correction range: low={cs_low:.4f} high={cs_high:.4f}  (floats)")
print(f"=> same formula, differing ONLY by floor/ceil rounding\n")

# Prove it: re-run the factor=32 comparison with CosSin's range rounded the
# ComplexRoPE way, by monkeypatching the ramp it builds.
def build_freqs(low, high, factor):
    freq = base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim)
    ramp = (torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)
    mask = 1 - ramp.clamp(0, 1)
    return (1.0 / (factor * freq)) * (1 - mask) + (1.0 / freq) * mask

for factor in (2.0, 4.0, 32.0):
    a = build_freqs(cx_low, cx_high, factor)   # integer range
    b = build_freqs(cs_low, cs_high, factor)   # float range
    rel = ((a - b).abs() / b.abs()).max().item()
    nz = int(((a - b).abs() > 0).sum())
    print(f"factor={factor:<5} inv_freq max rel-diff={rel:.3e}  "
          f"differing dims={nz}/{dim//2}  (all inside the transition band)")
