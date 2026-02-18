import triton
import triton.language as tl
DEFAULT_BLOCK_T = 16
DEFAULT_BLOCK_N = 128

@triton.jit
def mamba3_complex_scan_interleaved_kernel(x_ptr, out_ptr, a_ptr, b_ptr, state_ptr, n_lanes, BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    Interleaved complex scan kernel for Mamba-3 style SSM:
      h_t = a * h_{t-1} + b * x_t

    Layout:
      x/out   : [n_lanes, BLOCK_T, 2] interleaved (real, imag)
      a/b     : [n_lanes, 2] interleaved (real, imag)
      state   : [n_lanes, 2] interleaved (real, imag)

    Internals are float32 for stability, output/state are cast on store.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < n_lanes
    s_base = offs * 2
    hr = tl.load(state_ptr + s_base + 0, mask=mask, other=0).to(tl.float32)
    hi = tl.load(state_ptr + s_base + 1, mask=mask, other=0).to(tl.float32)
    ar = tl.load(a_ptr + s_base + 0, mask=mask, other=0).to(tl.float32)
    ai = tl.load(a_ptr + s_base + 1, mask=mask, other=0).to(tl.float32)
    br = tl.load(b_ptr + s_base + 0, mask=mask, other=0).to(tl.float32)
    bi = tl.load(b_ptr + s_base + 1, mask=mask, other=0).to(tl.float32)
    for t in tl.static_range(BLOCK_T):
        base = (offs * BLOCK_T + t) * 2
        xr = tl.load(x_ptr + base + 0, mask=mask, other=0).to(tl.float32)
        xi = tl.load(x_ptr + base + 1, mask=mask, other=0).to(tl.float32)
        nhr = ar * hr - ai * hi + br * xr - bi * xi
        nhi = ar * hi + ai * hr + br * xi + bi * xr
        hr = nhr
        hi = nhi
        tl.store(out_ptr + base + 0, hr, mask=mask)
        tl.store(out_ptr + base + 1, hi, mask=mask)
    tl.store(state_ptr + s_base + 0, hr, mask=mask)
    tl.store(state_ptr + s_base + 1, hi, mask=mask)
