import triton
import triton.language as tl


@triton.jit
def gnprox_update_kernel(
    p_ptr,
    g_ptr,
    m_ptr,
    h_ptr,
    n_elements,
    lr,
    beta1,
    beta2,
    weight_decay,
    bc1_inv,
    bc2_inv,
    eps,
    damping,
    damping_gain,
    clip_grad,
    BLOCK_SIZE: tl.constexpr,
    NS_STEPS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    p = tl.load(p_ptr + offs, mask=mask, other=0).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask, other=0).to(tl.float32)
    m = tl.load(m_ptr + offs, mask=mask, other=0).to(tl.float32)
    h = tl.load(h_ptr + offs, mask=mask, other=0).to(tl.float32)

    if clip_grad > 0:
        g = tl.maximum(tl.minimum(g, clip_grad), -clip_grad)

    m = beta1 * m + (1.0 - beta1) * g
    h = beta2 * h + (1.0 - beta2) * (g * g)
    m_hat = m * bc1_inv
    h_hat = h * bc2_inv

    curvature = tl.sqrt(h_hat + eps)
    stiff = tl.minimum(tl.abs(m_hat) / (curvature + eps), 8.0)
    lam = damping * (1.0 + (damping_gain * stiff))
    a = h_hat + lam + eps

    expo = tl.floor(tl.log2(a))
    x = tl.exp2(-expo)
    for _ in tl.static_range(NS_STEPS):
        x = x * (2.0 - (a * x))

    step = m_hat * x
    p = p * (1.0 - (lr * weight_decay)) - (lr * step)

    tl.store(m_ptr + offs, m, mask=mask)
    tl.store(h_ptr + offs, h, mask=mask)
    tl.store(p_ptr + offs, p, mask=mask)
