// src/kernels/cuda/portable_expf.cuh - an exp(x) that rounds the same on NVIDIA and AMD.
//
// WHY.  `__expf` is the hardware approximation: ex2.approx on NVIDIA, v_exp_f32 on AMD.  Both are within ~2 ulp of
// exp, but not of each other: over 2^24 inputs in [-16, 16] an RTX 4060 and a Radeon Pro V620 (gfx1030) disagree
// on 49% of __expf results (max 2 ulp) and on 16% of the SwiGLU products built on them (max 4 ulp).  The expert
// kernels quantize that product to q8_1, which absorbs most of it, but about one routed entry in 2000 lands on a
// rounding edge and then its whole output row differs.  A HIP helper next to CUDA cards is meant to return the
// CUDA kernels' bytes, so with STRATA_PORTABLE_EXPF the expert SwiGLU uses this function on both backends.
//
// HOW.  Only IEEE-rounded operations that both compilers must honour: a round-to-nearest-even reduction
// n = rint(x / ln 2), r = x - n ln 2 in two fmaf steps (Cody-Waite), Cephes' degree-5 polynomial evaluated with
// fmaf, then a scale by 2^n as an exact power-of-two product (split in two near the ends of the range).  No
// operation is left for the compiler to contract: CUDA takes the _rn intrinsics, HIP turns contraction off.
// Accuracy: within 1 ulp of exp over the finite range (better than __expf); x > 88.72 gives +inf, x < -103.97 0.
#pragma once

#if defined(__HIPCC__)
#define STRATA_PX_MUL(a, b) ((a) * (b))
#define STRATA_PX_ADD(a, b) ((a) + (b))
#else
#define STRATA_PX_MUL(a, b) __fmul_rn((a), (b))
#define STRATA_PX_ADD(a, b) __fadd_rn((a), (b))
#endif

__device__ __forceinline__ float strata_portable_expf(float x) {
#if defined(__HIPCC__)
#pragma clang fp contract(off)
#endif
    if (x > 88.72283905206835f) return __int_as_float(0x7f800000);
    if (x < -103.97207708f) return 0.0f;
    if (x != x) return x;
    const float n = rintf(STRATA_PX_MUL(x, 1.44269504088896341f));
    float r = fmaf(-n, 0.693359375f, x);          // ln 2, high part (exact product with any n in range)
    r = fmaf(-n, -2.12194440e-4f, r);             // ln 2, low part
    float p = 1.9875691500e-4f;
    p = fmaf(p, r, 1.3981999507e-3f);
    p = fmaf(p, r, 8.3334519073e-3f);
    p = fmaf(p, r, 4.1665795894e-2f);
    p = fmaf(p, r, 1.6666665459e-1f);
    p = fmaf(p, r, 5.0000001201e-1f);
    const float r2 = STRATA_PX_MUL(r, r);
    float y = STRATA_PX_ADD(fmaf(r2, p, r), 1.0f);
    int k = (int) n;
    if (k > 127) { y = STRATA_PX_MUL(y, 2.0f); k -= 1; }
    if (k < -126) { y = STRATA_PX_MUL(y, __int_as_float((-126 + 127) << 23)); k += 126; }
    return STRATA_PX_MUL(y, __int_as_float((k + 127) << 23));
}

// The expert SwiGLU's exponential: the portable one with STRATA_PORTABLE_EXPF, else the hardware approximation.
#if defined(STRATA_PORTABLE_EXPF)
#define STRATA_SWIGLU_EXPF(x) strata_portable_expf(x)
#else
#define STRATA_SWIGLU_EXPF(x) __expf(x)
#endif
