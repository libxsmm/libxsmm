/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_MATH_H
#define LIBXSMM_MATH_H

#include "libxsmm_typedefs.h"

/** Helper macro to setup a matrix with some initial values. */
#define LIBXSMM_MATINIT_AUX(OMP, TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE) do { \
  /*const*/ double libxsmm_matinit_seed_ = SEED; /* avoid constant conditional */ \
  const double libxsmm_matinit_scale_ = libxsmm_matinit_seed_ * (SCALE) + (SCALE); \
  const libxsmm_blasint libxsmm_matinit_nrows_ = (libxsmm_blasint)(NROWS); \
  const libxsmm_blasint libxsmm_matinit_ncols_ = (libxsmm_blasint)(NCOLS); \
  const libxsmm_blasint libxsmm_matinit_ld_ = (libxsmm_blasint)(LD); \
  libxsmm_blasint libxsmm_matinit_i_ = 0, libxsmm_matinit_j_ = 0; \
  LIBXSMM_OMP_VAR(libxsmm_matinit_i_); LIBXSMM_OMP_VAR(libxsmm_matinit_j_); \
  if (0 != libxsmm_matinit_seed_) { \
    OMP(parallel for private(libxsmm_matinit_i_, libxsmm_matinit_j_)) \
    for (libxsmm_matinit_i_ = 0; libxsmm_matinit_i_ < libxsmm_matinit_ncols_; ++libxsmm_matinit_i_) { \
      for (libxsmm_matinit_j_ = 0; libxsmm_matinit_j_ < libxsmm_matinit_nrows_; ++libxsmm_matinit_j_) { \
        const libxsmm_blasint libxsmm_matinit_k_ = libxsmm_matinit_i_ * libxsmm_matinit_ld_ + libxsmm_matinit_j_; \
        ((TYPE*)(DST))[libxsmm_matinit_k_] = (TYPE)(libxsmm_matinit_scale_ * (1.0 + \
          (double)libxsmm_matinit_i_ * libxsmm_matinit_nrows_ + libxsmm_matinit_j_)); \
      } \
      for (; libxsmm_matinit_j_ < libxsmm_matinit_ld_; ++libxsmm_matinit_j_) { \
        const libxsmm_blasint libxsmm_matinit_k_ = libxsmm_matinit_i_ * libxsmm_matinit_ld_ + libxsmm_matinit_j_; \
        ((TYPE*)(DST))[libxsmm_matinit_k_] = (TYPE)libxsmm_matinit_seed_; \
      } \
    } \
  } \
  else { /* shuffle based initialization */ \
    const libxsmm_blasint libxsmm_matinit_maxval_ = libxsmm_matinit_ncols_ * libxsmm_matinit_ld_; \
    const TYPE libxsmm_matinit_maxval2_ = (TYPE)((libxsmm_blasint)LIBXSMM_UPDIV(libxsmm_matinit_maxval_, 2)); /* non-zero */ \
    const TYPE libxsmm_matinit_inv_ = ((TYPE)(SCALE)) / libxsmm_matinit_maxval2_; \
    const size_t libxsmm_matinit_shuffle_ = libxsmm_coprime2((size_t)libxsmm_matinit_maxval_); \
    OMP(parallel for private(libxsmm_matinit_i_, libxsmm_matinit_j_)) \
    for (libxsmm_matinit_i_ = 0; libxsmm_matinit_i_ < libxsmm_matinit_ncols_; ++libxsmm_matinit_i_) { \
      for (libxsmm_matinit_j_ = 0; libxsmm_matinit_j_ < libxsmm_matinit_ld_; ++libxsmm_matinit_j_) { \
        const libxsmm_blasint libxsmm_matinit_k_ = libxsmm_matinit_i_ * libxsmm_matinit_ld_ + libxsmm_matinit_j_; \
        ((TYPE*)(DST))[libxsmm_matinit_k_] = libxsmm_matinit_inv_ * /* normalize values to an interval of [-1, +1] */ \
          ((TYPE)(libxsmm_matinit_shuffle_ * libxsmm_matinit_k_ % libxsmm_matinit_maxval_) - libxsmm_matinit_maxval2_); \
      } \
    } \
  } \
} while(0)

#define LIBXSMM_MATINIT(TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE) \
  LIBXSMM_MATINIT_AUX(LIBXSMM_ELIDE, TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE)
#define LIBXSMM_MATINIT_SEQ(TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE) \
  LIBXSMM_MATINIT(TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE)
#define LIBXSMM_MATINIT_OMP(TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE) \
  LIBXSMM_MATINIT_AUX(LIBXSMM_PRAGMA_OMP, TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE)

/** GEMM exercising the compiler's code generation. TODO: only NN is supported and SP/DP matrices. */
#define LIBXSMM_INLINE_XGEMM(ITYPE, OTYPE, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) do { \
  /* Use 'n' (instead of 'N') avoids warning about "no macro replacement within a character constant". */ \
  const char libxsmm_inline_xgemm_transa_ = (char)(NULL != ((void*)(TRANSA)) ? (*(const char*)(TRANSA)) : \
    (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & LIBXSMM_FLAGS) ? 'n' : 't')); \
  const char libxsmm_inline_xgemm_transb_ = (char)(NULL != ((void*)(TRANSB)) ? (*(const char*)(TRANSB)) : \
    (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & LIBXSMM_FLAGS) ? 'n' : 't')); \
  const libxsmm_blasint libxsmm_inline_xgemm_m_ = *(const libxsmm_blasint*)(M); /* must be specified */ \
  const libxsmm_blasint libxsmm_inline_xgemm_k_ = (NULL != ((void*)(K)) ? (*(const libxsmm_blasint*)(K)) : libxsmm_inline_xgemm_m_); \
  const libxsmm_blasint libxsmm_inline_xgemm_n_ = (NULL != ((void*)(N)) ? (*(const libxsmm_blasint*)(N)) : libxsmm_inline_xgemm_k_); \
  const libxsmm_blasint libxsmm_inline_xgemm_lda_ = (NULL != ((void*)(LDA)) ? (*(const libxsmm_blasint*)(LDA)) : \
    (('n' == libxsmm_inline_xgemm_transa_ || *"N" == libxsmm_inline_xgemm_transa_) ? libxsmm_inline_xgemm_m_ : libxsmm_inline_xgemm_k_)); \
  const libxsmm_blasint libxsmm_inline_xgemm_ldb_ = (NULL != ((void*)(LDB)) ? (*(const libxsmm_blasint*)(LDB)) : \
    (('n' == libxsmm_inline_xgemm_transb_ || *"N" == libxsmm_inline_xgemm_transb_) ? libxsmm_inline_xgemm_k_ : libxsmm_inline_xgemm_n_)); \
  const libxsmm_blasint libxsmm_inline_xgemm_ldc_ = (NULL != ((void*)(LDC)) ? (*(const libxsmm_blasint*)(LDC)) : libxsmm_inline_xgemm_m_); \
  const OTYPE libxsmm_inline_xgemm_alpha_ = (NULL != ((void*)(ALPHA)) ? (*(const OTYPE*)(ALPHA)) : ((OTYPE)LIBXSMM_ALPHA)); \
  const OTYPE libxsmm_inline_xgemm_beta_  = (NULL != ((void*)(BETA))  ? (*(const OTYPE*)(BETA))  : ((OTYPE)LIBXSMM_BETA)); \
  libxsmm_blasint libxsmm_inline_xgemm_ni_, libxsmm_inline_xgemm_mi_ = 0, libxsmm_inline_xgemm_ki_; /* loop induction variables */ \
  LIBXSMM_ASSERT('n' == libxsmm_inline_xgemm_transa_ || *"N" == libxsmm_inline_xgemm_transa_); \
  LIBXSMM_ASSERT('n' == libxsmm_inline_xgemm_transb_ || *"N" == libxsmm_inline_xgemm_transb_); \
  LIBXSMM_PRAGMA_SIMD \
  for (libxsmm_inline_xgemm_mi_ = 0; libxsmm_inline_xgemm_mi_ < libxsmm_inline_xgemm_m_; ++libxsmm_inline_xgemm_mi_) { \
    LIBXSMM_PRAGMA_LOOP_COUNT(1, LIBXSMM_CONFIG_MAX_DIM, LIBXSMM_CONFIG_AVG_DIM) \
    for (libxsmm_inline_xgemm_ki_ = 0; libxsmm_inline_xgemm_ki_ < libxsmm_inline_xgemm_k_; ++libxsmm_inline_xgemm_ki_) { \
      LIBXSMM_PRAGMA_UNROLL \
      for (libxsmm_inline_xgemm_ni_ = 0; libxsmm_inline_xgemm_ni_ < libxsmm_inline_xgemm_n_; ++libxsmm_inline_xgemm_ni_) { \
        ((OTYPE*)(C))[libxsmm_inline_xgemm_ni_*libxsmm_inline_xgemm_ldc_+libxsmm_inline_xgemm_mi_] \
          = ((const ITYPE*)(B))[libxsmm_inline_xgemm_ni_*libxsmm_inline_xgemm_ldb_+libxsmm_inline_xgemm_ki_] * \
           (((const ITYPE*)(A))[libxsmm_inline_xgemm_ki_*libxsmm_inline_xgemm_lda_+libxsmm_inline_xgemm_mi_] * libxsmm_inline_xgemm_alpha_) \
          + ((const OTYPE*)(C))[libxsmm_inline_xgemm_ni_*libxsmm_inline_xgemm_ldc_+libxsmm_inline_xgemm_mi_] * libxsmm_inline_xgemm_beta_; \
      } \
    } \
  } \
} while(0)


/**
 * Structure of differences with matrix norms according
 * to http://www.netlib.org/lapack/lug/node75.html).
 */
LIBXSMM_EXTERN_C typedef struct libxsmm_matdiff_info {
  /** One-norm */         double norm1_abs, norm1_rel;
  /** Infinity-norm */    double normi_abs, normi_rel;
  /** Froebenius-norm */  double normf_rel;
  /** Maximum difference, L2-norm (absolute and relative), and R-squared. */
  double linf_abs, linf_rel, l2_abs, l2_rel, rsq;
  /** Statistics: sum/l1, min., max., arith. avg., and variance. */
  double l1_ref, min_ref, max_ref, avg_ref, var_ref;
  /** Statistics: sum/l1, min., max., arith. avg., and variance. */
  double l1_tst, min_tst, max_tst, avg_tst, var_tst;
  /* Values(v_ref, v_tst) and location(m, n) of largest linf_abs. */
  double v_ref, v_tst;
  /**
   * If r is non-zero (i is not negative), values (v_ref, v_tst),
   * and the location (m, n) stem from the i-th reduction
   * (r calls of libxsmm_matdiff_reduce) of the largest
   * difference (libxsmm_matdiff_epsilon).
   */
  libxsmm_blasint m, n, i, r;
} libxsmm_matdiff_info;

/**
 * Utility function to calculate a collection of scalar differences between two matrices (libxsmm_matdiff_info).
 * The location (m, n) of the largest difference (linf_abs) is recorded (also in case of NaN). In case of NaN,
 * differences are set to infinity. If no difference is discovered, the location (m, n) is negative (OOB).
 * The return value does not judge the difference (norm) between reference and test data, but is about
 * missing support for the requested data-type or otherwise invalid input.
 */
LIBXSMM_API int libxsmm_matdiff(libxsmm_matdiff_info* info,
  libxsmm_datatype datatype, libxsmm_blasint m, libxsmm_blasint n, const void* ref, const void* tst,
  const libxsmm_blasint* ldref, const libxsmm_blasint* ldtst);

/**
 * Combine absolute and relative norms into a value which can be used to check against a margin.
 * A file or directory path given per environment variable LIBXSMM_MATDIFF=/path/to/file stores
 * the epsilon (followed by a line-break), which can be used to calibrate margins of a test case.
 * LIBXSMM_MATDIFF can carry optional space-separated arguments used to amend the file entry.
 */
LIBXSMM_API double libxsmm_matdiff_epsilon(const libxsmm_matdiff_info* input);
/**
 * Reduces input into output such that the difference is maintained or increased (max function).
 * The very first (initial) output should be zeroed (libxsmm_matdiff_clear).
 */
LIBXSMM_API void libxsmm_matdiff_reduce(libxsmm_matdiff_info* output, const libxsmm_matdiff_info* input);
/** Clears the given info-structure, e.g., for the initial reduction-value (libxsmm_matdiff_reduce). */
LIBXSMM_API void libxsmm_matdiff_clear(libxsmm_matdiff_info* info);

/** Greatest common divisor (corner case: the GCD of 0 and 0 is 1). */
LIBXSMM_API size_t libxsmm_gcd(size_t a, size_t b);
/** Least common multiple. */
LIBXSMM_API size_t libxsmm_lcm(size_t a, size_t b);

/**
 * This function finds prime-factors (up to 32) of an unsigned integer in ascending order, and
 * returns the number of factors found (zero if the given number is prime and unequal to two).
 */
LIBXSMM_API int libxsmm_primes_u32(unsigned int num, unsigned int num_factors_n32[]);

/** Co-prime R of N such that R <= MinCo (libxsmm_coprime2(0|1) == 0). */
LIBXSMM_API size_t libxsmm_coprime(size_t n, size_t minco);
/** Co-prime R of N such that R <= N/2 (libxsmm_coprime2(0|1) == 0). */
LIBXSMM_API size_t libxsmm_coprime2(size_t n);

/**
 * Minimizes the waste, if "a" can only be processed in multiples of "b".
 * The remainder r is such that ((i * b) % a) <= r with i := {1, ..., a}.
 * Return value of this function is (i * b) with i := {1, ..., a}.
 * Remainder and limit are considered for early-exit and relaxation.
 * If the remainder is not given (NULL), it is assumed to be zero.
 * For example: libxsmm_remainder(23, 8, NULL, NULL) => 184.
 */
LIBXSMM_API unsigned int libxsmm_remainder(unsigned int a, unsigned int b,
  /** Optional limit such that (i * b) <= limit or ((i * b) % a) <= r. */
  const unsigned int* limit,
  /** Optional remainder limiting ((i * b) % a) <= r. */
  const unsigned int* remainder);

/**
 * Divides the product into prime factors and selects factors such that the new product is within
 * the given limit (0/1-Knapsack problem), e.g., product=12=2*2*3 and limit=6 then result=2*3=6.
 * The limit is at least reached or exceeded with the minimal possible product (is_lower=true).
 */
LIBXSMM_API unsigned int libxsmm_product_limit(unsigned int product, unsigned int limit, int is_lower);

/* Kahan's summation returns accumulator += value and updates compensation. */
LIBXSMM_API double libxsmm_kahan_sum(double value, double* accumulator, double* compensation);

/** Convert BF8 to FP32 (scalar). */
LIBXSMM_API float libxsmm_convert_bf8_to_f32(libxsmm_bfloat8 in);
/** Convert HF8 to FP32 (scalar). */
LIBXSMM_API float libxsmm_convert_hf8_to_f32(libxsmm_hfloat8 in);
/** Convert BF16 to FP32 (scalar). */
LIBXSMM_API float libxsmm_convert_bf16_to_f32(libxsmm_bfloat16 in);
/** Convert FP16 to FP32 (scalar). */
LIBXSMM_API float libxsmm_convert_f16_to_f32(libxsmm_float16 in);

/** Convert FP32 to BF16 (scalar). */
LIBXSMM_API libxsmm_bfloat16 libxsmm_convert_f32_to_bf16_truncate(float in);
/** Convert FP32 to BF16 (scalar). */
LIBXSMM_API libxsmm_bfloat16 libxsmm_convert_f32_to_bf16_rnaz(float in);
/** Convert FP32 to BF16 (scalar). */
LIBXSMM_API libxsmm_bfloat16 libxsmm_convert_f32_to_bf16_rne(float in);
/* Convert FP32 to BF8 (scalar). */
LIBXSMM_API libxsmm_bfloat8 libxsmm_convert_f32_to_bf8_stochastic(float in,
  /** Random number may decide rounding direction (boolean/odd/even). */
  unsigned int seed);
/** Convert FP32 to BF8 (scalar). */
LIBXSMM_API libxsmm_bfloat8 libxsmm_convert_f32_to_bf8_rne(float in);
/** Convert FP16 to HF8 (scalar). */
LIBXSMM_API libxsmm_hfloat8 libxsmm_convert_f16_to_hf8_rne(libxsmm_float16 in);
/** Convert FP32 to HF8 (scalar). */
LIBXSMM_API libxsmm_hfloat8 libxsmm_convert_f32_to_hf8_rne(float in);
/** Convert FP32 to FP16 (scalar). */
LIBXSMM_API libxsmm_float16 libxsmm_convert_f32_to_f16(float in);

/**
 * create a new external state for thread-save execution managed
 * by the user. We do not provide a function for drawing the random numbers
 * the user is supposed to call the LIBXSMM_INTRINSICS_MM512_RNG_EXTSTATE_PS
 * or LIBXSMM_INTRINSICS_MM512_RNG_XOSHIRO128P_EXTSTATE_EPI32 intrinsic.
 * */
LIBXSMM_API unsigned int* libxsmm_rng_create_extstate(unsigned int/*uint32_t*/ seed);

/**
 * return the size of the state such that users can save it
 * and recreate the same sequence of PRNG numbers.
 */
LIBXSMM_API unsigned int libxsmm_rng_get_extstate_size(void);

/** free a previously created rng_avx512_extstate */
LIBXSMM_API void libxsmm_rng_destroy_extstate(unsigned int* stateptr);

/** Set the seed of libxsmm_rng_* (similar to srand). */
LIBXSMM_API void libxsmm_rng_set_seed(unsigned int/*uint32_t*/ seed);

/**
 * This SP-RNG is using xoshiro128+ 1.0, work done by
 * David Blackman and Sebastiano Vigna (vigna @ acm.org).
 * It is their best and fastest 32-bit generator for
 * 32-bit floating-point numbers. They suggest to use
 * its upper bits for floating-point generation, what
 * we do here and generate numbers in [0,1(.
 */
LIBXSMM_API void libxsmm_rng_f32_seq(float* rngs, libxsmm_blasint count);

/**
 * Returns a (pseudo-)random value based on rand/rand48 in the interval [0, n).
 * This function compensates for an n, which is not a factor of RAND_MAX.
 * Note: libxsmm_rng_set_seed must be used if one wishes to seed the generator.
 */
LIBXSMM_API unsigned int libxsmm_rng_u32(unsigned int n);

/** SQRT with Newton's method using integer arithmetic. */
LIBXSMM_API unsigned int libxsmm_isqrt_u64(unsigned long long x);
/** SQRT with Newton's method using integer arithmetic. */
LIBXSMM_API unsigned int libxsmm_isqrt_u32(unsigned int x);
/** Based on libxsmm_isqrt_u32; result is factor of x. */
LIBXSMM_API unsigned int libxsmm_isqrt2_u32(unsigned int x);
/** SQRT with Newton's method using double-precision. */
LIBXSMM_API double libxsmm_dsqrt(double x);
/** SQRT with Newton's method using single-precision. */
LIBXSMM_API float libxsmm_ssqrt(float x);

#endif /*LIBXSMM_MATH_H*/
