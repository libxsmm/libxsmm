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

#include "../libxsmm_typedefs.h"

/** Helper macro to setup a matrix with some initial values. */
#define LIBXSMM_MATINIT_AUX(OMP, TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE) do { \
  /*const*/ libxsmm_blasint libxsmm_matinit_seed_ = SEED; /* avoid constant conditional */ \
  const double libxsmm_matinit_scale_ = (SCALE) * libxsmm_matinit_seed_ + (SCALE); \
  const libxsmm_blasint libxsmm_matinit_nrows_ = (libxsmm_blasint)NROWS; \
  const libxsmm_blasint libxsmm_matinit_ld_ = (libxsmm_blasint)LD; \
  libxsmm_blasint libxsmm_matinit_i_ = 0, libxsmm_matinit_j_ = 0; \
  LIBXSMM_OMP_VAR(libxsmm_matinit_i_); LIBXSMM_OMP_VAR(libxsmm_matinit_j_); \
  if (0 != libxsmm_matinit_seed_) { \
    OMP(parallel for private(libxsmm_matinit_i_, libxsmm_matinit_j_)) \
    for (libxsmm_matinit_i_ = 0; libxsmm_matinit_i_ < ((libxsmm_blasint)NCOLS); ++libxsmm_matinit_i_) { \
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
    const unsigned int libxsmm_matinit_maxval_ = ((unsigned int)NCOLS) * ((unsigned int)libxsmm_matinit_ld_); \
    const TYPE libxsmm_matinit_maxval2_ = (TYPE)((unsigned int)LIBXSMM_UPDIV(libxsmm_matinit_maxval_, 2)); /* non-zero */ \
    const TYPE libxsmm_matinit_inv_ = ((TYPE)(SCALE)) / libxsmm_matinit_maxval2_; \
    const size_t libxsmm_matinit_shuffle_ = libxsmm_coprime2(libxsmm_matinit_maxval_); \
    OMP(parallel for private(libxsmm_matinit_i_, libxsmm_matinit_j_)) \
    for (libxsmm_matinit_i_ = 0; libxsmm_matinit_i_ < ((libxsmm_blasint)NCOLS); ++libxsmm_matinit_i_) { \
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
 */
LIBXSMM_API int libxsmm_matdiff(libxsmm_matdiff_info* info,
  libxsmm_datatype datatype, libxsmm_blasint m, libxsmm_blasint n, const void* ref, const void* tst,
  const libxsmm_blasint* ldref, const libxsmm_blasint* ldtst);

/** Combine absolute and relative norms into a value which can be used to check against a margin. */
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

/** Calculate co-prime number <= n/2 (libxsmm_coprime2(0|1) == 0). */
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

/* Convert BF8 to FP32 (scalar). */
LIBXSMM_API float libxsmm_convert_bf8_to_f32(libxsmm_bfloat8 in);
/* Convert BF16 to FP32 (scalar). */
LIBXSMM_API float libxsmm_convert_bf16_to_f32(libxsmm_bfloat16 in);
/* Convert FP16 to FP32 (scalar). */
LIBXSMM_API float libxsmm_convert_f16_to_f32(libxsmm_float16 in);
/* Convert FP16 to HF8 (scalar). */
LIBXSMM_API libxsmm_hfloat8 libxsmm_convert_f16_hf8_rne(libxsmm_float16 in);
/* Convert FP32 to FP16 (scalar). */
LIBXSMM_API libxsmm_float16 libxsmm_convert_f32_to_f16(float in);

/** SQRT with Newton's method using integer arithmetic. */
LIBXSMM_API unsigned int libxsmm_isqrt_u64(unsigned long long x);
/** SQRT with Newton's method using integer arithmetic. */
LIBXSMM_API unsigned int libxsmm_isqrt_u32(unsigned int x);
/** Based on libxsmm_isqrt_u32, but actual factor of x. */
LIBXSMM_API unsigned int libxsmm_isqrt2_u32(unsigned int x);
/** SQRT with Newton's method using double-precision. */
LIBXSMM_API double libxsmm_dsqrt(double x);
/** SQRT with Newton's method using single-precision. */
LIBXSMM_API float libxsmm_ssqrt(float x);

/** CBRT with Newton's method using integer arithmetic. */
LIBXSMM_API unsigned int libxsmm_icbrt_u64(unsigned long long x);
/** CBRT with Newton's method using integer arithmetic. */
LIBXSMM_API unsigned int libxsmm_icbrt_u32(unsigned int x);

/** Single-precision approximation of exponential function (base 2). */
LIBXSMM_API float libxsmm_sexp2(float x);

/**
 * Exponential function (base 2), which is limited to unsigned 8-bit input values.
 * This function reproduces bit-accurate results (single-precision).
 */
LIBXSMM_API float libxsmm_sexp2_u8(unsigned char x);

/**
* Exponential function (base 2), which is limited to signed 8-bit input values.
* This function reproduces bit-accurate results (single-precision).
*/
LIBXSMM_API float libxsmm_sexp2_i8(signed char x);

/** Similar to libxsmm_sexp2_i8, but takes an integer as signed 8-bit value (check). */
LIBXSMM_API float libxsmm_sexp2_i8i(int x);

#endif /*LIBXSMM_MATH_H*/
