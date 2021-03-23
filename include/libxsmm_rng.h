/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_RNG_H
#define LIBXSMM_RNG_H

#include "libxsmm_typedefs.h"

/**
 * create a new external state for thread-save execution managed
 * by the user. We do not provide a function for drawing the random numbers
 * the user is supposed to call the LIBXSMM_INTRINSICS_MM512_RNG_EXTSTATE_PS
 * or LIBXSMM_INTRINSICS_MM512_RNG_XOSHIRO128P_EXTSTATE_EPI32 intrinsic.
 * */
LIBXSMM_API unsigned int* libxsmm_rng_create_extstate(unsigned int/*uint32_t*/ seed);

/** free a previously created rng_avx512_extstate */
LIBXSMM_API void libxsmm_rng_destroy_extstate(unsigned int* stateptr);

/** Set the seed of libxsmm_rng_* (similar to srand). */
LIBXSMM_API void libxsmm_rng_set_seed(unsigned int/*uint32_t*/ seed);

/**
 * This SP-RNG is using xoshiro128+ 1.0, work done by
 * David Blackman and Sebastiano Vigna (vigna@acm.org).
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

/** Sequence of random data based on libxsmm_rng_u32. */
LIBXSMM_API void libxsmm_rng_seq(void* data, libxsmm_blasint nbytes);

/**
 * Similar to libxsmm_rng_u32, but returns a DP-value in the interval [0, 1).
 * Note: libxsmm_rng_set_seed must be used if one wishes to seed the generator.
 */
LIBXSMM_API double libxsmm_rng_f64(void);

#endif /* LIBXSMM_RNG_H */

