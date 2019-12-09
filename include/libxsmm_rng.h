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
LIBXSMM_API void libxsmm_rng_seq(void* data, libxsmm_blasint count);

/**
 * Similar to libxsmm_rng_u32, but returns a DP-value in the interval [0, 1).
 * Note: libxsmm_rng_set_seed must be used if one wishes to seed the generator.
 */
LIBXSMM_API double libxsmm_rng_f64(void);

#endif /* LIBXSMM_RNG_H */

