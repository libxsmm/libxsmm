/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#if !defined(INCLUDE_LIBXSMM_LAST)
# include <libxsmm.h>
#endif
#include <math.h>
#if defined(INCLUDE_LIBXSMM_LAST)
# include <libxsmm.h>
#endif

#if !defined(USE_EXPECTED) && 0
# define USE_EXPECTED
#else
# include <time.h>
#endif


int main(/*int argc, char* argv[]*/)
{
#if defined(USE_EXPECTED)
  const unsigned int seed = 25071975;
  const float rngs_expected[] = {
    0.438140392f, 0.284636021f, 0.808342457f, 0.140940785f, 0.740890265f, 0.0189954042f, 0.4811354880f, 0.616942167f,
    0.273835897f, 0.636928558f, 0.916998625f, 0.260923862f, 0.673431635f, 0.5160189870f, 0.0404732227f, 0.327739120f
  };
#endif
  libxsmm_blasint num_rngs = 1000, i;
  libxsmm_matdiff_info info;
  int result = EXIT_SUCCESS;

  float *const rngs = (float*)malloc((size_t)(sizeof(float) * num_rngs));
  if (NULL == rngs) num_rngs = 0;

  /* mute warning about potentially uninitialized variable */
  libxsmm_matdiff_clear(&info);

#if defined(USE_EXPECTED)
  /* setup reproducible sequence */
  libxsmm_rng_set_seed(seed);

  /* fill array with random floats */
  libxsmm_rng_f32_seq(rngs, num_rngs);

  /* check expected value (depends on reproducible seed) */
  for (i = 0; i < 16; ++i) {
    if (rngs_expected[i] != rngs[i]) result = EXIT_FAILURE;
  }
  /* reset state */
  libxsmm_rng_set_seed(seed);
  /* enforce scalar RNG */
  libxsmm_rng_f32_seq(rngs, 15);

  /* check expected value matches scalar RNG; check successful reset */
  for (i = 0; i < 16; ++i) {
    if (rngs_expected[i] != rngs[i]) result = EXIT_FAILURE;
  }

  if (EXIT_SUCCESS == result) { /* calculate quality of random numbers */
    result = libxsmm_matdiff(&info, LIBXSMM_DATATYPE_F32, 1/*m*/, num_rngs,
      NULL/*ref*/, rngs/*tst*/, NULL/*ldref*/, NULL/*ldtst*/);
  }
#else
  { int j; for (j = 0; j < 1000; ++j) {
    /* setup sequence */
    libxsmm_rng_set_seed((unsigned int)time(0));
    /* fill array with random floats */
    switch (j % 2) {
      case 1: {
        for (i = 0; i < num_rngs; ++i) rngs[i] = (float)libxsmm_rng_f64();
      } break;
      default: libxsmm_rng_f32_seq(rngs, num_rngs);
    }
    if (EXIT_SUCCESS == result) { /* calculate quality of random numbers */
      libxsmm_matdiff_info j_info;
      result = libxsmm_matdiff(&j_info, LIBXSMM_DATATYPE_F32, 1/*m*/, num_rngs,
        NULL/*ref*/, rngs/*tst*/, NULL/*ldref*/, NULL/*ldtst*/);
      if (EXIT_SUCCESS == result) libxsmm_matdiff_reduce(&info, &j_info);
    }
#endif
    if (EXIT_SUCCESS == result) {
      libxsmm_blasint num_odd = 0, num_even = 0;
      const double scale = 0xFFFFFFFF;
      for (i = 0; i < num_rngs; ++i) {
        const unsigned int u = (unsigned int)LIBXSMM_ROUND(rngs[i] * scale);
        if (u & 1) {
          ++num_odd;
        }
        else {
          ++num_even;
        }
      }
      if (num_rngs < 4 * LIBXSMM_DELTA(num_odd, num_even)) result = EXIT_FAILURE;
    }
#if !defined(USE_EXPECTED)
  }}
#endif

  if (EXIT_SUCCESS == result) {
    const double range = info.max_tst - info.min_tst, expected = 0.5;
    if (expected < 5 * LIBXSMM_DELTA(info.avg_tst, expected)) result = EXIT_FAILURE;
    if (expected < 5 * LIBXSMM_DELTA(0.5 * range, expected)) result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    const double expected = 1.0 / 12.0;
    if (expected < 5 * LIBXSMM_DELTA(info.var_tst, expected)) result = EXIT_FAILURE;
  }

  free(rngs);

  return result;
}

