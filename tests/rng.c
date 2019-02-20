/******************************************************************************
** Copyright (c) 2019, Intel Corporation                                     **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Hans Pabst, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <math.h>

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

  float *const rngs = (float*)malloc(num_rngs * sizeof(float));
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
#else
  { int j; for (j = 0; j < 100; ++j) {
    /* setup sequence */
    libxsmm_rng_set_seed((unsigned int)time(0));
    /* fill array with random floats */
    switch (j % 2) {
      case 1: {
        for (i = 0; i < num_rngs; ++i) rngs[i] = (float)libxsmm_rng_f64();
      } break;
      default: libxsmm_rng_f32_seq(rngs, num_rngs);
    }
#endif
    if (EXIT_SUCCESS == result) { /* calculate quality of random numbers */
      result = libxsmm_matdiff(&info, LIBXSMM_DATATYPE_F32, 1/*m*/, num_rngs,
        NULL/*ref*/, rngs/*tst*/, NULL/*ldref*/, NULL/*ldtst*/);
    }

    if (EXIT_SUCCESS == result) {
      const double expected = 0.5 * num_rngs;
      if (expected < 5 * LIBXSMM_DIFF(info.l1_tst, expected)) result = EXIT_FAILURE;
    }

    if (EXIT_SUCCESS == result) {
      const double expected = 1.0 / 12.0;
      if (expected < 5 * LIBXSMM_DIFF(info.var_tst, expected)) result = EXIT_FAILURE;
    }

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
      if (num_rngs < 4 * LIBXSMM_DIFF(num_odd, num_even)) result = EXIT_FAILURE;
    }
#if !defined(USE_EXPECTED)
  }}
#endif
  free(rngs);

  return result;
}

