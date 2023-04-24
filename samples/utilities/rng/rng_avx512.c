/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <utils/libxsmm_intrinsics_x86.h>
#include <stdio.h>


int main(int argc, char* argv[])
{
  double rng_stddev = 0;
  float* rngs;
#if defined(__AVX512F__)
  float  vrng[16];
#endif
  unsigned int* state = NULL;
  libxsmm_timer_tickint start;
  libxsmm_matdiff_info info;
  libxsmm_blasint num_rngs;
  libxsmm_blasint i;

  if (2 < argc) {
    fprintf(stderr, "Usage:\n  %s number_rngs\n", argv[0]);
    return EXIT_SUCCESS;
  }

  /* parse the command line and set up the test parameters */
  num_rngs = (1 < argc ? atoi(argv[1]) : 1024);
  /* avoid scalar remainder in timing loop */
  num_rngs = LIBXSMM_UP2(num_rngs, 16);
  assert(num_rngs >= 1);

  rngs = (float*)malloc((size_t)(sizeof(float) * num_rngs));
  if (NULL == rngs) num_rngs = 0;

  /* create thread-safe state */
  state = libxsmm_rng_create_extstate( (unsigned int)(time(0)) );

  /* fill array with random floats */
  for (i = 0; i < num_rngs; i+=16) {
#ifdef __AVX512F__
    _mm512_storeu_ps( rngs+i, LIBXSMM_INTRINSICS_MM512_RNG_EXTSTATE_PS( state ) );
#endif
  }

  /* some quality measure; variance is based on discovered average rather than expected value */
  if (EXIT_SUCCESS == libxsmm_matdiff(&info, LIBXSMM_DATATYPE_F32, 1/*m*/, num_rngs,
    NULL/*ref*/, rngs/*tst*/, NULL/*ldref*/, NULL/*ldtst*/))
  {
    rng_stddev = libxsmm_dsqrt( info.var_tst );
  }

  start = libxsmm_timer_tick();
  for (i = 0; i < num_rngs; ++i) {
#if defined(__AVX512F__)
    _mm512_storeu_ps( vrng, _mm512_add_ps( _mm512_load_ps(vrng), LIBXSMM_INTRINSICS_MM512_RNG_EXTSTATE_PS( state ) ) );
#endif
  }
  printf("\nlibxsmm_rng_float:  %llu cycles per random number (vlen=16)\n",
    libxsmm_timer_ncycles(start, libxsmm_timer_tick()) / ((size_t)num_rngs*16));

  /* free the state */
  libxsmm_rng_destroy_extstate( state );

  /* let's compute some values of the random numbers */
  printf("\n%lli random numbers generated, which are uniformly distributed in [0,1(\n", (long long)num_rngs);
  printf("Expected properties: avg=0.5, var=0.083333, stddev=0.288675\n\n");
  printf("minimum random number: %f\n", info.min_tst);
  printf("maximum random number: %f\n", info.max_tst);
  printf("sum of random numbers: %f\n", info.l1_tst);
  printf("avg of random numbers: %f\n", info.avg_tst);
  printf("var of random numbers: %f\n", info.var_tst);
  printf("dev of random numbers: %f\n\n", rng_stddev);

  free( rngs );

  return EXIT_SUCCESS;
}

