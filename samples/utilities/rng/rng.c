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
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#if defined(_OPENMP)
# include <omp.h>
#endif


int main(int argc, char* argv[])
{
  double rng_stddev = 0;
  float* rngs;
  float  vrng[16];
  libxsmm_matdiff_info info;
  libxsmm_blasint num_rngs;
  libxsmm_blasint i;
  unsigned long long start;

  if (2 < argc) {
    fprintf(stderr, "Usage:\n  %s number_rngs\n", argv[0]);
    return EXIT_SUCCESS;
  }

  /* parse the command line and set up the test parameters */
  num_rngs = (1 < argc ? atoi(argv[1]) : 1000);
  assert(num_rngs >= 1);

  rngs = (float*)malloc( num_rngs*sizeof(float) );
  if (NULL == rngs) num_rngs = 0;

  libxsmm_rng_set_seed( (uint32_t)(time(0)));

  /* fill array with random floats */
  libxsmm_rng_f32_seq( rngs, num_rngs );

  /* calculate quality of random numbers */
  if (EXIT_SUCCESS == libxsmm_matdiff(&info, LIBXSMM_DATATYPE_F32, 1/*m*/, num_rngs,
    NULL/*ref*/, rngs/*tst*/, NULL/*ldref*/, NULL/*ldtst*/))
  {
    rng_stddev = sqrt(info.var_tst);
  }

  start = libxsmm_timer_tick();
  for (i = 0; i < num_rngs; ++i) {
    libxsmm_rng_f32_seq( rngs, 1 );
  }
  printf("\nlibxsmm_rng_float:  %llu cycles per random number (scalar)\n",
    libxsmm_timer_cycles(start, libxsmm_timer_tick()) / num_rngs);

  start = libxsmm_timer_tick();
  for (i = 0; i < num_rngs; ++i) {
    libxsmm_rng_f32_seq( vrng, 16 );
  }
  printf("\nlibxsmm_rng_float:  %llu cycles per random number (vlen=16)\n",
    libxsmm_timer_cycles(start, libxsmm_timer_tick()) / ((size_t)num_rngs*16));

  /* let's compute some values of the random numbers */
  printf("\nWe have generated %i random numbers uniformly distributed in [0,1(\n", num_rngs);
  printf("We expect the following values E=0.5, Var=0.083333, Stddev=0.288675\n\n");
  printf("minimum random number is:            %f\n", info.min_tst);
  printf("maximum random number is:            %f\n", info.max_tst);
  printf("sum of random numbers is:            %f\n", info.l1_tst);
  printf("Expected Value of random numbers is: %f\n", info.avg_tst);
  printf("Variance of random numbers is:       %f\n", info.var_tst);
  printf("StdDev of random numbers is:         %f\n\n", rng_stddev);

  free( rngs );

  return EXIT_SUCCESS;
}

