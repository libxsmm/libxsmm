/******************************************************************************
** Copyright (c) 2014-2019, Intel Corporation                                **
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
  double rng_sum;
  double rng_exp;
  float  rng_min;
  float  rng_max;
  double rng_var;
  double rng_stddev;
  float* rngs;
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

  libxsmm_rng_float_set_seed( (uint32_t)(time(0)));

  /* fill array with random floats */
  libxsmm_rng_float_seq( rngs, num_rngs );

  /* calculate quality of random numbers */
  rng_sum = 0.0;
  rng_min = 2.0;
  rng_max = 0.0;
  for ( i = 0 ; i < num_rngs; ++i ) {
    rng_sum += (double)rngs[i];
    rng_min = (rngs[i] < rng_min) ? rngs[i] : rng_min;
    rng_max = (rngs[i] > rng_max) ? rngs[i] : rng_max;
  }
  rng_exp = rng_sum/(double)num_rngs;
  rng_var = 0.0;
  for( i = 0; i < num_rngs; ++i ) {
    rng_var += (rngs[i] - rng_exp) * (rngs[i] - rng_exp);
  }
  rng_var = rng_var/(double)num_rngs;
  rng_stddev = sqrt(rng_var);

  start = libxsmm_timer_tick();
  for (i = 0; i < num_rngs; ++i) {
    libxsmm_rng_float_seq( rngs, 1 );
  }
  printf("\nlibxsmm_rng_float:  %llu cycles per random number\n",
    libxsmm_timer_cycles(start, libxsmm_timer_tick()) / num_rngs);

  /* let's compute some values of the random numbers */
  printf("\nWe have generatred %i random numbers uniformly distributed in [0,1(\n", num_rngs);
  printf("We expect the following values E=0.5, Var=0.083333, Stddev=0.288675\n\n");
  printf("minimum random number is:            %f\n", rng_min);
  printf("maximum random number is:            %f\n", rng_max);
  printf("sum of random numbers is:            %f\n", rng_sum);
  printf("Expected Value of random numbers is: %f\n", rng_exp);
  printf("Variance of random numbers is:       %f\n", rng_var);
  printf("StdDev of random numbers is:         %f\n\n", rng_stddev);

  free( rngs );

  return EXIT_SUCCESS;
}

