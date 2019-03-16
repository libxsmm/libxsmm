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
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdio.h>
#include <math.h>


int main(int argc, char* argv[])
{
  const int insize = (1 < argc ? atoi(argv[1]) : 0);
  const int niters = (2 < argc ? atoi(argv[2]) : 1);
  const size_t n = (0 >= insize ? (((size_t)2 << 30/*2 GB*/) / sizeof(float)) : ((size_t)insize));
  float *inp, *out, *gold;
  size_t size, nrpt;
  int result;

  if (0 < niters) {
    nrpt = niters;
    size = n;
  }
  else {
    nrpt = n;
    size = LIBXSMM_MAX(LIBXSMM_ABS(niters), 1);
  }

  gold = (float*)(malloc(sizeof(float) * size));
  out = (float*)(malloc(sizeof(float) * size));
  inp = (float*)(malloc(sizeof(float) * size));

  if (NULL != gold && NULL != out && NULL != inp) {
    libxsmm_timer_tickint start;
    libxsmm_matdiff_info diff;
    size_t i, j;

    /* initialize the input data */
    libxsmm_rng_set_seed(25071975);
    libxsmm_rng_f32_seq(inp, (libxsmm_blasint)size);

    { start = libxsmm_timer_tick();
      for (j = 0; j < nrpt; ++j) {
        for (i = 0; i < size; ++i) {
          gold[i] = (float)LIBXSMM_EXP2((double)inp[i]);
        }
      }
      printf("standard exp2:\t\t%.3f s\n", libxsmm_timer_duration(start, libxsmm_timer_tick()));
    }

    { start = libxsmm_timer_tick();
      for (j = 0; j < nrpt; ++j) {
        for (i = 0; i < size; ++i) {
          out[i] = LIBXSMM_EXP2F(inp[i]);
        }
      }
      printf("standard exp2f:\t\t%.3f s", libxsmm_timer_duration(start, libxsmm_timer_tick()));
      if (EXIT_SUCCESS == libxsmm_matdiff(&diff, LIBXSMM_DATATYPE_F32, 1/*m*/,
        (libxsmm_blasint)size, gold, out, NULL/*ldref*/, NULL/*ldtst*/))
      {
        printf("\t\tdiff: L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
      }
      else printf("\n");
    }

    { start = libxsmm_timer_tick();
      for (j = 0; j < nrpt; ++j) {
        for (i = 0; i < size; ++i) {
          out[i] = libxsmm_sexp2(inp[i]);
        }
      }
      printf("libxsmm_sexp2:\t\t%.3f s", libxsmm_timer_duration(start, libxsmm_timer_tick()));
      if (EXIT_SUCCESS == libxsmm_matdiff(&diff, LIBXSMM_DATATYPE_F32, 1/*m*/,
        (libxsmm_blasint)size, gold, out, NULL/*ldref*/, NULL/*ldtst*/))
      {
        printf("\t\tdiff: L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
      }
      else printf("\n");
    }

    { start = libxsmm_timer_tick();
      for (j = 0; j < nrpt; ++j) {
        for (i = 0; i < size; ++i) {
          out[i] = libxsmm_sexp2_fast(inp[i], 13);
        }
      }
      printf("libxsmm_sexp2_fast13:\t%.3f s", libxsmm_timer_duration(start, libxsmm_timer_tick()));
      if (EXIT_SUCCESS == libxsmm_matdiff(&diff, LIBXSMM_DATATYPE_F32, 1/*m*/,
        (libxsmm_blasint)size, gold, out, NULL/*ldref*/, NULL/*ldtst*/))
      {
        printf("\t\tdiff: L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
      }
      else printf("\n");
    }

    { start = libxsmm_timer_tick();
      for (j = 0; j < nrpt; ++j) {
        for (i = 0; i < size; ++i) {
          out[i] = libxsmm_sexp2_fast(inp[i], 0);
        }
      }
      printf("libxsmm_sexp2_fast3:\t%.3f s", libxsmm_timer_duration(start, libxsmm_timer_tick()));
      if (EXIT_SUCCESS == libxsmm_matdiff(&diff, LIBXSMM_DATATYPE_F32, 1/*m*/,
        (libxsmm_blasint)size, gold, out, NULL/*ldref*/, NULL/*ldtst*/))
      {
        printf("\t\tdiff: L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
      }
      else printf("\n");
    }

    result = EXIT_SUCCESS;
  }
  else {
    result = EXIT_FAILURE;
  }

  free(gold);
  free(out);
  free(inp);

  return result;
}

