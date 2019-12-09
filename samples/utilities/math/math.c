/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
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

    /* collect gold data for exp2 function */
    { start = libxsmm_timer_tick();
      for (j = 0; j < nrpt; ++j) {
        for (i = 0; i < size; ++i) {
          gold[i] = (float)LIBXSMM_EXP2(inp[i]);
        }
      }
      printf("standard exp2:\t%.3f s\t\tgold\n", libxsmm_timer_duration(start, libxsmm_timer_tick()));
    }
    { start = libxsmm_timer_tick();
      for (j = 0; j < nrpt; ++j) {
        for (i = 0; i < size; ++i) {
          out[i] = LIBXSMM_EXP2F(inp[i]);
        }
      }
      printf("standard exp2f:\t%.3f s", libxsmm_timer_duration(start, libxsmm_timer_tick()));
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
      printf("libxsmm_sexp2:\t%.3f s", libxsmm_timer_duration(start, libxsmm_timer_tick()));
      if (EXIT_SUCCESS == libxsmm_matdiff(&diff, LIBXSMM_DATATYPE_F32, 1/*m*/,
        (libxsmm_blasint)size, gold, out, NULL/*ldref*/, NULL/*ldtst*/))
      {
        printf("\t\tdiff: L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
      }
      else printf("\n");
    }

    /* collect gold data for limited-range exp2 function */
    { start = libxsmm_timer_tick();
      for (j = 0; j < nrpt; ++j) {
        for (i = 0; i < size; ++i) {
          const unsigned char input = (unsigned char)(255.f * inp[i]);
          gold[i] = (float)LIBXSMM_EXP2(input);
        }
      }
      printf("low-range exp2:\t%.3f s\t\tgold\n", libxsmm_timer_duration(start, libxsmm_timer_tick()));
    }
    { start = libxsmm_timer_tick();
      for (j = 0; j < nrpt; ++j) {
        for (i = 0; i < size; ++i) {
          const unsigned char input = (unsigned char)(255.f * inp[i]);
          out[i] = libxsmm_sexp2_u8(input);
        }
      }
      printf("libxsmm_sexp2:\t%.3f s", libxsmm_timer_duration(start, libxsmm_timer_tick()));
      if (EXIT_SUCCESS == libxsmm_matdiff(&diff, LIBXSMM_DATATYPE_F32, 1/*m*/,
        (libxsmm_blasint)size, gold, out, NULL/*ldref*/, NULL/*ldtst*/))
      {
        printf("\t\tdiff: L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
      }
      else printf("\n");
    }

    /* collect gold data for sqrt function */
    { start = libxsmm_timer_tick();
      for (j = 0; j < nrpt; ++j) {
        for (i = 0; i < size; ++i) {
          gold[i] = (float)sqrt(inp[i]);
        }
      }
      printf("standard sqrt:\t%.3f s\t\tgold\n", libxsmm_timer_duration(start, libxsmm_timer_tick()));
    }
    { start = libxsmm_timer_tick();
      for (j = 0; j < nrpt; ++j) {
        for (i = 0; i < size; ++i) {
          out[i] = (float)libxsmm_dsqrt(inp[i]);
        }
      }
      printf("libxsmm_dsqrt:\t%.3f s", libxsmm_timer_duration(start, libxsmm_timer_tick()));
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
          out[i] = LIBXSMM_SQRTF(inp[i]);
        }
      }
      printf("standard sqrtf:\t%.3f s", libxsmm_timer_duration(start, libxsmm_timer_tick()));
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
          out[i] = libxsmm_ssqrt(inp[i]);
        }
      }
      printf("libxsmm_ssqrt:\t%.3f s", libxsmm_timer_duration(start, libxsmm_timer_tick()));
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

