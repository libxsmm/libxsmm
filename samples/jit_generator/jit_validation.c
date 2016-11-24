/******************************************************************************
** Copyright (c) 2015-2016, Intel Corporation                                **
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
#include <string.h>

static unsigned int g_jit_code_reps = 0;

void print_help(void) {
  printf("\n\n");
  printf("Usage (dense*dense=dense):\n");
  printf("    M\n");
  printf("    N\n");
  printf("    K\n");
  printf("    LDA\n");
  printf("    LDB\n");
  printf("    LDC\n");
  printf("    alpha: -1 or 1\n");
  printf("    beta: 0 or 1\n");
  printf("    0: unaligned A, otherwise aligned\n");
  printf("    0: unaligned C, otherwise aligned\n");
  printf("    PREFETCH: nopf (none), pfsigonly, BL2viaC, AL2, curAL2, AL2jpst, AL2_BL2viaC, curAL2_BL2viaC, AL2jpst_BL2viaC\n");
  printf("    PRECISION: SP, DP\n");
  printf("    #repetitions\n");
  printf("\n\n");
}

void init_double( double*                         io_a,
                  double*                         io_b,
                  double*                         io_c,
                  double*                         io_c_gold,
                  const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  unsigned int l_i, l_j;

  /* touch A */
  for ( l_i = 0; l_i < i_xgemm_desc->lda; l_i++) {
    for ( l_j = 0; l_j < i_xgemm_desc->k; l_j++) {
      io_a[(l_j * i_xgemm_desc->lda) + l_i] = (double)drand48();
    }
  }
  /* touch B */
  for ( l_i = 0; l_i < i_xgemm_desc->ldb; l_i++ ) {
    for ( l_j = 0; l_j < i_xgemm_desc->n; l_j++ ) {
      io_b[(l_j * i_xgemm_desc->ldb) + l_i] = (double)drand48();
    }
  }
  /* touch C */
  for ( l_i = 0; l_i < i_xgemm_desc->ldc; l_i++) {
    for ( l_j = 0; l_j < i_xgemm_desc->n; l_j++) {
      io_c[(l_j * i_xgemm_desc->ldc) + l_i] = (double)0.0;
      io_c_gold[(l_j * i_xgemm_desc->ldc) + l_i] = (double)0.0;
    }
  }
}

void init_float( float*                          io_a,
                 float*                          io_b,
                 float*                          io_c,
                 float*                          io_c_gold,
                 const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  unsigned int l_i, l_j;

  /* touch A */
  for ( l_i = 0; l_i < i_xgemm_desc->lda; l_i++) {
    for ( l_j = 0; l_j < i_xgemm_desc->k; l_j++) {
      io_a[(l_j * i_xgemm_desc->lda) + l_i] = (float)drand48();
    }
  }
  /* touch B */
  for ( l_i = 0; l_i < i_xgemm_desc->ldb; l_i++ ) {
    for ( l_j = 0; l_j < i_xgemm_desc->n; l_j++ ) {
      io_b[(l_j * i_xgemm_desc->ldb) + l_i] = (float)drand48();
    }
  }
  /* touch C */
  for ( l_i = 0; l_i < i_xgemm_desc->ldc; l_i++) {
    for ( l_j = 0; l_j < i_xgemm_desc->n; l_j++) {
      io_c[(l_j * i_xgemm_desc->ldc) + l_i] = (float)0.0;
      io_c_gold[(l_j * i_xgemm_desc->ldc) + l_i] = (float)0.0;
    }
  }
}

void run_gold_double( const double*                   i_a,
                      const double*                   i_b,
                      double*                         o_c,
                      const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  unsigned int l_m, l_n, l_k, l_t;
  double l_runtime = 0.0;

  const unsigned long long l_start = libxsmm_timer_tick();

  for ( l_t = 0; l_t < g_jit_code_reps; l_t++  ) {
    for ( l_n = 0; l_n < i_xgemm_desc->n; l_n++  ) {
      for ( l_k = 0; l_k < i_xgemm_desc->k; l_k++  ) {
        for ( l_m = 0; l_m < i_xgemm_desc->m; l_m++ ) {
          o_c[(l_n * i_xgemm_desc->ldc) + l_m] += i_a[(l_k * i_xgemm_desc->lda) + l_m] * i_b[(l_n * i_xgemm_desc->ldb) + l_k];
        }
      }
    }
  }

  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  printf("%fs for C\n", l_runtime);
  printf("%f GFLOPS for C\n", ((double)((double)g_jit_code_reps * (double)i_xgemm_desc->m * (double)i_xgemm_desc->n * (double)i_xgemm_desc->k) * 2.0) / (l_runtime * 1.0e9));
}


void run_gold_float( const float*                   i_a,
                     const float*                   i_b,
                     float*                         o_c,
                     const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  unsigned int l_m, l_n, l_k, l_t;
  double l_runtime = 0.0;

  const unsigned long long l_start = libxsmm_timer_tick();

  for ( l_t = 0; l_t < g_jit_code_reps; l_t++  ) {
    for ( l_n = 0; l_n < i_xgemm_desc->n; l_n++  ) {
      for ( l_k = 0; l_k < i_xgemm_desc->k; l_k++  ) {
        for ( l_m = 0; l_m < i_xgemm_desc->m; l_m++ ) {
          o_c[(l_n * i_xgemm_desc->ldc) + l_m] += i_a[(l_k * i_xgemm_desc->lda) + l_m] * i_b[(l_n * i_xgemm_desc->ldb) + l_k];
        }
      }
    }
  }

  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  printf("%fs for C\n", l_runtime);
  printf("%f GFLOPS for C\n", ((double)((double)g_jit_code_reps * (double)i_xgemm_desc->m * (double)i_xgemm_desc->n * (double)i_xgemm_desc->k) * 2.0) / (l_runtime * 1.0e9));
}

void run_jit_double( const double*                    i_a,
                     const double*                    i_b,
                     double*                          o_c,
                     const int                        i_M,
                     const int                        i_N,
                     const int                        i_K,
                     const libxsmm_gemm_prefetch_type i_prefetch ) {
  /* define function pointer */
  libxsmm_dmmfunction l_test_jit;
  double l_jittime = 0.0, l_runtime = 0.0;
  double l_alpha = 1.0;
  double l_beta = 1.0;
  unsigned long long l_start;
  unsigned int l_t;

  if ( l_beta != 0.0 && l_beta != 1.0 ) {
    fprintf(stderr, "JIT double: beta needs to be 0.0 or 1.0!\n");
    exit(-1);
  }
  if ( l_alpha != 1.0 ) {
    fprintf(stderr, "JIT double: alpha needs to be 1.0!\n");
    exit(-1);
  }

  l_start = libxsmm_timer_tick();
  l_test_jit = libxsmm_dmmdispatch(i_M, i_N, i_K, &i_M, &i_K, &i_M, &l_alpha, &l_beta, NULL, &i_prefetch );
  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  printf("function pointer address: %llx\n", (size_t)l_test_jit);

  l_start = libxsmm_timer_tick();

  if ( i_prefetch == LIBXSMM_PREFETCH_NONE ) {
    for ( l_t = 0; l_t < g_jit_code_reps; l_t++ ) {
      l_test_jit(i_a, i_b, o_c);
    }
  } else {
    for ( l_t = 0; l_t < g_jit_code_reps; l_t++ ) {
      l_test_jit(i_a, i_b, o_c, i_a, i_b, o_c);
    }
  }

  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  printf("%fs for creating jit\n", l_jittime);
  printf("%fs for executing jit\n", l_runtime);
  printf("%f GFLOPS for jit\n", ((double)((double)g_jit_code_reps * (double)i_M * (double)i_N * (double)i_K) * 2.0) / (l_runtime * 1.0e9));
}

void run_jit_float( const float*                     i_a,
                    const float*                     i_b,
                    float*                           o_c,
                    const int                        i_M,
                    const int                        i_N,
                    const int                        i_K,
                    const libxsmm_gemm_prefetch_type i_prefetch ) {
  /* define function pointer */
  libxsmm_smmfunction l_test_jit;
  double l_jittime = 0.0, l_runtime = 0.0;
  float l_alpha = 1.0f;
  float l_beta = 1.0f;
  unsigned long long l_start;
  unsigned int l_t;

  if ( l_beta != 0.0f && l_beta != 1.0f ) {
    fprintf(stderr, "JIT float: beta needs to be 0.0 or 1.0!\n");
    exit(-1);
  }
  if ( l_alpha != 1.0f ) {
    fprintf(stderr, "JIT float: alpha needs to be 1.0!\n");
    exit(-1);
  }

  l_start = libxsmm_timer_tick();
  l_test_jit = libxsmm_smmdispatch(i_M, i_N, i_K, &i_M, &i_K, &i_M, &l_alpha, &l_beta, NULL, &i_prefetch );
  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  printf("function pointer address: %llx\n", (size_t)l_test_jit);

  l_start = libxsmm_timer_tick();

  if ( i_prefetch == LIBXSMM_PREFETCH_NONE ) {
    for ( l_t = 0; l_t < g_jit_code_reps; l_t++ ) {
      l_test_jit(i_a, i_b, o_c);
    }
  } else {
    for ( l_t = 0; l_t < g_jit_code_reps; l_t++ ) {
      l_test_jit(i_a, i_b, o_c, i_a, i_b, o_c);
    }
  }

  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  printf("%fs for creating jit\n", l_jittime);
  printf("%fs for executing jit\n", l_runtime);
  printf("%f GFLOPS for jit\n", ((double)((double)g_jit_code_reps * (double)i_M * (double)i_N * (double)i_K) * 2.0) / (l_runtime * 1.0e9));
}

void max_error_double( const double*                   i_c,
                       const double*                   i_c_gold,
                       const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  unsigned int l_i, l_j;
  double l_max_error = 0.0;

  for ( l_i = 0; l_i < i_xgemm_desc->m; l_i++) {
    for ( l_j = 0; l_j < i_xgemm_desc->n; l_j++) {
#if 0
      printf("Entries in row %i, column %i, gold: %f, jit: %f\n", l_i+1, l_j+1, i_c_gold[(l_j*i_xgemm_desc->ldc)+l_i], i_c[(l_j*i_xgemm_desc->ldc)+l_i]);
#endif
      if (l_max_error < fabs( i_c_gold[(l_j * i_xgemm_desc->ldc) + l_i] - i_c[(l_j * i_xgemm_desc->ldc) + l_i]))
        l_max_error = fabs( i_c_gold[(l_j * i_xgemm_desc->ldc) + l_i] - i_c[(l_j * i_xgemm_desc->ldc) + l_i]);
    }
  }

  printf("max. error: %f\n", l_max_error);
}

void max_error_float( const float*                    i_c,
                      const float*                    i_c_gold,
                      const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  unsigned int l_i, l_j;
  double l_max_error = 0.0;

  for ( l_i = 0; l_i < i_xgemm_desc->m; l_i++) {
    for ( l_j = 0; l_j < i_xgemm_desc->n; l_j++) {
#if 0
      printf("Entries in row %i, column %i, gold: %f, jit: %f\n", l_i+1, l_j+1, i_c_gold[(l_j*i_xgemm_desc->ldc)+l_i], i_c[(l_j*i_xgemm_desc->ldc)+l_i]);
#endif
      if (l_max_error < fabs( (double)i_c_gold[(l_j * i_xgemm_desc->ldc) + l_i] - (double)i_c[(l_j * i_xgemm_desc->ldc) + l_i]))
        l_max_error = fabs( (double)i_c_gold[(l_j * i_xgemm_desc->ldc) + l_i] - (double)i_c[(l_j * i_xgemm_desc->ldc) + l_i]);
    }
  }

  printf("max. error: %f\n", l_max_error);
}

int main(int argc, char* argv []) {
  char* l_arch = NULL;
  char* l_precision = NULL;
  int l_m = 0;
  int l_n = 0;
  int l_k = 0;
  int l_lda = 0;
  int l_ldb = 0;
  int l_ldc = 0;
  int l_aligned_a = 0;
  int l_aligned_c = 0;
  int l_alpha = 0;
  int l_beta = 0;
  int l_single_precision = 0;
  libxsmm_gemm_prefetch_type l_prefetch = 0;

  libxsmm_gemm_descriptor l_xgemm_desc;
  /* init data structures */
  double* l_a_d;
  double* l_b_d;
  double* l_c_d;
  double* l_c_gold_d;
  float* l_a_f;
  float* l_b_f;
  float* l_c_f;
  float* l_c_gold_f;

  /* check argument count for a valid range */
  if ( argc != 14 ) {
    print_help();
    return -1;
  }

  /* xgemm sizes */
  l_m = atoi(argv[1]);
  l_n = atoi(argv[2]);
  l_k = atoi(argv[3]);
  l_lda = atoi(argv[4]);
  l_ldb = atoi(argv[5]);
  l_ldc = atoi(argv[6]);

  /* some sugar */
  l_alpha = atoi(argv[7]);
  l_beta = atoi(argv[8]);
  l_aligned_a = atoi(argv[9]);
  l_aligned_c = atoi(argv[10]);

  /* arch specific stuff */
  l_precision = argv[12];
  g_jit_code_reps = atoi(argv[13]);

  /* set value of prefetch flag */
  if (strcmp("nopf", argv[11]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_NONE;
  }
  else if (strcmp("pfsigonly", argv[11]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_SIGONLY;
  }
  else if (strcmp("BL2viaC", argv[11]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_BL2_VIA_C;
  }
  else if (strcmp("curAL2", argv[11]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL2_AHEAD;
  }
  else if (strcmp("curAL2_BL2viaC", argv[11]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD;
  }
  else if (strcmp("AL2", argv[11]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL2;
  }
  else if (strcmp("AL2_BL2viaC", argv[11]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL2BL2_VIA_C;
  }
  else if (strcmp("AL2jpst", argv[11]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL2_JPST;
  }
  else if (strcmp("AL2jpst_BL2viaC", argv[11]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST;
  }
  else {
    print_help();
    return -1;
  }

  /* check and evaluate precison flag */
  if ( strcmp(l_precision, "SP") == 0 ) {
    l_single_precision = 1;
  } else if ( strcmp(l_precision, "DP") == 0 ) {
    l_single_precision = 0;
  } else {
    print_help();
    return -1;
  }

  /* check alpha */
  if ((l_alpha != 1)) {
    print_help();
    return -1;
  }

  /* check beta */
  if ((l_beta != 0) && (l_beta != 1)) {
    print_help();
    return -1;
  }

  LIBXSMM_GEMM_DESCRIPTOR(l_xgemm_desc, 1,
    (0 == l_single_precision ? 0 : LIBXSMM_GEMM_FLAG_F32PREC)
      | (0 != l_aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0)
      | (0 != l_aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0),
    l_m, l_n, l_k, l_lda, l_ldb, l_ldc,
    l_alpha, l_beta, l_prefetch);

  if ( l_single_precision == 0 ) {
    l_a_d = (double*)libxsmm_aligned_malloc(l_xgemm_desc.lda * l_xgemm_desc.k * sizeof(double), 64);
    l_b_d = (double*)libxsmm_aligned_malloc(l_xgemm_desc.ldb * l_xgemm_desc.n * sizeof(double), 64);
    l_c_d = (double*)libxsmm_aligned_malloc(l_xgemm_desc.ldc * l_xgemm_desc.n * sizeof(double), 64);
    l_c_gold_d = (double*)libxsmm_aligned_malloc(l_xgemm_desc.ldc * l_xgemm_desc.n * sizeof(double), 64);
    init_double(l_a_d, l_b_d, l_c_d, l_c_gold_d, &l_xgemm_desc);
  } else {
    l_a_f = (float*)libxsmm_aligned_malloc(l_xgemm_desc.lda * l_xgemm_desc.k * sizeof(float), 64);
    l_b_f = (float*)libxsmm_aligned_malloc(l_xgemm_desc.ldb * l_xgemm_desc.n * sizeof(float), 64);
    l_c_f = (float*)libxsmm_aligned_malloc(l_xgemm_desc.ldc * l_xgemm_desc.n * sizeof(float), 64);
    l_c_gold_f = (float*)libxsmm_aligned_malloc(l_xgemm_desc.ldc * l_xgemm_desc.n * sizeof(float), 64);
    init_float(l_a_f, l_b_f, l_c_f, l_c_gold_f, &l_xgemm_desc);
  }

  /* print some output... */
  printf("------------------------------------------------\n");
  printf("RUNNING (%ix%i) X (%ix%i) = (%ix%i)", l_xgemm_desc.m, l_xgemm_desc.k, l_xgemm_desc.k, l_xgemm_desc.n, l_xgemm_desc.m, l_xgemm_desc.n);
  if ( l_single_precision == 0 ) {
    printf(", DP\n");
  } else {
    printf(", SP\n");
  }
  printf("------------------------------------------------\n");

  /* run C */
  if ( l_single_precision == 0 ) {
    run_gold_double( l_a_d, l_b_d, l_c_gold_d, &l_xgemm_desc );
  } else {
    run_gold_float( l_a_f, l_b_f, l_c_gold_f, &l_xgemm_desc );
  }

  /* run jit */
  if ( l_single_precision == 0 ) {
    run_jit_double( l_a_d, l_b_d, l_c_d, l_m, l_n, l_k, l_prefetch );
  } else {
    run_jit_float( l_a_f, l_b_f, l_c_f, l_m, l_n, l_k, l_prefetch );
  }

  /* test result */
  if ( l_single_precision == 0 ) {
    max_error_double( l_c_d, l_c_gold_d, &l_xgemm_desc );
  } else {
    max_error_float( l_c_f, l_c_gold_f, &l_xgemm_desc );
  }

  /* free */
  if ( l_single_precision == 0 ) {
    libxsmm_free(l_a_d);
    libxsmm_free(l_b_d);
    libxsmm_free(l_c_d);
    libxsmm_free(l_c_gold_d);
  } else {
    libxsmm_free(l_a_f);
    libxsmm_free(l_b_f);
    libxsmm_free(l_c_f);
    libxsmm_free(l_c_gold_f);
  }

  printf("------------------------------------------------\n");
  return 0;
}

