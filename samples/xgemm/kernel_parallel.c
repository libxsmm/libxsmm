/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
# if defined(__APPLE__) && defined(__arm64__)
#include <pthread.h>
# endif
# if defined(_OPENMP)
#include <omp.h>
#endif

int g_reps = 0;

#include "kernel_jit_exec.h"

LIBXSMM_INLINE void print_help(void) {
  printf("\n\n");
  printf("1. Usage (dense*dense=dense, correctness and performance):\n");
  printf("    M\n");
  printf("    N\n");
  printf("    K\n");
  printf("    LDA\n");
  printf("    LDB\n");
  printf("    LDC\n");
  printf("    alpha: 1\n");
  printf("    beta: 0 or 1\n");
  printf("    0: unaligned A, otherwise aligned\n");
  printf("    0: unaligned C, otherwise aligned\n");
  printf("    0: A normal, 1: A trans\n");
  printf("    0: B normal, 1: B trans\n");
  printf("    PREFETCH: nopf (none), pfsigonly, BL2viaC, AL2, curAL2, AL2_BL2viaC, curAL2_BL2viaC\n");
  printf("    PRECISION: SP, DP, I16I32, USI8I32, SUI8I32, SUI8UI8, BF16F32, BF16, BF1632_FLAT, BF16_FLAT\n");
  printf("    BRGEMM: nobr, addrbr, offsbr, strdbr\n");
  printf("    BRsize: 1 - N\n");
  printf("    BRunroll: 0/1\n");
  printf("    #repetitions\n");
  printf("    tile configuration: 1 - external, 0 - internal\n");
  printf("\n\n");
  printf("2. Usage (dense*dense=dense, performance only option available):\n");
  printf("    filename with space-sperated sizes (M N K LDA LDB LDC)\n");
  printf("    alpha: 1\n");
  printf("    beta: 0 or 1\n");
  printf("    0: unaligned A, otherwise aligned\n");
  printf("    0: unaligned C, otherwise aligned\n");
  printf("    0: A normal, 1: A trans\n");
  printf("    0: B normal, 1: B trans\n");
  printf("    PRECISION: SP, DP, I16I32, USI8I32, SUI8I32, SUI8UI8, BF16F32, BF16, BF1632_FLAT, BF16_FLAT\n");
  printf("    BRGEMM: nobr, addrbr, offsbr, strdbr\n");
  printf("    BRsize: 1 - N\n");
  printf("    BRunroll: 0/1\n");
  printf("    #repetitions\n");
  printf("    0: no check, otherwise: run check - IGNORE IN THIS TESTER\n");
  printf("    tile configuration: 1 - external, 0 - internal\n");
  printf("\n\n");
}

int main(int argc, char* argv []) {
  char* l_precision = NULL;
  libxsmm_blasint l_lda = 0, l_ldb = 0, l_ldc = 0;
  int l_m = 0, l_n = 0, l_k = 0;
  int l_aligned_a = 0;
  int l_aligned_c = 0;
  int l_trans_a = 0;
  int l_trans_b = 0;
  double l_alpha = 0;
  double l_beta = 0;
  int l_br = 1;
  int l_br_type = 0;
  int l_br_unroll = 0;

  libxsmm_gemm_prefetch_type l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  gemm_def l_gemm_def;
  double l_runtime_libxsmm = 0;
  int l_file_input = 0;
  char* l_file_name = NULL;
  FILE *l_file_handle = NULL;
  int l_run_check = 0;
  int l_tc_config = 0;
  int l_n_threads = 1;

# if defined(__APPLE__) && defined(__arm64__)
#  if 1
  pthread_set_qos_class_self_np( QOS_CLASS_USER_INTERACTIVE, 0 );
#  else
  pthread_set_qos_class_self_np( QOS_CLASS_BACKGROUND, 0 );
#  endif
# endif

  /* scaling factor */
  float l_scf = 1.0;

  /* check argument count for a valid range */
  if ( argc == 20 || argc == 19 ) {
    /* xgemm sizes */
    l_m = atoi(argv[1]);
    l_n = atoi(argv[2]);
    l_k = atoi(argv[3]);
    l_lda = atoi(argv[4]);
    l_ldb = atoi(argv[5]);
    l_ldc = atoi(argv[6]);

    /* some sugar */
    l_alpha = atof(argv[7]);
    l_beta = atof(argv[8]);
    l_aligned_a = atoi(argv[9]);
    l_aligned_c = atoi(argv[10]);
    l_trans_a = atoi(argv[11]);
    l_trans_b = atoi(argv[12]);

    /* arch specific stuff */
    l_precision = argv[14];
    l_br = atoi(argv[16]);
    l_br_unroll = atoi(argv[17]);
    g_reps = atoi(argv[18]);
    if ( argc == 20 ) {
      l_tc_config = atoi(argv[19]);
    } else {
      l_tc_config = 0;
    }

    /* set value of prefetch flag */
    if (strcmp("nopf", argv[13]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
    }
    else if (strcmp("pfsigonly", argv[13]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_SIGONLY;
    }
    else if (strcmp("BL2viaC", argv[13]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_BL2_VIA_C;
    }
    else if (strcmp("curAL2", argv[13]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_AL2_AHEAD;
    }
    else if (strcmp("curAL2_BL2viaC", argv[13]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD;
    }
    else if (strcmp("AL2", argv[13]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_AL2;
    }
    else if (strcmp("AL2_BL2viaC", argv[13]) == 0) {
      l_prefetch = LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C;
    }
    else {
      print_help();
      return EXIT_FAILURE;
    }

    if (strcmp("nobr", argv[15]) == 0) {
      l_br_type = 0;
    }
    else if (strcmp("addrbr", argv[15]) == 0) {
      l_br_type = 1;
    }
    else if (strcmp("offsbr", argv[15]) == 0) {
      l_br_type = 2;
    }
    else if (strcmp("strdbr", argv[15]) == 0) {
      l_br_type = 3;
    }
    else {
      print_help();
      return EXIT_FAILURE;
    }

    l_file_input = 0;
    l_run_check = 1;
  } else if ( argc == 15 || argc == 14 ) {
    l_file_input = 1;
    l_file_name = argv[1];
    l_alpha = atof(argv[2]);
    l_beta = atof(argv[3]);
    l_aligned_a = atoi(argv[4]);
    l_aligned_c = atoi(argv[5]);
    l_trans_a = atoi(argv[6]);
    l_trans_b = atoi(argv[7]);
    l_precision = argv[8];
    l_br = atoi(argv[10]);
    l_br_unroll = atoi(argv[11]);
    if ( argc == 15 ) {
      l_tc_config = atoi(argv[14]);
    } else {
      l_tc_config = 0;
    }

    if (strcmp("nobr", argv[9]) == 0) {
      l_br_type = 0;
    }
    else if (strcmp("addrbr", argv[9]) == 0) {
      l_br_type = 1;
    }
    else if (strcmp("offsbr", argv[9]) == 0) {
      l_br_type = 2;
    }
    else if (strcmp("strdbr", argv[9]) == 0) {
      l_br_type = 3;
    }
    else {
      print_help();
      return EXIT_FAILURE;
    }
    g_reps = atoi(argv[12]);
    l_run_check = atoi(argv[13]);
    l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  } else {
    print_help();
    return EXIT_FAILURE;
  }

  const char *env_arch = getenv("LIBXSMM_TARGET");
  const int is_env_SPR = (
      env_arch == libxsmm_stristr(env_arch, "spr") ||
      env_arch == libxsmm_stristr(env_arch, "amx"));
  int arch_cpuid = libxsmm_cpuid();

  if ((!is_env_SPR && arch_cpuid < LIBXSMM_X86_AVX512_SPR)
       && (l_tc_config)) {
    printf("Warning: external tile configuration will be ingnored\n");
    l_tc_config = 0;
  }

  l_br = (l_br < 1) ? 1 : l_br;
  l_br = (l_br_type == 0) ? 1 : l_br;
  l_br_unroll = (l_br_type == 0) ? 0 : l_br_unroll;

  /* check alpha */
  if ( LIBXSMM_NEQ(l_alpha, 1.0) ) {
    fprintf(stderr, "JIT: alpha needs to be 1.0!\n");
    exit(EXIT_FAILURE);
  }

  /* check beta */
  if ( LIBXSMM_NEQ(l_beta, 0.0) && LIBXSMM_NEQ(l_beta, 1.0) ) {
    fprintf(stderr, "JIT: beta needs to be 0.0 or 1.0!\n");
    exit(EXIT_FAILURE);
  }

  /* read the number of threads */
#if defined(_OPENMP)
  #pragma omp parallel
  {
    #pragma omp master
    {
      l_n_threads = omp_get_num_threads();
    }
  }
#endif

  if ( l_file_input != 0 ) {
    l_file_handle = fopen( l_file_name, "r" );
  } else {
    if ( l_trans_b == 0 ) {
      printf("------------------------------------------------\n");
      printf("RUNNING (%ix%i) X (%ix%i) = (%ix%i), %s, BR=%i\n", l_m, l_k, l_k, l_n, l_m, l_n, l_precision, l_br);
      printf("------------------------------------------------\n");
    } else {
      printf("------------------------------------------------\n");
      printf("RUNNING (%ix%i) X (%ix%i)^T = (%ix%i), %s, BR=%i\n", l_m, l_k, l_k, l_n, l_m, l_n, l_precision, l_br);
      printf("------------------------------------------------\n");
    }
  }

  if ((strcmp(l_precision, "DP") == 0) && (l_trans_b == 0)) {
    unsigned int l_keep_going = 0;
    do {
      if ( l_file_input != 0 ) {
        char l_line[512];
        if ( fgets( l_line, 512, l_file_handle) == NULL ) {
          l_keep_going = 0;
          break;
        } else {
          l_keep_going = 1;
        }
        if ( 6 != sscanf( l_line, "%i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_lda, &l_ldb, &l_ldc ) ) exit(EXIT_FAILURE);
      }

      l_gemm_def.m = l_m;
      l_gemm_def.n = l_n;
      l_gemm_def.k = l_k;
      l_gemm_def.lda = l_lda;
      l_gemm_def.ldb = l_ldb;
      l_gemm_def.ldc = l_ldc;
      l_gemm_def.alpha = l_alpha;
      l_gemm_def.beta = l_beta;
      l_gemm_def.trans_a = l_trans_a;
      l_gemm_def.trans_b = l_trans_b;
      l_gemm_def.aligned_a = l_aligned_a;
      l_gemm_def.aligned_c = l_aligned_c;
      l_gemm_def.prefetch = l_prefetch;
      l_gemm_def.br_type = l_br_type;
      l_gemm_def.br_count = l_br;
      l_gemm_def.br_unroll = l_br_unroll;
      l_gemm_def.tc_config = l_tc_config;

#if defined(_OPENMP)
#pragma omp parallel reduction(+:l_runtime_libxsmm)
#endif
      {
        unsigned int l_r, l_i, l_j;
        double* l_a_d = (double*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * sizeof(double), 64);
        double* l_b_d = (double*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * (size_t)l_br * sizeof(double), 64);
        double* l_c_d = (double*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(double), 64);

        /* touch A */
        for (l_r = 0; l_r < l_br; l_r++) {
          for (l_i = 0; l_i < l_lda; l_i++) {
            for (l_j = 0; l_j < l_k; l_j++) {
              l_a_d[(l_r * l_lda * l_k) + ((l_j * l_lda) + l_i)] = libxsmm_rng_f64();
            }
          }
        }
        /* touch B */
        for (l_r = 0; l_r < l_br; l_r++) {
          for (l_i = 0; l_i < l_ldb; l_i++) {
            for (l_j = 0; l_j < l_n; l_j++) {
              l_b_d[(l_r * l_ldb * l_n) + ((l_j * l_ldb) + l_i)] = libxsmm_rng_f64();
            }
          }
        }
        /* touch C */
        for (l_i = 0; l_i < l_ldc; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            l_c_d[(l_j * l_ldc) + l_i] = 0.0;
          }
        }

        l_runtime_libxsmm = run_jit_double( &l_gemm_def, l_a_d, l_b_d, l_c_d, l_file_input );

        libxsmm_free(l_a_d);
        libxsmm_free(l_b_d);
        libxsmm_free(l_c_d);
      }
      l_runtime_libxsmm /= (double)l_n_threads;

      if ( l_file_input == 0 ) {
        printf("avg %fs per core for libxsmm\n", l_runtime_libxsmm);
        printf("avg %f GFLOPS per core for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
      } else {
        printf("%i %i %i %i %i %i %i %i %i %s %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
      }
    } while ( l_keep_going );
  }
  else if ((strcmp(l_precision, "DP") == 0) && (l_trans_b != 0)) {
    unsigned int l_keep_going = 0;
    do {
      if ( l_file_input != 0 ) {
        char l_line[512];
        if ( fgets( l_line, 512, l_file_handle) == NULL ) {
          l_keep_going = 0;
          break;
        } else {
          l_keep_going = 1;
        }
        if ( 6 != sscanf( l_line, "%i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_lda, &l_ldb, &l_ldc ) ) exit(EXIT_FAILURE);
      }

      l_gemm_def.m = l_m;
      l_gemm_def.n = l_n;
      l_gemm_def.k = l_k;
      l_gemm_def.lda = l_lda;
      l_gemm_def.ldb = l_ldb;
      l_gemm_def.ldc = l_ldc;
      l_gemm_def.alpha = l_alpha;
      l_gemm_def.beta = l_beta;
      l_gemm_def.trans_a = l_trans_a;
      l_gemm_def.trans_b = l_trans_b;
      l_gemm_def.aligned_a = l_aligned_a;
      l_gemm_def.aligned_c = l_aligned_c;
      l_gemm_def.prefetch = l_prefetch;
      l_gemm_def.br_type = l_br_type;
      l_gemm_def.br_count = l_br;
      l_gemm_def.br_unroll = l_br_unroll;
      l_gemm_def.tc_config = l_tc_config;

#if defined(_OPENMP)
#pragma omp parallel reduction(+:l_runtime_libxsmm)
#endif
      {
        unsigned int l_r, l_i, l_j;
        double* l_a_d = (double*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * sizeof(double), 64);
        double* l_b_d = (double*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_k * (size_t)l_br * sizeof(double), 64);
        double* l_c_d = (double*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(double), 64);

        /* touch A */
        for (l_r = 0; l_r < l_br; l_r++) {
          for (l_i = 0; l_i < l_lda; l_i++) {
            for (l_j = 0; l_j < l_k; l_j++) {
              l_a_d[(l_r * l_lda * l_k) + ((l_j * l_lda) + l_i)] = libxsmm_rng_f64();
            }
          }
        }
        /* touch B */
        for (l_r = 0; l_r < l_br; l_r++) {
          for (l_i = 0; l_i < l_ldb; l_i++) {
            for (l_j = 0; l_j < l_n; l_j++) {
              l_b_d[(l_r * l_ldb * l_n) + ((l_j * l_ldb) + l_i)] = libxsmm_rng_f64();
            }
          }
        }
        /* touch C */
        for (l_i = 0; l_i < l_ldc; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            l_c_d[(l_j * l_ldc) + l_i] = 0.0;
          }
        }

        l_runtime_libxsmm = run_jit_double( &l_gemm_def, l_a_d, l_b_d, l_c_d, l_file_input );

        libxsmm_free(l_a_d);
        libxsmm_free(l_b_d);
        libxsmm_free(l_c_d);
      }
      l_runtime_libxsmm /= (double)l_n_threads;

      if ( l_file_input == 0 ) {
        printf("avg %fs per core for libxsmm\n", l_runtime_libxsmm);
        printf("avg %f GFLOPS per core for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
      } else {
        printf("%i %i %i %i %i %i %i %i %i %s %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
      }
    } while ( l_keep_going );
  }
  else if ((strcmp(l_precision, "SP") == 0) && (l_trans_b == 0)) {
    unsigned int l_keep_going = 0;
    do {
      if ( l_file_input != 0 ) {
        char l_line[512];
        if ( fgets( l_line, 512, l_file_handle) == NULL ) {
          l_keep_going = 0;
          break;
        } else {
          l_keep_going = 1;
        }
        if ( 6 != sscanf( l_line, "%i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_lda, &l_ldb, &l_ldc ) ) exit(EXIT_FAILURE);
      }

      l_gemm_def.m = l_m;
      l_gemm_def.n = l_n;
      l_gemm_def.k = l_k;
      l_gemm_def.lda = l_lda;
      l_gemm_def.ldb = l_ldb;
      l_gemm_def.ldc = l_ldc;
      l_gemm_def.alpha = l_alpha;
      l_gemm_def.beta = l_beta;
      l_gemm_def.trans_a = l_trans_a;
      l_gemm_def.trans_b = l_trans_b;
      l_gemm_def.aligned_a = l_aligned_a;
      l_gemm_def.aligned_c = l_aligned_c;
      l_gemm_def.prefetch = l_prefetch;
      l_gemm_def.br_type = l_br_type;
      l_gemm_def.br_count = l_br;
      l_gemm_def.br_unroll = l_br_unroll;
      l_gemm_def.tc_config = l_tc_config;

#if defined(_OPENMP)
#pragma omp parallel reduction(+:l_runtime_libxsmm)
#endif
      {
        unsigned int l_r, l_i, l_j;
        float* l_a_f = (float*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * sizeof(float), 64);
        float* l_b_f = (float*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * (size_t)l_br * sizeof(float), 64);
        float* l_c_f = (float*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(float), 64);

        /* touch A */
        for (l_r = 0; l_r < l_br; l_r++) {
          for (l_i = 0; l_i < l_lda; l_i++) {
            for (l_j = 0; l_j < l_k; l_j++) {
              l_a_f[(l_r * l_lda * l_k) + ((l_j * l_lda) + l_i)] = (float)libxsmm_rng_f64();
            }
          }
        }
        /* touch B */
        for (l_r = 0; l_r < l_br; l_r++) {
          for (l_i = 0; l_i < l_ldb; l_i++) {
            for (l_j = 0; l_j < l_n; l_j++) {
              l_b_f[(l_r * l_ldb * l_n) + ((l_j * l_ldb) + l_i)] = (float)libxsmm_rng_f64();
            }
          }
        }
        /* touch C */
        for (l_i = 0; l_i < l_ldc; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            l_c_f[(l_j * l_ldc) + l_i] = 0.f;
          }
        }

        l_runtime_libxsmm = run_jit_float( &l_gemm_def, l_a_f, l_b_f, l_c_f, l_file_input );

        libxsmm_free(l_a_f);
        libxsmm_free(l_b_f);
        libxsmm_free(l_c_f);
      }
      l_runtime_libxsmm /= (double)l_n_threads;

      if ( l_file_input == 0 ) {
        printf("avg %fs per core for libxsmm\n", l_runtime_libxsmm);
        printf("avg %f GFLOPS per core for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
      } else {
        printf("%i %i %i %i %i %i %i %i %i %s %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
      }
    } while ( l_keep_going );
  }
  else if ((strcmp(l_precision, "SP") == 0) && (l_trans_b != 0)) {
    unsigned int l_keep_going = 0;
    do {
      if ( l_file_input != 0 ) {
        char l_line[512];
        if ( fgets( l_line, 512, l_file_handle) == NULL ) {
          l_keep_going = 0;
          break;
        } else {
          l_keep_going = 1;
        }
        if ( 6 != sscanf( l_line, "%i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_lda, &l_ldb, &l_ldc ) ) exit(EXIT_FAILURE);
      }

      l_gemm_def.m = l_m;
      l_gemm_def.n = l_n;
      l_gemm_def.k = l_k;
      l_gemm_def.lda = l_lda;
      l_gemm_def.ldb = l_ldb;
      l_gemm_def.ldc = l_ldc;
      l_gemm_def.alpha = l_alpha;
      l_gemm_def.beta = l_beta;
      l_gemm_def.trans_a = l_trans_a;
      l_gemm_def.trans_b = l_trans_b;
      l_gemm_def.aligned_a = l_aligned_a;
      l_gemm_def.aligned_c = l_aligned_c;
      l_gemm_def.prefetch = l_prefetch;
      l_gemm_def.br_type = l_br_type;
      l_gemm_def.br_count = l_br;
      l_gemm_def.br_unroll = l_br_unroll;
      l_gemm_def.tc_config = l_tc_config;

#if defined(_OPENMP)
#pragma omp parallel reduction(+:l_runtime_libxsmm)
#endif
      {
        unsigned int l_r, l_i, l_j;
        float* l_a_f = (float*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * sizeof(float), 64);
        float* l_b_f = (float*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_k * (size_t)l_br * sizeof(float), 64);
        float* l_c_f = (float*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(float), 64);

        /* touch A */
        for (l_r = 0; l_r < l_br; l_r++) {
          for (l_i = 0; l_i < l_lda; l_i++) {
            for (l_j = 0; l_j < l_k; l_j++) {
              l_a_f[(l_r * l_lda * l_k) + ((l_j * l_lda) + l_i)] = (float)libxsmm_rng_f64();
            }
          }
        }
        /* touch B */
        for (l_r = 0; l_r < l_br; l_r++) {
          for (l_i = 0; l_i < l_ldb; l_i++) {
            for (l_j = 0; l_j < l_n; l_j++) {
              l_b_f[(l_r * l_ldb * l_n) + ((l_j * l_ldb) + l_i)] = (float)libxsmm_rng_f64();
            }
          }
        }
        /* touch C */
        for (l_i = 0; l_i < l_ldc; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            l_c_f[(l_j * l_ldc) + l_i] = 0.f;
          }
        }

        l_runtime_libxsmm = run_jit_float( &l_gemm_def, l_a_f, l_b_f, l_c_f, l_file_input );

        libxsmm_free(l_a_f);
        libxsmm_free(l_b_f);
        libxsmm_free(l_c_f);
      }
      l_runtime_libxsmm /= (double)l_n_threads;

      if ( l_file_input == 0 ) {
        printf("avg %fs per core for libxsmm\n", l_runtime_libxsmm);
        printf("avg %f GFLOPS per core for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
      } else {
        printf("%i %i %i %i %i %i %i %i %i %s %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
      }
    } while ( l_keep_going );
  } else if (strcmp(l_precision, "I16I32") == 0) {
    unsigned int l_keep_going = 0;
    do {
      if ( l_file_input != 0 ) {
        char l_line[512];
        if ( fgets( l_line, 512, l_file_handle) == NULL ) {
          l_keep_going = 0;
          break;
        } else {
          l_keep_going = 1;
        }
        if ( 6 != sscanf( l_line, "%i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_lda, &l_ldb, &l_ldc ) ) exit(EXIT_FAILURE);
      }

      l_gemm_def.m = l_m;
      l_gemm_def.n = l_n;
      l_gemm_def.k = l_k;
      l_gemm_def.lda = l_lda;
      l_gemm_def.ldb = l_ldb;
      l_gemm_def.ldc = l_ldc;
      l_gemm_def.alpha = l_alpha;
      l_gemm_def.beta = l_beta;
      l_gemm_def.trans_a = l_trans_a;
      l_gemm_def.trans_b = l_trans_b;
      l_gemm_def.aligned_a = l_aligned_a;
      l_gemm_def.aligned_c = l_aligned_c;
      l_gemm_def.prefetch = l_prefetch;
      l_gemm_def.br_type = l_br_type;
      l_gemm_def.br_count = l_br;
      l_gemm_def.br_unroll = l_br_unroll;
      l_gemm_def.tc_config = l_tc_config;

#if defined(_OPENMP)
#pragma omp parallel reduction(+:l_runtime_libxsmm)
#endif
      {
        unsigned int l_r, l_i, l_j;
        short* l_a_w = (short*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * sizeof(short), 64);
        short* l_b_w = (short*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * (size_t)l_br * sizeof(short), 64);
        int* l_c_w_i = (int*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(int), 64);

        /* touch A */
        for (l_r = 0; l_r < l_br; l_r++) {
          for (l_i = 0; l_i < l_lda; l_i++) {
            for (l_j = 0; l_j < l_k; l_j++) {
              l_a_w[(l_r * l_lda * l_k) + ((l_j * l_lda) + l_i)] = (short)(libxsmm_rng_f64() * 10.0);
            }
          }
        }
        /* touch B */
        for (l_r = 0; l_r < l_br; l_r++) {
          for (l_i = 0; l_i < l_ldb; l_i++) {
            for (l_j = 0; l_j < l_n; l_j++) {
              l_b_w[(l_r * l_ldb * l_n) + ((l_j * l_ldb) + l_i)] = (short)(libxsmm_rng_f64() * 10.0);
            }
          }
        }
        /* touch C */
        for (l_i = 0; l_i < l_ldc; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            l_c_w_i[(l_j * l_ldc) + l_i] = 0;
          }
        }

        l_runtime_libxsmm = run_jit_short_int( &l_gemm_def, l_a_w, l_b_w, l_c_w_i, l_file_input );

        libxsmm_free(l_a_w);
        libxsmm_free(l_b_w);
        libxsmm_free(l_c_w_i);
      }
      l_runtime_libxsmm /= (double)l_n_threads;

      if ( l_file_input == 0 ) {
        printf("avg %fs per core for libxsmm\n", l_runtime_libxsmm);
        printf("avg %f GFLOPS per oore for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
      } else {
        printf("%i %i %i %i %i %i %i %i %i %s %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
      }
    } while ( l_keep_going );
  } else if (strcmp(l_precision, "USI8I32") == 0) {
    unsigned int l_keep_going = 0;
    do {
      if ( l_file_input != 0 ) {
        char l_line[512];
        if ( fgets( l_line, 512, l_file_handle) == NULL ) {
          l_keep_going = 0;
          break;
        } else {
          l_keep_going = 1;
        }
        if ( 6 != sscanf( l_line, "%i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_lda, &l_ldb, &l_ldc ) ) exit(EXIT_FAILURE);
      }

      l_gemm_def.m = l_m;
      l_gemm_def.n = l_n;
      l_gemm_def.k = l_k;
      l_gemm_def.lda = l_lda;
      l_gemm_def.ldb = l_ldb;
      l_gemm_def.ldc = l_ldc;
      l_gemm_def.alpha = l_alpha;
      l_gemm_def.beta = l_beta;
      l_gemm_def.trans_a = l_trans_a;
      l_gemm_def.trans_b = l_trans_b;
      l_gemm_def.aligned_a = l_aligned_a;
      l_gemm_def.aligned_c = l_aligned_c;
      l_gemm_def.prefetch = l_prefetch;
      l_gemm_def.br_type = l_br_type;
      l_gemm_def.br_count = l_br;
      l_gemm_def.br_unroll = l_br_unroll;
      l_gemm_def.tc_config = l_tc_config;

#if defined(_OPENMP)
#pragma omp parallel reduction(+:l_runtime_libxsmm)
#endif
      {
        unsigned int l_r, l_i, l_j;
        unsigned char* l_ua_b = (unsigned char*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * sizeof(unsigned char), 64);
        char* l_sb_b = (char*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * (size_t)l_br * sizeof(char), 64);
        int* l_c_b_i = (int*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(int), 64);

        /* touch A */
        for (l_r = 0; l_r < l_br; l_r++) {
          for (l_i = 0; l_i < l_lda; l_i++) {
            for (l_j = 0; l_j < l_k; l_j++) {
              l_ua_b[(l_r * l_lda * l_k) + ((l_j * l_lda) + l_i)] = (unsigned char)(libxsmm_rng_f64() * 5.0);
            }
          }
        }
        /* touch B */
        for (l_r = 0; l_r < l_br; l_r++) {
          for (l_i = 0; l_i < l_ldb; l_i++) {
            for (l_j = 0; l_j < l_n; l_j++) {
              l_sb_b[(l_r * l_ldb * l_n) + ((l_j * l_ldb) + l_i)] = (char)(libxsmm_rng_f64() * 5.0);
            }
          }
        }
        /* touch C */
        for (l_i = 0; l_i < l_ldc; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            l_c_b_i[(l_j * l_ldc) + l_i] = 0;
          }
        }

        l_runtime_libxsmm = run_jit_uschar_int( &l_gemm_def, l_ua_b, l_sb_b, l_c_b_i, l_file_input );

        libxsmm_free(l_ua_b);
        libxsmm_free(l_sb_b);
        libxsmm_free(l_c_b_i);
      }
      l_runtime_libxsmm /= (double)l_n_threads;

      if ( l_file_input == 0 ) {
        printf("avg %fs per core for libxsmm\n", l_runtime_libxsmm);
        printf("avg %f GFLOPS per core for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
      } else {
        printf("%i %i %i %i %i %i %i %i %i %s %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
      }
    } while ( l_keep_going );
  } else if (strcmp(l_precision, "SUI8I32") == 0) {
    unsigned int l_keep_going = 0;
    do {
      if ( l_file_input != 0 ) {
        char l_line[512];
        if ( fgets( l_line, 512, l_file_handle) == NULL ) {
          l_keep_going = 0;
          break;
        } else {
          l_keep_going = 1;
        }
        if ( 6 != sscanf( l_line, "%i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_lda, &l_ldb, &l_ldc ) ) exit(EXIT_FAILURE);
      }

      l_gemm_def.m = l_m;
      l_gemm_def.n = l_n;
      l_gemm_def.k = l_k;
      l_gemm_def.lda = l_lda;
      l_gemm_def.ldb = l_ldb;
      l_gemm_def.ldc = l_ldc;
      l_gemm_def.alpha = l_alpha;
      l_gemm_def.beta = l_beta;
      l_gemm_def.trans_a = l_trans_a;
      l_gemm_def.trans_b = l_trans_b;
      l_gemm_def.aligned_a = l_aligned_a;
      l_gemm_def.aligned_c = l_aligned_c;
      l_gemm_def.prefetch = l_prefetch;
      l_gemm_def.br_type = l_br_type;
      l_gemm_def.br_count = l_br;
      l_gemm_def.br_unroll = l_br_unroll;
      l_gemm_def.tc_config = l_tc_config;

#if defined(_OPENMP)
#pragma omp parallel reduction(+:l_runtime_libxsmm)
#endif
      {
        unsigned int l_r, l_i, l_j;
        char* l_sa_b = (char*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * sizeof(char), 64);
        unsigned char* l_ub_b = (unsigned char*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * (size_t)l_br * sizeof(unsigned char), 64);
        int* l_c_b_i = (int*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(int), 64);

        /* touch A */
        for (l_r = 0; l_r < l_br; l_r++) {
          for (l_i = 0; l_i < l_lda; l_i++) {
            for (l_j = 0; l_j < l_k; l_j++) {
              l_sa_b[(l_r * l_lda * l_k) + ((l_j * l_lda) + l_i)] = (char)(libxsmm_rng_f64() * 5.0);
            }
          }
        }
        /* touch B */
        for (l_r = 0; l_r < l_br; l_r++) {
          for (l_i = 0; l_i < l_ldb; l_i++) {
            for (l_j = 0; l_j < l_n; l_j++) {
              l_ub_b[(l_r * l_ldb * l_n) + ((l_j * l_ldb) + l_i)] = (unsigned char)(libxsmm_rng_f64() * 5.0);
            }
          }
        }
        /* touch C */
        for (l_i = 0; l_i < l_ldc; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            l_c_b_i[(l_j * l_ldc) + l_i] = 0;
          }
        }

        l_runtime_libxsmm = run_jit_suchar_int( &l_gemm_def, l_sa_b, l_ub_b, l_c_b_i, l_file_input );

        libxsmm_free(l_sa_b);
        libxsmm_free(l_ub_b);
        libxsmm_free(l_c_b_i);
      }
      l_runtime_libxsmm /= (double)l_n_threads;

      if ( l_file_input == 0 ) {
        printf("avg %fs per core for libxsmm\n", l_runtime_libxsmm);
        printf("avg %f GFLOPS per core for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
      } else {
        printf("%i %i %i %i %i %i %i %i %i %s %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
      }
    } while ( l_keep_going );
  } else if (strcmp(l_precision, "SUI8UI8") == 0) {
    unsigned int l_keep_going = 0;
    do {
      if ( l_file_input != 0 ) {
        char l_line[512];
        if ( fgets( l_line, 512, l_file_handle) == NULL ) {
          l_keep_going = 0;
          break;
        } else {
          l_keep_going = 1;
        }
        if ( 6 != sscanf( l_line, "%i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_lda, &l_ldb, &l_ldc ) ) exit(EXIT_FAILURE);
      }

      l_gemm_def.m = l_m;
      l_gemm_def.n = l_n;
      l_gemm_def.k = l_k;
      l_gemm_def.lda = l_lda;
      l_gemm_def.ldb = l_ldb;
      l_gemm_def.ldc = l_ldc;
      l_gemm_def.alpha = l_alpha;
      l_gemm_def.beta = l_beta;
      l_gemm_def.trans_a = l_trans_a;
      l_gemm_def.trans_b = l_trans_b;
      l_gemm_def.aligned_a = l_aligned_a;
      l_gemm_def.aligned_c = l_aligned_c;
      l_gemm_def.prefetch = l_prefetch;
      l_gemm_def.br_type = l_br_type;
      l_gemm_def.br_count = l_br;
      l_gemm_def.br_unroll = l_br_unroll;
      l_gemm_def.tc_config = l_tc_config;

#if defined(_OPENMP)
#pragma omp parallel reduction(+:l_runtime_libxsmm)
#endif
      {
        unsigned int l_r, l_i, l_j;
        char* l_sa_b = (char*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * sizeof(char), 64);
        unsigned char* l_ub_b = (unsigned char*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * (size_t)l_br * sizeof(unsigned char), 64);
        unsigned char* l_c_b_ub = (unsigned char*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(unsigned char), 64);

        /* touch A */
        for (l_r = 0; l_r < l_br; l_r++) {
          for (l_i = 0; l_i < l_lda; l_i++) {
            for (l_j = 0; l_j < l_k; l_j++) {
              l_sa_b[(l_r * l_lda * l_k) + ((l_j * l_lda) + l_i)] = (char)(libxsmm_rng_f64() * 2.0);
            }
          }
        }
        /* touch B */
        for (l_r = 0; l_r < l_br; l_r++) {
          for (l_i = 0; l_i < l_ldb; l_i++) {
            for (l_j = 0; l_j < l_n; l_j++) {
              l_ub_b[(l_r * l_ldb * l_n) + ((l_j * l_ldb) + l_i)] = (unsigned char)(libxsmm_rng_f64() * 2.0);
            }
          }
        }
        /* touch C */
        for (l_i = 0; l_i < l_ldc; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            l_c_b_ub[(l_j * l_ldc) + l_i] = 0;
          }
        }

        l_runtime_libxsmm = run_jit_suchar_uchar( &l_gemm_def, l_sa_b, l_ub_b, l_c_b_ub, l_scf, l_file_input );

        libxsmm_free(l_sa_b);
        libxsmm_free(l_ub_b);
        libxsmm_free(l_c_b_ub);
      }
      l_runtime_libxsmm /= (double)l_n_threads;

      if ( l_file_input == 0 ) {
        printf("avg %fs per core for libxsmm\n", l_runtime_libxsmm);
        printf("avg %f GFLOPS per core for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
      } else {
        printf("%i %i %i %i %i %i %i %i %i %s %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
      }
    } while ( l_keep_going );
  } else if (strcmp(l_precision, "BF16F32") == 0) {
    unsigned int l_keep_going = 0;
    do {
      if ( l_file_input != 0 ) {
        char l_line[512];
        if ( fgets( l_line, 512, l_file_handle) == NULL ) {
          l_keep_going = 0;
          break;
        } else {
          l_keep_going = 1;
        }
        if ( 6 != sscanf( l_line, "%i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_lda, &l_ldb, &l_ldc ) ) exit(EXIT_FAILURE);
      }

      l_gemm_def.m = l_m;
      l_gemm_def.n = l_n;
      l_gemm_def.k = l_k;
      l_gemm_def.lda = l_lda;
      l_gemm_def.ldb = l_ldb;
      l_gemm_def.ldc = l_ldc;
      l_gemm_def.alpha = l_alpha;
      l_gemm_def.beta = l_beta;
      l_gemm_def.trans_a = l_trans_a;
      l_gemm_def.trans_b = l_trans_b;
      l_gemm_def.aligned_a = l_aligned_a;
      l_gemm_def.aligned_c = l_aligned_c;
      l_gemm_def.prefetch = l_prefetch;
      l_gemm_def.br_type = l_br_type;
      l_gemm_def.br_count = l_br;
      l_gemm_def.br_unroll = l_br_unroll;
      l_gemm_def.tc_config = l_tc_config;

#if defined(_OPENMP)
#pragma omp parallel reduction(+:l_runtime_libxsmm)
#endif
      {
        unsigned int l_r, l_i, l_j;
        libxsmm_bfloat16* l_a_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * sizeof(libxsmm_bfloat16), 64);
        libxsmm_bfloat16* l_b_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * (size_t)l_br * sizeof(libxsmm_bfloat16), 64);
        float* l_c_bf_f = (float*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(float), 64);

        /* touch A */
        for (l_r = 0; l_r < l_br; l_r++) {
          for (l_i = 0; l_i < l_lda; l_i++) {
            for (l_j = 0; l_j < l_k; l_j++) {
              union libxsmm_bfloat16_hp tmp;
              tmp.f = (float)libxsmm_rng_f64();
              l_a_bf[(l_r * l_lda * l_k) + (l_j * l_lda) + l_i] = tmp.i[1];
            }
          }
        }
        /* touch B */
        for (l_r = 0; l_r < l_br; l_r++) {
          for (l_i = 0; l_i < l_ldb; l_i++) {
            for (l_j = 0; l_j < l_n; l_j++) {
              union libxsmm_bfloat16_hp tmp;
              tmp.f = (float)libxsmm_rng_f64();
              l_b_bf[(l_r * l_ldb * l_n) + (l_j * l_ldb) + l_i] = tmp.i[1];
            }
          }
        }
        /* touch C */
        for (l_i = 0; l_i < l_ldc; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            l_c_bf_f[(l_j * l_ldc) + l_i] = 0.0f;
          }
        }

        l_runtime_libxsmm = run_jit_bfloat16_float( &l_gemm_def, l_a_bf, l_b_bf, l_c_bf_f, l_file_input );

        libxsmm_free(l_a_bf);
        libxsmm_free(l_b_bf);
        libxsmm_free(l_c_bf_f);
      }
      l_runtime_libxsmm /= (double)l_n_threads;

      if ( l_file_input == 0 ) {
        printf("avg %fs per core for libxsmm\n", l_runtime_libxsmm);
        printf("avg %f GFLOPS per core for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
      } else {
        printf("%i %i %i %i %i %i %i %i %i %s %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
      }
    } while ( l_keep_going );
  } else if (strcmp(l_precision, "BF16") == 0) {
    unsigned int l_keep_going = 0;
    do {
      if ( l_file_input != 0 ) {
        char l_line[512];
        if ( fgets( l_line, 512, l_file_handle) == NULL ) {
          l_keep_going = 0;
          break;
        } else {
          l_keep_going = 1;
        }
        if ( 6 != sscanf( l_line, "%i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_lda, &l_ldb, &l_ldc ) ) exit(EXIT_FAILURE);
      }

      l_gemm_def.m = l_m;
      l_gemm_def.n = l_n;
      l_gemm_def.k = l_k;
      l_gemm_def.lda = l_lda;
      l_gemm_def.ldb = l_ldb;
      l_gemm_def.ldc = l_ldc;
      l_gemm_def.alpha = l_alpha;
      l_gemm_def.beta = l_beta;
      l_gemm_def.trans_a = l_trans_a;
      l_gemm_def.trans_b = l_trans_b;
      l_gemm_def.aligned_a = l_aligned_a;
      l_gemm_def.aligned_c = l_aligned_c;
      l_gemm_def.prefetch = l_prefetch;
      l_gemm_def.br_type = l_br_type;
      l_gemm_def.br_count = l_br;
      l_gemm_def.br_unroll = l_br_unroll;
      l_gemm_def.tc_config = l_tc_config;

#if defined(_OPENMP)
#pragma omp parallel reduction(+:l_runtime_libxsmm)
#endif
      {
        unsigned int l_r, l_i, l_j;
        libxsmm_bfloat16* l_a_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * sizeof(libxsmm_bfloat16), 64);
        libxsmm_bfloat16* l_b_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * (size_t)l_br * sizeof(libxsmm_bfloat16), 64);
        libxsmm_bfloat16* l_c_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(libxsmm_bfloat16), 64);

        /* touch A */
        for (l_r = 0; l_r < l_br; l_r++) {
          for (l_i = 0; l_i < l_lda; l_i++) {
            for (l_j = 0; l_j < l_k; l_j++) {
              union libxsmm_bfloat16_hp tmp;
              tmp.f = (float)libxsmm_rng_f64();
              l_a_bf[(l_r * l_lda * l_k) + (l_j * l_lda) + l_i] = tmp.i[1];
            }
          }
        }
        /* touch B */
        for (l_r = 0; l_r < l_br; l_r++) {
          for (l_i = 0; l_i < l_ldb; l_i++) {
            for (l_j = 0; l_j < l_n; l_j++) {
              union libxsmm_bfloat16_hp tmp;
              tmp.f = (float)libxsmm_rng_f64();
              l_b_bf[(l_r * l_ldb * l_n) + (l_j * l_ldb) + l_i] = tmp.i[1];
            }
          }
        }
        /* touch C */
        for (l_i = 0; l_i < l_ldc; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            union libxsmm_bfloat16_hp tmp;
            tmp.f = 0.0f;
            l_c_bf[(l_j * l_ldc) + l_i] = tmp.i[1];
          }
        }

        l_runtime_libxsmm = run_jit_bfloat16( &l_gemm_def, l_a_bf, l_b_bf, l_c_bf, l_file_input );

        libxsmm_free(l_a_bf);
        libxsmm_free(l_b_bf);
        libxsmm_free(l_c_bf);
      }
      l_runtime_libxsmm /= (double)l_n_threads;

      if ( l_file_input == 0 ) {
        printf("avg %fs per core for libxsmm\n", l_runtime_libxsmm);
        printf("avg %f GFLOPS per core for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
      } else {
        printf("%i %i %i %i %i %i %i %i %i %s %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
      }
    } while ( l_keep_going );
  } else if (strcmp(l_precision, "BF16F32_FLAT") == 0) {
    unsigned int l_keep_going = 0;
    do {
      if ( l_file_input != 0 ) {
        char l_line[512];
        if ( fgets( l_line, 512, l_file_handle) == NULL ) {
          l_keep_going = 0;
          break;
        } else {
          l_keep_going = 1;
        }
        if ( 6 != sscanf( l_line, "%i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_lda, &l_ldb, &l_ldc ) ) exit(EXIT_FAILURE);
      }

      l_gemm_def.m = l_m;
      l_gemm_def.n = l_n;
      l_gemm_def.k = l_k;
      l_gemm_def.lda = l_lda;
      l_gemm_def.ldb = l_ldb;
      l_gemm_def.ldc = l_ldc;
      l_gemm_def.alpha = l_alpha;
      l_gemm_def.beta = l_beta;
      l_gemm_def.trans_a = l_trans_a;
      l_gemm_def.trans_b = l_trans_b;
      l_gemm_def.aligned_a = l_aligned_a;
      l_gemm_def.aligned_c = l_aligned_c;
      l_gemm_def.prefetch = l_prefetch;
      l_gemm_def.br_type = l_br_type;
      l_gemm_def.br_count = l_br;
      l_gemm_def.br_unroll = l_br_unroll;
      l_gemm_def.tc_config = l_tc_config;

#if defined(_OPENMP)
#pragma omp parallel reduction(+:l_runtime_libxsmm)
#endif
      {
        unsigned int l_r, l_i, l_j;
        libxsmm_bfloat16* l_a_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * sizeof(libxsmm_bfloat16), 64);
        libxsmm_bfloat16* l_b_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * (size_t)l_br * sizeof(libxsmm_bfloat16), 64);
        float* l_c_bf_f = (float*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(float), 64);

        /* touch A */
        for (l_r = 0; l_r < l_br; l_r++) {
          for (l_i = 0; l_i < l_lda; l_i++) {
            for (l_j = 0; l_j < l_k; l_j++) {
              union libxsmm_bfloat16_hp tmp;
              tmp.f = (float)libxsmm_rng_f64();
              l_a_bf[(l_r * l_lda * l_k) + (l_j * l_lda) + l_i] = tmp.i[1];
            }
          }
        }
        /* touch B */
        for (l_r = 0; l_r < l_br; l_r++) {
          for (l_i = 0; l_i < l_ldb; l_i++) {
            for (l_j = 0; l_j < l_n; l_j++) {
              union libxsmm_bfloat16_hp tmp;
              tmp.f = (float)libxsmm_rng_f64();
              l_b_bf[(l_r * l_ldb * l_n) + (l_j * l_ldb) + l_i] = tmp.i[1];
            }
          }
        }
        /* touch C */
        for (l_i = 0; l_i < l_ldc; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            l_c_bf_f[(l_j * l_ldc) + l_i] = 0.f;
          }
        }

        l_runtime_libxsmm = run_jit_bfloat16_float_flat( &l_gemm_def, l_a_bf, l_b_bf, l_c_bf_f, l_file_input );

        libxsmm_free(l_a_bf);
        libxsmm_free(l_b_bf);
        libxsmm_free(l_c_bf_f);
      }
      l_runtime_libxsmm /= (double)l_n_threads;

      if ( l_file_input == 0 ) {
        printf("avg %fs per core for libxsmm\n", l_runtime_libxsmm);
        printf("avg %f GFLOPS per core for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
      } else {
        printf("%i %i %i %i %i %i %i %i %i %s %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
      }
    } while ( l_keep_going );
  } else if (strcmp(l_precision, "BF16_FLAT") == 0) {
    unsigned int l_keep_going = 0;
    do {
      if ( l_file_input != 0 ) {
        char l_line[512];
        if ( fgets( l_line, 512, l_file_handle) == NULL ) {
          l_keep_going = 0;
          break;
        } else {
          l_keep_going = 1;
        }
        if ( 6 != sscanf( l_line, "%i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_lda, &l_ldb, &l_ldc ) ) exit(EXIT_FAILURE);
      }

      l_gemm_def.m = l_m;
      l_gemm_def.n = l_n;
      l_gemm_def.k = l_k;
      l_gemm_def.lda = l_lda;
      l_gemm_def.ldb = l_ldb;
      l_gemm_def.ldc = l_ldc;
      l_gemm_def.alpha = l_alpha;
      l_gemm_def.beta = l_beta;
      l_gemm_def.trans_a = l_trans_a;
      l_gemm_def.trans_b = l_trans_b;
      l_gemm_def.aligned_a = l_aligned_a;
      l_gemm_def.aligned_c = l_aligned_c;
      l_gemm_def.prefetch = l_prefetch;
      l_gemm_def.br_type = l_br_type;
      l_gemm_def.br_count = l_br;
      l_gemm_def.br_unroll = l_br_unroll;
      l_gemm_def.tc_config = l_tc_config;

#if defined(_OPENMP)
#pragma omp parallel reduction(+:l_runtime_libxsmm)
#endif
      {
        unsigned int l_r, l_i, l_j;
        libxsmm_bfloat16* l_a_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * sizeof(libxsmm_bfloat16), 64);
        libxsmm_bfloat16* l_b_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * (size_t)l_br * sizeof(libxsmm_bfloat16), 64);
        libxsmm_bfloat16* l_c_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(libxsmm_bfloat16), 64);

        /* touch A */
        for (l_r = 0; l_r < l_br; l_r++) {
          for (l_i = 0; l_i < l_lda; l_i++) {
            for (l_j = 0; l_j < l_k; l_j++) {
              union libxsmm_bfloat16_hp tmp;
              tmp.f = (float)libxsmm_rng_f64();
              l_a_bf[(l_r * l_lda * l_k) + (l_j * l_lda) + l_i] = tmp.i[1];
            }
          }
        }
        /* touch B */
        for (l_r = 0; l_r < l_br; l_r++) {
          for (l_i = 0; l_i < l_ldb; l_i++) {
            for (l_j = 0; l_j < l_n; l_j++) {
              union libxsmm_bfloat16_hp tmp;
              tmp.f = (float)libxsmm_rng_f64();
              l_b_bf[(l_r * l_ldb * l_n) + (l_j * l_ldb) + l_i] = tmp.i[1];
            }
          }
        }
        /* touch C */
        for (l_i = 0; l_i < l_ldc; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            union libxsmm_bfloat16_hp tmp;
            tmp.f = 0.0f;
            l_c_bf[(l_j * l_ldc) + l_i] = tmp.i[1];
          }
        }

        l_runtime_libxsmm = run_jit_bfloat16_flat( &l_gemm_def, l_a_bf, l_b_bf, l_c_bf, l_file_input );

        libxsmm_free(l_a_bf);
        libxsmm_free(l_b_bf);
        libxsmm_free(l_c_bf);
      }
      l_runtime_libxsmm /= (double)l_n_threads;

      if ( l_file_input == 0 ) {
        printf("avg %fs per core for libxsmm\n", l_runtime_libxsmm);
        printf("avg %f GFLOPS per core for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
      } else {
        printf("%i %i %i %i %i %i %i %i %i %s %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, l_br_type, l_br_unroll, l_precision, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
      }
    } while ( l_keep_going );
  }

  if ( l_file_input != 0 ) {
    fclose( l_file_handle );
  } else {
    printf("------------------------------------------------\n");
  }

  LIBXSMM_UNUSED( l_run_check );

  return EXIT_SUCCESS;
}

