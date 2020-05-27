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


typedef struct gemm_def {
  libxsmm_blasint m;
  libxsmm_blasint n;
  libxsmm_blasint k;
  libxsmm_blasint lda;
  libxsmm_blasint ldb;
  libxsmm_blasint ldc;
  double alpha;
  double beta;
  int trans_a;
  int trans_b;
  int aligned_a;
  int aligned_c;
  int prefetch;
  int br_type;
  libxsmm_blasint br_count;
  int br_unroll;
} gemm_def;

int g_reps = 0;

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
  printf("    PRECISION: SP, DP, I16I32, USI8I32, SUI8I32, SUI8UI8, BF16F32, BF16\n");
  printf("    BRGEMM: nobr, addrbr, offsbr, strdbr\n");
  printf("    brsize: 1 - N\n");
  printf("    #repetitions\n");
  printf("\n\n");
  printf("2. Usage (dense*dense=dense, performance only option available):\n");
  printf("    filename with space-sperated sizes (M N K LDA LDB LDC)\n");
  printf("    alpha: 1\n");
  printf("    beta: 0 or 1\n");
  printf("    0: unaligned A, otherwise aligned\n");
  printf("    0: unaligned C, otherwise aligned\n");
  printf("    0: A normal, 1: A trans\n");
  printf("    0: B normal, 1: B trans\n");
  printf("    PRECISION: SP, DP, I16I32, USI8I32, SUI8I32, SUI8UI8, BF16F32, BF16\n");
  printf("    BRGEMM: nobr, addrbr, offsbr, strdbr\n");
  printf("    brsize: 1 - N\n");
  printf("    #repetitions\n");
  printf("    0: no check, otherwise: run check\n");
  printf("\n\n");
}


LIBXSMM_INLINE
double run_jit_double( const gemm_def*     i_gemm_def,
                       const double*       i_a,
                       const double*       i_b,
                       double*             o_c,
                       const unsigned int  i_print_jit_info) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit;
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  double l_jittime, l_runtime;
  size_t l_t, l_r;
  const double** l_a_addr = (const double**)malloc(i_gemm_def->br_count*sizeof(double*));
  const double** l_b_addr = (const double**)malloc(i_gemm_def->br_count*sizeof(double*));
  unsigned long long* l_a_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  unsigned long long* l_b_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  double l_alpha = i_gemm_def->alpha;
  double l_beta = i_gemm_def->beta;
  unsigned long long l_br = (unsigned long long)i_gemm_def->br_count;

  if (0 == i_gemm_def) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  /* setup brgemm offsets */
  if ( i_gemm_def->br_type == 2 ) {
    for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
      l_a_offs[l_r] = l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k * sizeof(double);
      l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * sizeof(double);
    }
  }

  /* set up the flags */
  if ( i_gemm_def->trans_b != 0 ) {
    l_flags |= LIBXSMM_GEMM_FLAG_TRANS_B;
  }
  if ( i_gemm_def->trans_a != 0 ) {
    fprintf(stderr, "trans_a needs to be 0\n");
    return EXIT_FAILURE;
  }
  l_flags |= (0 != i_gemm_def->aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0);
  l_flags |= (0 != i_gemm_def->aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0);

  l_start = libxsmm_timer_tick();
  if (i_gemm_def->br_type == 0) {
    l_test_jit.dmm = libxsmm_dmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                         &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                         &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
  } else if (i_gemm_def->br_type == 1) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.dmra = libxsmm_dmmdispatch_reducebatch_addr(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.dmra = libxsmm_dmmdispatch_reducebatch_addr_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 2) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.dmro = libxsmm_dmmdispatch_reducebatch_offs(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.dmro = libxsmm_dmmdispatch_reducebatch_offs_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 3) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.dmrs = libxsmm_dmmdispatch_reducebatch_strd(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(double), i_gemm_def->ldb*i_gemm_def->n*sizeof(double),
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.dmrs = libxsmm_dmmdispatch_reducebatch_strd_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(double), i_gemm_def->ldb*i_gemm_def->n*sizeof(double), i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else {
    /* nothing */
  }
  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if (l_test_jit.xmm == 0) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(EXIT_FAILURE);
  }

  /* receive kernel information */
  libxsmm_get_mmkernel_info(l_test_jit, &l_info);

  l_start = libxsmm_timer_tick();
  if ( l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.dmm(i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (double*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (double*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.dmra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.dmro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.dmrs(i_a, i_b, o_c, &l_br);
      }
    }
  } else {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.dmm(i_a, i_b, o_c, i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (double*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (double*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.dmra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.dmro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.dmrs(i_a, i_b, o_c, &l_br);
      }
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  free( l_a_addr );
  free( l_b_addr );
  free( l_a_offs );
  free( l_b_offs );

  return l_runtime;
}


LIBXSMM_INLINE
double run_jit_float( const libxsmm_gemm_descriptor* i_xgemm_desc,
                      const float*                   i_a,
                      const float*                   i_b,
                      float*                         o_c,
                      const unsigned int             i_br,
                      const unsigned int             i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit;
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  double l_jittime, l_runtime;
  int l_t;

  if (0 == i_xgemm_desc) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  l_start = libxsmm_timer_tick();
  l_test_jit = libxsmm_xmmdispatch(i_xgemm_desc);
  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if (l_test_jit.xmm == 0) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(EXIT_FAILURE);
  }

  /* receive kernel information */
  libxsmm_get_mmkernel_info(l_test_jit, &l_info);

  l_start = libxsmm_timer_tick();
  if ( l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    for (l_t = 0; l_t < g_reps; l_t++) {
      l_test_jit.smm(i_a, i_b, o_c);
    }
  } else {
    for (l_t = 0; l_t < g_reps; l_t++) {
      l_test_jit.smm(i_a, i_b, o_c, i_a, i_b, o_c);
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  return l_runtime;
}


LIBXSMM_INLINE
double run_jit_short_int( const libxsmm_gemm_descriptor* i_xgemm_desc,
                          const short*                   i_a,
                          const short*                   i_b,
                          int*                           o_c,
                          const unsigned int             i_br,
                          const unsigned int             i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit;
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  double l_jittime, l_runtime;
  int l_t;

  if (0 == i_xgemm_desc) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  l_start = libxsmm_timer_tick();
  l_test_jit = libxsmm_xmmdispatch(i_xgemm_desc);
  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if (l_test_jit.xmm == 0) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(EXIT_FAILURE);
  }

  /* receive kernel information */
  libxsmm_get_mmkernel_info(l_test_jit, &l_info);

  l_start = libxsmm_timer_tick();
  if (l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    for ( l_t = 0; l_t < g_reps; l_t++ ) {
      l_test_jit.wimm(i_a, i_b, o_c, NULL, NULL, NULL);
    }
  } else {
    for ( l_t = 0; l_t < g_reps; l_t++ ) {
      l_test_jit.wimm(i_a, i_b, o_c, i_a, i_b, o_c);
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  return l_runtime;
}


LIBXSMM_INLINE
double run_jit_uschar_int( const libxsmm_gemm_descriptor* i_xgemm_desc,
                           const unsigned char*           i_a,
                           const char*                    i_b,
                           int*                           o_c,
                           const unsigned int             i_br,
                           const unsigned int             i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit;
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  double l_jittime, l_runtime;
  int l_t;

  if (0 == i_xgemm_desc) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  l_start = libxsmm_timer_tick();
  l_test_jit = libxsmm_xmmdispatch(i_xgemm_desc);
  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if (l_test_jit.xmm == 0) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(EXIT_FAILURE);
  }

  /* receive kernel information */
  libxsmm_get_mmkernel_info(l_test_jit, &l_info);

  l_start = libxsmm_timer_tick();
  if (l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    for ( l_t = 0; l_t < g_reps; l_t++ ) {
      l_test_jit.usbimm(i_a, i_b, o_c, NULL, NULL, NULL);
    }
  } else {
    for ( l_t = 0; l_t < g_reps; l_t++ ) {
      l_test_jit.usbimm(i_a, i_b, o_c, i_a, i_b, o_c);
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  return l_runtime;
}


LIBXSMM_INLINE
double run_jit_suchar_int( const libxsmm_gemm_descriptor* i_xgemm_desc,
                           const char*                    i_a,
                           const unsigned char*           i_b,
                           int*                           o_c,
                           const unsigned int             i_br,
                           const unsigned int             i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit;
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  double l_jittime, l_runtime;
  int l_t;

  if (0 == i_xgemm_desc) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  l_start = libxsmm_timer_tick();
  l_test_jit = libxsmm_xmmdispatch(i_xgemm_desc);
  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if (l_test_jit.xmm == 0) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(EXIT_FAILURE);
  }

  /* receive kernel information */
  libxsmm_get_mmkernel_info(l_test_jit, &l_info);

  l_start = libxsmm_timer_tick();
  if (l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    for ( l_t = 0; l_t < g_reps; l_t++ ) {
      l_test_jit.subimm(i_a, i_b, o_c, NULL, NULL, NULL);
    }
  } else {
    for ( l_t = 0; l_t < g_reps; l_t++ ) {
      l_test_jit.subimm(i_a, i_b, o_c, i_a, i_b, o_c);
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  return l_runtime;
}

#if 0
LIBXSMM_INLINE
double run_jit_uschar_uchar( const libxsmm_gemm_descriptor* i_xgemm_desc,
                             const unsigned char*           i_a,
                             const char*                    i_b,
                             unsigned char*                 o_c,
                             const unsigned int             i_br,
                             const unsigned int             i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit;
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  double l_jittime, l_runtime;
  int l_t;
  float l_scf = 1.0;

  if (0 == i_xgemm_desc) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  l_start = libxsmm_timer_tick();
  l_test_jit = libxsmm_xmmdispatch(i_xgemm_desc);
  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if (l_test_jit.xmm == 0) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(EXIT_FAILURE);
  }

  /* receive kernel information */
  libxsmm_get_mmkernel_info(l_test_jit, &l_info);

  l_start = libxsmm_timer_tick();
  if (l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    for ( l_t = 0; l_t < g_reps; l_t++ ) {
      l_test_jit.usbubmm(i_a, i_b, o_c, &l_scf, NULL, NULL, NULL);
    }
  } else {
    for ( l_t = 0; l_t < g_reps; l_t++ ) {
      l_test_jit.usbubmm(i_a, i_b, o_c, &l_scf, i_a, i_b, o_c);
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  return l_runtime;
}
#endif


LIBXSMM_INLINE
double run_jit_suchar_uchar( const libxsmm_gemm_descriptor* i_xgemm_desc,
                             const char*                    i_a,
                             const unsigned char*           i_b,
                             unsigned char*                 o_c,
                             float                          i_scf,
                             const unsigned int             i_br,
                             const unsigned int             i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit;
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  double l_jittime, l_runtime;
  int l_t;

  if (0 == i_xgemm_desc) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  l_start = libxsmm_timer_tick();
  l_test_jit = libxsmm_xmmdispatch(i_xgemm_desc);
  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if (l_test_jit.xmm == 0) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(EXIT_FAILURE);
  }

  /* receive kernel information */
  libxsmm_get_mmkernel_info(l_test_jit, &l_info);

  l_start = libxsmm_timer_tick();
  if (l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    for ( l_t = 0; l_t < g_reps; l_t++ ) {
      l_test_jit.sububmm(i_a, i_b, o_c, &i_scf, NULL, NULL, NULL);
    }
  } else {
    for ( l_t = 0; l_t < g_reps; l_t++ ) {
      l_test_jit.sububmm(i_a, i_b, o_c, &i_scf, i_a, i_b, o_c);
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  return l_runtime;
}

LIBXSMM_INLINE
double run_jit_bfloat16_float( const libxsmm_gemm_descriptor* i_xgemm_desc,
                               const libxsmm_bfloat16*        i_a,
                               const libxsmm_bfloat16*        i_b,
                               float*                         o_c,
                               const unsigned int             i_br,
                               const unsigned int             i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit;
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  double l_jittime, l_runtime;
  int l_t;

  if (0 == i_xgemm_desc) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  l_start = libxsmm_timer_tick();
  l_test_jit = libxsmm_xmmdispatch(i_xgemm_desc);
  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if (l_test_jit.xmm == 0) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(EXIT_FAILURE);
  }

  /* receive kernel information */
  libxsmm_get_mmkernel_info(l_test_jit, &l_info);

  l_start = libxsmm_timer_tick();
  if (l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    for ( l_t = 0; l_t < g_reps; l_t++ ) {
      l_test_jit.bsmm(i_a, i_b, o_c, NULL, NULL, NULL);
    }
  } else {
    for ( l_t = 0; l_t < g_reps; l_t++ ) {
      l_test_jit.bsmm(i_a, i_b, o_c, i_a, i_b, o_c);
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  return l_runtime;
}


LIBXSMM_INLINE
double run_jit_bfloat16( const libxsmm_gemm_descriptor* i_xgemm_desc,
                         const libxsmm_bfloat16*        i_a,
                         const libxsmm_bfloat16*        i_b,
                               libxsmm_bfloat16*        o_c,
                         const unsigned int             i_br,
                         const unsigned int             i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit;
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  double l_jittime, l_runtime;
  int l_t;

  if (0 == i_xgemm_desc) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  l_start = libxsmm_timer_tick();
  l_test_jit = libxsmm_xmmdispatch(i_xgemm_desc);
  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if (l_test_jit.xmm == 0) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(EXIT_FAILURE);
  }

  /* receive kernel information */
  libxsmm_get_mmkernel_info(l_test_jit, &l_info);

  l_start = libxsmm_timer_tick();
  if (l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    for ( l_t = 0; l_t < g_reps; l_t++ ) {
      l_test_jit.bmm(i_a, i_b, o_c, NULL, NULL, NULL);
    }
  } else {
    for ( l_t = 0; l_t < g_reps; l_t++ ) {
      l_test_jit.bmm(i_a, i_b, o_c, i_a, i_b, o_c);
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  return l_runtime;
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

  int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  libxsmm_gemm_prefetch_type l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  const libxsmm_gemm_descriptor* l_xgemm_desc = 0;
  libxsmm_descriptor_blob l_xgemm_blob;
  libxsmm_matdiff_info l_diff;
  gemm_def l_gemm_def;
  size_t l_i = 0, l_j = 0, l_s = 0, l_t = 0, l_r = 0;
  double l_runtime_c = 0;
  double l_runtime_libxsmm = 0;
  libxsmm_timer_tickint l_start;
  int l_file_input = 0;
  char* l_file_name = NULL;
  FILE *l_file_handle = NULL;
  int l_run_check = 0;

  /* input data */
  double *l_a_d = 0, *l_b_d = 0, *l_c_d = 0;
  float *l_a_f = 0, *l_b_f = 0, *l_c_f = 0;
  short *l_a_w = 0, *l_b_w = 0;
  libxsmm_bfloat16 *l_a_bf = 0, *l_b_bf = 0, *l_c_bf = 0;
  unsigned char *l_ua_b = 0, *l_ub_b;
  char *l_sa_b = 0, *l_sb_b = 0;
  int* l_c_b_i = 0;
  int* l_c_w_i = 0;
  unsigned char* l_c_b_ub = 0;
  float* l_c_bf_f = 0;
  /* Gold data */
  double* l_c_gold_d = 0;
  float* l_c_gold_f = 0;
  libxsmm_bfloat16* l_c_gold_bf = 0;
  int* l_c_gold_w_i = 0;
  int* l_c_gold_b_i = 0;
  unsigned char* l_c_gold_b_ub = 0;
  float* l_c_gold_bf_f = 0;
  double l_total_max_error = 0.0;

  /* scaling factor */
  float l_scf = 1.0;

  libxsmm_matdiff_clear(&l_diff);

  /* check argument count for a valid range */
  if ( argc == 18 ) {
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
    g_reps = atoi(argv[17]);

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
  } else if ( argc == 13 ) {
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
    g_reps = atoi(argv[11]);
    l_run_check = atoi(argv[12]);
    l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  } else {
    print_help();
    return EXIT_FAILURE;
  }

  l_br = (l_br < 1) ? 1 : l_br;
  l_br = (l_br_type == 0) ? 1 : l_br;

  if ( l_trans_b != 0 ) {
    l_flags |= LIBXSMM_GEMM_FLAG_TRANS_B;
  }
  if ( l_trans_a != 0 ) {
    fprintf(stderr, "trans_a needs to be 0\n");
    return EXIT_FAILURE;
  }

  l_flags |= (0 != l_aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0);
  l_flags |= (0 != l_aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0);

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

 if ( l_file_input != 0 ) {
    l_file_handle = fopen( l_file_name, "r" );
  } else {
    if ( l_trans_b == 0 ) {
      printf("------------------------------------------------\n");
      printf("RUNNING (%ix%i) X (%ix%i) = (%ix%i), %s\n", l_m, l_k, l_k, l_n, l_m, l_n, l_precision);
      printf("------------------------------------------------\n");
    } else {
      printf("------------------------------------------------\n");
      printf("RUNNING (%ix%i) X (%ix%i)^T = (%ix%i), %s\n", l_m, l_k, l_k, l_n, l_m, l_n, l_precision);
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
      l_gemm_def.br_unroll = 0;

      l_a_d = (double*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * (size_t)l_br * sizeof(double), 64);
      l_b_d = (double*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * (size_t)l_br * sizeof(double), 64);
      l_c_d = (double*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(double), 64);
      l_c_gold_d = (double*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(double), 64);
      /* touch A */
      for ( l_r = 0; l_r < l_br; l_r++) {
        for (l_i = 0; l_i < l_lda; l_i++) {
          for (l_j = 0; l_j < l_k; l_j++) {
            l_a_d[(l_r * l_lda * l_k) + ((l_j * l_lda) + l_i)] = libxsmm_rng_f64();
          }
        }
      }
      /* touch B */
      for ( l_r = 0; l_r < l_br; l_r++) {
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
          l_c_gold_d[(l_j * l_ldc) + l_i] = 0.0;
        }
      }

      l_runtime_libxsmm = run_jit_double( &l_gemm_def, l_a_d, l_b_d, l_c_d, l_file_input );

      if ( l_run_check == 1 ) {
        l_start = libxsmm_timer_tick();
        for (l_t = 0; l_t < g_reps; l_t++) {
          for (l_r = 0; l_r < l_br; l_r++) {
            for (l_j = 0; l_j < l_n; l_j++) {
              for (l_s = 0; l_s < l_k; l_s++) {
                for (l_i = 0; l_i < l_m; l_i++) {
                  l_c_gold_d[(l_j * l_ldc) + l_i] += l_a_d[(l_r * l_lda * l_k) + ((l_s * l_lda) + l_i)] * l_b_d[(l_r * l_ldb * l_n) + ((l_j * l_ldb) + l_s)];
                }
              }
            }
          }
        }
        l_runtime_c = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
        libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F64, l_m, l_n, l_c_gold_d, l_c_d, &l_ldc, &l_ldc);
      }

      if ( l_file_input == 0 ) {
        printf("%fs for C\n", l_runtime_c);
        printf("%f GFLOPS for C\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_c * 1.0e9));
        printf("%fs for libxsmm\n", l_runtime_libxsmm);
        printf("%f GFLOPS for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9));
        printf("max. error: %f\n", l_diff.linf_abs);
      } else {
        if ( l_run_check == 1 ) {
          printf("%i %i %i %i %i %i %i %f %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9), l_diff.linf_abs );
        } else {
          printf("%i %i %i %i %i %i %i %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_br, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k * (double)l_br) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
        }
      }

      if ( (l_total_max_error < l_diff.linf_abs) && (l_run_check == 1) ) {
        l_total_max_error = l_diff.linf_abs;
      }

      libxsmm_free(l_a_d);
      libxsmm_free(l_b_d);
      libxsmm_free(l_c_d);
      libxsmm_free(l_c_gold_d);
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
      l_gemm_def.br_type = 0;
      l_gemm_def.br_count = 1;
      l_gemm_def.br_unroll = 0;

      l_a_d = (double*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * sizeof(double), 64);
      l_b_d = (double*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_k * sizeof(double), 64);
      l_c_d = (double*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(double), 64);
      l_c_gold_d = (double*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(double), 64);
      /* touch A */
      for (l_i = 0; l_i < l_lda; l_i++) {
        for (l_j = 0; l_j < l_k; l_j++) {
          l_a_d[(l_j * l_lda) + l_i] = libxsmm_rng_f64();
        }
      }
      /* touch B */
      for (l_i = 0; l_i < l_ldb; l_i++) {
        for (l_j = 0; l_j < l_k; l_j++) {
          l_b_d[(l_j * l_ldb) + l_i] = libxsmm_rng_f64();
        }
      }
      /* touch C */
      for (l_i = 0; l_i < l_ldc; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          l_c_d[(l_j * l_ldc) + l_i] = 0.0;
          l_c_gold_d[(l_j * l_ldc) + l_i] = 0.0;
        }
      }

      l_runtime_libxsmm = run_jit_double( &l_gemm_def, l_a_d, l_b_d, l_c_d, l_file_input );

      if ( l_run_check == 1 ) {
        l_start = libxsmm_timer_tick();
        for (l_t = 0; l_t < g_reps; l_t++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            for (l_s = 0; l_s < l_k; l_s++) {
              for (l_i = 0; l_i < l_m; l_i++) {
                l_c_gold_d[(l_j * l_ldc) + l_i] += l_a_d[(l_s * l_lda) + l_i] * l_b_d[(l_s * l_ldb) + l_j];
              }
            }
          }
        }
        l_runtime_c = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
        libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F64, l_m, l_n, l_c_gold_d, l_c_d, &l_ldc, &l_ldc);
      }

      if ( l_file_input == 0 ) {
        printf("%fs for C\n", l_runtime_c);
        printf("%f GFLOPS for C\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_c * 1.0e9));
        printf("%fs for libxsmm\n", l_runtime_libxsmm);
        printf("%f GFLOPS for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9));
        printf("max. error: %f\n", l_diff.linf_abs);
      } else {
        if ( l_run_check == 1 ) {
          printf("%i %i %i %i %i %i %f %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9), l_diff.linf_abs );
        } else {
          printf("%i %i %i %i %i %i %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
        }
      }

      if ( (l_total_max_error < l_diff.linf_abs) && (l_run_check == 1) ) {
        l_total_max_error = l_diff.linf_abs;
      }

      libxsmm_free(l_a_d);
      libxsmm_free(l_b_d);
      libxsmm_free(l_c_d);
      libxsmm_free(l_c_gold_d);
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
      l_xgemm_desc = libxsmm_gemm_descriptor_dinit(&l_xgemm_blob, LIBXSMM_GEMM_PRECISION_F32,
        l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_alpha, l_beta, l_flags, l_prefetch);
      l_a_f = (float*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * sizeof(float), 64);
      l_b_f = (float*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * sizeof(float), 64);
      l_c_f = (float*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(float), 64);
      l_c_gold_f = (float*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(float), 64);
      /* touch A */
      for (l_i = 0; l_i < l_lda; l_i++) {
        for (l_j = 0; l_j < l_k; l_j++) {
          l_a_f[(l_j * l_lda) + l_i] = (float)libxsmm_rng_f64();
        }
      }
      /* touch B */
      for (l_i = 0; l_i < l_ldb; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          l_b_f[(l_j * l_ldb) + l_i] = (float)libxsmm_rng_f64();
        }
      }
      /* touch C */
      for (l_i = 0; l_i < l_ldc; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          l_c_f[(l_j * l_ldc) + l_i] = 0.f;
          l_c_gold_f[(l_j * l_ldc) + l_i] = 0.f;
        }
      }

      l_runtime_libxsmm = run_jit_float( l_xgemm_desc, l_a_f, l_b_f, l_c_f, l_br, l_file_input );

      if ( l_run_check == 1 ) {
        l_start = libxsmm_timer_tick();
        for (l_t = 0; l_t < g_reps; l_t++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            for (l_s = 0; l_s < l_k; l_s++) {
              for (l_i = 0; l_i < l_m; l_i++) {
                l_c_gold_f[(l_j * l_ldc) + l_i] += l_a_f[(l_s * l_lda) + l_i] * l_b_f[(l_j * l_ldb) + l_s];
              }
            }
          }
        }
        l_runtime_c = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
        libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, l_m, l_n, l_c_gold_f, l_c_f, &l_ldc, &l_ldc);
      }

      if ( l_file_input == 0 ) {
        printf("%fs for C\n", l_runtime_c);
        printf("%f GFLOPS for C\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_c * 1.0e9));
        printf("%fs for libxsmm\n", l_runtime_libxsmm);
        printf("%f GFLOPS for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9));
        printf("max. error: %f\n", l_diff.linf_abs);
      } else {
        if ( l_run_check == 1 ) {
          printf("%i %i %i %i %i %i %f %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9), l_diff.linf_abs );
        } else {
          printf("%i %i %i %i %i %i %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
        }
      }

      if ( (l_total_max_error < l_diff.linf_abs) && (l_run_check == 1) ) {
        l_total_max_error = l_diff.linf_abs;
      }

      libxsmm_free(l_a_f);
      libxsmm_free(l_b_f);
      libxsmm_free(l_c_f);
      libxsmm_free(l_c_gold_f);
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
      l_xgemm_desc = libxsmm_gemm_descriptor_dinit(&l_xgemm_blob, LIBXSMM_GEMM_PRECISION_F32,
        l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_alpha, l_beta, l_flags, l_prefetch);
      l_a_f = (float*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * sizeof(float), 64);
      l_b_f = (float*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_k * sizeof(float), 64);
      l_c_f = (float*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(float), 64);
      l_c_gold_f = (float*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(float), 64);
      /* touch A */
      for (l_i = 0; l_i < l_lda; l_i++) {
        for (l_j = 0; l_j < l_k; l_j++) {
          l_a_f[(l_j * l_lda) + l_i] = (float)libxsmm_rng_f64();
        }
      }
      /* touch B */
      for (l_i = 0; l_i < l_ldb; l_i++) {
        for (l_j = 0; l_j < l_k; l_j++) {
          l_b_f[(l_j * l_ldb) + l_i] = (float)libxsmm_rng_f64();
        }
      }
      /* touch C */
      for (l_i = 0; l_i < l_ldc; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          l_c_f[(l_j * l_ldc) + l_i] = 0.f;
          l_c_gold_f[(l_j * l_ldc) + l_i] = 0.f;
        }
      }

      l_runtime_libxsmm = run_jit_float( l_xgemm_desc, l_a_f, l_b_f, l_c_f, l_br, l_file_input );

      if ( l_run_check == 1 ) {
        l_start = libxsmm_timer_tick();
        for (l_t = 0; l_t < g_reps; l_t++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            for (l_s = 0; l_s < l_k; l_s++) {
              for (l_i = 0; l_i < l_m; l_i++) {
                l_c_gold_f[(l_j * l_ldc) + l_i] += l_a_f[(l_s * l_lda) + l_i] * l_b_f[(l_s * l_ldb) + l_j];
              }
            }
          }
        }
        l_runtime_c = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
        libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, l_m, l_n, l_c_gold_f, l_c_f, &l_ldc, &l_ldc);
      }

      if ( l_file_input == 0 ) {
        printf("%fs for C\n", l_runtime_c);
        printf("%f GFLOPS for C\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_c * 1.0e9));
        printf("%fs for libxsmm\n", l_runtime_libxsmm);
        printf("%f GFLOPS for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9));
        printf("max. error: %f\n", l_diff.linf_abs);
      } else {
        if ( l_run_check == 1 ) {
          printf("%i %i %i %i %i %i %f %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9), l_diff.linf_abs );
        } else {
          printf("%i %i %i %i %i %i %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
        }
      }

      if ( (l_total_max_error < l_diff.linf_abs) && (l_run_check == 1) ) {
        l_total_max_error = l_diff.linf_abs;
      }

      libxsmm_free(l_a_f);
      libxsmm_free(l_b_f);
      libxsmm_free(l_c_f);
      libxsmm_free(l_c_gold_f);
    } while ( l_keep_going );
  } else if (strcmp(l_precision, "I16I32") == 0) {
    const int l_k_block = 2;
    double l_max_error = 0;
    int l_k2;
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
      l_flags |= LIBXSMM_GEMM_FLAG_VNNI_A;
      l_xgemm_desc = libxsmm_gemm_descriptor_dinit2(&l_xgemm_blob,
        LIBXSMM_GEMM_PRECISION_I16, LIBXSMM_GEMM_PRECISION_I32,
        l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_alpha, l_beta, l_flags, l_prefetch);
      l_a_w = (short*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * sizeof(short), 64);
      l_b_w = (short*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * sizeof(short), 64);
      l_c_w_i = (int*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(int), 64);
      l_c_gold_w_i = (int*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(int), 64);

      /* touch A */
      for (l_i = 0; l_i < l_lda; l_i++) {
        for (l_j = 0; l_j < l_k; l_j++) {
          l_a_w[(l_j * l_lda) + l_i] = (short)(libxsmm_rng_f64() * 10.0);
        }
      }
      /* touch B */
      for (l_i = 0; l_i < l_ldb; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          l_b_w[(l_j * l_ldb) + l_i] = (short)(libxsmm_rng_f64() * 10.0);
        }
      }
      /* touch C */
      for (l_i = 0; l_i < l_ldc; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          l_c_w_i[(l_j * l_ldc) + l_i] = 0;
          l_c_gold_w_i[(l_j * l_ldc) + l_i] = 0;
        }
      }

      l_runtime_libxsmm = run_jit_short_int(l_xgemm_desc, l_a_w, l_b_w, l_c_w_i, l_br, l_file_input );

      if ( l_run_check == 1 ) {
        l_start = libxsmm_timer_tick();
        for (l_t = 0; l_t < g_reps; l_t++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            for (l_s = 0; l_s < (l_k / l_k_block); l_s++) {
              for (l_i = 0; l_i < l_m; l_i++) {
                for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
                  l_c_gold_w_i[(l_j * l_ldc) + l_i] += l_a_w[(l_s * (l_lda*l_k_block)) + (l_i*l_k_block) + l_k2] * l_b_w[(l_j * l_ldb) + (l_s*l_k_block) + l_k2];
                }
              }
            }
          }
        }
        l_runtime_c = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
        l_max_error = 0;
        for (l_i = 0; l_i < l_m; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            const double l_fabs = fabs((double)l_c_gold_w_i[(l_j * l_ldc) + l_i] - (double)l_c_w_i[(l_j * l_ldc) + l_i]);
            if (l_max_error < l_fabs) l_max_error = l_fabs;
          }
        }
      }

      if ( l_file_input == 0 ) {
        printf("%fs for C\n", l_runtime_c);
        printf("%f GFLOPS for C\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_c * 1.0e9));
        printf("%fs for libxsmm\n", l_runtime_libxsmm);
        printf("%f GFLOPS for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9));
        printf("max. error: %f\n", l_max_error);
      } else {
        if ( l_run_check == 1 ) {
          printf("%i %i %i %i %i %i %f %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9), l_max_error );
        } else {
          printf("%i %i %i %i %i %i %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
        }
      }

      if ( (l_total_max_error < l_max_error) && (l_run_check == 1) ) {
        l_total_max_error = l_max_error;
      }

      libxsmm_free(l_a_w);
      libxsmm_free(l_b_w);
      libxsmm_free(l_c_w_i);
      libxsmm_free(l_c_gold_w_i);
    } while ( l_keep_going );
  } else if (strcmp(l_precision, "USI8I32") == 0) {
    const int l_k_block = 4;
    double l_max_error = 0;
    int l_k2;
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
      l_flags |= LIBXSMM_GEMM_FLAG_A_UNSIGNED | LIBXSMM_GEMM_FLAG_VNNI_A;
      l_xgemm_desc = libxsmm_gemm_descriptor_dinit2(&l_xgemm_blob,
        LIBXSMM_GEMM_PRECISION_I8, LIBXSMM_GEMM_PRECISION_I32,
        l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_alpha, l_beta, l_flags, l_prefetch);
      l_ua_b = (unsigned char*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * sizeof(unsigned char), 64);
      l_sb_b = (char*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * sizeof(char), 64);
      l_c_b_i = (int*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(int), 64);
      l_c_gold_b_i = (int*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(int), 64);

      /* touch A */
      for (l_i = 0; l_i < l_lda; l_i++) {
        for (l_j = 0; l_j < l_k; l_j++) {
          l_ua_b[(l_j * l_lda) + l_i] = (unsigned char)(libxsmm_rng_f64() * 5.0);
        }
      }
      /* touch B */
      for (l_i = 0; l_i < l_ldb; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          l_sb_b[(l_j * l_ldb) + l_i] = (char)(libxsmm_rng_f64() * 5.0);
        }
      }
      /* touch C */
      for (l_i = 0; l_i < l_ldc; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          l_c_b_i[(l_j * l_ldc) + l_i] = 0;
          l_c_gold_b_i[(l_j * l_ldc) + l_i] = 0;
        }
      }

      l_runtime_libxsmm = run_jit_uschar_int(l_xgemm_desc, l_ua_b, l_sb_b, l_c_b_i, l_br, l_file_input );

      if ( l_run_check == 1 ) {
        l_start = libxsmm_timer_tick();
        for (l_t = 0; l_t < g_reps; l_t++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            for (l_s = 0; l_s < (l_k / l_k_block); l_s++) {
              for (l_i = 0; l_i < l_m; l_i++) {
                for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
                  l_c_gold_b_i[(l_j * l_ldc) + l_i] += l_ua_b[(l_s * (l_lda*l_k_block)) + (l_i*l_k_block) + l_k2] * l_sb_b[(l_j * l_ldb) + (l_s*l_k_block) + l_k2];
                }
              }
            }
          }
        }
        l_runtime_c = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
        l_max_error = 0;
        for (l_i = 0; l_i < l_m; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            const double l_fabs = fabs((double)l_c_gold_b_i[(l_j * l_ldc) + l_i] - (double)l_c_b_i[(l_j * l_ldc) + l_i]);
            if (l_max_error < l_fabs) l_max_error = l_fabs;
          }
        }
      }

      if ( l_file_input == 0 ) {
        printf("%fs for C\n", l_runtime_c);
        printf("%f GFLOPS for C\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_c * 1.0e9));
        printf("%fs for libxsmm\n", l_runtime_libxsmm);
        printf("%f GFLOPS for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9));
        printf("max. error: %f\n", l_max_error);
      } else {
        if ( l_run_check == 1 ) {
          printf("%i %i %i %i %i %i %f %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9), l_max_error );
        } else {
          printf("%i %i %i %i %i %i %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
        }
      }

      if ( (l_total_max_error < l_max_error) && (l_run_check == 1) ) {
        l_total_max_error = l_max_error;
      }

      libxsmm_free(l_ua_b);
      libxsmm_free(l_sb_b);
      libxsmm_free(l_c_b_i);
      libxsmm_free(l_c_gold_b_i);
    } while ( l_keep_going );
  } else if (strcmp(l_precision, "SUI8I32") == 0) {
    const int l_k_block = 4;
    double l_max_error = 0;
    int l_k2;
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
      l_flags |= LIBXSMM_GEMM_FLAG_B_UNSIGNED | LIBXSMM_GEMM_FLAG_VNNI_A;
      l_xgemm_desc = libxsmm_gemm_descriptor_dinit2(&l_xgemm_blob,
        LIBXSMM_GEMM_PRECISION_I8, LIBXSMM_GEMM_PRECISION_I32,
        l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_alpha, l_beta, l_flags, l_prefetch);
      l_sa_b = (char*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * sizeof(char), 64);
      l_ub_b = (unsigned char*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * sizeof(unsigned char), 64);
      l_c_b_i = (int*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(int), 64);
      l_c_gold_b_i = (int*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(int), 64);

      /* touch A */
      for (l_i = 0; l_i < l_lda; l_i++) {
        for (l_j = 0; l_j < l_k; l_j++) {
          l_sa_b[(l_j * l_lda) + l_i] = (char)(libxsmm_rng_f64() * 5.0);
        }
      }
      /* touch B */
      for (l_i = 0; l_i < l_ldb; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          l_ub_b[(l_j * l_ldb) + l_i] = (unsigned char)(libxsmm_rng_f64() * 5.0);
        }
      }
      /* touch C */
      for (l_i = 0; l_i < l_ldc; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          l_c_b_i[(l_j * l_ldc) + l_i] = 0;
          l_c_gold_b_i[(l_j * l_ldc) + l_i] = 0;
        }
      }

      l_runtime_libxsmm = run_jit_suchar_int(l_xgemm_desc, l_sa_b, l_ub_b, l_c_b_i, l_br, l_file_input );

      if ( l_run_check == 1 ) {
        l_start = libxsmm_timer_tick();
        for (l_t = 0; l_t < g_reps; l_t++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            for (l_s = 0; l_s < (l_k / l_k_block); l_s++) {
              for (l_i = 0; l_i < l_m; l_i++) {
                for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
                  l_c_gold_b_i[(l_j * l_ldc) + l_i] += l_sa_b[(l_s * (l_lda*l_k_block)) + (l_i*l_k_block) + l_k2] * l_ub_b[(l_j * l_ldb) + (l_s*l_k_block) + l_k2];
                }
              }
            }
          }
        }
        l_runtime_c = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
        l_max_error = 0;
        for (l_i = 0; l_i < l_m; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            const double l_fabs = fabs((double)l_c_gold_b_i[(l_j * l_ldc) + l_i] - (double)l_c_b_i[(l_j * l_ldc) + l_i]);
            if (l_max_error < l_fabs) l_max_error = l_fabs;
          }
        }
      }

      if ( l_file_input == 0 ) {
        printf("%fs for C\n", l_runtime_c);
        printf("%f GFLOPS for C\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_c * 1.0e9));
        printf("%fs for libxsmm\n", l_runtime_libxsmm);
        printf("%f GFLOPS for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9));
        printf("max. error: %f\n", l_max_error);
      } else {
        if ( l_run_check == 1 ) {
          printf("%i %i %i %i %i %i %f %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9), l_max_error );
        } else {
          printf("%i %i %i %i %i %i %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
        }
      }

      if ( (l_total_max_error < l_max_error) && (l_run_check == 1) ) {
        l_total_max_error = l_max_error;
      }

      libxsmm_free(l_sa_b);
      libxsmm_free(l_ub_b);
      libxsmm_free(l_c_b_i);
      libxsmm_free(l_c_gold_b_i);
    } while ( l_keep_going );
  } else if (strcmp(l_precision, "SUI8UI8") == 0) {
    const int l_k_block = 4;
    double l_max_error = 0;
    int l_k2;
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
      l_flags |= LIBXSMM_GEMM_FLAG_B_UNSIGNED | LIBXSMM_GEMM_FLAG_C_UNSIGNED | LIBXSMM_GEMM_FLAG_VNNI_A;
      l_xgemm_desc = libxsmm_gemm_descriptor_dinit2(&l_xgemm_blob,
        LIBXSMM_GEMM_PRECISION_I8, LIBXSMM_GEMM_PRECISION_I8,
        l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_alpha, l_beta, l_flags, l_prefetch);
      l_sa_b = (char*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * sizeof(char), 64);
      l_ub_b = (unsigned char*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * sizeof(unsigned char), 64);
      l_c_b_ub = (unsigned char*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(unsigned char), 64);
      l_c_gold_b_ub = (unsigned char*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(unsigned char), 64);

      /* touch A */
      for (l_i = 0; l_i < l_lda; l_i++) {
        for (l_j = 0; l_j < l_k; l_j++) {
          l_sa_b[(l_j * l_lda) + l_i] = (char)(libxsmm_rng_f64() * 2.0);
        }
      }
      /* touch B */
      for (l_i = 0; l_i < l_ldb; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          l_ub_b[(l_j * l_ldb) + l_i] = (unsigned char)(libxsmm_rng_f64() * 2.0);
        }
      }
      /* touch C */
      for (l_i = 0; l_i < l_ldc; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          l_c_b_ub[(l_j * l_ldc) + l_i] = 0;
          l_c_gold_b_ub[(l_j * l_ldc) + l_i] = 0;
        }
      }

      l_runtime_libxsmm = run_jit_suchar_uchar(l_xgemm_desc, l_sa_b, l_ub_b, l_c_b_ub, l_scf, l_br, l_file_input );

      if ( l_run_check == 1 ) {
        l_start = libxsmm_timer_tick();
        for (l_t = 0; l_t < g_reps; l_t++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            for (l_i = 0; l_i < l_m; l_i++) {
              int tmp = (int)l_c_gold_b_ub[(l_j * l_ldc) + l_i];
              float ftmp;
              for (l_s = 0; l_s < (l_k / l_k_block); l_s++) {
                for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
                  tmp += l_sa_b[(l_s * (l_lda*l_k_block)) + (l_i*l_k_block) + l_k2] * l_ub_b[(l_j * l_ldb) + (l_s*l_k_block) + l_k2];
                }
              }
              ftmp = (float)tmp;
              ftmp *= l_scf;
              l_c_gold_b_ub[(l_j * l_ldc) + l_i] = (unsigned char)ftmp;
            }
          }
        }
        l_runtime_c = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
        l_max_error = 0;
        for (l_i = 0; l_i < l_m; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            const double l_fabs = fabs((double)l_c_gold_b_ub[(l_j * l_ldc) + l_i] - (double)l_c_b_ub[(l_j * l_ldc) + l_i]);
            if (l_max_error < l_fabs) l_max_error = l_fabs;
          }
        }
      }

      if ( l_file_input == 0 ) {
        printf("%fs for C\n", l_runtime_c);
        printf("%f GFLOPS for C\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_c * 1.0e9));
        printf("%fs for libxsmm\n", l_runtime_libxsmm);
        printf("%f GFLOPS for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9));
        printf("max. error: %f\n", l_max_error);
      } else {
        if ( l_run_check == 1 ) {
          printf("%i %i %i %i %i %i %f %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9), l_max_error );
        } else {
          printf("%i %i %i %i %i %i %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
        }
      }

      if ( (l_total_max_error < l_max_error) && (l_run_check == 1) ) {
        l_total_max_error = l_max_error;
      }

      libxsmm_free(l_sa_b);
      libxsmm_free(l_ub_b);
      libxsmm_free(l_c_b_ub);
      libxsmm_free(l_c_gold_b_ub);
    } while ( l_keep_going );
   } else if (strcmp(l_precision, "BF16F32") == 0) {
    const int l_k_block = 2;
    double l_max_error = 0;
    int l_k2;
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
      l_flags |= LIBXSMM_GEMM_FLAG_VNNI_A;
      l_xgemm_desc = libxsmm_gemm_descriptor_dinit2(&l_xgemm_blob,
        LIBXSMM_GEMM_PRECISION_BF16, LIBXSMM_GEMM_PRECISION_F32,
        l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_alpha, l_beta, l_flags, l_prefetch);
      l_a_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * sizeof(libxsmm_bfloat16), 64);
      l_b_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * sizeof(libxsmm_bfloat16), 64);
      l_c_bf_f = (float*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(float), 64);
      l_c_gold_bf_f = (float*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(float), 64);
      /* touch A */
      for (l_i = 0; l_i < l_lda; l_i++) {
        for (l_j = 0; l_j < l_k; l_j++) {
          union libxsmm_bfloat16_hp tmp;
          tmp.f = (float)libxsmm_rng_f64();
          l_a_bf[(l_j * l_lda) + l_i] = tmp.i[1];
        }
      }
      /* touch B */
      for (l_i = 0; l_i < l_ldb; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          union libxsmm_bfloat16_hp tmp;
          tmp.f = (float)libxsmm_rng_f64();
          l_b_bf[(l_j * l_ldb) + l_i] = tmp.i[1];
        }
      }
      /* touch C */
      for (l_i = 0; l_i < l_ldc; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          l_c_bf_f[(l_j * l_ldc) + l_i] = 0.0f;
          l_c_gold_bf_f[(l_j * l_ldc) + l_i] = 0.0f;
        }
      }

      l_runtime_libxsmm = run_jit_bfloat16_float(l_xgemm_desc, l_a_bf, l_b_bf, l_c_bf_f, l_br, l_file_input );

      if ( l_run_check == 1 ) {
        l_start = libxsmm_timer_tick();
        for (l_t = 0; l_t < g_reps; l_t++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            for (l_s = 0; l_s < (l_k / l_k_block); l_s++) {
              for (l_i = 0; l_i < l_m; l_i++) {
                for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
                  union libxsmm_bfloat16_hp tmp_a_f;
                  union libxsmm_bfloat16_hp tmp_b_f;
                  tmp_a_f.i[1] = l_a_bf[(l_s * (l_lda*l_k_block)) + (l_i*l_k_block) + l_k2];
                  tmp_a_f.i[0] = 0;
                  tmp_b_f.i[1] = l_b_bf[(l_j * l_ldb) + (l_s*l_k_block) + l_k2];
                  tmp_b_f.i[0] = 0;
                  l_c_gold_bf_f[(l_j * l_ldc) + l_i] += (float)(tmp_a_f.f * tmp_b_f.f);
                }
              }
            }
          }
        }
        l_runtime_c = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
        l_max_error = 0;
        for (l_i = 0; l_i < l_m; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            const double l_fabs = fabs((double)l_c_gold_bf_f[(l_j * l_ldc) + l_i] - (double)l_c_bf_f[(l_j * l_ldc) + l_i]);
            if (l_max_error < l_fabs) l_max_error = l_fabs;
          }
        }
      }

      if ( l_file_input == 0 ) {
        printf("%fs for C\n", l_runtime_c);
        printf("%f GFLOPS for C\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_c * 1.0e9));
        printf("%fs for libxsmm\n", l_runtime_libxsmm);
        printf("%f GFLOPS for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9));
        printf("max. error: %f\n", l_max_error);
      } else {
        if ( l_run_check == 1 ) {
          printf("%i %i %i %i %i %i %f %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9), l_max_error );
        } else {
          printf("%i %i %i %i %i %i %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
        }
      }

      if ( (l_total_max_error < l_max_error) && (l_run_check == 1) ) {
        l_total_max_error = l_max_error;
      }

      libxsmm_free(l_a_bf);
      libxsmm_free(l_b_bf);
      libxsmm_free(l_c_bf_f);
      libxsmm_free(l_c_gold_bf_f);
    } while ( l_keep_going );
  } else if (strcmp(l_precision, "BF16") == 0) {
    const int l_k_block = 2;
    double l_max_error = 0;
    int l_k2;
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
      l_flags |= LIBXSMM_GEMM_FLAG_VNNI_A;
      l_xgemm_desc = libxsmm_gemm_descriptor_dinit2(&l_xgemm_blob,
        LIBXSMM_GEMM_PRECISION_BF16, LIBXSMM_GEMM_PRECISION_BF16,
        l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_alpha, l_beta, l_flags, l_prefetch);
      l_a_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_lda * (size_t)l_k * sizeof(libxsmm_bfloat16), 64);
      l_b_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_ldb * (size_t)l_n * sizeof(libxsmm_bfloat16), 64);
      l_c_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(libxsmm_bfloat16), 64);
      l_c_gold_bf = (libxsmm_bfloat16*)libxsmm_aligned_malloc((size_t)l_ldc * (size_t)l_n * sizeof(libxsmm_bfloat16), 64);
      /* touch A */
      for (l_i = 0; l_i < l_lda; l_i++) {
        for (l_j = 0; l_j < l_k; l_j++) {
          union libxsmm_bfloat16_hp tmp;
          tmp.f = (float)libxsmm_rng_f64();
          l_a_bf[(l_j * l_lda) + l_i] = tmp.i[1];
        }
      }
      /* touch B */
      for (l_i = 0; l_i < l_ldb; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          union libxsmm_bfloat16_hp tmp;
          tmp.f = (float)libxsmm_rng_f64();
          l_b_bf[(l_j * l_ldb) + l_i] = tmp.i[1];
        }
      }
      /* touch C */
      for (l_i = 0; l_i < l_ldc; l_i++) {
        for (l_j = 0; l_j < l_n; l_j++) {
          union libxsmm_bfloat16_hp tmp;
          tmp.f = 0.0f;
          l_c_bf[(l_j * l_ldc) + l_i] = tmp.i[1];
          l_c_gold_bf[(l_j * l_ldc) + l_i] = tmp.i[1];
        }
      }

      l_runtime_libxsmm = run_jit_bfloat16(l_xgemm_desc, l_a_bf, l_b_bf, l_c_bf, l_br, l_file_input );

      if ( l_run_check == 1 ) {
        l_start = libxsmm_timer_tick();
        for (l_t = 0; l_t < g_reps; l_t++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            for (l_i = 0; l_i < l_m; l_i++) {
              union libxsmm_bfloat16_hp fprod;
              fprod.i[1] = l_c_gold_bf[(l_j * l_ldc) + l_i];
              fprod.i[0] = 0;
              for (l_s = 0; l_s < (l_k / l_k_block); l_s++) {
                for (l_k2 = 0; l_k2 < l_k_block; l_k2++) {
                  union libxsmm_bfloat16_hp tmp_a_f;
                  union libxsmm_bfloat16_hp tmp_b_f;
                  tmp_a_f.i[1] = l_a_bf[(l_s * (l_lda*l_k_block)) + (l_i*l_k_block) + l_k2];
                  tmp_a_f.i[0] = 0;
                  tmp_b_f.i[1] = l_b_bf[(l_j * l_ldb) + (l_s*l_k_block) + l_k2];
                  tmp_b_f.i[0] = 0;
                  fprod.f += (float)(tmp_a_f.f * tmp_b_f.f);
                }
              }
              l_c_gold_bf[(l_j * l_ldc) + l_i] = fprod.i[1];
            }
          }
        }
        l_runtime_c = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
        l_max_error = 0;
        for (l_i = 0; l_i < l_m; l_i++) {
          for (l_j = 0; l_j < l_n; l_j++) {
            union libxsmm_bfloat16_hp tmp_c;
            union libxsmm_bfloat16_hp tmp_gold;
            double l_fabs;

            tmp_c.i[1] = l_c_bf[(l_j * l_ldc) + l_i];
            tmp_c.i[0] = 0;
            tmp_gold.i[1] = l_c_gold_bf[(l_j * l_ldc) + l_i];
            tmp_gold.i[0] = 0;
            l_fabs = fabs((double)tmp_gold.f - (double)tmp_c.f);
            if (l_max_error < l_fabs) l_max_error = l_fabs;
          }
        }
      }

      if ( l_file_input == 0 ) {
        printf("%fs for C\n", l_runtime_c);
        printf("%f GFLOPS for C\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_c * 1.0e9));
        printf("%fs for libxsmm\n", l_runtime_libxsmm);
        printf("%f GFLOPS for libxsmm\n", ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9));
        printf("max. error: %f\n", l_max_error);
      } else {
        if ( l_run_check == 1 ) {
          printf("%i %i %i %i %i %i %f %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9), l_max_error );
        } else {
          printf("%i %i %i %i %i %i %f\n", l_m, l_n, l_k, l_lda, l_ldb, l_ldc, ((double)((double)g_reps * (double)l_m * (double)l_n * (double)l_k) * 2.0) / (l_runtime_libxsmm * 1.0e9) );
        }
      }

      if ( (l_total_max_error < l_max_error) && (l_run_check == 1) ) {
        l_total_max_error = l_max_error;
      }

      libxsmm_free(l_a_bf);
      libxsmm_free(l_b_bf);
      libxsmm_free(l_c_bf);
      libxsmm_free(l_c_gold_bf);
    } while ( l_keep_going );
  }

  if ( l_file_input != 0 ) {
    fclose( l_file_handle );
  } else {
    printf("------------------------------------------------\n");
  }

  /* Print total max error */
  printf("\n\n Total Max Error %f\n\n", l_total_max_error );

  if ( l_total_max_error >= 0.00005 ) {
    return EXIT_FAILURE;
  } else {
    return EXIT_SUCCESS;
  }
}

