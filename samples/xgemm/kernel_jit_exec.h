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
  int tc_config;
} gemm_def;


LIBXSMM_INLINE
double run_jit_double( const gemm_def*     i_gemm_def,
                       const double*       i_a,
                       const double*       i_b,
                       double*             o_c,
                       const unsigned int  i_print_jit_info) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
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
      if (i_gemm_def->trans_b == 0) {
        l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * sizeof(double);
      } else {
        l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->k * sizeof(double);
      }
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
      if (i_gemm_def->trans_b == 0) {
        l_test_jit.dmrs = libxsmm_dmmdispatch_reducebatch_strd(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(double), i_gemm_def->ldb*i_gemm_def->n*sizeof(double),
                                                               &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                               &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
      } else {
        l_test_jit.dmrs = libxsmm_dmmdispatch_reducebatch_strd(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(double), i_gemm_def->ldb*i_gemm_def->k*sizeof(double),
                                                               &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                               &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
      }
    } else {
      if (i_gemm_def->trans_b == 0) {
        l_test_jit.dmrs = libxsmm_dmmdispatch_reducebatch_strd_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(double), i_gemm_def->ldb*i_gemm_def->n*sizeof(double), i_gemm_def->br_count,
                                                                      &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                      &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
      } else {
        l_test_jit.dmrs = libxsmm_dmmdispatch_reducebatch_strd_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(double), i_gemm_def->ldb*i_gemm_def->k*sizeof(double), i_gemm_def->br_count,
                                                                      &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                      &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
      }
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
          l_a_addr[l_r] = (const double*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          if (i_gemm_def->trans_b == 0) {
            l_b_addr[l_r] = (const double*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
          } else {
            l_b_addr[l_r] = (const double*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->k);
          }
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
        l_test_jit.dmm(i_a, i_b, o_c/*, i_a, i_b, o_c*/); /* @TODO fix prefetch */
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (const double*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          if (i_gemm_def->trans_b == 0) {
            l_b_addr[l_r] = (const double*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
          } else {
            l_b_addr[l_r] = (const double*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->k);
          }
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

  free( (void*)l_a_addr );
  free( (void*)l_b_addr );
  free( (void*)l_a_offs );
  free( (void*)l_b_offs );

  return l_runtime;
}


LIBXSMM_INLINE
double run_jit_float( const gemm_def*     i_gemm_def,
                      const float*        i_a,
                      const float*        i_b,
                      float*              o_c,
                      const unsigned int  i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  double l_jittime, l_runtime;
  size_t l_t, l_r;
  const float** l_a_addr = (const float**)malloc(i_gemm_def->br_count*sizeof(float*));
  const float** l_b_addr = (const float**)malloc(i_gemm_def->br_count*sizeof(float*));
  unsigned long long* l_a_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  unsigned long long* l_b_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  float l_alpha = (float)i_gemm_def->alpha;
  float l_beta = (float)i_gemm_def->beta;
  unsigned long long l_br = (unsigned long long)i_gemm_def->br_count;

  if (0 == i_gemm_def) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  /* setup brgemm offsets */
  if ( i_gemm_def->br_type == 2 ) {
    for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
      l_a_offs[l_r] = l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k * sizeof(float);
      if (i_gemm_def->trans_b == 0) {
        l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * sizeof(float);
      } else {
        l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->k * sizeof(float);
      }
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
    l_test_jit.smm = libxsmm_smmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                         &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                         &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
  } else if (i_gemm_def->br_type == 1) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.smra = libxsmm_smmdispatch_reducebatch_addr(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.smra = libxsmm_smmdispatch_reducebatch_addr_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 2) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.smro = libxsmm_smmdispatch_reducebatch_offs(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.smro = libxsmm_smmdispatch_reducebatch_offs_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 3) {
    if (i_gemm_def->br_unroll == 0) {
      if (i_gemm_def->trans_b == 0) {
        l_test_jit.smrs = libxsmm_smmdispatch_reducebatch_strd(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(float), i_gemm_def->ldb*i_gemm_def->n*sizeof(float),
                                                               &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                               &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
      } else {
        l_test_jit.smrs = libxsmm_smmdispatch_reducebatch_strd(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(float), i_gemm_def->ldb*i_gemm_def->k*sizeof(float),
                                                               &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                               &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
      }
    } else {
      if (i_gemm_def->trans_b == 0) {
        l_test_jit.smrs = libxsmm_smmdispatch_reducebatch_strd_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(float), i_gemm_def->ldb*i_gemm_def->n*sizeof(float), i_gemm_def->br_count,
                                                                      &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                      &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
      } else {
        l_test_jit.smrs = libxsmm_smmdispatch_reducebatch_strd_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(float), i_gemm_def->ldb*i_gemm_def->k*sizeof(float), i_gemm_def->br_count,
                                                                      &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                      &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
      }
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
        l_test_jit.smm(i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (float*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          if (i_gemm_def->trans_b == 0) {
            l_b_addr[l_r] = (float*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
          } else {
            l_b_addr[l_r] = (float*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->k);
          }
        }
        l_test_jit.smra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.smro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.smrs(i_a, i_b, o_c, &l_br);
      }
    }
  } else {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.smm(i_a, i_b, o_c/*, i_a, i_b, o_c*/); /* @TODO fix prefetch */
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (float*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          if (i_gemm_def->trans_b == 0) {
            l_b_addr[l_r] = (float*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
          } else {
            l_b_addr[l_r] = (float*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->k);
          }
        }
        l_test_jit.smra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.smro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.smrs(i_a, i_b, o_c, &l_br);
      }
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  free( (void*)l_a_addr );
  free( (void*)l_b_addr );
  free( (void*)l_a_offs );
  free( (void*)l_b_offs );

  return l_runtime;
}


LIBXSMM_INLINE
double run_jit_short_int( const gemm_def*     i_gemm_def,
                          const short*        i_a,
                          const short*        i_b,
                          int*                o_c,
                          const unsigned int  i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  double l_jittime, l_runtime;
  size_t l_t, l_r;
  const short** l_a_addr = (const short**)malloc(i_gemm_def->br_count*sizeof(short*));
  const short** l_b_addr = (const short**)malloc(i_gemm_def->br_count*sizeof(short*));
  unsigned long long* l_a_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  unsigned long long* l_b_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  int l_alpha = (int)i_gemm_def->alpha;
  int l_beta = (int)i_gemm_def->beta;
  unsigned long long l_br = (unsigned long long)i_gemm_def->br_count;
  l_flags |= LIBXSMM_GEMM_FLAG_A_UNSIGNED | LIBXSMM_GEMM_FLAG_VNNI_A;

  if (0 == i_gemm_def) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  /* setup brgemm offsets */
  if ( i_gemm_def->br_type == 2 ) {
    for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
      l_a_offs[l_r] = l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k * sizeof(short);
      l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * sizeof(short);
    }
  }

  /* set up the flags */
  if ( i_gemm_def->trans_b != 0 ) {
    fprintf(stderr, "trans_b needs to be 0\n");
    return EXIT_FAILURE;
  }
  if ( i_gemm_def->trans_a != 0 ) {
    fprintf(stderr, "trans_a needs to be 0\n");
    return EXIT_FAILURE;
  }
  l_flags |= (0 != i_gemm_def->aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0);
  l_flags |= (0 != i_gemm_def->aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0);

  libxsmm_xmmfunction cfg_tr = { NULL };
  libxsmm_xmmfunction rls_tr = { NULL };

  int l_cfg_flags = 0;
  int l_rls_flags = 0;
  if (i_gemm_def->tc_config) {
      l_cfg_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | l_flags;
      l_rls_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | l_flags;
  }

  l_start = libxsmm_timer_tick();
  if (i_gemm_def->tc_config) {
      cfg_tr.wimm  = libxsmm_wimmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                        &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                        NULL, &l_beta, &l_cfg_flags, NULL);
      rls_tr.wimm  = libxsmm_wimmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                             NULL, NULL, NULL, NULL, NULL, &l_rls_flags, NULL);
      l_flags |= (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
  }

  if (i_gemm_def->br_type == 0) {
    l_test_jit.wimm = libxsmm_wimmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                         &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                         &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
  } else if (i_gemm_def->br_type == 1) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.wimra = libxsmm_wimmdispatch_reducebatch_addr(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.wimra = libxsmm_wimmdispatch_reducebatch_addr_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 2) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.wimro = libxsmm_wimmdispatch_reducebatch_offs(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.wimro = libxsmm_wimmdispatch_reducebatch_offs_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 3) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.wimrs = libxsmm_wimmdispatch_reducebatch_strd(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(short), i_gemm_def->ldb*i_gemm_def->n*sizeof(short),
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.wimrs = libxsmm_wimmdispatch_reducebatch_strd_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(short), i_gemm_def->ldb*i_gemm_def->n*sizeof(short), i_gemm_def->br_count,
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
  if (i_gemm_def->tc_config) {
    cfg_tr.wimm(NULL, NULL, NULL);
  }
  l_start = libxsmm_timer_tick();
  if ( l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.wimm(i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (short*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (short*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.wimra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.wimro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.wimrs(i_a, i_b, o_c, &l_br);
      }
    }
  } else {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.wimm(i_a, i_b, o_c/*, i_a, i_b, o_c*/); /* @TODO fix prefetch */
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (short*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (short*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.wimra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.wimro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.wimrs(i_a, i_b, o_c, &l_br);
      }
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  if (i_gemm_def->tc_config) {
    rls_tr.wimm(NULL, NULL, NULL);
  }
  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  free( (void*)l_a_addr );
  free( (void*)l_b_addr );
  free( (void*)l_a_offs );
  free( (void*)l_b_offs );

  return l_runtime;
}


LIBXSMM_INLINE
double run_jit_uschar_int( const gemm_def*      i_gemm_def,
                           const unsigned char* i_a,
                           const char*          i_b,
                           int*                 o_c,
                           const unsigned int   i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  double l_jittime, l_runtime;
  size_t l_t, l_r;
  const unsigned char** l_a_addr = (const unsigned char**)malloc(i_gemm_def->br_count*sizeof(unsigned char*));
  const char** l_b_addr = (const char**)malloc(i_gemm_def->br_count*sizeof(char*));
  unsigned long long* l_a_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  unsigned long long* l_b_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  int l_alpha = (int)i_gemm_def->alpha;
  int l_beta = (int)i_gemm_def->beta;
  unsigned long long l_br = (unsigned long long)i_gemm_def->br_count;
  l_flags |= LIBXSMM_GEMM_FLAG_A_UNSIGNED | LIBXSMM_GEMM_FLAG_VNNI_A;

  if (0 == i_gemm_def) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  /* setup brgemm offsets */
  if ( i_gemm_def->br_type == 2 ) {
    for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
      l_a_offs[l_r] = l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k * sizeof(unsigned char);
      l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * sizeof(char);
    }
  }

  /* set up the flags */
  if ( i_gemm_def->trans_b != 0 ) {
    fprintf(stderr, "trans_b needs to be 0\n");
    return EXIT_FAILURE;
  }
  if ( i_gemm_def->trans_a != 0 ) {
    fprintf(stderr, "trans_a needs to be 0\n");
    return EXIT_FAILURE;
  }
  l_flags |= (0 != i_gemm_def->aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0);
  l_flags |= (0 != i_gemm_def->aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0);

  libxsmm_xmmfunction cfg_tr = { NULL };
  libxsmm_xmmfunction rls_tr = { NULL };

  int l_cfg_flags = 0;
  int l_rls_flags = 0;
  if (i_gemm_def->tc_config) {
      l_cfg_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | l_flags;
      l_rls_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | l_flags;
  }

  l_start = libxsmm_timer_tick();
  if (i_gemm_def->tc_config) {
      cfg_tr.usbimm  = libxsmm_usbimmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                        &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                        NULL, &l_beta, &l_cfg_flags, NULL);
      rls_tr.usbimm  = libxsmm_usbimmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                             NULL, NULL, NULL, NULL, NULL, &l_rls_flags, NULL);
      l_flags |= (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
  }

  if (i_gemm_def->br_type == 0) {
    l_test_jit.usbimm = libxsmm_usbimmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                         &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                         &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
  } else if (i_gemm_def->br_type == 1) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.usbimra = libxsmm_usbimmdispatch_reducebatch_addr(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.usbimra = libxsmm_usbimmdispatch_reducebatch_addr_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 2) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.usbimro = libxsmm_usbimmdispatch_reducebatch_offs(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.usbimro = libxsmm_usbimmdispatch_reducebatch_offs_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 3) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.usbimrs = libxsmm_usbimmdispatch_reducebatch_strd(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(unsigned char), i_gemm_def->ldb*i_gemm_def->n*sizeof(char),
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.usbimrs = libxsmm_usbimmdispatch_reducebatch_strd_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(unsigned char), i_gemm_def->ldb*i_gemm_def->n*sizeof(char), i_gemm_def->br_count,
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
  if (i_gemm_def->tc_config) {
    cfg_tr.usbimm(NULL, NULL, NULL);
  }
  l_start = libxsmm_timer_tick();
  if ( l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.usbimm(i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (unsigned char*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (char*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.usbimra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.usbimro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.usbimrs(i_a, i_b, o_c, &l_br);
      }
    }
  } else {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.usbimm(i_a, i_b, o_c/*, i_a, i_b, o_c*/); /* @TODO fix prefetch */
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (unsigned char*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (char*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.usbimra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.usbimro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.usbimrs(i_a, i_b, o_c, &l_br);
      }
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  if (i_gemm_def->tc_config) {
    rls_tr.usbimm(NULL, NULL, NULL);
  }
  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  free( (void*)l_a_addr );
  free( (void*)l_b_addr );
  free( (void*)l_a_offs );
  free( (void*)l_b_offs );

  return l_runtime;
}


LIBXSMM_INLINE
double run_jit_suchar_int( const gemm_def*      i_gemm_def,
                           const char*          i_a,
                           const unsigned char* i_b,
                           int*                 o_c,
                           const unsigned int   i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  double l_jittime, l_runtime;
  size_t l_t, l_r;
  const char** l_a_addr = (const char**)malloc(i_gemm_def->br_count*sizeof(char*));
  const unsigned char** l_b_addr = (const unsigned char**)malloc(i_gemm_def->br_count*sizeof(unsigned char*));
  unsigned long long* l_a_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  unsigned long long* l_b_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  int l_alpha = (int)i_gemm_def->alpha;
  int l_beta = (int)i_gemm_def->beta;
  unsigned long long l_br = (unsigned long long)i_gemm_def->br_count;
  l_flags |= LIBXSMM_GEMM_FLAG_B_UNSIGNED | LIBXSMM_GEMM_FLAG_VNNI_A;

  if (0 == i_gemm_def) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  /* setup brgemm offsets */
  if ( i_gemm_def->br_type == 2 ) {
    for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
      l_a_offs[l_r] = l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k * sizeof(char);
      l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * sizeof(unsigned char);
    }
  }

  /* set up the flags */
  if ( i_gemm_def->trans_b != 0 ) {
    fprintf(stderr, "trans_b needs to be 0\n");
    return EXIT_FAILURE;
  }
  if ( i_gemm_def->trans_a != 0 ) {
    fprintf(stderr, "trans_a needs to be 0\n");
    return EXIT_FAILURE;
  }
  l_flags |= (0 != i_gemm_def->aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0);
  l_flags |= (0 != i_gemm_def->aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0);

  libxsmm_xmmfunction cfg_tr = { NULL };
  libxsmm_xmmfunction rls_tr = { NULL };

  int l_cfg_flags = 0;
  int l_rls_flags = 0;
  if (i_gemm_def->tc_config) {
      l_cfg_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | l_flags;
      l_rls_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | l_flags;
  }

  l_start = libxsmm_timer_tick();
  if (i_gemm_def->tc_config) {
      cfg_tr.subimm  = libxsmm_subimmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                        &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                        NULL, &l_beta, &l_cfg_flags, NULL);
      rls_tr.subimm  = libxsmm_subimmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                             NULL, NULL, NULL, NULL, NULL, &l_rls_flags, NULL);
      l_flags |= (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
  }

  if (i_gemm_def->br_type == 0) {
    l_test_jit.subimm = libxsmm_subimmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                         &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                         &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
  } else if (i_gemm_def->br_type == 1) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.subimra = libxsmm_subimmdispatch_reducebatch_addr(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.subimra = libxsmm_subimmdispatch_reducebatch_addr_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 2) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.subimro = libxsmm_subimmdispatch_reducebatch_offs(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.subimro = libxsmm_subimmdispatch_reducebatch_offs_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 3) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.subimrs = libxsmm_subimmdispatch_reducebatch_strd(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(char), i_gemm_def->ldb*i_gemm_def->n*sizeof(unsigned char),
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.subimrs = libxsmm_subimmdispatch_reducebatch_strd_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(char), i_gemm_def->ldb*i_gemm_def->n*sizeof(unsigned char), i_gemm_def->br_count,
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
  if (i_gemm_def->tc_config) {
    cfg_tr.subimm(NULL, NULL, NULL);
  }
  l_start = libxsmm_timer_tick();
  if ( l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.subimm(i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (char*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (unsigned char*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.subimra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.subimro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.subimrs(i_a, i_b, o_c, &l_br);
      }
    }
  } else {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.subimm(i_a, i_b, o_c/*, i_a, i_b, o_c*/); /* @TODO fix prefetch */
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (char*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (unsigned char*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.subimra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.subimro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.subimrs(i_a, i_b, o_c, &l_br);
      }
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  if (i_gemm_def->tc_config) {
    rls_tr.subimm(NULL, NULL, NULL);
  }
  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  free( (void*)l_a_addr );
  free( (void*)l_b_addr );
  free( (void*)l_a_offs );
  free( (void*)l_b_offs );

  return l_runtime;
}


#if 0
LIBXSMM_INLINE
double run_jit_uschar_uchar( const gemm_def*       i_gemm_def,
                             const unsigned char*  i_a,
                             const char*           i_b,
                             unsigned char*        o_c,
                             const unsigned int    i_print_jit_info ) {
  return 0.0;
}
#endif


LIBXSMM_INLINE
double run_jit_suchar_uchar( const gemm_def*        i_gemm_def,
                             const char*            i_a,
                             const unsigned char*   i_b,
                             unsigned char*         o_c,
                             float                  i_scf,
                             const unsigned int     i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  double l_jittime, l_runtime;
  size_t l_t, l_r;
  const char** l_a_addr = (const char**)malloc(i_gemm_def->br_count*sizeof(char*));
  const unsigned char** l_b_addr = (const unsigned char**)malloc(i_gemm_def->br_count*sizeof(unsigned char*));
  unsigned long long* l_a_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  unsigned long long* l_b_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  int l_alpha = (int)i_gemm_def->alpha;
  int l_beta = (int)i_gemm_def->beta;
  unsigned long long l_br = (unsigned long long)i_gemm_def->br_count;
  l_flags |= LIBXSMM_GEMM_FLAG_B_UNSIGNED | LIBXSMM_GEMM_FLAG_C_UNSIGNED | LIBXSMM_GEMM_FLAG_VNNI_A;

  if (0 == i_gemm_def) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  /* setup brgemm offsets */
  if ( i_gemm_def->br_type == 2 ) {
    for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
      l_a_offs[l_r] = l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k * sizeof(char);
      l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * sizeof(unsigned char);
    }
  }

  /* set up the flags */
  if ( i_gemm_def->trans_b != 0 ) {
    fprintf(stderr, "trans_b needs to be 0\n");
    return EXIT_FAILURE;
  }
  if ( i_gemm_def->trans_a != 0 ) {
    fprintf(stderr, "trans_a needs to be 0\n");
    return EXIT_FAILURE;
  }
  l_flags |= (0 != i_gemm_def->aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0);
  l_flags |= (0 != i_gemm_def->aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0);

  libxsmm_xmmfunction cfg_tr = { NULL };
  libxsmm_xmmfunction rls_tr = { NULL };

  int l_cfg_flags = 0;
  int l_rls_flags = 0;
  if (i_gemm_def->tc_config) {
      l_cfg_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | l_flags;
      l_rls_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | l_flags;
  }

  l_start = libxsmm_timer_tick();
  if (i_gemm_def->tc_config) {
      cfg_tr.subimm  = libxsmm_subimmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                        &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                        NULL, &l_beta, &l_cfg_flags, NULL);
      rls_tr.subimm  = libxsmm_subimmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                             NULL, NULL, NULL, NULL, NULL, &l_rls_flags, NULL);
      l_flags |= (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
  }

  if (i_gemm_def->br_type == 0) {
    l_test_jit.sububmm = libxsmm_sububmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                         &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                         &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
  } else if (i_gemm_def->br_type == 1) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.sububmra = libxsmm_sububmmdispatch_reducebatch_addr(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.sububmra = libxsmm_sububmmdispatch_reducebatch_addr_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 2) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.sububmro = libxsmm_sububmmdispatch_reducebatch_offs(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.sububmro = libxsmm_sububmmdispatch_reducebatch_offs_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 3) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.sububmrs = libxsmm_sububmmdispatch_reducebatch_strd(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(char), i_gemm_def->ldb*i_gemm_def->n*sizeof(unsigned char),
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.sububmrs = libxsmm_sububmmdispatch_reducebatch_strd_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(char), i_gemm_def->ldb*i_gemm_def->n*sizeof(unsigned char), i_gemm_def->br_count,
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
  if (i_gemm_def->tc_config) {
    cfg_tr.subimm(NULL, NULL, NULL);
  }
  l_start = libxsmm_timer_tick();
  if ( l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.sububmm(i_a, i_b, o_c, &i_scf);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (char*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (unsigned char*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.sububmra(l_a_addr, l_b_addr, o_c, &l_br, &i_scf);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.sububmro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs, &i_scf);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.sububmrs(i_a, i_b, o_c, &l_br, &i_scf);
      }
    }
  } else {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.sububmm(i_a, i_b, o_c, &i_scf);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (char*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (unsigned char*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.sububmra(l_a_addr, l_b_addr, o_c, &l_br, &i_scf);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.sububmro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs, &i_scf);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.sububmrs(i_a, i_b, o_c, &l_br, &i_scf);
      }
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  if (i_gemm_def->tc_config) {
    rls_tr.subimm(NULL, NULL, NULL);
  }
  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  free( (void*)l_a_addr );
  free( (void*)l_b_addr );
  free( (void*)l_a_offs );
  free( (void*)l_b_offs );

  return l_runtime;
}


LIBXSMM_INLINE
double run_jit_bfloat16_float( const gemm_def*         i_gemm_def,
                               const libxsmm_bfloat16* i_a,
                               const libxsmm_bfloat16* i_b,
                               float*                  o_c,
                               const unsigned int      i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  double l_jittime, l_runtime;
  size_t l_t, l_r;
  const libxsmm_bfloat16** l_a_addr = (const libxsmm_bfloat16**)malloc(i_gemm_def->br_count*sizeof(libxsmm_bfloat16*));
  const libxsmm_bfloat16** l_b_addr = (const libxsmm_bfloat16**)malloc(i_gemm_def->br_count*sizeof(libxsmm_bfloat16*));
  unsigned long long* l_a_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  unsigned long long* l_b_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  float l_alpha = (float)i_gemm_def->alpha;
  float l_beta = (float)i_gemm_def->beta;
  unsigned long long l_br = (unsigned long long)i_gemm_def->br_count;
  l_flags |= LIBXSMM_GEMM_FLAG_VNNI_A;

  if (0 == i_gemm_def) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  /* setup brgemm offsets */
  if ( i_gemm_def->br_type == 2 ) {
    for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
      l_a_offs[l_r] = l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k * sizeof(libxsmm_bfloat16);
      l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * sizeof(libxsmm_bfloat16);
    }
  }

  /* set up the flags */
  if ( i_gemm_def->trans_b != 0 ) {
    fprintf(stderr, "trans_b needs to be 0\n");
    return EXIT_FAILURE;
  }
  if ( i_gemm_def->trans_a != 0 ) {
    fprintf(stderr, "trans_a needs to be 0\n");
    return EXIT_FAILURE;
  }
  l_flags |= (0 != i_gemm_def->aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0);
  l_flags |= (0 != i_gemm_def->aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0);

  libxsmm_xmmfunction cfg_tr = { NULL };
  libxsmm_xmmfunction rls_tr = { NULL };

  int l_cfg_flags = 0;
  int l_rls_flags = 0;
  if (i_gemm_def->tc_config) {
      l_cfg_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | l_flags;
      l_rls_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | l_flags;
  }

  l_start = libxsmm_timer_tick();
  if (i_gemm_def->tc_config) {
      cfg_tr.bsmm  = libxsmm_bsmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                        &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                        NULL, &l_beta, &l_cfg_flags, NULL);
      rls_tr.bsmm  = libxsmm_bsmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                             NULL, NULL, NULL, NULL, NULL, &l_rls_flags, NULL);
      l_flags |= (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
  }
  if (i_gemm_def->br_type == 0) {
    l_test_jit.bsmm = libxsmm_bsmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                         &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                         &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
  } else if (i_gemm_def->br_type == 1) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.bsmra = libxsmm_bsmmdispatch_reducebatch_addr(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.bsmra = libxsmm_bsmmdispatch_reducebatch_addr_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 2) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.bsmro = libxsmm_bsmmdispatch_reducebatch_offs(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.bsmro = libxsmm_bsmmdispatch_reducebatch_offs_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 3) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.bsmrs = libxsmm_bsmmdispatch_reducebatch_strd(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(libxsmm_bfloat16), i_gemm_def->ldb*i_gemm_def->n*sizeof(libxsmm_bfloat16),
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.bsmrs = libxsmm_bsmmdispatch_reducebatch_strd_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(libxsmm_bfloat16), i_gemm_def->ldb*i_gemm_def->n*sizeof(libxsmm_bfloat16), i_gemm_def->br_count,
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
  if (i_gemm_def->tc_config) {
    cfg_tr.bsmm(NULL, NULL, NULL);
  }
  l_start = libxsmm_timer_tick();
  if ( l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bsmm(i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (libxsmm_bfloat16*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (libxsmm_bfloat16*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.bsmra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bsmro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bsmrs(i_a, i_b, o_c, &l_br);
      }
    }
  } else {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bsmm(i_a, i_b, o_c/*, i_a, i_b, o_c*/); /* @TODO fix prefetch */
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (libxsmm_bfloat16*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (libxsmm_bfloat16*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.bsmra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bsmro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bsmrs(i_a, i_b, o_c, &l_br);
      }
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  if (i_gemm_def->tc_config) {
    rls_tr.bsmm(NULL, NULL, NULL);
  }
  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  free( (void*)l_a_addr );
  free( (void*)l_b_addr );
  free( (void*)l_a_offs );
  free( (void*)l_b_offs );

  return l_runtime;
}


LIBXSMM_INLINE
double run_jit_bfloat16( const gemm_def*         i_gemm_def,
                         const libxsmm_bfloat16* i_a,
                         const libxsmm_bfloat16* i_b,
                               libxsmm_bfloat16* o_c,
                         const unsigned int      i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  double l_jittime, l_runtime;
  size_t l_t, l_r;
  const libxsmm_bfloat16** l_a_addr = (const libxsmm_bfloat16**)malloc(i_gemm_def->br_count*sizeof(libxsmm_bfloat16*));
  const libxsmm_bfloat16** l_b_addr = (const libxsmm_bfloat16**)malloc(i_gemm_def->br_count*sizeof(libxsmm_bfloat16*));
  unsigned long long* l_a_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  unsigned long long* l_b_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  float l_alpha = (float)i_gemm_def->alpha;
  float l_beta = (float)i_gemm_def->beta;
  unsigned long long l_br = (unsigned long long)i_gemm_def->br_count;
  l_flags |= LIBXSMM_GEMM_FLAG_VNNI_A;

  if (0 == i_gemm_def) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  /* setup brgemm offsets */
  if ( i_gemm_def->br_type == 2 ) {
    for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
      l_a_offs[l_r] = l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k * sizeof(libxsmm_bfloat16);
      l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * sizeof(libxsmm_bfloat16);
    }
  }

  /* set up the flags */
  if ( i_gemm_def->trans_b != 0 ) {
    fprintf(stderr, "trans_b needs to be 0\n");
    return EXIT_FAILURE;
  }
  if ( i_gemm_def->trans_a != 0 ) {
    fprintf(stderr, "trans_a needs to be 0\n");
    return EXIT_FAILURE;
  }
  l_flags |= (0 != i_gemm_def->aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0);
  l_flags |= (0 != i_gemm_def->aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0);

  libxsmm_xmmfunction cfg_tr = { NULL };
  libxsmm_xmmfunction rls_tr = { NULL };

  int l_cfg_flags = 0;
  int l_rls_flags = 0;
  if (i_gemm_def->tc_config) {
      l_cfg_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | l_flags;
      l_rls_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | l_flags;
  }

  l_start = libxsmm_timer_tick();
  if (i_gemm_def->tc_config) {
      cfg_tr.bsmm  = libxsmm_bsmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                        &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                        NULL, &l_beta, &l_cfg_flags, NULL);
      rls_tr.bsmm  = libxsmm_bsmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                             NULL, NULL, NULL, NULL, NULL, &l_rls_flags, NULL);
      l_flags |= (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
  }

  if (i_gemm_def->br_type == 0) {
    l_test_jit.bmm = libxsmm_bmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                         &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                         &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
  } else if (i_gemm_def->br_type == 1) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.bmra = libxsmm_bmmdispatch_reducebatch_addr(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.bmra = libxsmm_bmmdispatch_reducebatch_addr_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 2) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.bmro = libxsmm_bmmdispatch_reducebatch_offs(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.bmro = libxsmm_bmmdispatch_reducebatch_offs_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 3) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.bmrs = libxsmm_bmmdispatch_reducebatch_strd(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(libxsmm_bfloat16), i_gemm_def->ldb*i_gemm_def->n*sizeof(libxsmm_bfloat16),
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.bmrs = libxsmm_bmmdispatch_reducebatch_strd_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(libxsmm_bfloat16), i_gemm_def->ldb*i_gemm_def->n*sizeof(libxsmm_bfloat16), i_gemm_def->br_count,
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
  if (i_gemm_def->tc_config) {
    cfg_tr.bsmm(NULL, NULL, NULL);
  }
  l_start = libxsmm_timer_tick();
  if ( l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bmm(i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (libxsmm_bfloat16*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (libxsmm_bfloat16*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.bmra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bmro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bmrs(i_a, i_b, o_c, &l_br);
      }
    }
  } else {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bmm(i_a, i_b, o_c/*, i_a, i_b, o_c*/); /* @TODO fix prefetch */
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (libxsmm_bfloat16*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (libxsmm_bfloat16*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.bmra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bmro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bmrs(i_a, i_b, o_c, &l_br);
      }
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  if (i_gemm_def->tc_config) {
    rls_tr.bsmm(NULL, NULL, NULL);
  }
  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  free( (void*)l_a_addr );
  free( (void*)l_b_addr );
  free( (void*)l_a_offs );
  free( (void*)l_b_offs );

  return l_runtime;
}

LIBXSMM_INLINE
double run_jit_bfloat16_float_flat( const gemm_def*         i_gemm_def,
                                    const libxsmm_bfloat16* i_a,
                                    const libxsmm_bfloat16* i_b,
                                    float*                  o_c,
                                    const unsigned int      i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  double l_jittime, l_runtime;
  size_t l_t, l_r;
  const libxsmm_bfloat16** l_a_addr = (const libxsmm_bfloat16**)malloc(i_gemm_def->br_count*sizeof(libxsmm_bfloat16*));
  const libxsmm_bfloat16** l_b_addr = (const libxsmm_bfloat16**)malloc(i_gemm_def->br_count*sizeof(libxsmm_bfloat16*));
  unsigned long long* l_a_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  unsigned long long* l_b_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  float l_alpha = (float)i_gemm_def->alpha;
  float l_beta = (float)i_gemm_def->beta;
  unsigned long long l_br = (unsigned long long)i_gemm_def->br_count;

  if (0 == i_gemm_def) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  /* setup brgemm offsets */
  if ( i_gemm_def->br_type == 2 ) {
    for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
      l_a_offs[l_r] = l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k * sizeof(libxsmm_bfloat16);
      l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * sizeof(libxsmm_bfloat16);
    }
  }

  /* set up the flags */
  if ( i_gemm_def->trans_b != 0 ) {
    fprintf(stderr, "trans_b needs to be 0\n");
    return EXIT_FAILURE;
  }
  if ( i_gemm_def->trans_a != 0 ) {
    fprintf(stderr, "trans_a needs to be 0\n");
    return EXIT_FAILURE;
  }
  l_flags |= (0 != i_gemm_def->aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0);
  l_flags |= (0 != i_gemm_def->aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0);

  libxsmm_xmmfunction cfg_tr = { NULL };
  libxsmm_xmmfunction rls_tr = { NULL };

  int l_cfg_flags = 0;
  int l_rls_flags = 0;
  if (i_gemm_def->tc_config) {
      l_cfg_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | l_flags;
      l_rls_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | l_flags;
  }

  l_start = libxsmm_timer_tick();
  if (i_gemm_def->tc_config) {
      cfg_tr.bsmm  = libxsmm_bsmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                        &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                        NULL, &l_beta, &l_cfg_flags, NULL);
      rls_tr.bsmm  = libxsmm_bsmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                             NULL, NULL, NULL, NULL, NULL, &l_rls_flags, NULL);
      l_flags |= (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
  }
  if (i_gemm_def->br_type == 0) {
    l_test_jit.bsmm = libxsmm_bsmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                         &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                         &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
  } else if (i_gemm_def->br_type == 1) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.bsmra = libxsmm_bsmmdispatch_reducebatch_addr(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.bsmra = libxsmm_bsmmdispatch_reducebatch_addr_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 2) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.bsmro = libxsmm_bsmmdispatch_reducebatch_offs(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.bsmro = libxsmm_bsmmdispatch_reducebatch_offs_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 3) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.bsmrs = libxsmm_bsmmdispatch_reducebatch_strd(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(libxsmm_bfloat16), i_gemm_def->ldb*i_gemm_def->n*sizeof(libxsmm_bfloat16),
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.bsmrs = libxsmm_bsmmdispatch_reducebatch_strd_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(libxsmm_bfloat16), i_gemm_def->ldb*i_gemm_def->n*sizeof(libxsmm_bfloat16), i_gemm_def->br_count,
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
  if (i_gemm_def->tc_config) {
    cfg_tr.bsmm(NULL, NULL, NULL);
  }
  l_start = libxsmm_timer_tick();
  if ( l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bsmm(i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (libxsmm_bfloat16*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (libxsmm_bfloat16*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.bsmra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bsmro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bsmrs(i_a, i_b, o_c, &l_br);
      }
    }
  } else {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bsmm(i_a, i_b, o_c/*, i_a, i_b, o_c*/); /* @TODO fix prefetch */
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (libxsmm_bfloat16*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (libxsmm_bfloat16*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.bsmra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bsmro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bsmrs(i_a, i_b, o_c, &l_br);
      }
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  if (i_gemm_def->tc_config) {
    rls_tr.bsmm(NULL, NULL, NULL);
  }
  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  free( (void*)l_a_addr );
  free( (void*)l_b_addr );
  free( (void*)l_a_offs );
  free( (void*)l_b_offs );

  return l_runtime;
}


LIBXSMM_INLINE
double run_jit_bfloat16_flat( const gemm_def*         i_gemm_def,
                              const libxsmm_bfloat16* i_a,
                              const libxsmm_bfloat16* i_b,
                                    libxsmm_bfloat16* o_c,
                              const unsigned int      i_print_jit_info ) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
  libxsmm_timer_tickint l_start;
  libxsmm_mmkernel_info l_info;
  int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  double l_jittime, l_runtime;
  size_t l_t, l_r;
  const libxsmm_bfloat16** l_a_addr = (const libxsmm_bfloat16**)malloc(i_gemm_def->br_count*sizeof(libxsmm_bfloat16*));
  const libxsmm_bfloat16** l_b_addr = (const libxsmm_bfloat16**)malloc(i_gemm_def->br_count*sizeof(libxsmm_bfloat16*));
  unsigned long long* l_a_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  unsigned long long* l_b_offs = (unsigned long long*)malloc(i_gemm_def->br_count*sizeof(unsigned long long));
  float l_alpha = (float)i_gemm_def->alpha;
  float l_beta = (float)i_gemm_def->beta;
  unsigned long long l_br = (unsigned long long)i_gemm_def->br_count;

  if (0 == i_gemm_def) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  /* setup brgemm offsets */
  if ( i_gemm_def->br_type == 2 ) {
    for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
      l_a_offs[l_r] = l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k * sizeof(libxsmm_bfloat16);
      l_b_offs[l_r] = l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n * sizeof(libxsmm_bfloat16);
    }
  }

  /* set up the flags */
  if ( i_gemm_def->trans_b != 0 ) {
    fprintf(stderr, "trans_b needs to be 0\n");
    return EXIT_FAILURE;
  }
  if ( i_gemm_def->trans_a != 0 ) {
    fprintf(stderr, "trans_a needs to be 0\n");
    return EXIT_FAILURE;
  }
  l_flags |= (0 != i_gemm_def->aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0);
  l_flags |= (0 != i_gemm_def->aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0);

  libxsmm_xmmfunction cfg_tr = { NULL };
  libxsmm_xmmfunction rls_tr = { NULL };

  int l_cfg_flags = 0;
  int l_rls_flags = 0;
  if (i_gemm_def->tc_config) {
      l_cfg_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | l_flags;
      l_rls_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | l_flags;
  }

  l_start = libxsmm_timer_tick();
  if (i_gemm_def->tc_config) {
      cfg_tr.bsmm  = libxsmm_bsmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                        &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                        NULL, &l_beta, &l_cfg_flags, NULL);
      rls_tr.bsmm  = libxsmm_bsmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                             NULL, NULL, NULL, NULL, NULL, &l_rls_flags, NULL);
      l_flags |= (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
  }

  if (i_gemm_def->br_type == 0) {
    l_test_jit.bmm = libxsmm_bmmdispatch(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                         &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                         &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
  } else if (i_gemm_def->br_type == 1) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.bmra = libxsmm_bmmdispatch_reducebatch_addr(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.bmra = libxsmm_bmmdispatch_reducebatch_addr_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 2) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.bmro = libxsmm_bmmdispatch_reducebatch_offs(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k,
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.bmro = libxsmm_bmmdispatch_reducebatch_offs_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->br_count,
                                                                    &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                                    &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    }
  } else if (i_gemm_def->br_type == 3) {
    if (i_gemm_def->br_unroll == 0) {
      l_test_jit.bmrs = libxsmm_bmmdispatch_reducebatch_strd(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(libxsmm_bfloat16), i_gemm_def->ldb*i_gemm_def->n*sizeof(libxsmm_bfloat16),
                                                             &(i_gemm_def->lda), &(i_gemm_def->ldb), &(i_gemm_def->ldc),
                                                             &l_alpha, &l_beta, &l_flags, &(i_gemm_def->prefetch));
    } else {
      l_test_jit.bmrs = libxsmm_bmmdispatch_reducebatch_strd_unroll(i_gemm_def->m, i_gemm_def->n, i_gemm_def->k, i_gemm_def->lda*i_gemm_def->k*sizeof(libxsmm_bfloat16), i_gemm_def->ldb*i_gemm_def->n*sizeof(libxsmm_bfloat16), i_gemm_def->br_count,
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
  if (i_gemm_def->tc_config) {
    cfg_tr.bsmm(NULL, NULL, NULL);
  }
  l_start = libxsmm_timer_tick();
  if ( l_info.prefetch == LIBXSMM_GEMM_PREFETCH_NONE ) {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bmm(i_a, i_b, o_c);
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (libxsmm_bfloat16*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (libxsmm_bfloat16*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.bmra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bmro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bmrs(i_a, i_b, o_c, &l_br);
      }
    }
  } else {
    if (i_gemm_def->br_type == 0) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bmm(i_a, i_b, o_c/*, i_a, i_b, o_c*/); /* @TODO fix prefetch */
      }
    } else if (i_gemm_def->br_type == 1) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        for ( l_r = 0 ; l_r < i_gemm_def->br_count; l_r++ ) {
          l_a_addr[l_r] = (libxsmm_bfloat16*)i_a + (l_r * (size_t)i_gemm_def->lda * (size_t)i_gemm_def->k);
          l_b_addr[l_r] = (libxsmm_bfloat16*)i_b + (l_r * (size_t)i_gemm_def->ldb * (size_t)i_gemm_def->n);
        }
        l_test_jit.bmra(l_a_addr, l_b_addr, o_c, &l_br);
      }
    } else if (i_gemm_def->br_type == 2) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bmro(i_a, i_b, o_c, &l_br, l_a_offs, l_b_offs);
      }
    } else if (i_gemm_def->br_type == 3) {
      for (l_t = 0; l_t < g_reps; l_t++) {
        l_test_jit.bmrs(i_a, i_b, o_c, &l_br);
      }
    }
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  if (i_gemm_def->tc_config) {
    rls_tr.bsmm(NULL, NULL, NULL);
  }
  if ( i_print_jit_info == 0 ) {
    printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
    printf("%fs for creating jit\n", l_jittime);
  }

  free( (void*)l_a_addr );
  free( (void*)l_b_addr );
  free( (void*)l_a_offs );
  free( (void*)l_b_offs );

  return l_runtime;
}

