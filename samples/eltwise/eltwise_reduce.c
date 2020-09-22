/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>


LIBXSMM_INLINE
void sfill_matrix ( float *matrix, unsigned int ld, unsigned int m, unsigned int n )
{
  unsigned int i, j;
  double dtmp;

  if ( ld < m )
  {
     fprintf(stderr,"Error is sfill_matrix: ld=%u m=%u mismatched!\n",ld,m);
     exit(EXIT_FAILURE);
  }
  for ( j = 1; j <= n; j++ )
  {
     /* Fill through the leading dimension */
     for ( i = 1; i <= ld; i++ )
     {
        dtmp = 1.0 - 2.0*libxsmm_rng_f64();
        matrix [ (j-1)*ld + (i-1) ] = (float) dtmp;
     }
  }
}

int main(int argc, char* argv[])
{
  unsigned int m = 64, n = 64, reduce_elts = 1, reduce_elts_squared = 1, reduce_rows = 1, result_size, i, j, jj, k, iters = 10000, n_cols_idx = 0;
  libxsmm_blasint ld_in = 64/*, ld_out = 64*/;
  float  *sinp, *result_reduce_elts, *result_reduce_elts_squared, *ref_result_reduce_elts, *ref_result_reduce_elts_squared;
  unsigned long long *cols_ind_array;
  libxsmm_meltw_redu_flags jit_flags = LIBXSMM_MELTW_FLAG_REDUCE_NONE;
  libxsmm_meltwfunction_reduce kernel;
  libxsmm_meltwfunction_reduce_cols_idx kernel2;
  libxsmm_meltw_reduce_param params;
  libxsmm_meltw_reduce_cols_idx_param params2;
  libxsmm_matdiff_info norms_elts, norms_elts_squared;
  unsigned long long l_start, l_end;
  double l_total = 0.0, l_total2 = 0.0;

  libxsmm_init();

  libxsmm_matdiff_clear(&norms_elts);
  libxsmm_matdiff_clear(&norms_elts_squared);

  if ( argc > 1 ) m = atoi(argv[1]);
  if ( argc > 2 ) n = atoi(argv[2]);
  if ( argc > 3 ) ld_in = atoi(argv[3]);
  if ( argc > 4 ) reduce_elts = atoi(argv[4]);
  if ( argc > 5 ) reduce_elts_squared = atoi(argv[5]);
  if ( argc > 6 ) reduce_rows = atoi(argv[6]);
  if ( argc > 7 ) iters = atoi(argv[7]);
  if ( argc > 8 ) n_cols_idx = atoi(argv[8]);

  m = LIBXSMM_MAX(m,1);
  n = LIBXSMM_MAX(n,1);
  ld_in = LIBXSMM_MAX(ld_in,(libxsmm_blasint)m);
  result_size = (reduce_rows == 1) ? n : m;

  /* Allocate arrays  */
  sinp  = (float*) malloc( ld_in*n*sizeof(float) );
  result_reduce_elts = (float*) malloc(result_size*sizeof(float) );
  result_reduce_elts_squared = (float*) malloc(result_size*sizeof(float) );
  ref_result_reduce_elts = (float*) malloc(result_size*sizeof(float) );
  ref_result_reduce_elts_squared = (float*) malloc(result_size*sizeof(float) );
  cols_ind_array = (unsigned long long*) malloc(n_cols_idx*sizeof(unsigned long long));

  /* Fill matrices with random data */
  sfill_matrix ( sinp, ld_in, m, n );

  /* Calculate reference results...  */
  if (reduce_rows == 1) {
    for (j = 0; j < n; j++) {
      ref_result_reduce_elts[j] = 0;
      ref_result_reduce_elts_squared[j] = 0;
      for (i = 0; i < m; i++) {
        ref_result_reduce_elts[j] += sinp[j*ld_in + i];
        ref_result_reduce_elts_squared[j] += sinp[j*ld_in + i] * sinp[j*ld_in + i];
      }
    }
  } else {
    if (n_cols_idx == 0) {
      /* In this case we reduce columns */
      for (i = 0; i < m; i++) {
        ref_result_reduce_elts[i] = 0;
        ref_result_reduce_elts_squared[i] = 0;
        for (j = 0; j < n; j++) {
          ref_result_reduce_elts[i] += sinp[j*ld_in + i];
          ref_result_reduce_elts_squared[i] += sinp[j*ld_in + i] * sinp[j*ld_in + i];
        }
      }
    } else {
      /* In this case we reduce columns */
      for (j = 0; j < n_cols_idx; j++) {
        cols_ind_array[j] = rand() % n_cols_idx;
      }
      for (i = 0; i < m; i++) {
        ref_result_reduce_elts[i] = 0;
        ref_result_reduce_elts_squared[i] = 0;
        for (jj = 0; jj < n_cols_idx; jj++) {
          j = cols_ind_array[jj];
          ref_result_reduce_elts[i] += sinp[j*ld_in + i];
        }
      }
    }
  }

  /* Generate JITED kernel */
  if (reduce_rows == 1) {
    jit_flags = LIBXSMM_MELTW_FLAG_REDUCE_OP_ADD | LIBXSMM_MELTW_FLAG_REDUCE_ROWS;
  } else {
    jit_flags = LIBXSMM_MELTW_FLAG_REDUCE_OP_ADD | LIBXSMM_MELTW_FLAG_REDUCE_COLS;
  }
  if (reduce_elts == 1) {
    jit_flags |=  LIBXSMM_MELTW_FLAG_REDUCE_ELTS;
  }
  if (reduce_elts_squared == 1) {
    jit_flags |=  LIBXSMM_MELTW_FLAG_REDUCE_ELTS_SQUARED;
  }


  printf("JITing reduce kernel... \n");
  if (n_cols_idx == 0) {
    kernel = libxsmm_dispatch_meltw_reduce(m, n, &ld_in, &ld_in, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_flags);
    /* Call JITed kernel and compare result  */
    printf("Calling JITed reduce kernel... \n");
    params.in_ptr = sinp;
    params.out_ptr_0 = result_reduce_elts;
    params.out_ptr_1 = result_reduce_elts_squared;
    kernel( &params );
  } else {
    kernel2 = libxsmm_dispatch_meltw_reduce_cols_idx(m, &ld_in, &ld_in, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_I64);
    /* Call JITed kernel and compare result  */
    printf("Calling JITed reduce cols idx kernel... \n");
    params2.n = n_cols_idx;
    params2.ind_ptr = cols_ind_array;
    params2.inp_ptr = sinp;
    params2.out_ptr = result_reduce_elts;
    kernel2( &params2 );
  }

  /* compare */
  printf("##########################################\n");
  if (n_cols_idx == 0) {
    printf("#   Correctness - Eltwise reduce         #\n");
  } else {
    printf("#   Correctness - Eltwise reduce colsidx #\n");
  }
  printf("##########################################\n");
  libxsmm_matdiff(&norms_elts, LIBXSMM_DATATYPE_F32, result_size, 1, ref_result_reduce_elts, result_reduce_elts, 0, 0);
  printf("L1 reference  : %.25g\n", norms_elts.l1_ref);
  printf("L1 test       : %.25g\n", norms_elts.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_elts.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_elts.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_elts.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_elts.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_elts.normf_rel);

  /* compare */
  if (n_cols_idx == 0) {
    printf("##########################################\n");
    printf("#   Correctness - Eltwise-square reduce  #\n");
    printf("##########################################\n");
    libxsmm_matdiff(&norms_elts_squared, LIBXSMM_DATATYPE_F32, result_size, 1, ref_result_reduce_elts_squared, result_reduce_elts_squared, 0, 0);
    printf("L1 reference  : %.25g\n", norms_elts_squared.l1_ref);
    printf("L1 test       : %.25g\n", norms_elts_squared.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_elts_squared.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_elts_squared.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_elts_squared.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_elts_squared.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_elts_squared.normf_rel);
  }


  l_start = libxsmm_timer_tick();
  /* Calculate reference results...  */
  for (k = 0; k < iters; k++) {
    if (reduce_rows == 1) {
      for (j = 0; j < n; j++) {
        ref_result_reduce_elts[j] = 0;
        ref_result_reduce_elts_squared[j] = 0;
        for (i = 0; i < m; i++) {
          ref_result_reduce_elts[j] += sinp[j*ld_in + i];
          ref_result_reduce_elts_squared[j] += sinp[j*ld_in + i] * sinp[j*ld_in + i];
        }
      }
    } else {
      if (n_cols_idx == 0) {
        /* In this case we reduce columns */
        for (i = 0; i < m; i++) {
          ref_result_reduce_elts[i] = 0;
          ref_result_reduce_elts_squared[i] = 0;
          for (j = 0; j < n; j++) {
            ref_result_reduce_elts[i] += sinp[j*ld_in + i];
            ref_result_reduce_elts_squared[i] += sinp[j*ld_in + i] * sinp[j*ld_in + i];
          }
        }
      } else {
        /* In this case we reduce columns */
        for (j = 0; j < n_cols_idx; j++) {
          cols_ind_array[j] = rand() % n_cols_idx;
        }
        for (i = 0; i < m; i++) {
          ref_result_reduce_elts[i] = 0;
          ref_result_reduce_elts_squared[i] = 0;
          for (jj = 0; jj < n_cols_idx; jj++) {
            j = cols_ind_array[jj];
            ref_result_reduce_elts[i] += sinp[j*ld_in + i];
          }
        }
      }
    }
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("Reference time = %.5g\n", ((double)(l_total)));

  l_start = libxsmm_timer_tick();
  if (n_cols_idx == 0) {
    for (k = 0; k < iters; k++) {
      kernel( &params );
    }
  } else {
    for (k = 0; k < iters; k++) {
      kernel2( &params2 );
    }
  }
  l_end = libxsmm_timer_tick();
  l_total2 = libxsmm_timer_duration(l_start, l_end);
  printf("Optimized time = %.5g\n", ((double)(l_total2)));
  printf("Speedup is = %.5g\n", ((double)(l_total/l_total2)));

  free(sinp);
  free(result_reduce_elts);
  free(result_reduce_elts_squared);
  free(ref_result_reduce_elts);
  free(ref_result_reduce_elts_squared);

  return EXIT_SUCCESS;
}

