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

#include "common_edge_proxy.h"


int main(int argc, char* argv[]) {
  int M = ( argc == 7 ) ? atoi(argv[1]) : 9;
  int N = ( argc == 7 ) ? atoi(argv[2]) : 10;
  int K = ( argc == 7 ) ? atoi(argv[3]) : 9;
  unsigned int N_CRUNS = ( argc == 7 ) ? atoi(argv[4]) : 8;
  unsigned int REPS =    ( argc == 7 ) ? atoi(argv[5]) : 1;
  char* l_csr_file =     ( argc == 7 ) ?      argv[6]  : "file.csr";

  const libxsmm_gemm_prefetch_type prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  const int flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  const REALTYPE alpha = 1, beta = 1;

  REALTYPE* l_a_de = (REALTYPE*)libxsmm_aligned_malloc(sizeof(REALTYPE) * K * K, 64);
  REALTYPE* l_a_sp = NULL;
  REALTYPE* l_b = (REALTYPE*)libxsmm_aligned_malloc(sizeof(REALTYPE) * K * N * N_CRUNS, 64);
  unsigned int* l_rowptr = NULL;
  unsigned int* l_colidx = NULL;
  unsigned int l_rowcount, l_colcount, l_elements;
  REALTYPE* l_c = (REALTYPE*)libxsmm_aligned_malloc(sizeof(REALTYPE) * K * N * N_CRUNS, 64);
  REALTYPE* l_c_gold = (REALTYPE*)libxsmm_aligned_malloc(sizeof(REALTYPE) * K * N * N_CRUNS, 64);
  REALTYPE* l_c_asm = (REALTYPE*)libxsmm_aligned_malloc(sizeof(REALTYPE) * K * N * N_CRUNS, 64);
  REALTYPE l_max_error = 0.0;
  unsigned int l_k, l_n;
  int l_i, l_j, l_jj;

  LIBXSMM_VLA_DECL(3, REALTYPE, l_p_b, l_b, N, N_CRUNS);
  LIBXSMM_VLA_DECL(3, REALTYPE, l_p_c_asm, l_c_asm, N, N_CRUNS);
  LIBXSMM_VLA_DECL(3, REALTYPE, l_p_c_gold, l_c_gold, N, N_CRUNS);

  libxsmm_descriptor_blob l_xgemm_blob;
  const libxsmm_gemm_descriptor* l_xgemm_desc = 0;
  LIBXSMM_MMFUNCTION_TYPE(REALTYPE) mykernel = NULL;

  unsigned long long l_start, l_end;
  double l_total;
  unsigned long long l_libxsmmflops;
  libxsmm_kernel_info l_kinfo;

  if (argc != 7) {
    fprintf( stderr, "arguments: M #iters CSR-file!\n" );
    return -1;
  }

  /* touch B */
  for ( l_i = 0; l_i < K; l_i++) {
    for ( l_j = 0; l_j < N; l_j++) {
      for ( l_k = 0; l_k < N_CRUNS; l_k++ ) {
        LIBXSMM_VLA_ACCESS(3, l_p_b, l_i, l_j, l_k, N, N_CRUNS) = (REALTYPE)libxsmm_rng_f64();
      }
    }
  }

  /* touch C */
  for ( l_i = 0; l_i < K; l_i++) {
    for ( l_j = 0; l_j < N; l_j++) {
      for ( l_k = 0; l_k < N_CRUNS; l_k++ ) {
        LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_i, l_j, l_k, N, N_CRUNS) = (REALTYPE)0.0;
        LIBXSMM_VLA_ACCESS(3, l_p_c_asm,  l_i, l_j, l_k, N, N_CRUNS) = (REALTYPE)0.0;
      }
    }
  }

  /* read A, CSR */
  libxsmm_sparse_csr_reader(  l_csr_file,
                             &l_rowptr,
                             &l_colidx,
                             &l_a_sp,
                             &l_rowcount, &l_colcount, &l_elements );

  /* copy b to dense */
  printf("CSR matrix data structure we just read:\n");
  printf("rows: %u, columns: %u, elements: %u\n", l_rowcount, l_colcount, l_elements);

  for ( l_n = 0; l_n < (((unsigned int)K) * K); l_n++) {
    l_a_de[l_n] = 0.0;
  }

  for ( l_n = 0; l_n < (unsigned int)K; l_n++) {
    const unsigned int l_rowelems = l_rowptr[l_n+1] - l_rowptr[l_n];
    assert(l_rowptr[l_n+1] >= l_rowptr[l_n]);

    for ( l_k = 0; l_k < l_rowelems; l_k++) {
      l_a_de[(l_n * K) + l_colidx[l_rowptr[l_n] + l_k]] = l_a_sp[l_rowptr[l_n] + l_k];
    }
  }

  /* dense routine */
  l_start = libxsmm_timer_tick();
#if 1
  for ( l_n = 0; l_n < REPS; l_n++) {
    for ( l_i = 0; l_i < K; l_i++) {
      for ( l_j = 0; l_j < N; l_j++) {
        for ( l_jj = 0; l_jj < K; l_jj++) {
          LIBXSMM_PRAGMA_SIMD
          for (l_k = 0; l_k < N_CRUNS; l_k++) {
            LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_i, l_j, l_k, N, N_CRUNS)
              +=   l_a_de[(l_i*K)+l_jj]
                 * LIBXSMM_VLA_ACCESS(3, l_p_b, l_jj, l_j, l_k, N, N_CRUNS);
          }
        }
      }
    }
  }
#endif
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("%fs for dense\n", l_total);
  printf("%f GFLOPS for dense\n", ((double)((double)REPS * (double)K * (double)K * (double)N * (double)N_CRUNS) * 2.0) / (l_total * 1.0e9));

  l_xgemm_desc = libxsmm_gemm_descriptor_dinit(&l_xgemm_blob, LIBXSMM_GEMM_PRECISION(REALTYPE),
    K, N, K, 0, N, N, alpha, beta, flags, prefetch);

  /* sparse routine */
#if defined(__EDGE_EXECUTE_F32__)
  mykernel = libxsmm_create_packed_spxgemm_csr( l_xgemm_desc, N_CRUNS, l_rowptr, l_colidx, (const void*)l_a_sp ).smm;
#else
  mykernel = libxsmm_create_packed_spxgemm_csr( l_xgemm_desc, N_CRUNS, l_rowptr, l_colidx, (const void*)l_a_sp ).dmm;
#endif

  l_start = libxsmm_timer_tick();
  for ( l_n = 0; l_n < REPS; l_n++) {
    mykernel( l_a_sp, l_b, l_c_asm );
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  libxsmm_get_kernel_info( LIBXSMM_CONST_VOID_PTR(mykernel), &l_kinfo);
  l_libxsmmflops = l_kinfo.nflops;
  printf("%fs for sparse (asm)\n", l_total);
  printf("%f GFLOPS for sparse (asm), calculated\n", ((double)((double)REPS * (double)N * (double)l_elements * (double)N_CRUNS) * 2.0) / (l_total * 1.0e9));
  printf("%f GFLOPS for sparse (asm), libxsmm   \n", ((double)((double)REPS * (double)l_libxsmmflops)) / (l_total * 1.0e9));

  /* check for errors */
  l_max_error = (REALTYPE)0.0;
  for ( l_i = 0; l_i < K; l_i++) {
    for ( l_j = 0; l_j < N; l_j++) {
      for ( l_k = 0; l_k < N_CRUNS; l_k++ ) {
        if (fabs( LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_i, l_j, l_k, N, N_CRUNS)
                    - LIBXSMM_VLA_ACCESS(3, l_p_c_asm, l_i, l_j, l_k, N, N_CRUNS) ) > l_max_error ) {
          l_max_error = (REALTYPE)fabs( LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_i, l_j, l_k, N, N_CRUNS)
                                       -LIBXSMM_VLA_ACCESS(3, l_p_c_asm, l_i, l_j, l_k, N, N_CRUNS) );
        }
      }
    }
  }
  printf("max error: %f\n", l_max_error);

  printf("PERFDUMP,%s,%u,%i,%i,%i,%u,%u,%f,%f,%f\n", l_csr_file, REPS, M, N, K, l_elements, K * l_elements * N_CRUNS * 2, l_max_error, l_total, ((double)((double)REPS * (double)K * (double)l_elements * (double)N_CRUNS) * 2.0) / (l_total * 1.0e9) );

  /* free */
  libxsmm_free( l_a_de );
  libxsmm_free( l_b );
  libxsmm_free( l_c );
  libxsmm_free( l_c_gold );
  libxsmm_free( l_c_asm );

  free( l_a_sp );
  free( l_rowptr );
  free( l_colidx );

  return 0;
}

