/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <utils/libxsmm_utils.h>
#include <libxsmm.h>

#if defined(__EDGE_EXECUTE_F32__)
#define REALTYPE float
#else
#define REALTYPE double
#endif


LIBXSMM_INLINE
void matMulFusedAC(              libxsmm_blasint  i_r,
                                 libxsmm_blasint  i_m,
                                 libxsmm_blasint  i_n,
                                 libxsmm_blasint  i_k,
                                 libxsmm_blasint  i_ldA,
                                 libxsmm_blasint  i_ldB,
                                 libxsmm_blasint  i_ldC,
                                 REALTYPE         i_beta,
                           const REALTYPE        *i_a,
                           const REALTYPE        *i_b,
                                 REALTYPE        *o_c ) {
  libxsmm_blasint l_m = 0;
  libxsmm_blasint l_n = 0;
  libxsmm_blasint l_r = 0;
  libxsmm_blasint l_k = 0;

  /* init result matrix */
  for ( l_m = 0; l_m < i_m; l_m++ ) {
    for ( l_n = 0; l_n < i_n; l_n++ ) {
      for ( l_r = 0; l_r < i_r; l_r++ ) {
        o_c[l_m*i_ldC*i_r + l_n*i_r + l_r] = (i_beta != (REALTYPE)0) ? o_c[l_m*i_ldC*i_r + l_n*i_r + l_r] * i_beta : 0;
      }
    }
  }
  /* perform matmul */
  for ( l_k = 0; l_k < i_k; l_k++ ) {
    for ( l_m = 0; l_m < i_m; l_m++ ) {
      for ( l_n = 0; l_n < i_n; l_n++ ) {
        for ( l_r = 0; l_r < i_r; l_r++ ) {
          o_c[l_m*i_ldC*i_r + l_n*i_r + l_r] += i_a[l_m*i_ldA*i_r + l_k*i_r + l_r] * i_b[l_k*i_ldB + l_n];
        }
      }
    }
  }
}


int main(int argc, char* argv[]) {
#if defined(__EDGE_EXECUTE_F32__)
  libxsmm_blasint l_r = 16;
#else
  libxsmm_blasint l_r = 8;
#endif
  libxsmm_blasint l_m = 1 < argc ? atoi(argv[1]) : 0;
  libxsmm_blasint l_n = 2 < argc ? atoi(argv[2]) : 0;
  libxsmm_blasint l_k = 3 < argc ? atoi(argv[3]) : 0;
  REALTYPE l_beta = (REALTYPE)(4 < argc ? atof(argv[4]) : 0);
  libxsmm_blasint l_reps = 5 < argc ? atoi(argv[5]) : 0;
  double flops = 2.0 * (double)l_m * (double)l_n * (double)l_k * (double)l_r * (double)l_reps;

  REALTYPE* a = (REALTYPE*)  libxsmm_aligned_malloc( l_m*l_k*l_r*sizeof(REALTYPE), 64 );
  REALTYPE* b = (REALTYPE*)  libxsmm_aligned_malloc( l_k*l_n*sizeof(REALTYPE), 64 );
  REALTYPE* c1 = (REALTYPE*) libxsmm_aligned_malloc( l_m*l_n*l_r*sizeof(REALTYPE), 64 );
  REALTYPE* c2 = (REALTYPE*) libxsmm_aligned_malloc( l_m*l_n*l_r*sizeof(REALTYPE), 64 );

  libxsmm_gemmfunction mykernel = NULL;
  const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(
    l_m, l_n, l_k, l_k, l_n, l_n, LIBXSMM_DATATYPE(REALTYPE),
    LIBXSMM_DATATYPE(REALTYPE), LIBXSMM_DATATYPE(REALTYPE), LIBXSMM_DATATYPE(REALTYPE) );
  const libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N') | ( ( l_beta == 0 ) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0 );
  const libxsmm_bitfield l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
  libxsmm_gemm_param gemm_param;

  libxsmm_timer_tickint l_start, l_end;
  double l_total_ref, l_total_opt;
  double max_error = 0.0;
  double gflops_ref = 0.0;
  double gflops_opt = 0.0;
  double gflops_opt2 = 0.0;
  libxsmm_blasint i = 0;
  unsigned long long l_libxsmmflops;
  libxsmm_kernel_info l_kinfo;
  int result = EXIT_SUCCESS;

  if (NULL != a && NULL != b && NULL != c1 && NULL != c2) {
    for (i = 0; i < l_m * l_n * l_r; ++i) {
      c1[i] = (REALTYPE)libxsmm_rng_f64();
    }
    for (i = 0; i < l_m * l_n * l_r; ++i) {
      c2[i] = c1[i];
    }
    for (i = 0; i < l_m * l_k * l_r; ++i) {
      a[i] = (REALTYPE)libxsmm_rng_f64();
    }
    for (i = 0; i < l_k * l_n; ++i) {
      b[i] = (REALTYPE)libxsmm_rng_f64();
    }

    /* JIT code */
    mykernel = libxsmm_create_packed_gemm_ac_rm_v2(gemm_shape, l_flags, l_prefetch_flags, l_r);

    /* run reference */
    matMulFusedAC(l_r,
      l_m, l_n, l_k,
      l_k,
      l_n,
      l_n,
      l_beta,
      a,
      b,
      c1);

    /* run optimized */
    memset(&gemm_param, 0, sizeof(libxsmm_gemm_param));
    gemm_param.a.primary = (void*)a;
    gemm_param.b.primary = (void*)b;
    gemm_param.c.primary = (void*)c2;
    mykernel(&gemm_param);

    /* check correctness */
    for (i = 0; i < l_m * l_n * l_r; ++i) {
      if (max_error < LIBXSMM_FABS(c1[i] - c2[i])) {
        max_error = LIBXSMM_FABS(c1[i] - c2[i]);
      }
    }

    printf("max. error: %f\n", max_error);

    /* lets run some performance test */
    l_start = libxsmm_timer_tick();
    for (i = 0; i < l_reps; ++i) {
      /* run reference */
      matMulFusedAC(l_r,
        l_m, l_n, l_k,
        l_k,
        l_n,
        l_n,
        l_beta,
        a,
        b,
        c1);
    }
    l_end = libxsmm_timer_tick();
    l_total_ref = libxsmm_timer_duration(l_start, l_end);

    l_start = libxsmm_timer_tick();
    for (i = 0; i < l_reps; ++i) {
      /* run optimized */
      mykernel(&gemm_param);
    }
    l_end = libxsmm_timer_tick();
    l_total_opt = libxsmm_timer_duration(l_start, l_end);
    libxsmm_get_kernel_info(LIBXSMM_CONST_VOID_PTR(mykernel), &l_kinfo);
    l_libxsmmflops = l_kinfo.nflops;

    gflops_ref = (flops / l_total_ref) / 1e9;
    gflops_opt = (flops / l_total_opt) / 1e9;
    gflops_opt2 = (((double)l_libxsmmflops * l_reps) / l_total_opt) / 1e9;

    printf("GFLOPS ref: %f\n", gflops_ref);
    printf("GFLOPS opt, calculated: %f\n", gflops_opt);
    printf("GFLOPS opt, libxsmm:    %f\n", gflops_opt2);
  }
  else result = EXIT_FAILURE;

  libxsmm_free( a );
  libxsmm_free( b );
  libxsmm_free( c1 );
  libxsmm_free( c2 );

  return result;
}
