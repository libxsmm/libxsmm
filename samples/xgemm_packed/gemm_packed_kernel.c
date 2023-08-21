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
#include <libxsmm.h>

#define REALTYPE float

LIBXSMM_INLINE
void matMulpacked(             libxsmm_blasint  i_r,
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
        o_c[l_n*i_ldC*i_r + l_m*i_r + l_r] = (i_beta != (REALTYPE)0) ? o_c[l_n*i_ldC*i_r + l_m*i_r + l_r] * i_beta : 0;
      }
    }
  }
  /* perform matmul */
  for ( l_k = 0; l_k < i_k; l_k++ ) {
    for ( l_m = 0; l_m < i_m; l_m++ ) {
      for ( l_n = 0; l_n < i_n; l_n++ ) {
        for ( l_r = 0; l_r < i_r; l_r++ ) {
          o_c[l_n*i_ldC*i_r + l_m*i_r + l_r] += i_a[l_k*i_ldA*i_r + l_m*i_r + l_r] * i_b[l_n*i_ldB*i_r + l_k*i_r + l_r];
        }
      }
    }
  }
}

LIBXSMM_INLINE
void print_help(void) {
  printf("\n\n");
  printf("1. Usage (dense*dense=dense, correctness and performance):\n");
  printf("    A Precision (F32, F64)\n");
  printf("    B Precision (F32, F64)\n");
  printf("    Compute Precision (F32, F64)\n");
  printf("    C Precision (F32, F64)\n");
  printf("    M\n");
  printf("    N\n");
  printf("    K\n");
  printf("    LDA\n");
  printf("    LDB\n");
  printf("    LDC\n");
  printf("    packed width\n");
  printf("    alpha: 1\n");
  printf("    beta: 0 or 1\n");
  printf("    0: A normal, 1: A trans\n");
  printf("    0: B normal, 1: B trans\n");
  printf("    #repetitions\n");
  printf("\n\n");
  printf("2. Usage (dense*dense=dense, performance only option available):\n");
  printf("    A Precision (F32, F64)\n");
  printf("    B Precision (F32, F64)\n");
  printf("    Compute Precision (F32, F64)\n");
  printf("    C Precision (F32, F64)\n");
  printf("    filename with space-sperated sizes (M N K LDA LDB LDC)\n");
  printf("    alpha: 1\n");
  printf("    beta: 0 or 1\n");
  printf("    0: A normal, 1: A trans\n");
  printf("    0: B normal, 1: B trans\n");
  printf("    #repetitions\n");
  printf("\n\n");
}


LIBXSMM_INLINE
libxsmm_datatype char_to_libxsmm_datatype( const char* dt ) {
  libxsmm_datatype dtype = LIBXSMM_DATATYPE_UNSUPPORTED;

  if ( (strcmp(dt, "F64") == 0) ) {
    dtype = LIBXSMM_DATATYPE_F64;
  } else if ( (strcmp(dt, "I64") == 0) ) {
    dtype = LIBXSMM_DATATYPE_I64;
  } else if ( (strcmp(dt, "F32") == 0) ) {
    dtype = LIBXSMM_DATATYPE_F32;
  } else if ( (strcmp(dt, "I32") == 0) ) {
    dtype = LIBXSMM_DATATYPE_I32;
  } else if ( (strcmp(dt, "F16") == 0) ) {
    dtype = LIBXSMM_DATATYPE_F16;
  } else if ( (strcmp(dt, "BF16") == 0) ) {
    dtype = LIBXSMM_DATATYPE_BF16;
  } else if ( (strcmp(dt, "I16") == 0) ) {
    dtype = LIBXSMM_DATATYPE_I16;
  } else if ( (strcmp(dt, "BF8") == 0) ) {
    dtype = LIBXSMM_DATATYPE_BF8;
  } else if ( (strcmp(dt, "HF8") == 0) ) {
    dtype = LIBXSMM_DATATYPE_HF8;
  } else if ( (strcmp(dt, "I8") == 0) ) {
    dtype = LIBXSMM_DATATYPE_I8;
  } else if ( (strcmp(dt, "U8") == 0) ) {
    dtype = LIBXSMM_DATATYPE_U8;
  } else if ( (strcmp(dt, "IMPLICIT") == 0) ) {
    dtype = LIBXSMM_DATATYPE_IMPLICIT;
  } else {
    dtype = LIBXSMM_DATATYPE_UNSUPPORTED;
  }

  return dtype;
}

int main(int argc, char* argv[]) {
  char* l_a_dt = NULL;
  char* l_b_dt = NULL;
  char* l_comp_dt = NULL;
  char* l_c_dt = NULL;
  libxsmm_datatype l_dtype_a, l_dtype_b, l_dtype_comp, l_dtype_c;
  libxsmm_blasint l_m = 0, l_n = 0, l_k = 0;
  libxsmm_blasint l_lda = 0, l_ldb = 0, l_ldc = 0;
  libxsmm_blasint l_r = 0;
  double l_alpha = 0;
  double l_beta = 0;
  int l_trans_a = 0;
  int l_trans_b = 0;
  int l_reps;
  char* l_file_name;

  if ( argc == 17 ) {
    l_a_dt = argv[1];
    l_b_dt = argv[2];
    l_comp_dt = argv[3];
    l_c_dt = argv[4];
    l_m = atoi(argv[5]);
    l_n = atoi(argv[6]);
    l_k = atoi(argv[7]);
    l_lda = atoi(argv[8]);
    l_ldb = atoi(argv[9]);
    l_ldc = atoi(argv[10]);
    l_r = atoi(argv[11]);
    l_alpha = atof(argv[12]);
    l_beta = atof(argv[13]);
    l_trans_a = atoi(argv[14]);
    l_trans_b = atoi(argv[15]);
    l_reps = atoi(argv[16]);
  } else if ( argc == 11 ) {
    l_a_dt = argv[1];
    l_b_dt = argv[2];
    l_comp_dt = argv[3];
    l_c_dt = argv[4];
    l_file_name = argv[5];
    l_alpha = atof(argv[6]);
    l_beta = atof(argv[7]);
    l_trans_a = atoi(argv[8]);
    l_trans_b = atoi(argv[9]);
    l_reps = atoi(argv[10]);
  } else {
    print_help();
    return EXIT_FAILURE;
  }

  l_dtype_a    = char_to_libxsmm_datatype( l_a_dt );
  l_dtype_b    = char_to_libxsmm_datatype( l_b_dt );
  l_dtype_comp = char_to_libxsmm_datatype( l_comp_dt );
  l_dtype_c    = char_to_libxsmm_datatype( l_c_dt );

  LIBXSMM_UNUSED( l_lda );
  LIBXSMM_UNUSED( l_ldb );
  LIBXSMM_UNUSED( l_ldc );
  LIBXSMM_UNUSED( l_file_name );
  LIBXSMM_UNUSED( l_alpha );
  LIBXSMM_UNUSED( l_beta );
  LIBXSMM_UNUSED( l_trans_a );
  LIBXSMM_UNUSED( l_trans_b );

  if ( !(
         ((l_dtype_a == LIBXSMM_DATATYPE_F64)  && (l_dtype_b == LIBXSMM_DATATYPE_F64)  && (l_dtype_comp == LIBXSMM_DATATYPE_F64) && (l_dtype_c == LIBXSMM_DATATYPE_F64))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_F32)  && (l_dtype_b == LIBXSMM_DATATYPE_F32)  && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_F32))
        ) ) {
    fprintf(stderr, "Unsupported precion combination: a: %s, b: %s, comp: %s, c: %s!\n", l_a_dt, l_b_dt, l_comp_dt, l_c_dt);
    exit(EXIT_FAILURE);
  }

  double flops = 2.0 * (double)l_m * (double)l_n * (double)l_k * (double)l_r * (double)l_reps;

  REALTYPE* a = (REALTYPE*)  libxsmm_aligned_malloc( l_m*l_k*l_r*sizeof(REALTYPE), 64 );
  REALTYPE* b = (REALTYPE*)  libxsmm_aligned_malloc( l_k*l_n*l_r*sizeof(REALTYPE), 64 );
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
    for (i = 0; i < l_k * l_n * l_r; ++i) {
      b[i] = (REALTYPE)libxsmm_rng_f64();
    }

    /* JIT code */
    mykernel = libxsmm_create_packed_gemm(gemm_shape, l_flags, l_prefetch_flags, l_r);

    /* run reference */
    matMulpacked(l_r,
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
      matMulpacked(l_r,
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
