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
#include <libxsmm_utils.h>

LIBXSMM_INLINE
void matMulpacked(             libxsmm_blasint  i_r,
                               libxsmm_blasint  i_m,
                               libxsmm_blasint  i_n,
                               libxsmm_blasint  i_k,
                               libxsmm_blasint  i_ldA,
                               libxsmm_blasint  i_ldB,
                               libxsmm_blasint  i_ldC,
                               libxsmm_blasint  i_beta_one,
                               libxsmm_datatype i_dt_a,
                               libxsmm_datatype i_dt_b,
                               libxsmm_datatype i_dt_c,
                               libxsmm_datatype i_dt_comp,
                         const char             *i_a,
                         const char             *i_b,
                               char             *o_c ) {
  libxsmm_blasint l_m = 0;
  libxsmm_blasint l_n = 0;
  libxsmm_blasint l_r = 0;
  libxsmm_blasint l_k = 0;

  /* init result matrix */
  for ( l_m = 0; l_m < i_m; l_m++ ) {
    for ( l_n = 0; l_n < i_n; l_n++ ) {
      for ( l_r = 0; l_r < i_r; l_r++ ) {
        if ( (i_dt_a == LIBXSMM_DATATYPE_F32) && (i_dt_b == LIBXSMM_DATATYPE_F32) && (i_dt_c == LIBXSMM_DATATYPE_F32) && (i_dt_comp == LIBXSMM_DATATYPE_F32) ) {
          float* f_c = (float*)o_c;
          f_c[l_n*i_ldC*i_r + l_m*i_r + l_r] = (i_beta_one != 0) ? f_c[l_n*i_ldC*i_r + l_m*i_r + l_r] : 0.0f;
        } else if ( (i_dt_a == LIBXSMM_DATATYPE_F64) && (i_dt_b == LIBXSMM_DATATYPE_F64) && (i_dt_c == LIBXSMM_DATATYPE_F64) && (i_dt_comp == LIBXSMM_DATATYPE_F64) ) {
          double* d_c = (double*)o_c;
          d_c[l_n*i_ldC*i_r + l_m*i_r + l_r] = (i_beta_one != 0) ? d_c[l_n*i_ldC*i_r + l_m*i_r + l_r] : 0.0;
        } else {
          /* shouldn't happen */
        }
      }
    }
  }
  /* perform matmul */
  for ( l_k = 0; l_k < i_k; l_k++ ) {
    for ( l_m = 0; l_m < i_m; l_m++ ) {
      for ( l_n = 0; l_n < i_n; l_n++ ) {
        for ( l_r = 0; l_r < i_r; l_r++ ) {
          if ( (i_dt_a == LIBXSMM_DATATYPE_F32) && (i_dt_b == LIBXSMM_DATATYPE_F32) && (i_dt_c == LIBXSMM_DATATYPE_F32) && (i_dt_comp == LIBXSMM_DATATYPE_F32) ) {
            float* f_c = (float*)o_c;
            const float* f_a = (const float*)i_a;
            const float* f_b = (const float*)i_b;
            f_c[l_n*i_ldC*i_r + l_m*i_r + l_r] += f_a[l_k*i_ldA*i_r + l_m*i_r + l_r] * f_b[l_n*i_ldB*i_r + l_k*i_r + l_r];
          } else if ( (i_dt_a == LIBXSMM_DATATYPE_F64) && (i_dt_b == LIBXSMM_DATATYPE_F64) && (i_dt_c == LIBXSMM_DATATYPE_F64) && (i_dt_comp == LIBXSMM_DATATYPE_F64) ) {
            double* d_c = (double*)o_c;
            const double* d_a = (const double*)i_a;
            const double* d_b = (const double*)i_b;
            d_c[l_n*i_ldC*i_r + l_m*i_r + l_r] += d_a[l_k*i_ldA*i_r + l_m*i_r + l_r] * d_b[l_n*i_ldB*i_r + l_k*i_r + l_r];
          } else {
            /* shouldn't happen */
          }
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

LIBXSMM_INLINE
void init_data( libxsmm_blasint  i_r,
                libxsmm_blasint  i_m,
                libxsmm_blasint  i_n,
                libxsmm_blasint  i_ld,
                libxsmm_datatype i_dt,
                char             *io_data ) {
  libxsmm_blasint l_m = 0;
  libxsmm_blasint l_n = 0;
  libxsmm_blasint l_r = 0;

  /* init result matrix */
  for ( l_m = 0; l_m < i_m; l_m++ ) {
    for ( l_n = 0; l_n < i_n; l_n++ ) {
      for ( l_r = 0; l_r < i_r; l_r++ ) {
        if ( i_dt == LIBXSMM_DATATYPE_F32 ) {
          float* f_data = (float*)io_data;
          f_data[l_n*i_ld*i_r + l_m*i_r + l_r] = (float)libxsmm_rng_f64();
        } else if ( i_dt == LIBXSMM_DATATYPE_F64 ) {
          double* d_data = (double*)io_data;
          d_data[l_n*i_ld*i_r + l_m*i_r + l_r] = libxsmm_rng_f64();
        } else {
          /* shouldn't happen */
        }
      }
    }
  }
}

LIBXSMM_INLINE
void copy_data( libxsmm_blasint  i_r,
                libxsmm_blasint  i_m,
                libxsmm_blasint  i_n,
                libxsmm_blasint  i_ld,
                libxsmm_datatype i_dt,
                const char       *i_data,
                char             *o_data ) {
  libxsmm_blasint l_m = 0;
  libxsmm_blasint l_n = 0;
  libxsmm_blasint l_r = 0;

  /* init result matrix */
  for ( l_m = 0; l_m < i_m; l_m++ ) {
    for ( l_n = 0; l_n < i_n; l_n++ ) {
      for ( l_r = 0; l_r < i_r; l_r++ ) {
        if ( i_dt == LIBXSMM_DATATYPE_F32 ) {
          const float* fi_data = (const float*)i_data;
          float* fo_data = (float*)o_data;
          fo_data[l_n*i_ld*i_r + l_m*i_r + l_r] = fi_data[l_n*i_ld*i_r + l_m*i_r + l_r];
        } else if ( i_dt == LIBXSMM_DATATYPE_F64 ) {
          const double* di_data = (const double*)i_data;
          double* do_data = (double*)o_data;
          do_data[l_n*i_ld*i_r + l_m*i_r + l_r] = di_data[l_n*i_ld*i_r + l_m*i_r + l_r];
        } else {
          /* shouldn't happen */
        }
      }
    }
  }
}

LIBXSMM_INLINE
double check_data( libxsmm_blasint  i_r,
                   libxsmm_blasint  i_m,
                   libxsmm_blasint  i_n,
                   libxsmm_blasint  i_ld,
                   libxsmm_datatype i_dt,
                   const char       *i_gold,
                   const char       *o_data ) {
  libxsmm_matdiff_info l_diff;
  libxsmm_blasint l_2d_ld = i_ld*i_r;
  double error = 0.0;

  libxsmm_matdiff_clear(&l_diff);
  if ( i_dt == LIBXSMM_DATATYPE_F64 ) {
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F64, i_m*i_r, i_n, i_gold, o_data, &l_2d_ld, &l_2d_ld);
    error = libxsmm_matdiff_epsilon(&l_diff);
  } else if ( i_dt == LIBXSMM_DATATYPE_F32 ) {
#if 0
    float* data_gold_f = (float*)i_gold;
    float* data_f      = (float*)o_data;
    libxsmm_blasint l_m = 0;
    libxsmm_blasint l_n = 0;
    libxsmm_blasint l_r = 0;

    for (l_m = 0; l_m < i_m; l_m++) {
      for (l_n = 0; l_n < i_n; l_n++) {
        for (l_r = 0; l_r < i_r; l_r++) {
          printf("gold: %10.10f, computed: %10.10f, diff: %10.10f\n", data_gold_f[l_n*i_ld*i_r + l_m*i_r + l_r], data_f[l_n*i_ld*i_r + l_m*i_r + l_r], data_gold_f[l_n*i_ld*i_r + l_m*i_r + l_r]-data_f[l_n*i_ld*i_r + l_m*i_r + l_r] );
        }
      }
    }
#endif
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, i_m*i_r, i_n, i_gold, o_data, &l_2d_ld, &l_2d_ld);
    error = libxsmm_matdiff_epsilon(&l_diff);
  } else {
    error = 100.0;
  }

  printf("\nPrinting Norms:\n");
  printf("L1 reference  : %.25g\n", l_diff.l1_ref);
  printf("L1 test       : %.25g\n", l_diff.l1_tst);
  printf("L2 abs.error  : %.24f\n", l_diff.l2_abs);
  printf("L2 rel.error  : %.24f\n", l_diff.l2_rel);
  printf("Linf abs.error: %.24f\n", l_diff.linf_abs);
  printf("Linf rel.error: %.24f\n", l_diff.linf_rel);
  printf("Check-norm    : %.24f\n\n", error);

  return error;
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
  int result = EXIT_SUCCESS;
  unsigned int l_keep_going = 0;
  FILE *l_file_handle = NULL;
  unsigned int l_file_input = 0;

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
    l_file_input = 1;
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

  LIBXSMM_UNUSED( l_file_name );
  LIBXSMM_UNUSED( l_alpha );
  LIBXSMM_UNUSED( l_trans_a );

  LIBXSMM_UNUSED( l_trans_b );

  if ( (l_trans_a != 0) || (l_trans_b != 0) ) {
    fprintf(stderr, "Unsupported transpose combination: a: %i, b: %i\n", l_trans_a, l_trans_b);
    exit(EXIT_FAILURE);
  }

  l_dtype_a    = char_to_libxsmm_datatype( l_a_dt );
  l_dtype_b    = char_to_libxsmm_datatype( l_b_dt );
  l_dtype_comp = char_to_libxsmm_datatype( l_comp_dt );
  l_dtype_c    = char_to_libxsmm_datatype( l_c_dt );

  /* check for supported precision */
  if ( !(
         ((l_dtype_a == LIBXSMM_DATATYPE_F64)  && (l_dtype_b == LIBXSMM_DATATYPE_F64)  && (l_dtype_comp == LIBXSMM_DATATYPE_F64) && (l_dtype_c == LIBXSMM_DATATYPE_F64))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_F32)  && (l_dtype_b == LIBXSMM_DATATYPE_F32)  && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_F32))
        ) ) {
    fprintf(stderr, "Unsupported precision combination: a: %s, b: %s, comp: %s, c: %s!\n", l_a_dt, l_b_dt, l_comp_dt, l_c_dt);
    exit(EXIT_FAILURE);
  }

  if ( l_file_input != 0 ) {
    l_file_handle = fopen( l_file_name, "r" );
  } else {
    printf("------------------------------------------------\n");
    printf("RUNNING (%ix%iX%i) X (%ix%iX%i) = (%ix%iX%i)\na:%s, b:%s, comp:%s, c:%s\n", l_m, l_k, l_r, l_k, l_n, l_r, l_m, l_n, l_r, l_a_dt, l_b_dt, l_comp_dt, l_c_dt);
    printf("------------------------------------------------\n");
  }

  l_keep_going = 0;
  do {
    if ( l_file_input != 0 ) {
      char l_line[512];
      if ( fgets( l_line, 512, l_file_handle) == NULL ) {
        l_keep_going = 0;
        break;
      } else {
        l_keep_going = 1;
      }
      if ( 7 != sscanf( l_line, "%i %i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_lda, &l_ldb, &l_ldc, &l_r ) ) exit(EXIT_FAILURE);

      if (l_keep_going == 0) break;
    }
    printf("CMDLINE: %s %s %s %s %s %i %i %i %i %i %i %i %f %f %i %i %i\n", argv[0], l_a_dt, l_b_dt, l_comp_dt, l_c_dt, l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_r, l_alpha, l_beta, l_trans_a, l_trans_b, l_reps);
    {
      /* calculate flops */
      double l_flops = 2.0 * (double)l_m * (double)l_n * (double)l_k * (double)l_r * (double)l_reps;
      /* allocate data */
      char* a  = (char*) libxsmm_aligned_malloc( l_lda*l_k*l_r*LIBXSMM_TYPESIZE(l_dtype_a), 64 );
      char* b  = (char*) libxsmm_aligned_malloc( l_ldb*l_n*l_r*LIBXSMM_TYPESIZE(l_dtype_b), 64 );
      char* c1 = (char*) libxsmm_aligned_malloc( l_ldc*l_n*l_r*LIBXSMM_TYPESIZE(l_dtype_c), 64 );
      char* c2 = (char*) libxsmm_aligned_malloc( l_ldc*l_n*l_r*LIBXSMM_TYPESIZE(l_dtype_c), 64 );
      /* init libxsmm kernel */
      libxsmm_gemmfunction mykernel = NULL;
      const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(
        l_m, l_n, l_k, l_lda, l_ldb, l_ldc, l_dtype_a,
        l_dtype_b, l_dtype_c, l_dtype_comp );
      const libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N') | ( ( l_beta == 0 ) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0 );
      const libxsmm_bitfield l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
      unsigned long long l_libxsmmflops;
      libxsmm_kernel_info l_kinfo;
      libxsmm_gemm_param gemm_param;
      libxsmm_timer_tickint l_start, l_end;
      double l_total_opt;
      double l_max_error = 0.0;
      double l_gflops_opt = 0.0;
      double l_gflops_opt2 = 0.0;
      libxsmm_blasint i = 0;

      /* JIT code */
      mykernel = libxsmm_create_packed_gemm(gemm_shape, l_flags, l_prefetch_flags, l_r);

      if ( mykernel == NULL ) {
        printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
        exit(-1);
      }

      /* init data */
      init_data( l_r, l_m, l_n, l_ldc, l_dtype_c, c1 );
      copy_data( l_r, l_m, l_n, l_ldc, l_dtype_c, c1, c2 );
      init_data( l_r, l_m, l_k, l_lda, l_dtype_a, a  );
      init_data( l_r, l_k, l_n, l_ldb, l_dtype_b, b  );

      /* run reference */
      matMulpacked(l_r,
        l_m, l_n, l_k,
        l_lda,
        l_ldb,
        l_ldc,
        l_beta,
        l_dtype_a,
        l_dtype_b,
        l_dtype_c,
        l_dtype_comp,
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
      l_max_error = check_data( l_r, l_m, l_n, l_ldc, l_dtype_c, c1, c2 );
      if ( l_max_error > 0.00001 ) {
        result = EXIT_FAILURE;
      }

      l_start = libxsmm_timer_tick();
      for (i = 0; i < l_reps; ++i) {
        /* run optimized */
        mykernel(&gemm_param);
      }
      l_end = libxsmm_timer_tick();
      l_total_opt = libxsmm_timer_duration(l_start, l_end);
      libxsmm_get_kernel_info(LIBXSMM_CONST_VOID_PTR(mykernel), &l_kinfo);
      l_libxsmmflops = l_kinfo.nflops;

      l_gflops_opt = (l_flops / l_total_opt) / 1e9;
      l_gflops_opt2 = (((double)l_libxsmmflops * l_reps) / l_total_opt) / 1e9;

      printf("max. error: %f\n", l_max_error);
      printf("GFLOPS opt, calculated: %f\n", l_gflops_opt);
      printf("GFLOPS opt, libxsmm:    %f\n", l_gflops_opt2);

      libxsmm_free( a );
      libxsmm_free( b );
      libxsmm_free( c1 );
      libxsmm_free( c2 );
    }
  } while ( l_keep_going );

  return result;
}
