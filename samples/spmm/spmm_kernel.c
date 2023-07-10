/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <float.h>
#if defined(__APPLE__) && defined(__arm64__)
# include <pthread.h>
#endif

typedef struct spmm_def {
  libxsmm_datatype a_type;
  libxsmm_datatype b_type;
  libxsmm_datatype c_type;
  libxsmm_datatype comp_type;
  libxsmm_blasint m;
  libxsmm_blasint n;
  libxsmm_blasint k;
  libxsmm_blasint bk;
  libxsmm_blasint bn;
  libxsmm_blasint m_blocks;
  double beta;
  int trans_a;
  int trans_b;
  int vnni_a;
  int vnni_b;
  int vnni_c;
  int unsigned_a;
  int unsigned_b;
  int unsigned_c;
  int prefetch;
  int tc_config;
} spmm_def;

int ullcompare( const void* a , const void* b ) {
  const unsigned long long aull = *( const unsigned long long* )a;
  const unsigned long long bull = *( const unsigned long long* )b;
  if ( aull < bull ) {
    return -1;
  } else if( aull > bull ) {
    return 1;
  } else {
    return 0;
  }
}

void shuffle_array(unsigned long long *array, int n) {
  if (n > 1)
  {
    int i;
    for (i = 0; i < n - 1; i++)
    {
      int j = i + rand() / (RAND_MAX / (n - i) + 1);
      unsigned long long t = array[j];
      array[j] = array[i];
      array[i] = t;
    }
  }
}

int is_dense_grid_point(unsigned long long grid_point_id, int n_dense_grid_points, unsigned long long *grid_point_array) {
  unsigned long long key = grid_point_id;
  unsigned long long *found_ptr = (unsigned long long*) bsearch(&key, grid_point_array, n_dense_grid_points, sizeof(unsigned long long), ullcompare);
  return ((found_ptr == NULL) ? 0 : 1);
}

void dense_gemm_ref(spmm_def *i_spmm_def, char *l_a, char *l_b, char *l_c_gold) {
  spmm_def l_spmm_def = *i_spmm_def;
  libxsmm_blasint l_m = l_spmm_def.m;
  libxsmm_blasint l_n = l_spmm_def.n;
  libxsmm_blasint l_k = l_spmm_def.k;
  libxsmm_blasint l_m_blocks = l_spmm_def.m_blocks;
  libxsmm_blasint l_j, l_i, l_jj, l_m_block_id;

  /* dense routine */
  if (l_spmm_def.b_type == LIBXSMM_DATATYPE_F32) {
    float *B = (float*)l_b;
    float *A = (float*)l_a;
    float *C = (float*)l_c_gold;
    for (l_m_block_id = 0; l_m_block_id < l_m_blocks; l_m_block_id++) {
      for ( l_j = 0; l_j < l_n; l_j++) {
        LIBXSMM_PRAGMA_SIMD
        for ( l_i = 0; l_i < l_m; l_i++) {
          if ( l_spmm_def.beta == 0 ) {
            C[l_j * l_m + l_i] = 0.0;
          }
          for ( l_jj = 0; l_jj < l_k; l_jj++) {
            C[l_j * l_m + l_i] += A[l_jj * l_m + l_i] * B[l_j*l_k + l_jj];
          }
        }
      }
      A = A + l_k * l_m;
      C = C + l_n * l_m;
    }
  } else if (l_spmm_def.b_type == LIBXSMM_DATATYPE_BF16) {
    libxsmm_bfloat16 *B = (libxsmm_bfloat16*)l_b;
    libxsmm_bfloat16 *A = (libxsmm_bfloat16*)l_a;
    libxsmm_bfloat16 *C = (libxsmm_bfloat16*)l_c_gold;
    for (l_m_block_id = 0; l_m_block_id < l_m_blocks; l_m_block_id++) {
      for ( l_j = 0; l_j < l_n; l_j++) {
        LIBXSMM_PRAGMA_SIMD
        for ( l_i = 0; l_i < l_m; l_i++) {
          float acc = 0.0;
          libxsmm_bfloat16 h_acc;
          if ( l_spmm_def.beta == 0 ) {
            acc = 0.0f;
          } else {
            libxsmm_bfloat16_f32 tmp/* = { 0 }*/;
            tmp.i[0] = 0;
            tmp.i[1] = C[l_j * l_m + l_i];
            acc = tmp.f;
          }
          for ( l_jj = 0; l_jj < l_k; l_jj++) {
            union libxsmm_bfloat16_f32 tmp_a_f;
            union libxsmm_bfloat16_f32 tmp_b_f;
            tmp_a_f.i[0] = 0;
            tmp_a_f.i[1] = A[l_jj * l_m + l_i];
            tmp_b_f.i[0] = 0;
            tmp_b_f.i[1] = B[l_j*l_k + l_jj];
            acc += tmp_a_f.f * tmp_b_f.f;
          }
          libxsmm_rne_convert_fp32_bf16( &acc, &h_acc, 1 );
          C[l_j * l_m + l_i] = h_acc;
        }
      }
      A = A + l_k * l_m;
      C = C + l_n * l_m;
    }
  } else if (l_spmm_def.b_type == LIBXSMM_DATATYPE_I8) {
    if ((l_spmm_def.unsigned_a == 1) && (l_spmm_def.unsigned_b == 0)) {
      char *B = (char*)l_b;
      unsigned char *A = (unsigned char*)l_a;
      int *C = (int*)l_c_gold;
      for (l_m_block_id = 0; l_m_block_id < l_m_blocks; l_m_block_id++) {
        for ( l_j = 0; l_j < l_n; l_j++) {
          LIBXSMM_PRAGMA_SIMD
          for ( l_i = 0; l_i < l_m; l_i++) {
            if ( l_spmm_def.beta == 0 ) {
              C[l_j * l_m + l_i] = 0;
            }
            for ( l_jj = 0; l_jj < l_k; l_jj++) {
              C[l_j * l_m + l_i] += (unsigned char)A[l_jj * l_m + l_i] * (int)B[l_j*l_k + l_jj];
            }
          }
        }
        A = A + l_k * l_m;
        C = C + l_n * l_m;
      }
    } else if ((l_spmm_def.unsigned_a == 0) && (l_spmm_def.unsigned_b == 1)) {
      unsigned char *B = (unsigned char*)l_b;
      char *A = (char*)l_a;
      int *C = (int*)l_c_gold;
      for (l_m_block_id = 0; l_m_block_id < l_m_blocks; l_m_block_id++) {
        for ( l_j = 0; l_j < l_n; l_j++) {
          LIBXSMM_PRAGMA_SIMD
          for ( l_i = 0; l_i < l_m; l_i++) {
            if ( l_spmm_def.beta == 0 ) {
              C[l_j * l_m + l_i] = 0;
            }
            for ( l_jj = 0; l_jj < l_k; l_jj++) {
              C[l_j * l_m + l_i] += (int)A[l_jj * l_m + l_i] * (unsigned char)B[l_j*l_k + l_jj];
            }
          }
        }
        A = A + l_k * l_m;
        C = C + l_n * l_m;
      }
    } else {
      /* Should not happen */
    }
  } else {
    /* Should not happen  */
  }
}

void create_spmm_inputs(spmm_def *i_spmm_def, double l_sparsity_frac, char *l_a, char *l_b, char *l_a_spmm, char **l_b_sp_bcsc_data_ptr, unsigned int **l_colptr_ptr, unsigned int **l_rowidx_ptr) {
  spmm_def l_spmm_def = *i_spmm_def;
  libxsmm_blasint l_m = l_spmm_def.m;
  libxsmm_blasint l_n = l_spmm_def.n;
  libxsmm_blasint l_k = l_spmm_def.k;
  libxsmm_blasint l_bn = l_spmm_def.bn;
  libxsmm_blasint l_bk = l_spmm_def.bk;
  libxsmm_blasint l_vnni_b = l_spmm_def.vnni_b;
  libxsmm_blasint l_trans_b = l_spmm_def.trans_b;
  libxsmm_blasint l_j, l_i, l_jj, l_m_block_id;
  libxsmm_blasint l_m_blocks = l_spmm_def.m_blocks;
  unsigned int n_grid_points = (l_n/l_bn) * (l_k/l_bk);
  unsigned long long *grid_point_array = (unsigned long long *) malloc(n_grid_points * sizeof(unsigned long long));
  long long n_dense_grid_points = (long long) ((double)(1.0-l_sparsity_frac) * n_grid_points);
  unsigned int nnz = 0;
  unsigned int l_val_idx = 0;
  unsigned int l_nz_block_id = 0;
  int l_a_vnni_factor =  libxsmm_cpuid_dot_pack_factor(l_spmm_def.a_type);
  unsigned int *l_colptr;
  unsigned int *l_rowidx;
  char *l_b_sp_bcsc_data;

  if (l_a_vnni_factor != 1) {
    for (l_m_block_id = 0; l_m_block_id < l_m_blocks; l_m_block_id++) {
      for ( l_i = 0; l_i < l_m; l_i++ ) {
        for ( l_j = 0; l_j < l_k/l_a_vnni_factor; l_j++ ) {
          for ( l_jj = 0; l_jj < l_a_vnni_factor; l_jj++ ) {
            if (l_spmm_def.a_type == LIBXSMM_DATATYPE_BF16) {
              libxsmm_bfloat16 *l_a_bf16 = (libxsmm_bfloat16*)l_a + l_m_block_id * l_k * l_m;
              libxsmm_bfloat16 *l_a_spmm_bf16 = (libxsmm_bfloat16*)l_a_spmm + l_m_block_id * l_k * l_m;
              l_a_spmm_bf16[l_j * (l_m*l_a_vnni_factor) + l_i * l_a_vnni_factor + l_jj] = l_a_bf16[(l_j * l_a_vnni_factor + l_jj) * l_m + l_i];
            } else if (l_spmm_def.a_type == LIBXSMM_DATATYPE_I8) {
              char *l_a_i8 = (char*)l_a + l_m_block_id * l_k * l_m;
              char *l_a_spmm_i8 = (char*)l_a_spmm + l_m_block_id * l_k * l_m;
              l_a_spmm_i8[l_j * (l_m*l_a_vnni_factor) + l_i * l_a_vnni_factor + l_jj] = l_a_i8[(l_j * l_a_vnni_factor + l_jj) * l_m + l_i];
            } else {
              /* Should not happen  */
            }
          }
        }
      }
    }
  }

  /* Sparsify B matrix and generate matrix in BCSC format */
  for (l_i = 0; l_i < n_grid_points; l_i++) {
    grid_point_array[l_i] = l_i;
  }
  /* Pemute array of n grid points and sparsify the ones with id > n_dense_grid_points */
  shuffle_array(grid_point_array, n_grid_points);
  qsort(grid_point_array, n_dense_grid_points, sizeof(unsigned long long), ullcompare);
  for ( l_i = 0; l_i < l_n/l_bn; l_i++ ) {
    for ( l_j = 0; l_j < l_k/l_bk; l_j++ ) {
      if (is_dense_grid_point(l_i * (l_k/l_bk) + l_j, n_dense_grid_points, grid_point_array) == 0) {
        unsigned int l_ui = l_i * l_bn;
        unsigned int l_uj = l_j * l_bk;
        unsigned int l_di = 0, l_dj = 0;
        for (l_di = 0; l_di < l_bn; l_di++) {
          for (l_dj = 0; l_dj < l_bk; l_dj++) {
            if (l_spmm_def.b_type == LIBXSMM_DATATYPE_F32) {
              float *ptr_f32 = (float*)l_b + ((l_ui + l_di) * l_k + (l_uj + l_dj));
              *ptr_f32 = (float)0;
            } else if (l_spmm_def.b_type == LIBXSMM_DATATYPE_BF16) {
              libxsmm_bfloat16 *ptr_bf16 = (libxsmm_bfloat16*)l_b + ((l_ui + l_di) * l_k + (l_uj + l_dj));
              *ptr_bf16 = (libxsmm_bfloat16)0;
            } else if (l_spmm_def.b_type == LIBXSMM_DATATYPE_I8) {
              char *ptr_char = (char*)l_b + ((l_ui + l_di) * l_k + (l_uj + l_dj));
              *ptr_char = (char)0;
            } else {
              /* Should not happen  */
            }
          }
        }
      } else {
        nnz += l_bn*l_bk;
      }
    }
  }

  l_colptr   = (unsigned int*) libxsmm_aligned_malloc( (l_n/l_bn+1)*sizeof(unsigned int), 64 );
  l_rowidx   = (unsigned int*) libxsmm_aligned_malloc( nnz/(l_bn*l_bk)*sizeof(unsigned int), 64 );
  l_b_sp_bcsc_data = (char*) libxsmm_aligned_malloc( nnz*LIBXSMM_TYPESIZE(l_spmm_def.b_type), 64 );

  *l_colptr_ptr = l_colptr;
  *l_rowidx_ptr = l_rowidx;
  *l_b_sp_bcsc_data_ptr = l_b_sp_bcsc_data;

  l_nz_block_id = 0;
  l_colptr[l_n/l_bn] = nnz/(l_bn*l_bk);
  for ( l_i = 0; l_i < l_n/l_bn; l_i++ ) {
    l_colptr[l_i] = l_nz_block_id;
    for ( l_j = 0; l_j < l_k/l_bk; l_j++ ) {
      if (is_dense_grid_point(l_i * (l_k/l_bk) + l_j, n_dense_grid_points, grid_point_array) > 0) {
        unsigned int l_ui = l_i * l_bn;
        unsigned int l_uj = l_j * l_bk;
        unsigned int l_di = 0, l_dj = 0;
        l_rowidx[l_nz_block_id] = l_j;
        for (l_di = 0; l_di < l_bn; l_di++) {
          for (l_dj = 0; l_dj < l_bk; l_dj++) {
            if (l_spmm_def.b_type == LIBXSMM_DATATYPE_F32) {
              float *ptr_f32 = (float*)l_b + ((l_ui + l_di) * l_k + (l_uj + l_dj));
              float *sp_ptr_f32 = (float*)l_b_sp_bcsc_data + l_val_idx;
              *sp_ptr_f32 = *ptr_f32;
            } else if (l_spmm_def.b_type == LIBXSMM_DATATYPE_BF16) {
              libxsmm_bfloat16 *ptr_bf16 = (libxsmm_bfloat16*)l_b + ((l_ui + l_di) * l_k + (l_uj + l_dj));
              libxsmm_bfloat16 *sp_ptr_bf16 = (libxsmm_bfloat16*)l_b_sp_bcsc_data + l_val_idx;
              *sp_ptr_bf16 = *ptr_bf16;
            } else if (l_spmm_def.b_type == LIBXSMM_DATATYPE_I8) {
              char *ptr_char = (char*)l_b + ((l_ui + l_di) * l_k + (l_uj + l_dj));
              char *sp_ptr_char = (char*)l_b_sp_bcsc_data + l_val_idx;
              *sp_ptr_char = *ptr_char;
            } else {
              /* Should not happen  */
            }
            l_val_idx++;
          }
        }
        l_nz_block_id++;
      }
    }
  }

  if (l_vnni_b > 0 && l_trans_b > 0) {
    unsigned int l_di = 0, l_dj = 0;
    int l_b_vnni_factor =  libxsmm_cpuid_dot_pack_factor(l_spmm_def.b_type);
    for ( l_i = 0; l_i < nnz/(l_bk*l_bn); l_i++) {
      char tmp_block[l_bk*l_bn*LIBXSMM_TYPESIZE(l_spmm_def.b_type)];
      memcpy(tmp_block, &l_b_sp_bcsc_data[l_i*(l_bk*l_bn)*LIBXSMM_TYPESIZE(l_spmm_def.b_type)], (l_bk*l_bn)*LIBXSMM_TYPESIZE(l_spmm_def.b_type));
      for (l_di = 0; l_di < l_bn; l_di++) {
        for (l_dj = 0; l_dj < l_bk; l_dj++) {
          if (l_spmm_def.b_type == LIBXSMM_DATATYPE_BF16) {
            libxsmm_bfloat16 *l_tmp_bf16 = (libxsmm_bfloat16*)tmp_block;
            libxsmm_bfloat16 *l_b_sp_bcsc_data_bf16 = (libxsmm_bfloat16*)l_b_sp_bcsc_data;
            l_b_sp_bcsc_data_bf16[l_i*(l_bk*l_bn) + (l_dj/l_b_vnni_factor) * (l_bn * l_b_vnni_factor) + l_di * l_b_vnni_factor + l_dj % l_b_vnni_factor] = l_tmp_bf16[l_di * l_bk + l_dj];
          } else if (l_spmm_def.b_type == LIBXSMM_DATATYPE_I8) {
            char *l_tmp_i8 = (char*)tmp_block;
            char *l_b_sp_bcsc_data_i8 = (char*)l_b_sp_bcsc_data;
            l_b_sp_bcsc_data_i8[l_i*(l_bk*l_bn) + (l_dj/l_b_vnni_factor) * (l_bn * l_b_vnni_factor) + l_di * l_b_vnni_factor + l_dj % l_b_vnni_factor] = l_tmp_i8[l_di * l_bk + l_dj];
          } else {
            /* Should not happen */
          }
        }
      }
    }
  }

  free(grid_point_array);
}

LIBXSMM_INLINE
double jit_matmul( const spmm_def*    i_spmm_def,
                   const void*        i_a,
                   const void*        i_b,
                   unsigned int*      i_colptr,
                   unsigned int*      i_rowidx,
                   void*              o_c,
                   void*              o_c_perf,
                   const int          i_reps) {
  /* define function pointer */
  libxsmm_xmmfunction l_test_jit = { NULL };
  libxsmm_xmmfunction cfg_tr = { NULL };
  libxsmm_xmmfunction rls_tr = { NULL };
  libxsmm_timer_tickint l_start;
  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  libxsmm_bitfield l_prefetch_flags = 0;
  libxsmm_gemm_param gemm_param;
  libxsmm_spgemm_config spgemm_config;

  double l_jittime, l_runtime;
  size_t l_t;
  double l_beta = i_spmm_def->beta;
  int l_cfg_flags = 0;
  int l_rls_flags = 0;
  unsigned long long N = i_spmm_def->n/i_spmm_def->bn;

  if (0 == i_spmm_def) {
    fprintf(stderr, "JIT: unsupported descriptor arguments or data type!\n");
    return EXIT_FAILURE;
  }

  /* set up the flags */
  if ( i_spmm_def->unsigned_a != 0 ) {
    l_flags |= LIBXSMM_GEMM_FLAG_A_UNSIGNED;
  }
  if ( i_spmm_def->unsigned_b != 0 ) {
    l_flags |= LIBXSMM_GEMM_FLAG_B_UNSIGNED;
  }

  l_flags |= (0 != i_spmm_def->trans_b ? LIBXSMM_GEMM_FLAG_TRANS_B : 0);
  l_flags |= (0 != i_spmm_def->vnni_a ? LIBXSMM_GEMM_FLAG_VNNI_A : 0);
  l_flags |= (0 != i_spmm_def->vnni_b ? LIBXSMM_GEMM_FLAG_VNNI_B : 0);

  l_flags |= ( l_beta == 0 ) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0;

  /* setting update GEMM struct */
  l_shape = libxsmm_create_gemm_shape( i_spmm_def->m_blocks,  0, i_spmm_def->k,
      i_spmm_def->k, 0, i_spmm_def->n,
      i_spmm_def->a_type, i_spmm_def->b_type, i_spmm_def->c_type, i_spmm_def->comp_type );

  /* setting prefetch flags */
  l_prefetch_flags = i_spmm_def->prefetch;

  spgemm_config.packed_width = i_spmm_def->m;
  spgemm_config.bk = i_spmm_def->bk;
  spgemm_config.bn = i_spmm_def->bn;

  l_start = libxsmm_timer_tick();
  if (i_spmm_def->tc_config) {
    l_cfg_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | l_flags;
    l_rls_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | l_flags;
    l_flags |= (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
    cfg_tr.gemm = libxsmm_create_packed_spgemm_bcsc(l_shape, l_cfg_flags, l_prefetch_flags, spgemm_config);
    rls_tr.gemm = libxsmm_create_packed_spgemm_bcsc(l_shape, l_rls_flags, l_prefetch_flags, spgemm_config);
  }
  l_test_jit.gemm = libxsmm_create_packed_spgemm_bcsc(l_shape, l_flags, l_prefetch_flags, spgemm_config);

  l_jittime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());
  if (l_test_jit.xmm == NULL) {
    printf("JIT failed, please run with LIBXSMM_VERBOSE=-1 and/or with debug mode LIBXSMM library!\n");
    exit(-1);
  }

  /* run external tileconfig */
  if (i_spmm_def->tc_config) {
    cfg_tr.gemm( NULL );
  }

  /* run correctness */
  gemm_param.a.primary = (void*)i_a;
  gemm_param.b.primary = (void*)i_b;
  gemm_param.b.secondary = (void*)i_colptr;
  gemm_param.b.tertiary  = (void*)i_rowidx;
  gemm_param.b.quaternary = (void*)&N;
  gemm_param.c.primary = (void*)o_c;
  l_test_jit.gemm( &gemm_param );

  /* run performance */
  l_start = libxsmm_timer_tick();
  gemm_param.a.primary = (void*)i_a;
  gemm_param.b.primary = (void*)i_b;
  gemm_param.b.secondary = (void*)i_colptr;
  gemm_param.b.tertiary  = (void*)i_rowidx;
  gemm_param.b.quaternary = (void*)&N;
  gemm_param.c.primary = (void*)o_c_perf;
  for (l_t = 0; l_t < (size_t)i_reps; l_t++) {
    l_test_jit.gemm( &gemm_param );
  }
  l_runtime = libxsmm_timer_duration(l_start, libxsmm_timer_tick());

  /* run external tilerelease */
  if (i_spmm_def->tc_config) {
    rls_tr.gemm( NULL );
  }

  printf("function pointer address: %llx\n", (unsigned long long)l_test_jit.xmm);
  printf("%fs for creating jit\n", l_jittime);

  return l_runtime;
}

LIBXSMM_INLINE
double get_random_posneg_p5_num(void) {
  double tmp = libxsmm_rng_f64()-0.5;

  if ( tmp < -0.4 ) {
    tmp = -0.4;
  } else if ( tmp < -0.3 ) {
    tmp = -0.3;
  } else if ( tmp < -0.2 ) {
    tmp = -0.2;
  } else if ( tmp < -0.1 ) {
    tmp = -0.1;
  } else if ( tmp < 0 ) {
    tmp = 0;
  } else if ( tmp < 0.1 ) {
    tmp = 0.1;
  } else if ( tmp < 0.2 ) {
    tmp = 0.2;
  } else if ( tmp < 0.3 ) {
    tmp = 0.3;
  } else if ( tmp < 0.4 ) {
    tmp = 0.4;
  } else if ( tmp < 0.5 ) {
    tmp = 0.5;
  } else {
    tmp = 0.5;
  }

  return tmp;
}

LIBXSMM_INLINE
double get_random_pos_p5_num(void) {
  double tmp = libxsmm_rng_f64();

  if ( tmp < 0.1 ) {
    tmp = 0.1;
  } else if ( tmp < 0.2 ) {
    tmp = 0.2;
  } else if ( tmp < 0.3 ) {
    tmp = 0.3;
  } else if ( tmp < 0.4 ) {
    tmp = 0.4;
  } else if ( tmp < 0.5 ) {
    tmp = 0.5;
  } else if ( tmp < 0.6 ) {
    tmp = 0.6;
  } else if ( tmp < 0.7 ) {
    tmp = 0.7;
  } else if ( tmp < 0.8 ) {
    tmp = 0.8;
  } else if ( tmp < 0.9 ) {
    tmp = 0.9;
  } else if ( tmp < 1.0 ) {
    tmp = 1.0;
  } else {
    tmp = 1.0;
  }

  return tmp;
}

LIBXSMM_INLINE
void init_random_matrix( const libxsmm_datatype dtype, void* data, const libxsmm_blasint br, const libxsmm_blasint ld, const libxsmm_blasint n, const libxsmm_blasint pos_val_only ) {
  double* d_data = (double*) data;
  float* f_data = (float*) data;
  libxsmm_bfloat16* bf16_data = (libxsmm_bfloat16*) data;
  libxsmm_bfloat8* bf8_data = (libxsmm_bfloat8*) data;
  libxsmm_hfloat8* hf8_data = (libxsmm_hfloat8*) data;
  int* i_data = (int*) data;
  short* s_data = (short*) data;
  char* sc_data = (char*) data;
  unsigned char* uc_data = (unsigned char*) data;
  libxsmm_blasint l_r, l_i, l_j;

  for (l_r = 0; l_r < br; l_r++) {
    for (l_i = 0; l_i < ld; l_i++) {
      for (l_j = 0; l_j < n; l_j++) {
        if ( dtype == LIBXSMM_DATATYPE_F64 ) {
          d_data[(l_r * ld * n) + (l_j * ld) + l_i] = (pos_val_only > 0 ) ? get_random_pos_p5_num() :  get_random_posneg_p5_num();
        } else if ( dtype == LIBXSMM_DATATYPE_F32 ) {
          f_data[(l_r * ld * n) + (l_j * ld) + l_i] = (pos_val_only > 0 ) ? (float)get_random_pos_p5_num() : (float)get_random_posneg_p5_num();
        } else if ( dtype == LIBXSMM_DATATYPE_BF16 ) {
          libxsmm_bfloat16_f32 tmp /*= { 0 }*/;
          tmp.f = (pos_val_only > 0 ) ? (float)get_random_pos_p5_num() : (float)get_random_posneg_p5_num();
          bf16_data[(l_r * ld * n) + (l_j * ld) + l_i] = tmp.i[1];
        } else if ( dtype == LIBXSMM_DATATYPE_BF8 ) {
          union libxsmm_bfloat8_f16 tmp/* = { 0 }*/;
          tmp.hf = libxsmm_convert_f32_to_f16( (float)get_random_posneg_p5_num() );
          bf8_data[(l_r * ld * n) + (l_j * ld) + l_i] = tmp.i[1];
        } else if ( dtype == LIBXSMM_DATATYPE_HF8 ) {
          float tmp_rnd = (float)get_random_posneg_p5_num();
          libxsmm_rne_convert_fp32_hf8( &tmp_rnd, &hf8_data[(l_r * ld * n) + (l_j * ld) + l_i], 1 );
        } else if ( dtype == LIBXSMM_DATATYPE_I32 ) {
          i_data[(l_r * ld * n) + (l_j * ld) + l_i] = (int)  (get_random_posneg_p5_num() * 40.0);
        } else if ( dtype == LIBXSMM_DATATYPE_I16 ) {
          s_data[(l_r * ld * n) + (l_j * ld) + l_i] = (short)(get_random_posneg_p5_num() * 40.0);
        } else if ( dtype == LIBXSMM_DATATYPE_I8 ) {
          if ( pos_val_only != 0 ) {
            uc_data[(l_r * ld * n) + (l_j * ld) + l_i] = (unsigned char) (get_random_pos_p5_num() * 20.0);
          } else {
            sc_data[(l_r * ld * n) + (l_j * ld) + l_i] = (char)(get_random_posneg_p5_num() * 40.0);
          }
        } else {
        }
      }
    }
  }
}

LIBXSMM_INLINE
void init_zero_matrix( const libxsmm_datatype dtype, void* data, const libxsmm_blasint br, const libxsmm_blasint ld, const libxsmm_blasint n ) {
  char* l_data = (char*) data;
  memset( l_data, 0x0, (size_t)br*ld*n*LIBXSMM_TYPESIZE(dtype) );
}

LIBXSMM_INLINE
void init_garbage_matrix( const libxsmm_datatype dtype, void* data, const libxsmm_blasint br, const libxsmm_blasint ld, const libxsmm_blasint n ) {
  char* l_data = (char*) data;
  memset( l_data, 0xdeadbeef, (size_t)br*ld*n*LIBXSMM_TYPESIZE(dtype) );
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
  } else {
    dtype = LIBXSMM_DATATYPE_UNSUPPORTED;
  }
  return dtype;
}

LIBXSMM_INLINE
double check_matrix( const libxsmm_datatype dtype, const void* data_gold, const void* data, const libxsmm_blasint ld, const libxsmm_blasint m, const libxsmm_blasint n ) {
  libxsmm_matdiff_info l_diff;
  double error = 0.0;

  libxsmm_matdiff_clear(&l_diff);
  if ( dtype == LIBXSMM_DATATYPE_F64 ) {
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F64, m, n, data_gold, data, &ld, &ld);
    error = libxsmm_matdiff_epsilon(&l_diff);
  } else if ( dtype == LIBXSMM_DATATYPE_F32 ) {
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, m, n, data_gold, data, &ld, &ld);
    error = libxsmm_matdiff_epsilon(&l_diff);
  } else if ( dtype == LIBXSMM_DATATYPE_BF16 ) {
    float* data_gold_f = (float*)malloc( sizeof(float) * ld * n );
    float* data_f      = (float*)malloc( sizeof(float) * ld * n );

    libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)data_gold, data_gold_f, ld*n );
    libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)data,      data_f,      ld*n );
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, m, n, data_gold_f, data_f, &ld, &ld);
    error = libxsmm_matdiff_epsilon(&l_diff);

    free( data_f );
    free( data_gold_f );
  } else if ( dtype == LIBXSMM_DATATYPE_BF8 ) {
    float* data_gold_f = malloc( ld * n * sizeof(float) );
    float* data_f      = malloc( ld * n * sizeof(float) );

    libxsmm_convert_bf8_f32( (libxsmm_bfloat8*)data_gold, data_gold_f, ld*n );
    libxsmm_convert_bf8_f32( (libxsmm_bfloat8*)data,      data_f,      ld*n );
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, m, n, data_gold_f, data_f, &ld, &ld);
    error = l_diff.normf_rel;

    free( data_f );
    free( data_gold_f );
  } else if ( dtype == LIBXSMM_DATATYPE_HF8 ) {
    float* data_gold_f = malloc( ld * n * sizeof(float) );
    float* data_f      = malloc( ld * n * sizeof(float) );

    libxsmm_convert_hf8_f32( (libxsmm_hfloat8*)data_gold, data_gold_f, ld*n );
    libxsmm_convert_hf8_f32( (libxsmm_hfloat8*)data,      data_f,      ld*n );
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, m, n, data_gold_f, data_f, &ld, &ld);
    error = l_diff.normf_rel;
    free( data_f );
    free( data_gold_f );
  } else if ( dtype == LIBXSMM_DATATYPE_I32 ) {
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_I32, m, n, data_gold, data, &ld, &ld);
    error = libxsmm_matdiff_epsilon(&l_diff);
  } else if ( dtype == LIBXSMM_DATATYPE_I8 ) {
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_I8, m, n, data_gold, data, &ld, &ld);
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
  printf("Check-norm    : %.24f\n", error);
  printf("\n");

  return error;
}

int main(int argc, char* argv []) {
  char* l_a_dt = NULL;
  char* l_b_dt = NULL;
  char* l_comp_dt = NULL;
  char* l_c_dt = NULL;
  libxsmm_datatype l_dtype_a, l_dtype_b, l_dtype_comp, l_dtype_c;
  libxsmm_blasint l_m = 0, l_n = 0, l_k = 0, l_m_blocks = 1;
  libxsmm_blasint l_bk = 1, l_bn = 1;
  double l_sparsity_frac = 0.5;
  int l_trans_a = 0;
  int l_trans_b = 0;
  int l_vnni_a = 1;
  int l_vnni_b = 0;
  int l_vnni_c = 0;
  double l_beta = 0;
  double l_runtime_libxsmm = 0;
  int l_file_input = 0;
  char* l_file_name = NULL;
  FILE *l_file_handle = NULL;
  int l_run_check = 0;
  double l_total_max_error = 0.0;
  int l_tc_config = 0;
  int l_reps;
  libxsmm_gemm_prefetch_type l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  spmm_def l_spmm_def;
  int l_n_threads = 1;
  unsigned int l_keep_going = 0;

# if defined(__APPLE__) && defined(__arm64__)
#  if 1
  pthread_set_qos_class_self_np( QOS_CLASS_USER_INTERACTIVE, 0 );
#  else
  pthread_set_qos_class_self_np( QOS_CLASS_BACKGROUND, 0 );
#  endif
# endif

  /* check argument count for a valid range */
  if ( argc == 19 || argc == 20 ) {
    /* datatypes */
    l_a_dt = argv[1];
    l_b_dt = argv[2];
    l_comp_dt = argv[3];
    l_c_dt = argv[4];
    l_dtype_a    = char_to_libxsmm_datatype( l_a_dt );
    l_dtype_b    = char_to_libxsmm_datatype( l_b_dt );
    l_dtype_comp = char_to_libxsmm_datatype( l_comp_dt );
    l_dtype_c    = char_to_libxsmm_datatype( l_c_dt );

    /* xgemm sizes */
    l_m = atoi(argv[5]);
    l_n = atoi(argv[6]);
    l_k = atoi(argv[7]);
    l_m_blocks = atoi(argv[8]);

    /* sparsity related params */
    l_sparsity_frac =  atof(argv[9]);
    l_bk =  atoi(argv[10]);
    l_bn =  atoi(argv[11]);

    l_beta = atof(argv[12]);

    l_trans_a = atoi(argv[13]);
    l_trans_b = atoi(argv[14]);
    l_vnni_a = atoi(argv[15]);
    l_vnni_b = atoi(argv[16]);
    l_vnni_c = atoi(argv[17]);

    l_reps = atoi(argv[18]);

    /* optional flags */
    if ( argc >= 20 ) {
      l_tc_config = atoi(argv[19]);
    } else {
      l_tc_config = 0;
    }

    l_file_input = 0;
    l_run_check = 1;
  } else if ( argc == 15 || argc == 16 ) {
    l_file_input = 1;
    /* datatypes */
    l_a_dt = argv[1];
    l_b_dt = argv[2];
    l_comp_dt = argv[3];
    l_c_dt = argv[4];
    l_dtype_a    = char_to_libxsmm_datatype( l_a_dt );
    l_dtype_b    = char_to_libxsmm_datatype( l_b_dt );
    l_dtype_comp = char_to_libxsmm_datatype( l_comp_dt );
    l_dtype_c    = char_to_libxsmm_datatype( l_c_dt );

    l_file_name = argv[5];
    l_sparsity_frac = atof(argv[6]);
    l_beta = atof(argv[7]);

    l_trans_a = atoi(argv[8]);
    l_trans_b = atoi(argv[9]);
    l_vnni_a = atoi(argv[10]);
    l_vnni_b = atoi(argv[11]);
    l_vnni_c = atoi(argv[12]);

    l_reps = atoi(argv[13]);
    l_run_check = atoi(argv[14]);

    /* optional flags */
    if ( argc >= 16 ) {
      l_tc_config = atoi(argv[15]);
    } else {
      l_tc_config = 0;
    }
  } else {
    return EXIT_FAILURE;
  }

  { const char* env_arch = getenv("LIBXSMM_TARGET");
    const int is_env_SPR = (
      env_arch == libxsmm_stristr(env_arch, "spr") ||
      env_arch == libxsmm_stristr(env_arch, "amx"));
    int arch_cpuid = libxsmm_cpuid(NULL);

    if ((!is_env_SPR && arch_cpuid < LIBXSMM_X86_AVX512_SPR)
      && (l_tc_config)) {
      printf("Warning: external tile configuration will be ingnored\n");
      l_tc_config = 0;
    }
  }

  /* check beta */
  if ( LIBXSMM_NEQ(l_beta, 0.0) && LIBXSMM_NEQ(l_beta, 1.0) ) {
    fprintf(stderr, "JIT: beta needs to be 0.0 or 1.0!\n");
    exit(EXIT_FAILURE);
  }

  /* check if we have entered supported datatpes */
  if ( !(
         ((l_dtype_a == LIBXSMM_DATATYPE_F32)  && (l_dtype_b == LIBXSMM_DATATYPE_F32)  && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_F32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_U8)   && (l_dtype_b == LIBXSMM_DATATYPE_I8)   && (l_dtype_comp == LIBXSMM_DATATYPE_I32) && (l_dtype_c == LIBXSMM_DATATYPE_I32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_I8)   && (l_dtype_b == LIBXSMM_DATATYPE_U8)   && (l_dtype_comp == LIBXSMM_DATATYPE_I32) && (l_dtype_c == LIBXSMM_DATATYPE_I32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_BF16) && (l_dtype_b == LIBXSMM_DATATYPE_BF16) && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_BF16))
        ) ) {
    fprintf(stderr, "Unsupported precion combination: a: %s, b: %s, comp: %s, c: %s!\n", l_a_dt, l_b_dt, l_comp_dt, l_c_dt);
    exit(EXIT_FAILURE);
  }

  l_spmm_def.unsigned_a = 0;
  l_spmm_def.unsigned_b = 0;
  l_spmm_def.unsigned_c = 0;
  /* handle unsigned cases */
  if ( l_dtype_a == LIBXSMM_DATATYPE_U8 ) {
    l_dtype_a = LIBXSMM_DATATYPE_I8;
    l_spmm_def.unsigned_a = 1;
  }
  if ( l_dtype_b == LIBXSMM_DATATYPE_U8 ) {
    l_dtype_b = LIBXSMM_DATATYPE_I8;
    l_spmm_def.unsigned_b = 1;
  }
  /* setting static GEMM parameters */
  l_spmm_def.a_type = l_dtype_a;
  l_spmm_def.b_type = l_dtype_b;
  l_spmm_def.comp_type = l_dtype_comp;
  l_spmm_def.c_type = l_dtype_c;
  l_spmm_def.beta = l_beta;
  l_spmm_def.vnni_a = l_vnni_a;
  l_spmm_def.vnni_b = l_vnni_b;
  l_spmm_def.vnni_c = l_vnni_c;
  l_spmm_def.trans_a = l_trans_a;
  l_spmm_def.trans_b = l_trans_b;
  l_spmm_def.prefetch = l_prefetch;
  l_spmm_def.tc_config = l_tc_config;

  if ( l_file_input != 0 ) {
    l_file_handle = fopen( l_file_name, "r" );
  } else {
    printf("------------------------------------------------\n");
    printf("RUNNING (%ix%i) X (%ix%i) = (%ix%i)\na:%s, b:%s, comp:%s, c:%s\n", l_m, l_k, l_k, l_n, l_m, l_n, l_a_dt, l_b_dt, l_comp_dt, l_c_dt);
    printf("------------------------------------------------\n");
  }

  do {
    double error = 0.0;

    if ( l_file_input != 0 ) {
      char l_line[512];
      if ( fgets( l_line, 512, l_file_handle) == NULL ) {
        l_keep_going = 0;
        break;
      } else {
        l_keep_going = 1;
      }
      if ( 6 != sscanf( l_line, "%i %i %i %i %i %i", &l_m, &l_n, &l_k, &l_bk, &l_bn, &l_m_blocks ) ) exit(EXIT_FAILURE);
      printf("Command line:\n%s %s %s %s %s %i %i %i %i %f %i %i %f %i %i %i %i %i %i %i\n\n",
          argv[0], l_a_dt, l_b_dt, l_comp_dt, l_c_dt,
          l_m, l_n, l_k, l_m_blocks, l_sparsity_frac, l_bk, l_bn, l_beta, l_trans_a, l_trans_b, l_vnni_a, l_vnni_b, l_vnni_c, l_reps, l_tc_config);

      if (l_keep_going == 0) break;
    }

    l_spmm_def.m = l_m;
    l_spmm_def.m_blocks = l_m_blocks;
    l_spmm_def.n = l_n;
    l_spmm_def.k = l_k;
    l_spmm_def.bk = l_bk;
    l_spmm_def.bn = l_bn;

    /* set rng seed */
    libxsmm_rng_set_seed( 555 );

    {
      char *l_a, *l_b, *l_c, *l_c_perf, *l_c_gold;
      char *l_a_spmm;
      /* BCSC DS for B */
      unsigned int *l_colptr;
      unsigned int *l_rowidx ;
      char *l_b_sp_bcsc_data;
      int l_a_vnni_factor =  libxsmm_cpuid_dot_pack_factor(l_spmm_def.a_type);

      l_a      = (char*)libxsmm_aligned_malloc((size_t)l_m * l_m_blocks * (size_t)l_k * LIBXSMM_TYPESIZE(l_spmm_def.a_type), 64);
      if (l_a_vnni_factor == 1) {
        l_a_spmm = l_a;
      } else {
        l_a_spmm = (char*)libxsmm_aligned_malloc((size_t)l_m * l_m_blocks * (size_t)l_k * LIBXSMM_TYPESIZE(l_spmm_def.a_type), 64);
      }
      l_b      = (char*)libxsmm_aligned_malloc((size_t)l_k * (size_t)l_n * LIBXSMM_TYPESIZE(l_spmm_def.b_type), 64);
      l_c      = (char*)libxsmm_aligned_malloc((size_t)l_m * l_m_blocks * (size_t)l_n * LIBXSMM_TYPESIZE(l_spmm_def.c_type), 64);
      l_c_perf = (char*)libxsmm_aligned_malloc((size_t)l_m * l_m_blocks * (size_t)l_n * LIBXSMM_TYPESIZE(l_spmm_def.c_type), 64);
      l_c_gold = (char*)libxsmm_aligned_malloc((size_t)l_m * l_m_blocks * (size_t)l_n * LIBXSMM_TYPESIZE(l_spmm_def.c_type), 64);

      init_random_matrix( l_spmm_def.a_type, l_a, 1, l_m * l_m_blocks, l_k, l_spmm_def.unsigned_a );
      init_random_matrix( l_spmm_def.b_type, l_b, 1, l_k, l_n, l_spmm_def.unsigned_b  );

      /* Create spmm inputs */
      create_spmm_inputs(&l_spmm_def, l_sparsity_frac, l_a, l_b, l_a_spmm, &l_b_sp_bcsc_data, &l_colptr, &l_rowidx);

      if ( l_beta == 0 ) {
        init_garbage_matrix( l_spmm_def.c_type, l_c,      1, l_m * l_m_blocks, l_n );
        init_garbage_matrix( l_spmm_def.c_type, l_c_perf, 1, l_m * l_m_blocks, l_n );
        init_garbage_matrix( l_spmm_def.c_type, l_c_gold, 1, l_m * l_m_blocks, l_n );
      } else {
        init_zero_matrix( l_spmm_def.c_type, l_c,      1, l_m * l_m_blocks, l_n );
        init_zero_matrix( l_spmm_def.c_type, l_c_perf, 1, l_m * l_m_blocks, l_n );
        init_zero_matrix( l_spmm_def.c_type, l_c_gold, 1, l_m * l_m_blocks, l_n );
      }

      /* Run reference */
      dense_gemm_ref(&l_spmm_def, l_a, l_b, l_c_gold);

      /* run LIBXSMM solution */
      l_runtime_libxsmm = jit_matmul( &l_spmm_def, l_a_spmm, l_b_sp_bcsc_data, l_colptr, l_rowidx, l_c, l_c_perf, l_reps);

      /* run compare */
      error = check_matrix( l_spmm_def.c_type, l_c_gold, l_c, l_m * l_m_blocks, l_m * l_m_blocks, l_n );

      libxsmm_free(l_a);
      if (l_a_vnni_factor != 1) {
        libxsmm_free(l_a_spmm);
      }
      libxsmm_free(l_b);
      libxsmm_free(l_c);
      libxsmm_free(l_c_perf);
      libxsmm_free(l_c_gold);
      libxsmm_free(l_colptr);
      libxsmm_free(l_rowidx);
      libxsmm_free(l_b_sp_bcsc_data);
    }

    if ( l_file_input == 0 ) {
      printf("%fs for libxsmm\n", l_runtime_libxsmm);
      printf("%f GFLOPS for libxsmm\n", ((double)((double)l_reps * (double)l_m * l_m_blocks * (double)l_n * (double)l_k) * (double)l_n_threads * 2.0) / (l_runtime_libxsmm * 1.0e9));
      printf("max. error: %f\n", error);
    } else {
      {
        const char *prefetch = NULL;
        switch (l_prefetch) {
          case LIBXSMM_GEMM_PREFETCH_NONE: prefetch = "nopf"; break;
          case LIBXSMM_GEMM_PREFETCH_SIGONLY: prefetch = "pfsigonly"; break;
          case LIBXSMM_GEMM_PREFETCH_BL2_VIA_C: prefetch = "BL2viaC"; break;
          case LIBXSMM_GEMM_PREFETCH_AL2_AHEAD: prefetch = "curAL2"; break;
          case LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD: prefetch = "curAL2_BL2viaC"; break;
          case LIBXSMM_GEMM_PREFETCH_AL2: prefetch = "AL2"; break;
          case LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C: prefetch = "AL2_BL2viaC"; break;
          default: prefetch = "unknown";
        }

        assert(NULL != prefetch);
        l_runtime_libxsmm /= (double)l_n_threads;
      }
      printf("%fs for LIBXSMM\n", l_runtime_libxsmm);
      printf("%f GFLOPS\n", ((double)((double)l_reps * (double)l_m * l_m_blocks * (double)l_n * (double)l_k * (double)l_n_threads) * 2.0) / (l_runtime_libxsmm * 1.0e9));
      printf("max. error: %f\n", error);
    }

    if ( (l_total_max_error < error) && (l_run_check == 1) ) {
      l_total_max_error = error;
    }
  } while ( l_keep_going );

  return 0;
}

