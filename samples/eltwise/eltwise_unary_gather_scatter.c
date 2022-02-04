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

#define COLS 0
#define ROWS 1
#define OFFS 2
#define DTYPE_32BIT 0
#define DTYPE_16BIT 1
#define IDX_32BIT 0
#define IDX_64BIT 1
#define GATHER 0
#define SCATTER 1
#define EXPANSION_FACTOR 4

void create_unique_random_array(unsigned long long *inout_array, int n) {
  if (n > 1)
  {
    int i;
    for (i = 0; i < n; i++) {
      inout_array[i] = i;
    }
    for (i = 0; i < n - 1; i++) {
      int j = i + rand() / (RAND_MAX / (n - i) + 1);
      unsigned long long t = inout_array[j];
      inout_array[j] = inout_array[i];
      inout_array[i] = t;
    }
  }
}

LIBXSMM_INLINE
void sfill_matrix ( float *matrix, unsigned int ld, unsigned int m, unsigned int n ) {
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

LIBXSMM_INLINE
void reference_gather_scatter(float *sinp, float *sout, libxsmm_bfloat16 *binp, libxsmm_bfloat16 *bout,
    unsigned long long *ind_array_64bit, unsigned int * ind_array_32bit,
    libxsmm_blasint inp_m, libxsmm_blasint inp_n, libxsmm_blasint inp_ld,
    libxsmm_blasint out_m, libxsmm_blasint out_n, libxsmm_blasint out_ld,
    unsigned int use_gather_or_scatter, unsigned int use_rows_cols_offs, unsigned int use_16bit_dtype, unsigned int use_64bit_index) {
  libxsmm_blasint i, j, ind, ind2;

  if (use_16bit_dtype == DTYPE_32BIT) {
    if (use_64bit_index == IDX_64BIT) {
      if (use_gather_or_scatter == GATHER) {
        if (use_rows_cols_offs == COLS) {
          for (ind = 0; ind < out_n; ind++) {
            j = ind_array_64bit[ind];
            for (i = 0; i < out_m; i++) {
              sout[i + ind * out_ld] = sinp[i + j * inp_ld];
            }
          }
        } else if (use_rows_cols_offs == ROWS) {
          for (ind = 0; ind < out_m; ind++) {
            i = ind_array_64bit[ind];
            for (j = 0; j < out_n; j++) {
              sout[ind + j * out_ld] = sinp[i + j * inp_ld];
            }
          }
        } else {
          for (ind2 = 0; ind2 < out_n; ind2++) {
            for (ind = 0; ind < out_m; ind++) {
              i = ind_array_64bit[ind + ind2 * out_m];
              sout[ind + ind2 * out_ld] = sinp[i];
            }
          }
        }
      } else {
        if (use_rows_cols_offs == COLS) {
          for (ind = 0; ind < inp_n; ind++) {
            j = ind_array_64bit[ind];
            for (i = 0; i < out_m; i++) {
              sout[i + j * out_ld] = sinp[i + ind * inp_ld];
            }
          }
        } else if (use_rows_cols_offs == ROWS) {
          for (ind = 0; ind < inp_m; ind++) {
            i = ind_array_64bit[ind];
            for (j = 0; j < inp_n; j++) {
              sout[i + j * out_ld] = sinp[ind + j * inp_ld];
            }
          }
        } else  {
          for (ind2 = 0; ind2 < inp_n; ind2++) {
            for (ind = 0; ind < inp_m; ind++) {
              i = ind_array_64bit[ind + ind2 * inp_m];
              sout[i] = sinp[ind + ind2 * inp_ld ];
            }
          }
        }
      }
    } else {
      if (use_gather_or_scatter == GATHER) {
        if (use_rows_cols_offs == COLS) {
          for (ind = 0; ind < out_n; ind++) {
            j = ind_array_32bit[ind];
            for (i = 0; i < out_m; i++) {
              sout[i + ind * out_ld] = sinp[i + j * inp_ld];
            }
          }
        } else if (use_rows_cols_offs == ROWS) {
          for (ind = 0; ind < out_m; ind++) {
            i = ind_array_32bit[ind];
            for (j = 0; j < out_n; j++) {
              sout[ind + j * out_ld] = sinp[i + j * inp_ld];
            }
          }
        } else  {
          for (ind2 = 0; ind2 < out_n; ind2++) {
            for (ind = 0; ind < out_m; ind++) {
              i = ind_array_32bit[ind + ind2 * out_m];
              sout[ind + ind2 * out_ld] = sinp[i];
            }
          }
        }
      } else {
        if (use_rows_cols_offs == COLS) {
          for (ind = 0; ind < inp_n; ind++) {
            j = ind_array_32bit[ind];
            for (i = 0; i < out_m; i++) {
              sout[i + j * out_ld] = sinp[i + ind * inp_ld];
            }
          }
        } else if (use_rows_cols_offs == ROWS) {
          for (ind = 0; ind < inp_m; ind++) {
            i = ind_array_32bit[ind];
            for (j = 0; j < inp_n; j++) {
              sout[i + j * out_ld] = sinp[ind + j * inp_ld];
            }
          }
        } else  {
          for (ind2 = 0; ind2 < inp_n; ind2++) {
            for (ind = 0; ind < inp_m; ind++) {
              i = ind_array_32bit[ind + ind2 * inp_m];
              sout[i] = sinp[ind + ind2 * inp_ld ];
            }
          }
        }
      }
    }
  } else {
    if (use_64bit_index == IDX_64BIT) {
      if (use_gather_or_scatter == GATHER) {
        if (use_rows_cols_offs == COLS) {
          for (ind = 0; ind < out_n; ind++) {
            j = ind_array_64bit[ind];
            for (i = 0; i < out_m; i++) {
              bout[i + ind * out_ld] = binp[i + j * inp_ld];
            }
          }
        } else if (use_rows_cols_offs == ROWS) {
          for (ind = 0; ind < out_m; ind++) {
            i = ind_array_64bit[ind];
            for (j = 0; j < out_n; j++) {
              bout[ind + j * out_ld] = binp[i + j * inp_ld];
            }
          }
        } else {
          for (ind2 = 0; ind2 < out_n; ind2++) {
            for (ind = 0; ind < out_m; ind++) {
              i = ind_array_64bit[ind + ind2 * out_m];
              bout[ind + ind2 * out_ld] = binp[i];
            }
          }
        }
      } else {
        if (use_rows_cols_offs == COLS) {
          for (ind = 0; ind < inp_n; ind++) {
            j = ind_array_64bit[ind];
            for (i = 0; i < out_m; i++) {
              bout[i + j * out_ld] = binp[i + ind * inp_ld];
            }
          }
        } else if (use_rows_cols_offs == ROWS) {
          for (ind = 0; ind < inp_m; ind++) {
            i = ind_array_64bit[ind];
            for (j = 0; j < inp_n; j++) {
              bout[i + j * out_ld] = binp[ind + j * inp_ld];
            }
          }
        } else  {
          for (ind2 = 0; ind2 < inp_n; ind2++) {
            for (ind = 0; ind < inp_m; ind++) {
              i = ind_array_64bit[ind + ind2 * inp_m];
              bout[i] = binp[ind + ind2 * inp_ld ];
            }
          }
        }
      }
    } else {
      if (use_gather_or_scatter == GATHER) {
        if (use_rows_cols_offs == COLS) {
          for (ind = 0; ind < out_n; ind++) {
            j = ind_array_32bit[ind];
            for (i = 0; i < out_m; i++) {
              bout[i + ind * out_ld] = binp[i + j * inp_ld];
            }
          }
        } else if (use_rows_cols_offs == ROWS) {
          for (ind = 0; ind < out_m; ind++) {
            i = ind_array_32bit[ind];
            for (j = 0; j < out_n; j++) {
              bout[ind + j * out_ld] = binp[i + j * inp_ld];
            }
          }
        } else  {
          for (ind2 = 0; ind2 < out_n; ind2++) {
            for (ind = 0; ind < out_m; ind++) {
              i = ind_array_32bit[ind + ind2 * out_m];
              bout[ind + ind2 * out_ld] = binp[i];
            }
          }
        }
      } else {
        if (use_rows_cols_offs == COLS) {
          for (ind = 0; ind < inp_n; ind++) {
            j = ind_array_32bit[ind];
            for (i = 0; i < out_m; i++) {
              bout[i + j * out_ld] = binp[i + ind * inp_ld];
            }
          }
        } else if (use_rows_cols_offs == ROWS) {
          for (ind = 0; ind < inp_m; ind++) {
            i = ind_array_32bit[ind];
            for (j = 0; j < inp_n; j++) {
              bout[i + j * out_ld] = binp[ind + j * inp_ld];
            }
          }
        } else  {
          for (ind2 = 0; ind2 < inp_n; ind2++) {
            for (ind = 0; ind < inp_m; ind++) {
              i = ind_array_32bit[ind + ind2 * inp_m];
              bout[i] = binp[ind + ind2 * inp_ld ];
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE
void setup_tpp_kernel_and_param_struct( libxsmm_meltwfunction_unary *kernel, libxsmm_meltw_unary_param *unary_param,
    float *sinp, float *sout, libxsmm_bfloat16 *binp, libxsmm_bfloat16 *bout,
    unsigned long long *ind_array_64bit, unsigned int * ind_array_32bit,
    libxsmm_blasint inp_m, libxsmm_blasint inp_n, libxsmm_blasint inp_ld,
    libxsmm_blasint out_m, libxsmm_blasint out_n, libxsmm_blasint out_ld,
    unsigned int use_gather_or_scatter, unsigned int use_rows_cols_offs, unsigned int use_16bit_dtype, unsigned int use_64bit_index) {

  libxsmm_blasint ld_in_kernel = inp_ld, ld_out_kernel = out_ld;
  libxsmm_meltw_unary_flags unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltw_unary_type  unary_type = LIBXSMM_MELTW_TYPE_UNARY_NONE;
  libxsmm_meltwfunction_unary l_kernel = NULL;
  libxsmm_meltw_unary_param l_unary_param;
  libxsmm_dnn_datatype dtype = (use_16bit_dtype == DTYPE_32BIT) ? LIBXSMM_DATATYPE_F32 : LIBXSMM_DATATYPE_BF16;
  libxsmm_blasint m_kernel = 0, n_kernel = 0;

  if (use_gather_or_scatter == GATHER) {
    m_kernel = out_m;
    n_kernel = out_n;
  } else {
    m_kernel = inp_m;
    n_kernel = inp_n;
  }

  if (use_rows_cols_offs == COLS) {
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_GS_COLS;
  } else if (use_rows_cols_offs == ROWS) {
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_GS_ROWS;
  } else {
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_GS_OFFS;
  }
  unary_flags = (use_64bit_index == IDX_64BIT) ? (LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES | unary_flags) : (LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES | unary_flags) ;
  unary_type  = (use_gather_or_scatter == GATHER) ? LIBXSMM_MELTW_TYPE_UNARY_GATHER :LIBXSMM_MELTW_TYPE_UNARY_SCATTER;
  l_kernel    = libxsmm_dispatch_meltw_unary(m_kernel, n_kernel, &ld_in_kernel, &ld_out_kernel, dtype, dtype, dtype, unary_flags, unary_type);
  l_unary_param.in.primary   = (use_16bit_dtype == DTYPE_32BIT) ? (void*)sinp : (void*)binp;
  if (use_gather_or_scatter == GATHER) {
    l_unary_param.in.secondary = (use_64bit_index == IDX_64BIT) ? (void*)ind_array_64bit : (void*)ind_array_32bit;
  } else {
    l_unary_param.out.secondary = (use_64bit_index == IDX_64BIT) ? (void*)ind_array_64bit : (void*)ind_array_32bit;
  }
  l_unary_param.out.primary  = (use_16bit_dtype == DTYPE_32BIT) ? (void*)sout : (void*)bout;

  *kernel      = l_kernel;
  *unary_param = l_unary_param;
}

LIBXSMM_INLINE
int compare_results(float *sout, float *sout_ref, libxsmm_bfloat16 *bout, libxsmm_bfloat16 *bout_ref,
    libxsmm_blasint inp_m, libxsmm_blasint inp_n, libxsmm_blasint inp_ld,
    libxsmm_blasint out_m, libxsmm_blasint out_n, libxsmm_blasint out_ld,
    unsigned int use_gather_or_scatter, unsigned int use_rows_cols_offs, unsigned int use_16bit_dtype, unsigned int use_64bit_index) {
  int ret = EXIT_SUCCESS;
  libxsmm_blasint result_size_check;
  libxsmm_matdiff_info norms_elts, diff;
  libxsmm_matdiff_clear(&norms_elts);
  libxsmm_matdiff_clear(&diff);

  if (use_16bit_dtype == DTYPE_32BIT) {
    if (use_64bit_index == IDX_64BIT) {
      if (use_gather_or_scatter == GATHER) {
        if (use_rows_cols_offs == COLS) {
          printf("# Correctness FP32 GATHER COLS (64-Bit indices)  #\n");
          result_size_check = out_ld * out_n;
        } else if (use_rows_cols_offs == ROWS) {
          printf("# Correctness FP32 GATHER ROWS (64-Bit indices)  #\n");
          result_size_check = out_ld * out_n;
        } else  {
          printf("# Correctness FP32 GATHER OFFS (64-Bit indices)  #\n");
          result_size_check = out_ld * out_n;
        }
      } else {
        if (use_rows_cols_offs == COLS) {
          printf("# Correctness FP32 SCATTER COLS (64-Bit indices)  #\n");
          result_size_check = out_ld * out_n;
        } else if (use_rows_cols_offs == ROWS) {
          printf("# Correctness FP32 SCATTER ROWS (64-Bit indices)  #\n");
          result_size_check = out_ld * out_n;
        } else  {
          printf("# Correctness FP32 SCATTER OFFS (64-Bit indices)  #\n");
          result_size_check = out_ld * out_n;
        }
      }
    } else {
      if (use_gather_or_scatter == GATHER) {
        if (use_rows_cols_offs == COLS) {
          printf("# Correctness FP32 GATHER COLS (32-Bit indices)  #\n");
          result_size_check = out_ld * out_n;
        } else if (use_rows_cols_offs == ROWS) {
          printf("# Correctness FP32 GATHER ROWS (32-Bit indices)  #\n");
          result_size_check = out_ld * out_n;
        } else  {
          printf("# Correctness FP32 GATHER OFFS (32-Bit indices)  #\n");
          result_size_check = out_ld * out_n;
        }
      } else {
        if (use_rows_cols_offs == COLS) {
          printf("# Correctness FP32 SCATTER COLS (32-Bit indices)  #\n");
          result_size_check = out_ld * out_n;
        } else if (use_rows_cols_offs == ROWS) {
          printf("# Correctness FP32 SCATTER ROWS (32-Bit indices)  #\n");
          result_size_check = out_ld * out_n;
        } else  {
          printf("# Correctness FP32 SCATTER OFFS (32-Bit indices)  #\n");
          result_size_check = out_ld * out_n;
        }
      }
    }
  } else {
    if (use_64bit_index == IDX_64BIT) {
      if (use_gather_or_scatter == GATHER) {
        if (use_rows_cols_offs == COLS) {
          printf("# Correctness BF16 GATHER COLS (64-Bit indices)  #\n");
          result_size_check = out_ld * out_n;
        } else if (use_rows_cols_offs == ROWS) {
          printf("# Correctness BF16 GATHER ROWS (64-Bit indices)  #\n");
          result_size_check = out_ld * out_n;
        } else  {
          printf("# Correctness BF16 GATHER OFFS (64-Bit indices)  #\n");
          result_size_check = out_ld * out_n;
        }
      } else {
        if (use_rows_cols_offs == COLS) {
          printf("# Correctness BF16 SCATTER COLS (64-Bit indices)  #\n");
          result_size_check = out_ld * out_n;
        } else if (use_rows_cols_offs == ROWS) {
          printf("# Correctness BF16 SCATTER ROWS (64-Bit indices)  #\n");
          result_size_check = out_ld * out_n;
        } else  {
          printf("# Correctness BF16 SCATTER OFFS (64-Bit indices)  #\n");
          result_size_check = out_ld * out_n;
        }
      }
    } else {
      if (use_gather_or_scatter == GATHER) {
        if (use_rows_cols_offs == COLS) {
          printf("# Correctness BF16 GATHER COLS (32-Bit indices)  #\n");
          result_size_check = out_ld * out_n;
        } else if (use_rows_cols_offs == ROWS) {
          printf("# Correctness BF16 GATHER ROWS (32-Bit indices)  #\n");
          result_size_check = out_ld * out_n;
        } else  {
          printf("# Correctness BF16 GATHER OFFS (32-Bit indices)  #\n");
          result_size_check = out_ld * out_n;
        }
      } else {
        if (use_rows_cols_offs == COLS) {
          printf("# Correctness BF16 SCATTER COLS (32-Bit indices)  #\n");
          result_size_check = out_ld * out_n;
        } else if (use_rows_cols_offs == ROWS) {
          printf("# Correctness BF16 SCATTER ROWS (32-Bit indices)  #\n");
          result_size_check = out_ld * out_n;
        } else  {
          printf("# Correctness BF16 SCATTER OFFS (32-Bit indices)  #\n");
          result_size_check = out_ld * out_n;
        }
      }
    }
    libxsmm_convert_bf16_f32( bout_ref, sout_ref, result_size_check );
    libxsmm_convert_bf16_f32( bout, sout, result_size_check );
  }

  libxsmm_matdiff(&norms_elts, LIBXSMM_DATATYPE_F32, result_size_check, 1, sout_ref, sout, 0, 0);
  printf("L1 reference  : %.25g\n", norms_elts.l1_ref);
  printf("L1 test       : %.25g\n", norms_elts.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_elts.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_elts.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_elts.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_elts.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_elts.normf_rel);

  if ( norms_elts.normf_rel > 0 ) {
    ret = EXIT_FAILURE;
  }

  return ret;
}

int main(int argc, char* argv[])
{
  unsigned int m = 64, n = 64, M = 0, N = 0, inp_m = 64, inp_n = 64, out_m = 64, out_n = 64, i = 0, j = 0;
  libxsmm_blasint ld_in = 64, ld_out = 64, LDI, LDO, inp_ld, out_ld;
  unsigned int use_gather_or_scatter = 0, use_rows_cols_offs = 0, use_16bit_dtype = 0, use_64bit_index = 0, iters = 100;

  float  *sinp = NULL, *sout = NULL, *sout_ref = NULL;
  libxsmm_bfloat16 *binp = NULL, *bout = NULL, *bout_ref = NULL;

  unsigned long long *ind_array_64bit;
  unsigned int       *ind_array_32bit;
  unsigned long long *unique_random_array;

  libxsmm_meltwfunction_unary kernel = NULL;
  libxsmm_meltw_unary_param unary_param;
  int ret = EXIT_FAILURE;

  unsigned long long l_start, l_end;
  double l_total = 0.0, l_total2 = 0.0;

  libxsmm_init();

  if ( argc > 1 ) m = atoi(argv[1]);
  if ( argc > 2 ) n = atoi(argv[2]);
  if ( argc > 3 ) ld_in = atoi(argv[3]);
  if ( argc > 4 ) ld_out = atoi(argv[4]);
  if ( argc > 5 ) use_gather_or_scatter = atoi(argv[5]);
  if ( argc > 6 ) use_rows_cols_offs = atoi(argv[6]);
  if ( argc > 7 ) use_16bit_dtype = atoi(argv[7]);
  if ( argc > 8 ) use_64bit_index = atoi(argv[8]);
  if ( argc > 9 ) iters = atoi(argv[9]);


  m = LIBXSMM_MAX(m,1);
  n = LIBXSMM_MAX(n,1);
  ld_in = LIBXSMM_MAX(ld_in,(libxsmm_blasint)m);
  ld_out = LIBXSMM_MAX(ld_out,(libxsmm_blasint)m);

  M = m * EXPANSION_FACTOR;
  N = n * EXPANSION_FACTOR;
  LDI = ld_in * EXPANSION_FACTOR;
  LDO = ld_out * EXPANSION_FACTOR;

  unique_random_array = (unsigned long long*) malloc(M * N * sizeof(unsigned long long));

  /* Allocate arrays  */
  /* Gather cols : input has larger N (output mxn)  */
  /* Scatter cols: output has larger N (input mxn)  */
  /* Gather rows : input has larger M (output mxn)  */
  /* Scatter rows: output has larger M (input mxn)  */
  /* Gather offs : input is larger MxN (output mxn)  */
  /* Scatter offs: output is larger MxN (input mxn)  */

  if (use_gather_or_scatter == GATHER) {
    if (use_rows_cols_offs == COLS) {
      inp_m = m;
      inp_n = N;
      out_m = m;
      out_n = n;
      inp_ld = ld_in;
      out_ld = ld_out;
      ind_array_64bit = (unsigned long long*)  libxsmm_aligned_malloc( out_n * sizeof(unsigned long long), 2097152);
      ind_array_32bit = (unsigned int*)  libxsmm_aligned_malloc( out_n * sizeof(unsigned int), 2097152);
      create_unique_random_array(unique_random_array, inp_n);
      for (i = 0; i < out_n; i++) {
        ind_array_64bit[i] = (unsigned long long) unique_random_array[i];
        ind_array_32bit[i] = (unsigned int) ind_array_64bit[i];
      }
    } else if (use_rows_cols_offs == ROWS) {
      inp_m = M;
      inp_n = n;
      out_m = m;
      out_n = n;
      inp_ld = LDI;
      out_ld = ld_out;
      ind_array_64bit = (unsigned long long*)  libxsmm_aligned_malloc( out_m * sizeof(unsigned long long), 2097152);
      ind_array_32bit = (unsigned int*)  libxsmm_aligned_malloc( out_m * sizeof(unsigned int), 2097152);
      create_unique_random_array(unique_random_array, inp_m);
      for (i = 0; i < out_m; i++) {
        ind_array_64bit[i] = (unsigned long long) unique_random_array[i];
        ind_array_32bit[i] = (unsigned int) ind_array_64bit[i];
      }
    } else if (use_rows_cols_offs == OFFS) {
      inp_m = M;
      inp_n = N;
      out_m = m;
      out_n = n;
      inp_ld = LDI;
      out_ld = ld_out;
      ind_array_64bit = (unsigned long long*)  libxsmm_aligned_malloc( out_m * out_n * sizeof(unsigned long long), 2097152);
      ind_array_32bit = (unsigned int*)  libxsmm_aligned_malloc(  out_m * out_n * sizeof(unsigned int), 2097152);
      create_unique_random_array(unique_random_array, inp_m * inp_n);
      for (j = 0; j < out_n; j++) {
        for (i = 0; i < out_m; i++) {
          unsigned long long tmp_ind = (unsigned long long) unique_random_array[i + j * out_m];
          ind_array_64bit[i + j * out_m] = (unsigned long long) ( (tmp_ind/inp_m) * LDI + tmp_ind % inp_m );
          ind_array_32bit[i + j * out_m] = (unsigned int) ind_array_64bit[i];
        }
      }
    } else {
      fprintf(stdout, "Unsupported OP!\n");
      exit(EXIT_FAILURE);
    }
  } else if (use_gather_or_scatter == SCATTER) {
    if (use_rows_cols_offs == COLS) {
      inp_m = m;
      inp_n = n;
      out_m = m;
      out_n = N;
      inp_ld = ld_in;
      out_ld = ld_out;
      ind_array_64bit = (unsigned long long*)  libxsmm_aligned_malloc( inp_n * sizeof(unsigned long long), 2097152);
      ind_array_32bit = (unsigned int*)  libxsmm_aligned_malloc( inp_n * sizeof(unsigned int), 2097152);
      create_unique_random_array(unique_random_array, out_n);
      for (i = 0; i < inp_n; i++) {
        ind_array_64bit[i] = (unsigned long long) unique_random_array[i];
        ind_array_32bit[i] = (unsigned int) ind_array_64bit[i];
      }
    } else if (use_rows_cols_offs == ROWS) {
      inp_m = m;
      inp_n = n;
      out_m = M;
      out_n = n;
      inp_ld = ld_in;
      out_ld = LDO;
      ind_array_64bit = (unsigned long long*)  libxsmm_aligned_malloc( inp_m * sizeof(unsigned long long), 2097152);
      ind_array_32bit = (unsigned int*)  libxsmm_aligned_malloc( inp_m * sizeof(unsigned int), 2097152);
      create_unique_random_array(unique_random_array, out_m);
      for (i = 0; i < inp_m; i++) {
        ind_array_64bit[i] = (unsigned long long) unique_random_array[i];
        ind_array_32bit[i] = (unsigned int) ind_array_64bit[i];
      }
    } else if (use_rows_cols_offs == OFFS) {
      inp_m = m;
      inp_n = n;
      out_m = M;
      out_n = N;
      inp_ld = ld_in;
      out_ld = LDO;
      ind_array_64bit = (unsigned long long*)  libxsmm_aligned_malloc( inp_m * inp_n * sizeof(unsigned long long), 2097152);
      ind_array_32bit = (unsigned int*)  libxsmm_aligned_malloc( inp_m * inp_n * sizeof(unsigned int), 2097152);
      create_unique_random_array(unique_random_array, out_m * out_n);
      for (j = 0; j < inp_n; j++) {
        for (i = 0; i < inp_m; i++) {
          unsigned long long tmp_ind = (unsigned long long) unique_random_array[i + j * inp_m];
          ind_array_64bit[i + j * inp_m] = (unsigned long long) ( (tmp_ind/out_m) * LDO + tmp_ind % out_m );
          ind_array_32bit[i + j * inp_m] = (unsigned int) ind_array_64bit[i];
        }
      }
    } else {
      fprintf(stdout, "Unsupported OP!\n");
      exit(EXIT_FAILURE);
    }
  } else {
    fprintf(stdout, "Unsupported OP!\n");
    exit(EXIT_FAILURE);
  }

  sinp      = (float*) libxsmm_aligned_malloc( inp_ld * inp_n * sizeof(float), 2097152);
  sout      = (float*) libxsmm_aligned_malloc( out_ld * out_n * sizeof(float), 2097152);
  sout_ref  = (float*) libxsmm_aligned_malloc( out_ld * out_n * sizeof(float), 2097152);
  sfill_matrix ( sinp, inp_ld, inp_m, inp_n );
  sfill_matrix ( sout, out_ld, out_m, out_n );
  memcpy( sout_ref, sout, out_ld * out_n * sizeof(float)  );

  if ( use_16bit_dtype == DTYPE_16BIT) {
    binp      = (libxsmm_bfloat16*) libxsmm_aligned_malloc( inp_ld * inp_n * sizeof(libxsmm_bfloat16), 2097152);
    bout      = (libxsmm_bfloat16*) libxsmm_aligned_malloc( out_ld * out_n * sizeof(libxsmm_bfloat16), 2097152);
    bout_ref  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( out_ld * out_n * sizeof(libxsmm_bfloat16), 2097152);
    libxsmm_rne_convert_fp32_bf16( sinp, binp, inp_ld * inp_n);
    libxsmm_rne_convert_fp32_bf16( sout, bout, out_ld * out_n);
    libxsmm_rne_convert_fp32_bf16( sout_ref, bout_ref, out_ld * out_n);
  }

  /* Run reference code */
  reference_gather_scatter(sinp, sout_ref, binp, bout_ref, ind_array_64bit, ind_array_32bit, inp_m, inp_n, inp_ld, out_m, out_n, out_ld,
    use_gather_or_scatter, use_rows_cols_offs, use_16bit_dtype, use_64bit_index);

  /* Setup TPP and param struct  */
  setup_tpp_kernel_and_param_struct( &kernel, &unary_param, sinp, sout, binp, bout, ind_array_64bit, ind_array_32bit, inp_m, inp_n, inp_ld, out_m, out_n, out_ld,
    use_gather_or_scatter, use_rows_cols_offs, use_16bit_dtype, use_64bit_index);

  /* Call TPP kernel  */
  kernel( &unary_param );

  /* compare results*/
  ret = compare_results(sout, sout_ref, bout, bout_ref, inp_m, inp_n, inp_ld, out_m, out_n, out_ld,
    use_gather_or_scatter, use_rows_cols_offs, use_16bit_dtype, use_64bit_index);

  reference_gather_scatter(sinp, sout_ref, binp, bout_ref, ind_array_64bit, ind_array_32bit, inp_m, inp_n, inp_ld, out_m, out_n, out_ld,
    use_gather_or_scatter, use_rows_cols_offs, use_16bit_dtype, use_64bit_index);
  l_start = libxsmm_timer_tick();
  /* Calculate reference results...  */
  for (j = 0; j < iters; j++) {
    reference_gather_scatter(sinp, sout_ref, binp, bout_ref, ind_array_64bit, ind_array_32bit, inp_m, inp_n, inp_ld, out_m, out_n, out_ld,
      use_gather_or_scatter, use_rows_cols_offs, use_16bit_dtype, use_64bit_index);
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("Reference time = %.5g\n", ((double)(l_total)));

  kernel( &unary_param );
  l_start = libxsmm_timer_tick();
  for (j = 0; j < iters; j++) {
    kernel( &unary_param );
  }
  l_end = libxsmm_timer_tick();
  l_total2 = libxsmm_timer_duration(l_start, l_end);
  printf("Optimized time = %.5g\n", ((double)(l_total2)));
  printf("Speedup is = %.5g\n", ((double)(l_total/l_total2)));

  libxsmm_free(sinp);
  libxsmm_free(sout);
  libxsmm_free(sout_ref);
  if (use_16bit_dtype == DTYPE_16BIT) {
    libxsmm_free(binp);
    libxsmm_free(bout);
    libxsmm_free(bout_ref);
  }
  libxsmm_free(ind_array_64bit);
  libxsmm_free(ind_array_32bit);
  free( unique_random_array );

  if (ret == EXIT_FAILURE) {
    fprintf(stderr, "ERROR at test: M = %d, N = %d, ldi = %d, ldo = %d, gs = %d, rowcolsoffs = %d, 32b/16b = %d, idxtype = %d\n", m, n, ld_in, ld_out, use_gather_or_scatter, use_rows_cols_offs, use_16bit_dtype, use_64bit_index );
  }

  return ret;
}

