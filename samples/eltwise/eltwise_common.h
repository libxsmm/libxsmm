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
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

void init_random_matrix( libxsmm_datatype dtype, void* data, libxsmm_blasint br, libxsmm_blasint ld, libxsmm_blasint n ) {
  double* d_data = (double*) data;
  float* f_data = (float*) data;
  libxsmm_bfloat16* bf_data = (libxsmm_bfloat16*) data;
  int* i_data = (int*) data;
  short* s_data = (short*) data;
  char* c_data = (char*) data;
  unsigned int l_r, l_i, l_j;

  for (l_r = 0; l_r < br; l_r++) {
    for (l_i = 0; l_i < ld; l_i++) {
      for (l_j = 0; l_j < n; l_j++) {
        if ( dtype == LIBXSMM_DATATYPE_F64 ) {
          d_data[(l_r * ld * n) + (l_j * ld) + l_i] = libxsmm_rng_f64();
        } else if ( dtype == LIBXSMM_DATATYPE_F32 ) {
          f_data[(l_r * ld * n) + (l_j * ld) + l_i] = (float)libxsmm_rng_f64();
        } else if ( dtype == LIBXSMM_DATATYPE_BF16 ) {
          union libxsmm_bfloat16_hp tmp;
          tmp.f = (float)libxsmm_rng_f64();
          bf_data[(l_r * ld * n) + (l_j * ld) + l_i] = tmp.i[1];
        } else if ( dtype == LIBXSMM_DATATYPE_I32 ) {
          i_data[(l_r * ld * n) + (l_j * ld) + l_i] = (int)  (libxsmm_rng_f64() * 20.0);
        } else if ( dtype == LIBXSMM_DATATYPE_I16 ) {
          s_data[(l_r * ld * n) + (l_j * ld) + l_i] = (short)(libxsmm_rng_f64() * 20.0);
        } else if ( dtype == LIBXSMM_DATATYPE_I8 ) {
          c_data[(l_r * ld * n) + (l_j * ld) + l_i] = (char) (libxsmm_rng_f64() * 20.0);
        } else {
        }
      }
    }
  }
}

void init_zero_matrix( libxsmm_datatype dtype, void* data, libxsmm_blasint br, libxsmm_blasint ld, libxsmm_blasint n ) {
  char* l_data = (char*) data;
  memset( l_data, 0x0, br*ld*n*LIBXSMM_TYPESIZE(dtype) );
}

void init_garbage_matrix( libxsmm_datatype dtype, void* data, libxsmm_blasint br, libxsmm_blasint ld, libxsmm_blasint n ) {
  char* l_data = (char*) data;
  memset( l_data, 0xdeadbeef, br*ld*n*LIBXSMM_TYPESIZE(dtype) );
}

double check_matrix( libxsmm_datatype dtype, void* data_gold, void* data, libxsmm_blasint ld, libxsmm_blasint m, libxsmm_blasint n ) {
  libxsmm_matdiff_info l_diff;
  double max_error = 0.0;

  libxsmm_matdiff_clear(&l_diff);

  if ( dtype == LIBXSMM_DATATYPE_F64 ) {
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F64, m, n, data_gold, data, &ld, &ld);
    max_error = l_diff.linf_abs;
  } else if ( dtype == LIBXSMM_DATATYPE_F32 ) {
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, m, n, data_gold, data, &ld, &ld);
    max_error = l_diff.linf_abs;
  } else if ( dtype == LIBXSMM_DATATYPE_BF16 ) {
    unsigned int l_i, l_j;
    libxsmm_bfloat16* h_data =      (libxsmm_bfloat16*)data;
    libxsmm_bfloat16* h_data_gold = (libxsmm_bfloat16*)data_gold;
    for (l_i = 0; l_i < m; l_i++) {
      for (l_j = 0; l_j < n; l_j++) {
        union libxsmm_bfloat16_hp tmp_c;
        union libxsmm_bfloat16_hp tmp_gold;
        double l_fabs;

        tmp_c.i[1] = h_data[(l_j * ld) + l_i];
        tmp_c.i[0] = 0;
        tmp_gold.i[1] = h_data_gold[(l_j * ld) + l_i];
        tmp_gold.i[0] = 0;
        l_fabs = fabs((double)tmp_gold.f - (double)tmp_c.f);
        if (max_error < l_fabs) max_error = l_fabs;
      }
    }
  } else if ( dtype == LIBXSMM_DATATYPE_I32 ) {
    unsigned int l_i, l_j;
    int* l_data = (int*)data;
    int* l_data_gold = (int*)data_gold;
    for (l_i = 0; l_i < m; l_i++) {
      for (l_j = 0; l_j < n; l_j++) {
        const double l_fabs = fabs((double)l_data_gold[(l_j * ld) + l_i] - (double)l_data[(l_j * ld) + l_i]);
        if (max_error < l_fabs) max_error = l_fabs;
      }
    }
  } else if ( dtype == LIBXSMM_DATATYPE_I8 ) {
    unsigned int l_i, l_j;
    unsigned char* l_data      = (unsigned char*)data;
    unsigned char* l_data_gold = (unsigned char*)data_gold;
    for (l_i = 0; l_i < m; l_i++) {
      for (l_j = 0; l_j < n; l_j++) {
        const double l_fabs = fabs((double)l_data_gold[(l_j * ld) + l_i] - (double)l_data[(l_j * ld) + l_i]);
        if (max_error < l_fabs) max_error = l_fabs;
      }
    }
  } else {
    max_error = 100.0;
  }

  return max_error;
}


