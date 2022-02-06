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

void init_random_matrix( const libxsmm_datatype dtype, void* data, const libxsmm_blasint br, const libxsmm_blasint ld, const libxsmm_blasint n, const libxsmm_blasint neg_values ) {
  double* d_data = (double*) data;
  float* f_data = (float*) data;
  libxsmm_bfloat16* bf_data = (libxsmm_bfloat16*) data;
  int* i_data = (int*) data;
  short* s_data = (short*) data;
  char* c_data = (char*) data;
  size_t l_r, l_i, l_j;

  for (l_r = 0; l_r < br; l_r++) {
    for (l_i = 0; l_i < ld; l_i++) {
      for (l_j = 0; l_j < n; l_j++) {
        if ( dtype == LIBXSMM_DATATYPE_F64 ) {
          d_data[(l_r * ld * n) + (l_j * ld) + l_i] = (neg_values) ? (0.05 - libxsmm_rng_f64()/10.0) : libxsmm_rng_f64();
        } else if ( dtype == LIBXSMM_DATATYPE_F32 ) {
          f_data[(l_r * ld * n) + (l_j * ld) + l_i] = (neg_values) ? (float)(0.05 - libxsmm_rng_f64()/10.0) : (float)libxsmm_rng_f64();
        } else if ( dtype == LIBXSMM_DATATYPE_BF16 ) {
          union libxsmm_bfloat16_hp tmp;
          tmp.f = (neg_values) ? (float)(0.05 - libxsmm_rng_f64()/10.0) : (float)libxsmm_rng_f64();
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

void init_zero_matrix( const libxsmm_datatype dtype, void* data, const libxsmm_blasint br, const libxsmm_blasint ld, const libxsmm_blasint n ) {
  char* l_data = (char*) data;
  memset( l_data, 0x0, br*ld*n*LIBXSMM_TYPESIZE(dtype) );
}

void init_garbage_matrix( const libxsmm_datatype dtype, void* data, const libxsmm_blasint br, const libxsmm_blasint ld, const libxsmm_blasint n ) {
  char* l_data = (char*) data;
  memset( l_data, 0xdeadbeef, br*ld*n*LIBXSMM_TYPESIZE(dtype) );
}

void apply_row_bcast_matrix( const libxsmm_datatype dtype, void* data, const libxsmm_blasint ld, const libxsmm_blasint m, const libxsmm_blasint n ) {
  double* d_data = (double*) data;
  float* f_data = (float*) data;
  unsigned short* s_data = (unsigned short*) data;
  unsigned char* c_data = (unsigned char*) data;
  size_t i,j;

  for ( i = 0; i < n; ++i ) {
    for ( j = 0; j < LIBXSMM_MAX(m,ld); ++j ) {
      if ( dtype == LIBXSMM_DATATYPE_F64 ) {
        d_data[(i*ld)+j] = d_data[i*ld];
      } else if ( (dtype == LIBXSMM_DATATYPE_F32) || (dtype == LIBXSMM_DATATYPE_I32) ) {
        f_data[(i*ld)+j] = f_data[i*ld];
      } else if ( (dtype == LIBXSMM_DATATYPE_BF16) || (dtype == LIBXSMM_DATATYPE_BF16) || (dtype == LIBXSMM_DATATYPE_I16) ) {
        s_data[(i*ld)+j] = s_data[i*ld];
      } else if ( dtype == LIBXSMM_DATATYPE_I8 ) {
        c_data[(i*ld)+j] = c_data[i*ld];
      } else {
      }
    }
  }
}

void apply_col_bcast_matrix( const libxsmm_datatype dtype, void* data, const libxsmm_blasint ld, const libxsmm_blasint m, const libxsmm_blasint n ) {
  double* d_data = (double*) data;
  float* f_data = (float*) data;
  unsigned short* s_data = (unsigned short*) data;
  unsigned char* c_data = (unsigned char*) data;
  size_t i,j;

  for ( i = 0; i < n; ++i ) {
    for ( j = 0; j < LIBXSMM_MAX(m,ld); ++j ) {
      if ( dtype == LIBXSMM_DATATYPE_F64 ) {
        d_data[(i*ld)+j] = d_data[j];
      } else if ( (dtype == LIBXSMM_DATATYPE_F32) || (dtype == LIBXSMM_DATATYPE_I32) ) {
        f_data[(i*ld)+j] = f_data[j];
      } else if ( (dtype == LIBXSMM_DATATYPE_BF16) || (dtype == LIBXSMM_DATATYPE_BF16) || (dtype == LIBXSMM_DATATYPE_I16) ) {
        s_data[(i*ld)+j] = s_data[j];
      } else if ( dtype == LIBXSMM_DATATYPE_I8 ) {
        c_data[(i*ld)+j] = c_data[j];
      } else {
      }
    }
  }
}

void apply_scalar_bcast_matrix( const libxsmm_datatype dtype, void* data, const libxsmm_blasint ld, const libxsmm_blasint m, const libxsmm_blasint n ) {
  double* d_data = (double*) data;
  float* f_data = (float*) data;
  unsigned short* s_data = (unsigned short*) data;
  unsigned char* c_data = (unsigned char*) data;
  size_t i,j;

  for ( i = 0; i < n; ++i ) {
    for ( j = 0; j < LIBXSMM_MAX(m,ld); ++j ) {
      if ( dtype == LIBXSMM_DATATYPE_F64 ) {
        d_data[(i*ld)+j] = d_data[0];
      } else if ( (dtype == LIBXSMM_DATATYPE_F32) || (dtype == LIBXSMM_DATATYPE_I32) ) {
        f_data[(i*ld)+j] = f_data[0];
      } else if ( (dtype == LIBXSMM_DATATYPE_BF16) || (dtype == LIBXSMM_DATATYPE_BF16) || (dtype == LIBXSMM_DATATYPE_I16) ) {
        s_data[(i*ld)+j] = s_data[0];
      } else if ( dtype == LIBXSMM_DATATYPE_I8 ) {
        c_data[(i*ld)+j] = c_data[0];
      } else {
      }
    }
  }
}

libxsmm_matdiff_info check_matrix( const libxsmm_datatype dtype, const void* data_gold, const void* data, const libxsmm_blasint ld, const libxsmm_blasint m, const libxsmm_blasint n ) {
  libxsmm_matdiff_info l_diff;

  libxsmm_matdiff_clear(&l_diff);

  if ( dtype == LIBXSMM_DATATYPE_F64 ) {
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F64, m, n, data_gold, data, &ld, &ld);
  } else if ( dtype == LIBXSMM_DATATYPE_F32 ) {
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, m, n, data_gold, data, &ld, &ld);
  } else if ( dtype == LIBXSMM_DATATYPE_BF16 ) {
    float* f_data_gold = (float*) malloc( sizeof(float)*n*ld );
    float* f_data      = (float*) malloc( sizeof(float)*n*ld );
    libxsmm_convert_bf16_f32( data_gold, f_data_gold, n*ld );
    libxsmm_convert_bf16_f32( data,      f_data,      n*ld );
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, m, n, f_data_gold, f_data, &ld, &ld);
    free( f_data );
    free( f_data_gold );
  } else if ( dtype == LIBXSMM_DATATYPE_I32 ) {
    const int* i_data_gold = (const int*)data_gold;
    const int* i_data      = (const int*)data;
    double* f_data_gold = (double*) malloc( sizeof(double)*n*ld );
    double* f_data      = (double*) malloc( sizeof(double)*n*ld );
    size_t i;
    for ( i = 0; i < (size_t)n*ld; ++i ) f_data_gold[i] = (double)i_data_gold[i];
    for ( i = 0; i < (size_t)n*ld; ++i ) f_data[i]      = (double)i_data[i];
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F64, m, n, f_data_gold, f_data, &ld, &ld);
    free( f_data );
    free( f_data_gold );
  } else if ( dtype == LIBXSMM_DATATYPE_I8 ) {
    const char* i_data_gold = (const char*)data_gold;
    const char* i_data      = (const char*)data;
    double* f_data_gold = (double*) malloc( sizeof(double)*n*ld );
    double* f_data      = (double*) malloc( sizeof(double)*n*ld );
    size_t i;
    for ( i = 0; i < (size_t)n*ld; ++i ) f_data_gold[i] = (double)i_data_gold[i];
    for ( i = 0; i < (size_t)n*ld; ++i ) f_data[i]      = (double)i_data[i];
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F64, m, n, f_data_gold, f_data, &ld, &ld);
    free( f_data );
    free( f_data_gold );
  } else {
  }

  return l_diff;
}


