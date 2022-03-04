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

typedef unsigned short float16;
typedef union float_uint {
  float f;
  unsigned int u;
} float_uint;


float convert_f16__f32( float16 in ) {
  unsigned int f32_bias = 127;
  unsigned int f16_bias = 15;
  unsigned int s = ( in & 0x8000 ) << 16;
  unsigned int e = ( in & 0x7c00 ) >> 10;
  unsigned int m = ( in & 0x03ff );
  unsigned int e_norm = e + (f32_bias - f16_bias);
  float_uint res;

  /* convert denormal fp16 number into a normal fp32 number */
  if ( (e == 0) && (m != 0) ) {
    unsigned int lz_cnt = 9;
    lz_cnt = ( m >   0x1 ) ? 8 : lz_cnt;
    lz_cnt = ( m >   0x3 ) ? 7 : lz_cnt;
    lz_cnt = ( m >   0x7 ) ? 6 : lz_cnt;
    lz_cnt = ( m >   0xf ) ? 5 : lz_cnt;
    lz_cnt = ( m >  0x1f ) ? 4 : lz_cnt;
    lz_cnt = ( m >  0x3f ) ? 3 : lz_cnt;
    lz_cnt = ( m >  0x7f ) ? 2 : lz_cnt;
    lz_cnt = ( m >  0xff ) ? 1 : lz_cnt;
    lz_cnt = ( m > 0x1ff ) ? 0 : lz_cnt;
    e_norm -= lz_cnt;
    m = (m << (lz_cnt+1)) & 0x03ff;
  } else if ( (e == 0) && (m == 0) ) {
    e_norm = 0;
  } else if ( e == 0x1f ) {
    e_norm = 0xff;
    m |= ( m == 0 ) ? 0 : 0x0200; /* making first mantissa bit 1 */
  }

  /* set result to 0 */
  res.u = 0x0;
  /* set exp and mant */
  res.u |= (e_norm << 23);
  res.u |= (m << 13);
  /* sign it */
  res.u |= s;

  return res.f;
}

float16 convert_f32_f16( float in ) {
  unsigned int f32_bias = 127;
  unsigned int f16_bias = 15;
  float_uint hybrid_in;
  float16 res = 0;
  unsigned int s, e, m, e_f32, m_f32;
  unsigned int fixup;
  hybrid_in.f = in;

  /* DAZ */
  hybrid_in.u = ( (hybrid_in.u & 0x7f800000) == 0x0 ) ? ( hybrid_in.u & 0x80000000 ) : ( hybrid_in.u & 0xffffffff );

  s = ( hybrid_in.u & 0x80000000 ) >> 16;
  e_f32 = ( hybrid_in.u & 0x7f800000 ) >> 23;
  m_f32 = ( hybrid_in.u & 0x007fffff );

  /* special value */
  if ( e_f32 == 0xff ) {
    e = 0x1f;
    m = (m_f32 == 0) ? 0 : (m_f32 >> 13) | 0x200;
  /* overflow */
  } else if ( e_f32 > (f32_bias + f16_bias) ) {
    e = 0x1f;
    m = 0x0;
  /* smaller than denormal f16 */
  } else if ( e_f32 < f32_bias - f16_bias - 10 ) {
    e = 0x0;
    m = 0x0;
  /* denormal */
  } else if ( e_f32 <= f32_bias - f16_bias ) {
    /* RNE */
    /* denormalized mantissa */
    m = m_f32 | 0x00800000;
    /* addtionally subnormal shift */
    m = m >> ((f32_bias - f16_bias) + 1 - e_f32);
    /* preserve sticky bit (some sticky bits are lost when denormalizing) */
    m |= (((m_f32 & 0x1fff) + 0x1fff) >> 13);
    /* RNE Round */
    fixup = (m >> 13) & 0x1;
    m = m + 0x000000fff + fixup;
    m = m >> 13;
    e = 0x0;
  /* normal */
  } else {
    /* RNE round */
    fixup = (m_f32 >> 13) & 0x1;
    hybrid_in.u = hybrid_in.u + 0x000000fff + fixup;
    e = ( hybrid_in.u & 0x7f800000 ) >> 23;
    m = ( hybrid_in.u & 0x007fffff );
    e -= (f32_bias - f16_bias);
    m = m >> 13;
  }

  /* set result to 0 */
  res = 0x0;
  /* set exp and mant */
  res |= e << 10;
  res |= m;
  /* sign it */
  res |= s;

  return res;
}

void init_random_matrix( const libxsmm_datatype dtype, void* data, const libxsmm_blasint br, const libxsmm_blasint ld, const libxsmm_blasint n, const libxsmm_blasint neg_values ) {
  double* d_data = (double*) data;
  float* f_data = (float*) data;
  libxsmm_bfloat16* bf_data = (libxsmm_bfloat16*) data;
  libxsmm_bfloat8* bf8_data = (libxsmm_bfloat8*) data;
  int* i_data = (int*) data;
  short* s_data = (short*) data;
  char* c_data = (char*) data;
  libxsmm_blasint l_r, l_i, l_j;

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
        } else if ( dtype == LIBXSMM_DATATYPE_BF8 ) {
          union libxsmm_bfloat8_qp tmp;
          tmp.hf = convert_f32_f16 ((neg_values) ? (float)(0.05 - libxsmm_rng_f64()/10.0) : (float)libxsmm_rng_f64());
          bf8_data[(l_r * ld * n) + (l_j * ld) + l_i] = tmp.i[1];
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
  libxsmm_blasint i,j;

  for ( i = 0; i < n; ++i ) {
    for ( j = 0; j < LIBXSMM_MAX(m,ld); ++j ) {
      if ( dtype == LIBXSMM_DATATYPE_F64 ) {
        d_data[(i*ld)+j] = d_data[i*ld];
      } else if ( (dtype == LIBXSMM_DATATYPE_F32) || (dtype == LIBXSMM_DATATYPE_I32) ) {
        f_data[(i*ld)+j] = f_data[i*ld];
      } else if ( (dtype == LIBXSMM_DATATYPE_BF16) || (dtype == LIBXSMM_DATATYPE_BF16) || (dtype == LIBXSMM_DATATYPE_I16) ) {
        s_data[(i*ld)+j] = s_data[i*ld];
      } else if ( (dtype == LIBXSMM_DATATYPE_I8) || (dtype == LIBXSMM_DATATYPE_BF8) ) {
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
  libxsmm_blasint i,j;

  for ( i = 0; i < n; ++i ) {
    for ( j = 0; j < LIBXSMM_MAX(m,ld); ++j ) {
      if ( dtype == LIBXSMM_DATATYPE_F64 ) {
        d_data[(i*ld)+j] = d_data[j];
      } else if ( (dtype == LIBXSMM_DATATYPE_F32) || (dtype == LIBXSMM_DATATYPE_I32) ) {
        f_data[(i*ld)+j] = f_data[j];
      } else if ( (dtype == LIBXSMM_DATATYPE_BF16) || (dtype == LIBXSMM_DATATYPE_BF16) || (dtype == LIBXSMM_DATATYPE_I16) ) {
        s_data[(i*ld)+j] = s_data[j];
      } else if ( (dtype == LIBXSMM_DATATYPE_I8) || (dtype == LIBXSMM_DATATYPE_BF8) ) {
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
  libxsmm_blasint i,j;

  for ( i = 0; i < n; ++i ) {
    for ( j = 0; j < LIBXSMM_MAX(m,ld); ++j ) {
      if ( dtype == LIBXSMM_DATATYPE_F64 ) {
        d_data[(i*ld)+j] = d_data[0];
      } else if ( (dtype == LIBXSMM_DATATYPE_F32) || (dtype == LIBXSMM_DATATYPE_I32) ) {
        f_data[(i*ld)+j] = f_data[0];
      } else if ( (dtype == LIBXSMM_DATATYPE_BF16) || (dtype == LIBXSMM_DATATYPE_BF16) || (dtype == LIBXSMM_DATATYPE_I16) ) {
        s_data[(i*ld)+j] = s_data[0];
      } else if ( (dtype == LIBXSMM_DATATYPE_I8) || (dtype == LIBXSMM_DATATYPE_BF8) ) {
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
  } else if ( dtype == LIBXSMM_DATATYPE_BF8 ) {
    float* f_data_gold = (float*) malloc( sizeof(float)*n*ld );
    float* f_data      = (float*) malloc( sizeof(float)*n*ld );
    libxsmm_convert_bf8_f32( data_gold, f_data_gold, n*ld );
    libxsmm_convert_bf8_f32( data,      f_data,      n*ld );
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, m, n, f_data_gold, f_data, &ld, &ld);
    free( f_data );
    free( f_data_gold );
  } else if ( dtype == LIBXSMM_DATATYPE_I32 ) {
    const int* i_data_gold = (const int*)data_gold;
    const int* i_data      = (const int*)data;
    double* f_data_gold = (double*) malloc( sizeof(double)*n*ld );
    double* f_data      = (double*) malloc( sizeof(double)*n*ld );
    libxsmm_blasint i;
    for ( i = 0; i < n*ld; ++i ) f_data_gold[i] = (double)i_data_gold[i];
    for ( i = 0; i < n*ld; ++i ) f_data[i]      = (double)i_data[i];
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F64, m, n, f_data_gold, f_data, &ld, &ld);
    free( f_data );
    free( f_data_gold );
  } else if ( dtype == LIBXSMM_DATATYPE_I8 ) {
    const char* i_data_gold = (const char*)data_gold;
    const char* i_data      = (const char*)data;
    double* f_data_gold = (double*) malloc( sizeof(double)*n*ld );
    double* f_data      = (double*) malloc( sizeof(double)*n*ld );
    libxsmm_blasint i;
    for ( i = 0; i < n*ld; ++i ) f_data_gold[i] = (double)i_data_gold[i];
    for ( i = 0; i < n*ld; ++i ) f_data[i]      = (double)i_data[i];
    libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F64, m, n, f_data_gold, f_data, &ld, &ld);
    free( f_data );
    free( f_data_gold );
  } else {
  }

  return l_diff;
}


