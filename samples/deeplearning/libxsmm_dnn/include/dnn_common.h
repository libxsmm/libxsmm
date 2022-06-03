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
#include <libxsmm_intrinsics_x86.h>

#if defined(_OPENMP)
# include <omp.h>
#endif

typedef struct {
  int nImg;
  int nIfm;
  int nOfm;
  int ifhp;
  int ifwp;
  int ifh;
  int ifw;
  int ofhp;
  int ofwp;
  int ofh;
  int ofw;
  int pad_h;
  int pad_w;
  int pad_h_in;
  int pad_w_in;
  int pad_h_out;
  int pad_w_out;
  int kh;
  int kw;
  int stride_h;
  int stride_w;
} naive_conv_t;

typedef struct {
  int N;
  int C;
  int H;
  int W;
  int pad_h_in;
  int pad_w_in;
  int pad_h_out;
  int pad_w_out;
  int stride_h;
  int stride_w;
  int norm_type;  /* 0: full batchnorm, 1: batch scaling only */
  int fuse_type;  /* 0: nothing fused, 1: relu fused, 2: elementwise fused, 3: relu and elementwise fused */
} naive_fusedbatchnorm_t;

typedef struct {
  int N;
  int C;
  int G;
  int H;
  int W;
  int pad_h_in;
  int pad_w_in;
  int pad_h_out;
  int pad_w_out;
  int stride_h;
  int stride_w;
  int fuse_type;  /* 0: nothing fused, 1: relu fused, 2: elementwise fused, 3: relu and elementwise fused */
} naive_fusedgroupnorm_t;

typedef struct {
  int N;
  int C;
  int K;
  int fuse_type;  /* 0: nothing fused */
} naive_fullyconnected_t;

typedef struct {
  int N;
  int C;
  int H;
  int W;
  int R;
  int S;
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  int type;
} naive_pooling_t;

LIBXSMM_INLINE void mask_compress_uint8 (unsigned char * src_mask_uncompressed, unsigned char * dst_mask_compressed, size_t length) {
  int i;
  for (i = 0; i < (int)length; ++i) {
    if (i%8 == 0)
      dst_mask_compressed[i/8] = 0;
    dst_mask_compressed[i/8] |= (src_mask_uncompressed[i] == 0 ? 0x0 : (1 << i%8));;
  }
}

/* it's fine to alias in and out */
LIBXSMM_INLINE void truncate_mask_fp32_bf16(float* in, float* out, unsigned int len) {
  unsigned int i = 0;

  /* truncate buffer to bf16 */
  for ( i = 0; i < len; ++i ) {
    union libxsmm_bfloat16_hp t;

    t.f = in[i];
    t.i[0] = 0;
    out[i] = t.f;
  }
}

/* it's fine to alias in and out */
LIBXSMM_INLINE void rnaz_mask_fp32_bf16(float* in, float* out, unsigned int len) {
  unsigned int i = 0;

  /* rnaz buffer to bf16 */
  for ( i = 0; i < len; ++i ) {
    unsigned int int_round = 0;
    unsigned int do_round = 1;
    const void *const ptr = &int_round;

    int_round = *((unsigned int*)&(in[i]));

    /* we don't round NaN and inf */
    if ( (int_round & 0x7f800000) == 0x7f800000 ) {
      do_round = 0;
    }

    /* perform round nearest tie away from zero */
    if ( do_round != 0 ) {
      int_round = int_round + 0x00008000;
    }

    /* chop bits to create BFP16 in FP32 */
    int_round = int_round & 0xffff0000;

    out[i] = *((float*)ptr);
  }
}

/* it's fine to alias in and out */
LIBXSMM_INLINE void rne_mask_fp32_bf16(float* in, float* out, unsigned int len) {
  unsigned int i = 0;

  /* rnaz buffer to bf16 */
  for ( i = 0; i < len; ++i ) {
    unsigned int int_round = 0;
    unsigned int do_round = 1;
    const void *const ptr = &int_round;

    int_round = *((unsigned int*)&(in[i]));

    /* we don't round NaN and inf */
    if ( (int_round & 0x7f800000) == 0x7f800000 ) {
      do_round = 0;
    }

    /* perform round nearest tie even */
    if ( do_round != 0 ) {
      unsigned int fixup = (int_round >> 16) & 1;
      int_round = int_round + 0x00007fff + fixup;
    }

    /* chop bits to create BFP16 in FP32 */
    int_round = int_round & 0xffff0000;

    out[i] = *((float*)ptr);
  }
}

/* it's fine to alias in and out */
LIBXSMM_INLINE void rne_mask_fp32_bfp16(float* in, float* out, unsigned int len) {
  unsigned int i = 0;

  /* rnaz buffer to bfp16 */
  for ( i = 0; i < len; ++i ) {
    unsigned int int_round = 0;
    unsigned int do_round = 1;
    const void *const ptr = &int_round;

    int_round = *((unsigned int*)&(in[i]));

    /* we don't round NaN and inf */
    if ( (int_round & 0x7f800000) == 0x7f800000 ) {
      do_round = 0;
    }

    /* perform round nearest tie even */
    if ( do_round != 0 ) {
      unsigned int fixup = (int_round >> 16) & 1;
      int_round = int_round + 0x00007fff + fixup;
    }

    /* chop bits to create BFP16 in FP32 */
    int_round = int_round & 0xffff0000;

    out[i] = *((float*)ptr);
  }
}

LIBXSMM_INLINE void zero_buf(float* buf, size_t size) {
  int i;
#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(i);
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    buf[i] = 0.0f;
  }
}

LIBXSMM_INLINE void zero_buf_fp64(double* buf, size_t size) {
  int i;
#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(i);
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    buf[i] = 0.0f;
  }
}

LIBXSMM_INLINE void zero_buf_bf16(libxsmm_bfloat16* buf, size_t size) {
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    buf[i] = 0;
  }
}

LIBXSMM_INLINE void zero_buf_int16(short* buf, size_t size) {
  int i;
#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(i);
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    buf[i] = 0;
  }
}

LIBXSMM_INLINE void zero_buf_int32(int* buf, size_t size) {
  int i;
#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(i);
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    buf[i] = 0;
  }
}

LIBXSMM_INLINE void zero_buf_int8(char* buf, size_t size) {
  int i;
#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(i);
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    buf[i] = 0;
  }
}

LIBXSMM_INLINE void zero_buf_uint8(unsigned char* buf, size_t size) {
  int i;
#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(i);
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    buf[i] = 0;
  }
}

LIBXSMM_INLINE void copy_buf(float* src, float* dst, size_t size) {
  int i;
#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(i);
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    dst[i] = src[i];
  }
}

LIBXSMM_INLINE void copy_buf_int16(short* src, short* dst, size_t size) {
  int i;
#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(i);
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    dst[i] = src[i];
  }
}

LIBXSMM_INLINE void copy_buf_int8(char* src, char* dst, size_t size) {
  int i;
#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(i);
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    dst[i] = src[i];
  }
}

LIBXSMM_INLINE void copy_buf_uint8(unsigned char* src, unsigned char* dst, size_t size) {
  int i;
#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(i);
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    dst[i] = src[i];
  }
}

LIBXSMM_INLINE void init_buf(float* buf, size_t size, int initPos, int initOne)
{
  int i;
  zero_buf(buf, size);
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    buf[i] = (float)((initOne != 0) ? 1.0 : ((initPos != 0) ? libxsmm_rng_f64() : (0.05 - libxsmm_rng_f64()/10.0)));
  }
}

LIBXSMM_INLINE void init_buf_bf16(libxsmm_bfloat16* buf, size_t size, int initPos, int initOne)
{
  int i;
  zero_buf_bf16(buf, size);
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    libxsmm_bfloat16_hp tmp;
    tmp.f = (float)((initOne != 0) ? 1.0 : ((initPos != 0) ? libxsmm_rng_f64() : (0.05 - libxsmm_rng_f64()/10.0)));
    buf[i] = tmp.i[1];
  }
}

LIBXSMM_INLINE void dequantize_buffer_char( char* in_buffer, float* out_buffer, int length, unsigned char scf ) {
  const float val_exp = libxsmm_sexp2_i8i(-scf);
  int i = 0;
#ifdef _OPENMP
# pragma omp parallel for private(i)
#endif
  for ( i = 0; i < length; ++i ) {
    out_buffer[i] = ((float)in_buffer[i])*val_exp;
  }
}

LIBXSMM_INLINE float libxsmm_internal_get_max_common( float* in_buffer, int length ) {
  float absmax_value = LIBXSMM_ABS(in_buffer[0]);
  int i = 0;
  for (i = 1; i < length; ++i ) {
    if (LIBXSMM_ABS(in_buffer[i]) > absmax_value) {
      absmax_value = LIBXSMM_ABS(in_buffer[i]);
    }
  }
  return absmax_value;
}

LIBXSMM_INLINE void quantize_buffer_char(float *in_buffer, char *out_buffer, int size, unsigned char add_shift, unsigned char* scf) {
  int i;
  const float max_value = libxsmm_internal_get_max_common(in_buffer, size);
  int maxexp = 0;
  /* take return value of LIBXSMM_FREXPF to mute static analysis issue */
  float scfq = LIBXSMM_FREXPF(max_value, &maxexp);
  maxexp -= (7 - add_shift);
  scfq = libxsmm_sexp2_i8i(-maxexp);
  for (i=0; i<size; i++) {
    out_buffer[i] = (char)LIBXSMM_ROUNDF(in_buffer[i]*scfq);
  }
  *scf = (unsigned char)(-maxexp);
}

LIBXSMM_INLINE void quantize_buffer_uchar(float *in_buffer, unsigned char *out_buffer, int size, unsigned char add_shift, unsigned char* scf) {
  int i;
  const float max_value = libxsmm_internal_get_max_common(in_buffer, size);
  int maxexp = 0;
  /* take return value of LIBXSMM_FREXPF to mute static analysis issue */
  float scfq = LIBXSMM_FREXPF(max_value, &maxexp);
  maxexp -= (7 - add_shift);
  scfq = libxsmm_sexp2_i8i(-maxexp);
  for (i=0; i<size; i++) {
    out_buffer[i] = (unsigned char)LIBXSMM_ROUNDF(in_buffer[i]*scfq);
  }
  *scf = (unsigned char)(-maxexp);
}

LIBXSMM_INLINE void init_buf_range(float* buf, size_t size, float low, float high)
{
  int i;
  float range = high - low;
  zero_buf(buf, size);
  for (i = 0; i < (int)size; ++i) {
    buf[i] = (((float)rand())/RAND_MAX)*range+low;
  }
}

LIBXSMM_INLINE void init_buf_int16(short* buf, size_t size, int initPos, int initOne)
{
  int i;
  zero_buf_int16(buf, size);
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    buf[i] = (short)((initOne != 0) ? 1 : ((initPos != 0) ? (rand()%7) : (rand()%7-3)));
  }
}

LIBXSMM_INLINE void init_buf_int32(int* buf, size_t size, int initPos, int initOne)
{
  int i;
  zero_buf_int32(buf, size);
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    buf[i] = (int)((initOne != 0) ? 1 : ((initPos != 0) ? (rand()%7) : (rand()%7-3)));
  }
}

LIBXSMM_INLINE void init_buf_int8(char* buf, size_t size, int initPos, int initOne)
{
  int i;
  zero_buf_int8(buf, size);
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    buf[i] = (char)((initOne != 0) ? 1 : ((initPos != 0) ? (rand()%3) : (rand()%3)-1));
  }
}

LIBXSMM_INLINE void init_buf_uint8(unsigned char* buf, size_t size, int initPos, int initOne)
{
  int i;
  LIBXSMM_UNUSED(initPos);
  zero_buf_uint8(buf, size);
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    buf[i] = (unsigned char)((initOne != 0) ? 1 : (rand()%3));
  }
}

/* src_size is in float elements */
LIBXSMM_INLINE void extend_buf_fp32_to_fp64 (const float *src, double *dst, size_t src_size)
{
  int i;
  for (i = 0; i < (int)src_size; i++)
    dst[i] = (double)(src[i]);
}

/* src_size is in double elements */
LIBXSMM_INLINE void truncate_buf_fp64_to_fp32 (const double *src, float *dst, size_t src_size)
{
  int i;
  for (i = 0; i < (int)src_size; i++)
    dst[i] = (float)(src[i]);
}

LIBXSMM_INLINE void set_zeropad_nchw(float* nchw, int N, int C, int H, int W, int pad_h, int pad_w)
{
  LIBXSMM_VLA_DECL(4, float, input, nchw, C, H, W);
  int n, h, w, c;

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c); LIBXSMM_OMP_VAR(h); LIBXSMM_OMP_VAR(w);
# pragma omp parallel for private(n,c,h,w)
#endif
  for ( n = 0; n < N; n++ ) {
    for ( c = 0; c < C; c++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          if (h < pad_h || h >= H-pad_h || w < pad_w || w >= W-pad_w) {
            LIBXSMM_VLA_ACCESS(4, input, n, c, h, w, C, H, W) = 0.0;
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void set_zeropad_nchw_int16(short* nchw, int N, int C, int H, int W, int pad_h, int pad_w)
{
  LIBXSMM_VLA_DECL(4, short, input, nchw, C, H, W);
  int n, h, w, c;

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c); LIBXSMM_OMP_VAR(h); LIBXSMM_OMP_VAR(w);
# pragma omp parallel for private(n,c,h,w)
#endif
  for ( n = 0; n < N; n++ ) {
    for ( c = 0; c < C; c++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          if (h < pad_h || h >= H-pad_h || w < pad_w || w >= W-pad_w) {
            LIBXSMM_VLA_ACCESS(4, input, n, c, h, w, C, H, W) = 0;
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void set_zeropad_nchw_int32(int* nchw, int N, int C, int H, int W, int pad_h, int pad_w)
{
  LIBXSMM_VLA_DECL(4, int, input, nchw, C, H, W);
  int n, h, w, c;

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c); LIBXSMM_OMP_VAR(h); LIBXSMM_OMP_VAR(w);
# pragma omp parallel for private(n,c,h,w)
#endif
  for ( n = 0; n < N; n++ ) {
    for ( c = 0; c < C; c++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          if (h < pad_h || h >= H-pad_h || w < pad_w || w >= W-pad_w) {
            LIBXSMM_VLA_ACCESS(4, input, n, c, h, w, C, H, W) = 0;
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void set_zeropad_nchw_uint8(unsigned char* nchw, int N, int C, int H, int W, int pad_h, int pad_w)
{
  LIBXSMM_VLA_DECL(4, unsigned char, input, nchw, C, H, W);
  int n, h, w, c;

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c); LIBXSMM_OMP_VAR(h); LIBXSMM_OMP_VAR(w);
# pragma omp parallel for private(n,c,h,w)
#endif
  for ( n = 0; n < N; n++ ) {
    for ( c = 0; c < C; c++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          if (h < pad_h || h >= H-pad_h || w < pad_w || w >= W-pad_w) {
            LIBXSMM_VLA_ACCESS(4, input, n, c, h, w, C, H, W) = 0;
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void copy_internal_nchw(float* dst , float* src, int N, int C, int H, int W, int pad_h, int pad_w)
{
  LIBXSMM_VLA_DECL(4, float, input, src, C, H, W);
  LIBXSMM_VLA_DECL(4, float, new_input, dst, C, H+2*pad_h, W+2*pad_w);
  int n, h, w, c;

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c); LIBXSMM_OMP_VAR(h); LIBXSMM_OMP_VAR(w);
# pragma omp parallel for private(n,c,h,w)
#endif
  for ( n = 0; n < N; n++ ) {
    for ( c = 0; c < C; c++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          LIBXSMM_VLA_ACCESS(4, new_input, n, c, h+pad_h, w+pad_w, C, H+2*pad_h, W+2*pad_w) =  LIBXSMM_VLA_ACCESS(4, input, n, c, h, w, C, H, W);
        }
      }
    }
  }
}

LIBXSMM_INLINE void copy_internal_nchw_int16(short* dst , short* src, int N, int C, int H, int W, int pad_h, int pad_w)
{
  LIBXSMM_VLA_DECL(4, short, input, src, C, H, W);
  LIBXSMM_VLA_DECL(4, short, new_input, dst, C, H+2*pad_h, W+2*pad_w);
  int n, h, w, c;

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c); LIBXSMM_OMP_VAR(h); LIBXSMM_OMP_VAR(w);
# pragma omp parallel for private(n,c,h,w)
#endif
  for ( n = 0; n < N; n++ ) {
    for ( c = 0; c < C; c++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          LIBXSMM_VLA_ACCESS(4, new_input, n, c, h+pad_h, w+pad_w, C, H+2*pad_h, W+2*pad_w) =  LIBXSMM_VLA_ACCESS(4, input, n, c, h, w, C, H, W);
        }
      }
    }
  }
}

LIBXSMM_INLINE void copy_internal_nchw_uint8(unsigned char* dst , unsigned char* src, int N, int C, int H, int W, int pad_h, int pad_w)
{
  LIBXSMM_VLA_DECL(4, unsigned char, input, src, C, H, W);
  LIBXSMM_VLA_DECL(4, unsigned char, new_input, dst, C, H+2*pad_h, W+2*pad_w);
  int n, h, w, c;

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c); LIBXSMM_OMP_VAR(h); LIBXSMM_OMP_VAR(w);
# pragma omp parallel for private(n,c,h,w)
#endif
  for ( n = 0; n < N; n++ ) {
    for ( c = 0; c < C; c++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          LIBXSMM_VLA_ACCESS(4, new_input, n, c, h+pad_h, w+pad_w, C, H+2*pad_h, W+2*pad_w) =  LIBXSMM_VLA_ACCESS(4, input, n, c, h, w, C, H, W);
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_copy_NCHW_to_NHWC(const float* nchw, float* nhwc, int N, int H, int W, int C)
{
  LIBXSMM_VLA_DECL(4,       float, output, nhwc, H, W, C);
  LIBXSMM_VLA_DECL(4, const float,  input, nchw, C, H, W);
  int n, h, w, c;

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c); LIBXSMM_OMP_VAR(h); LIBXSMM_OMP_VAR(w);
# pragma omp parallel for private(n,c,h,w)
#endif
  for ( n = 0; n < N; n++ ) {
    for ( h = 0; h < H; h++ ) {
      for ( w = 0; w < W; w++ ) {
        for ( c = 0; c < C; c++ ) {
          LIBXSMM_VLA_ACCESS(4, output, n, h, w, c, H, W, C) =
          LIBXSMM_VLA_ACCESS(4, input, n, c, h, w, C, H, W);
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_copy_NHWC_to_NCHW(const float* nhwc, float* nchw, int N, int H, int W, int C)
{
  LIBXSMM_VLA_DECL(4,       float, output, nchw, C, H, W);
  LIBXSMM_VLA_DECL(4, const float,  input, nhwc, H, W, C);
  int n, h, w, c;

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c); LIBXSMM_OMP_VAR(h); LIBXSMM_OMP_VAR(w);
# pragma omp parallel for private(n,c,h,w)
#endif
  for ( n = 0; n < N; n++ ) {
    for ( h = 0; h < H; h++ ) {
      for ( w = 0; w < W; w++ ) {
        for ( c = 0; c < C; c++ ) {
          LIBXSMM_VLA_ACCESS(4, output, n, c, h, w, C, H, W) =
          LIBXSMM_VLA_ACCESS(4, input, n, h, w, c, H, W, C);
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_copy_KCRS_to_RSCK(const float* kcrs, float* rsck, int R, int S, int C, int K)
{
  LIBXSMM_VLA_DECL(4,       float, output, rsck, S, C, K);
  LIBXSMM_VLA_DECL(4, const float,  input, kcrs, C, R, S);
  int r, s, c, k;

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(s); LIBXSMM_OMP_VAR(c); LIBXSMM_OMP_VAR(k);
# pragma omp parallel for private(r,s,c,k)
#endif
  for ( r = 0; r < R; r++ ) {
    for ( s = 0; s < S; s++ ) {
      for ( c = 0; c < C; c++ ) {
        for ( k = 0; k < K; k++ ) {
          LIBXSMM_VLA_ACCESS(4, output, r, s, c, k, S, C, K) =
          LIBXSMM_VLA_ACCESS(4, input, k, c, r, s, C, R, S);
        }
      }
    }
  }
}


LIBXSMM_INLINE void naive_copy_RSCK_to_KCRS(const float* rsck, float* kcrs, int R, int S, int C, int K)
{
  LIBXSMM_VLA_DECL(4, const float,  input, rsck, S, C, K);
  LIBXSMM_VLA_DECL(4,       float, output, kcrs, C, R, S);
  int r, s, c, k;

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(s); LIBXSMM_OMP_VAR(c); LIBXSMM_OMP_VAR(k);
# pragma omp parallel for private(r,s,c,k)
#endif
  for ( r = 0; r < R; r++ ) {
    for ( s = 0; s < S; s++ ) {
      for ( c = 0; c < C; c++ ) {
        for ( k = 0; k < K; k++ ) {
          LIBXSMM_VLA_ACCESS(4, output, k, c, r, s, C, R, S) =
            LIBXSMM_VLA_ACCESS(4, input, r, s, c, k, S, C, K);
        }
      }
    }
  }
}

LIBXSMM_INLINE void matrix_copy_NC_to_NCNC(float *src, float *dst, int T, int N, int C, int bn, int bc)
{
  int t, n1, n2, c1, c2;
  int nBlocks = N/bn;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(3, float, real_src, src, N, C);
  LIBXSMM_VLA_DECL(5, float, real_dst, dst, nBlocks, cBlocks, bn, bc);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(n1); LIBXSMM_OMP_VAR(c1); LIBXSMM_OMP_VAR(n2); LIBXSMM_OMP_VAR(c2);
# pragma omp parallel for private(t,n1,c1,n2,c2)
#endif
  for (t = 0; t < T; t++) {
    for (n1 = 0; n1 < nBlocks; n1++) {
      for (c1 = 0; c1 < cBlocks; c1++) {
        for (n2 = 0; n2 < bn; n2++) {
          for (c2 = 0; c2 < bc; c2++) {
            LIBXSMM_VLA_ACCESS(5, real_dst, t, n1, c1, n2, c2, nBlocks, cBlocks, bn, bc) =
              LIBXSMM_VLA_ACCESS(3, real_src, t, n1*bn+n2, c1*bc+c2, N, C);
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void matrix_copy_NCNC_to_NC(float *src, float *dst, int T, int N, int C, int bn, int bc)
{
  int t, n1, n2, c1, c2;
  int nBlocks = N/bn;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(3, float, real_dst, dst, N, C);
  LIBXSMM_VLA_DECL(5, float, real_src, src, nBlocks, cBlocks, bn, bc);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(n1); LIBXSMM_OMP_VAR(c1); LIBXSMM_OMP_VAR(n2); LIBXSMM_OMP_VAR(c2);
# pragma omp parallel for private(t,n1,c1,n2,c2)
#endif
  for (t = 0; t < T; t++) {
    for (n1 = 0; n1 < nBlocks; n1++) {
      for (c1 = 0; c1 < cBlocks; c1++) {
        for (n2 = 0; n2 < bn; n2++) {
          for (c2 = 0; c2 < bc; c2++) {
            LIBXSMM_VLA_ACCESS(3, real_dst, t, n1*bn+n2, c1*bc+c2, N, C) =
              LIBXSMM_VLA_ACCESS(5, real_src, t, n1, c1, n2, c2, nBlocks, cBlocks, bn, bc);
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void matrix_copy_NC_to_NCNC_bf16(libxsmm_bfloat16 *src, libxsmm_bfloat16 *dst, int T, int N, int C, int bn, int bc)
{
  int t, n1, n2, c1, c2;
  int nBlocks = N/bn;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, real_src, src, N, C);
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, real_dst, dst, nBlocks, cBlocks, bn, bc);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(n1); LIBXSMM_OMP_VAR(c1); LIBXSMM_OMP_VAR(n2); LIBXSMM_OMP_VAR(c2);
# pragma omp parallel for private(t,n1,c1,n2,c2)
#endif
  for (t = 0; t < T; t++) {
    for (n1 = 0; n1 < nBlocks; n1++) {
      for (c1 = 0; c1 < cBlocks; c1++) {
        for (n2 = 0; n2 < bn; n2++) {
          for (c2 = 0; c2 < bc; c2++) {
            LIBXSMM_VLA_ACCESS(5, real_dst, t, n1, c1, n2, c2, nBlocks, cBlocks, bn, bc) =
              LIBXSMM_VLA_ACCESS(3, real_src, t, n1*bn+n2, c1*bc+c2, N, C);
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void matrix_copy_NC_to_NCNC_bf16_serial(libxsmm_bfloat16 *src, libxsmm_bfloat16 *dst, int T, int N, int C, int bn, int bc)
{
  int t, n1, n2, c1, c2;
  int nBlocks = N/bn;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, real_src, src, N, C);
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, real_dst, dst, nBlocks, cBlocks, bn, bc);

  for (t = 0; t < T; t++) {
    for (n1 = 0; n1 < nBlocks; n1++) {
      for (c1 = 0; c1 < cBlocks; c1++) {
        for (n2 = 0; n2 < bn; n2++) {
          for (c2 = 0; c2 < bc; c2++) {
            LIBXSMM_VLA_ACCESS(5, real_dst, t, n1, c1, n2, c2, nBlocks, cBlocks, bn, bc) =
              LIBXSMM_VLA_ACCESS(3, real_src, t, n1*bn+n2, c1*bc+c2, N, C);
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void matrix_copy_NCNC_to_NC_bf16(libxsmm_bfloat16 *src, libxsmm_bfloat16 *dst, int T, int N, int C, int bn, int bc)
{
  int t, n1, n2, c1, c2;
  int nBlocks = N/bn;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, real_dst, dst, N, C);
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, real_src, src, nBlocks, cBlocks, bn, bc);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(n1); LIBXSMM_OMP_VAR(c1); LIBXSMM_OMP_VAR(n2); LIBXSMM_OMP_VAR(c2);
# pragma omp parallel for private(t,n1,c1,n2,c2)
#endif
  for (t = 0; t < T; t++) {
    for (n1 = 0; n1 < nBlocks; n1++) {
      for (c1 = 0; c1 < cBlocks; c1++) {
        for (n2 = 0; n2 < bn; n2++) {
          for (c2 = 0; c2 < bc; c2++) {
            LIBXSMM_VLA_ACCESS(3, real_dst, t, n1*bn+n2, c1*bc+c2, N, C) =
              LIBXSMM_VLA_ACCESS(5, real_src, t, n1, c1, n2, c2, nBlocks, cBlocks, bn, bc);
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void matrix_copy_CK_to_KCCK(float *src, float *dst, int C, int K, int bc, int bk)
{
  int k1, k2, c1, c2;
  int kBlocks = K/bk;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(2, float, real_src, src, K);
  LIBXSMM_VLA_DECL(4, float, real_dst, dst, cBlocks, bc, bk);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c1); LIBXSMM_OMP_VAR(c2); LIBXSMM_OMP_VAR(k2);
# pragma omp parallel for private(k1,c1,c2,k2)
#endif
  for (k1 = 0; k1 < kBlocks; k1++) {
    for (c1 = 0; c1 < cBlocks; c1++) {
      for (c2 = 0; c2 < bc; c2++) {
        for (k2 = 0; k2 < bk; k2++) {
          LIBXSMM_VLA_ACCESS(4, real_dst, k1, c1, c2, k2, cBlocks, bc, bk) =
            LIBXSMM_VLA_ACCESS(2, real_src, c1*bc+c2, k1*bk+k2, K);
        }
      }
    }
  }
}

LIBXSMM_INLINE void matrix_copy_CK_to_CKKC(float *src, float *dst, int C, int K, int bc, int bk)
{
  int k1, k2, c1, c2;
  int kBlocks = K/bk;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(2, float, real_src, src, K);
  LIBXSMM_VLA_DECL(4, float, real_dst, dst, kBlocks, bk, bc);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(k1); LIBXSMM_OMP_VAR(c1); LIBXSMM_OMP_VAR(c2); LIBXSMM_OMP_VAR(k2);
# pragma omp parallel for private(k1,c1,c2,k2)
#endif
  for (c1 = 0; c1 < cBlocks; c1++) {
    for (k1 = 0; k1 < kBlocks; k1++) {
      for (k2 = 0; k2 < bk; k2++) {
        for (c2 = 0; c2 < bc; c2++) {
          LIBXSMM_VLA_ACCESS(4, real_dst, c1, k1, k2, c2, kBlocks, bk, bc) =
            LIBXSMM_VLA_ACCESS(2, real_src, c1*bc+c2, k1*bk+k2, K);
        }
      }
    }
  }
}

LIBXSMM_INLINE void matrix_copy_KC_to_KCCK(float *src, float *dst, int C, int K, int bc, int bk)
{
  int k1, k2, c1, c2;
  int kBlocks = K/bk;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(2, float, real_src, src, C);
  LIBXSMM_VLA_DECL(4, float, real_dst, dst, cBlocks, bc, bk);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c1); LIBXSMM_OMP_VAR(c2); LIBXSMM_OMP_VAR(k2);
# pragma omp parallel for private(k1,c1,c2,k2)
#endif
  for (k1 = 0; k1 < kBlocks; k1++) {
    for (c1 = 0; c1 < cBlocks; c1++) {
      for (c2 = 0; c2 < bc; c2++) {
        for (k2 = 0; k2 < bk; k2++) {
          LIBXSMM_VLA_ACCESS(4, real_dst, k1, c1, c2, k2, cBlocks, bc, bk) =
            LIBXSMM_VLA_ACCESS(2, real_src, k1*bk+k2, c1*bc+c2, C);
        }
      }
    }
  }
}

LIBXSMM_INLINE void matrix_copy_KCCK_to_KC(float *src, float *dst, int C, int K, int bc, int bk)
{
  int k1, k2, c1, c2;
  int kBlocks = K/bk;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(2, float, real_dst, dst, C);
  LIBXSMM_VLA_DECL(4, float, real_src, src, cBlocks, bc, bk);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c1); LIBXSMM_OMP_VAR(c2); LIBXSMM_OMP_VAR(k2);
# pragma omp parallel for private(k1,c1,c2,k2)
#endif
  for (k1 = 0; k1 < kBlocks; k1++) {
    for (c1 = 0; c1 < cBlocks; c1++) {
      for (c2 = 0; c2 < bc; c2++) {
        for (k2 = 0; k2 < bk; k2++) {
          LIBXSMM_VLA_ACCESS(2, real_dst, k1*bk+k2, c1*bc+c2, C) =
            LIBXSMM_VLA_ACCESS(4, real_src, k1, c1, c2, k2, cBlocks, bc, bk);
        }
      }
    }
  }
}

LIBXSMM_INLINE void matrix_copy_KCCK_to_CK(float *src, float *dst, int C, int K, int bc, int bk)
{
  int k1, k2, c1, c2;
  int kBlocks = K/bk;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(2, float, real_dst, dst, K);
  LIBXSMM_VLA_DECL(4, float, real_src, src, cBlocks, bc, bk);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c1); LIBXSMM_OMP_VAR(c2); LIBXSMM_OMP_VAR(k2);
# pragma omp parallel for private(k1,c1,c2,k2)
#endif
  for (k1 = 0; k1 < kBlocks; k1++) {
    for (c1 = 0; c1 < cBlocks; c1++) {
      for (c2 = 0; c2 < bc; c2++) {
        for (k2 = 0; k2 < bk; k2++) {
          LIBXSMM_VLA_ACCESS(2, real_dst, c1*bc+c2, k1*bk+k2, K) =
            LIBXSMM_VLA_ACCESS(4, real_src, k1, c1, c2, k2, cBlocks, bc, bk);
        }
      }
    }
  }
}

LIBXSMM_INLINE void matrix_copy_CK_to_KCCK_bf16(libxsmm_bfloat16 *src, libxsmm_bfloat16 *dst, int C, int K, int bc, int bk)
{
  int k1, k2, c1, c2;
  int kBlocks = K/bk;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, real_src, src, K);
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, real_dst, dst, cBlocks, bc/2, bk, 2);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c1); LIBXSMM_OMP_VAR(c2); LIBXSMM_OMP_VAR(k2);
# pragma omp parallel for private(k1,c1,c2,k2)
#endif
  for (k1 = 0; k1 < kBlocks; k1++) {
    for (c1 = 0; c1 < cBlocks; c1++) {
      for (c2 = 0; c2 < bc; c2++) {
        for (k2 = 0; k2 < bk; k2++) {
          LIBXSMM_VLA_ACCESS(5, real_dst, k1, c1, c2/2, k2, c2%2, cBlocks, bc/2, bk, 2) =
            LIBXSMM_VLA_ACCESS(2, real_src, c1*bc+c2, k1*bk+k2, K);
        }
      }
    }
  }
}

LIBXSMM_INLINE void matrix_copy_CK_to_CKKC_bf16(libxsmm_bfloat16 *src, libxsmm_bfloat16 *dst, int C, int K, int bc, int bk)
{
  int k1, k2, c1, c2;
  int kBlocks = K/bk;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, real_src, src, K);
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, real_dst, dst, kBlocks, bk/2, bc, 2);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(k1); LIBXSMM_OMP_VAR(c1); LIBXSMM_OMP_VAR(c2); LIBXSMM_OMP_VAR(k2);
# pragma omp parallel for private(k1,c1,c2,k2)
#endif
  for (c1 = 0; c1 < cBlocks; c1++) {
    for (k1 = 0; k1 < kBlocks; k1++) {
      for (k2 = 0; k2 < bk; k2++) {
        for (c2 = 0; c2 < bc; c2++) {
          LIBXSMM_VLA_ACCESS(5, real_dst, c1, k1, k2/2, c2, k2%2, kBlocks, bk/2, bc, 2) =
            LIBXSMM_VLA_ACCESS(2, real_src, c1*bc+c2, k1*bk+k2, K);
        }
      }
    }
  }
}

LIBXSMM_INLINE void matrix_copy_KC_to_KCCK_bf16(libxsmm_bfloat16 *src, libxsmm_bfloat16 *dst, int C, int K, int bc, int bk)
{
  int k1, k2, c1, c2;
  int kBlocks = K/bk;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, real_src, src, C);
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, real_dst, dst, cBlocks, bc/2, bk, 2);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c1); LIBXSMM_OMP_VAR(c2); LIBXSMM_OMP_VAR(k2);
# pragma omp parallel for private(k1,c1,c2,k2)
#endif
  for (k1 = 0; k1 < kBlocks; k1++) {
    for (c1 = 0; c1 < cBlocks; c1++) {
      for (c2 = 0; c2 < bc; c2++) {
        for (k2 = 0; k2 < bk; k2++) {
          LIBXSMM_VLA_ACCESS(5, real_dst, k1, c1, c2/2, k2, c2%2, cBlocks, bc/2, bk, 2) =
            LIBXSMM_VLA_ACCESS(2, real_src, k1*bk+k2, c1*bc+c2, C);
        }
      }
    }
  }
}

LIBXSMM_INLINE void matrix_copy_KCCK_to_KC_bf16(libxsmm_bfloat16 *src, libxsmm_bfloat16 *dst, int C, int K, int bc, int bk)
{
  int k1, k2, c1, c2;
  int kBlocks = K/bk;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, real_dst, dst, C);
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, real_src, src, cBlocks, bc/2, bk, 2);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c1); LIBXSMM_OMP_VAR(c2); LIBXSMM_OMP_VAR(k2);
# pragma omp parallel for private(k1,c1,c2,k2)
#endif
  for (k1 = 0; k1 < kBlocks; k1++) {
    for (c1 = 0; c1 < cBlocks; c1++) {
      for (c2 = 0; c2 < bc; c2++) {
        for (k2 = 0; k2 < bk; k2++) {
          LIBXSMM_VLA_ACCESS(2, real_dst, k1*bk+k2, c1*bc+c2, C) =
            LIBXSMM_VLA_ACCESS(5, real_src, k1, c1, c2/2, k2, c2%2, cBlocks, bc/2, bk, 2);
        }
      }
    }
  }
}

LIBXSMM_INLINE void matrix_copy_KCCK_to_CK_bf16(libxsmm_bfloat16 *src, libxsmm_bfloat16 *dst, int C, int K, int bc, int bk)
{
  int k1, k2, c1, c2;
  int kBlocks = K/bk;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, real_dst, dst, K);
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, real_src, src, cBlocks, bc/2, bk, 2);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c1); LIBXSMM_OMP_VAR(c2); LIBXSMM_OMP_VAR(k2);
# pragma omp parallel for private(k1,c1,c2,k2)
#endif
  for (k1 = 0; k1 < kBlocks; k1++) {
    for (c1 = 0; c1 < cBlocks; c1++) {
      for (c2 = 0; c2 < bc; c2++) {
        for (k2 = 0; k2 < bk; k2++) {
          LIBXSMM_VLA_ACCESS(2, real_dst, c1*bc+c2, k1*bk+k2, K) =
            LIBXSMM_VLA_ACCESS(5, real_src, k1, c1, c2/2, k2, c2%2, cBlocks, bc/2, bk, 2);
        }
      }
    }
  }
}

LIBXSMM_INLINE void matrix_copy_KCCK_to_CKKC_bf16(libxsmm_bfloat16 *src, libxsmm_bfloat16 *dst, int C, int K, int bc, int bk)
{
  int k1, k2, c1, c2;
  int kBlocks = K/bk;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, real_dst, dst, kBlocks, bk/2, bc, 2);
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, real_src, src, cBlocks, bc/2, bk, 2);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c1); LIBXSMM_OMP_VAR(c2); LIBXSMM_OMP_VAR(k2);
# pragma omp parallel for private(k1,c1,c2,k2)
#endif
  for (k1 = 0; k1 < kBlocks; k1++) {
    for (c1 = 0; c1 < cBlocks; c1++) {
      for (c2 = 0; c2 < bc; c2++) {
        for (k2 = 0; k2 < bk; k2++) {
          LIBXSMM_VLA_ACCESS(5, real_dst, c1, k1, k2/2, c2, k2%2, kBlocks, bk/2, bc, 2) =
          LIBXSMM_VLA_ACCESS(5, real_src, k1, c1, c2/2, k2, c2%2, cBlocks, bc/2, bk, 2);
        }
      }
    }
  }
}

LIBXSMM_INLINE void tensor_copy_NCHW_to_NCHWc(float *src, float *dst, int N, int C, int H, int W, int bc)
{
  int n, h, w, c1, c2;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(4, float, in, src, C, H, W);
  LIBXSMM_VLA_DECL(5, float,out, dst, cBlocks, H, W, bc);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c2); LIBXSMM_OMP_VAR(h); LIBXSMM_OMP_VAR(w);
# pragma omp parallel for private(c1,c2,h,w)
#endif
  for (n = 0; n < N; n++) {
    for (c1 = 0; c1 < cBlocks; c1++) {
      for (c2 = 0; c2 < bc; c2++) {
        for (h = 0; h < H; h++) {
          for (w = 0; w < W; w++) {
            LIBXSMM_VLA_ACCESS(5, out, n, c1,       h, w, c2, cBlocks, H, W, bc) =
            LIBXSMM_VLA_ACCESS(4, in,  n, c1*bc+c2, h, w, C, H, W);
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void tensor_copy_NCHW_to_NCHWc_uint8(unsigned char *src, unsigned char *dst, int N, int C, int H, int W, int bc)
{
  int n, h, w, c1, c2;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(4, unsigned char, in, src, C, H, W);
  LIBXSMM_VLA_DECL(5, unsigned char,out, dst, cBlocks, H, W, bc);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c2); LIBXSMM_OMP_VAR(h); LIBXSMM_OMP_VAR(w);
# pragma omp parallel for private(c1,c2,h,w)
#endif
  for (n = 0; n < N; n++) {
    for (c1 = 0; c1 < cBlocks; c1++) {
      for (c2 = 0; c2 < bc; c2++) {
        for (h = 0; h < H; h++) {
          for (w = 0; w < W; w++) {
            LIBXSMM_VLA_ACCESS(5, out, n, c1,       h, w, c2, cBlocks, H, W, bc) =
            LIBXSMM_VLA_ACCESS(4, in,  n, c1*bc+c2, h, w, C, H, W);
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void tensor_pollute_rim_NCHWc(float *dst, int N, int C, int H, int W, int bc, int pad_h, int pad_w, float polute_val)
{
  int n, h, w, c1, c2;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(5, float,out, dst, cBlocks, H, W, bc);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c2); LIBXSMM_OMP_VAR(h); LIBXSMM_OMP_VAR(w);
# pragma omp parallel for private(c1,c2,h,w)
#endif
  for (n = 0; n < N; n++) {
    for (c1 = 0; c1 < cBlocks; c1++) {
      for (h = 0; h < H; h++) {
        for (w = 0; w < W; w++) {
          for (c2 = 0; c2 < bc; c2++) {
            if ( (h < pad_h) || (h >= H-pad_h)) {
              LIBXSMM_VLA_ACCESS(5, out, n, c1, h, w, c2, cBlocks, H, W, bc) = polute_val;
            } else {
              if ( (w < pad_w) || (w >= W-pad_w)) {
                LIBXSMM_VLA_ACCESS(5, out, n, c1, h, w, c2, cBlocks, H, W, bc) = polute_val;
              }
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void tensor_copy_NCHWc_to_NCHW(float *src, float *dst, int N, int C, int H, int W, int bc)
{
  int n, h, w, c1, c2;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(4, float,out, dst, C, H, W);
  LIBXSMM_VLA_DECL(5, float, in, src, cBlocks, H, W, bc);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c1); LIBXSMM_OMP_VAR(c2); LIBXSMM_OMP_VAR(h); LIBXSMM_OMP_VAR(w);
# pragma omp parallel for private(c1,c2,h,w)
#endif
  for (n = 0; n < N; n++) {
    for (c1 = 0; c1 < cBlocks; c1++) {
      for (c2 = 0; c2 < bc; c2++) {
        for (h = 0; h < H; h++) {
          for (w = 0; w < W; w++) {
            LIBXSMM_VLA_ACCESS(4,out,  n, c1*bc+c2, h, w, C, H, W) =
            LIBXSMM_VLA_ACCESS(5, in, n, c1,       h, w, c2, cBlocks, H, W, bc);
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void tensor_copy_NCHW_to_NCHWc_bf16(libxsmm_bfloat16 *src, libxsmm_bfloat16 *dst, int N, int C, int H, int W, int bc)
{
  int n, h, w, c1, c2;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16,  in, src, C, H, W);
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, out, dst, cBlocks, H, W, bc);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c2); LIBXSMM_OMP_VAR(h); LIBXSMM_OMP_VAR(w);
# pragma omp parallel for private(c1,c2,h,w)
#endif
  for (n = 0; n < N; n++) {
    for (c1 = 0; c1 < cBlocks; c1++) {
      for (c2 = 0; c2 < bc; c2++) {
        for (h = 0; h < H; h++) {
          for (w = 0; w < W; w++) {
            LIBXSMM_VLA_ACCESS(5, out, n, c1,       h, w, c2, cBlocks, H, W, bc) =
            LIBXSMM_VLA_ACCESS(4, in,  n, c1*bc+c2, h, w, C, H, W);
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void tensor_pollute_rim_NCHWc_bf16(libxsmm_bfloat16 *dst, int N, int C, int H, int W, int bc, int pad_h, int pad_w, float polute_val)
{
  int n, h, w, c1, c2;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, out, dst, cBlocks, H, W, bc);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c2); LIBXSMM_OMP_VAR(h); LIBXSMM_OMP_VAR(w);
# pragma omp parallel for private(c1,c2,h,w)
#endif
  for (n = 0; n < N; n++) {
    for (c1 = 0; c1 < cBlocks; c1++) {
      for (h = 0; h < H; h++) {
        for (w = 0; w < W; w++) {
          for (c2 = 0; c2 < bc; c2++) {
            if ( (h < pad_h) || (h >= H-pad_h)) {
              LIBXSMM_VLA_ACCESS(5, out, n, c1, h, w, c2, cBlocks, H, W, bc) = polute_val;
            } else {
              if ( (w < pad_w) || (w >= W-pad_w)) {
                LIBXSMM_VLA_ACCESS(5, out, n, c1, h, w, c2, cBlocks, H, W, bc) = polute_val;
              }
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void tensor_copy_NCHWc_to_NCHW_bf16(libxsmm_bfloat16 *src, libxsmm_bfloat16 *dst, int N, int C, int H, int W, int bc)
{
  int n, h, w, c1, c2;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16, out, dst, C, H, W);
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16,  in, src, cBlocks, H, W, bc);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c1); LIBXSMM_OMP_VAR(c2); LIBXSMM_OMP_VAR(h); LIBXSMM_OMP_VAR(w);
# pragma omp parallel for private(c1,c2,h,w)
#endif
  for (n = 0; n < N; n++) {
    for (c1 = 0; c1 < cBlocks; c1++) {
      for (c2 = 0; c2 < bc; c2++) {
        for (h = 0; h < H; h++) {
          for (w = 0; w < W; w++) {
            LIBXSMM_VLA_ACCESS(4,out,  n, c1*bc+c2, h, w, C, H, W) =
            LIBXSMM_VLA_ACCESS(5, in, n, c1,       h, w, c2, cBlocks, H, W, bc);
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void tensor_copy_KCRS_to_KCRSck(float *src, float *dst, int K, int C, int R, int S, int bc, int bk)
{
  int k1, k2, c1, c2, r, s;
  int cBlocks = C/bc;
  int kBlocks = K/bk;
  LIBXSMM_VLA_DECL(4, float, in, src, C, R, S);
  LIBXSMM_VLA_DECL(6, float,out, dst, cBlocks, R, S, bc, bk);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c1); LIBXSMM_OMP_VAR(c2); LIBXSMM_OMP_VAR(r); LIBXSMM_OMP_VAR(s);
# pragma omp parallel for private(k2,c1,c2,r,s)
#endif
  for (k1 = 0; k1 < kBlocks; k1++) {
    for (k2 = 0; k2 < bk; k2++) {
      for (c1 = 0; c1 < cBlocks; c1++) {
        for (c2 = 0; c2 < bc; c2++) {
          for (r = 0; r < R; r++) {
            for (s = 0; s < S; s++) {
              LIBXSMM_VLA_ACCESS(6, out, k1,     c1,         r, s, c2, k2, cBlocks, R, S, bc, bk) =
              LIBXSMM_VLA_ACCESS(4, in,  k1*bk+k2, c1*bc+c2, r, s, C, R, S);
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void tensor_copy_KCRS_to_KCRSck_bf16(float *src, libxsmm_bfloat16 *dst, int K, int C, int R, int S, int bc, int bk)
{
  int k1, k2, c1, c2, r, s;
  int cBlocks = C/bc;
  int kBlocks = K/bk;
  LIBXSMM_VLA_DECL(4, float, in, src, C, R, S);
  LIBXSMM_VLA_DECL(7, libxsmm_bfloat16, out, dst, cBlocks, R, S, bc/2, bk, 2);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c1); LIBXSMM_OMP_VAR(c2); LIBXSMM_OMP_VAR(r); LIBXSMM_OMP_VAR(s);
# pragma omp parallel for private(k2,c1,c2,r,s)
#endif
  for (k1 = 0; k1 < kBlocks; k1++) {
    for (k2 = 0; k2 < bk; k2++) {
      for (c1 = 0; c1 < cBlocks; c1++) {
        for (c2 = 0; c2 < bc; c2++) {
          for (r = 0; r < R; r++) {
            for (s = 0; s < S; s++) {
              libxsmm_rne_convert_fp32_bf16( &LIBXSMM_VLA_ACCESS(4, in,  k1*bk+k2, c1*bc+c2, r, s, C, R, S),
                                             &LIBXSMM_VLA_ACCESS(7, out, k1,     c1,         r, s, c2/2, k2, c2%2, cBlocks, R, S, bc/2, bk, 2),     1);
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void tensor_copy_KCRSck_to_KCRS(float *src, float *dst, int K, int C, int R, int S, int bc, int bk)
{
  int k1, k2, c1, c2, r, s;
  int cBlocks = C/bc;
  int kBlocks = K/bk;
  LIBXSMM_VLA_DECL(4, float, out, dst, C, R, S);
  LIBXSMM_VLA_DECL(6, float,  in, src, cBlocks, R, S, bc, bk);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c1); LIBXSMM_OMP_VAR(c2); LIBXSMM_OMP_VAR(r); LIBXSMM_OMP_VAR(s);
# pragma omp parallel for private(k2,c1,c2,r,s)
#endif
  for (k1 = 0; k1 < kBlocks; k1++) {
    for (k2 = 0; k2 < bk; k2++) {
      for (c1 = 0; c1 < cBlocks; c1++) {
        for (c2 = 0; c2 < bc; c2++) {
          for (r = 0; r < R; r++) {
            for (s = 0; s < S; s++) {
              LIBXSMM_VLA_ACCESS(4, out,  k1*bk+k2, c1*bc+c2, r, s, C, R, S) =
                LIBXSMM_VLA_ACCESS(6, in, k1,     c1,         r, s, c2, k2, cBlocks, R, S, bc, bk);
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void tensor_copy_KCRSck_vnni_to_norm_f32(libxsmm_bfloat16 *src, float *dst, int K, int C, int R, int S, int bc, int bk)
{
  int k1, k2, c1, c2, r, s;
  int cBlocks = C/bc;
  int kBlocks = K/bk;
  LIBXSMM_VLA_DECL(7, libxsmm_bfloat16, in, src, cBlocks, R, S, bc/2, bk, 2);
  LIBXSMM_VLA_DECL(6, float, out, dst, cBlocks, R, S, bc, bk);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c1); LIBXSMM_OMP_VAR(c2); LIBXSMM_OMP_VAR(r); LIBXSMM_OMP_VAR(s);
# pragma omp parallel for private(k2,c1,c2,r,s)
#endif
  for (k1 = 0; k1 < kBlocks; k1++) {
    for (k2 = 0; k2 < bk; k2++) {
      for (c1 = 0; c1 < cBlocks; c1++) {
        for (c2 = 0; c2 < bc; c2++) {
          for (r = 0; r < R; r++) {
            for (s = 0; s < S; s++) {
              float val;
              libxsmm_convert_bf16_f32( &LIBXSMM_VLA_ACCESS(7, in, k1, c1, r, s, c2/2, k2, c2%2, cBlocks, R, S, bc/2, bk, 2), &val, 1 );
              LIBXSMM_VLA_ACCESS(6, out, k1, c1, r, s, c2, k2, cBlocks, R, S, bc, bk) = val;
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void tensor_transpose_KCRSck_to_CKRSkc(float *src, float *dst, int K, int C, int R, int S, int bc, int bk)
{
  int k1, k2, c1, c2, r, s;
  int cBlocks = C/bc;
  int kBlocks = K/bk;
  LIBXSMM_VLA_DECL(6, float, out, dst, kBlocks, R, S, bk, bc);
  LIBXSMM_VLA_DECL(6, float, in , src, cBlocks, R, S, bc, bk);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c1); LIBXSMM_OMP_VAR(c2); LIBXSMM_OMP_VAR(r); LIBXSMM_OMP_VAR(s);
# pragma omp parallel for private(k2,c1,c2,r,s)
#endif
  for (k1 = 0; k1 < kBlocks; k1++) {
    for (k2 = 0; k2 < bk; k2++) {
      for (c1 = 0; c1 < cBlocks; c1++) {
        for (c2 = 0; c2 < bc; c2++) {
          for (r = 0; r < R; r++) {
            for (s = 0; s < S; s++) {
              LIBXSMM_VLA_ACCESS(6, out, c1, k1, R-1-r, S-1-s, k2, c2, kBlocks, R, S, bk, bc) =
              LIBXSMM_VLA_ACCESS(6,  in, k1, c1, r, s, c2, k2, cBlocks, R, S, bc, bk);
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void tensor_transpose_KCRSck_to_CKRSkc_bf16(float *src, libxsmm_bfloat16 *dst, int K, int C, int R, int S, int bc, int bk)
{
  int k1, k2, c1, c2, r, s;
  int cBlocks = C/bc;
  int kBlocks = K/bk;
  LIBXSMM_VLA_DECL(7, libxsmm_bfloat16, out, dst, kBlocks, R, S, bk/2, bc, 2);
  LIBXSMM_VLA_DECL(6, float, in , src, cBlocks, R, S, bc, bk);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c1); LIBXSMM_OMP_VAR(c2); LIBXSMM_OMP_VAR(r); LIBXSMM_OMP_VAR(s);
# pragma omp parallel for private(k2,c1,c2,r,s)
#endif
  for (k1 = 0; k1 < kBlocks; k1++) {
    for (k2 = 0; k2 < bk; k2++) {
      for (c1 = 0; c1 < cBlocks; c1++) {
        for (c2 = 0; c2 < bc; c2++) {
          for (r = 0; r < R; r++) {
            for (s = 0; s < S; s++) {
              libxsmm_rne_convert_fp32_bf16( &LIBXSMM_VLA_ACCESS(6,  in, k1, c1, r, s, c2, k2, cBlocks, R, S, bc, bk),
                                             &LIBXSMM_VLA_ACCESS(7, out, c1, k1, R-1-r, S-1-s, k2/2, c2, k2%2, kBlocks, R, S, bk/2, bc, 2),     1);
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void matrix_add(int size, float *a, float *b, float *c)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}

LIBXSMM_INLINE void matrix_eltwise_mult(int size, float *a, float *b, float *c)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    c[i] = a[i] * b[i];
  }
}

LIBXSMM_INLINE void matrix_eltwise_fma(int size, float *a, float *b, float *c)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    c[i] += a[i] * b[i];
  }
}

LIBXSMM_INLINE void matrix_eltwise_mult_ld_a(int m, int n, int ld, float *a, float *b, float *c)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < m*n; i++) {
    int row = i / m;
    int col = i % m;
    c[i] = a[row*ld + col] * b[i];
  }
}

LIBXSMM_INLINE void matrix_eltwise_mult_ld_ab(int m, int n, int ld, float *a, float *b, float *c)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < m*n; i++) {
    int row = i / m;
    int col = i % m;
    c[i] = a[row*ld + col] * b[row*ld + col];
  }
}

LIBXSMM_INLINE void matrix_eltwise_mult_ld_c(int m, int n, int ld, float *a, float *b, float *c)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < m*n; i++) {
    int row = i / m;
    int col = i % m;
    c[row*ld + col] = a[i] * b[i];
  }
}

LIBXSMM_INLINE void matrix_sigmoid(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    const float exp_value = (float)exp((double) -src[i]);
    dst[i] = 1.0f / (1.0f + exp_value);
  }
}

LIBXSMM_INLINE void matrix_sigmoid_ld(int m, int n, int ld, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < m*n; i++) {
    int row = i / m;
    int col = i % m;
    const float exp_value = (float)exp((double) -src[row*ld + col]);
    dst[row*ld + col] = 1.0f / (1.0f + exp_value);
  }
}

LIBXSMM_INLINE void matrix_tanh(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = (float)tanh((double)src[i]);
  }
}

LIBXSMM_INLINE void matrix_tanh_ld(int m, int n, int ld, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < m*n; i++) {
    int row = i / m;
    int col = i % m;
    dst[row*ld + col] = (float)tanh((double)src[row*ld + col]);
  }
}

LIBXSMM_INLINE void matrix_relu(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = (src[i] > 0.0f) ? src[i] : 0.0f;
  }
}

LIBXSMM_INLINE void matrix_sigmoid_inverse(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    const float exp_value = (float)exp((double) -src[i]);
    const float sig_exp = 1.0f / (1.0f + exp_value);
    dst[i] = (1.0f - sig_exp)*sig_exp;
  }
}

LIBXSMM_INLINE void matrix_tanh_inverse(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    const float tanh_value = (float)tanh((double)src[i]);
    dst[i] = 1.0f - (tanh_value * tanh_value);
  }
}

LIBXSMM_INLINE void matrix_relu_inverse(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = (src[i] > 0.0f) ? 1.0f : 0.0f;
  }
}

LIBXSMM_INLINE void matrix_transpose(int rows, int cols, float *src, float *dst)
{
#if defined(_OPENMP)
  libxsmm_otrans(dst, src, sizeof(float), cols, rows, cols/*ldi*/, rows/*ldo*/);
#else
  LIBXSMM_USEOMP(libxsmm_otrans)(dst, src, sizeof(float), cols, rows, cols/*ldi*/, rows/*ldo*/);
#endif
}

LIBXSMM_INLINE void matrix_copy(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = src[i];
  }
}

LIBXSMM_INLINE void matrix_copy_f32_bf16(int size, float *src, libxsmm_bfloat16 *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    libxsmm_bfloat16_hp t;
    t.f = src[i];
    dst[i] = t.i[1];
  }
}

LIBXSMM_INLINE void matrix_copy_bf16_f32(int size, libxsmm_bfloat16 *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    libxsmm_bfloat16_hp t;
    t.i[1] = src[i];
    t.i[0] = 0;
    dst[i] = t.f;
  }
}

LIBXSMM_INLINE void matrix_copy_ld(int m, int n, int ld, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < m*n; i++) {
    int row = i / m;
    int col = i % m;
    dst[i] = src[row*ld + col];
  }
}

LIBXSMM_INLINE void matrix_copy_bias(int m, int n, int ld, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < m*n; i++) {
    int row = i / m;
    int col = i % m;
    dst[row*ld + col] = src[col];
  }
}

LIBXSMM_INLINE void matrix_copy_forget_bias(int m, int n, int ld, float *src, float *dst, float forget_bias)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < m*n; i++) {
    int row = i / m;
    int col = i % m;
    dst[row*ld + col] = src[col] + forget_bias;
  }
}

LIBXSMM_INLINE void matrix_complement(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = 1.0f - src[i];
  }
}

LIBXSMM_INLINE void matrix_complement_ld(int m, int n, int ld, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < m*n; i++) {
    int row = i / m;
    int col = i % m;
    dst[i] = 1.0f - src[row*ld + col];
  }
}


LIBXSMM_INLINE void matrix_complement_square(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = 1.0f - (src[i] * src[i]);
  }
}

LIBXSMM_INLINE void matrix_complement_square_ld(int m, int n, int ld, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < m*n; i++) {
    int row = i / m;
    int col = i % m;
    dst[i] = 1.0f - (src[row*ld + col] * src[row*ld + col]);
  }
}

LIBXSMM_INLINE void convert_ck_c4k_offset(int C, int K, int offset, float *src, float *dst)
{
  /* offsets: i--0, c--1, f--2, o--3 */
  int x, y;
#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(x);
# pragma omp parallel for private(x, y)
#endif
  for (y = 0; y < C; y++) {
    for (x = 0; x < K; x++) {
      dst[y*4*K + offset*K + x] = src[y*K + x];
    }
  }
}

LIBXSMM_INLINE void convert_ck_c4k(int C, int K, float *src, float *dst)
{
  convert_ck_c4k_offset(C, K, 0, src, dst);
}

LIBXSMM_INLINE void convert_ck_f32_to_c4k_bf16(int C, int K, float *src, libxsmm_bfloat16 *dst)
{
  int x, y;
#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(x);
# pragma omp parallel for private(x, y)
#endif
  for (y = 0; y < C; y++) {
    for (x = 0; x < K; x++) {
      libxsmm_bfloat16_hp t;
      t.f = src[y*K + x];
      dst[y*4*K + x] = t.i[1];
    }
  }
}

LIBXSMM_INLINE void convert_c4k_4ck(int C, int K, float *src, float *dst)
{
  /* offsets: i--0, c--1, f--2, o--3 */
  int x, y, offset;
#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(x); LIBXSMM_OMP_VAR(y);
# pragma omp parallel for private(x, y, offset)
#endif
  for (offset = 0; offset < 4; offset++) {
    for (y = 0; y < C; y++) {
      for (x = 0; x < K; x++) {
        dst[offset*C*K + y*K + x] = src[y*4*K + offset*K + x];
      }
    }
  }
}

LIBXSMM_INLINE void convert_ck_c3k(int C, int K, float *src, float *dst)
{
  int x, y;
#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(x);
# pragma omp parallel for private(x, y)
#endif
  for (y = 0; y < C; y++) {
    for (x = 0; x < K; x++) {
      dst[y*3*K + x] = src[y*K + x];
    }
  }
}

LIBXSMM_INLINE void convert_nk_nck(int N, int K, int CK, float *src, float *dst)
{
  int x, y;
#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(x);
# pragma omp parallel for private(x, y)
#endif
  for (y = 0; y < N; y++) {
    for (x = 0; x < K; x++) {
      dst[y*CK + x] = src[y*K + x];
    }
  }
}

LIBXSMM_INLINE void naive_conv_fp(naive_conv_t* param, const float* input, float* output, const float* filter, const float* bias)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ifh       = param->ifh;
  int ifw       = param->ifw;
  int ofh       = param->ofh;
  int ofw       = param->ofw;
  int pad_h     = param->pad_h;
  int pad_w     = param->pad_w;
  int pad_h_in  = param->pad_h_in;
  int pad_w_in  = param->pad_w_in;
  int pad_h_out = param->pad_h_out;
  int pad_w_out = param->pad_w_out;
  int kh        = param->kh;
  int kw        = param->kw;
  int stride_h  = param->stride_h;
  int stride_w  = param->stride_w;
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

  LIBXSMM_VLA_DECL(4,       float, output_t, output + (pad_h_out * ofwp + pad_w_out), nOfm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4, const float,  input_t,  input + (pad_h_in * ifwp + pad_w_in), nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4, const float, filter_t, filter, nIfm, kh, kw);

#if defined(USE_FUSED_BIAS) || defined(USE_FUSED_BIAS_RELU)
#if defined(_OPENMP)
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ofm = 0; ofm < nOfm; ++ofm) {
      for (oj = 0; oj < ofh; ++oj) {
        for (oi = 0; oi < ofw; ++oi) {
          LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) = bias[ofm];
        }
      }
    }
  }
#else
  LIBXSMM_UNUSED(bias);
#endif

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(ofm); LIBXSMM_OMP_VAR(oj);  LIBXSMM_OMP_VAR(oi);
  LIBXSMM_OMP_VAR(ifm); LIBXSMM_OMP_VAR(ij);  LIBXSMM_OMP_VAR(ii);  LIBXSMM_OMP_VAR(kj);  LIBXSMM_OMP_VAR(ki);
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ofm = 0; ofm < nOfm; ++ofm) {
      for (ifm = 0; ifm < nIfm; ++ifm) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h - pad_h;
          for (oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w - pad_w;
            for (kj = 0; kj < kh; ++kj) {
              if (ij+kj < 0 || ij+kj >= ifh) continue;
              for (ki = 0; ki < kw; ++ki) {
                if (ii+ki < 0 || ii+ki >= ifw) continue;
                LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) +=
                  LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp)
                  * LIBXSMM_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw);
              }
            }
          }
        }
      }
#if defined(USE_FUSED_RELU) || defined(USE_FUSED_BIAS_RELU)
      for (oj = 0; oj < ofh; ++oj) {
        for (oi = 0; oi < ofw; ++oi) {
          LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) =
            (LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) < 0.0f) ? 0.0f : LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp);
        }
      }
#endif
    }
  }
}

LIBXSMM_INLINE void naive_fused_conv_fp(naive_conv_t* param, const float* input, float* output, const float* filter, const float* bias, libxsmm_blasint fuse_type)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ifh       = param->ifh;
  int ifw       = param->ifw;
  int ofh       = param->ofh;
  int ofw       = param->ofw;
  int pad_h     = param->pad_h;
  int pad_w     = param->pad_w;
  int pad_h_in  = param->pad_h_in;
  int pad_w_in  = param->pad_w_in;
  int pad_h_out = param->pad_h_out;
  int pad_w_out = param->pad_w_out;
  int kh        = param->kh;
  int kw        = param->kw;
  int stride_h  = param->stride_h;
  int stride_w  = param->stride_w;
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

  LIBXSMM_VLA_DECL(4,       float, output_t, output + (pad_h_out * ofwp + pad_w_out), nOfm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4, const float,  input_t,  input + (pad_h_in * ifwp + pad_w_in), nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4, const float, filter_t, filter, nIfm, kh, kw);


  if (fuse_type == 1 || fuse_type == 3) {
#if defined(_OPENMP)
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
    for (img = 0; img < nImg; ++img) {
      for (ofm = 0; ofm < nOfm; ++ofm) {
        for (oj = 0; oj < ofh; ++oj) {
          for (oi = 0; oi < ofw; ++oi) {
            LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) += bias[ofm];
          }
        }
      }
    }
  }


#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(ofm); LIBXSMM_OMP_VAR(oj);  LIBXSMM_OMP_VAR(oi);
  LIBXSMM_OMP_VAR(ifm); LIBXSMM_OMP_VAR(ij);  LIBXSMM_OMP_VAR(ii);  LIBXSMM_OMP_VAR(kj);  LIBXSMM_OMP_VAR(ki);
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ofm = 0; ofm < nOfm; ++ofm) {
      for (ifm = 0; ifm < nIfm; ++ifm) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h - pad_h;
          for (oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w - pad_w;
            for (kj = 0; kj < kh; ++kj) {
              if (ij+kj < 0 || ij+kj >= ifh) continue;
              for (ki = 0; ki < kw; ++ki) {
                if (ii+ki < 0 || ii+ki >= ifw) continue;
                LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) +=
                  LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp)
                  * LIBXSMM_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw);
              }
            }
          }
        }
      }
      if (fuse_type == 2 || fuse_type == 3) {
        for (oj = 0; oj < ofh; ++oj) {
          for (oi = 0; oi < ofw; ++oi) {
            LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) =
              (LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) < 0.0f) ? 0.0f : LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp);
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_conv_bp(naive_conv_t* param, float* input, const float* output, const float* filter, const float* naive_input_save)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ifh       = param->ifh;
  int ifw       = param->ifw;
  int ofh       = param->ofh;
  int ofw       = param->ofw;
  int pad_h     = param->pad_h;
  int pad_w     = param->pad_w;
  int pad_h_in  = param->pad_h_in;
  int pad_w_in  = param->pad_w_in;
  int pad_h_out = param->pad_h_out;
  int pad_w_out = param->pad_w_out;
  int kh        = param->kh;
  int kw        = param->kw;
  int stride_h  = param->stride_h;
  int stride_w  = param->stride_w;
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

  LIBXSMM_VLA_DECL(4, const float, output_t, output + (pad_h_out * ofwp + pad_w_out), nOfm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4,       float,  input_t,  input + (pad_h_in * ifwp + pad_w_in), nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4, const float, filter_t, filter, nIfm, kh, kw);
#if (defined(USE_FUSED_RELU_BWD) || defined(USE_FUSED_BATCH_STATS_BWD))
  LIBXSMM_VLA_DECL(4, const float, naive_input_t, naive_input_save + (pad_h_in * ifwp + pad_w_in), nIfm, ifhp, ifwp);
#else
  LIBXSMM_UNUSED(naive_input_save);
#endif

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(ofm); LIBXSMM_OMP_VAR(oj);  LIBXSMM_OMP_VAR(oi);
  LIBXSMM_OMP_VAR(ifm); LIBXSMM_OMP_VAR(ij);  LIBXSMM_OMP_VAR(ii);  LIBXSMM_OMP_VAR(kj);  LIBXSMM_OMP_VAR(ki);
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ifm = 0; ifm < nIfm; ++ifm) {
      for (ofm = 0; ofm < nOfm; ++ofm) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h - pad_h;
          for (oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w - pad_w;
            for (kj = 0; kj < kh; ++kj) {
              if (ij+kj < 0 || ij+kj >= ifh) continue;
              for (ki = 0; ki < kw; ++ki) {
                if (ii+ki < 0 || ii+ki >= ifw) continue;
                LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp) +=
                  LIBXSMM_VLA_ACCESS(4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp)
                  * LIBXSMM_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw);
              }
            }
          }
        }
      }
#if (defined(USE_FUSED_RELU_BWD) || defined(USE_FUSED_BATCH_STATS_BWD))
      for (ij = 0; ij < ifh; ij++) {
        for (ii = 0; ii < ifw; ii++) {
          if ( LIBXSMM_VLA_ACCESS(4,  naive_input_t, img, ifm, ij, ii , nIfm, ifhp, ifwp) == 0.0 ) {
            LIBXSMM_VLA_ACCESS(4, input_t, img, ifm, ij, ii , nIfm, ifhp, ifwp) = 0.0;
          }
        }
      }
#endif
    }
  }
}

LIBXSMM_INLINE void naive_conv_wu(naive_conv_t* param, const float* input, const float* output, float* filter)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ifh       = param->ifh;
  int ifw       = param->ifw;
  int ofh       = param->ofh;
  int ofw       = param->ofw;
  int pad_h     = param->pad_h;
  int pad_w     = param->pad_w;
  int pad_h_in  = param->pad_h_in;
  int pad_w_in  = param->pad_w_in;
  int pad_h_out = param->pad_h_out;
  int pad_w_out = param->pad_w_out;
  int kh        = param->kh;
  int kw        = param->kw;
  int stride_h  = param->stride_h;
  int stride_w  = param->stride_w;
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

  LIBXSMM_VLA_DECL(4, const float, output_t, output + (pad_h_out * ofwp + pad_w_out), nOfm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4, const float,  input_t,  input + (pad_h_in * ifwp + pad_w_in), nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4,       float, filter_t, filter, nIfm, kh, kw);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(ofm); LIBXSMM_OMP_VAR(oj);  LIBXSMM_OMP_VAR(oi);
  LIBXSMM_OMP_VAR(ifm); LIBXSMM_OMP_VAR(ij);  LIBXSMM_OMP_VAR(ii);  LIBXSMM_OMP_VAR(kj);  LIBXSMM_OMP_VAR(ki);
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (ofm = 0; ofm < nOfm; ++ofm) {
    for (ifm = 0; ifm < nIfm; ++ifm) {
      for (img = 0; img < nImg; ++img) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h - pad_h;
          for (oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w - pad_w;
            for (kj = 0; kj < kh; ++kj) {
              if (ij+kj < 0 || ij+kj >= ifh) continue;
              for (ki = 0; ki < kw; ++ki) {
                if (ii+ki < 0 || ii+ki >= ifw) continue;
                LIBXSMM_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw) +=
                  LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp)
                  * LIBXSMM_VLA_ACCESS(4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp);
              }
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_conv_fp_int16fp32(naive_conv_t* param, const short* input, float* output, const short* filter)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ifh       = param->ifh;
  int ifw       = param->ifw;
  int ofh       = param->ofh;
  int ofw       = param->ofw;
  int pad_h     = param->pad_h;
  int pad_w     = param->pad_w;
  int pad_h_in  = param->pad_h_in;
  int pad_w_in  = param->pad_w_in;
  int pad_h_out = param->pad_h_out;
  int pad_w_out = param->pad_w_out;
  int kh        = param->kh;
  int kw        = param->kw;
  int stride_h  = param->stride_h;
  int stride_w  = param->stride_w;
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

  LIBXSMM_VLA_DECL(4,       float,     output_t, output + (pad_w_out * ofwp + pad_h_out), nOfm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4, const short,      input_t,  input + (pad_w_in * ifwp + pad_h_in), nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4, const short,     filter_t, filter, nIfm, kh, kw);


#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(ofm); LIBXSMM_OMP_VAR(oj);  LIBXSMM_OMP_VAR(oi);
  LIBXSMM_OMP_VAR(ifm); LIBXSMM_OMP_VAR(ij);  LIBXSMM_OMP_VAR(ii);  LIBXSMM_OMP_VAR(kj);  LIBXSMM_OMP_VAR(ki);
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ofm = 0; ofm < nOfm; ++ofm) {
      for (ifm = 0; ifm < nIfm; ++ifm) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h - pad_h;
          for (oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w - pad_w;
            for (kj = 0; kj < kh; ++kj) {
              if (ij+kj < 0 || ij+kj >= ifh) continue;
              for (ki = 0; ki < kw; ++ki) {
                if (ii+ki < 0 || ii+ki >= ifw) continue;
                LIBXSMM_VLA_ACCESS(4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) +=
                  (1.f * LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp))
                * (1.f * LIBXSMM_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw));
              }
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_conv_fp_int16int32(naive_conv_t* param, const short* input, int* output, const short* filter)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ifh       = param->ifh;
  int ifw       = param->ifw;
  int ofh       = param->ofh;
  int ofw       = param->ofw;
  int pad_h     = param->pad_h;
  int pad_w     = param->pad_w;
  int pad_h_in  = param->pad_h_in;
  int pad_w_in  = param->pad_w_in;
  int pad_h_out = param->pad_h_out;
  int pad_w_out = param->pad_w_out;
  int kh        = param->kh;
  int kw        = param->kw;
  int stride_h  = param->stride_h;
  int stride_w  = param->stride_w;
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

  LIBXSMM_VLA_DECL(4,         int,     output_t, output + (pad_w_out * ofwp + pad_h_out), nOfm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4, const short,      input_t,  input + (pad_w_in * ifwp + pad_h_in), nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4, const short,     filter_t, filter, nIfm, kh, kw);


#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(ofm); LIBXSMM_OMP_VAR(oj);  LIBXSMM_OMP_VAR(oi);
  LIBXSMM_OMP_VAR(ifm); LIBXSMM_OMP_VAR(ij);  LIBXSMM_OMP_VAR(ii);  LIBXSMM_OMP_VAR(kj);  LIBXSMM_OMP_VAR(ki);
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ofm = 0; ofm < nOfm; ++ofm) {
      for (ifm = 0; ifm < nIfm; ++ifm) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h - pad_h;
          for (oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w - pad_w;
            for (kj = 0; kj < kh; ++kj) {
              if (ij+kj < 0 || ij+kj >= ifh) continue;
              for (ki = 0; ki < kw; ++ki) {
                if (ii+ki < 0 || ii+ki >= ifw) continue;
                LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) += (int)
                 ( (int)LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp))
                * ( (int)  LIBXSMM_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw));
              }
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_conv_fp_int8int32(naive_conv_t* param, const unsigned char* input, int* output, const char* filter)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ifh       = param->ifh;
  int ifw       = param->ifw;
  int ofh       = param->ofh;
  int ofw       = param->ofw;
  int pad_h     = param->pad_h;
  int pad_w     = param->pad_w;
  int pad_h_in  = param->pad_h_in;
  int pad_w_in  = param->pad_w_in;
  int pad_h_out = param->pad_h_out;
  int pad_w_out = param->pad_w_out;
  int kh        = param->kh;
  int kw        = param->kw;
  int stride_h  = param->stride_h;
  int stride_w  = param->stride_w;
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

  LIBXSMM_VLA_DECL(4,         int,     output_t, output + (pad_w_out * ofwp + pad_h_out), nOfm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4, const unsigned char,      input_t,  input + (pad_w_in * ifwp + pad_h_in), nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4, const char,     filter_t, filter, nIfm, kh, kw);


#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(ofm); LIBXSMM_OMP_VAR(oj);  LIBXSMM_OMP_VAR(oi);
  LIBXSMM_OMP_VAR(ifm); LIBXSMM_OMP_VAR(ij);  LIBXSMM_OMP_VAR(ii);  LIBXSMM_OMP_VAR(kj);  LIBXSMM_OMP_VAR(ki);
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ofm = 0; ofm < nOfm; ++ofm) {
      for (ifm = 0; ifm < nIfm; ++ifm) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h - pad_h;
          for (oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w - pad_w;
            for (kj = 0; kj < kh; ++kj) {
              if (ij+kj < 0 || ij+kj >= ifh) continue;
              for (ki = 0; ki < kw; ++ki) {
                if (ii+ki < 0 || ii+ki >= ifw) continue;
                LIBXSMM_VLA_ACCESS(4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) += (int)
                LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp)
                * LIBXSMM_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw);
              }
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_fullyconnected_fp(naive_fullyconnected_t* param, const float* input_ptr, float* output_ptr, const float* filter_ptr)
{
  const int nImg = param->N;
  const int nIFm = param->C;
  const int nOFm = param->K;

  int img, ifm, ofm;

  LIBXSMM_VLA_DECL(2, const float, input,  input_ptr,  nIFm);
  LIBXSMM_VLA_DECL(2, const float, filter, filter_ptr, nIFm);
  LIBXSMM_VLA_DECL(2,       float, output, output_ptr, nOFm);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(ifm); LIBXSMM_OMP_VAR(ofm);
# pragma omp parallel for private(img, ofm, ifm)
#endif
  for (ofm = 0; ofm < nOFm; ++ofm) {
    for(img = 0; img < nImg; ++img) {
      LIBXSMM_VLA_ACCESS(2, output, img, ofm, nOFm) = (float)0;
      for (ifm = 0; ifm < nIFm; ++ifm) {
        LIBXSMM_VLA_ACCESS(2, output, img, ofm, nOFm) +=
          LIBXSMM_VLA_ACCESS(2, filter, ofm, ifm, nIFm) * LIBXSMM_VLA_ACCESS(2, input, img, ifm, nIFm);
      }
    }
  }
}

LIBXSMM_INLINE void naive_fullyconnected_bp(naive_fullyconnected_t* param, float* delinput_ptr, const float* deloutput_ptr, const float* filter_ptr)
{
  const int nImg = param->N;
  const int nIFm = param->C;
  const int nOFm = param->K;

  int img, ifm, ofm;

  LIBXSMM_VLA_DECL(2,       float,  dinput,  delinput_ptr, nIFm);
  LIBXSMM_VLA_DECL(2, const float,  filter,    filter_ptr, nIFm);
  LIBXSMM_VLA_DECL(2, const float, doutput, deloutput_ptr, nOFm);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(ofm); LIBXSMM_OMP_VAR(ifm);
# pragma omp parallel for private(img, ofm, ifm)
#endif
  for (ifm = 0; ifm < nIFm; ++ifm) {
    for(img = 0; img < nImg; ++img) {
      LIBXSMM_VLA_ACCESS(2, dinput, img, ifm, nIFm) = (float)0;
      for (ofm = 0; ofm < nOFm; ++ofm) {
        LIBXSMM_VLA_ACCESS(2, dinput, img, ifm, nIFm) +=
          LIBXSMM_VLA_ACCESS(2, filter, ofm, ifm, nIFm) * LIBXSMM_VLA_ACCESS(2, doutput, img, ofm, nOFm);
      }
    }
  }
}

LIBXSMM_INLINE void naive_fullyconnected_fused_fp(naive_fullyconnected_t* param, const float* input_ptr, float* output_ptr, const float* filter_ptr, const float* bias_ptr)
{
  const int nImg = param->N;
  const int nIFm = param->C;
  const int nOFm = param->K;

  int img, ifm, ofm;

  LIBXSMM_VLA_DECL(2, const float, input,  input_ptr,  nIFm);
  LIBXSMM_VLA_DECL(2, const float, filter, filter_ptr, nIFm);
  LIBXSMM_VLA_DECL(2,       float, output, output_ptr, nOFm);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(ifm); LIBXSMM_OMP_VAR(ofm);
# pragma omp parallel for private(img, ofm, ifm)
#endif
  for (ofm = 0; ofm < nOFm; ++ofm) {
    for(img = 0; img < nImg; ++img) {
      float accum = 0.f;
      for (ifm = 0; ifm < nIFm; ++ifm) {
        accum += LIBXSMM_VLA_ACCESS(2, filter, ofm, ifm, nIFm) * LIBXSMM_VLA_ACCESS(2, input, img, ifm, nIFm);
      }
      if ( param->fuse_type == 1 ) {
        accum += bias_ptr[ofm];
      } else if ( param->fuse_type == 2 ) {
        accum = ( accum > 0 ) ? accum : 0;
      } else if ( param->fuse_type == 4 ) {
        accum = ( accum > 0 ) ? accum : 0;
      } else if ( param->fuse_type == 8 ) {
        accum = ((float)tanh((double)accum/2.0)+1.0f)/2.0f;
      } else if ( param->fuse_type == 3 ) {
        accum += bias_ptr[ofm];
        accum = ( accum > 0 ) ? accum : 0;
      } else if ( param->fuse_type == 5 ) {
        accum += bias_ptr[ofm];
        accum = ( accum > 0 ) ? accum : 0;
      } else if ( param->fuse_type == 9 ) {
        accum += bias_ptr[ofm];
        accum = ((float)tanh((double)accum/2.0)+1.0f)/2.0f;
      }
      LIBXSMM_VLA_ACCESS(2, output, img, ofm, nOFm) = accum;
    }
  }
}

LIBXSMM_INLINE void naive_fullyconnected_fused_bp(naive_fullyconnected_t* param, float* delinput_ptr, float* deloutput_ptr, const float* filter_ptr, float* delbias_ptr, const float* output_ptr)
{
  const int nImg = param->N;
  const int nIFm = param->C;
  const int nOFm = param->K;

  int img, ifm, ofm;

  LIBXSMM_VLA_DECL(2,       float,  dinput,  delinput_ptr, nIFm);
  LIBXSMM_VLA_DECL(2, const float,  filter,    filter_ptr, nIFm);
  LIBXSMM_VLA_DECL(2,       float, doutput, deloutput_ptr, nOFm);
  LIBXSMM_VLA_DECL(2, const float,  output,    output_ptr, nOFm);

  if ( param->fuse_type != 0 ) {
#if defined(_OPENMP)
    LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(ofm);
# pragma omp parallel for private(img, ofm)
#endif
    for (ofm = 0; ofm < nOFm; ++ofm) {
      float dbias = 0.0f;
      for(img = 0; img < nImg; ++img) {
        if ( param->fuse_type == 1 ) {
          dbias += LIBXSMM_VLA_ACCESS(2, doutput, img, ofm, nOFm);
        } else if ( param->fuse_type == 2 ) {
          LIBXSMM_VLA_ACCESS(2, doutput, img, ofm, nOFm) = ( LIBXSMM_VLA_ACCESS(2, output, img, ofm, nOFm) > 0 ) ? LIBXSMM_VLA_ACCESS(2, doutput, img, ofm, nOFm) : 0;
        } else if ( param->fuse_type == 4 ) {
          LIBXSMM_VLA_ACCESS(2, doutput, img, ofm, nOFm) = ( LIBXSMM_VLA_ACCESS(2, output, img, ofm, nOFm) > 0 ) ? LIBXSMM_VLA_ACCESS(2, doutput, img, ofm, nOFm) : 0;
        } else if ( param->fuse_type == 8 ) {
          LIBXSMM_VLA_ACCESS(2, doutput, img, ofm, nOFm) = LIBXSMM_VLA_ACCESS(2, doutput, img, ofm, nOFm)*(1.0f-LIBXSMM_VLA_ACCESS(2, doutput, img, ofm, nOFm));
        } else if ( param->fuse_type == 3 ) {
          LIBXSMM_VLA_ACCESS(2, doutput, img, ofm, nOFm) = ( LIBXSMM_VLA_ACCESS(2, output, img, ofm, nOFm) > 0 ) ? LIBXSMM_VLA_ACCESS(2, doutput, img, ofm, nOFm) : 0;
          dbias += LIBXSMM_VLA_ACCESS(2, doutput, img, ofm, nOFm);
        } else if ( param->fuse_type == 5 ) {
          LIBXSMM_VLA_ACCESS(2, doutput, img, ofm, nOFm) = ( LIBXSMM_VLA_ACCESS(2, output, img, ofm, nOFm) > 0 ) ? LIBXSMM_VLA_ACCESS(2, doutput, img, ofm, nOFm) : 0;
          dbias += LIBXSMM_VLA_ACCESS(2, doutput, img, ofm, nOFm);
        } else if ( param->fuse_type == 9 ) {
          LIBXSMM_VLA_ACCESS(2, doutput, img, ofm, nOFm) = LIBXSMM_VLA_ACCESS(2, doutput, img, ofm, nOFm)*(1.0f-LIBXSMM_VLA_ACCESS(2, doutput, img, ofm, nOFm));
          dbias += LIBXSMM_VLA_ACCESS(2, doutput, img, ofm, nOFm);
        }
      }
      delbias_ptr[ofm] = dbias;
    }
  }

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(ofm); LIBXSMM_OMP_VAR(ifm);
# pragma omp parallel for private(img, ofm, ifm)
#endif
  for (ifm = 0; ifm < nIFm; ++ifm) {
    for(img = 0; img < nImg; ++img) {
      LIBXSMM_VLA_ACCESS(2, dinput, img, ifm, nIFm) = (float)0;
      for (ofm = 0; ofm < nOFm; ++ofm) {
        LIBXSMM_VLA_ACCESS(2, dinput, img, ifm, nIFm) +=
          LIBXSMM_VLA_ACCESS(2, filter, ofm, ifm, nIFm) * LIBXSMM_VLA_ACCESS(2, doutput, img, ofm, nOFm);
      }
    }
  }
}

LIBXSMM_INLINE void naive_fullyconnected_wu(naive_fullyconnected_t* param, const float* input_ptr, const float* deloutput_ptr, float* delfilter_ptr)
{
  const int nImg = param->N;
  const int nIFm = param->C;
  const int nOFm = param->K;

  int img, ifm, ofm;

  LIBXSMM_VLA_DECL(2, const float,   input,     input_ptr, nIFm);
  LIBXSMM_VLA_DECL(2,       float, dfilter, delfilter_ptr, nIFm);
  LIBXSMM_VLA_DECL(2, const float, doutput, deloutput_ptr, nOFm);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(ofm); LIBXSMM_OMP_VAR(ifm);
# pragma omp parallel for private(img, ofm, ifm)
#endif
  for (ofm = 0; ofm < nOFm; ++ofm) {
    for (ifm = 0; ifm < nIFm; ++ifm) {
      LIBXSMM_VLA_ACCESS(2, dfilter, ofm, ifm, nIFm) = (float)0;
      for(img = 0; img < nImg; ++img) {
        LIBXSMM_VLA_ACCESS(2, dfilter, ofm, ifm, nIFm) +=
          LIBXSMM_VLA_ACCESS(2, doutput, img, ofm, nOFm) * LIBXSMM_VLA_ACCESS(2, input, img, ifm, nIFm);
      }
    }
  }
}

LIBXSMM_INLINE void naive_pooling_fp(naive_pooling_t* param, const float* input_ptr, float* output_ptr, int* mask_ptr)
{
  const int nImg = param->N;
  const int nFm = param->C;
  const int ifh = param->H;
  const int ifw = param->W;
  const int sh = param->stride_h;
  const int sw = param->stride_w;
  const int r = param->R;
  const int s = param->S;
  const int pad_h = param->pad_h;
  const int pad_w = param->pad_w;
  const int ofh = (ifh + 2*pad_h - r)/sh + 1;
  const int ofw = (ifw + 2*pad_w - s)/sw + 1;


  int img, fm;

  LIBXSMM_VLA_DECL(4, const float, input,   input_ptr, nFm, ifh, ifw);
  LIBXSMM_VLA_DECL(4,       int,   mask,     mask_ptr, nFm, ofh, ofw);
  LIBXSMM_VLA_DECL(4,       float, output, output_ptr, nFm, ofh, ofw);

#if defined(_OPENMP)
  float* tmp_buffer = (float*)malloc(sizeof(float)*ofh*ofw*omp_get_max_threads());
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(fm);
# pragma omp parallel for private(img, fm)
#else
  float* tmp_buffer = (float*)malloc(sizeof(float)*ofh*ofw);
#endif
  for (img = 0; img < nImg; img++) {
    for (fm = 0; fm < nFm; fm++) {
#if defined(_OPENMP)
      float* lcl_buffer_ptr = tmp_buffer + (ofh*ofw*omp_get_thread_num());
#else
      float* lcl_buffer_ptr = tmp_buffer;
#endif
      LIBXSMM_VLA_DECL(2, float, lcl_buffer, lcl_buffer_ptr, ofw);
      int i, ho, wo, hi, wi, kh, kw;

      if (param->type == 0 ) {
        for ( i = 0; i < ofh*ofw; i++ ) {
          lcl_buffer_ptr[i] = -FLT_MAX;
        }
      } else if (param->type == 1) {
        for ( i = 0; i < ofh*ofw; i++ ) {
          lcl_buffer_ptr[i] = 0.0;
        }
      } else {
        /* shouldn't happen */
      }

      for( ho = 0; ho < ofh; ho++ ) {
        hi = (ho * sh) - pad_h;
        for( wo = 0; wo < ofw; wo++ ) {
          wi = (wo * sw) - pad_w;
          for( kh = 0; kh < r; kh++ ) {
            if (hi+kh < 0 || hi+kh >= ifh) continue;
            for( kw = 0; kw < s; kw++ ) {
              if (wi+kw < 0 || wi+kw >= ifw) continue;
              if ( param->type == 0 ) {
                const int index = (hi+kh)*ifw + wi+kw;
                if ( LIBXSMM_VLA_ACCESS(4, input, img, fm, hi+kh, wi+kw, nFm, ifh, ifw) >= LIBXSMM_VLA_ACCESS(2, lcl_buffer, ho, wo, ofw) ) {
                  LIBXSMM_VLA_ACCESS(2, lcl_buffer, ho, wo, ofw) = LIBXSMM_VLA_ACCESS(4, input, img, fm, hi+kh, wi+kw, nFm, ifh, ifw);
                  LIBXSMM_VLA_ACCESS(4, mask, img, fm, ho, wo, nFm, ofh, ofw) = index;
                }
              } else if ( param->type == 1 ) {
                LIBXSMM_VLA_ACCESS(2, lcl_buffer, ho, wo, ofw) += LIBXSMM_VLA_ACCESS(4, input, img, fm, hi+kh, wi+kw, nFm, ifh, ifw);
              } else {
                /* shouldn't happen */
              }
            }
          }
        }
      }

      if (param->type == 0 ) {
        for( ho = 0; ho < ofh; ho++ ) {
          for( wo = 0; wo < ofw; wo++ ) {
            LIBXSMM_VLA_ACCESS(4, output, img, fm, ho, wo, nFm, ofh, ofw) = LIBXSMM_VLA_ACCESS(2, lcl_buffer, ho, wo, ofw);
          }
        }
      } else if (param->type == 1) {
        for( ho = 0; ho < ofh; ho++ ) {
          for( wo = 0; wo < ofw; wo++ ) {
            LIBXSMM_VLA_ACCESS(4, output, img, fm, ho, wo, nFm, ofh, ofw) = LIBXSMM_VLA_ACCESS(2, lcl_buffer, ho, wo, ofw) * (1.0f/(((float)r) * ((float)s)));
          }
        }
      } else {
        /* shouldn't happen */
      }
    }
  }

  free( tmp_buffer );
}

LIBXSMM_INLINE void naive_pooling_bp(naive_pooling_t* param, float* dinput_ptr, const float* doutput_ptr, const int* mask_ptr)
{
  const int nImg = param->N;
  const int nFm = param->C;
  const int ifh = param->H;
  const int ifw = param->W;
  const int sh = param->stride_h;
  const int sw = param->stride_w;
  const int r = param->R;
  const int s = param->S;
  const int pad_h = param->pad_h;
  const int pad_w = param->pad_w;
  const int ofh = (ifh + 2*pad_h - r)/sh + 1;
  const int ofw = (ifw + 2*pad_w - s)/sw + 1;

  int img, fm;

  LIBXSMM_VLA_DECL(4,       float, dinput,   dinput_ptr, nFm, ifh, ifw);
  LIBXSMM_VLA_DECL(4, const int  ,  mask,      mask_ptr, nFm, ofh, ofw);
  LIBXSMM_VLA_DECL(4, const float, doutput, doutput_ptr, nFm, ofh, ofw);

#if defined(_OPENMP)
  float* tmp_buffer = (float*)malloc(sizeof(float)*ifh*ifw*omp_get_max_threads());
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(fm);
# pragma omp parallel for private(img, fm)
#else
  float* tmp_buffer = (float*)malloc(sizeof(float)*ofh*ofw);
#endif
  for (img = 0; img < nImg; img++) {
    for (fm = 0; fm < nFm; fm++) {
#if defined(_OPENMP)
      float* lcl_buffer_ptr = tmp_buffer + (ifh*ifw*omp_get_thread_num());
#else
      float* lcl_buffer_ptr = tmp_buffer;
#endif
      LIBXSMM_VLA_DECL(2, float, lcl_buffer, lcl_buffer_ptr, ifw);
      int i, ho, wo, hi, wi, kh, kw;

      for ( i = 0; i < ifh*ifw; i++ ) {
        lcl_buffer_ptr[i] = 0.0;
      }

      if (param->type == 0 ) {
        for( ho = 0; ho < ofh; ho++ ) {
          for( wo = 0; wo < ofw; wo++ ) {
            lcl_buffer_ptr[LIBXSMM_VLA_ACCESS(4, mask, img, fm, ho, wo, nFm, ofh, ofw)] += LIBXSMM_VLA_ACCESS(4, doutput, img, fm, ho, wo, nFm, ofh, ofw);
          }
        }
      } else if ( param->type == 1 ) {
        for( ho = 0; ho < ofh; ho++ ) {
          hi = (ho * sh) - pad_h;
          for( wo = 0; wo < ofw; wo++ ) {
            wi = (wo * sw) - pad_w;
            for( kh = 0; kh < r; kh++ ) {
              if (hi+kh < 0 || hi+kh >= ifh) continue;
              for( kw = 0; kw < s; kw++ ) {
                if (wi+kw < 0 || wi+kw >= ifw) continue;
                LIBXSMM_VLA_ACCESS(2, lcl_buffer, hi+kh, wi+kw, ifw) += ( LIBXSMM_VLA_ACCESS(4, doutput, img, fm, ho, wo, nFm, ofh, ofw) * (1.0f/(((float)r) * ((float)s))) );
              }
            }
          }
        }
      } else {
        /* shouldn't happen */
      }

      for( hi = 0; hi < ifh; hi++ ) {
        for( wi = 0; wi < ifw; wi++ ) {
          LIBXSMM_VLA_ACCESS(4, dinput, img, fm, hi, wi, nFm, ifh, ifw) = LIBXSMM_VLA_ACCESS(2, lcl_buffer, hi, wi, ifw);
        }
      }
    }
  }

  free( tmp_buffer );
}

/* relumask is created (optionally) in the format without compression, full char element per entry (which is in fact just one bit) */
LIBXSMM_INLINE void naive_fusedbatchnorm_fp(naive_fusedbatchnorm_t* param, const float* input_ptr, float* output_ptr, const float* input_add_ptr,
                                     const float* beta_ptr, const float* gamma_ptr, float eps, float* expectval_ptr, float* rcpstddev_ptr, float* variance_ptr, unsigned char *relumask_ptr)
{
  const int nImg = param->N;
  const int nFm = param->C;
  const int ifh = param->H;
  const int ifw = param->W;
  const int ifhp = param->H + 2 * param->pad_h_in;
  const int ifwp = param->W + 2 * param->pad_w_in;
  const int hi_start = param->pad_h_in;
  const int wi_start = param->pad_w_in;
  const int hi_end = param->pad_h_in + param->H;
  const int wi_end = param->pad_w_in + param->W;
  const int sh = param->stride_h;
  const int sw = param->stride_w;
  /*
  const int ofh = ifh/sh;
  const int ofw = ifw/sw;
  */
  const int ofhp = param->H + 2 * param->pad_h_out;
  const int ofwp = param->W + 2 * param->pad_w_out;
  const int ho_start = param->pad_h_out;
  const int wo_start = param->pad_w_out;
  const int ho_end = param->pad_h_out + param->H;
  const int wo_end = param->pad_w_out + param->W;
  const float nhw = (float)(nImg * ifh * ifw);
  const float recp_nhw = 1.0f/nhw;

  int img, fm, hi, wi, ho, wo;

  LIBXSMM_VLA_DECL(4, const float,   input,     input_ptr,     nFm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4, const float,   input_add, input_add_ptr, nFm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4,       float,   output,    output_ptr,    nFm, ofhp, ofwp);

  LIBXSMM_VLA_DECL(4, unsigned char, relumask,  relumask_ptr,  nFm, ofhp, ofwp); /* no compression, 1 char per entry (only 1 bit used) */

  if ( param->norm_type == 0 ) {
#if defined(_OPENMP)
    LIBXSMM_OMP_VAR(wi); LIBXSMM_OMP_VAR(hi);
#   pragma omp parallel for private(img, fm, hi, wi)
#endif
    for (fm = 0; fm < nFm; fm++) {
      float ch_sum = 0.0f;
      float ch_sumsq = 0.0f;
      float tbmean = 0.0f;
      float tbmeansq = 0.0f;
      float tsqbmean = 0.0f;
      float tbrstd = 0.0f;
      float tvariance = 0.0f;

      for ( img = 0; img < nImg; img++ ) {
        for ( hi = hi_start; hi < hi_start + ifh; hi++ ) {
          for ( wi = wi_start; wi < wi_start + ifw; wi++ ) {
            const float input_val = LIBXSMM_VLA_ACCESS(4, input, img, fm, hi, wi, nFm, ifhp, ifwp);
            ch_sum   += input_val;
            ch_sumsq += (input_val * input_val);
          }
        }
      }

      tbmean = recp_nhw * ch_sum;
      tbmeansq  = tbmean * tbmean;
      tsqbmean = recp_nhw * ch_sumsq;
      tvariance = tsqbmean - tbmeansq;
      tbrstd = (float)(1.0/sqrt(tvariance + eps));
      expectval_ptr[fm] = tbmean;
      rcpstddev_ptr[fm] = tbrstd;
      variance_ptr[fm] = tvariance;
    }
  }

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(ho); LIBXSMM_OMP_VAR(wo);
# pragma omp parallel for private(img, fm, hi, wi, ho, wo)
#endif
  for ( img = 0; img < nImg; img++ ) {
    for ( fm = 0; fm < nFm; fm++ ) {
      /* Handling the padding at the start */
      for (ho = 0; ho < ho_start; ho++) {
        for (wo = 0; wo < ofwp; wo++) {
          float* output_ptr2   = &LIBXSMM_VLA_ACCESS(4, output,    img, fm, ho, wo, nFm, ofhp, ofwp);
          *output_ptr2 = 0;
          unsigned char* relumask_ptr2 = &LIBXSMM_VLA_ACCESS(4, relumask, img, fm, ho, wo, nFm, ofhp, ofwp);
          *relumask_ptr2 = 0;
        }
      }
      for (wo = 0; wo < wo_start; wo++) {
        for (ho = 0; ho < ofhp; ho++) {
          float* output_ptr2   = &LIBXSMM_VLA_ACCESS(4, output,    img, fm, ho, wo, nFm, ofhp, ofwp);
          *output_ptr2 = 0;
          unsigned char* relumask_ptr2 = &LIBXSMM_VLA_ACCESS(4, relumask, img, fm, ho, wo, nFm, ofhp, ofwp);
          *relumask_ptr2 = 0;
        }
      }

      /* Computing the actual batchnorm for the middle (internal) part */
      for ( hi = hi_start, ho = ho_start; hi < hi_end; hi += sh, ho++ ) {
        for ( wi = wi_start, wo = wo_start; wi < wi_end; wi += sw, wo++ ) {
          const float  input_val     =  LIBXSMM_VLA_ACCESS(4, input,     img, fm, hi, wi, nFm, ifhp, ifwp);
          /*const float  input_add_val =  LIBXSMM_VLA_ACCESS(4, input_add, img, fm, hi, wi, nFm, ifh, ifw);*/
                float* output_ptr2   = &LIBXSMM_VLA_ACCESS(4, output,    img, fm, ho, wo, nFm, ofhp, ofwp);

          /* BN + scale (gamma, beta) */
          float o = gamma_ptr[fm]*(input_val - expectval_ptr[fm])*rcpstddev_ptr[fm] + beta_ptr[fm];
          /* Eltwise */
          if ( (param->fuse_type == 2) || (param->fuse_type == 3) || (param->fuse_type == 5) ) {
            const float input_add_val = LIBXSMM_VLA_ACCESS(4, input_add, img, fm, hi, wi, nFm, ifhp, ifwp);
            o += input_add_val;
          }
          /* ReLU */
          if ( (param->fuse_type == 1) || (param->fuse_type == 3) || (param->fuse_type == 4) || (param->fuse_type == 5) ) {
            o = ( o < 0.0f ) ? 0.0f : o;
          }
          *output_ptr2 = o;
          /* Mask */
          if ( (param->fuse_type == 4) || (param->fuse_type == 5) ) {

            /* without compression */
            unsigned char* relumask_ptr2 = &LIBXSMM_VLA_ACCESS(4, relumask, img, fm, ho, wo, nFm, ofhp, ofwp);
            *relumask_ptr2 = (unsigned char)(( o <= 0.0f ) ? 0x0 : 1/*(1 << (i%8))*/ );
          }
        }
      }

      /* Handling the padding at the end */
      for (ho = ho_end; ho < ofhp; ho++) {
        for (wo = 0; wo < ofwp; wo++) {
          float* output_ptr2   = &LIBXSMM_VLA_ACCESS(4, output,    img, fm, ho, wo, nFm, ofhp, ofwp);
          *output_ptr2 = 0;
          unsigned char* relumask_ptr2 = &LIBXSMM_VLA_ACCESS(4, relumask, img, fm, ho, wo, nFm, ofhp, ofwp);
          *relumask_ptr2 = 0;
        }
      }
      for (wo = wo_end; wo < ofwp; wo++) {
        for (ho = 0; ho < ofhp; ho++) {
          float* output_ptr2   = &LIBXSMM_VLA_ACCESS(4, output,    img, fm, ho, wo, nFm, ofhp, ofwp);
          *output_ptr2 = 0;
          unsigned char* relumask_ptr2 = &LIBXSMM_VLA_ACCESS(4, relumask, img, fm, ho, wo, nFm, ofhp, ofwp);
          *relumask_ptr2 = 0;
        }
      }

    }
  }
}

/* relumask is created (optionally) in the format without compression, full char element per entry (which is in fact just one bit) */
LIBXSMM_INLINE void naive_fusedbatchnorm_fp_fp64(naive_fusedbatchnorm_t* param, const double* input_ptr, double* output_ptr, const double* input_add_ptr,
                                     const double* beta_ptr, const double* gamma_ptr, double eps, double* expectval_ptr, double* rcpstddev_ptr, double* variance_ptr, unsigned char *relumask_ptr)
{
  const int nImg = param->N;
  const int nFm = param->C;
  const int ifh = param->H;
  const int ifw = param->W;
  const int sh = param->stride_h;
  const int sw = param->stride_w;
  const int ofh = ifh/sh;
  const int ofw = ifw/sw;
  const double nhw = (double)(nImg * ifh * ifw);
  const double recp_nhw = 1.0f/nhw;

  int img, fm, hi, wi, ho, wo;

  LIBXSMM_VLA_DECL(4, const double,  input,     input_ptr,     nFm, ifh, ifw);
  LIBXSMM_VLA_DECL(4, const double,  input_add, input_add_ptr, nFm, ifh, ifw);
  LIBXSMM_VLA_DECL(4,       double,  output,    output_ptr,    nFm, ofh, ofw);

  LIBXSMM_VLA_DECL(4, unsigned char, relumask,  relumask_ptr,  nFm, ofh, ofw); /* no compression, 1 char per entry (only 1 bit used) */

  if (param->pad_h_in != 0 || param->pad_w_in != 0 || param->pad_h_out != 0 || param->pad_w_out != 0) {
    printf("Error: naive_fusedbatchnorm_fp_fp64 does not support padding!\n");
    return;
  }

  if ( param->norm_type == 0 ) {
#if defined(_OPENMP)
    LIBXSMM_OMP_VAR(wi); LIBXSMM_OMP_VAR(hi);
#   pragma omp parallel for private(img, fm, hi, wi)
#endif
    for (fm = 0; fm < nFm; fm++) {
      double ch_sum = 0.0f;
      double ch_sumsq = 0.0f;
      double tbmean = 0.0f;
      double tbmeansq = 0.0f;
      double tsqbmean = 0.0f;
      double tbrstd = 0.0f;
      double tvariance = 0.0f;

      for ( img = 0; img < nImg; img++ ) {
        for ( hi = 0; hi < ifh; hi++ ) {
          for ( wi = 0; wi < ifw; wi++ ) {
            const double input_val = LIBXSMM_VLA_ACCESS(4, input, img, fm, hi, wi, nFm, ifh, ifw);
            ch_sum   += input_val;
            ch_sumsq += (input_val * input_val);
          }
        }
      }

      tbmean = recp_nhw * ch_sum;
      tbmeansq  = tbmean * tbmean;
      tsqbmean = recp_nhw * ch_sumsq;
      tvariance = tsqbmean - tbmeansq;
      tbrstd = (double)(1.0/sqrt(tvariance + eps));
      expectval_ptr[fm] = tbmean;
      rcpstddev_ptr[fm] = tbrstd;
      variance_ptr[fm] = tvariance;
    }
  }

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(ho); LIBXSMM_OMP_VAR(wo);
# pragma omp parallel for private(img, fm, hi, wi, ho, wo)
#endif
  for ( img = 0; img < nImg; img++ ) {
    for ( fm = 0; fm < nFm; fm++ ) {
      for ( hi = 0, ho = 0; hi < ifh; hi += sh, ho++ ) {
        for ( wi = 0, wo = 0; wi < ifw; wi += sw, wo++ ) {
          const double  input_val     =  LIBXSMM_VLA_ACCESS(4, input,     img, fm, hi, wi, nFm, ifh, ifw);
          /*const double  input_add_val =  LIBXSMM_VLA_ACCESS(4, input_add, img, fm, hi, wi, nFm, ifh, ifw); */
                double* output_ptr2   = &LIBXSMM_VLA_ACCESS(4, output,    img, fm, ho, wo, nFm, ofh, ofw);

          /* BN + scale (gamma, beta) */
          double o = gamma_ptr[fm]*(input_val - expectval_ptr[fm])*rcpstddev_ptr[fm] + beta_ptr[fm];
          /* Eltwise */
          if ( (param->fuse_type == 2) || (param->fuse_type == 3) || (param->fuse_type == 5) ) {
            const double  input_add_val =  LIBXSMM_VLA_ACCESS(4, input_add, img, fm, hi, wi, nFm, ifh, ifw);
            o += input_add_val;
          }
          /* ReLU */
          if ( (param->fuse_type == 1) || (param->fuse_type == 3) || (param->fuse_type == 4) || (param->fuse_type == 5) ) {
            o = ( o < 0.0f ) ? 0.0f : o;
          }
          *output_ptr2 = o;

          /* Mask */
          if ( (param->fuse_type == 4) || (param->fuse_type == 5) ) {
            /* without compression */
            unsigned char* relumask_ptr2 = &LIBXSMM_VLA_ACCESS(4, relumask, img, fm, ho, wo, nFm, ofh, ofw);
            *relumask_ptr2 = (unsigned char)(( o <= 0.0f ) ? 0x0 : 1/*(1 << (i%8))*/ );
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_fusedbatchnorm_bp(naive_fusedbatchnorm_t* param, const float* input_ptr, float* dinput_ptr, const float* output_ptr, float* doutput_ptr, float* dinput_add_ptr,
                                     const float* beta_ptr, float* del_beta_ptr, const float* gamma_ptr, float* del_gamma_ptr,
                                     const float* expectval_ptr, const float* rcpstddev_ptr)
{
  const int nImg = param->N;
  const int nFm = param->C;
  const int ifh = param->H;
  const int ifw = param->W;
  const int ifhp = param->H + 2 * param->pad_h_in;
  const int ifwp = param->W + 2 * param->pad_w_in;
  const int hi_start = param->pad_h_in;
  const int wi_start = param->pad_w_in;
  const int hi_end = param->pad_h_in + param->H;
  const int wi_end = param->pad_w_in + param->W;
  const int sh = param->stride_h;
  const int sw = param->stride_w;
  /*
  const int ofh = ifh/sh;
  const int ofw = ifw/sw;
  */
  const int ofhp = param->H + 2 * param->pad_h_out;
  const int ofwp = param->W + 2 * param->pad_w_out;
  const int ho_start = param->pad_h_out;
  const int wo_start = param->pad_w_out;
  const int ho_end = param->pad_h_out + param->H;
  const int wo_end = param->pad_w_out + param->W;
  const float nhw = (float)(nImg * ifh * ifw);
  const float recp_nhw = 1.0f/nhw;

  int img, fm, hi, wi, ho, wo;

  LIBXSMM_VLA_DECL(4, const float,         input,      input_ptr,      nFm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4,       float,         dinput,     dinput_ptr,     nFm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4,       float,         dinput_add, dinput_add_ptr, nFm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4, const float,         output,     output_ptr,     nFm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4,       float,         doutput,    doutput_ptr,    nFm, ofhp, ofwp);
  LIBXSMM_UNUSED(beta_ptr);

  if ( param->norm_type == 0 ) {
#if defined(_OPENMP)
    LIBXSMM_OMP_VAR(hi); LIBXSMM_OMP_VAR(wi); LIBXSMM_OMP_VAR(ho); LIBXSMM_OMP_VAR(wo);
#   pragma omp parallel for private(img, fm, hi, wi, ho, wo)
#endif
    for ( fm = 0; fm < nFm; fm++ ) {
      del_gamma_ptr[fm] = 0.0f;
      del_beta_ptr[fm] = 0.0f;

      for ( img = 0; img < nImg; img++ ) {

        /* Handling the padding at the start */
        if ( (param->fuse_type == 1) || (param->fuse_type == 3) || (param->fuse_type == 4) || (param->fuse_type == 5) ) {
          for (ho = 0; ho < ho_start; ho++) {
            for (wo = 0; wo < ofwp; wo++) {
              float* del_output_ptr    = &LIBXSMM_VLA_ACCESS(4,    doutput, img, fm, ho, wo, nFm, ofhp, ofwp);
              *del_output_ptr = 0;
            }
          }
          for (wo = 0; wo < wo_start; wo++) {
            for (ho = 0; ho < ofhp; ho++) {
              float* del_output_ptr    = &LIBXSMM_VLA_ACCESS(4,    doutput, img, fm, ho, wo, nFm, ofhp, ofwp);
              *del_output_ptr = 0;
            }
          }
        }
        if ( (param->fuse_type == 2) || (param->fuse_type == 3) || (param->fuse_type == 5) ) {
          for (hi = 0; hi < hi_start; hi++) {
            for (wi = 0; wi < ifwp; wi++) {
              float* del_input_add_ptr = &LIBXSMM_VLA_ACCESS(4, dinput_add, img, fm, hi, wi, nFm, ifhp, ifwp);
              *del_input_add_ptr = 0;
            }
          }
          for (wi = 0; wi < wi_start; wi++) {
            for (hi = 0; hi < ifhp; hi++) {
              float* del_input_add_ptr = &LIBXSMM_VLA_ACCESS(4, dinput_add, img, fm, hi, wi, nFm, ifhp, ifwp);
              *del_input_add_ptr = 0;
            }
          }
        }

        /* main (middle ~ internal) part */
        for ( hi = hi_start, ho = ho_start; hi < hi_end; hi += sh, ho++ ) {
          for ( wi = wi_start, wo = wo_start; wi < wi_end; wi += sw, wo++ ) {
                  float* del_input_add_ptr = &LIBXSMM_VLA_ACCESS(4, dinput_add, img, fm, hi, wi, nFm, ifhp, ifwp);
            const float  output_val        =  LIBXSMM_VLA_ACCESS(4,     output, img, fm, ho, wo, nFm, ofhp, ofwp);
            const float  input_val         =  LIBXSMM_VLA_ACCESS(4,      input, img, fm, hi, wi, nFm, ifhp, ifwp);
                  float* del_output_ptr    = &LIBXSMM_VLA_ACCESS(4,    doutput, img, fm, ho, wo, nFm, ofhp, ofwp);

            /* (inv) ReLU/mask */
            if ( (param->fuse_type == 1) || (param->fuse_type == 3) || (param->fuse_type == 4) || (param->fuse_type == 5) ) {
              *del_output_ptr = (output_val == 0) ? 0 : *del_output_ptr;
            }

            /* elementwise */
            if ( (param->fuse_type == 2) || (param->fuse_type == 3) || (param->fuse_type == 5) ) {
              *del_input_add_ptr = *del_output_ptr;
            }
            del_gamma_ptr[fm] += (input_val - expectval_ptr[fm]) * (*del_output_ptr) * rcpstddev_ptr[fm];
            del_beta_ptr[fm]  += *del_output_ptr;
          }
        }

        /* Handling the padding at the end */
        if ( (param->fuse_type == 1) || (param->fuse_type == 3) || (param->fuse_type == 4) || (param->fuse_type == 5) ) {
          for (ho = ho_end; ho < ofhp; ho++) {
            for (wo = 0; wo < ofwp; wo++) {
              float* del_output_ptr = &LIBXSMM_VLA_ACCESS(4, doutput, img, fm, ho, wo, nFm, ofhp, ofwp);
              *del_output_ptr = 0;
            }
          }
          for (wo = wo_end; wo < ofwp; wo++) {
            for (ho = 0; ho < ofhp; ho++) {
              float* del_output_ptr = &LIBXSMM_VLA_ACCESS(4, doutput, img, fm, ho, wo, nFm, ofhp, ofwp);
              *del_output_ptr = 0;
            }
          }
        }
        if ( (param->fuse_type == 2) || (param->fuse_type == 3) || (param->fuse_type == 5) ) {
          for (hi = hi_end; hi < ifhp; hi++) {
            for (wi = 0; wi < ifwp; wi++) {
              float* del_input_add_ptr = &LIBXSMM_VLA_ACCESS(4, dinput_add, img, fm, hi, wi, nFm, ifhp, ifwp);
              *del_input_add_ptr = 0;
            }
          }
          for (wi = wi_end; wi < ifwp; wi++) {
            for (hi = 0; hi < ifhp; hi++) {
              float* del_input_add_ptr = &LIBXSMM_VLA_ACCESS(4, dinput_add, img, fm, hi, wi, nFm, ifhp, ifwp);
              *del_input_add_ptr = 0;
            }
          }
        }
      }
    }
  }

#if defined(_OPENMP)
# pragma omp parallel for private(img, fm, hi, wi, ho, wo)
#endif
  for ( img = 0; img < nImg; img++ ) {
    for ( fm = 0; fm < nFm; fm++ ) {
      /* Handling the padding at the start */
      for (hi = 0; hi < hi_start; hi++) {
        for (wi = 0; wi < ifwp; wi++) {
          float* del_input_ptr  = &LIBXSMM_VLA_ACCESS(4,     dinput, img, fm, hi, wi, nFm, ifhp, ifwp);
          *del_input_ptr = 0;
        }
      }
      for (wi = 0; wi < wi_start; wi++) {
        for (hi = 0; hi < ifhp; hi++) {
          float* del_input_ptr  = &LIBXSMM_VLA_ACCESS(4,     dinput, img, fm, hi, wi, nFm, ifhp, ifwp);
          *del_input_ptr = 0;
        }
      }

      /* main (middle ~ internal) part */
      for ( hi = hi_start, ho = ho_start; hi < hi_end; hi += sh, ho++ ) {
        for ( wi = wi_start, wo = wo_start; wi < wi_end; wi += sw, wo++) {
                float* del_input_ptr  = &LIBXSMM_VLA_ACCESS(4,     dinput, img, fm, hi, wi, nFm, ifhp, ifwp);
          const float  input_val      =  LIBXSMM_VLA_ACCESS(4,      input, img, fm, hi, wi, nFm, ifhp, ifwp);
          const float  del_output_val =  LIBXSMM_VLA_ACCESS(4,    doutput, img, fm, ho, wo, nFm, ofhp, ofwp);

          *del_input_ptr = gamma_ptr[fm] * rcpstddev_ptr[fm] * recp_nhw * (nhw * del_output_val -
                    (del_beta_ptr[fm] + (input_val - expectval_ptr[fm]) * del_gamma_ptr[fm] * rcpstddev_ptr[fm]));
        }
      }

      /* Handling the padding at the end */
      for (hi = hi_end; hi < ifhp; hi++) {
        for (wi = 0; wi < ifwp; wi++) {
          float* del_input_ptr  = &LIBXSMM_VLA_ACCESS(4,     dinput, img, fm, hi, wi, nFm, ifhp, ifwp);
          *del_input_ptr = 0;
        }
      }
      for (wi = wi_end; wi < ifwp; wi++) {
        for (hi = 0; hi < ifhp; hi++) {
          float* del_input_ptr  = &LIBXSMM_VLA_ACCESS(4,     dinput, img, fm, hi, wi, nFm, ifhp, ifwp);
          *del_input_ptr = 0;
        }
      }

    }
  }
}

LIBXSMM_INLINE void naive_fusedbatchnorm_bp_fp64(naive_fusedbatchnorm_t* param, const double* input_ptr, double* dinput_ptr, const double* output_ptr, double* doutput_ptr, double* dinput_add_ptr,
                                     const double* beta_ptr, double* del_beta_ptr, const double* gamma_ptr, double* del_gamma_ptr,
                                     const double* expectval_ptr, const double* rcpstddev_ptr)
{
  const int nImg = param->N;
  const int nFm = param->C;
  const int ifh = param->H;
  const int ifw = param->W;
  const int sh = param->stride_h;
  const int sw = param->stride_w;
  const int ofh = ifh/sh;
  const int ofw = ifw/sw;
  const double nhw = (double)(nImg * ifh * ifw);
  const double recp_nhw = 1.0f/nhw;

  int img, fm, hi, wi, ho, wo;

  if (param->pad_h_in != 0 || param->pad_w_in != 0 || param->pad_h_out != 0 || param->pad_w_out != 0) {
    printf("Error: naive_fusedbatchnorm_bp_fp64 does not support padding!\n");
    return;
  }

  LIBXSMM_VLA_DECL(4, const double, input,      input_ptr,      nFm, ifh, ifw);
  LIBXSMM_VLA_DECL(4,       double, dinput,     dinput_ptr,     nFm, ifh, ifw);
  LIBXSMM_VLA_DECL(4,       double, dinput_add, dinput_add_ptr, nFm, ifh, ifw);
  LIBXSMM_VLA_DECL(4, const double, output,     output_ptr,     nFm, ofh, ofw);
  LIBXSMM_VLA_DECL(4,       double, doutput,    doutput_ptr,    nFm, ofh, ofw);
  LIBXSMM_UNUSED(beta_ptr);

  if ( param->norm_type == 0 ) {
#if defined(_OPENMP)
    LIBXSMM_OMP_VAR(hi); LIBXSMM_OMP_VAR(wi); LIBXSMM_OMP_VAR(ho); LIBXSMM_OMP_VAR(wo);
#   pragma omp parallel for private(img, fm, hi, wi, ho, wo)
#endif
    for ( fm = 0; fm < nFm; fm++ ) {
      del_gamma_ptr[fm] = 0.0f;
      del_beta_ptr[fm] = 0.0f;

      for ( img = 0; img < nImg; img++ ) {
        for ( hi = 0, ho = 0; hi < ifh; hi += sh, ho++ ) {
          for ( wi = 0, wo = 0; wi < ifw; wi += sw, wo++ ) {
                  double* del_input_add_ptr = &LIBXSMM_VLA_ACCESS(4, dinput_add, img, fm, hi, wi, fm, ifh, ifw);
            const double  output_val        =  LIBXSMM_VLA_ACCESS(4,     output, img, fm, ho, wo, fm, ofh, ofw);
            const double  input_val         =  LIBXSMM_VLA_ACCESS(4,      input, img, fm, hi, wi, fm, ifh, ifw);
                  double* del_output_ptr    = &LIBXSMM_VLA_ACCESS(4,    doutput, img, fm, ho, wo, fm, ofh, ofw);

            /* ReLU */
            if ( (param->fuse_type == 1) || (param->fuse_type == 3) || (param->fuse_type == 4) || (param->fuse_type == 5) ) {
              *del_output_ptr = (output_val == 0) ? 0 : *del_output_ptr;
            }
            /* elementwise */
            if ( (param->fuse_type == 2) || (param->fuse_type == 3) || (param->fuse_type == 5) ) {
              *del_input_add_ptr = *del_output_ptr;
            }
            del_gamma_ptr[fm] += (input_val - expectval_ptr[fm]) * (*del_output_ptr) * rcpstddev_ptr[fm];
            del_beta_ptr[fm]  += *del_output_ptr;
          }
        }
      }
    }
  }

#if defined(_OPENMP)
# pragma omp parallel for private(img, fm, hi, wi, ho, wo)
#endif
  for ( img = 0; img < nImg; img++ ) {
    for ( fm = 0; fm < nFm; fm++ ) {
      for ( hi = 0, ho = 0; hi < ifh; hi += sh, ho++ ) {
        for ( wi = 0, wo = 0; wi < ifw; wi += sw, wo++) {
                double* del_input_ptr  = &LIBXSMM_VLA_ACCESS(4,     dinput, img, fm, hi, wi, fm, ifh, ifw);
          const double  input_val      =  LIBXSMM_VLA_ACCESS(4,      input, img, fm, hi, wi, fm, ifh, ifw);
          const double  del_output_val =  LIBXSMM_VLA_ACCESS(4,    doutput, img, fm, ho, wo, fm, ofh, ofw);

          *del_input_ptr = gamma_ptr[fm] * rcpstddev_ptr[fm] * recp_nhw * (nhw * del_output_val -
                    (del_beta_ptr[fm] + (input_val - expectval_ptr[fm]) * del_gamma_ptr[fm] * rcpstddev_ptr[fm]));
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_fusedgroupnorm_fp(naive_fusedgroupnorm_t* param, const float* input_ptr, float* output_ptr, const float* input_add_ptr,
                                     const float* beta_ptr, const float* gamma_ptr, float eps, float* expectval_ptr, float* rcpstddev_ptr, float* variance_ptr, unsigned char *relumask_ptr)
{
  const int nImg = param->N;
  const int nFm = param->C;
  const int ifh = param->H;
  const int ifw = param->W;
  const int ifhp = param->H + 2 * param->pad_h_in;
  const int ifwp = param->W + 2 * param->pad_w_in;
  const int hi_start = param->pad_h_in;
  const int wi_start = param->pad_w_in;
  const int hi_end = param->pad_h_in + param->H;
  const int wi_end = param->pad_w_in + param->W;
  const int sh = param->stride_h;
  const int sw = param->stride_w;
  /*
  const int ofh = ifh/sh;
  const int ofw = ifw/sw;
  */
  const int nG = param->G;
  const int ofhp = param->H + 2 * param->pad_h_out;
  const int ofwp = param->W + 2 * param->pad_w_out;
  const int ho_start = param->pad_h_out;
  const int wo_start = param->pad_w_out;
  const int ho_end = param->pad_h_out + param->H;
  const int wo_end = param->pad_w_out + param->W;

  const int nFMG = nFm/nG;
  const float ghw = (float)(nFMG * ifh * ifw);
  const float recp_ghw = 1.0f/ghw;

  int img, g, fmg, hi, wi, ho, wo;

  LIBXSMM_VLA_DECL(5, const float, input,     input_ptr,     nG,  nFMG, ifhp, ifwp);
  LIBXSMM_VLA_DECL(5, const float, input_add, input_add_ptr, nG,  nFMG, ifhp, ifwp);
  LIBXSMM_VLA_DECL(5,       float, output,    output_ptr,    nG,  nFMG, ofhp, ofwp);

  LIBXSMM_VLA_DECL(5, unsigned char, relumask,  relumask_ptr, nG, nFMG, ofhp, ofwp); /* no compression, 1 char per entry (only 1 bit used) */

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(g); LIBXSMM_OMP_VAR(fmg); LIBXSMM_OMP_VAR(hi); LIBXSMM_OMP_VAR(wi);
# pragma omp parallel for private(img, g, fmg, hi, wi)
#endif
  for ( img = 0; img < nImg; img++ ) {
    for (g = 0; g < nG; g++) {
      float ch_sum = 0.0f;
      float ch_sumsq = 0.0f;
      float tbmean = 0.0f;
      float tbmeansq = 0.0f;
      float tsqbmean = 0.0f;
      float tbrstd = 0.0f;
      float tvariance = 0.0f;

      for ( fmg = 0; fmg < nFMG; fmg++) {
        for ( hi = hi_start; hi < hi_end; hi++ ) {
          for ( wi = wi_start; wi < wi_end; wi++ ) {
            const float input_val = LIBXSMM_VLA_ACCESS(5, input, img, g, fmg, hi, wi, nG, nFMG, ifhp, ifwp);
            ch_sum   += input_val;
            ch_sumsq += (input_val * input_val);
          }
        }
      }

      tbmean = recp_ghw * ch_sum;
      tbmeansq  = tbmean * tbmean;
      tsqbmean = recp_ghw * ch_sumsq;
      tvariance = tsqbmean - tbmeansq;
      tbrstd = (float)(1.0/sqrt(tvariance + eps));
      expectval_ptr[img*nG+g] = tbmean;
      rcpstddev_ptr[img*nG+g] = tbrstd;
      variance_ptr[img*nG+g] = tvariance;
    }
  }

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(ho); LIBXSMM_OMP_VAR(wo);
# pragma omp parallel for private(img, g, fmg, hi, wi, ho, wo)
#endif
  for ( img = 0; img < nImg; img++ ) {
    for ( g = 0; g < nG; g++ ) {
      for ( fmg = 0; fmg < nFMG; fmg++ ) {
        /* Handling the padding at the start */
        for (ho = 0; ho < ho_start; ho++) {
          for (wo = 0; wo < ofwp; wo++) {
            float* output_ptr2           = &LIBXSMM_VLA_ACCESS(5, output,   img, g, fmg, ho, wo, nG, nFMG, ofhp, ofwp);
            *output_ptr2 = 0;
            unsigned char* relumask_ptr2 = &LIBXSMM_VLA_ACCESS(5, relumask, img, g, fmg, ho, wo, nG, nFMG, ofhp, ofwp);
            *relumask_ptr2 = 0;
          }
        }
        for (wo = 0; wo < wo_start; wo++) {
          for (ho = 0; ho < ofhp; ho++) {
            float* output_ptr2           = &LIBXSMM_VLA_ACCESS(5, output,   img, g, fmg, ho, wo, nG, nFMG, ofhp, ofwp);
            *output_ptr2 = 0;
            unsigned char* relumask_ptr2 = &LIBXSMM_VLA_ACCESS(5, relumask, img, g, fmg, ho, wo, nG, nFMG, ofhp, ofwp);
            *relumask_ptr2 = 0;
          }
        }

        /* Computing the actual groupnorm for the middle (internal) part */
        for ( hi = hi_start, ho = ho_start; hi < hi_end; hi += sh, ho++ ) {
          for ( wi = wi_start, wo = wo_start; wi < wi_end; wi += sw, wo++ ) {

            const float  input_val      =  LIBXSMM_VLA_ACCESS(5, input,     img, g, fmg, hi, wi, nG, nFMG, ifhp, ifwp);
            const float  input_add_val  =  LIBXSMM_VLA_ACCESS(5, input_add, img, g, fmg, hi, wi, nG, nFMG, ifhp, ifwp);
            float* output_ptr2          = &LIBXSMM_VLA_ACCESS(5, output,    img, g, fmg, ho, wo, nG, nFMG, ofhp, ofwp);

            /* BN + scale (gamma, beta) */
            float o = gamma_ptr[g*nFMG+fmg]*(input_val - expectval_ptr[img*nG+g])*rcpstddev_ptr[img*nG+g] + beta_ptr[g*nFMG+fmg];
            /* Eltwise */
            if ( (param->fuse_type == 2) || (param->fuse_type == 3) || (param->fuse_type == 5) ) {
              o += input_add_val;
            }
            /* ReLU */
            if ( (param->fuse_type == 1) || (param->fuse_type == 3) || (param->fuse_type == 4) || (param->fuse_type == 5) ) {
              o = ( o < 0.0f ) ? 0.0f : o;
            }
            *output_ptr2 = o;

            /* Mask */
            if ( (param->fuse_type == 4) || (param->fuse_type == 5) ) {
              /* without compression */
              unsigned char* relumask_ptr2 = &LIBXSMM_VLA_ACCESS(5, relumask, img, g, fmg, ho, wo, nG, nFMG, ofhp, ofwp);
              *relumask_ptr2 = (unsigned char)(( o <= 0.0f ) ? 0x0 : 1/*(1 << (i%8))*/ );
            }
          }
        }

        /* Handling the padding at the end */
        for (ho = ho_end; ho < ofhp; ho++) {
          for (wo = 0; wo < ofwp; wo++) {
            float* output_ptr2           = &LIBXSMM_VLA_ACCESS(5, output,   img, g, fmg, ho, wo, nG, nFMG, ofhp, ofwp);
            *output_ptr2 = 0;
            unsigned char* relumask_ptr2 = &LIBXSMM_VLA_ACCESS(5, relumask, img, g, fmg, ho, wo, nG, nFMG, ofhp, ofwp);
            *relumask_ptr2 = 0;
          }
        }
        for (wo = wo_end; wo < ofwp; wo++) {
          for (ho = 0; ho < ofhp; ho++) {
            float* output_ptr2           = &LIBXSMM_VLA_ACCESS(5, output,   img, g, fmg, ho, wo, nG, nFMG, ofhp, ofwp);
            *output_ptr2 = 0;
            unsigned char* relumask_ptr2 = &LIBXSMM_VLA_ACCESS(5, relumask, img, g, fmg, ho, wo, nG, nFMG, ofhp, ofwp);
            *relumask_ptr2 = 0;
          }
        }

      }
    }
  }
}

LIBXSMM_INLINE void naive_fusedgroupnorm_fp_fp64(naive_fusedgroupnorm_t* param, const double* input_ptr, double* output_ptr, const double* input_add_ptr,
                                     const double* beta_ptr, const double* gamma_ptr, double eps, double* expectval_ptr, double* rcpstddev_ptr, double* variance_ptr, unsigned char *relumask_ptr)
{
  const int nImg = param->N;
  const int nFm = param->C;
  const int ifh = param->H;
  const int ifw = param->W;
  const int sh = param->stride_h;
  const int sw = param->stride_w;
  const int ofh = ifh/sh;
  const int ofw = ifw/sw;
  const int nG = param->G;
  const int nFMG = nFm/nG;
  const double ghw = (double)(nFMG * ifh * ifw);
  const double recp_ghw = 1.0f/ghw;

  int img, g, fmg, hi, wi, ho, wo;

  if (param->pad_h_in != 0 || param->pad_w_in != 0 || param->pad_h_out != 0 || param->pad_w_out != 0) {
    printf("Error: naive_fusedgroupnorm_fp_fp64 does not support padding!\n");
    return;
  }

  LIBXSMM_VLA_DECL(5, const double, input,     input_ptr,     nG,  nFMG, ifh, ifw);
  LIBXSMM_VLA_DECL(5, const double, input_add, input_add_ptr, nG,  nFMG, ifh, ifw);
  LIBXSMM_VLA_DECL(5,       double, output,    output_ptr,    nG,  nFMG, ofh, ofw);

  LIBXSMM_VLA_DECL(5, unsigned char, relumask,  relumask_ptr, nG, nFMG, ofh, ofw); /* no compression, 1 char per entry (only 1 bit used) */

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(g); LIBXSMM_OMP_VAR(fmg); LIBXSMM_OMP_VAR(hi); LIBXSMM_OMP_VAR(wi);
# pragma omp parallel for private(img, g, fmg, hi, wi)
#endif
  for ( img = 0; img < nImg; img++ ) {
    for (g = 0; g < nG; g++) {
      double ch_sum = 0.0f;
      double ch_sumsq = 0.0f;
      double tbmean = 0.0f;
      double tbmeansq = 0.0f;
      double tsqbmean = 0.0f;
      double tbrstd = 0.0f;
      double tvariance = 0.0f;

      for ( fmg = 0; fmg < nFMG; fmg++) {
        for ( hi = 0; hi < ifh; hi++ ) {
          for ( wi = 0; wi < ifw; wi++ ) {
            const double input_val = LIBXSMM_VLA_ACCESS(5, input, img, g, fmg, hi, wi, nG, nFMG, ifh, ifw);
            ch_sum   += input_val;
            ch_sumsq += (input_val * input_val);
          }
        }
      }

      tbmean = recp_ghw * ch_sum;
      tbmeansq  = tbmean * tbmean;
      tsqbmean = recp_ghw * ch_sumsq;
      tvariance = tsqbmean - tbmeansq;
      tbrstd = (double)(1.0/sqrt(tvariance + eps));
      expectval_ptr[img*nG+g] = tbmean;
      rcpstddev_ptr[img*nG+g] = tbrstd;
      variance_ptr[img*nG+g] = tvariance;
    }
  }

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(ho); LIBXSMM_OMP_VAR(wo);
# pragma omp parallel for private(img, g, fmg, hi, wi, ho, wo)
#endif
  for ( img = 0; img < nImg; img++ ) {
    for ( g = 0; g < nG; g++ ) {
      for ( fmg = 0; fmg < nFMG; fmg++ ) {
        for ( hi = 0, ho = 0; hi < ifh; hi += sh, ho++ ) {
          for ( wi = 0, wo = 0; wi < ifw; wi += sw, wo++ ) {
            const double  input_val      =  LIBXSMM_VLA_ACCESS(5, input,     img, g,  fmg, hi, wi, nG, nFMG, ifh, ifw);
            const double  input_add_val  =  LIBXSMM_VLA_ACCESS(5, input_add, img, g,  fmg, hi, wi, nG, nFMG, ifh, ifw);
            double* output_ptr2          = &LIBXSMM_VLA_ACCESS(5, output,    img, g,  fmg, ho, wo, nG, nFMG, ofh, ofw);

            /* BN + scale (gamma, beta) */
            double o = gamma_ptr[g*nFMG+fmg]*(input_val - expectval_ptr[img*nG+g])*rcpstddev_ptr[img*nG+g] + beta_ptr[g*nFMG+fmg];
            /* Eltwise */
            if ( (param->fuse_type == 2) || (param->fuse_type == 3) || (param->fuse_type == 5) ) {
              o += input_add_val;
            }
            /* ReLU */
            if ( (param->fuse_type == 1) || (param->fuse_type == 3) || (param->fuse_type == 4) || (param->fuse_type == 5) ) {
              o = ( o < 0.0f ) ? 0.0f : o;
            }
            *output_ptr2 = o;

            /* Mask */
            if ( (param->fuse_type == 4) || (param->fuse_type == 5) ) {
              /* without compression */
              unsigned char* relumask_ptr2 = &LIBXSMM_VLA_ACCESS(5, relumask, img, g, fmg, ho, wo, nG, nFMG, ofh, ofw);
              *relumask_ptr2 = (unsigned char)(( o <= 0.0f ) ? 0x0 : 1/*(1 << (i%8))*/ );
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_fusedgroupnorm_bp(naive_fusedgroupnorm_t* param, const float* input_ptr, float* dinput_ptr, const float* output_ptr, float* doutput_ptr, float* dinput_add_ptr,
                                     const float* beta_ptr, float* del_beta_ptr, const float* gamma_ptr, float* del_gamma_ptr,
                                     const float* expectval_ptr, const float* rcpstddev_ptr, const float* variance_ptr)
{
  const int nImg = param->N;
  const int nFm = param->C;
  const int ifh = param->H;
  const int ifw = param->W;
  const int ifhp = param->H + 2 * param->pad_h_in;
  const int ifwp = param->W + 2 * param->pad_w_in;
  const int hi_start = param->pad_h_in;
  const int wi_start = param->pad_w_in;
  const int hi_end = param->pad_h_in + param->H;
  const int wi_end = param->pad_w_in + param->W;
  const int sh = param->stride_h;
  const int sw = param->stride_w;
  /*
  const int ofh = ifh/sh;
  const int ofw = ifw/sw;
  */
  const int ofhp = param->H + 2 * param->pad_h_out;
  const int ofwp = param->W + 2 * param->pad_w_out;
  const int ho_start = param->pad_h_out;
  const int wo_start = param->pad_w_out;
  const int ho_end = param->pad_h_out + param->H;
  const int wo_end = param->pad_w_out + param->W;

  const int nG = param->G;
  const int nFMG = nFm/nG;
  const float ghw = (float)(nFMG * ifh * ifw);
  const float recp_ghw = 1.0f/ghw;

  int img, g, fmg, fm, hi, wi, ho, wo;

  LIBXSMM_VLA_DECL(5, const float, input,      input_ptr,      nG,  nFMG, ifhp, ifwp);
  LIBXSMM_VLA_DECL(5,       float, dinput,     dinput_ptr,     nG,  nFMG, ifhp, ifwp);
  /*LIBXSMM_VLA_DECL(5, const float, output,     output_ptr,     nG,  nFMG, ofhp, ofwp);*/
  LIBXSMM_VLA_DECL(5,       float, doutput,    doutput_ptr,    nG,  nFMG, ofhp, ofwp);

  LIBXSMM_VLA_DECL(4, const float, input_gb,      input_ptr,      nFm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4, const float, output_gb,     output_ptr,     nFm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4,       float, doutput_gb,    doutput_ptr,    nFm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4,       float, dinput_add,    dinput_add_ptr, nFm, ifhp, ifwp);

  LIBXSMM_UNUSED(beta_ptr);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(hi); LIBXSMM_OMP_VAR(wi); LIBXSMM_OMP_VAR(ho); LIBXSMM_OMP_VAR(wo); LIBXSMM_OMP_VAR(g);
# pragma omp parallel for private(img, fm, hi, wi, ho, wo, g)
#endif
  for ( fm = 0; fm < nFm; fm++ ) {
    del_gamma_ptr[fm] = 0.0f;
    del_beta_ptr[fm] = 0.0f;

    for ( img = 0; img < nImg; img++ ) {

      /* Handling the padding at the start */
      if ( (param->fuse_type == 1) || (param->fuse_type == 3) || (param->fuse_type == 4) || (param->fuse_type == 5) ) {
        for (ho = 0; ho < ho_start; ho++) {
          for (wo = 0; wo < ofwp; wo++) {
            float* del_output_ptr    = &LIBXSMM_VLA_ACCESS(4,    doutput_gb, img, fm, ho, wo, nFm, ofhp, ofwp);
            *del_output_ptr = 0;
          }
        }
        for (wo = 0; wo < wo_start; wo++) {
          for (ho = 0; ho < ofhp; ho++) {
            float* del_output_ptr    = &LIBXSMM_VLA_ACCESS(4,    doutput_gb, img, fm, ho, wo, nFm, ofhp, ofwp);
            *del_output_ptr = 0;
          }
        }
      }
      if ( (param->fuse_type == 2) || (param->fuse_type == 3) || (param->fuse_type == 5) ) {
        for (hi = 0; hi < hi_start; hi++) {
          for (wi = 0; wi < ifwp; wi++) {
            float* del_input_add_ptr = &LIBXSMM_VLA_ACCESS(4, dinput_add, img, fm, hi, wi, nFm, ifhp, ifwp);
            *del_input_add_ptr = 0;
          }
        }
        for (wi = 0; wi < wi_start; wi++) {
          for (hi = 0; hi < ifhp; hi++) {
            float* del_input_add_ptr = &LIBXSMM_VLA_ACCESS(4, dinput_add, img, fm, hi, wi, nFm, ifhp, ifwp);
            *del_input_add_ptr = 0;
          }
        }
      }

      /* main (middle ~ internal) part */
      for ( hi = hi_start, ho = ho_start; hi < hi_end; hi += sh, ho++ ) {
        for ( wi = wi_start, wo = wo_start; wi < wi_end; wi += sw, wo++ ) {
                float* del_input_add_ptr = &LIBXSMM_VLA_ACCESS(4,    dinput_add, img, fm, hi, wi, nFm, ifhp, ifwp);
          const float  output_val        =  LIBXSMM_VLA_ACCESS(4,     output_gb, img, fm, ho, wo, nFm, ofhp, ofwp);
          const float  input_val         =  LIBXSMM_VLA_ACCESS(4,      input_gb, img, fm, hi, wi, nFm, ifhp, ifwp);
                float* del_output_ptr    = &LIBXSMM_VLA_ACCESS(4,    doutput_gb, img, fm, ho, wo, nFm, ofhp, ofwp);

          /* ReLU */
          if ( (param->fuse_type == 1) || (param->fuse_type == 3) || (param->fuse_type == 4) || (param->fuse_type == 5) ) {
            *del_output_ptr    = (output_val == 0) ? 0 : *del_output_ptr;
          }
          /* elementwise */
          if ( (param->fuse_type == 2) || (param->fuse_type == 3) || (param->fuse_type == 5) ) {
            *del_input_add_ptr = *del_output_ptr;
          }
          g = fm/nFMG;
          del_gamma_ptr[fm] += (input_val - expectval_ptr[img*nG+g]) * (*del_output_ptr) * rcpstddev_ptr[img*nG+g];
          del_beta_ptr[fm]  += *del_output_ptr;
        }
      }

      /* Handling the padding at the end */
      if ( (param->fuse_type == 1) || (param->fuse_type == 3) || (param->fuse_type == 4) || (param->fuse_type == 5) ) {
        for (ho = ho_end; ho < ofhp; ho++) {
          for (wo = 0; wo < ofwp; wo++) {
            float* del_output_ptr = &LIBXSMM_VLA_ACCESS(4, doutput_gb, img, fm, ho, wo, nFm, ofhp, ofwp);
            *del_output_ptr = 0;
          }
        }
        for (wo = wo_end; wo < ofwp; wo++) {
          for (ho = 0; ho < ofhp; ho++) {
            float* del_output_ptr = &LIBXSMM_VLA_ACCESS(4, doutput_gb, img, fm, ho, wo, nFm, ofhp, ofwp);
            *del_output_ptr = 0;
          }
        }
      }
      if ( (param->fuse_type == 2) || (param->fuse_type == 3) || (param->fuse_type == 5) ) {
        for (hi = hi_end; hi < ifhp; hi++) {
          for (wi = 0; wi < ifwp; wi++) {
            float* del_input_add_ptr = &LIBXSMM_VLA_ACCESS(4, dinput_add, img, fm, hi, wi, nFm, ifhp, ifwp);
            *del_input_add_ptr = 0;
          }
        }
        for (wi = wi_end; wi < ifwp; wi++) {
          for (hi = 0; hi < ifhp; hi++) {
            float* del_input_add_ptr = &LIBXSMM_VLA_ACCESS(4, dinput_add, img, fm, hi, wi, nFm, ifhp, ifwp);
            *del_input_add_ptr = 0;
          }
        }
      }
    }
  }

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(fmg);
# pragma omp parallel for private(img, g, fmg, hi, wi, ho, wo)
#endif
  for ( img = 0; img < nImg; img++ ) {
    for ( g = 0; g < nG; g++ ) {
      float d1_val = 0.0;
      float d2_val = 0.0;

      for ( fmg = 0; fmg < nFMG; fmg++ ) {
        for ( hi = hi_start, ho = ho_start; hi < hi_end; hi += sh, ho++ ) {
          for ( wi = wi_start, wo = wo_start; wi < wi_end; wi += sw, wo++ ) {
            const float  input_val      =  LIBXSMM_VLA_ACCESS(5,      input, img, g, fmg, hi, wi, nG, nFMG, ifhp, ifwp);
            const float  del_output_val =  LIBXSMM_VLA_ACCESS(5,    doutput, img, g, fmg, ho, wo, nG, nFMG, ofhp, ofwp);

            d1_val += del_output_val * (input_val - expectval_ptr[img*nG+g]) * gamma_ptr[g*nFMG+fmg];
            d2_val += del_output_val * gamma_ptr[g*nFMG+fmg];
          }
        }
      }

      for ( fmg = 0; fmg < nFMG; fmg++ ) {

        /* Handling the padding at the start */
        for (hi = 0; hi < hi_start; hi++) {
          for (wi = 0; wi < ifwp; wi++) {
            float* del_input_ptr  = &LIBXSMM_VLA_ACCESS(5,     dinput, img, g, fmg, hi, wi, nG, nFMG, ifhp, ifwp);
            *del_input_ptr = 0;
          }
        }
        for (wi = 0; wi < wi_start; wi++) {
          for (hi = 0; hi < ifhp; hi++) {
            float* del_input_ptr  = &LIBXSMM_VLA_ACCESS(5,     dinput, img, g, fmg, hi, wi, nG, nFMG, ifhp, ifwp);
            *del_input_ptr = 0;
          }
        }

        /* main (middle ~ internal) part */
        for ( hi = hi_start, ho = ho_start; hi < hi_end; hi += sh, ho++ ) {
          for ( wi = wi_start, wo = wo_start; wi < wi_end; wi += sw, wo++) {
            const float  input_val      =  LIBXSMM_VLA_ACCESS(5,      input, img, g, fmg, hi, wi, nG, nFMG, ifhp, ifwp);
            const float  del_output_val =  LIBXSMM_VLA_ACCESS(5,    doutput, img, g, fmg, ho, wo, nG, nFMG, ofhp, ofwp);
                  float* del_input_ptr  = &LIBXSMM_VLA_ACCESS(5,     dinput, img, g, fmg, hi, wi, nG, nFMG, ifhp, ifwp);

            float t0_val = rcpstddev_ptr[img*nG+g] * recp_ghw;
            *del_input_ptr = t0_val * ((gamma_ptr[g*nFMG+fmg] * ghw * del_output_val) - d2_val - ((input_val - expectval_ptr[img*nG+g]) * d1_val * rcpstddev_ptr[img*nG+g] * rcpstddev_ptr[img*nG+g] /*(1.0f/(variance_ptr[img*nG+g]+eps))*/));
          }
        }

        /* Handling the padding at the end */
        for (hi = hi_end; hi < ifhp; hi++) {
          for (wi = 0; wi < ifwp; wi++) {
            float* del_input_ptr  = &LIBXSMM_VLA_ACCESS(5,     dinput, img, g, fmg, hi, wi, nG, nFMG, ifhp, ifwp);
            *del_input_ptr = 0;
          }
        }
        for (wi = wi_end; wi < ifwp; wi++) {
          for (hi = 0; hi < ifhp; hi++) {
            float* del_input_ptr  = &LIBXSMM_VLA_ACCESS(5,     dinput, img, g, fmg, hi, wi, nG, nFMG, ifhp, ifwp);
            *del_input_ptr = 0;
          }
        }

      }
    }
  }
}

LIBXSMM_INLINE void naive_fusedgroupnorm_bp_fp64(naive_fusedgroupnorm_t* param, const double* input_ptr, double* dinput_ptr, const double* output_ptr, double* doutput_ptr, double* dinput_add_ptr,
                                     const double* beta_ptr, double* del_beta_ptr, const double* gamma_ptr, double* del_gamma_ptr,
                                     const double* expectval_ptr, const double* rcpstddev_ptr, const double* variance_ptr)
{
  const int nImg = param->N;
  const int nFm = param->C;
  const int ifh = param->H;
  const int ifw = param->W;
  const int sh = param->stride_h;
  const int sw = param->stride_w;
  const int ofh = ifh/sh;
  const int ofw = ifw/sw;
  const int nG = param->G;
  const int nFMG = nFm/nG;
  const double ghw = (double)(nFMG * ifh * ifw);
  const double recp_ghw = 1.0f/ghw;

  int img, g, fmg, fm, hi, wi, ho, wo;

  if (param->pad_h_in != 0 || param->pad_w_in != 0 || param->pad_h_out != 0 || param->pad_w_out != 0) {
    printf("Error: naive_fusedgroupnorm_bp_fp64 does not support padding!\n");
    return;
  }

  LIBXSMM_VLA_DECL(5, const double, input,      input_ptr,      nG,  nFMG, ifh, ifw);
  LIBXSMM_VLA_DECL(5,       double, dinput,     dinput_ptr,     nG,  nFMG, ifh, ifw);
  /*LIBXSMM_VLA_DECL(5, const double, output,     output_ptr,     nG,  nFMG, ofh, ofw);*/
  LIBXSMM_VLA_DECL(5,       double, doutput,    doutput_ptr,    nG,  nFMG, ofh, ofw);

  LIBXSMM_VLA_DECL(4, const double, input_gb,      input_ptr,      nFm,  ifh, ifw);
  LIBXSMM_VLA_DECL(4, const double, output_gb,     output_ptr,     nFm,  ofh, ofw);
  LIBXSMM_VLA_DECL(4,       double, doutput_gb,    doutput_ptr,    nFm,  ofh, ofw);
  LIBXSMM_VLA_DECL(4,       double, dinput_add,    dinput_add_ptr, nFm, ifh, ifw);

  LIBXSMM_UNUSED(beta_ptr);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(hi); LIBXSMM_OMP_VAR(wi); LIBXSMM_OMP_VAR(ho); LIBXSMM_OMP_VAR(wo); LIBXSMM_OMP_VAR(g);
# pragma omp parallel for private(img, fm, hi, wi, ho, wo, g)
#endif
  for ( fm = 0; fm < nFm; fm++ ) {
    del_gamma_ptr[fm] = 0.0f;
    del_beta_ptr[fm] = 0.0f;

    for ( img = 0; img < nImg; img++ ) {
      for ( hi = 0, ho = 0; hi < ifh; hi += sh, ho++ ) {
        for ( wi = 0, wo = 0; wi < ifw; wi += sw, wo++ ) {
                double* del_input_add_ptr = &LIBXSMM_VLA_ACCESS(4,    dinput_add, img, fm, hi, wi, nFm, ifh, ifw);
          const double  output_val        =  LIBXSMM_VLA_ACCESS(4,     output_gb, img, fm, ho, wo, nFm, ofh, ofw);
          const double  input_val         =  LIBXSMM_VLA_ACCESS(4,      input_gb, img, fm, hi, wi, nFm, ifh, ifw);
                double* del_output_ptr    = &LIBXSMM_VLA_ACCESS(4,    doutput_gb, img, fm, ho, wo, nFm, ofh, ofw);

          /* ReLU */
          if ( (param->fuse_type == 1) || (param->fuse_type == 3) || (param->fuse_type == 4) || (param->fuse_type == 5) ) {
            *del_output_ptr    = (output_val == 0) ? 0 : *del_output_ptr;
          }
          /* elementwise */
          if ( (param->fuse_type == 2) || (param->fuse_type == 3) || (param->fuse_type == 5) ) {
            *del_input_add_ptr = *del_output_ptr;
          }
          g = fm/nFMG;
          del_gamma_ptr[fm] += (input_val - expectval_ptr[img*nG+g]) * (*del_output_ptr) * rcpstddev_ptr[img*nG+g];
          del_beta_ptr[fm]  += *del_output_ptr;
        }
      }
    }
  }

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(fmg);
# pragma omp parallel for private(img, g, fmg, hi, wi, ho, wo)
#endif
  for ( img = 0; img < nImg; img++ ) {
    for ( g = 0; g < nG; g++ ) {
      double d1_val = 0.0;
      double d2_val = 0.0;

      for ( fmg = 0; fmg < nFMG; fmg++ ) {
        for ( hi = 0, ho = 0; hi < ifh; hi += sh, ho++ ) {
          for ( wi = 0, wo = 0; wi < ifw; wi += sw, wo++) {
            const double  input_val      =  LIBXSMM_VLA_ACCESS(5,      input, img, g, fmg, hi, wi, nG, nFMG, ifh, ifw);
            const double  del_output_val =  LIBXSMM_VLA_ACCESS(5,    doutput, img, g, fmg, ho, wo, nG, nFMG, ofh, ofw);

            d1_val += del_output_val * (input_val - expectval_ptr[img*nG+g]) * gamma_ptr[g*nFMG+fmg];
            d2_val += del_output_val * gamma_ptr[g*nFMG+fmg];
          }
        }
      }

      for ( fmg = 0; fmg < nFMG; fmg++ ) {
        for ( hi = 0, ho = 0; hi < ifh; hi += sh, ho++ ) {
          for ( wi = 0, wo = 0; wi < ifw; wi += sw, wo++) {
            const double  input_val      =  LIBXSMM_VLA_ACCESS(5,      input, img, g, fmg, hi, wi, nG, nFMG, ifh, ifw);
            const double  del_output_val =  LIBXSMM_VLA_ACCESS(5,    doutput, img, g, fmg, ho, wo, nG, nFMG, ofh, ofw);
                  double* del_input_ptr  = &LIBXSMM_VLA_ACCESS(5,     dinput, img, g, fmg, hi, wi, nG, nFMG, ifh, ifw);

            double t0_val = rcpstddev_ptr[img*nG+g] * recp_ghw;
            *del_input_ptr = t0_val * ((gamma_ptr[g*nFMG+fmg] * ghw * del_output_val) - d2_val - ((input_val - expectval_ptr[img*nG+g]) * d1_val * rcpstddev_ptr[img*nG+g] * rcpstddev_ptr[img*nG+g]/*(1.0f/(variance_ptr[img*nG+g]+eps))*/));
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void lstm_fwd_copy_bias(int N, int K, float *bigold, float *bcgold, float *bfgold, float *bogold, float forget_bias, float *icfogoldt, int j)
{
  LIBXSMM_VLA_DECL(3, float, icfogold, icfogoldt, N, 4 * K);
  int i, l;
#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(i); LIBXSMM_OMP_VAR(l);
# pragma omp parallel for private(i, l) LIBXSMM_OPENMP_COLLAPSE(2)
#endif
  for (i = 0; i < N; i++) {
    for (l = 0; l < K; l++) {
      LIBXSMM_VLA_ACCESS(3, icfogold, j, i, l,     N, 4 * K) = bigold[l];
      LIBXSMM_VLA_ACCESS(3, icfogold, j, i, l+K,   N, 4 * K) = bcgold[l];
      LIBXSMM_VLA_ACCESS(3, icfogold, j, i, l+2*K, N, 4 * K) = bfgold[l] + forget_bias;
      LIBXSMM_VLA_ACCESS(3, icfogold, j, i, l+3*K, N, 4 * K) = bogold[l];
    }
  }
}

LIBXSMM_INLINE void lstm_fwd_eltwise_merged(int N, int K, float *i, float *c, float *f, float *o, float *csp, float *cs, float *co, float *h)
{
  int j;
#if defined(__AVX512F__)
  int l;
  int rem = (K/16)*16;
  __m512 minus1 = _mm512_set1_ps (-1.0f);
  __m512 plus1  = _mm512_set1_ps (1.0f);
#if defined(_OPENMP)
# pragma omp parallel for private(j, l) LIBXSMM_OPENMP_COLLAPSE(2)
#endif
  for (j = 0; j < N; j++) {
    for (l = 0; l < rem; l+=16) {
      __m512 iv   = LIBXSMM_INTRINSICS_MM512_LOAD_PS (&(i[j*4*K + l]));
      __m512 cv   = LIBXSMM_INTRINSICS_MM512_LOAD_PS (&(c[j*4*K + l]));
      __m512 fv   = LIBXSMM_INTRINSICS_MM512_LOAD_PS (&(f[j*4*K + l]));
      __m512 ov   = LIBXSMM_INTRINSICS_MM512_LOAD_PS (&(o[j*4*K + l]));
      __m512 cspv = LIBXSMM_INTRINSICS_MM512_LOAD_PS (&(csp[j*K + l]));
      __m512 csv, cov, hv;
      /* i = sigmoid(i) */
      iv = _mm512_mul_ps (iv, minus1);
      iv = LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS (iv);
      iv = _mm512_add_ps (iv, plus1);
      iv = _mm512_div_ps (plus1, iv);
      /* c = tanh(c) */
      cv = LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2 (cv);
      /* f = sigmoid(f) */
      fv = _mm512_mul_ps (fv, minus1);
      fv = LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS (fv);
      fv = _mm512_add_ps (fv, plus1);
      fv = _mm512_div_ps (plus1, fv);
      /* o = sigmoid(o) */
      ov = _mm512_mul_ps (ov, minus1);
      ov = LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS (ov);
      ov = _mm512_add_ps (ov, plus1);
      ov = _mm512_div_ps (plus1, ov);
      /* cs = f.csp + i.c */
      csv = _mm512_mul_ps (fv, cspv);
      csv = _mm512_fmadd_ps (iv, cv, csv);
      /* co = tanh(cs) */
      cov = LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2 (csv);
      /* h = o.co */
      hv = _mm512_mul_ps (ov, cov);
      _mm512_storeu_ps (&(i[j*4*K + l]), iv);
      _mm512_storeu_ps (&(c[j*4*K + l]), cv);
      _mm512_storeu_ps (&(f[j*4*K + l]), fv);
      _mm512_storeu_ps (&(o[j*4*K + l]), ov);
      _mm512_storeu_ps (&(cs[j*K + l]),  csv);
      _mm512_storeu_ps (&(co[j*K + l]),  cov);
      _mm512_storeu_ps (&(h[j*K + l]),   hv);
    }
  }
#if defined(_OPENMP)
# pragma omp parallel for private(j, l) LIBXSMM_OPENMP_COLLAPSE(2)
#endif
  for (j = 0; j < N; j++) {
    for (l = rem; l < K; l++) {
      float exp_value;
      /* i = sigmoid(i) */
      exp_value = (float)exp((double) -i[j*4*K + l]);
      i[j*4*K + l] = 1.0f / (1.0f + exp_value);
      /* c = tanh(c) */
      c[j*4*K + l] = (float)tanh((double)c[j*4*K + l]);
      /* f = sigmoid(f) */
      exp_value = (float)exp((double) -f[j*4*K + l]);
      f[j*4*K + l] = 1.0f / (1.0f + exp_value);
      /* o = sigmoid(o) */
      exp_value = (float)exp((double) -o[j*4*K + l]);
      o[j*4*K + l] = 1.0f / (1.0f + exp_value);
      /* cs = f.csp + i.c */
      cs[j*K + l] = f[j*4*K + l]*csp[j*K + l] + i[j*4*K + l]*c[j*4*K + l];
      /* co = tanh(cs) */
      co[j*K + l] = (float)tanh((double)cs[j*K + l]);
      /* h = o.co */
      h[j*K + l] = o[j*4*K + l] * co[j*K + l];
    }
  }
#else
#if defined(_OPENMP)
# pragma omp parallel for private(j)
#endif
  for (j = 0; j < N*K; j++) {
    const int row = j / K;
    const int col = j % K;
    float exp_value;
    /* i = sigmoid(i) */
    exp_value = (float)exp((double) -i[row*4*K + col]);
    i[row*4*K + col] = 1.0f / (1.0f + exp_value);
    /* c = tanh(c) */
    c[row*4*K + col] = (float)tanh((double)c[row*4*K + col]);
    /* f = sigmoid(f) */
    exp_value = (float)exp((double) -f[row*4*K + col]);
    f[row*4*K + col] = 1.0f / (1.0f + exp_value);
    /* o = sigmoid(o) */
    exp_value = (float)exp((double) -o[row*4*K + col]);
    o[row*4*K + col] = 1.0f / (1.0f + exp_value);
    /* cs = f.csp + i.c */
    cs[j] = f[row*4*K + col]*csp[j] + i[row*4*K + col]*c[row*4*K + col];
    /* co = tanh(cs) */
    co[j] = (float)tanh((double)cs[j]);
    /* h = o.co */
    h[j] = o[row*4*K + col] * co[j];
  }
#endif
}

LIBXSMM_INLINE void lstm_bwd_upd_eltwise_merged(int N, int K, float *i, float *c, float *f, float *o, float *csp, float *co,
                                                float *dh, float *dout, float *di, float *dc, float *df, float *dp, float *dcsp, float *dcs)
{
  int j;
#if defined(__AVX512F__)
  int l;
  int rem = (K/16)*16;
  __m512 plus1  = _mm512_set1_ps (1.0f);
#if defined(_OPENMP)
# pragma omp parallel for private(j, l) LIBXSMM_OPENMP_COLLAPSE(2)
#endif
  for (j = 0; j < N; j++) {
    for (l = 0; l < rem; l+=16) {
      __m512 iv       = LIBXSMM_INTRINSICS_MM512_LOAD_PS (&(i[j*4*K + l]));
      __m512 cv       = LIBXSMM_INTRINSICS_MM512_LOAD_PS (&(c[j*4*K + l]));
      __m512 fv       = LIBXSMM_INTRINSICS_MM512_LOAD_PS (&(f[j*4*K + l]));
      __m512 ov       = LIBXSMM_INTRINSICS_MM512_LOAD_PS (&(o[j*4*K + l]));
      __m512 cspv     = LIBXSMM_INTRINSICS_MM512_LOAD_PS (&(csp[j*K + l]));
      __m512 cov      = LIBXSMM_INTRINSICS_MM512_LOAD_PS (&(co[j*K + l]));
      __m512 dcsv     = LIBXSMM_INTRINSICS_MM512_LOAD_PS (&(dcs[j*K + l]));
      __m512 dhv, doutv, div, dcv, dfv, dov, dcspv, deltav, tv;
      /* compute delta */
      if (NULL == dout) {
        deltav = LIBXSMM_INTRINSICS_MM512_LOAD_PS (&(dh[j*K + l]));
      } else {
        dhv    = LIBXSMM_INTRINSICS_MM512_LOAD_PS (&(dh[j*K + l]));
        doutv  = LIBXSMM_INTRINSICS_MM512_LOAD_PS (&(dout[j*K + l]));
        deltav = _mm512_add_ps (dhv, doutv);
      }
      /* compute dcsp */
      /* dcsp = delta.o.(1 - (co.co)) + dcs */
      tv    = _mm512_mul_ps (cov, cov);
      tv    = _mm512_sub_ps (plus1, tv);
      dcspv = _mm512_mul_ps (deltav, ov);
      dcspv = _mm512_fmadd_ps (dcspv, tv, dcsv);
      /* compute di */
      /* di = dcsp.c.i.(1 - i) */
      tv  = _mm512_sub_ps (plus1, iv);
      tv  = _mm512_mul_ps (iv, tv);
      div = _mm512_mul_ps (dcspv, cv);
      div = _mm512_mul_ps (div, tv);
      /* compute dc */
      /* dc = dcsp.i.(1 - (c.c)) */
      tv  = _mm512_mul_ps (cv, cv);
      tv  = _mm512_sub_ps (plus1, tv);
      dcv = _mm512_mul_ps (dcspv, iv);
      dcv = _mm512_mul_ps (dcv, tv);
      /* compute df */
      /* df = dcsp.csp.f.(1 - f) */
      tv  = _mm512_sub_ps (plus1, fv);
      tv  = _mm512_mul_ps (fv, tv);
      dfv = _mm512_mul_ps (dcspv, cspv);
      dfv = _mm512_mul_ps (dfv, tv);
      /* compute do */
      /* do = delta.co.o.(1 - o) */
      tv  = _mm512_sub_ps (plus1, ov);
      tv  = _mm512_mul_ps (ov, tv);
      dov = _mm512_mul_ps (deltav, cov);
      dov = _mm512_mul_ps (dov, tv);
      /* update dcsp */
      /* dcsp = dcsp.f */
      dcspv = _mm512_mul_ps (dcspv, fv);
      _mm512_storeu_ps (&(di[j*4*K + l]), div);
      _mm512_storeu_ps (&(dc[j*4*K + l]), dcv);
      _mm512_storeu_ps (&(df[j*4*K + l]), dfv);
      _mm512_storeu_ps (&(dp[j*4*K + l]), dov);
      _mm512_storeu_ps (&(dcsp[j*K + l]), dcspv);
    }
  }
#if defined(_OPENMP)
# pragma omp parallel for private(j, l) LIBXSMM_OPENMP_COLLAPSE(2)
#endif
  for (j = 0; j < N; j++) {
    for (l = rem; l < K; l++) {
      float delta;
      /* compute delta */
      if (NULL == dout) {
        delta = dh[j*K + l];
      } else {
        delta = dh[j*K + l] + dout[j*K + l];
      }
      /* compute dcsp */
      dcsp[j*K + l] = delta * o[j*4*K + l] * (1.0f - (co[j*K + l]*co[j*K + l])) + dcs[j*K + l];
      /* compute di */
      di[j*4*K + l] = dcsp[j*K + l] * c[j*4*K + l] * i[j*4*K + l] * (1.0f - i[j*4*K + l]);
      /* compute dc */
      dc[j*4*K + l] = dcsp[j*K + l] * i[j*4*K + l] * (1.0f - (c[j*4*K + l]*c[j*4*K + l]));
      /* compute df */
      df[j*4*K + l] = dcsp[j*K + l] * csp[j*K + l] * f[j*4*K + l] * (1.0f - f[j*4*K + l]);
      /* compute do */
      dp[j*4*K + l] = delta * co[j*K + l] * o[j*4*K + l] * (1.0f - o[j*4*K + l]);
      /* update dcsp */
      dcsp[j*K + l] = dcsp[j*K + l] * f[j*4*K + l];
    }
  }
#else
#if defined(_OPENMP)
# pragma omp parallel for private(j)
#endif
  for (j = 0; j < N*K; j++) {
    const int row = j / K;
    const int col = j % K;
    float delta;
    /* compute delta */
    if (NULL == dout) {
      delta = dh[j];
    } else {
      delta = dh[j] + dout[j];
    }
    /* compute dcsp */
    dcsp[j] = delta * o[row*4*K + col] * (1.0f - (co[j]*co[j])) + dcs[j];
    /* compute di */
    di[row*4*K + col] = dcsp[j] * c[row*4*K + col] * i[row*4*K + col] * (1.0f - i[row*4*K + col]);
    /* compute dc */
    dc[row*4*K + col] = dcsp[j] * i[row*4*K + col] * (1.0f - (c[row*4*K + col]*c[row*4*K + col]));
    /* compute df */
    df[row*4*K + col] = dcsp[j] * csp[j] * f[row*4*K + col] * (1.0f - f[row*4*K + col]);
    /* compute do */
    dp[row*4*K + col] = delta * co[j] * o[row*4*K + col] * (1.0f - o[row*4*K + col]);
    /* update dcsp */
    dcsp[j] = dcsp[j] * f[row*4*K + col];
  }
#endif
}

LIBXSMM_INLINE void lstm_ref_fwd( int N, int C, int K, int t, float forget_bias,
                   float *wigold, float *wcgold, float *wfgold, float *wogold,
                   float *rigold, float *rcgold, float *rfgold, float *rogold,
                   float *bigold, float *bcgold, float *bfgold, float *bogold,
                   float *xgoldt, float *cspgold, float *hpgold,
                   float *csgoldt, float *cogoldt, float *hgoldt,
                   float *icfogoldt, float *wgold, float *rgold, float *scratch )
{
#if !defined(TWO_GEMMS)
  float *xhgold = scratch;
#endif
  const char transa = 'N', transb = 'N';   /* no transposes */
  const float alpha = 1, beta = 1;
  int j;
  int K4 = K * 4;
  int CK = C + K;
  LIBXSMM_VLA_DECL(2, float, xgold, xgoldt, N * C);
  LIBXSMM_VLA_DECL(2, float, csgold, csgoldt, K * N);
  LIBXSMM_VLA_DECL(2, float, cogold, cogoldt, K * N);
  LIBXSMM_VLA_DECL(2, float, hgold, hgoldt, K * N);
  LIBXSMM_VLA_DECL(3, float, icfogold, icfogoldt, N, 4 * K);
#if defined(PROFILE)
  Gbl_conv_start = libxsmm_timer_tick();
#endif
#if defined(TWO_GEMMS)
  convert_ck_c4k(C, K, wigold, wgold);
  convert_ck_c4k(C, K, wcgold, &(wgold[K]));
  convert_ck_c4k(C, K, wfgold, &(wgold[2*K]));
  convert_ck_c4k(C, K, wogold, &(wgold[3*K]));
  convert_ck_c4k(K, K, rigold, rgold);
  convert_ck_c4k(K, K, rcgold, &(rgold[K]));
  convert_ck_c4k(K, K, rfgold, &(rgold[2*K]));
  convert_ck_c4k(K, K, rogold, &(rgold[3*K]));
#else
  LIBXSMM_UNUSED(rgold);
  convert_ck_c4k(C, K, wigold, wgold);
  convert_ck_c4k(C, K, wcgold, &(wgold[K]));
  convert_ck_c4k(C, K, wfgold, &(wgold[2*K]));
  convert_ck_c4k(C, K, wogold, &(wgold[3*K]));
  convert_ck_c4k(K, K, rigold, &(wgold[C*K*4]));
  convert_ck_c4k(K, K, rcgold, &(wgold[C*K*4 + K]));
  convert_ck_c4k(K, K, rfgold, &(wgold[C*K*4 + 2*K]));
  convert_ck_c4k(K, K, rogold, &(wgold[C*K*4 + 3*K]));
#endif
#if defined(PROFILE)
  Gbl_conv_end = libxsmm_timer_tick();
  Gbl_conv_total += libxsmm_timer_duration(Gbl_conv_start, Gbl_conv_end);
#endif
  for (j = 0; j < t; ++j) {
    /* Initialization with bias */
#if defined(PROFILE)
    Gbl_copy_bias_start = libxsmm_timer_tick();
#endif
    lstm_fwd_copy_bias(N, K, bigold, bcgold, bfgold, bogold, forget_bias, icfogoldt, j);
#if defined(PROFILE)
    Gbl_copy_bias_end = libxsmm_timer_tick();
    Gbl_copy_bias_total += libxsmm_timer_duration(Gbl_copy_bias_start, Gbl_copy_bias_end);
    Gbl_blas_start = libxsmm_timer_tick();
#endif
#if defined(TWO_GEMMS)
    /* icfo += W * x */
    LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K4, &N, &C, &alpha, wgold, &K4, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, N * C), &C, &beta, &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 0, N, 4 * K), &K4);
    /* icfo += R * h */
    if (j == 0) {
      LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K4, &N, &K, &alpha, rgold, &K4, hpgold, &K, &beta, &LIBXSMM_VLA_ACCESS(3, icfogold, 0, 0, 0, N, 4 * K), &K4);
    } else {
      LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K4, &N, &K, &alpha, rgold, &K4, &LIBXSMM_VLA_ACCESS(2, hgold, j-1, 0, K * N), &K, &beta, &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 0, N, 4 * K), &K4);
    }
#else
    /* Concatenate x and h */
    convert_nk_nck(N, C, C+K, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, N * C), xhgold);
    if (j == 0) {
      convert_nk_nck(N, K, C+K, hpgold, &(xhgold[C]));
    } else {
      convert_nk_nck(N, K, C+K, &LIBXSMM_VLA_ACCESS(2, hgold, j-1, 0, K * N), &(xhgold[C]));
    }
    /* icfo += (W * x) + (R * h) */
    LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K4, &N, &CK, &alpha, wgold, &K4, xhgold, &CK, &beta, &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 0, N, 4 * K), &K4);
#endif
#if defined(PROFILE)
    Gbl_blas_end = libxsmm_timer_tick();
    Gbl_blas_total += libxsmm_timer_duration(Gbl_blas_start, Gbl_blas_end);
    Gbl_eltwise_start = libxsmm_timer_tick();
#endif
    if (j == 0) {
      lstm_fwd_eltwise_merged( N, K,
                               &LIBXSMM_VLA_ACCESS(3, icfogold, 0, 0, 0,   N, 4 * K),
                               &LIBXSMM_VLA_ACCESS(3, icfogold, 0, 0, K,   N, 4 * K),
                               &LIBXSMM_VLA_ACCESS(3, icfogold, 0, 0, 2*K, N, 4 * K),
                               &LIBXSMM_VLA_ACCESS(3, icfogold, 0, 0, 3*K, N, 4 * K),
                               cspgold,
                               &LIBXSMM_VLA_ACCESS(2, csgold, 0, 0, K * N),
                               &LIBXSMM_VLA_ACCESS(2, cogold, 0, 0, K * N),
                               &LIBXSMM_VLA_ACCESS(2, hgold, 0, 0, K * N) );
    } else {
      lstm_fwd_eltwise_merged( N, K,
                               &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 0,   N, 4 * K),
                               &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, K,   N, 4 * K),
                               &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 2*K, N, 4 * K),
                               &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 3*K, N, 4 * K),
                               &LIBXSMM_VLA_ACCESS(2, csgold, j-1, 0, K * N),
                               &LIBXSMM_VLA_ACCESS(2, csgold, j, 0, K * N),
                               &LIBXSMM_VLA_ACCESS(2, cogold, j, 0, K * N),
                               &LIBXSMM_VLA_ACCESS(2, hgold, j, 0, K * N) );
    }
#if defined(PROFILE)
    Gbl_eltwise_end = libxsmm_timer_tick();
    Gbl_eltwise_total += libxsmm_timer_duration(Gbl_eltwise_start, Gbl_eltwise_end);
#endif
  }
}

LIBXSMM_INLINE void lstm_ref_bwd_upd( int N, int C, int K, int t,
                       float *xgoldt, float *cspgold, float *hpgold,
                       float *csgoldt, float *cogoldt, float *hgoldt,
                       float *icfogoldt, float *wgold, float *rgold,
                       float *dcsgold, float *dhgoldt,
                       float *dwgold, float *drgold, float *dbgold,
                       float *dxgoldt, float *dcspgold, float *dhpgold, float *scratch )
{
#if !defined(TWO_GEMMS)
  float *xhgold   = &(scratch[K*N*t*5]);
  float *dxhgold  = &(scratch[K*N*t*5 + (C+K)*N]);
#endif
  float *dicfogoldt = scratch;
  float *doutgoldt  = &(scratch[K*N*t*4]);
  float *dout, *dcs, *csp;
  const char transa = 'N', transb = 'N';   /* no transposes */
  const char transaT = 'T', transbT = 'T'; /* transposes */
  const float alpha = 1, beta = 1, beta0 = 0;
  int j, l, p;
  int K4 = K * 4;
  int CK = C + K;
  LIBXSMM_VLA_DECL(2, float, xgold, xgoldt, N * C);
  LIBXSMM_VLA_DECL(2, float, csgold, csgoldt, K * N);
  LIBXSMM_VLA_DECL(2, float, cogold, cogoldt, K * N);
  LIBXSMM_VLA_DECL(2, float, hgold, hgoldt, K * N);
  LIBXSMM_VLA_DECL(3, float, icfogold, icfogoldt, N, 4 * K);
  LIBXSMM_VLA_DECL(2, float, dxgold, dxgoldt, N * C);
  LIBXSMM_VLA_DECL(2, float, dhgold, dhgoldt, K * N);
  LIBXSMM_VLA_DECL(3, float, dicfogold, dicfogoldt, N, 4 * K);
  LIBXSMM_VLA_DECL(2, float, doutgold, doutgoldt, K * N);
  for (j = t-1; j >= 0; --j) {
#if defined(PROFILE)
    Gbl_eltwise_start = libxsmm_timer_tick();
#endif
    if (t-1 == j) {
      dout = NULL;
      dcs = dcsgold;
    } else {
      dout = &LIBXSMM_VLA_ACCESS(2, doutgold, j, 0, K * N);
      dcs = dcspgold;
    }
    if (0 == j) {
      csp = cspgold;
    } else {
      csp = &LIBXSMM_VLA_ACCESS(2, csgold, j-1, 0, K * N);
    }
    lstm_bwd_upd_eltwise_merged( N, K,
                                 &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 0,   N, 4 * K),
                                 &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, K,   N, 4 * K),
                                 &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 2*K, N, 4 * K),
                                 &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 3*K, N, 4 * K),
                                 csp,
                                 &LIBXSMM_VLA_ACCESS(2, cogold, j, 0, K * N),
                                 &LIBXSMM_VLA_ACCESS(2, dhgold, j, 0, K * N),
                                 dout,
                                 &LIBXSMM_VLA_ACCESS(3, dicfogold, j, 0, 0,   N, 4 * K),
                                 &LIBXSMM_VLA_ACCESS(3, dicfogold, j, 0, K,   N, 4 * K),
                                 &LIBXSMM_VLA_ACCESS(3, dicfogold, j, 0, 2*K, N, 4 * K),
                                 &LIBXSMM_VLA_ACCESS(3, dicfogold, j, 0, 3*K, N, 4 * K),
                                 dcspgold, dcs);
#if defined(PROFILE)
    Gbl_eltwise_end = libxsmm_timer_tick();
    Gbl_eltwise_total += libxsmm_timer_duration(Gbl_eltwise_start, Gbl_eltwise_end);
    Gbl_blas_start = libxsmm_timer_tick();
#endif
#if defined(TWO_GEMMS)
    if (j > 0) {
      /* compute dout */
      LIBXSMM_XBLAS_SYMBOL(float)(&transaT, &transb, &K, &N, &K4, &alpha, rgold, &K4, &LIBXSMM_VLA_ACCESS(3, dicfogold, j, 0, 0, N, 4 * K), &K4, &beta0, &LIBXSMM_VLA_ACCESS(2, doutgold, j-1, 0, K * N), &K);
    } else {
      /* compute dhp */
      LIBXSMM_XBLAS_SYMBOL(float)(&transaT, &transb, &K, &N, &K4, &alpha, rgold, &K4, &LIBXSMM_VLA_ACCESS(3, dicfogold, 0, 0, 0, N, 4 * K), &K4, &beta0, dhpgold, &K);
    }

    /* compute dx */
    LIBXSMM_XBLAS_SYMBOL(float)(&transaT, &transb, &C, &N, &K4, &alpha, wgold, &K4, &LIBXSMM_VLA_ACCESS(3, dicfogold, j, 0, 0, N, 4 * K), &K4, &beta, &LIBXSMM_VLA_ACCESS(2, dxgold, j, 0, N * C), &C);

    /* compute dw */
    LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transbT, &K4, &C, &N, &alpha, &LIBXSMM_VLA_ACCESS(3, dicfogold, j, 0, 0, N, 4 * K), &K4, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, N * C), &C, &beta, dwgold, &K4);

    /* compute dr */
    if (j == 0) {
      LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transbT, &K4, &K, &N, &alpha, &LIBXSMM_VLA_ACCESS(3, dicfogold, j, 0, 0, N, 4 * K), &K4, hpgold, &K, &beta, drgold, &K4);
    } else {
      LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transbT, &K4, &K, &N, &alpha, &LIBXSMM_VLA_ACCESS(3, dicfogold, j, 0, 0, N, 4 * K), &K4, &LIBXSMM_VLA_ACCESS(2, hgold, j-1, 0, K * N), &K, &beta, drgold, &K4);
    }
#else
    LIBXSMM_UNUSED(rgold); LIBXSMM_UNUSED(drgold);
    LIBXSMM_XBLAS_SYMBOL(float)(&transaT, &transb, &CK, &N, &K4, &alpha, wgold, &K4, &LIBXSMM_VLA_ACCESS(3, dicfogold, j, 0, 0, N, 4 * K), &K4, &beta0, dxhgold, &CK);
    matrix_copy_ld(C, N, C+K, dxhgold, &LIBXSMM_VLA_ACCESS(2, dxgold, j, 0, N * C));
    if (j > 0) {
      matrix_copy_ld(K, N, C+K, &(dxhgold[C]), &LIBXSMM_VLA_ACCESS(2, doutgold, j-1, 0, K * N));
    } else {
      matrix_copy_ld(K, N, C+K, &(dxhgold[C]), dhpgold);
    }

    /* Concatenate x and h */
    convert_nk_nck(N, C, C+K, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, N * C), xhgold);
    if (j == 0) {
      convert_nk_nck(N, K, C+K, hpgold, &(xhgold[C]));
    } else {
      convert_nk_nck(N, K, C+K, &LIBXSMM_VLA_ACCESS(2, hgold, j-1, 0, K * N), &(xhgold[C]));
    }
    LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transbT, &K4, &CK, &N, &alpha, &LIBXSMM_VLA_ACCESS(3, dicfogold, j, 0, 0, N, 4 * K), &K4, xhgold, &CK, &beta, dwgold, &K4);
#endif
#if defined(PROFILE)
    Gbl_blas_end = libxsmm_timer_tick();
    Gbl_blas_total += libxsmm_timer_duration(Gbl_blas_start, Gbl_blas_end);
#endif
    /* compute db */
#if defined(_OPENMP)
    LIBXSMM_OMP_VAR(p);
# pragma omp parallel for private(l, p)
#endif
    for (l = 0; l < K; l++) {
      for (p = 0; p < N; p++) {
        dbgold[l]       += LIBXSMM_VLA_ACCESS(3, dicfogold, j, p, l,       N, 4 * K);
        dbgold[l + K]   += LIBXSMM_VLA_ACCESS(3, dicfogold, j, p, l + K,   N, 4 * K);
        dbgold[l + 2*K] += LIBXSMM_VLA_ACCESS(3, dicfogold, j, p, l + 2*K, N, 4 * K);
        dbgold[l + 3*K] += LIBXSMM_VLA_ACCESS(3, dicfogold, j, p, l + 3*K, N, 4 * K);
      }
    }
  }
}

LIBXSMM_INLINE void gru_ref_fwd( int N, int C, int K, int t,
                  float *wi, float *wc, float *wf,
                  float *ri, float *rc, float *rf,
                  float *bi, float *bc, float *bf,
                  float *xt, float *hp, float *ht,
                  float *it, float *ct, float *ft, float *ot )
{
  const char transa = 'N', transb = 'N';   /* no transposes */
  const float alpha = 1, beta = 1;
  int j;
  LIBXSMM_VLA_DECL(2, float, x, xt, N * C);
  LIBXSMM_VLA_DECL(2, float, h, ht, K * N);
  LIBXSMM_VLA_DECL(2, float, i, it, K * N);
  LIBXSMM_VLA_DECL(2, float, c, ct, K * N);
  LIBXSMM_VLA_DECL(2, float, f, ft, K * N);
  LIBXSMM_VLA_DECL(2, float, o, ot, K * N);
  for (j = 0; j < t; ++j) {
    /* i_t = b_i */
    matrix_copy_bias(K, N, K, bi, &LIBXSMM_VLA_ACCESS(2, i, j, 0, K * N));
    /* i_t += W_i * x_t */
    LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &C, &alpha, wi, &K, &LIBXSMM_VLA_ACCESS(2, x, j, 0, N * C), &C, &beta, &LIBXSMM_VLA_ACCESS(2, i, j, 0, K * N), &K);
    /* i_t += R_i * h_{t-1} */
    if (0 == j) {
      LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, ri, &K, hp,                                       &K, &beta, &LIBXSMM_VLA_ACCESS(2, i, j, 0, K * N), &K);
    } else {
      LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, ri, &K, &LIBXSMM_VLA_ACCESS(2, h, j-1, 0, K * N), &K, &beta, &LIBXSMM_VLA_ACCESS(2, i, j, 0, K * N), &K);
    }
    /* i_t = sigmoid(i_t) */
    matrix_sigmoid(N*K, &LIBXSMM_VLA_ACCESS(2, i, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, i, j, 0, K * N));
    /* c_t = b_c */
    matrix_copy_bias(K, N, K, bc, &LIBXSMM_VLA_ACCESS(2, c, j, 0, K * N));
    /* c_t += W_c * x_t */
    LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &C, &alpha, wc, &K, &LIBXSMM_VLA_ACCESS(2, x, j, 0, N * C), &C, &beta, &LIBXSMM_VLA_ACCESS(2, c, j, 0, K * N), &K);
    /* c_t += R_c * h_{t-1} */
    if (0 == j) {
      LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rc, &K, hp,                                       &K, &beta, &LIBXSMM_VLA_ACCESS(2, c, j, 0, K * N), &K);
    } else {
      LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rc, &K, &LIBXSMM_VLA_ACCESS(2, h, j-1, 0, K * N), &K, &beta, &LIBXSMM_VLA_ACCESS(2, c, j, 0, K * N), &K);
    }
    /* c_t = sigmoid(c_t) */
    matrix_sigmoid(N*K, &LIBXSMM_VLA_ACCESS(2, c, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, c, j, 0, K * N));
    /* o_t = h_{t-1} . i_t */
    if (0 == j) {
      matrix_eltwise_mult(N*K, hp,                                       &LIBXSMM_VLA_ACCESS(2, i, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, o, j, 0, K * N));
    } else {
      matrix_eltwise_mult(N*K, &LIBXSMM_VLA_ACCESS(2, h, j-1, 0, K * N), &LIBXSMM_VLA_ACCESS(2, i, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, o, j, 0, K * N));
    }
    /* f_t = b_f */
    matrix_copy_bias(K, N, K, bf, &LIBXSMM_VLA_ACCESS(2, f, j, 0, K * N));
    /* f_t += W_f * x_t */
    LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &C, &alpha, wf, &K, &LIBXSMM_VLA_ACCESS(2, x, j, 0, N * C), &C, &beta, &LIBXSMM_VLA_ACCESS(2, f, j, 0, K * N), &K);
    /* f_t += R_f * o_t */
    LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rf, &K, &LIBXSMM_VLA_ACCESS(2, o, j, 0, K * N), &K, &beta, &LIBXSMM_VLA_ACCESS(2, f, j, 0, K * N), &K);
    /* f_t = tanh(f_t) */
    matrix_tanh(N*K, &LIBXSMM_VLA_ACCESS(2, f, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, f, j, 0, K * N));
    /* h_t = (1 - c_t) . f_t */
    matrix_complement  (N*K, &LIBXSMM_VLA_ACCESS(2, c, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, h, j, 0, K * N));
    matrix_eltwise_mult(N*K, &LIBXSMM_VLA_ACCESS(2, h, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, f, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, h, j, 0, K * N));
    /* h_t += c_t . h_{t-1} */
    if (0 == j) {
      matrix_eltwise_fma(N*K, &LIBXSMM_VLA_ACCESS(2, c, j, 0, K * N), hp,                                       &LIBXSMM_VLA_ACCESS(2, h, j, 0, K * N));
    } else {
      matrix_eltwise_fma(N*K, &LIBXSMM_VLA_ACCESS(2, c, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, h, j-1, 0, K * N), &LIBXSMM_VLA_ACCESS(2, h, j, 0, K * N));
    }
  }
}

LIBXSMM_INLINE void gru_ref_bwd_upd( int N, int C, int K, int t,
                      float *xt,  float *hpD,  float *ht,
                      float *it,  float *ct,   float *ft, float *ot,
                      float *wi,  float *wc,   float *wf,
                      float *ri,  float *rc,   float *rf,
                      float *dht, float *dw,   float *dr, float *db,
                      float *dxt, float *dhpD, float *scratch )
{
  const char transa = 'N', transb = 'N';   /* no transposes */
  const char transaT = 'T', transbT = 'T'; /* transposes */
  const float alpha = 1, beta = 1, beta0 = 0;
  int j, l, p;
  float *dwi = dw;
  float *dwc = &(dw[C*K]);
  float *dwf = &(dw[2*C*K]);
  float *dri = dr;
  float *drc = &(dr[K*K]);
  float *drf = &(dr[2*K*K]);
  float *dbi = db;
  float *dbc = &(db[K]);
  float *dbf = &(db[2*K]);
  float *deltaD = scratch;
  float *doutD  = &(scratch[N*K]);
  float *diD    = &(scratch[2*N*K]);
  float *dcD    = &(scratch[3*N*K]);
  float *dfD    = &(scratch[4*N*K]);
  float *doD    = &(scratch[5*N*K]);
  LIBXSMM_VLA_DECL(3, float, x,     xt,     N, C);
  LIBXSMM_VLA_DECL(2, float, hp,    hpD,    K);
  LIBXSMM_VLA_DECL(3, float, h,     ht,     N, K);
  LIBXSMM_VLA_DECL(3, float, i,     it,     N, K);
  LIBXSMM_VLA_DECL(3, float, c,     ct,     N, K);
  LIBXSMM_VLA_DECL(3, float, f,     ft,     N, K);
  LIBXSMM_VLA_DECL(3, float, o,     ot,     N, K);
  LIBXSMM_VLA_DECL(3, float, dx,    dxt,    N, C);
  LIBXSMM_VLA_DECL(2, float, dhp,   dhpD,   K);
  LIBXSMM_VLA_DECL(3, float, dh,    dht,    N, K);
  LIBXSMM_VLA_DECL(2, float, di,    diD,    K);
  LIBXSMM_VLA_DECL(2, float, dc,    dcD,    K);
  LIBXSMM_VLA_DECL(2, float, df,    dfD,    K);
  LIBXSMM_VLA_DECL(2, float, dp,    doD,    K);
  LIBXSMM_VLA_DECL(2, float, dout,  doutD,  K);
  LIBXSMM_VLA_DECL(2, float, delta, deltaD, K);
  for (j = t-1; j >= 0; j--) {
#if defined(_OPENMP)
    LIBXSMM_OMP_VAR(p);
#   pragma omp parallel for private(l, p) LIBXSMM_OPENMP_COLLAPSE(2)
#endif
    for (l = 0; l < N; l++) {
      for (p = 0; p < K; p++) {
        if (t-1 == j) {
          LIBXSMM_VLA_ACCESS(2, delta, l, p, K) = LIBXSMM_VLA_ACCESS(3, dh, t-1, l, p, N, K);
        } else {
          LIBXSMM_VLA_ACCESS(2, delta, l, p, K) = LIBXSMM_VLA_ACCESS(3, dh, j,   l, p, N, K) + LIBXSMM_VLA_ACCESS(2, dout, l, p, K);
        }
        /* df = delta . (1 - c_t) . (1 - (f_t . f_t)) */
        LIBXSMM_VLA_ACCESS(2, df, l, p, K) = LIBXSMM_VLA_ACCESS(2, delta, l, p, K) * (1.0f - LIBXSMM_VLA_ACCESS(3, c, j, l, p, N, K)) * (1.0f - (LIBXSMM_VLA_ACCESS(3, f, j, l, p, N, K) * LIBXSMM_VLA_ACCESS(3, f, j, l, p, N, K)));
        /* dc = delta . (h_{t-1} - f_t) . c_t . (1 - c_t) */
        if (0 == j) {
          LIBXSMM_VLA_ACCESS(2, dc, l, p, K) = LIBXSMM_VLA_ACCESS(2, delta, l, p, K) * (LIBXSMM_VLA_ACCESS(2, hp, l, p, K) -        LIBXSMM_VLA_ACCESS(3, f, j, l, p, N, K)) * LIBXSMM_VLA_ACCESS(3, c, j, l, p, N, K) * (1.0f - LIBXSMM_VLA_ACCESS(3, c, j, l, p, N, K));
        } else {
          LIBXSMM_VLA_ACCESS(2, dc, l, p, K) = LIBXSMM_VLA_ACCESS(2, delta, l, p, K) * (LIBXSMM_VLA_ACCESS(3, h, j-1, l, p, N, K) - LIBXSMM_VLA_ACCESS(3, f, j, l, p, N, K)) * LIBXSMM_VLA_ACCESS(3, c, j, l, p, N, K) * (1.0f - LIBXSMM_VLA_ACCESS(3, c, j, l, p, N, K));
        }
      }
    }
    /* do = {R_f}^T * df */
    LIBXSMM_XBLAS_SYMBOL(float)(&transaT, &transb, &K, &N, &K, &alpha, rf, &K, dfD, &K, &beta0, doD, &K);
    /* di = do . h_{t-1} . i_t . (1 - i_t) */
    if (0 == j) {
#if defined(_OPENMP)
#     pragma omp parallel for private(l, p) LIBXSMM_OPENMP_COLLAPSE(2)
#endif
      for (l = 0; l < N; l++) {
        for (p = 0; p < K; p++) {
          LIBXSMM_VLA_ACCESS(2, di, l, p, K) = LIBXSMM_VLA_ACCESS(2, dp, l, p, K) * LIBXSMM_VLA_ACCESS(2, hp, l, p, K)        * LIBXSMM_VLA_ACCESS(3, i, 0, l, p, N, K) * (1.0f - LIBXSMM_VLA_ACCESS(3, i, 0, l, p, N, K));
        }
      }
    } else {
#if defined(_OPENMP)
#     pragma omp parallel for private(l, p) LIBXSMM_OPENMP_COLLAPSE(2)
#endif
      for (l = 0; l < N; l++) {
        for (p = 0; p < K; p++) {
          LIBXSMM_VLA_ACCESS(2, di, l, p, K) = LIBXSMM_VLA_ACCESS(2, dp, l, p, K) * LIBXSMM_VLA_ACCESS(3, h, j-1, l, p, N, K) * LIBXSMM_VLA_ACCESS(3, i, j, l, p, N, K) * (1.0f - LIBXSMM_VLA_ACCESS(3, i, j, l, p, N, K));
        }
      }
    }
    /* dx_t  = {W_i}^T * di */
    LIBXSMM_XBLAS_SYMBOL(float)(&transaT, &transb, &C, &N, &K, &alpha, wi, &K, diD, &K, &beta0, &LIBXSMM_VLA_ACCESS(3, dx, j, 0, 0, N, C), &C);
    /* dx_t += {W_c}^T * dc */
    LIBXSMM_XBLAS_SYMBOL(float)(&transaT, &transb, &C, &N, &K, &alpha, wc, &K, dcD, &K, &beta,  &LIBXSMM_VLA_ACCESS(3, dx, j, 0, 0, N, C), &C);
    /* dx_t += {W_f}^T * df */
    LIBXSMM_XBLAS_SYMBOL(float)(&transaT, &transb, &C, &N, &K, &alpha, wf, &K, dfD, &K, &beta,  &LIBXSMM_VLA_ACCESS(3, dx, j, 0, 0, N, C), &C);
    /* dh_{t-1}  = {R_i}^T * di */
    /* dh_{t-1} += {R_c}^T * dc */
    if (0 == j) {
      LIBXSMM_XBLAS_SYMBOL(float)(&transaT, &transb, &K, &N, &K, &alpha, ri, &K, diD, &K, &beta0, dhpD, &K);
      LIBXSMM_XBLAS_SYMBOL(float)(&transaT, &transb, &K, &N, &K, &alpha, rc, &K, dcD, &K, &beta,  dhpD, &K);
    } else {
      LIBXSMM_XBLAS_SYMBOL(float)(&transaT, &transb, &K, &N, &K, &alpha, ri, &K, diD, &K, &beta0, doutD, &K);
      LIBXSMM_XBLAS_SYMBOL(float)(&transaT, &transb, &K, &N, &K, &alpha, rc, &K, dcD, &K, &beta,  doutD, &K);
    }
    /* dh_{t-1} += do * i_t + delta * c_t */
    if (0 == j) {
#if defined(_OPENMP)
#     pragma omp parallel for private(l, p) LIBXSMM_OPENMP_COLLAPSE(2)
#endif
      for (l = 0; l < N; l++) {
        for (p = 0; p < K; p++) {
          LIBXSMM_VLA_ACCESS(2, dhp,  l, p, K) += LIBXSMM_VLA_ACCESS(2, dp, l, p, K) * LIBXSMM_VLA_ACCESS(3, i, j, l, p, N, K) + LIBXSMM_VLA_ACCESS(2, delta, l, p, K) * LIBXSMM_VLA_ACCESS(3, c, j, l, p, N, K);
        }
      }
    } else {
#if defined(_OPENMP)
#     pragma omp parallel for private(l, p) LIBXSMM_OPENMP_COLLAPSE(2)
#endif
      for (l = 0; l < N; l++) {
        for (p = 0; p < K; p++) {
          LIBXSMM_VLA_ACCESS(2, dout, l, p, K) += LIBXSMM_VLA_ACCESS(2, dp, l, p, K) * LIBXSMM_VLA_ACCESS(3, i, j, l, p, N, K) + LIBXSMM_VLA_ACCESS(2, delta, l, p, K) * LIBXSMM_VLA_ACCESS(3, c, j, l, p, N, K);
        }
      }
    }
    /* dw_i += di * {x_t}^T */
    /* dw_c += dc * {x_t}^T */
    /* dw_f += df * {x_t}^T */
    LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transbT, &K, &C, &N, &alpha, diD, &K, &LIBXSMM_VLA_ACCESS(3, x, j, 0, 0, N, C), &C, &beta, dwi, &K);
    LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transbT, &K, &C, &N, &alpha, dcD, &K, &LIBXSMM_VLA_ACCESS(3, x, j, 0, 0, N, C), &C, &beta, dwc, &K);
    LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transbT, &K, &C, &N, &alpha, dfD, &K, &LIBXSMM_VLA_ACCESS(3, x, j, 0, 0, N, C), &C, &beta, dwf, &K);
    /* dr_i += di * {o_t}^T */
    /* dr_c += dc * {o_t}^T */
    /* dr_f += df * {h_{t-1}}^T */
    LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transbT, &K, &K, &N, &alpha, diD, &K, &LIBXSMM_VLA_ACCESS(3, o, j, 0, 0, N, K), &K, &beta, dri, &K);
    LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transbT, &K, &K, &N, &alpha, dcD, &K, &LIBXSMM_VLA_ACCESS(3, o, j, 0, 0, N, K), &K, &beta, drc, &K);
    if (0 == j) {
      LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transbT, &K, &K, &N, &alpha, dfD, &K, &LIBXSMM_VLA_ACCESS(2, hp, 0, 0, K),        &K, &beta, drf, &K);
    } else {
      LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transbT, &K, &K, &N, &alpha, dfD, &K, &LIBXSMM_VLA_ACCESS(3, h, j-1, 0, 0, N, K), &K, &beta, drf, &K);
    }
    /* compute db */
#if defined(_OPENMP)
#   pragma omp parallel for private(l, p)
#endif
    for (l = 0; l < K; l++) {
      for (p = 0; p < N; p++) {
        dbi[l] += LIBXSMM_VLA_ACCESS(2, di, p, l, K);
        dbc[l] += LIBXSMM_VLA_ACCESS(2, dc, p, l, K);
        dbf[l] += LIBXSMM_VLA_ACCESS(2, df, p, l, K);
      }
    }
  }
}

