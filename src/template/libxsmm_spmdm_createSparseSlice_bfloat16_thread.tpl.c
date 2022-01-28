/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Nadathur Satish (Intel Corp.)
******************************************************************************/

int i, k;
int mb, kb;
#if SIMD_WIDTH_FP32 == 8
const __m256i *const shufmasks = internal_spmdm_shufmasks_32;
#endif
#if SIMD_WIDTH_FP32 > 1
const __m256i *const shufmasks2 = internal_spmdm_shufmasks_16;
#endif
int block_offset_base, block_offset;

LIBXSMM_UNUSED(nthreads);
LIBXSMM_UNUSED(tid);

kb = block_id / handle->mb;
mb = block_id % handle->mb;

if ('T' == transa || 't' == transa) {
  block_offset_base = mb * handle->bm;
  block_offset = block_offset_base + kb * handle->m * handle->bk;
}
else {
  block_offset_base = kb * handle->bk;
  block_offset = block_offset_base + mb * handle->k * handle->bm;
}
{
  libxsmm_CSR_sparseslice slice = libxsmm_output_csr_a[kb*handle->mb + mb];
  int nrows = ((mb + 1)*handle->bm > handle->m)?(handle->m - (mb)*handle->bm):handle->bm;
  int ncols = ((kb + 1)*handle->bk > handle->k)?(handle->k - (kb)*handle->bk):handle->bk;
  /*printf("nrows: %d, ncols: %d\n", nrows, ncols);*/
  const uint16_t * input_ptr = a + block_offset;
  uint16_t * rowidx_ptr = slice.rowidx;
  uint16_t * colidx_ptr = slice.colidx;
  float * values_ptr = (float *)(slice.values);
  uint16_t cnt = 0;
#if SIMD_WIDTH_FP32 > 1
  const SIMDTYPE_INT32 vzero = _MM_SETZERO_INT32();
  const SIMDTYPE_FP32 vzerof = _MM_SETZERO_FP32();
  const int ncols_aligned = ncols / (4*SIMD_WIDTH_FP32)*(4*SIMD_WIDTH_FP32);
#else
  const int ncols_aligned = 0;
#endif
  for (i = 0; i < nrows; i++) {
    rowidx_ptr[i] = cnt;
    if ('T' == transa || 't' == transa) {
#if SIMD_WIDTH_FP32 > 1
      for (k = 0; k < ncols_aligned; k += 4*SIMD_WIDTH_FP32) {
        int vals[32];
        int kk;
        for (kk = 0; kk < 4*SIMD_WIDTH_FP32; kk += 2) { vals[kk/2] = (int)input_ptr[(k+kk)*handle->m + i]; vals[kk/2] |= ((int)(input_ptr[(k+kk+1)*handle->m + i]) << 16); }
        {
          SIMDTYPE_INT32 v1tmp = _MM_LOADU_INT32(vals);
          SIMDTYPE_INT32 v2tmp = _MM_LOADU_INT32(vals + SIMD_WIDTH_FP32);
          SIMDTYPE_FP32 v1, v2, v3, v4;
          SIMDMASKTYPE_FP32 m1, m2, m3, m4;
          EXPAND_BFLOAT16(v1tmp, v1, v2);
          EXPAND_BFLOAT16(v2tmp, v3, v4);
          m1 = _MM_CMPNEQ_FP32(v1, vzerof);
          m2 = _MM_CMPNEQ_FP32(v2, vzerof);
          m3 = _MM_CMPNEQ_FP32(v3, vzerof);
          m4 = _MM_CMPNEQ_FP32(v4, vzerof);
          COMPRESS_FP32(v1, k, m1, cnt);
          COMPRESS_FP32(v2, k + SIMD_WIDTH_FP32, m2, cnt);
          COMPRESS_FP32(v3, k + 2*SIMD_WIDTH_FP32, m3, cnt);
          COMPRESS_FP32(v4, k + 3*SIMD_WIDTH_FP32, m4, cnt);
        }
      }
#endif
      for (k = ncols_aligned; k < ncols; k++) {
        uint16_t v1tmp = input_ptr[k*handle->m + i];
        union {int i; float f; } v1tmp_int;
        v1tmp_int.i = v1tmp;
        v1tmp_int.i <<= 16;
        {
          const int m1 = LIBXSMM_FEQ(0, v1tmp_int.f) ? 0 : 1;
          if (m1) { colidx_ptr[cnt] = (uint16_t)k; values_ptr[cnt] = v1tmp_int.f; cnt++; }
        }
      }
    }
    else {
#if SIMD_WIDTH_FP32 > 1
      for (k = 0; k < ncols_aligned; k += 4*SIMD_WIDTH_FP32) {
        SIMDTYPE_INT32 v1tmp, v2tmp;
        SIMDTYPE_FP32 v1, v2, v3, v4;
        SIMDMASKTYPE_FP32 m1, m2, m3, m4;
        v1tmp = _MM_LOADU_INT32((const SIMDTYPE_INT32*)(input_ptr + (size_t)i*handle->k + k));
        _MM_PREFETCH((char *)(input_ptr + ((size_t)i+2)*handle->k + k), _MM_HINT_T0);
        v2tmp = _MM_LOADU_INT32((const SIMDTYPE_INT32*)(input_ptr + (size_t)i*handle->k + k + 2*SIMD_WIDTH_FP32));
        _MM_PREFETCH((char *)(input_ptr + ((size_t)i+2)*handle->k + k + SIMD_WIDTH_FP32), _MM_HINT_T0);
        EXPAND_BFLOAT16(v1tmp, v1, v2);
        EXPAND_BFLOAT16(v2tmp, v3, v4);
        m1 = _MM_CMPNEQ_FP32(v1, vzerof);
        m2 = _MM_CMPNEQ_FP32(v2, vzerof);
        m3 = _MM_CMPNEQ_FP32(v3, vzerof);
        m4 = _MM_CMPNEQ_FP32(v4, vzerof);
        COMPRESS_FP32(v1, k, m1, cnt);
        COMPRESS_FP32(v2, k + SIMD_WIDTH_FP32, m2, cnt);
        COMPRESS_FP32(v3, k + 2*SIMD_WIDTH_FP32, m3, cnt);
        COMPRESS_FP32(v4, k + 3*SIMD_WIDTH_FP32, m4, cnt);
      }
#endif
      for (k = ncols_aligned; k < ncols; k++) {
        uint16_t v1tmp = input_ptr[i*handle->k + k];
        union {int i; float f; } v1tmp_int;
        v1tmp_int.i = v1tmp;
        v1tmp_int.i <<= 16;
        {
          int m1 = LIBXSMM_FEQ(0, v1tmp_int.f) ? 0 : 1;
          if (m1) { colidx_ptr[cnt] = (uint16_t)k; values_ptr[cnt] = v1tmp_int.f; cnt++; }
        }
      }
    }
  }
  rowidx_ptr[nrows] = cnt;
}

