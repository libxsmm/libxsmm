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
SIMDTYPE_INT32 vindex = _MM_SETZERO_INT32();
int idx_array[16];
#endif
int block_offset_base, block_offset;

LIBXSMM_UNUSED(nthreads);
LIBXSMM_UNUSED(tid);

kb = block_id / handle->mb;
mb = block_id % handle->mb;
if ('T' == transa || 't' == transa) {
#if SIMD_WIDTH_FP32 > 1
  int kk;
  for (kk = 0; kk < SIMD_WIDTH_FP32; kk++) idx_array[kk] = kk * handle->m;
  vindex = _MM_LOADU_INT32(idx_array);
#endif
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
  const float * input_ptr = a + block_offset;
  uint16_t * rowidx_ptr = slice.rowidx;
  uint16_t * colidx_ptr = slice.colidx;
  float    * values_ptr = (float *)(slice.values);
  uint16_t cnt = 0;
#if SIMD_WIDTH_FP32 > 1
  const SIMDTYPE_FP32 vzero = _MM_SETZERO_FP32();
  const int ncols_aligned = ncols / (4*SIMD_WIDTH_FP32)*(4*SIMD_WIDTH_FP32);
  const int ncols_aligned_2 = ncols / (SIMD_WIDTH_FP32)*(SIMD_WIDTH_FP32);
#else
  const int ncols_aligned_2 = 0;
#endif
  for (i = 0; i < nrows; i++) {
    rowidx_ptr[i] = cnt;
    if ('T' == transa || 't' == transa) {
#if SIMD_WIDTH_FP32 > 1
      for (k = 0; k < ncols_aligned; k += 4*SIMD_WIDTH_FP32) {
        SIMDTYPE_FP32 v1 = _MM_GATHER_FP32(input_ptr + (size_t)k * handle->m + i, vindex, 4);
        SIMDTYPE_FP32 v2 = _MM_GATHER_FP32(input_ptr + ((size_t)k+1*SIMD_WIDTH_FP32) * handle->m + i, vindex, 4);
        SIMDTYPE_FP32 v3 = _MM_GATHER_FP32(input_ptr + ((size_t)k+2*SIMD_WIDTH_FP32) * handle->m + i, vindex, 4);
        SIMDTYPE_FP32 v4 = _MM_GATHER_FP32(input_ptr + ((size_t)k+3*SIMD_WIDTH_FP32) * handle->m + i, vindex, 4);
        SIMDMASKTYPE_FP32 m1 = _MM_CMPNEQ_FP32(v1, vzero);
        SIMDMASKTYPE_FP32 m2 = _MM_CMPNEQ_FP32(v2, vzero);
        SIMDMASKTYPE_FP32 m3 = _MM_CMPNEQ_FP32(v3, vzero);
        SIMDMASKTYPE_FP32 m4 = _MM_CMPNEQ_FP32(v4, vzero);
        COMPRESS_FP32(v1, k, m1, cnt);
        COMPRESS_FP32(v2, k + SIMD_WIDTH_FP32, m2, cnt);
        COMPRESS_FP32(v3, k + 2*SIMD_WIDTH_FP32, m3, cnt);
        COMPRESS_FP32(v4, k + 3*SIMD_WIDTH_FP32, m4, cnt);
      }
      for (k = ncols_aligned; k < ncols_aligned_2; k += SIMD_WIDTH_FP32) {
        SIMDTYPE_FP32 v1 = _MM_GATHER_FP32(input_ptr + (size_t)k * handle->m + i, vindex, 4);
        SIMDMASKTYPE_FP32 m1 = _MM_CMPNEQ_FP32(v1, vzero);
        COMPRESS_FP32(v1, k, m1, cnt);
      }
#endif
      for (k = ncols_aligned_2; k < ncols; k++) {
        const float v1 = input_ptr[i + k*handle->m];
        const int m1 = LIBXSMM_FEQ(0, v1) ? 0 : 1;
        if (m1) { colidx_ptr[cnt] = (uint16_t)k; values_ptr[cnt] = v1; cnt++; }
      }
    }
    else {
#if SIMD_WIDTH_FP32 > 1
      for (k = 0; k < ncols_aligned; k += 4*SIMD_WIDTH_FP32) {
        SIMDTYPE_FP32 v1, v2, v3, v4;
        SIMDMASKTYPE_FP32 m1, m2, m3, m4;
        v1 = _MM_LOADU_FP32(input_ptr + ((size_t)i)   * handle->k + (size_t)k);
        _MM_PREFETCH((char*)input_ptr + ((size_t)i+2) * handle->k + (size_t)k, _MM_HINT_T0);
        v2 = _MM_LOADU_FP32(input_ptr + ((size_t)i)   * handle->k + (size_t)k + (size_t)SIMD_WIDTH_FP32);
        _MM_PREFETCH((char*)input_ptr + ((size_t)i+2) * handle->k + (size_t)k + (size_t)SIMD_WIDTH_FP32, _MM_HINT_T0);
        v3 = _MM_LOADU_FP32(input_ptr + ((size_t)i)   * handle->k + (size_t)k + (size_t)2 * SIMD_WIDTH_FP32);
        _MM_PREFETCH((char*)input_ptr + ((size_t)i+2) * handle->k + (size_t)k + (size_t)2 * SIMD_WIDTH_FP32, _MM_HINT_T0);
        v4 = _MM_LOADU_FP32(input_ptr + ((size_t)i)   * handle->k + (size_t)k + (size_t)3 * SIMD_WIDTH_FP32);
        _MM_PREFETCH((char*)input_ptr + ((size_t)i+2) * handle->k + (size_t)k + (size_t)3 * SIMD_WIDTH_FP32, _MM_HINT_T0);
        m1 = _MM_CMPNEQ_FP32(v1, vzero);
        m2 = _MM_CMPNEQ_FP32(v2, vzero);
        m3 = _MM_CMPNEQ_FP32(v3, vzero);
        m4 = _MM_CMPNEQ_FP32(v4, vzero);
        COMPRESS_FP32(v1, k, m1, cnt);
        COMPRESS_FP32(v2, k + SIMD_WIDTH_FP32, m2, cnt);
        COMPRESS_FP32(v3, k + 2*SIMD_WIDTH_FP32, m3, cnt);
        COMPRESS_FP32(v4, k + 3*SIMD_WIDTH_FP32, m4, cnt);
      }
      for (k = ncols_aligned; k < ncols_aligned_2; k += SIMD_WIDTH_FP32) {
        SIMDTYPE_FP32 v1;
        SIMDMASKTYPE_FP32 m1;
        v1 = _MM_LOADU_FP32(input_ptr + ((size_t)i)   * handle->k + (size_t)k);
        _MM_PREFETCH((char*)input_ptr + ((size_t)i+2) * handle->k + (size_t)k, _MM_HINT_T0);
        m1 = _MM_CMPNEQ_FP32(v1, vzero);
        COMPRESS_FP32(v1, k, m1, cnt);
      }
#endif
      for (k = ncols_aligned_2; k < ncols; k++) {
        const float v1 = input_ptr[i*handle->k + k];
        const int m1 = LIBXSMM_FEQ(0, v1) ? 0 : 1;
        if (m1) { colidx_ptr[cnt] = (uint16_t)k; values_ptr[cnt] = v1; cnt++; }
      }
    }
  }
  rowidx_ptr[nrows] = cnt;
}

