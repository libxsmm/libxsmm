/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
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
int index[16];
SIMDTYPE_INT32 vindex = _MM_SETZERO_INT32();

LIBXSMM_UNUSED(nthreads);
LIBXSMM_UNUSED(tid);

kb = block_id / handle->mb;
mb = block_id % handle->mb;
if ('T' == transA || 't' == transA) {
  int kk;
  block_offset_base = mb * handle->bm;
  block_offset = block_offset_base + kb * handle->m * handle->bk;
  for (kk = 0; kk < SIMD_WIDTH_FP32; kk++) index[kk] = kk*handle->m;
  vindex = _MM_LOADU_INT32(index);
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
  int ncols_aligned = ncols / (4*SIMD_WIDTH_FP32)*(4*SIMD_WIDTH_FP32);
  int ncols_aligned_2 = ncols / (SIMD_WIDTH_FP32)*(SIMD_WIDTH_FP32);
  const float * input_ptr = A + block_offset;
  uint16_t * rowidx_ptr = slice.rowidx;
  uint16_t * colidx_ptr = slice.colidx;
  float    * values_ptr = (float *)(slice.values);
  SIMDTYPE_FP32 vzero = _MM_SET1_FP32(0.0);
  uint16_t cnt = 0;
#if (1 == SIMD_WIDTH_FP32)
  ncols_aligned = 0;
  ncols_aligned_2 = 0;
#endif
  for (i = 0; i < nrows; i++) {
    rowidx_ptr[i] = cnt;
    if ('T' == transA || 't' == transA) {
      for (k = 0; k < ncols_aligned; k += 4*SIMD_WIDTH_FP32) {
        SIMDTYPE_FP32 v1 = _MM_GATHER_FP32(input_ptr + k*handle->m + i, vindex, 4);
        SIMDTYPE_FP32 v2 = _MM_GATHER_FP32(input_ptr + (k+SIMD_WIDTH_FP32)*handle->m + i, vindex, 4);
        SIMDTYPE_FP32 v3 = _MM_GATHER_FP32(input_ptr + (k+2*SIMD_WIDTH_FP32)*handle->m + i, vindex, 4);
        SIMDTYPE_FP32 v4 = _MM_GATHER_FP32(input_ptr + (k+3*SIMD_WIDTH_FP32)*handle->m + i, vindex, 4);
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
        SIMDTYPE_FP32 v1 = _MM_GATHER_FP32(input_ptr + k*handle->m + i, vindex, 4);
        SIMDMASKTYPE_FP32 m1 = _MM_CMPNEQ_FP32(v1, vzero);
        COMPRESS_FP32(v1, k, m1, cnt);
      }
      for (k = ncols_aligned_2; k < ncols; k++) {
        const float v1 = input_ptr[i + k*handle->m];
        const int m1 = LIBXSMM_FEQ(0, v1) ? 0 : 1;
        if (m1) { colidx_ptr[cnt] = (uint16_t)k; values_ptr[cnt] = v1; cnt++; }
      }
    }
    else {
      for (k = 0; k < ncols_aligned; k += 4*SIMD_WIDTH_FP32) {
        SIMDTYPE_FP32 v1, v2, v3, v4;
        SIMDMASKTYPE_FP32 m1, m2, m3, m4;
        v1 = _MM_LOADU_FP32(input_ptr + i*handle->k + k);
        _MM_PREFETCH((char *)(input_ptr + (i+2)*handle->k + k), _MM_HINT_T0);
        v2 = _MM_LOADU_FP32(input_ptr + i*handle->k + k + SIMD_WIDTH_FP32);
        _MM_PREFETCH((char *)(input_ptr + (i+2)*handle->k + k + SIMD_WIDTH_FP32), _MM_HINT_T0);
        v3 = _MM_LOADU_FP32(input_ptr + i*handle->k + k + 2*SIMD_WIDTH_FP32);
        _MM_PREFETCH((char *)(input_ptr + (i+2)*handle->k + k + 2*SIMD_WIDTH_FP32), _MM_HINT_T0);
        v4 = _MM_LOADU_FP32(input_ptr + i*handle->k + k + 3*SIMD_WIDTH_FP32);
        _MM_PREFETCH((char *)(input_ptr + (i+2)*handle->k + k + 3*SIMD_WIDTH_FP32), _MM_HINT_T0);
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
        v1 = _MM_LOADU_FP32(input_ptr + i*handle->k + k);
        _MM_PREFETCH((char *)(input_ptr + (i+2)*handle->k + k), _MM_HINT_T0);
        m1 = _MM_CMPNEQ_FP32(v1, vzero);
        COMPRESS_FP32(v1, k, m1, cnt);
      }
      for (k = ncols_aligned_2; k < ncols; k++) {
        const float v1 = input_ptr[i*handle->k + k];
        const int m1 = LIBXSMM_FEQ(0, v1) ? 0 : 1;
        if (m1) { colidx_ptr[cnt] = (uint16_t)k; values_ptr[cnt] = v1; cnt++; }
      }
    }
  }
  rowidx_ptr[nrows] = cnt;
#if 0
  printf("cnt: %d\n", cnt);
  for (i = 0; i <= nrows; i++) {
    int j;
    for (j = slice.rowidx[i]; j < slice.rowidx[i+1]; j++) {
      printf("(%d, %d): %f ", i, colidx_ptr[j], values_ptr[j]);
    }
  }
#endif
}

