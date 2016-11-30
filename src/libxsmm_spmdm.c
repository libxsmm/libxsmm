/******************************************************************************
** Copyright (c) 2016, Intel Corporation                                     **
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

/* NOTE: This code currently ignores alpha, beta and trans inputs to the matrix multiply */
#include <libxsmm_spmdm.h>
#include <libxsmm_intrinsics_x86.h>
#include <libxsmm_malloc.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXSMM_SPMDM_MALLOC_INTRINSIC)
# define LIBXSMM_SPMDM_MALLOC_INTRINSIC
#endif
#if defined(LIBXSMM_SPMDM_MALLOC_INTRINSIC)
# define LIBXSMM_SPMDM_MALLOC(SIZE, ALIGNMENT) _mm_malloc(SIZE, ALIGNMENT)
# define LIBXSMM_SPMDM_FREE(BUFFER) _mm_free((void*)(BUFFER))
#else
# define LIBXSMM_SPMDM_MALLOC(SIZE, ALIGNMENT) libxsmm_aligned_malloc(SIZE, -(ALIGNMENT))
# define LIBXSMM_SPMDM_FREE(BUFFER) libxsmm_free(BUFFER)
#endif

#ifndef LIBXSMM_STATIC_TARGET_ARCH
#error "LIBXSMM_STATIC_TARGET_ARCH undefined"
#endif

#if LIBXSMM_STATIC_TARGET_ARCH==LIBXSMM_X86_AVX512_CORE
#define SIMD_WIDTH_FP32 (16)
#define SIMDTYPE_FP32 __m512
#define SIMDTYPE_INT32 __m512i
#define SIMDMASKTYPE_FP32 __mmask16
#define _MM_SETZERO_FP32 _mm512_setzero_ps
#define _MM_SETZERO_INT32 _mm512_setzero_epi32
#define _MM_SET1_FP32 _mm512_set1_ps
#define _MM_SET1_INT32 _mm512_set1_epi32
#define _MM_SET1_INT16 _mm512_set1_epi16
#define _MM_SET_INT32 _mm512_set_epi32
#define _MM_LOAD_FP32 _mm512_load_ps
#define _MM_LOADU_FP32 _mm512_loadu_ps
#define _MM_LOAD_INT32 _mm512_load_epi32
#define _MM_STORE_INT32 _mm512_store_epi32
#define _MM_LOADU_INT32(x) _mm512_loadu_si512( (void const *)(x))
#define _MM_GATHER_INT32(Addr, idx, scale) _mm512_i32gather_epi32((idx), (Addr), (scale))
#define _MM_GATHER_FP32(Addr, idx, scale) _mm512_i32gather_ps((idx), (Addr), (scale))
#define _MM_CMPNEQ_FP32(v1,v2) _mm512_cmp_ps_mask(v1,v2,12)
#define _MM_STORE_FP32 _mm512_store_ps
#define _MM_ADD_FP32 _mm512_add_ps
#define _MM_FMADD_FP32 _mm512_fmadd_ps
#define _MM_PREFETCH(x, y) _mm_prefetch(x, y)

LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE __m256i internal_spmdm_shufmasks_32[256];
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE __m256i internal_spmdm_shufmasks_16[256];

LIBXSMM_INLINE LIBXSMM_RETARGETABLE void _mm512_print(__m512 a, char * s)
{
  float *v=(float *)(&a);
  int i;
  printf("[%8s(%3lud)]: ", s, (size_t)16);
  for(i=0; i < 16; i++)
    printf("%4f ", v[i]);
  printf("\n");
}

LIBXSMM_INLINE LIBXSMM_RETARGETABLE void _mm512i_print(__m512i a, char * s)
{
  int *v=(int *)(&a);
  int i;
  printf("[%8s(%3lud)]: ", s, (size_t)16);
  for(i=0; i < 16; i++)
    printf("%4d ", v[i]);
  printf("\n");
}

LIBXSMM_INLINE LIBXSMM_RETARGETABLE void _mm512i_epi16_print(__m512i a, char * s)
{
  uint16_t *v=(uint16_t*)(&a);
  int i;
  printf("[%8s(%3lud)]: ", s, (size_t)32);
  for(i=0; i < 32; i++)
    printf("%4d ", v[i]);
  printf("\n");
}

LIBXSMM_INLINE LIBXSMM_RETARGETABLE void _mm256i_epi16_print(__m256i a, char * s)
{
  uint16_t *v=(uint16_t*)(&a);
  int i;
  printf("[%8s(%3lud)]: ", s, (size_t)16);
  for(i=0; i < 16; i++)
    printf("%4d ", v[i]);
  printf("\n");
}


#define COMPRESS_FP32(v, k, m, cnt) \
{ \
  _mm512_mask_compressstoreu_ps(values_ptr + (cnt), m, v); \
  { \
    __m256i vk1 = _mm256_set1_epi16((short)(k)); \
    __m256i vk2 = _mm256_set1_epi16((short)((k) + 8)); \
    __m256i v_idx = _mm256_add_epi32(vk1, _mm256_load_si256(&shufmasks2[(m)&0xFF])); \
    __m256i v_idx_2 = _mm256_add_epi32(vk2, _mm256_load_si256(&shufmasks2[((m)>>8)&0xFF])); \
    _mm256_storeu_si256((__m256i *)(colidx_ptr + (cnt)), v_idx); \
    cnt = (unsigned short)((cnt) + _mm_popcnt_u32((m)&0xFF)); \
    _mm256_storeu_si256((__m256i *)(colidx_ptr + (cnt)), v_idx_2); \
    cnt = (unsigned short)((cnt) + _mm_popcnt_u32(((m)>>8)&0xFF)); \
  } \
}

#define EXPAND_BFLOAT16(v, vlo_final, vhi_final) \
  { \
  const __m512i vlo = _mm512_unpacklo_epi16(vzero, v); \
  const __m512i vhi = _mm512_unpackhi_epi16(vzero, v); \
  const __m512i permmask1 = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0); \
  const __m512i permmask2 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4); \
  vlo_final = _mm512_castsi512_ps(_mm512_permutex2var_epi64(vlo, permmask1, vhi)); \
  vhi_final = _mm512_castsi512_ps(_mm512_permutex2var_epi64(vlo, permmask2, vhi)); \
  }

#define COMPRESS_BFLOAT16(vlo, vhi, v) \
  { \
  const __m512i permmask1 = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0); \
  const __m512i permmask2 = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2); \
  const __m512i va = _mm512_castps_si512(vlo), vb = _mm512_castps_si512(vhi); \
  const __m512i vtmp1 =  _mm512_permutex2var_epi64(va, permmask1, vb); \
  const __m512i vtmp2 =  _mm512_permutex2var_epi64(va, permmask2, vb); \
  const __m512i a = _mm512_srli_epi32(vtmp1, 16), b = _mm512_srli_epi32(vtmp2, 16); \
  v = _mm512_packus_epi32(a, b); \
  }

#elif LIBXSMM_STATIC_TARGET_ARCH==LIBXSMM_X86_AVX2
#define SIMD_WIDTH_FP32 (8)
#define SIMDTYPE_FP32 __m256
#define SIMDTYPE_INT32 __m256i
#define SIMDMASKTYPE_FP32 __m256
#define _MM_SETZERO_FP32 _mm256_setzero_ps
#define _MM_SETZERO_INT32 _mm256_setzero_si256
#define _MM_SET1_FP32 _mm256_set1_ps
#define _MM_SET1_INT32 _mm256_set1_epi32
#define _MM_SET1_INT16 _mm256_set1_epi16
#define _MM_SET_INT32 _mm256_set_epi32
#define _MM_LOAD_FP32 _mm256_load_ps
#define _MM_LOADU_FP32 _mm256_loadu_ps
#define _MM_LOAD_INT32 _mm256_load_si256
#define _MM_STORE_INT32 _mm256_store_si256
#define _MM_LOADU_INT32(x) _mm256_loadu_si256( (__m256i const *)(x))
#define _MM_GATHER_INT32(Addr, idx, scale) _mm256_i32gather_epi32((Addr), (idx), (scale))
#define _MM_GATHER_FP32(Addr, idx, scale) _mm256_i32gather_ps(((float const *)(Addr)), (idx), (scale))
#define _MM_CMPNEQ_FP32(v1,v2) _mm256_cmp_ps(v1,v2,12)
#define _MM_STORE_FP32 _mm256_store_ps
#define _MM_ADD_FP32 _mm256_add_ps
#define _MM_FMADD_FP32 _mm256_fmadd_ps
#define _MM_PREFETCH(x, y) _mm_prefetch(x, y)

LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE __m256i internal_spmdm_shufmasks_32[256];
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE __m256i internal_spmdm_shufmasks_16[256];

LIBXSMM_INLINE LIBXSMM_RETARGETABLE void _mm256_print(__m256 a, char * s)
{
  float *v=(float *)(&a);
  int i;
  printf("[%8s(%3lud)]: ", s, (size_t)8);
  for(i=0; i < 8; i++)
    printf("%4f ", v[i]);
  printf("\n");
}

LIBXSMM_INLINE LIBXSMM_RETARGETABLE void _mm256i_print(__m256i a, char * s)
{
  int *v=(int *)(&a);
  int i;
  printf("[%8s(%3lud)]: ", s, (size_t)8);
  for(i=0; i < 8; i++)
    printf("%4d ", v[i]);
  printf("\n");
}

LIBXSMM_INLINE LIBXSMM_RETARGETABLE void _mm256i_epi16_print(__m256i a, char * s)
{
  uint16_t *v=(uint16_t*)(&a);
  int i;
  printf("[%8s(%3lud)]: ", s, (size_t)16);
  for(i=0; i < 16; i++)
    printf("%4d ", v[i]);
  printf("\n");
}

#define COMPRESS_FP32(v, k, m, cnt) \
  { \
  unsigned int mask = _mm256_movemask_ps(m); \
  SIMDTYPE_INT32 vk = _MM_SET1_INT16((short)(k)); \
  __m256i perm_ctrl = _mm256_load_si256(&shufmasks[mask]); \
  __m256 v_packed = _mm256_permutevar8x32_ps(v, perm_ctrl); \
  __m256i v_idx = _mm256_add_epi32(vk, _mm256_load_si256(&shufmasks2[mask])); \
  _mm256_storeu_ps(values_ptr + (cnt), v_packed); \
  _mm256_storeu_si256((__m256i *)(colidx_ptr + (cnt)), v_idx); \
  cnt = (unsigned short)((cnt) + _mm_popcnt_u32(mask)); \
  }

#define EXPAND_BFLOAT16(v, vlo_final, vhi_final) \
  { \
  __m256i vlo = _mm256_unpacklo_epi16(vzero, v); \
  __m256i vhi = _mm256_unpackhi_epi16(vzero, v); \
  vlo_final = _mm256_castsi256_ps(_mm256_permute2f128_si256(vlo, vhi, 0x20)); \
  vhi_final = _mm256_castsi256_ps(_mm256_permute2f128_si256(vlo, vhi, 0x31)); \
  }

#define COMPRESS_BFLOAT16(vlo, vhi, v) \
  { \
  const __m256i vtmp1 =  _mm256_castps_si256(_mm256_permute2f128_ps(vlo, vhi, 0x20)); \
  const __m256i vtmp2 =  _mm256_castps_si256(_mm256_permute2f128_ps(vlo, vhi, 0x31)); \
  const __m256i a = _mm256_srli_epi32(vtmp1, 16), b = _mm256_srli_epi32(vtmp2,16); \
  v = _mm256_packus_epi32(a, b); \
  }

#else
#define SIMD_WIDTH_FP32 (1)
#define SIMDTYPE_FP32 float
#define SIMDTYPE_INT32 int
#define SIMDMASKTYPE_FP32 int
#define _MM_SETZERO_FP32() (0)
#define _MM_SETZERO_INT32() (0)
#define _MM_SET1_FP32(x) (x)
#define _MM_SET1_INT32(x) (x)
#define _MM_SET1_INT16 (x)
#define _MM_LOAD_FP32(x) (*(x))
#define _MM_LOADU_FP32(x) (*(x))
#define _MM_LOAD_INT32(x) (*(x))
#define _MM_STORE_INT32(x,y) ((*(x)) = (y))
#define _MM_LOADU_INT32(x) (*(x))
#define _MM_GATHER_FP32(Addr, idx, scale) (*(Addr + (idx)))
#define _MM_CMPNEQ_FP32(v1,v2) (LIBXSMM_FEQ(v1, v2) ? 0 : 1)
#define _MM_STORE_FP32(x,y) ((*(x)) = (y))
#define _MM_ADD_FP32(x,y) ((x) + (y))
#define _MM_FMADD_FP32(x,y,z) (((x)*(y))+(z))
#define _MM_PREFETCH(x, y)

#define COMPRESS_FP32(v, k, m, cnt) \
  { \
  if(m) \
  { \
    values_ptr[cnt] = v; \
    colidx_ptr[cnt] = (uint16_t)(k); \
    cnt++; \
  } \
  }

#define EXPAND_BFLOAT16(v, vlo_final, vhi_final) \
  { \
    union { int i; float f; } vlo_tmp, vhi_tmp; \
    vlo_tmp.i = (v) & 0xFFFF; vlo_tmp.i <<= 16; \
    vlo_final = vlo_tmp.f; \
    vhi_tmp.i = (v) & 0x0000FFFF; \
    vhi_final = vhi_tmp.f; \
  }

#define COMPRESS_BFLOAT16(vlo, vhi, v) \
  { \
    union { int i; float f; } vlo_tmp, vhi_tmp; \
    vlo_tmp.f = vlo; \
    v = (vlo_tmp.i >> 16); \
    vhi_tmp.f = vhi; \
    v = v | (vhi_tmp.i & 0xFFFF0000); \
  }

#endif


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_spmdm_init_shufmask()
{
#if SIMD_WIDTH_FP32 != 1
  unsigned int i,j, c, last_bit;
  LIBXSMM_ALIGNED(int temp_shufmasks[8], 64);
  LIBXSMM_ALIGNED(uint16_t temp_shufmasks2[16], 64);
  int cnt;
  for(i = 0; i < 256; i++) {
    cnt = 0;
    j = i;
    for(c = 0; c < 8; c++) temp_shufmasks[c] = 0;
    for(c = 0; c < 16; c++) temp_shufmasks2[c] = 0;
    while ( j ) {
      last_bit = _bit_scan_forward(j);
      temp_shufmasks[cnt] = last_bit;
      temp_shufmasks2[cnt] = (uint16_t)last_bit;
      j &= (~(1<<last_bit));
      cnt++;
    }
    internal_spmdm_shufmasks_32[i] = _mm256_loadu_si256((const __m256i*)temp_shufmasks);
    internal_spmdm_shufmasks_16[i] = _mm256_loadu_si256((const __m256i*)temp_shufmasks2);
  }
#endif
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_spmdm_allocate_csr_a( const libxsmm_spmdm_handle* handle, libxsmm_CSR_sparseslice ** libxsmm_output_csr)
{
  int kb, mb;
  int m_blocks = handle->mb;
  int k_blocks = handle->kb;

  libxsmm_CSR_sparseslice* libxsmm_output_csr_a = (libxsmm_CSR_sparseslice *)libxsmm_aligned_malloc( handle->mb * handle->kb * sizeof(libxsmm_CSR_sparseslice), 2097152);
  for ( kb = 0; kb < k_blocks; kb++ ) {
    for ( mb = 0; mb < m_blocks; mb++ ) {
      int i = kb*m_blocks + mb;
      libxsmm_output_csr_a[i].rowidx = (uint16_t *)libxsmm_aligned_malloc((handle->bm + 1)*sizeof(uint16_t), 2097152);
      libxsmm_output_csr_a[i].colidx = (uint16_t *)libxsmm_aligned_malloc((handle->bm)*(handle->bk)*sizeof(uint16_t), 2097152);
      libxsmm_output_csr_a[i].values = (float *)libxsmm_aligned_malloc((handle->bm)*(handle->bk)*sizeof(float), 2097152);
    }
  }

  *libxsmm_output_csr = libxsmm_output_csr_a;
}


LIBXSMM_API_DEFINITION void libxsmm_spmdm_init(int M, int N, int K, libxsmm_spmdm_handle * handle, libxsmm_CSR_sparseslice ** libxsmm_output_csr)
{
  handle->m  = M;
  handle->n  = N;
  handle->k  = K;

  handle->bm = 512;
#if LIBXSMM_STATIC_TARGET_ARCH==LIBXSMM_X86_AVX512_CORE
  handle->bn = 96;
#elif LIBXSMM_STATIC_TARGET_ARCH==LIBXSMM_X86_AVX2
  handle->bn = 48;
#else
  handle->bn = 6;
#endif
  handle->bk = 128;
  handle->mb = (handle->m + handle->bm - 1) / handle->bm;
  handle->nb = (handle->n + handle->bn - 1) / handle->bn;
  handle->kb = (handle->k + handle->bk - 1) / handle->bk;

  /* This is temporary space needed; allocate for each different size of A */
  internal_spmdm_allocate_csr_a( handle, libxsmm_output_csr);

  /* Initialize shuffle masks for the computation */
  internal_spmdm_init_shufmask();
}


/* This converts a dense representation of the sparse matrix to 2D array of sparse slices. */
LIBXSMM_API_DEFINITION void libxsmm_spmdm_createSparseSlice_fp32_notrans_thread(
  const libxsmm_spmdm_handle* handle,
  char transA,
  const float * A,
  libxsmm_CSR_sparseslice* libxsmm_output_csr_a,
  int mb, int kb,
  int tid, int nthreads)
{
   int i,k;
#if SIMD_WIDTH_FP32 == 8
   __m256i * shufmasks = internal_spmdm_shufmasks_32;
#endif

#if SIMD_WIDTH_FP32 > 1
   __m256i * shufmasks2 = internal_spmdm_shufmasks_16;
#endif
   int block_offset_base, block_offset;
   int index[16];
   SIMDTYPE_INT32 vindex;

   LIBXSMM_UNUSED(nthreads);
   LIBXSMM_UNUSED(tid);

   if(transA == 'Y')
   {
     int kk;
     block_offset_base = mb * handle->bm;
     block_offset = block_offset_base + kb * handle->m * handle->bk;
     for(kk = 0; kk < SIMD_WIDTH_FP32; kk++) index[kk] = kk*handle->m;
     vindex = _MM_LOADU_INT32(index);
   }
   else
   {
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
     if(SIMD_WIDTH_FP32 == 1) { ncols_aligned = 0; ncols_aligned_2 = 0; }
     for(i = 0; i < nrows; i++) {
       rowidx_ptr[i] = cnt;
       if(transA == 'Y')
       {
         for(k = 0; k < ncols_aligned; k+= 4*SIMD_WIDTH_FP32) {
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
         for(k = ncols_aligned; k < ncols_aligned_2; k+= SIMD_WIDTH_FP32) {
           SIMDTYPE_FP32 v1 = _MM_GATHER_FP32(input_ptr + k*handle->m + i, vindex, 4);
           SIMDMASKTYPE_FP32 m1 = _MM_CMPNEQ_FP32(v1, vzero);
           COMPRESS_FP32(v1, k, m1, cnt);
         }

         for(k = ncols_aligned_2; k < ncols; k++) {
           const float v1 = input_ptr[i + k*handle->m];
           const int m1 = LIBXSMM_FEQ(0, v1) ? 0 : 1;
           if(m1) { colidx_ptr[cnt] = (uint16_t)k; values_ptr[cnt] = v1; cnt++; }
         }
       }
       else
       {
         for(k = 0; k < ncols_aligned; k+= 4*SIMD_WIDTH_FP32) {
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
         for(k = ncols_aligned; k < ncols_aligned_2; k+= SIMD_WIDTH_FP32) {
           SIMDTYPE_FP32 v1;
           SIMDMASKTYPE_FP32 m1;
           v1 = _MM_LOADU_FP32(input_ptr + i*handle->k + k);
           _MM_PREFETCH((char *)(input_ptr + (i+2)*handle->k + k), _MM_HINT_T0);
           m1 = _MM_CMPNEQ_FP32(v1, vzero);
           COMPRESS_FP32(v1, k, m1, cnt);
         }
         for(k = ncols_aligned_2; k < ncols; k++) {
           const float v1 = input_ptr[i*handle->k + k];
           const int m1 = LIBXSMM_FEQ(0, v1) ? 0 : 1;
           if(m1) { colidx_ptr[cnt] = (uint16_t)k; values_ptr[cnt] = v1; cnt++; }
         }
       }
     }
     rowidx_ptr[nrows] = cnt;
#if 0
     printf("cnt: %d\n", cnt);
     for(i = 0; i <= nrows; i++) {
       int j;
       for(j = slice.rowidx[i]; j < slice.rowidx[i+1]; j++) {
         printf("(%d, %d): %f ", i, colidx_ptr[j], values_ptr[j]);
       }
     }
#endif
   }
}


LIBXSMM_API_DEFINITION void libxsmm_spmdm_createSparseSlice_bfloat16_notrans_thread(
  const libxsmm_spmdm_handle* handle,
  char transA,
  const uint16_t * A,
  libxsmm_CSR_sparseslice* libxsmm_output_csr_a,
  int mb, int kb,
  int tid, int nthreads)
{
   int i,k;
#if SIMD_WIDTH_FP32 == 8
   __m256i * shufmasks = internal_spmdm_shufmasks_32;
#endif
#if SIMD_WIDTH_FP32 > 1
   __m256i * shufmasks2 = internal_spmdm_shufmasks_16;
#endif
   int block_offset_base, block_offset;

   LIBXSMM_UNUSED(nthreads);
   LIBXSMM_UNUSED(tid);

   if(transA == 'Y')
   {
     block_offset_base = mb * handle->bm;
     block_offset = block_offset_base + kb * handle->m * handle->bk;
   }
   else
   {
     block_offset_base = kb * handle->bk;
     block_offset = block_offset_base + mb * handle->k * handle->bm;
   }
   {
     libxsmm_CSR_sparseslice slice = libxsmm_output_csr_a[kb*handle->mb + mb];
     int nrows = ((mb + 1)*handle->bm > handle->m)?(handle->m - (mb)*handle->bm):handle->bm;
     int ncols = ((kb + 1)*handle->bk > handle->k)?(handle->k - (kb)*handle->bk):handle->bk;
     /*printf("nrows: %d, ncols: %d\n", nrows, ncols);*/
     int ncols_aligned = ncols / (4*SIMD_WIDTH_FP32)*(4*SIMD_WIDTH_FP32);
     const uint16_t * input_ptr = A + block_offset;
     uint16_t * rowidx_ptr = slice.rowidx;
     uint16_t * colidx_ptr = slice.colidx;
     float * values_ptr = (float *)(slice.values);
#if SIMD_WIDTH_FP32 > 1
     SIMDTYPE_INT32 vzero = _MM_SET1_INT32(0);
#endif
     SIMDTYPE_FP32 vzerof = _MM_SET1_FP32(0.0);
     uint16_t cnt = 0;
     if(SIMD_WIDTH_FP32 == 1) { ncols_aligned = 0; }
     for(i = 0; i < nrows; i++) {
       rowidx_ptr[i] = cnt;
       if(transA == 'Y')
       {
         for(k = 0; k < ncols_aligned; k+= 4*SIMD_WIDTH_FP32) {
           int vals[32];
           int kk;
           for(kk = 0; kk < 4*SIMD_WIDTH_FP32; kk+=2) { vals[kk/2] = (int)input_ptr[(k+kk)*handle->m + i]; vals[kk/2] |= ((int)(input_ptr[(k+kk+1)*handle->m + i]) << 16); }
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

         for(k = ncols_aligned; k < ncols; k++) {
           uint16_t v1tmp = input_ptr[k*handle->m + i];
           union {int i; float f; } v1tmp_int;
           v1tmp_int.i = v1tmp;
           v1tmp_int.i <<= 16;
           {
             const int m1 = LIBXSMM_FEQ(0, v1tmp_int.f) ? 0 : 1;
             if(m1) { colidx_ptr[cnt] = (uint16_t)k; values_ptr[cnt] = v1tmp_int.f; cnt++; }
           }
         }
       }
       else
       {
         for(k = 0; k < ncols_aligned; k+= 4*SIMD_WIDTH_FP32) {
           SIMDTYPE_INT32 v1tmp, v2tmp;
           SIMDTYPE_FP32 v1, v2, v3, v4;
           SIMDMASKTYPE_FP32 m1, m2, m3, m4;
           v1tmp = _MM_LOADU_INT32((const SIMDTYPE_INT32* )(input_ptr + i*handle->k + k));
           _MM_PREFETCH((char *)(input_ptr + (i+2)*handle->k + k), _MM_HINT_T0);
           v2tmp = _MM_LOADU_INT32((const SIMDTYPE_INT32*)(input_ptr + i*handle->k + k + 2*SIMD_WIDTH_FP32));
           _MM_PREFETCH((char *)(input_ptr + (i+2)*handle->k + k + SIMD_WIDTH_FP32), _MM_HINT_T0);
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
         for(k = ncols_aligned; k < ncols; k++) {
           uint16_t v1tmp = input_ptr[i*handle->k + k];
           union {int i; float f; } v1tmp_int;
           v1tmp_int.i = v1tmp;
           v1tmp_int.i <<= 16;
           {
             int m1 = LIBXSMM_FEQ(0, v1tmp_int.f) ? 0 : 1;
             if(m1) { colidx_ptr[cnt] = (uint16_t)k; values_ptr[cnt] = v1tmp_int.f; cnt++; }
           }
         }
       }
     }
     rowidx_ptr[nrows] = cnt;
#if 0
     printf("cnt: %d\n", cnt);
     for(i = 0; i <= nrows; i++) {
       for(j = slice.rowidx[i]; j < slice.rowidx[i+1]; j++) {
         printf("(%d, %d): %f ", i, colidx_ptr[j], values_ptr[j]);
       }
     }
#endif
   }
}


LIBXSMM_API_DEFINITION void libxsmm_spmdm_compute_fp32_thread(
  const libxsmm_spmdm_handle* handle,
  char transA,
  char transB,
  const float *alpha,
  libxsmm_CSR_sparseslice* A_sparse,
  const float *B,
  const float *beta,
  float* C,
  int mb, int num_m_blocks, int nb,
  int tid, int nthreads)
{
  const int m_blocks = handle->mb;
  /* const int n_blocks = handle->nb; */
  const int k_blocks = handle->kb;
  const int m_block_size = handle->bm;
  const int n_block_size = handle->bn;
  const int k_block_size = handle->bk;

#define num_regs (6)
  int m_overall_start = mb*m_block_size;
  int m_overall_end   = (mb + num_m_blocks)*m_block_size;
  int num_m;
  int num_m_aligned;

  int n_overall_start = nb*n_block_size;
  int n_overall_end   = (nb + 1)*n_block_size;
  int num_n;
  int m, n, k, kb;
  int last_block_n, num_full_regs, last_n_start;

  int k_overall_start, k_overall_end, num_k;

  float *const scratch_C = (float*)LIBXSMM_SPMDM_MALLOC(num_m_blocks*m_block_size*n_block_size*sizeof(float), 64);
  float *const scratch_B = (float*)LIBXSMM_SPMDM_MALLOC(k_block_size*n_block_size*sizeof(float), 64);
  SIMDTYPE_FP32 sum[2*num_regs];
  float* LIBXSMM_RESTRICT ptr_result;

  LIBXSMM_UNUSED(nthreads);
  LIBXSMM_UNUSED(transA);
  LIBXSMM_UNUSED(alpha);
  LIBXSMM_UNUSED(beta);
  LIBXSMM_UNUSED(tid);

  /* really is twice this */
  assert(n_block_size == num_regs*SIMD_WIDTH_FP32);

  if (m_overall_end   > handle->m) m_overall_end   = handle->m;
  num_m = (m_overall_end - m_overall_start);
  num_m_aligned = (num_m / 2) * 2;

  if (n_overall_end   > handle->n) n_overall_end   = handle->n;
  num_n = (n_overall_end - n_overall_start);
  last_block_n = (num_n != n_block_size);
  num_full_regs = 0; /* (num_n / SIMD_WIDTH_FP32);*/
  last_n_start = num_full_regs*SIMD_WIDTH_FP32;

#if 0
  printf("Block: m_overall_start: %d, m_overall_end: %d, num_m: %d, num_m_aligned: %d\n", m_overall_start, m_overall_end, num_m, num_m_aligned);
  printf("Block: n_overall_start: %d, n_overall_end: %d, num_n: %d, last_block_n: %d\n", n_overall_start, n_overall_end, num_n, last_block_n);
  printf("Block: k_blocks: %d\n", k_blocks);
#endif
  /* Copy in C matrix to buffer*/
  ptr_result = C + m_overall_start*handle->n + n_overall_start;
  if(!last_block_n) {
    for (m = 0; m < num_m; m++) {
      _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 0*SIMD_WIDTH_FP32, _MM_LOAD_FP32(ptr_result + m*handle->n + 0*SIMD_WIDTH_FP32));
      _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 1*SIMD_WIDTH_FP32, _MM_LOAD_FP32(ptr_result + m*handle->n + 1*SIMD_WIDTH_FP32));
      _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 2*SIMD_WIDTH_FP32, _MM_LOAD_FP32(ptr_result + m*handle->n + 2*SIMD_WIDTH_FP32));
      _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 3*SIMD_WIDTH_FP32, _MM_LOAD_FP32(ptr_result + m*handle->n + 3*SIMD_WIDTH_FP32));
      _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 4*SIMD_WIDTH_FP32, _MM_LOAD_FP32(ptr_result + m*handle->n + 4*SIMD_WIDTH_FP32));
      _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 5*SIMD_WIDTH_FP32, _MM_LOAD_FP32(ptr_result + m*handle->n + 5*SIMD_WIDTH_FP32));
    }
  } else {
    for (m = 0; m < num_m; m++) {
      for (n = 0; n < num_full_regs; n+=2) {
        _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + n*SIMD_WIDTH_FP32, _MM_LOAD_FP32(ptr_result + m*handle->n + n*SIMD_WIDTH_FP32));
        _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + (n+1)*SIMD_WIDTH_FP32, _MM_LOAD_FP32(ptr_result + m*handle->n + (n+1)*SIMD_WIDTH_FP32));
      }
      for (n = last_n_start; n < num_n; n++) {
        scratch_C[m*num_regs*SIMD_WIDTH_FP32 + n] = ptr_result[m*handle->n + n];
      }
    }
  }

  for (kb = 0; kb < k_blocks; kb++) {
    const float * LIBXSMM_RESTRICT ptr_dense;
    float * LIBXSMM_RESTRICT scratch_C_base;
    const float * LIBXSMM_RESTRICT scratch_B_base;
    int block_A = kb * m_blocks + mb;
    libxsmm_CSR_sparseslice slice = A_sparse[block_A];
    int m_local = 0;

    k_overall_start = kb*k_block_size;
    k_overall_end   = (kb+1)*k_block_size;
    num_k = (k_overall_end - k_overall_start);

    /* Copy in B matrix*/
    if(transB == 'Y')
    {
      SIMDTYPE_INT32 vindex;
      int index[16];
      int kk;
      for(kk = 0; kk < SIMD_WIDTH_FP32; kk++) index[kk] = kk*handle->k;
      vindex = _MM_LOADU_INT32(index);
      ptr_dense = B + n_overall_start*handle->k + k_overall_start;
      if(!last_block_n) {
        for (k = 0; k < num_k; k++) {
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + 0*SIMD_WIDTH_FP32, _MM_GATHER_FP32(ptr_dense + k + 0*SIMD_WIDTH_FP32*handle->k, vindex, 4));
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + 1*SIMD_WIDTH_FP32, _MM_GATHER_FP32(ptr_dense + k + 1*SIMD_WIDTH_FP32*handle->k, vindex, 4));
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + 2*SIMD_WIDTH_FP32, _MM_GATHER_FP32(ptr_dense + k + 2*SIMD_WIDTH_FP32*handle->k, vindex, 4));
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + 3*SIMD_WIDTH_FP32, _MM_GATHER_FP32(ptr_dense + k + 3*SIMD_WIDTH_FP32*handle->k, vindex, 4));
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + 4*SIMD_WIDTH_FP32, _MM_GATHER_FP32(ptr_dense + k + 4*SIMD_WIDTH_FP32*handle->k, vindex, 4));
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + 5*SIMD_WIDTH_FP32, _MM_GATHER_FP32(ptr_dense + k + 5*SIMD_WIDTH_FP32*handle->k, vindex, 4));
        }
      }
      else {
        for (k = 0; k < num_k; k++) {
          for (n = 0; n < num_n; n++) {
            scratch_B[k*num_regs*SIMD_WIDTH_FP32 + n] = ptr_dense[n*handle->k + k];
          }
        }
      }
    }
    else
    {
      ptr_dense = B + k_overall_start*handle->n + n_overall_start;
      if(!last_block_n) {
        for (k = 0; k < num_k; k++) {
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + 0*SIMD_WIDTH_FP32, _MM_LOAD_FP32(ptr_dense + k*handle->n + 0*SIMD_WIDTH_FP32));
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + 1*SIMD_WIDTH_FP32, _MM_LOAD_FP32(ptr_dense + k*handle->n + 1*SIMD_WIDTH_FP32));
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + 2*SIMD_WIDTH_FP32, _MM_LOAD_FP32(ptr_dense + k*handle->n + 2*SIMD_WIDTH_FP32));
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + 3*SIMD_WIDTH_FP32, _MM_LOAD_FP32(ptr_dense + k*handle->n + 3*SIMD_WIDTH_FP32));
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + 4*SIMD_WIDTH_FP32, _MM_LOAD_FP32(ptr_dense + k*handle->n + 4*SIMD_WIDTH_FP32));
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + 5*SIMD_WIDTH_FP32, _MM_LOAD_FP32(ptr_dense + k*handle->n + 5*SIMD_WIDTH_FP32));
        }
      } else {
        for (k = 0; k < num_k; k++) {
          for (n = 0; n < num_n; n++) {
            scratch_B[k*num_regs*SIMD_WIDTH_FP32 + n] = ptr_dense[k*handle->n + n];
          }
        }
      }
    }
#if 0
    printf("B_col\n");
    for (k = 0; k < num_k; k++) {
      printf(" %lf ", ptr_dense[k*handle->n]);
    }
    printf("\n");
#endif
    scratch_C_base = scratch_C - m_overall_start*num_regs*SIMD_WIDTH_FP32;
    scratch_B_base = scratch_B; /* - k_overall_start*num_regs*SIMD_WIDTH_FP32;*/

    for (m = m_overall_start; m < m_overall_start + num_m_aligned; m+=2, m_local+=2) {
      int start_j, end_j, end_j_2, num_j, num_j_2;
      const uint16_t*  LIBXSMM_RESTRICT sp_c_ptr_base;
      const uint16_t*  LIBXSMM_RESTRICT sp_c_ptr_base_2;
      const float* LIBXSMM_RESTRICT sp_v_ptr_base;
      const float* LIBXSMM_RESTRICT sp_v_ptr_base_2;
      float* LIBXSMM_RESTRICT result_m_index;
      float* LIBXSMM_RESTRICT result_m_index_2;

      if( m_local >= m_block_size) { block_A++; slice = A_sparse[block_A]; m_local = 0; }

      start_j =  slice.rowidx[m_local];
      end_j   =  slice.rowidx[m_local + 1];
      end_j_2 =  slice.rowidx[m_local + 2];
      num_j   = (end_j - start_j);
      num_j_2   = (end_j_2 - end_j);
      sp_c_ptr_base = slice.colidx + start_j;
      sp_c_ptr_base_2 = slice.colidx + end_j;
      sp_v_ptr_base = (float *)(slice.values) + start_j;
      sp_v_ptr_base_2 = (float *)(slice.values) + end_j;
      result_m_index = scratch_C_base + (m)*num_regs*SIMD_WIDTH_FP32;
      result_m_index_2 = scratch_C_base + (m+1)*num_regs*SIMD_WIDTH_FP32;

      if(!last_block_n)
      {
        int64_t j = 0, j2 = 0;
        sum[0] = _MM_LOAD_FP32(result_m_index + 0*SIMD_WIDTH_FP32);
        sum[0+num_regs] = _MM_LOAD_FP32(result_m_index_2 + 0*SIMD_WIDTH_FP32);
        sum[1] = _MM_LOAD_FP32(result_m_index + 1*SIMD_WIDTH_FP32);
        sum[1+num_regs] = _MM_LOAD_FP32(result_m_index_2 + 1*SIMD_WIDTH_FP32);
        sum[2] = _MM_LOAD_FP32(result_m_index + 2*SIMD_WIDTH_FP32);
        sum[2+num_regs] = _MM_LOAD_FP32(result_m_index_2 + 2*SIMD_WIDTH_FP32);
        sum[3] = _MM_LOAD_FP32(result_m_index + 3*SIMD_WIDTH_FP32);
        sum[3+num_regs] = _MM_LOAD_FP32(result_m_index_2 + 3*SIMD_WIDTH_FP32);
        sum[4] = _MM_LOAD_FP32(result_m_index + 4*SIMD_WIDTH_FP32);
        sum[4+num_regs] = _MM_LOAD_FP32(result_m_index_2 + 4*SIMD_WIDTH_FP32);
        sum[5] = _MM_LOAD_FP32(result_m_index + 5*SIMD_WIDTH_FP32);
        sum[5+num_regs] = _MM_LOAD_FP32(result_m_index_2 + 5*SIMD_WIDTH_FP32);
        for (; j < num_j && j2 < num_j_2; j++, j2++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index = scratch_B_base +  sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          const float* const LIBXSMM_RESTRICT sp_col_dense_index_2 = scratch_B_base + sp_c_ptr_base_2[j2]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]);
          SIMDTYPE_FP32 v_v_2 = _MM_SET1_FP32(sp_v_ptr_base_2[j2]);
          sum[0] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 0*SIMD_WIDTH_FP32), sum[0]);
          sum[0 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 0*SIMD_WIDTH_FP32), sum[0+num_regs]);
          sum[1] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 1*SIMD_WIDTH_FP32), sum[1]);
          sum[1 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 1*SIMD_WIDTH_FP32), sum[1+num_regs]);
          sum[2] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 2*SIMD_WIDTH_FP32), sum[2]);
          sum[2 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 2*SIMD_WIDTH_FP32), sum[2+num_regs]);
          sum[3] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 3*SIMD_WIDTH_FP32), sum[3]);
          sum[3 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 3*SIMD_WIDTH_FP32), sum[3+num_regs]);
          sum[4] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 4*SIMD_WIDTH_FP32), sum[4]);
          sum[4 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 4*SIMD_WIDTH_FP32), sum[4+num_regs]);
          sum[5] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 5*SIMD_WIDTH_FP32), sum[5]);
          sum[5 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 5*SIMD_WIDTH_FP32), sum[5+num_regs]);
        }
        for (; j < num_j; j++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index = scratch_B_base +  sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]);
          sum[0] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 0*SIMD_WIDTH_FP32), sum[0]);
          sum[1] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 1*SIMD_WIDTH_FP32), sum[1]);
          sum[2] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 2*SIMD_WIDTH_FP32), sum[2]);
          sum[3] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 3*SIMD_WIDTH_FP32), sum[3]);
          sum[4] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 4*SIMD_WIDTH_FP32), sum[4]);
          sum[5] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 5*SIMD_WIDTH_FP32), sum[5]);
        }
        for (; j2 < num_j_2; j2++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index_2 = scratch_B_base + sp_c_ptr_base_2[j2]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v_2 = _MM_SET1_FP32(sp_v_ptr_base_2[j2]);
          sum[0 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 0*SIMD_WIDTH_FP32), sum[0+num_regs]);
          sum[1 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 1*SIMD_WIDTH_FP32), sum[1+num_regs]);
          sum[2 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 2*SIMD_WIDTH_FP32), sum[2+num_regs]);
          sum[3 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 3*SIMD_WIDTH_FP32), sum[3+num_regs]);
          sum[4 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 4*SIMD_WIDTH_FP32), sum[4+num_regs]);
          sum[5 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 5*SIMD_WIDTH_FP32), sum[5+num_regs]);
        }
        _MM_STORE_FP32(result_m_index + 0*SIMD_WIDTH_FP32, sum[0]);
        _MM_STORE_FP32(result_m_index_2 + 0*SIMD_WIDTH_FP32, sum[0+num_regs]);
        _MM_STORE_FP32(result_m_index + 1*SIMD_WIDTH_FP32, sum[1]);
        _MM_STORE_FP32(result_m_index_2 + 1*SIMD_WIDTH_FP32, sum[1+num_regs]);
        _MM_STORE_FP32(result_m_index + 2*SIMD_WIDTH_FP32, sum[2]);
        _MM_STORE_FP32(result_m_index_2 + 2*SIMD_WIDTH_FP32, sum[2+num_regs]);
        _MM_STORE_FP32(result_m_index + 3*SIMD_WIDTH_FP32, sum[3]);
        _MM_STORE_FP32(result_m_index_2 + 3*SIMD_WIDTH_FP32, sum[3+num_regs]);
        _MM_STORE_FP32(result_m_index + 4*SIMD_WIDTH_FP32, sum[4]);
        _MM_STORE_FP32(result_m_index_2 + 4*SIMD_WIDTH_FP32, sum[4+num_regs]);
        _MM_STORE_FP32(result_m_index + 5*SIMD_WIDTH_FP32, sum[5]);
        _MM_STORE_FP32(result_m_index_2 + 5*SIMD_WIDTH_FP32, sum[5+num_regs]);
      }
      else {
        int64_t j = 0, j2 = 0;
        for (n = 0; n < num_full_regs; n+=2) {
          sum[n] = _MM_SETZERO_FP32();
          sum[n+num_regs] = _MM_SETZERO_FP32();
          sum[n+1] = _MM_SETZERO_FP32();
          sum[n+1+num_regs] = _MM_SETZERO_FP32();
        }
        for (; j < num_j && j2 < num_j_2; j++, j2++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index = scratch_B_base +  sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          const float* const LIBXSMM_RESTRICT sp_col_dense_index_2 = scratch_B_base + sp_c_ptr_base_2[j2]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]);
          SIMDTYPE_FP32 v_v_2 = _MM_SET1_FP32(sp_v_ptr_base_2[j2]);
          for (n = 0; n < num_full_regs; n+=2) {
            sum[n] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + n*SIMD_WIDTH_FP32), sum[n]);
            sum[n + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + n*SIMD_WIDTH_FP32), sum[n+num_regs]);
            sum[n+1] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + (n+1)*SIMD_WIDTH_FP32), sum[n+1]);
            sum[n+1 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + (n+1)*SIMD_WIDTH_FP32), sum[n+1+num_regs]);
          }
          for (n = last_n_start; n < num_n; n++) {
            result_m_index[n] += sp_col_dense_index[n]*sp_v_ptr_base[j];
            result_m_index_2[n] += sp_col_dense_index_2[n]*sp_v_ptr_base_2[j2];
          }
        }
        for (; j < num_j; j++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index = scratch_B_base +  sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]);
          for (n = 0; n < num_full_regs; n+=2) {
            sum[n] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + n*SIMD_WIDTH_FP32), sum[n]);
            sum[n+1] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + (n+1)*SIMD_WIDTH_FP32), sum[n+1]);
          }
          for (n = last_n_start; n < num_n; n++) {
            result_m_index[n] += sp_col_dense_index[n]*sp_v_ptr_base[j];
          }
        }
        for (; j2 < num_j_2; j2++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index_2 = scratch_B_base + sp_c_ptr_base_2[j2]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v_2 = _MM_SET1_FP32(sp_v_ptr_base_2[j2]);
          for (n = 0; n < num_full_regs; n+=2) {
            sum[n + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + n*SIMD_WIDTH_FP32), sum[n+num_regs]);
            sum[n+1 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + (n+1)*SIMD_WIDTH_FP32), sum[n+1+num_regs]);
          }
          for (n = last_n_start; n < num_n; n++) {
            result_m_index_2[n] += sp_col_dense_index_2[n]*sp_v_ptr_base_2[j2];
          }
        }
        for (n = 0; n < num_full_regs; n+=2) {
          _MM_STORE_FP32(result_m_index + n*SIMD_WIDTH_FP32,  _MM_ADD_FP32(sum[n], _MM_LOAD_FP32(result_m_index + n*SIMD_WIDTH_FP32)));
          _MM_STORE_FP32(result_m_index_2 + n*SIMD_WIDTH_FP32,  _MM_ADD_FP32(sum[n+num_regs], _MM_LOAD_FP32(result_m_index_2 + n*SIMD_WIDTH_FP32)));
          _MM_STORE_FP32(result_m_index + (n+1)*SIMD_WIDTH_FP32,  _MM_ADD_FP32(sum[n+1], _MM_LOAD_FP32(result_m_index + (n+1)*SIMD_WIDTH_FP32)));
          _MM_STORE_FP32(result_m_index_2 + (n+1)*SIMD_WIDTH_FP32,  _MM_ADD_FP32(sum[n+1+num_regs], _MM_LOAD_FP32(result_m_index_2 + (n+1)*SIMD_WIDTH_FP32)));
        }
      }
    }
    for (m = m_overall_start + num_m_aligned; m < m_overall_end; m++, m_local++) {
      int start_j, end_j, num_j;
      const uint16_t*  LIBXSMM_RESTRICT sp_c_ptr_base;
      const float* LIBXSMM_RESTRICT sp_v_ptr_base;
      float* LIBXSMM_RESTRICT result_m_index;

      if( m_local >= m_block_size) { block_A++; slice = A_sparse[block_A]; m_local = 0; }

      start_j =  slice.rowidx[m_local];
      end_j   =  slice.rowidx[m_local + 1];
      num_j   = (end_j - start_j);
      sp_c_ptr_base = slice.colidx + start_j;
      sp_v_ptr_base = slice.values + start_j;
      result_m_index = scratch_C_base + (m)*num_regs*SIMD_WIDTH_FP32;

      if(!last_block_n) {
        int64_t j = 0;
        sum[0] = _MM_LOAD_FP32(result_m_index + 0*SIMD_WIDTH_FP32);
        sum[1] = _MM_LOAD_FP32(result_m_index + 1*SIMD_WIDTH_FP32);
        sum[2] = _MM_LOAD_FP32(result_m_index + 2*SIMD_WIDTH_FP32);
        sum[3] = _MM_LOAD_FP32(result_m_index + 3*SIMD_WIDTH_FP32);
        sum[4] = _MM_LOAD_FP32(result_m_index + 4*SIMD_WIDTH_FP32);
        sum[5] = _MM_LOAD_FP32(result_m_index + 5*SIMD_WIDTH_FP32);
        for (; j < num_j; j++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index = scratch_B_base +  sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]);
          sum[0] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 0*SIMD_WIDTH_FP32), sum[0]);
          sum[1] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 1*SIMD_WIDTH_FP32), sum[1]);
          sum[2] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 2*SIMD_WIDTH_FP32), sum[2]);
          sum[3] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 3*SIMD_WIDTH_FP32), sum[3]);
          sum[4] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 4*SIMD_WIDTH_FP32), sum[4]);
          sum[5] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 5*SIMD_WIDTH_FP32), sum[5]);
        }
        _MM_STORE_FP32(result_m_index + 0*SIMD_WIDTH_FP32, sum[0]);
        _MM_STORE_FP32(result_m_index + 1*SIMD_WIDTH_FP32, sum[1]);
        _MM_STORE_FP32(result_m_index + 2*SIMD_WIDTH_FP32, sum[2]);
        _MM_STORE_FP32(result_m_index + 3*SIMD_WIDTH_FP32, sum[3]);
        _MM_STORE_FP32(result_m_index + 4*SIMD_WIDTH_FP32, sum[4]);
        _MM_STORE_FP32(result_m_index + 5*SIMD_WIDTH_FP32, sum[5]);
      }
      else {
        int64_t j = 0;
        for (n = 0; n < num_full_regs; n+=2) {
          sum[n] = _MM_SETZERO_FP32();
          sum[n+1] = _MM_SETZERO_FP32();
        }
        for (; j < num_j; j++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index = scratch_B_base +  sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]);
          for (n = 0; n < num_full_regs; n+=2) {
            sum[n] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + n*SIMD_WIDTH_FP32), sum[n]);
            sum[n+1] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + (n+1)*SIMD_WIDTH_FP32), sum[n+1]);
          }
          for (n = last_n_start; n < num_n; n++) {
            result_m_index[n] += sp_col_dense_index[n]*sp_v_ptr_base[j];
          }
        }
        for (n = 0; n < num_full_regs; n+=2) {
          _MM_STORE_FP32(result_m_index + n*SIMD_WIDTH_FP32, _MM_ADD_FP32(sum[n], _MM_LOAD_FP32(result_m_index + n*SIMD_WIDTH_FP32)));
          _MM_STORE_FP32(result_m_index + (n+1)*SIMD_WIDTH_FP32, _MM_ADD_FP32(sum[n+1], _MM_LOAD_FP32(result_m_index + (n+1)*SIMD_WIDTH_FP32)));
        }
      }
    }
  } /* kb */
#if 0
  for (m = 0; m < 3; m++) {
    for (n = 0; n < num_n; n++) {
      printf("%f ", scratch_C[m*num_regs*SIMD_WIDTH_FP32 + n]);
    }
    printf("\n");
  }
#endif
  /* Copy out C matrix */
  if(!last_block_n) {
    for (m = 0; m < num_m; m++) {
      _MM_STORE_FP32(ptr_result + m*handle->n + 0*SIMD_WIDTH_FP32, _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 0*SIMD_WIDTH_FP32));
      _MM_STORE_FP32(ptr_result + m*handle->n + 1*SIMD_WIDTH_FP32, _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 1*SIMD_WIDTH_FP32));
      _MM_STORE_FP32(ptr_result + m*handle->n + 2*SIMD_WIDTH_FP32, _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 2*SIMD_WIDTH_FP32));
      _MM_STORE_FP32(ptr_result + m*handle->n + 3*SIMD_WIDTH_FP32, _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 3*SIMD_WIDTH_FP32));
      _MM_STORE_FP32(ptr_result + m*handle->n + 4*SIMD_WIDTH_FP32, _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 4*SIMD_WIDTH_FP32));
      _MM_STORE_FP32(ptr_result + m*handle->n + 5*SIMD_WIDTH_FP32, _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 5*SIMD_WIDTH_FP32));
    }
  } else {
    for (m = 0; m < num_m; m++) {
      for (n = 0; n < num_full_regs; n+=2) {
        _MM_STORE_FP32(ptr_result + m*handle->n + n*SIMD_WIDTH_FP32, _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + n*SIMD_WIDTH_FP32));
        _MM_STORE_FP32(ptr_result + m*handle->n + (n+1)*SIMD_WIDTH_FP32, _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + (n+1)*SIMD_WIDTH_FP32));
      }
      for (n = last_n_start; n < num_n; n++) {
        ptr_result[m*handle->n + n] = scratch_C[m*num_regs*SIMD_WIDTH_FP32 + n];
      }
    }
  }

  LIBXSMM_SPMDM_FREE(scratch_C);
  LIBXSMM_SPMDM_FREE(scratch_B);
}


LIBXSMM_API_DEFINITION void libxsmm_spmdm_compute_bfloat16_thread(
  const libxsmm_spmdm_handle* handle,
  char transA,
  char transB,
  const uint16_t *alpha,
  libxsmm_CSR_sparseslice* A_sparse,
  const uint16_t *B,
  const uint16_t *beta,
  uint16_t* C,
  int mb, int num_m_blocks, int nb,
  int tid, int nthreads)
{
  const int m_blocks = handle->mb;
  /*const int n_blocks = handle->nb;*/
  const int k_blocks = handle->kb;
  const int m_block_size = handle->bm;
  const int n_block_size = handle->bn;
  const int k_block_size = handle->bk;

#define num_regs (6)
  int m_overall_start = mb*m_block_size;
  int m_overall_end   = (mb + num_m_blocks)*m_block_size;
  int num_m;
  int num_m_aligned;

  int n_overall_start = nb*n_block_size;
  int n_overall_end   = (nb + 1)*n_block_size;
  int num_n;
  int m, n, k, kb;
  int last_block_n;

  int k_overall_start, k_overall_end, num_k;

  float *const scratch_C = (float*)LIBXSMM_SPMDM_MALLOC(num_m_blocks*m_block_size*n_block_size*sizeof(float), 64);
  float *const scratch_B = (float*)LIBXSMM_SPMDM_MALLOC(k_block_size*n_block_size*sizeof(float), 64);

  SIMDTYPE_FP32 sum[2*num_regs];
  uint16_t* LIBXSMM_RESTRICT ptr_result;
#if SIMD_WIDTH_FP32 > 1
  SIMDTYPE_INT32 vzero = _MM_SETZERO_INT32();
#endif

  LIBXSMM_UNUSED(nthreads);
  LIBXSMM_UNUSED(transA);
  LIBXSMM_UNUSED(alpha);
  LIBXSMM_UNUSED(beta);
  LIBXSMM_UNUSED(tid);

  /* really is twice this */
  assert(n_block_size == num_regs*SIMD_WIDTH_FP32);

  if (m_overall_end   > handle->m) m_overall_end   = handle->m;
  num_m = (m_overall_end - m_overall_start);
  num_m_aligned = (num_m / 2) * 2;

  if (n_overall_end   > handle->n) n_overall_end   = handle->n;
  num_n = (n_overall_end - n_overall_start);
  last_block_n = (num_n != n_block_size);
#if 0
  printf("Block: m_overall_start: %d, m_overall_end: %d, num_m: %d, num_m_aligned: %d\n", m_overall_start, m_overall_end, num_m, num_m_aligned);
  printf("Block: n_overall_start: %d, n_overall_end: %d, num_n: %d, last_block_n: %d\n", n_overall_start, n_overall_end, num_n, last_block_n);
  printf("Block: k_blocks: %d\n", k_blocks);
#endif
  /* Copy in C matrix to buffer */
  ptr_result = C + m_overall_start*handle->n + n_overall_start;
  if(!last_block_n) {
    for (m = 0; m < num_m; m++) {
      SIMDTYPE_INT32 vload_0 =  _MM_LOAD_INT32((const SIMDTYPE_INT32 *)(ptr_result + m*handle->n + 2*0*SIMD_WIDTH_FP32));
      SIMDTYPE_INT32 vload_1, vload_2;
      SIMDTYPE_FP32 v1_0, v2_0;
      SIMDTYPE_FP32 v1_1, v2_1;
      SIMDTYPE_FP32 v1_2, v2_2;
      EXPAND_BFLOAT16(vload_0, v1_0, v2_0);
      _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 2*0*SIMD_WIDTH_FP32, v1_0);
      _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + (2*0+1)*SIMD_WIDTH_FP32, v2_0);
      vload_1 =  _MM_LOAD_INT32((const SIMDTYPE_INT32 *)(ptr_result + m*handle->n + 2*1*SIMD_WIDTH_FP32));
      EXPAND_BFLOAT16(vload_1, v1_1, v2_1);
      _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 2*1*SIMD_WIDTH_FP32, v1_1);
      _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + (2*1+1)*SIMD_WIDTH_FP32, v2_1);
      vload_2 =  _MM_LOAD_INT32((const SIMDTYPE_INT32 *)(ptr_result + m*handle->n + 2*2*SIMD_WIDTH_FP32));
      EXPAND_BFLOAT16(vload_2, v1_2, v2_2);
      _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 2*2*SIMD_WIDTH_FP32, v1_2);
      _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + (2*2+1)*SIMD_WIDTH_FP32, v2_2);
    }
  } else {
    for (m = 0; m < num_m; m++) {
      for (n = 0; n < num_n; n++) {
        uint16_t restmp = ptr_result[m*handle->n + n];
        union { int i; float f; } res;
        res.i = restmp;
        res.i <<= 16;
        {
          scratch_C[m*num_regs*SIMD_WIDTH_FP32 + n] = res.f;
        }
      }
    }
  }

  for (kb = 0; kb < k_blocks; kb++) {
    const uint16_t* LIBXSMM_RESTRICT ptr_dense;
    float * LIBXSMM_RESTRICT scratch_C_base;
    const float * LIBXSMM_RESTRICT scratch_B_base;
    int block_A = kb * m_blocks + mb;
    libxsmm_CSR_sparseslice slice = A_sparse[block_A];
    int m_local = 0;

    k_overall_start = kb*k_block_size;
    k_overall_end   = (kb+1)*k_block_size;
    num_k = (k_overall_end - k_overall_start);

    /* Copy in B matrix */
    if(transB == 'Y')
    {
      ptr_dense = B + n_overall_start*handle->k + k_overall_start;
      for (k = 0; k < num_k; k++) {
        for (n = 0; n < num_n; n++) {
          uint16_t restmp = ptr_dense[n*handle->k + k];
          union { int i; float f; } res;
          res.i = restmp;
          res.i <<= 16;
          {
            scratch_B[k*num_regs*SIMD_WIDTH_FP32 + n] = res.f;
          }
        }
      }
    }
    else
    {
      ptr_dense = B + k_overall_start*handle->n + n_overall_start;
      if(!last_block_n) {
        for (k = 0; k < num_k; k++) {
          SIMDTYPE_INT32 vload_0 =  _MM_LOAD_INT32((const SIMDTYPE_INT32 *)(ptr_dense + k*handle->n + 2*0*SIMD_WIDTH_FP32));
          SIMDTYPE_INT32 vload_1, vload_2;
          SIMDTYPE_FP32 v1_0, v2_0;
          SIMDTYPE_FP32 v1_1, v2_1;
          SIMDTYPE_FP32 v1_2, v2_2;
          EXPAND_BFLOAT16(vload_0, v1_0, v2_0);
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + 2*0*SIMD_WIDTH_FP32, v1_0);
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + (2*0+1)*SIMD_WIDTH_FP32, v2_0);
          vload_1 =  _MM_LOAD_INT32((const SIMDTYPE_INT32 *)(ptr_dense + k*handle->n + 2*1*SIMD_WIDTH_FP32));
          EXPAND_BFLOAT16(vload_1, v1_1, v2_1);
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + 2*1*SIMD_WIDTH_FP32, v1_1);
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + (2*1+1)*SIMD_WIDTH_FP32, v2_1);
          vload_2 =  _MM_LOAD_INT32((const SIMDTYPE_INT32 *)(ptr_dense + k*handle->n + 2*2*SIMD_WIDTH_FP32));
          EXPAND_BFLOAT16(vload_2, v1_2, v2_2);
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + 2*2*SIMD_WIDTH_FP32, v1_2);
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + (2*2+1)*SIMD_WIDTH_FP32, v2_2);
        }
      } else {
        for (k = 0; k < num_k; k++) {
          for (n = 0; n < num_n; n++) {
            uint16_t restmp = ptr_dense[k*handle->n + n];
            union { int i; float f; } res;
            res.i = restmp;
            res.i <<= 16;
            {
              scratch_B[k*num_regs*SIMD_WIDTH_FP32 + n] = res.f;
            }
          }
        }
      }
    }
#if 0
    printf("B_col\n");
    for (k = 0; k < num_k; k++) {
      printf(" %lf ", ptr_dense[k*handle->n]);
    }
    printf("\n");
#endif
    scratch_C_base = scratch_C - m_overall_start*num_regs*SIMD_WIDTH_FP32;
    scratch_B_base = scratch_B; /* - k_overall_start*num_regs*SIMD_WIDTH_FP32; */

    for (m = m_overall_start; m < m_overall_start + num_m_aligned; m+=2, m_local+=2) {
      int start_j, end_j, end_j_2, num_j, num_j_2;
      const uint16_t*  LIBXSMM_RESTRICT sp_c_ptr_base;
      const uint16_t*  LIBXSMM_RESTRICT sp_c_ptr_base_2;
      const float* LIBXSMM_RESTRICT sp_v_ptr_base;
      const float* LIBXSMM_RESTRICT sp_v_ptr_base_2;
      float* const LIBXSMM_RESTRICT result_m_index = scratch_C_base + (m)*num_regs*SIMD_WIDTH_FP32;
      float* const LIBXSMM_RESTRICT result_m_index_2 = scratch_C_base + (m+1)*num_regs*SIMD_WIDTH_FP32;

      if( m_local >= m_block_size) { block_A++; slice = A_sparse[block_A]; m_local = 0; }

      start_j =  slice.rowidx[m_local];
      end_j   =  slice.rowidx[m_local + 1];
      end_j_2 =  slice.rowidx[m_local + 2];
      num_j   = (end_j - start_j);
      num_j_2   = (end_j_2 - end_j);
      sp_c_ptr_base = slice.colidx + start_j;
      sp_c_ptr_base_2 = slice.colidx + end_j;
      sp_v_ptr_base = (float *)(slice.values) + start_j;
      sp_v_ptr_base_2 = (float *)(slice.values) + end_j;

      if(!last_block_n)
      {
        int64_t j = 0, j2 = 0;
        sum[0] = _MM_LOAD_FP32(result_m_index + 0*SIMD_WIDTH_FP32);
        sum[0+num_regs] = _MM_LOAD_FP32(result_m_index_2 + 0*SIMD_WIDTH_FP32);
        sum[1] = _MM_LOAD_FP32(result_m_index + 1*SIMD_WIDTH_FP32);
        sum[1+num_regs] = _MM_LOAD_FP32(result_m_index_2 + 1*SIMD_WIDTH_FP32);
        sum[2] = _MM_LOAD_FP32(result_m_index + 2*SIMD_WIDTH_FP32);
        sum[2+num_regs] = _MM_LOAD_FP32(result_m_index_2 + 2*SIMD_WIDTH_FP32);
        sum[3] = _MM_LOAD_FP32(result_m_index + 3*SIMD_WIDTH_FP32);
        sum[3+num_regs] = _MM_LOAD_FP32(result_m_index_2 + 3*SIMD_WIDTH_FP32);
        sum[4] = _MM_LOAD_FP32(result_m_index + 4*SIMD_WIDTH_FP32);
        sum[4+num_regs] = _MM_LOAD_FP32(result_m_index_2 + 4*SIMD_WIDTH_FP32);
        sum[5] = _MM_LOAD_FP32(result_m_index + 5*SIMD_WIDTH_FP32);
        sum[5+num_regs] = _MM_LOAD_FP32(result_m_index_2 + 5*SIMD_WIDTH_FP32);
        for (; j < num_j && j2 < num_j_2; j++, j2++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index = scratch_B_base +  sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          const float* const LIBXSMM_RESTRICT sp_col_dense_index_2 = scratch_B_base + sp_c_ptr_base_2[j2]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]);
          SIMDTYPE_FP32 v_v_2 = _MM_SET1_FP32(sp_v_ptr_base_2[j2]);
          sum[0] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 0*SIMD_WIDTH_FP32), sum[0]);
          sum[0 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 0*SIMD_WIDTH_FP32), sum[0+num_regs]);
          sum[1] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 1*SIMD_WIDTH_FP32), sum[1]);
          sum[1 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 1*SIMD_WIDTH_FP32), sum[1+num_regs]);
          sum[2] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 2*SIMD_WIDTH_FP32), sum[2]);
          sum[2 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 2*SIMD_WIDTH_FP32), sum[2+num_regs]);
          sum[3] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 3*SIMD_WIDTH_FP32), sum[3]);
          sum[3 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 3*SIMD_WIDTH_FP32), sum[3+num_regs]);
          sum[4] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 4*SIMD_WIDTH_FP32), sum[4]);
          sum[4 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 4*SIMD_WIDTH_FP32), sum[4+num_regs]);
          sum[5] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 5*SIMD_WIDTH_FP32), sum[5]);
          sum[5 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 5*SIMD_WIDTH_FP32), sum[5+num_regs]);
        }
        for (; j < num_j; j++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index = scratch_B_base +  sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]);
          sum[0] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 0*SIMD_WIDTH_FP32), sum[0]);
          sum[1] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 1*SIMD_WIDTH_FP32), sum[1]);
          sum[2] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 2*SIMD_WIDTH_FP32), sum[2]);
          sum[3] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 3*SIMD_WIDTH_FP32), sum[3]);
          sum[4] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 4*SIMD_WIDTH_FP32), sum[4]);
          sum[5] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 5*SIMD_WIDTH_FP32), sum[5]);
        }
        for (; j2 < num_j_2; j2++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index_2 = scratch_B_base + sp_c_ptr_base_2[j2]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v_2 = _MM_SET1_FP32(sp_v_ptr_base_2[j2]);
          sum[0 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 0*SIMD_WIDTH_FP32), sum[0+num_regs]);
          sum[1 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 1*SIMD_WIDTH_FP32), sum[1+num_regs]);
          sum[2 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 2*SIMD_WIDTH_FP32), sum[2+num_regs]);
          sum[3 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 3*SIMD_WIDTH_FP32), sum[3+num_regs]);
          sum[4 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 4*SIMD_WIDTH_FP32), sum[4+num_regs]);
          sum[5 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 5*SIMD_WIDTH_FP32), sum[5+num_regs]);
        }
        _MM_STORE_FP32(result_m_index + 0*SIMD_WIDTH_FP32, sum[0]);
        _MM_STORE_FP32(result_m_index_2 + 0*SIMD_WIDTH_FP32, sum[0+num_regs]);
        _MM_STORE_FP32(result_m_index + 1*SIMD_WIDTH_FP32, sum[1]);
        _MM_STORE_FP32(result_m_index_2 + 1*SIMD_WIDTH_FP32, sum[1+num_regs]);
        _MM_STORE_FP32(result_m_index + 2*SIMD_WIDTH_FP32, sum[2]);
        _MM_STORE_FP32(result_m_index_2 + 2*SIMD_WIDTH_FP32, sum[2+num_regs]);
        _MM_STORE_FP32(result_m_index + 3*SIMD_WIDTH_FP32, sum[3]);
        _MM_STORE_FP32(result_m_index_2 + 3*SIMD_WIDTH_FP32, sum[3+num_regs]);
        _MM_STORE_FP32(result_m_index + 4*SIMD_WIDTH_FP32, sum[4]);
        _MM_STORE_FP32(result_m_index_2 + 4*SIMD_WIDTH_FP32, sum[4+num_regs]);
        _MM_STORE_FP32(result_m_index + 5*SIMD_WIDTH_FP32, sum[5]);
        _MM_STORE_FP32(result_m_index_2 + 5*SIMD_WIDTH_FP32, sum[5+num_regs]);
      }
      else {
        int64_t j = 0, j2 = 0;
        for (; j < num_j && j2 < num_j_2; j++, j2++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index = scratch_B_base +  sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          const float* const LIBXSMM_RESTRICT sp_col_dense_index_2 = scratch_B_base + sp_c_ptr_base_2[j2]*num_regs*SIMD_WIDTH_FP32;
          float v_v = sp_v_ptr_base[j];
          float v_v_2 = sp_v_ptr_base_2[j2];
          for (n = 0; n < num_n; n++) {
            result_m_index[n] += sp_col_dense_index[n]*v_v;
            result_m_index_2[n] += sp_col_dense_index_2[n]*v_v_2;
          }
        }
        for (; j < num_j; j++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index = scratch_B_base +  sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          float v_v = sp_v_ptr_base[j];
          for (n = 0; n < num_n; n++) {
            result_m_index[n] += sp_col_dense_index[n]*v_v;
          }
        }
        for (; j2 < num_j_2; j2++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index_2 = scratch_B_base + sp_c_ptr_base_2[j2]*num_regs*SIMD_WIDTH_FP32;
          float v_v_2 = sp_v_ptr_base_2[j2];
          for (n = 0; n < num_n; n++) {
            result_m_index_2[n] += sp_col_dense_index_2[n]*v_v_2;
          }
        }
      }
    }
    for (m = m_overall_start + num_m_aligned; m < m_overall_end; m++, m_local++) {
      int start_j, end_j, num_j;
      const uint16_t*  LIBXSMM_RESTRICT sp_c_ptr_base;
      const float* LIBXSMM_RESTRICT sp_v_ptr_base;
      float* LIBXSMM_RESTRICT result_m_index;

      if( m_local >= m_block_size) { block_A++; slice = A_sparse[block_A]; m_local = 0; }

      start_j =  slice.rowidx[m_local];
      end_j   =  slice.rowidx[m_local + 1];
      num_j   = (end_j - start_j);
      sp_c_ptr_base = slice.colidx + start_j;
      sp_v_ptr_base = slice.values + start_j;
      result_m_index = scratch_C_base + (m)*num_regs*SIMD_WIDTH_FP32;

      if(!last_block_n) {
        int64_t j = 0;
        sum[0] = _MM_LOAD_FP32(result_m_index + 0*SIMD_WIDTH_FP32);
        sum[1] = _MM_LOAD_FP32(result_m_index + 1*SIMD_WIDTH_FP32);
        sum[2] = _MM_LOAD_FP32(result_m_index + 2*SIMD_WIDTH_FP32);
        sum[3] = _MM_LOAD_FP32(result_m_index + 3*SIMD_WIDTH_FP32);
        sum[4] = _MM_LOAD_FP32(result_m_index + 4*SIMD_WIDTH_FP32);
        sum[5] = _MM_LOAD_FP32(result_m_index + 5*SIMD_WIDTH_FP32);
        for (; j < num_j; j++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index = scratch_B_base +  sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]);
          sum[0] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 0*SIMD_WIDTH_FP32), sum[0]);
          sum[1] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 1*SIMD_WIDTH_FP32), sum[1]);
          sum[2] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 2*SIMD_WIDTH_FP32), sum[2]);
          sum[3] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 3*SIMD_WIDTH_FP32), sum[3]);
          sum[4] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 4*SIMD_WIDTH_FP32), sum[4]);
          sum[5] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 5*SIMD_WIDTH_FP32), sum[5]);
        }
        _MM_STORE_FP32(result_m_index + 0*SIMD_WIDTH_FP32, sum[0]);
        _MM_STORE_FP32(result_m_index + 1*SIMD_WIDTH_FP32, sum[1]);
        _MM_STORE_FP32(result_m_index + 2*SIMD_WIDTH_FP32, sum[2]);
        _MM_STORE_FP32(result_m_index + 3*SIMD_WIDTH_FP32, sum[3]);
        _MM_STORE_FP32(result_m_index + 4*SIMD_WIDTH_FP32, sum[4]);
        _MM_STORE_FP32(result_m_index + 5*SIMD_WIDTH_FP32, sum[5]);
      }
      else {
        int64_t j = 0;
        for (; j < num_j; j++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index = scratch_B_base +  sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          float v_v = sp_v_ptr_base[j];
          for (n = 0; n < num_n; n++) {
            result_m_index[n] += sp_col_dense_index[n]*v_v;
          }
        }
      }
    }
  } /* kb */
#if 0
  for (m = 0; m < 3; m++) {
    for (n = 0; n < num_n; n++) {
      printf("%f ", scratch_C[m*num_regs*SIMD_WIDTH_FP32 + n]);
    }
    printf("\n");
  }
#endif
  /* Copy out C matrix */
  if(!last_block_n) {
    for (m = 0; m < num_m; m++) {
      SIMDTYPE_FP32 vload1_0 =  _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 2*0*SIMD_WIDTH_FP32);
      SIMDTYPE_FP32 vload2_0 =  _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + (2*0+1)*SIMD_WIDTH_FP32);
      SIMDTYPE_FP32 vload1_1, vload2_1, vload1_2, vload2_2;
      SIMDTYPE_INT32 v_0, v_1, v_2;
      COMPRESS_BFLOAT16(vload1_0, vload2_0, v_0);
      _MM_STORE_INT32((SIMDTYPE_INT32 *)(ptr_result + m*handle->n + 2*0*SIMD_WIDTH_FP32), v_0);
      vload1_1 =  _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 2*1*SIMD_WIDTH_FP32);
      vload2_1 =  _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + (2*1+1)*SIMD_WIDTH_FP32);
      COMPRESS_BFLOAT16(vload1_1, vload2_1, v_1);
      _MM_STORE_INT32((SIMDTYPE_INT32 *)(ptr_result + m*handle->n + 2*1*SIMD_WIDTH_FP32), v_1);
      vload1_2 =  _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 2*2*SIMD_WIDTH_FP32);
      vload2_2 =  _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + (2*2+1)*SIMD_WIDTH_FP32);
      COMPRESS_BFLOAT16(vload1_2, vload2_2, v_2);
      _MM_STORE_INT32((SIMDTYPE_INT32 *)(ptr_result + m*handle->n + 2*2*SIMD_WIDTH_FP32), v_2);
    }
  } else {
    for (m = 0; m < num_m; m++) {
      for (n = 0; n < num_n; n++) {
         int v = *(int *)(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + n);
         ptr_result[m*handle->n + n] = (uint16_t)(v >> 16);
      }
    }
  }

  LIBXSMM_SPMDM_FREE(scratch_C);
  LIBXSMM_SPMDM_FREE(scratch_B);
}

