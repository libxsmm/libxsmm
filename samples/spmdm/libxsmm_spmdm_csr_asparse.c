/******************************************************************************
 * ** Copyright (c) 2016, Intel Corporation                                     **
 * ** All rights reserved.                                                      **
 * **                                                                           **
 * ** Redistribution and use in source and binary forms, with or without        **
 * ** modification, are permitted provided that the following conditions        **
 * ** are met:                                                                  **
 * ** 1. Redistributions of source code must retain the above copyright         **
 * **    notice, this list of conditions and the following disclaimer.          **
 * ** 2. Redistributions in binary form must reproduce the above copyright      **
 * **    notice, this list of conditions and the following disclaimer in the    **
 * **    documentation and/or other materials provided with the distribution.   **
 * ** 3. Neither the name of the copyright holder nor the names of its          **
 * **    contributors may be used to endorse or promote products derived        **
 * **    from this software without specific prior written permission.          **
 * **                                                                           **
 * ** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
 * ** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
 * ** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
 * ** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
 * ** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
 * ** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
 * ** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
 * ** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
 * ** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
 * ** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
 * ** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
 * ******************************************************************************/
/* Nadathur Satish (Intel Corp.)
 * ******************************************************************************/

/* NOTE: This code currently ignores alpha, beta and trans inputs to the matrix multiply */
#include <libxsmm.h>
#include <libxsmm_intrinsics_x86.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#if defined(_WIN32) || defined(__CYGWIN__)
/* note: this does not reproduce 48-bit RNG quality */
# define drand48() ((double)rand() / RAND_MAX)
# define srand48 srand
#endif

#define USE_BFLOAT
#ifdef USE_BFLOAT
typedef uint16_t real;
#else
typedef float real;
#endif

typedef enum libxsmm_spmdm_datatype {
  LIBXSMM_SPMDM_DATATYPE_F32,
  LIBXSMM_SPMDM_DATATYPE_BFLOAT16
} libxsmm_spmdm_datatype;

typedef struct libxsmm_spmdm_handle {
  /* The following are the matrix multiply dimensions: A (sparse): m X k, B (dense): k X n, Output C (dense): m X n */
  int m;
  int n;
  int k;
  /* The block sizes for A, B and C. */
  /* Here we fix A to be divided into 128 X 128 blocks, B/C to be 128 X 48 for HSW/BDW and 128 X 96 for SKX */
  int bm;
  int bn;
  int bk;
  /* The number of blocks for the m, n and k dimensions */
  int mb;
  int nb;
  int kb;
  libxsmm_spmdm_datatype datatype;
} libxsmm_spmdm_handle;

/* This stores a single sparse splice (or block) of sparse matrix A using a CSR representation (rowidx, colidx, and values */
/* Each splice corresponds to a bm X bk region of A, and stores local indices */
typedef struct libxsmm_CSR_sparseslice {
  /* Since bm and bk are assumed to be <=256, a 16-bit integer is enough to store the local rowidx, colidx */
  uint16_t * rowidx;
  uint16_t * colidx;
  float*     values; 
} libxsmm_CSR_sparseslice;



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
#define _MM_LOAD_FP32 _mm512_load_ps
#define _MM_LOAD_INT32 _mm512_load_epi32
#define _MM_STORE_INT32 _mm512_store_epi32
#define _MM_LOADU_INT32 _mm512_loadu_si512
#define _MM_CMPNEQ_FP32(v1,v2) _mm512_cmp_ps_mask(v1,v2,12)
#define _MM_STORE_FP32 _mm512_store_ps
#define _MM_ADD_FP32 _mm512_add_ps
#define _MM_FMADD_FP32 _mm512_fmadd_ps
static void _mm512_print(__m512 a, char * s)
{
  float *v=(float *)(&a);
  int i;
  printf("[%8s(%3lud)]: ", s, (size_t)16);
  for(i=0; i < 16; i++)
    printf("%4f ", v[i]);
  printf("\n");
}

static void _mm512i_print(__m512i a, char * s)
{
  int *v=(int *)(&a);
  int i;
  printf("[%8s(%3lud)]: ", s, (size_t)16);
  for(i=0; i < 16; i++)
    printf("%4d ", v[i]);
  printf("\n");
}

static void _mm512i_epi16_print(__m512i a, char * s)
{
  uint16_t *v=(uint16_t*)(&a);
  int i;
  printf("[%8s(%3lud)]: ", s, (size_t)32);
  for(i=0; i < 32; i++)
    printf("%4d ", v[i]);
  printf("\n");
}

static void _mm256i_epi16_print(__m256i a, char * s)
{
  uint16_t *v=(uint16_t*)(&a);
  int i;
  printf("[%8s(%3lud)]: ", s, (size_t)16);
  for(i=0; i < 16; i++)
    printf("%4d ", v[i]);
  printf("\n");
}


  //__m512i vtmp = _mm512_set_epi32(0,0,0,0,0,0,0,0,0xF000E,0xD000C,0xB000A,0x90008,0x70006,0x50004,0x30002,0x10000);\
//_mm256_add_epi32(vk1, _mm256_load_si256(&shufmasks2[m&0xFF])); \

#define COMPRESS_FP32(v, k, m, cnt) \
  { \
  _mm512_mask_compressstoreu_ps(values_ptr +  cnt, m, v); \
  __m256i vk1 = _mm256_set1_epi16((short)k); \
  __m256i vk2 = _mm256_set1_epi16((short)(k + 8)); \
  __m256i v_idx = _mm256_add_epi32(vk1, _mm256_load_si256(&shufmasks2[m&0xFF])); \
  __m256i v_idx_2 = _mm256_add_epi32(vk2, _mm256_load_si256(&shufmasks2[(m>>8)&0xFF])); \
  _mm256_storeu_si256((__m256i *)(colidx_ptr +  cnt), v_idx); \
  cnt += _mm_countbits_32(m&0xFF); \
  _mm256_storeu_si256((__m256i *)(colidx_ptr +  cnt), v_idx_2); \
  cnt += _mm_countbits_32((m>>8)&0xFF); \
  }

#define EXPAND_BFLOAT16(v, vlo_final, vhi_final) \
  { \
  __m512i vlo = _mm512_unpacklo_epi16(vzero, v); \
  __m512i vhi = _mm512_unpackhi_epi16(vzero, v); \
  __m512i permmask1 = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0); \
  __m512i permmask2 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4); \
  vlo_final = _mm512_castsi512_ps(_mm512_permutex2var_epi64(vlo, permmask1, vhi)); \
  vhi_final = _mm512_castsi512_ps(_mm512_permutex2var_epi64(vlo, permmask2, vhi)); \
  } 

#define COMPRESS_BFLOAT16(vlo, vhi, v) \
  { \
  __m512i permmask1 = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0); \
  __m512i permmask2 = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2); \
  __m512i vtmp1 =  _mm512_permutex2var_epi64(_mm512_castps_si512(vlo), permmask1, _mm512_castps_si512(vhi)); \
  __m512i vtmp2 =  _mm512_permutex2var_epi64(_mm512_castps_si512(vlo), permmask2, _mm512_castps_si512(vhi)); \
  v = _mm512_packus_epi32(_mm512_srli_epi32(vtmp1,16), _mm512_srli_epi32(vtmp2,16)); \
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
#define _MM_LOAD_FP32 _mm256_load_ps
#define _MM_LOAD_INT32 _mm256_load_si256
#define _MM_STORE_INT32 _mm256_store_si256
#define _MM_LOADU_INT32 _mm256_loadu_si256
#define _MM_CMPNEQ_FP32(v1,v2) _mm256_cmp_ps(v1,v2,12)
#define _MM_STORE_FP32 _mm256_store_ps
#define _MM_ADD_FP32 _mm256_add_ps
#define _MM_FMADD_FP32 _mm256_fmadd_ps
static void _mm256_print(__m256 a, char * s)
{
  float *v=(float *)(&a);
  int i;
  printf("[%8s(%3lud)]: ", s, (size_t)8);
  for(i=0; i < 8; i++)
    printf("%4f ", v[i]);
  printf("\n");
}

static void _mm256i_print(__m256i a, char * s)
{
  int *v=(int *)(&a);
  int i;
  printf("[%8s(%3lud)]: ", s, (size_t)8);
  for(i=0; i < 8; i++)
    printf("%4d ", v[i]);
  printf("\n");
}

static void _mm256i_epi16_print(__m256i a, char * s)
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
  SIMDTYPE_INT32 vk = _MM_SET1_INT16(k); \
  __m256i perm_ctrl = _mm256_load_si256(&shufmasks[mask]); \
  __m256 v_packed = _mm256_permutevar8x32_ps(v, perm_ctrl); \
  __m256i v_idx = _mm256_add_epi32(vk, _mm256_load_si256(&shufmasks2[mask])); \
  _mm256_storeu_ps(values_ptr +  cnt, v_packed); \
  _mm256_storeu_si256((__m256i *)(colidx_ptr +  cnt), v_idx); \
  cnt += _mm_countbits_32(mask); \
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
  __m256i vtmp1 =  _mm256_castps_si256(_mm256_permute2f128_ps(vlo, vhi, 0x20)); \
  __m256i vtmp2 =  _mm256_castps_si256(_mm256_permute2f128_ps(vlo, vhi, 0x31)); \
  v = _mm256_packus_epi32(_mm256_srli_epi32(vtmp1,16), _mm256_srli_epi32(vtmp2,16)); \
  }

#endif

void libxsmm_spmdm_init_shufmask( __m256i * shufmasks_32, __m256i * shufmasks_16)
{
  unsigned int i,j, c, last_bit;
  int __attribute__((aligned (64))) temp_shufmasks[8];
  uint16_t __attribute__((aligned (64))) temp_shufmasks2[16];
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
    shufmasks_32[i] = _mm256_loadu_si256((const __m256i*)temp_shufmasks);
    shufmasks_16[i] = _mm256_loadu_si256((const __m256i*)temp_shufmasks2);
  }
}


void libxsmm_spmdm_allocate_csr_a( const libxsmm_spmdm_handle* handle, libxsmm_CSR_sparseslice ** libxsmm_output_csr, char transA)
{
    int kb, mb;
    int m_blocks = handle->mb;
    int k_blocks = handle->kb;

    libxsmm_CSR_sparseslice* libxsmm_output_csr_a = (libxsmm_CSR_sparseslice *)libxsmm_aligned_malloc( handle->mb * handle->kb * sizeof(libxsmm_CSR_sparseslice), 2097152);
    for ( kb = 0; kb < k_blocks; kb++ ) {
      for ( mb = 0; mb < m_blocks; mb++ ) {
        int i = kb*m_blocks + mb;
        if(transA == 'Y') {
          libxsmm_output_csr_a[i].rowidx = (uint16_t *)libxsmm_aligned_malloc((handle->bk + 1)*sizeof(uint16_t), 2097152); 
        } else {
          libxsmm_output_csr_a[i].rowidx = (uint16_t *)libxsmm_aligned_malloc((handle->bm + 1)*sizeof(uint16_t), 2097152); 
        }
        libxsmm_output_csr_a[i].colidx = (uint16_t *)libxsmm_aligned_malloc((handle->bm)*(handle->bk)*sizeof(uint16_t), 2097152); 
        libxsmm_output_csr_a[i].values = (float *)libxsmm_aligned_malloc((handle->bm)*(handle->bk)*sizeof(float), 2097152); 
      }
    }

    *libxsmm_output_csr = libxsmm_output_csr_a;

}

/* This converts a dense representation of the sparse matrix to 2D array of sparse slices. */
void libxsmm_spmdm_createSparseSlice_fp32_notrans_thread( const libxsmm_spmdm_handle* handle,
				const float * A, 
				const int transA,
				libxsmm_CSR_sparseslice* libxsmm_output_csr_a,
                                int mb, int kb,
				int tid, int nthreads, 
                                __m256i * shufmasks,
                                __m256i * shufmasks2
                               ) 
{
   int i,k;
   int block_offset_base = kb * handle->bk;
   int block_offset = block_offset_base + mb * handle->k * handle->bm;
   libxsmm_CSR_sparseslice slice = libxsmm_output_csr_a[kb*handle->mb + mb];
   int nrows = ((mb + 1)*handle->bm > handle->m)?(handle->m - (mb)*handle->bm):handle->bm;
   int ncols = ((kb + 1)*handle->bk > handle->k)?(handle->k - (kb)*handle->bk):handle->bk;
   //printf("nrows: %d, ncols: %d\n", nrows, ncols);
   int ncols_aligned = ncols / (4*SIMD_WIDTH_FP32)*(4*SIMD_WIDTH_FP32);
   int ncols_aligned_2 = ncols / (SIMD_WIDTH_FP32)*(SIMD_WIDTH_FP32);
   const float * input_ptr = A + block_offset;
   uint16_t * rowidx_ptr = slice.rowidx;
   uint16_t * colidx_ptr = slice.colidx;
   float    * values_ptr = (float *)(slice.values);
   SIMDTYPE_FP32 vzero = _MM_SET1_FP32(0.0);
   uint16_t cnt = 0;
   for(i = 0; i < nrows; i++) {
     rowidx_ptr[i] = cnt;
     for(k = 0; k < ncols_aligned; k+= 4*SIMD_WIDTH_FP32) {
       SIMDTYPE_FP32 v1 = _MM_LOAD_FP32(input_ptr + i*handle->k + k);
       _mm_prefetch((char *)(input_ptr + (i+2)*handle->k + k), _MM_HINT_T0);
       SIMDTYPE_FP32 v2 = _MM_LOAD_FP32(input_ptr + i*handle->k + k + SIMD_WIDTH_FP32);
       _mm_prefetch((char *)(input_ptr + (i+2)*handle->k + k + SIMD_WIDTH_FP32), _MM_HINT_T0);
       SIMDTYPE_FP32 v3 = _MM_LOAD_FP32(input_ptr + i*handle->k + k + 2*SIMD_WIDTH_FP32);
       _mm_prefetch((char *)(input_ptr + (i+2)*handle->k + k + 2*SIMD_WIDTH_FP32), _MM_HINT_T0);
       SIMDTYPE_FP32 v4 = _MM_LOAD_FP32(input_ptr + i*handle->k + k + 3*SIMD_WIDTH_FP32);
       _mm_prefetch((char *)(input_ptr + (i+2)*handle->k + k + 3*SIMD_WIDTH_FP32), _MM_HINT_T0);
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
       SIMDTYPE_FP32 v1 = _MM_LOAD_FP32(input_ptr + i*handle->k + k);
       _mm_prefetch((char *)(input_ptr + (i+2)*handle->k + k), _MM_HINT_T0);
       SIMDMASKTYPE_FP32 m1 = _MM_CMPNEQ_FP32(v1, vzero);
       COMPRESS_FP32(v1, k, m1, cnt);
     }
     for(k = ncols_aligned_2; k < ncols; k++) {
       float v1 = input_ptr[i*handle->k + k];
       int m1 = (v1 != 0.0);
       if(m1) { colidx_ptr[cnt] = (uint16_t)k; values_ptr[cnt] = v1; cnt++; }
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

void libxsmm_spmdm_createSparseSlice_bfloat16_notrans_thread( const libxsmm_spmdm_handle* handle,
				const uint16_t * A, 
				const int transA,
				libxsmm_CSR_sparseslice* libxsmm_output_csr_a,
                                int mb, int kb,
				int tid, int nthreads,
                                __m256i * shufmasks,
                                __m256i * shufmasks2) 
{
   int i,k;
   int block_offset_base = kb * handle->bk;
   int block_offset = block_offset_base + mb * handle->k * handle->bm;
   libxsmm_CSR_sparseslice slice = libxsmm_output_csr_a[kb*handle->mb + mb];
   int nrows = ((mb + 1)*handle->bm > handle->m)?(handle->m - (mb)*handle->bm):handle->bm;
   int ncols = ((kb + 1)*handle->bk > handle->k)?(handle->k - (kb)*handle->bk):handle->bk;
   //printf("nrows: %d, ncols: %d\n", nrows, ncols);
   int ncols_aligned = ncols / (4*SIMD_WIDTH_FP32)*(4*SIMD_WIDTH_FP32);
   const uint16_t * input_ptr = A + block_offset;
   uint16_t * rowidx_ptr = slice.rowidx;
   uint16_t * colidx_ptr = slice.colidx;
   float * values_ptr = (float *)(slice.values);
   SIMDTYPE_INT32 vzero = _MM_SET1_INT32(0);
   SIMDTYPE_FP32 vzerof = _MM_SET1_FP32(0.0);
   uint16_t cnt = 0;
   for(i = 0; i < nrows; i++) {
     rowidx_ptr[i] = cnt;
     for(k = 0; k < ncols_aligned; k+= 4*SIMD_WIDTH_FP32) {
       SIMDTYPE_INT32 v1tmp = _MM_LOAD_INT32((const SIMDTYPE_INT32* )(input_ptr + i*handle->k + k));
       _mm_prefetch((char *)(input_ptr + (i+2)*handle->k + k), _MM_HINT_T0);
       SIMDTYPE_INT32 v2tmp = _MM_LOAD_INT32((const SIMDTYPE_INT32*)(input_ptr + i*handle->k + k + 2*SIMD_WIDTH_FP32));
       _mm_prefetch((char *)(input_ptr + (i+2)*handle->k + k + SIMD_WIDTH_FP32), _MM_HINT_T0);
       SIMDTYPE_FP32 v1, v2, v3, v4; 
       EXPAND_BFLOAT16(v1tmp, v1, v2);
       EXPAND_BFLOAT16(v2tmp, v3, v4);
       SIMDMASKTYPE_FP32 m1 = _MM_CMPNEQ_FP32(v1, vzerof);
       SIMDMASKTYPE_FP32 m2 = _MM_CMPNEQ_FP32(v2, vzerof);
       SIMDMASKTYPE_FP32 m3 = _MM_CMPNEQ_FP32(v3, vzerof);
       SIMDMASKTYPE_FP32 m4 = _MM_CMPNEQ_FP32(v4, vzerof);
       COMPRESS_FP32(v1, k, m1, cnt);
       COMPRESS_FP32(v2, k + SIMD_WIDTH_FP32, m2, cnt);
       COMPRESS_FP32(v3, k + 2*SIMD_WIDTH_FP32, m3, cnt);
       COMPRESS_FP32(v4, k + 3*SIMD_WIDTH_FP32, m4, cnt);
     }
     for(k = ncols_aligned; k < ncols; k++) {
       uint16_t v1tmp = input_ptr[i*handle->k + k];
       int v1tmp_int  = v1tmp; v1tmp_int <<= 16;
       float v1 = *(float *)&v1tmp_int;
       int m1 = (v1 != 0.0);
       if(m1) { colidx_ptr[cnt] = (uint16_t)k; values_ptr[cnt] = v1; cnt++; }
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



void libxsmm_spmdm_createSparseSlices_fp32( const libxsmm_spmdm_handle* handle,
				const float* A, /* This is of type libxsmm_spmdm_datatype */
				const int transA,
				libxsmm_CSR_sparseslice* A_sparse,
                                __m256i * shufmasks,
                                __m256i * shufmasks2) 
{
  int m_blocks = handle->mb;
  int k_blocks = handle->kb;
  int mb, kb;
  if( transA == 'Y') {
  }
  else {
    #pragma omp parallel for collapse(2)
    for ( kb = 0; kb < k_blocks; kb++ ) {
      for ( mb = 0; mb < m_blocks; mb++ ) {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        libxsmm_spmdm_createSparseSlice_fp32_notrans_thread( handle, A, transA, A_sparse, mb, kb, tid, nthreads, shufmasks, shufmasks2);
      }
    }
  }		
}

void libxsmm_spmdm_createSparseSlices_bfloat16( const libxsmm_spmdm_handle* handle,
				const uint16_t* A, /* This is of type libxsmm_spmdm_datatype */
				const int transA,
				libxsmm_CSR_sparseslice* A_sparse,
                                __m256i * shufmasks,
                                __m256i * shufmasks2) 
{
  int m_blocks = handle->mb;
  int k_blocks = handle->kb;
  int mb, kb;
  if( transA == 'Y') {
  }
  else {
    #pragma omp parallel for collapse(2)
    for ( kb = 0; kb < k_blocks; kb++ ) {
      for ( mb = 0; mb < m_blocks; mb++ ) {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        libxsmm_spmdm_createSparseSlice_bfloat16_notrans_thread( handle, A, transA, A_sparse, mb, kb, tid, nthreads, shufmasks, shufmasks2);
      }
    }
  }		
}


void libxsmm_spmdm_compute_fp32_thread( const libxsmm_spmdm_handle* handle,
                            const float *alpha, 
                            libxsmm_CSR_sparseslice* A_sparse, 
                            const float *B, 
                            const float *beta, 
                            float* C,
                            int mb, int num_m_blocks, int nb, 
                            int tid, int nthreads) 
{
  const int m_blocks = handle->mb;
  const int n_blocks = handle->nb;
  const int k_blocks = handle->kb;
  const int m_block_size = handle->bm;
  const int n_block_size = handle->bn;
  const int k_block_size = handle->bk;

#define num_regs (6) 
// really is twice this
  assert(n_block_size == num_regs*SIMD_WIDTH_FP32);
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
 
  float __attribute__((aligned (64))) scratch_C[num_m_blocks*m_block_size*n_block_size];
  float __attribute__((aligned (64))) scratch_B[k_block_size*n_block_size];
  SIMDTYPE_FP32 sum[2*num_regs];
  float* __restrict__ ptr_result;
    
  if (m_overall_end   > handle->m) m_overall_end   = handle->m;
  num_m = (m_overall_end - m_overall_start);
  num_m_aligned = (num_m / 2) * 2;

  if (n_overall_end   > handle->n) n_overall_end   = handle->n;
  num_n = (n_overall_end - n_overall_start);
  last_block_n = (num_n != n_block_size);
  num_full_regs = 0; // (num_n / SIMD_WIDTH_FP32);
  last_n_start = num_full_regs*SIMD_WIDTH_FP32;

  #if 0
  printf("Block: m_overall_start: %d, m_overall_end: %d, num_m: %d, num_m_aligned: %d\n", m_overall_start, m_overall_end, num_m, num_m_aligned); 
  printf("Block: n_overall_start: %d, n_overall_end: %d, num_n: %d, last_block_n: %d\n", n_overall_start, n_overall_end, num_n, last_block_n); 
  printf("Block: k_blocks: %d\n", k_blocks);
  #endif
  // Copy in C matrix to buffer
  ptr_result = C + m_overall_start*handle->n + n_overall_start;
  if(!last_block_n) {
    for (m = 0; m < num_m; m++) {
      #pragma unroll (num_regs)
      for (n = 0; n < num_regs; n++) {
        _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + n*SIMD_WIDTH_FP32, _MM_LOAD_FP32(ptr_result + m*handle->n + n*SIMD_WIDTH_FP32));
      }
    }
  } else {
    for (m = 0; m < num_m; m++) {
      #pragma unroll (2)
      for (n = 0; n < num_full_regs; n++) {
        _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + n*SIMD_WIDTH_FP32, _MM_LOAD_FP32(ptr_result + m*handle->n + n*SIMD_WIDTH_FP32));
      }
      for (n = last_n_start; n < num_n; n++) {
        scratch_C[m*num_regs*SIMD_WIDTH_FP32 + n] = ptr_result[m*handle->n + n];
      }
    }
  }	

  for (kb = 0; kb < k_blocks; kb++) {
    const float * __restrict__ ptr_dense;
    float * __restrict__ scratch_C_base;
    const float * __restrict__ scratch_B_base;
    int block_A = kb * m_blocks + mb;
    libxsmm_CSR_sparseslice slice = A_sparse[block_A];
    int m_local = 0;

    k_overall_start = kb*k_block_size;
    k_overall_end   = (kb+1)*k_block_size;
    num_k = (k_overall_end - k_overall_start);
     
    // Copy in B matrix
    ptr_dense = B + k_overall_start*handle->n + n_overall_start;
    if(!last_block_n) {
      for (k = 0; k < num_k; k++) {
        #pragma unroll (num_regs)
        for (n = 0; n < num_regs; n++) {
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + n*SIMD_WIDTH_FP32, _MM_LOAD_FP32(ptr_dense + k*handle->n + n*SIMD_WIDTH_FP32));
        }
      }
    } else {
      for (k = 0; k < num_k; k++) {
        #pragma unroll (2)
        for (n = 0; n < num_full_regs; n++) {
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + n*SIMD_WIDTH_FP32, _MM_LOAD_FP32(ptr_dense + k*handle->n + n*SIMD_WIDTH_FP32));
        }
        for (n = last_n_start; n < num_n; n++) {
          scratch_B[k*num_regs*SIMD_WIDTH_FP32 + n] = ptr_dense[k*handle->n + n];
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
    scratch_B_base = scratch_B; // - k_overall_start*num_regs*SIMD_WIDTH_FP32;
    
    for (m = m_overall_start; m < m_overall_start + num_m_aligned; m+=2, m_local+=2) {
      int start_j, end_j, end_j_2, num_j, num_j_2;
      const uint16_t*  __restrict__ sp_c_ptr_base; 
      const uint16_t*  __restrict__ sp_c_ptr_base_2; 
      const float* __restrict__ sp_v_ptr_base;
      const float* __restrict__ sp_v_ptr_base_2;

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
      float* const __restrict__ result_m_index = scratch_C_base + (m)*num_regs*SIMD_WIDTH_FP32;
      float* const __restrict__ result_m_index_2 = scratch_C_base + (m+1)*num_regs*SIMD_WIDTH_FP32;
      
      if(!last_block_n) 
      {
        int64_t j = 0, j2 = 0;
        #pragma unroll (num_regs)
        for (n = 0; n < num_regs; n++) {
          sum[n] = _MM_LOAD_FP32(result_m_index + n*SIMD_WIDTH_FP32);
          sum[n+num_regs] = _MM_LOAD_FP32(result_m_index_2 + n*SIMD_WIDTH_FP32);
        }
        for (; j < num_j && j2 < num_j_2; j++, j2++) {
          const float* const __restrict__ sp_col_dense_index = scratch_B_base +  sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          const float* const __restrict__ sp_col_dense_index_2 = scratch_B_base + sp_c_ptr_base_2[j2]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]); 
          SIMDTYPE_FP32 v_v_2 = _MM_SET1_FP32(sp_v_ptr_base_2[j2]); 
          #pragma unroll (num_regs)
          for (n = 0; n < num_regs; n++) {
            sum[n] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + n*SIMD_WIDTH_FP32), sum[n]);
            sum[n + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + n*SIMD_WIDTH_FP32), sum[n+num_regs]);
          }
        }
        for (; j < num_j; j++) {
          const float* const __restrict__ sp_col_dense_index = scratch_B_base +  sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]); 
          #pragma unroll (num_regs)
          for (n = 0; n < num_regs; n++) {
            sum[n] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + n*SIMD_WIDTH_FP32), sum[n]);
          }
        }
        for (; j2 < num_j_2; j2++) {
          const float* const __restrict__ sp_col_dense_index_2 = scratch_B_base + sp_c_ptr_base_2[j2]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v_2 = _MM_SET1_FP32(sp_v_ptr_base_2[j2]); 
          #pragma unroll (num_regs)
          for (n = 0; n < num_regs; n++) {
            sum[n + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + n*SIMD_WIDTH_FP32), sum[n+num_regs]);
          }
        }
        #pragma unroll (num_regs)
        for (n = 0; n < num_regs; n++) {
          _MM_STORE_FP32(result_m_index + n*SIMD_WIDTH_FP32, sum[n]);
          _MM_STORE_FP32(result_m_index_2 + n*SIMD_WIDTH_FP32, sum[n+num_regs]);
        }
      }
      else {
        int64_t j = 0, j2 = 0;
        #pragma unroll (2)
        for (n = 0; n < num_full_regs; n++) {
          sum[n] = _MM_SETZERO_FP32();
          sum[n+num_regs] = _MM_SETZERO_FP32();
        }
        for (; j < num_j && j2 < num_j_2; j++, j2++) {
          const float* const __restrict__ sp_col_dense_index = scratch_B_base +  sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          const float* const __restrict__ sp_col_dense_index_2 = scratch_B_base + sp_c_ptr_base_2[j2]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]); 
          SIMDTYPE_FP32 v_v_2 = _MM_SET1_FP32(sp_v_ptr_base_2[j2]); 
          #pragma unroll (2)
          for (n = 0; n < num_full_regs; n++) {
            sum[n] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + n*SIMD_WIDTH_FP32), sum[n]);
            sum[n + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + n*SIMD_WIDTH_FP32), sum[n+num_regs]);
          }
          for (n = last_n_start; n < num_n; n++) {
            result_m_index[n] += sp_col_dense_index[n]*sp_v_ptr_base[j];
            result_m_index_2[n] += sp_col_dense_index_2[n]*sp_v_ptr_base_2[j2];
          } 
        }
        for (; j < num_j; j++) {
          const float* const __restrict__ sp_col_dense_index = scratch_B_base +  sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]); 
          #pragma unroll (2)
          for (n = 0; n < num_full_regs; n++) {
            sum[n] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + n*SIMD_WIDTH_FP32), sum[n]);
          }
          for (n = last_n_start; n < num_n; n++) {
            result_m_index[n] += sp_col_dense_index[n]*sp_v_ptr_base[j];
          } 
        }
        for (; j2 < num_j_2; j2++) {
          const float* const __restrict__ sp_col_dense_index_2 = scratch_B_base + sp_c_ptr_base_2[j2]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v_2 = _MM_SET1_FP32(sp_v_ptr_base_2[j2]); 
          #pragma unroll (2)
          for (n = 0; n < num_full_regs; n++) {
            sum[n + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + n*SIMD_WIDTH_FP32), sum[n+num_regs]);
          }
          for (n = last_n_start; n < num_n; n++) {
            result_m_index_2[n] += sp_col_dense_index_2[n]*sp_v_ptr_base_2[j2];
          } 
        }
        #pragma unroll (2)
        for (n = 0; n < num_full_regs; n++) {
          _MM_STORE_FP32(result_m_index + n*SIMD_WIDTH_FP32,  _MM_ADD_FP32(sum[n], _MM_LOAD_FP32(result_m_index + n*SIMD_WIDTH_FP32)));
          _MM_STORE_FP32(result_m_index_2 + n*SIMD_WIDTH_FP32,  _MM_ADD_FP32(sum[n+num_regs], _MM_LOAD_FP32(result_m_index_2 + n*SIMD_WIDTH_FP32)));
        }
      }
    }
    for (m = m_overall_start + num_m_aligned; m < m_overall_end; m++, m_local++) {
      int start_j, end_j, num_j;
      const uint16_t*  __restrict__ sp_c_ptr_base; 
      const float* __restrict__ sp_v_ptr_base;
      float* __restrict__ result_m_index;

      if( m_local >= m_block_size) { block_A++; slice = A_sparse[block_A]; m_local = 0; }

      start_j =  slice.rowidx[m_local];
      end_j   =  slice.rowidx[m_local + 1];
      num_j   = (end_j - start_j);
      sp_c_ptr_base = slice.colidx + start_j;	
      sp_v_ptr_base = slice.values + start_j;
      result_m_index = scratch_C_base + (m)*num_regs*SIMD_WIDTH_FP32;
      
      if(!last_block_n) {
        int64_t j = 0;
        #pragma unroll (num_regs)
        for (n = 0; n < num_regs; n++) {
          sum[n] = _MM_LOAD_FP32(result_m_index + n*SIMD_WIDTH_FP32);
        }
        for (; j < num_j; j++) {
          const float* const __restrict__ sp_col_dense_index = scratch_B_base +  sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]); 
          #pragma unroll (num_regs)
          for (n = 0; n < num_regs; n++) {
            sum[n] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + n*SIMD_WIDTH_FP32), sum[n]);
          }
        }
        #pragma unroll (num_regs)
        for (n = 0; n < num_regs; n++) {
          _MM_STORE_FP32(result_m_index + n*SIMD_WIDTH_FP32, sum[n]);
        }
      }
      else {
        int64_t j = 0;
        #pragma unroll (2)
        for (n = 0; n < num_full_regs; n++) {
          sum[n] = _MM_SETZERO_FP32();
        }
        for (; j < num_j; j++) {
          const float* const __restrict__ sp_col_dense_index = scratch_B_base +  sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]); 
          #pragma unroll (2)
          for (n = 0; n < num_full_regs; n++) {
            sum[n] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + n*SIMD_WIDTH_FP32), sum[n]);
          }
          for (n = last_n_start; n < num_n; n++) {
            result_m_index[n] += sp_col_dense_index[n]*sp_v_ptr_base[j];
          } 
        }
        #pragma unroll (2)
        for (n = 0; n < num_full_regs; n++) {
          _MM_STORE_FP32(result_m_index + n*SIMD_WIDTH_FP32, _MM_ADD_FP32(sum[n], _MM_LOAD_FP32(result_m_index + n*SIMD_WIDTH_FP32)));
        }
      }
    }
  } // kb
  #if 0
  for (m = 0; m < 3; m++) {
    for (n = 0; n < num_n; n++) {
      printf("%f ", scratch_C[m*num_regs*SIMD_WIDTH_FP32 + n]);
    }
    printf("\n");
  }
  #endif
  // Copy out C matrix
  if(!last_block_n) {
    for (m = 0; m < num_m; m++) {
      #pragma unroll (num_regs)
      for (n = 0; n < num_regs; n++) {
        _MM_STORE_FP32(ptr_result + m*handle->n + n*SIMD_WIDTH_FP32, _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + n*SIMD_WIDTH_FP32));
      }
    }
  } else {
    for (m = 0; m < num_m; m++) {
      #pragma unroll (2)
      for (n = 0; n < num_full_regs; n++) {
        _MM_STORE_FP32(ptr_result + m*handle->n + n*SIMD_WIDTH_FP32, _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + n*SIMD_WIDTH_FP32));
      }
      for (n = last_n_start; n < num_n; n++) {
        ptr_result[m*handle->n + n] = scratch_C[m*num_regs*SIMD_WIDTH_FP32 + n];
      }
    }
  }
}

void libxsmm_spmdm_compute_bfloat16_thread( const libxsmm_spmdm_handle* handle,
                            const uint16_t *alpha, 
                            libxsmm_CSR_sparseslice* A_sparse, 
                            const uint16_t *B, 
                            const uint16_t *beta, 
                            uint16_t* C,
                            int mb, int num_m_blocks, int nb, 
                            int tid, int nthreads) 
{
  const int m_blocks = handle->mb;
  const int n_blocks = handle->nb;
  const int k_blocks = handle->kb;
  const int m_block_size = handle->bm;
  const int n_block_size = handle->bn;
  const int k_block_size = handle->bk;

#define num_regs (6) 
// really is twice this
  assert(n_block_size == num_regs*SIMD_WIDTH_FP32);
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
 
  float __attribute__((aligned (64))) scratch_C[num_m_blocks*m_block_size*n_block_size];
  float __attribute__((aligned (64))) scratch_B[k_block_size*n_block_size];
  SIMDTYPE_FP32 sum[2*num_regs];
  uint16_t* __restrict__ ptr_result;
    
  if (m_overall_end   > handle->m) m_overall_end   = handle->m;
  num_m = (m_overall_end - m_overall_start);
  num_m_aligned = (num_m / 2) * 2;

  if (n_overall_end   > handle->n) n_overall_end   = handle->n;
  num_n = (n_overall_end - n_overall_start);
  last_block_n = (num_n != n_block_size);
  SIMDTYPE_INT32 vzero = _MM_SETZERO_INT32(); 
  #if 0
  printf("Block: m_overall_start: %d, m_overall_end: %d, num_m: %d, num_m_aligned: %d\n", m_overall_start, m_overall_end, num_m, num_m_aligned); 
  printf("Block: n_overall_start: %d, n_overall_end: %d, num_n: %d, last_block_n: %d\n", n_overall_start, n_overall_end, num_n, last_block_n); 
  printf("Block: k_blocks: %d\n", k_blocks);
  #endif
  // Copy in C matrix to buffer
  ptr_result = C + m_overall_start*handle->n + n_overall_start;
  if(!last_block_n) {
    for (m = 0; m < num_m; m++) {
      #pragma unroll (num_regs/2)
      for (n = 0; n < num_regs/2; n++) {
	SIMDTYPE_INT32 vload =  _MM_LOAD_INT32((const SIMDTYPE_INT32 *)(ptr_result + m*handle->n + 2*n*SIMD_WIDTH_FP32));
        SIMDTYPE_FP32 v1, v2;
	EXPAND_BFLOAT16(vload, v1, v2);
        _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 2*n*SIMD_WIDTH_FP32, v1);
        _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + (2*n+1)*SIMD_WIDTH_FP32, v2);
      }
    }
  } else {
    for (m = 0; m < num_m; m++) {
      for (n = 0; n < num_n; n++) {
        uint16_t restmp = ptr_result[m*handle->n + n];
        int res = restmp; res <<= 16;
        float v1 = *(float *)&res;
        scratch_C[m*num_regs*SIMD_WIDTH_FP32 + n] = v1;
      }
    }
  }	

  for (kb = 0; kb < k_blocks; kb++) {
    const uint16_t* __restrict__ ptr_dense;
    float * __restrict__ scratch_C_base;
    const float * __restrict__ scratch_B_base;
    int block_A = kb * m_blocks + mb;
    libxsmm_CSR_sparseslice slice = A_sparse[block_A];
    int m_local = 0;

    k_overall_start = kb*k_block_size;
    k_overall_end   = (kb+1)*k_block_size;
    num_k = (k_overall_end - k_overall_start);
     
    // Copy in B matrix
    ptr_dense = B + k_overall_start*handle->n + n_overall_start;
    if(!last_block_n) {
      for (k = 0; k < num_k; k++) {
        #pragma unroll (num_regs/2)
        for (n = 0; n < num_regs/2; n++) {
	  SIMDTYPE_INT32 vload =  _MM_LOAD_INT32((const SIMDTYPE_INT32 *)(ptr_dense + k*handle->n + 2*n*SIMD_WIDTH_FP32));
          SIMDTYPE_FP32 v1, v2;
	  EXPAND_BFLOAT16(vload, v1, v2);
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + 2*n*SIMD_WIDTH_FP32, v1);
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + (2*n+1)*SIMD_WIDTH_FP32, v2);
        }
      }
    } else {
      for (k = 0; k < num_k; k++) {
        for (n = 0; n < num_n; n++) {
          uint16_t restmp = ptr_dense[k*handle->n + n];
          int res = restmp; res <<= 16;
          float v1 = *(float *)&res;
          scratch_B[k*num_regs*SIMD_WIDTH_FP32 + n] = v1;
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
    scratch_B_base = scratch_B; // - k_overall_start*num_regs*SIMD_WIDTH_FP32;
    
    for (m = m_overall_start; m < m_overall_start + num_m_aligned; m+=2, m_local+=2) {
      int start_j, end_j, end_j_2, num_j, num_j_2;
      const uint16_t*  __restrict__ sp_c_ptr_base; 
      const uint16_t*  __restrict__ sp_c_ptr_base_2; 
      const float* __restrict__ sp_v_ptr_base;
      const float* __restrict__ sp_v_ptr_base_2;

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
      float* const __restrict__ result_m_index = scratch_C_base + (m)*num_regs*SIMD_WIDTH_FP32;
      float* const __restrict__ result_m_index_2 = scratch_C_base + (m+1)*num_regs*SIMD_WIDTH_FP32;
      
      if(!last_block_n) 
      {
        int64_t j = 0, j2 = 0;
        #pragma unroll (num_regs)
        for (n = 0; n < num_regs; n++) {
          sum[n] = _MM_LOAD_FP32(result_m_index + n*SIMD_WIDTH_FP32);
          sum[n+num_regs] = _MM_LOAD_FP32(result_m_index_2 + n*SIMD_WIDTH_FP32);
        }
        for (; j < num_j && j2 < num_j_2; j++, j2++) {
          const float* const __restrict__ sp_col_dense_index = scratch_B_base +  sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          const float* const __restrict__ sp_col_dense_index_2 = scratch_B_base + sp_c_ptr_base_2[j2]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]); 
          SIMDTYPE_FP32 v_v_2 = _MM_SET1_FP32(sp_v_ptr_base_2[j2]); 
          #pragma unroll (num_regs)
          for (n = 0; n < num_regs; n++) {
            sum[n] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + n*SIMD_WIDTH_FP32), sum[n]);
            sum[n + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + n*SIMD_WIDTH_FP32), sum[n+num_regs]);
          }
        }
        for (; j < num_j; j++) {
          const float* const __restrict__ sp_col_dense_index = scratch_B_base +  sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]); 
          #pragma unroll (num_regs)
          for (n = 0; n < num_regs; n++) {
            sum[n] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + n*SIMD_WIDTH_FP32), sum[n]);
          }
        }
        for (; j2 < num_j_2; j2++) {
          const float* const __restrict__ sp_col_dense_index_2 = scratch_B_base + sp_c_ptr_base_2[j2]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v_2 = _MM_SET1_FP32(sp_v_ptr_base_2[j2]); 
          #pragma unroll (num_regs)
          for (n = 0; n < num_regs; n++) {
            sum[n + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + n*SIMD_WIDTH_FP32), sum[n+num_regs]);
          }
        }
        #pragma unroll (num_regs)
        for (n = 0; n < num_regs; n++) {
          _MM_STORE_FP32(result_m_index + n*SIMD_WIDTH_FP32, sum[n]);
          _MM_STORE_FP32(result_m_index_2 + n*SIMD_WIDTH_FP32, sum[n+num_regs]);
        }
      }
      else {
        int64_t j = 0, j2 = 0;
        for (; j < num_j && j2 < num_j_2; j++, j2++) {
          const float* const __restrict__ sp_col_dense_index = scratch_B_base +  sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          const float* const __restrict__ sp_col_dense_index_2 = scratch_B_base + sp_c_ptr_base_2[j2]*num_regs*SIMD_WIDTH_FP32;
          float v_v = sp_v_ptr_base[j];
          float v_v_2 = sp_v_ptr_base_2[j2];
          for (n = 0; n < num_n; n++) {
            result_m_index[n] += sp_col_dense_index[n]*v_v;
            result_m_index_2[n] += sp_col_dense_index_2[n]*v_v_2;
          } 
        }
        for (; j < num_j; j++) {
          const float* const __restrict__ sp_col_dense_index = scratch_B_base +  sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          float v_v = sp_v_ptr_base[j];
          for (n = 0; n < num_n; n++) {
            result_m_index[n] += sp_col_dense_index[n]*v_v;
          } 
        }
        for (; j2 < num_j_2; j2++) {
          const float* const __restrict__ sp_col_dense_index_2 = scratch_B_base + sp_c_ptr_base_2[j2]*num_regs*SIMD_WIDTH_FP32;
          float v_v_2 = sp_v_ptr_base_2[j2];
          for (n = 0; n < num_n; n++) {
            result_m_index_2[n] += sp_col_dense_index_2[n]*v_v_2;
          } 
        }
      }
    }
    for (m = m_overall_start + num_m_aligned; m < m_overall_end; m++, m_local++) {
      int start_j, end_j, num_j;
      const uint16_t*  __restrict__ sp_c_ptr_base; 
      const float* __restrict__ sp_v_ptr_base;
      float* __restrict__ result_m_index;

      if( m_local >= m_block_size) { block_A++; slice = A_sparse[block_A]; m_local = 0; }

      start_j =  slice.rowidx[m_local];
      end_j   =  slice.rowidx[m_local + 1];
      num_j   = (end_j - start_j);
      sp_c_ptr_base = slice.colidx + start_j;	
      sp_v_ptr_base = slice.values + start_j;
      result_m_index = scratch_C_base + (m)*num_regs*SIMD_WIDTH_FP32;
      
      if(!last_block_n) {
        int64_t j = 0;
        #pragma unroll (num_regs)
        for (n = 0; n < num_regs; n++) {
          sum[n] = _MM_LOAD_FP32(result_m_index + n*SIMD_WIDTH_FP32);
        }
        for (; j < num_j; j++) {
          const float* const __restrict__ sp_col_dense_index = scratch_B_base +  sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]); 
          #pragma unroll (num_regs)
          for (n = 0; n < num_regs; n++) {
            sum[n] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + n*SIMD_WIDTH_FP32), sum[n]);
          }
        }
        #pragma unroll (num_regs)
        for (n = 0; n < num_regs; n++) {
          _MM_STORE_FP32(result_m_index + n*SIMD_WIDTH_FP32, sum[n]);
        }
      }
      else {
        int64_t j = 0;
        for (; j < num_j; j++) {
          const float* const __restrict__ sp_col_dense_index = scratch_B_base +  sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          float v_v = sp_v_ptr_base[j];
          for (n = 0; n < num_n; n++) {
            result_m_index[n] += sp_col_dense_index[n]*v_v;
          } 
        }
      }
    }
  } // kb
  #if 0
  //for (m = 0; m < 3; m++) {
  //  for (n = 0; n < num_n; n++) {
  //    printf("%f ", scratch_C[m*num_regs*SIMD_WIDTH_FP32 + n]);
  //  }
  //  printf("\n");
  //}
  #endif
  // Copy out C matrix
  if(!last_block_n) {
    for (m = 0; m < num_m; m++) {
      #pragma unroll (num_regs/2)
      for (n = 0; n < num_regs/2; n++) {
	SIMDTYPE_FP32 vload1 =  _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 2*n*SIMD_WIDTH_FP32);
	SIMDTYPE_FP32 vload2 =  _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + (2*n+1)*SIMD_WIDTH_FP32);
        SIMDTYPE_INT32 v;
	COMPRESS_BFLOAT16(vload1, vload2, v);
        _MM_STORE_INT32((SIMDTYPE_INT32 *)(ptr_result + m*handle->n + 2*n*SIMD_WIDTH_FP32), v);
      }
    }
  } else {
    for (m = 0; m < num_m; m++) {
      for (n = 0; n < num_n; n++) {
         int v = *(int *)(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + n);
         ptr_result[m*handle->n + n] = (uint16_t)(v >> 16);
      }
    }
  }
}


void libxsmm_spmdm_compute_fp32( const libxsmm_spmdm_handle* handle,
                            const float *alpha, 
                            libxsmm_CSR_sparseslice* A_sparse, 
                            const float *B, 
                            const float *beta, 
                            float* C) {
  int m_blocks = handle->mb;
  int n_blocks = handle->nb;
  int k_blocks = handle->kb;
  int mb, nb;
  // Parallelization is over a 4X1 block grid of the output
  int num_m_blocks = 1;
  #pragma omp parallel for collapse(2)
  for (mb= 0; mb < m_blocks; mb += num_m_blocks) {
    for ( nb = 0; nb < n_blocks; nb++ ) {
      int tid = omp_get_thread_num();
      int nthreads = omp_get_num_threads();
      libxsmm_spmdm_compute_fp32_thread( handle, alpha, A_sparse, B, beta, C, mb, num_m_blocks, nb, tid, nthreads);
    }
  }
}

void libxsmm_spmdm_compute_bfloat16( const libxsmm_spmdm_handle* handle,
                            const uint16_t *alpha, 
                            libxsmm_CSR_sparseslice* A_sparse, 
                            const uint16_t *B, 
                            const uint16_t *beta, 
                            uint16_t* C) {
  int m_blocks = handle->mb;
  int n_blocks = handle->nb;
  int k_blocks = handle->kb;
  int mb, nb;

  // Parallelization is over a 4X1 block grid of the output
  int num_m_blocks = 1;
  #pragma omp parallel for collapse(2)
  for (mb= 0; mb < m_blocks; mb += num_m_blocks) {
    for ( nb = 0; nb < n_blocks; nb++ ) {
      int tid = omp_get_thread_num();
      int nthreads = omp_get_num_threads();
      libxsmm_spmdm_compute_bfloat16_thread( handle, alpha, A_sparse, B, beta, C, mb, num_m_blocks, nb, tid, nthreads);
    }
  }
}


void libxsmm_test_a ( const libxsmm_spmdm_handle* handle,
				const real * libxsmm_input_dense_a_with_zeros, /* This is of type libxsmm_spmdm_datatype */
				const int transA,
				libxsmm_CSR_sparseslice* libxsmm_output_csr_a) 
{
  real* libxsmm_test_dense_a_with_zeros;
  int m_blocks = handle->mb;
  int k_blocks = handle->kb;
  int mb, kb;
  /* Create a new dense matrix and compare element wise with libxsmm_input_dense_a_with_zeros */
  libxsmm_test_dense_a_with_zeros = (real *)libxsmm_aligned_malloc( handle->m * handle->k * sizeof(real), 2097152);
  if( transA == 'Y' ) {
    assert(0);
    for ( mb = 0; mb < m_blocks; mb++ ) {
      int block_offset_base = mb * handle->k * handle->bm;
      for ( kb = 0; kb < k_blocks; kb++ ) {
        int block_offset = block_offset_base + kb * handle->bk;
        libxsmm_CSR_sparseslice slice = libxsmm_output_csr_a[mb * k_blocks + kb];
      }
    }    
  } 
  else {
    for ( kb = 0; kb < k_blocks; kb++ ) {
      int block_offset_base = kb * handle->bk;
      for ( mb = 0; mb < m_blocks; mb++ ) {
        int block_offset = block_offset_base + mb * handle->k * handle->bm;
        libxsmm_CSR_sparseslice slice = libxsmm_output_csr_a[kb * m_blocks + mb];
        int row;	
	for( row = 0; row < handle->bm; row++ ) {
	  int i;
	  for(i = slice.rowidx[row]; i < slice.rowidx[row+1]; i++) {
            int col = slice.colidx[i];
            if ( handle->datatype == LIBXSMM_SPMDM_DATATYPE_F32 ) {
              *((float *)(libxsmm_test_dense_a_with_zeros) + block_offset + row * handle->k + col) = *((float *)(slice.values) + i);
            }
            else if ( handle->datatype == LIBXSMM_SPMDM_DATATYPE_BFLOAT16 ) {
              int val = *((int *)(slice.values) + i);
              *((uint16_t *)(libxsmm_test_dense_a_with_zeros) + block_offset + row * handle->k + col) = (uint16_t)(val >> 16);
            }
          }
        }
      }
    }
    
    double max_error = 0.0;
    double src_norm = 0.0;
    double dst_norm = 0.0;
    size_t l;
  
    for ( l = 0; l < (size_t)handle->m * (size_t)handle->k; l++ ) {
      const double dstval = (double)libxsmm_test_dense_a_with_zeros[l];
      const double srcval = (double)libxsmm_input_dense_a_with_zeros[l];
      const double local_error = fabs(dstval - srcval);
      if (local_error > max_error) {
        max_error = local_error;
      }
      src_norm += srcval;
      dst_norm += dstval;
    }

    printf("A conversion: max error: %f, sum BLAS: %f, sum LIBXSMM: %f \n", max_error, src_norm, dst_norm );
  }

}

void libxsmm_spmdm_check_c( const libxsmm_spmdm_handle* handle,
                               real* test,
                               real* gold) {
  int mb, nb, bm, bn;
  double max_error = 0.0;
  double src_norm = 0.0;
  double dst_norm = 0.0;
  size_t l;

  for ( l = 0; l < (size_t)handle->m * (size_t)handle->n; l++ ) {
    const double dstval = (double)test[l];
    const double srcval = (double)gold[l];
    const double local_error = fabs(dstval - srcval);
    //if(local_error > 0.01) printf("l: %lld, gold: %lf actual: %lf local_error: %lf\n", l, srcval, dstval, local_error);
    if (local_error > max_error) {
      max_error = local_error;
    }
    src_norm += srcval;
    dst_norm += dstval;
  }

  printf(" max error: %f, sum BLAS: %f, sum LIBXSMM: %f \n", max_error, src_norm, dst_norm );
}

void libxsmm_spmdm_exec_fp32( const libxsmm_spmdm_handle* handle,
                            const char transA,
                            const float* alpha,
                            const float* A,
                            const float* B,
                            const float* beta,
                            float* C,
                            libxsmm_CSR_sparseslice* A_sparse,
                            __m256i * shufmasks,
                            __m256i * shufmasks2 ) {

  int m_blocks = handle->mb;
  int n_blocks = handle->nb;
  int k_blocks = handle->kb;
  int mb, nb, kb;
  if( transA == 'Y') {
  }
  else {
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int nthreads = omp_get_num_threads();
      //double st = omp_get_wtime();
      #pragma omp for collapse(2)
      for ( kb = 0; kb < k_blocks; kb++ ) {
        for ( mb = 0; mb < m_blocks; mb++ ) {
          libxsmm_spmdm_createSparseSlice_fp32_notrans_thread( handle, A, transA, A_sparse, mb, kb, tid, nthreads, shufmasks, shufmasks2);
        }
      }
      //double end = omp_get_wtime();
      #if 1
      int num_m_blocks = 1;
      #pragma omp for collapse(2)
      for (mb= 0; mb < m_blocks; mb += num_m_blocks) {
        for ( nb = 0; nb < n_blocks; nb++ ) {
          libxsmm_spmdm_compute_fp32_thread( handle, alpha, A_sparse, B, beta, C, mb, num_m_blocks, nb, tid, nthreads);
        }
      }
      #endif
      //double end2 = omp_get_wtime();
      //printf("T1: %lf, T2: %lf\n", (end - st), (end2 - end));
    }
  }

  //libxsmm_spmdm_createSparseSlices( handle, A, transA, A_sparse, shufmasks, shufmasks2);
  //libxsmm_spmdm_compute( handle, alpha, A_sparse, B, beta, C); 
}

void libxsmm_spmdm_exec_bfloat16( const libxsmm_spmdm_handle* handle,
                            const char transA,
                            const uint16_t* alpha,
                            const uint16_t* A,
                            const uint16_t* B,
                            const uint16_t* beta,
                            uint16_t* C,
                            libxsmm_CSR_sparseslice* A_sparse,
                            __m256i * shufmasks,
                            __m256i * shufmasks2 ) {

  int m_blocks = handle->mb;
  int n_blocks = handle->nb;
  int k_blocks = handle->kb;
  int mb, nb, kb;
  if( transA == 'Y') {
  }
  else {
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int nthreads = omp_get_num_threads();
      //double st = omp_get_wtime();
      #pragma omp for collapse(2)
      for ( kb = 0; kb < k_blocks; kb++ ) {
        for ( mb = 0; mb < m_blocks; mb++ ) {
          libxsmm_spmdm_createSparseSlice_bfloat16_notrans_thread( handle, A, transA, A_sparse, mb, kb, tid, nthreads, shufmasks, shufmasks2);
        }
      }
      //double end = omp_get_wtime();
      #if 1
      int num_m_blocks = 1;
      #pragma omp for collapse(2)
      for (mb= 0; mb < m_blocks; mb += num_m_blocks) {
        for ( nb = 0; nb < n_blocks; nb++ ) {
          libxsmm_spmdm_compute_bfloat16_thread( handle, alpha, A_sparse, B, beta, C, mb, num_m_blocks, nb, tid, nthreads);
        }
      }
      #endif
      //double end2 = omp_get_wtime();
      //printf("T1: %lf, T2: %lf\n", (end - st), (end2 - end));
    }
  }

  //libxsmm_spmdm_createSparseSlices( handle, A, transA, A_sparse, shufmasks, shufmasks2);
  //libxsmm_spmdm_compute( handle, alpha, A_sparse, B, beta, C); 
}


int main(int argc, char **argv)
{
  real *A_gold, *B_gold, *C_gold, *C;
  libxsmm_CSR_sparseslice* A_sparse;

  int M, N, K;
  real alpha, beta;
  int reps;
  double start, end, flops; 
  char trans;
  int i, j, k;

  /* Step 1: Initalize handle */
  libxsmm_spmdm_handle handle;
  M = 0; N = 0; K = 0; alpha = (real)1.0; beta = (real)1.0;   reps = 0; trans = 'N';

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("\nUsage: ./block_gemm [M] [N] [K] [bm] [bn] [bk] [reps]\n\n");
    return 0;
  }

  /* setup defaults */
  handle.m = 2048;
  handle.n = 2048;
  handle.k = 2048;
  handle.bm = 512;
#if LIBXSMM_STATIC_TARGET_ARCH==LIBXSMM_X86_AVX512_CORE
  handle.bn = 96;
#else
  handle.bn = 48;
#endif
  handle.bk = 128;
  reps = 100;

  /* reading new values from cli */
  i = 1;
  if (argc > i) handle.m      = atoi(argv[i++]);
  if (argc > i) handle.n      = atoi(argv[i++]);
  if (argc > i) handle.k      = atoi(argv[i++]);
  if (argc > i) reps          = atoi(argv[i++]);
  M = handle.m;
  N = handle.n;
  K = handle.k;
  alpha = (real)1.0;
  beta = (real)1.0;
  #ifdef USE_BFLOAT 
  handle.datatype =  LIBXSMM_SPMDM_DATATYPE_BFLOAT16;
  #else
  handle.datatype =  LIBXSMM_SPMDM_DATATYPE_F32;
  #endif
  handle.mb = (handle.m + handle.bm - 1) / handle.bm;
  handle.nb = (handle.n + handle.bn - 1) / handle.bn;
  handle.kb = (handle.k + handle.bk - 1) / handle.bk;
 
  printf(" running with: M=%i, N=%i, K=%i, bm=%i, bn=%i, bk=%i, mb=%i, nb=%i, kb=%i, reps=%i\n", M, N, K, handle.bm, handle.bn, handle.bk, handle.mb, handle.nb, handle.kb, reps );
  srand48(1);

  /* allocate data */
  A_gold = (real*)libxsmm_aligned_malloc( M*K*sizeof(real), 2097152 );
  B_gold = (real*)libxsmm_aligned_malloc( K*N*sizeof(real), 2097152 );
  C_gold = (real*)libxsmm_aligned_malloc( M*N*sizeof(real), 2097152 );
  C      = (real*)libxsmm_aligned_malloc( M*N*sizeof(real), 2097152 );
  libxsmm_spmdm_allocate_csr_a( &handle, &A_sparse, trans);

  /* init data */
  size_t l;
  for ( l = 0; l < (size_t)M * (size_t)N; l++ ) {
    double random = drand48();
    #ifdef USE_BFLOAT 
    float  random_f = (float)random;
    int    random_int = *(int *)(&random_f);
    uint16_t val = (random_int>>16);
    #else
    float  val = (float)random;
    #endif 
    if(random > 0.85) A_gold[l] = val;
    else              A_gold[l] = (real)0.0;
  }
  for ( l = 0; l < (size_t)K * (size_t)N; l++ ) {
    double random = drand48();
    #ifdef USE_BFLOAT 
    float  random_f = (float)random;
    int    random_int = *(int *)(&random_f);
    uint16_t val = (random_int>>16);
    #else
    float  val = (float)random;
    #endif 
    B_gold[l] = val;
  }
  for ( l = 0; l < (size_t)M * (size_t)N; l++ ) {
    C_gold[l] = (real)0.0;
  }
  for ( l = 0; l < (size_t)M * (size_t)N; l++ ) {
    C[l]      = (real)0.0;
  }
  flops = (double)M * (double)N * (double)K * 2.0;
  #if 0
  printf("First row A: \n");
  for( l = 0; l < (size_t)K; l++ ) printf("%lf ", A_gold[l]);
  printf("\n"); 
  printf("First col B: \n");
  for( l = 0; l < (size_t)K * (size_t)N; l+= (size_t)N ) printf("%lf ", B_gold[l]);
  printf("\n"); 
  double sum = C_gold[0];
  for( l = 0; l < (size_t)K; l++ ) sum += A_gold[l] * B_gold[N*l];
  printf("Sum: %lf\n", sum);
  #endif
 
  /* Initialize shuffle masks for the computation */ 
  __m256i shufmasks_32[256];
  __m256i shufmasks_16[256];
  libxsmm_spmdm_init_shufmask( shufmasks_32, shufmasks_16);

  //libxsmm_spmdm_createSparseSlices( &handle, A_gold, trans, A_sparse, shufmasks_32, shufmasks_16);
  
  /* The overall function that takes in matrix inputs in dense format, does the conversion of A to sparse format and does the matrix multiply */
  /* Currently ignores alpha, beta and trans */
  /* TODO: fix alpha, beta and trans inputs */
  #ifdef USE_BFLOAT 
  libxsmm_spmdm_exec_bfloat16( &handle, trans, &alpha, A_gold, B_gold, &beta, C, A_sparse, shufmasks_32, shufmasks_16);
  #else
  libxsmm_spmdm_exec_fp32( &handle, trans, &alpha, A_gold, B_gold, &beta, C, A_sparse, shufmasks_32, shufmasks_16);
  #endif

  /* Checks */
  /* Has A been correctly converted? */
  libxsmm_test_a ( &handle, A_gold, trans, A_sparse);

  /* Compute a "gold" answer sequentially - we can also use MKL; not using MKL now due to difficulty for bfloat16 */
  #pragma omp parallel for collapse(2)
  for(i = 0; i < M; i++) {
    for(j = 0; j < N; j++) {
      float sum = 0.0;
      for(k = 0; k < K; k++) {
        #ifdef USE_BFLOAT
        uint16_t Atmp = A_gold[i*K + k];
        int Atmp_int  = Atmp; Atmp_int <<= 16;
        float Aval = *(float *)&Atmp_int;
        uint16_t Btmp = B_gold[k*N + j];
        int Btmp_int  = Btmp; Btmp_int <<= 16;
        float Bval = *(float *)&Btmp_int;
        #else
        float Aval = A_gold[i*K + k];
        float Bval = B_gold[k*N + j];
        #endif 
        sum += Aval * Bval;
      }
      #ifdef USE_BFLOAT
      int v = *(int *)(&sum);
      uint16_t Cval = (v >> 16);
      #else
      float Cval = sum;
      #endif
      C_gold[i*N + j] += Cval;
    }
  }
  //LIBXSMM_FSYMBOL(sgemm)(&trans, &trans, &N, &M, &K, &alpha, B_gold, &N, A_gold, &K, &beta, C_gold, &N);
  
  /* Compute the max difference between gold and computed results. */
  libxsmm_spmdm_check_c( &handle, C, C_gold );
 
  /* Timing loop starts */ 
  start = omp_get_wtime();  
  for( i = 0; i < reps; i++) {
    #ifdef USE_BFLOAT 
    libxsmm_spmdm_exec_bfloat16( &handle, trans, &alpha, A_gold, B_gold, &beta, C, A_sparse, shufmasks_32, shufmasks_16);
    #else
    libxsmm_spmdm_exec_fp32( &handle, trans, &alpha, A_gold, B_gold, &beta, C, A_sparse, shufmasks_32, shufmasks_16);
    #endif
  }
  end = omp_get_wtime();  
  printf("Time = %lf Time/rep = %lf, TFlops/s = %lf\n", (end - start), (end - start)*1.0/reps, flops/1000./1000./1000./1000./(end-start)*reps);
}

