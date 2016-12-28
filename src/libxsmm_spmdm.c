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
/* Nadathur Satish, Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm_spmdm.h>
#include <libxsmm_intrinsics_x86.h>
#include <libxsmm_malloc.h>
#include "libxsmm_main.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXSMM_SPMDM_MALLOC_INTRINSIC) && !defined(LIBXSMM_INTRINSICS_NONE)
# define LIBXSMM_SPMDM_MALLOC_INTRINSIC
#endif
#if defined(LIBXSMM_SPMDM_MALLOC_INTRINSIC)
# define LIBXSMM_SPMDM_MALLOC(SIZE, ALIGNMENT) _mm_malloc(SIZE, ALIGNMENT)
# define LIBXSMM_SPMDM_FREE(BUFFER) _mm_free((void*)(BUFFER))
#else
# define LIBXSMM_SPMDM_MALLOC(SIZE, ALIGNMENT) libxsmm_aligned_malloc(SIZE, -(ALIGNMENT))
# define LIBXSMM_SPMDM_FREE(BUFFER) libxsmm_free(BUFFER)
#endif


#if !defined(LIBXSMM_INTRINSICS_NONE) && (LIBXSMM_X86_AVX <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE __m256i internal_spmdm_shufmasks_32[256];
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE __m256i internal_spmdm_shufmasks_16[256];
#endif


LIBXSMM_INLINE LIBXSMM_RETARGETABLE LIBXSMM_INTRINSICS void internal_spmdm_init_shufmask_avx()
{
#if !defined(LIBXSMM_INTRINSICS_NONE) && (LIBXSMM_X86_AVX <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
  unsigned int i, j, c, last_bit;
  LIBXSMM_ALIGNED(int temp_shufmasks[8], 64);
  LIBXSMM_ALIGNED(uint16_t temp_shufmasks2[16], 64);
  int cnt;
  for (i = 0; i < 256; i++) {
    cnt = 0;
    j = i;
    for (c = 0; c < 8; c++) temp_shufmasks[c] = 0;
    for (c = 0; c < 16; c++) temp_shufmasks2[c] = 0;
    while ( j) {
      last_bit = LIBXSMM_INTRINSICS_BITSCANFWD(j);
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


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_spmdm_allocate_csr_a(libxsmm_spmdm_handle* handle, libxsmm_CSR_sparseslice** libxsmm_output_csr)
{
  int kb, mb;
  int m_blocks = handle->mb;
  int k_blocks = handle->kb;

  size_t sz_block = ((handle->bm + 1)*sizeof(uint16_t) + (handle->bm)*(handle->bk)*sizeof(uint16_t) + (handle->bm)*(handle->bk)*sizeof(float) + sizeof(libxsmm_CSR_sparseslice));
  size_t sz_all_blocks = sz_block * handle->mb * handle->kb;

  char * memory_block = (char *)LIBXSMM_SPMDM_MALLOC( sz_all_blocks, 2097152);
  char * memory_head  = memory_block;

  libxsmm_CSR_sparseslice* libxsmm_output_csr_a = (libxsmm_CSR_sparseslice*)(memory_head);
  memory_head += handle->mb * handle->kb * sizeof(libxsmm_CSR_sparseslice);

  for (kb = 0; kb < k_blocks; kb++) {
    for (mb = 0; mb < m_blocks; mb++) {
      int i = kb*m_blocks + mb;
      libxsmm_output_csr_a[i].rowidx = (uint16_t *)(memory_head);
      memory_head += (handle->bm + 1)*sizeof(uint16_t);
      libxsmm_output_csr_a[i].colidx = (uint16_t *)(memory_head);
      memory_head += (handle->bm)*(handle->bk)*sizeof(uint16_t);
      libxsmm_output_csr_a[i].values = (float*)(memory_head);
      memory_head += (handle->bm)*(handle->bk)*sizeof(float);
    }
  }
  assert(memory_head == (memory_block + sz_all_blocks));
  *libxsmm_output_csr = libxsmm_output_csr_a;
  handle->base_ptr_scratch_A = memory_block;
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_spmdm_allocate_scratch(libxsmm_spmdm_handle* handle, int max_threads)
{
  size_t sz_memory_for_scratch_per_thread = ((handle->bm)*(handle->bn)*sizeof(float) + (handle->bk)*(handle->bn)*sizeof(float))*max_threads, sz_total_memory;
  sz_memory_for_scratch_per_thread = (sz_memory_for_scratch_per_thread + 4095)/4096 * 4096;
  sz_total_memory = sz_memory_for_scratch_per_thread * max_threads;

  handle->base_ptr_scratch_B_scratch_C = (char *)LIBXSMM_SPMDM_MALLOC( sz_total_memory, 2097152 );
  handle->memory_for_scratch_per_thread = (int)sz_memory_for_scratch_per_thread;
}


LIBXSMM_API_DEFINITION void libxsmm_spmdm_init(int M, int N, int K, int max_threads, libxsmm_spmdm_handle* handle, libxsmm_CSR_sparseslice** libxsmm_output_csr)
{
  handle->m  = M;
  handle->n  = N;
  handle->k  = K;

  handle->bm = (M >= 4096 || M <= 1024) ? 512 : 256;
  if (LIBXSMM_X86_AVX512_CORE <= libxsmm_target_archid) {
    handle->bn = 96;
  }
  else if (LIBXSMM_X86_AVX2 <= libxsmm_target_archid) {
    handle->bn = 48;
  }
  else {
    handle->bn = 6;
  }
  handle->bk = 128;
  handle->mb = (handle->m + handle->bm - 1) / handle->bm;
  handle->nb = (handle->n + handle->bn - 1) / handle->bn;
  handle->kb = (handle->k + handle->bk - 1) / handle->bk;

  /* This is temporary space needed; allocate for each different size of A */
  internal_spmdm_allocate_csr_a( handle, libxsmm_output_csr);
  internal_spmdm_allocate_scratch( handle, max_threads);

  /* Initialize shuffle masks for the computation */
  if (LIBXSMM_X86_AVX <= libxsmm_target_archid) {
    internal_spmdm_init_shufmask_avx();
  }
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_spmdm_deallocate_csr_a(libxsmm_spmdm_handle* handle)
{
  LIBXSMM_SPMDM_FREE(handle->base_ptr_scratch_A);
  handle->base_ptr_scratch_A= NULL;
  LIBXSMM_SPMDM_FREE(handle->base_ptr_scratch_B_scratch_C);
  handle->base_ptr_scratch_B_scratch_C = NULL;
}


LIBXSMM_API_DEFINITION void libxsmm_spmdm_destroy(libxsmm_spmdm_handle* handle)
{
  internal_spmdm_deallocate_csr_a(handle);
}


LIBXSMM_API_DEFINITION int libxsmm_spmdm_get_num_createSparseSlice_blocks(const libxsmm_spmdm_handle* handle)
{
  return handle->mb * handle->kb;
}


LIBXSMM_API_DEFINITION int libxsmm_spmdm_get_num_compute_blocks(const libxsmm_spmdm_handle* handle)
{
  return handle->mb * handle->nb;
}


LIBXSMM_API_DEFINITION LIBXSMM_INTRINSICS void libxsmm_spmdm_createSparseSlice_fp32_thread(
  const libxsmm_spmdm_handle* handle,
  char transA,
  const float* A,
  libxsmm_CSR_sparseslice* libxsmm_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
#if !defined(LIBXSMM_INTRINSICS_NONE)
# if (LIBXSMM_X86_AVX512_CORE <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
#   if (LIBXSMM_X86_AVX512_CORE > LIBXSMM_STATIC_TARGET_ARCH)
  if (LIBXSMM_X86_AVX512_CORE <= libxsmm_target_archid)
#   endif
  {
#   include "libxsmm_spmdm_begin_avx512.h"
#   include "template/libxsmm_spmdm_createSparseSlice_fp32_thread.tpl.c"
#   include "libxsmm_spmdm_end.h"
  }
#   if (LIBXSMM_X86_AVX512_CORE > LIBXSMM_STATIC_TARGET_ARCH)
  else
#   endif
# endif
# if (LIBXSMM_X86_AVX2 <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
#   if (LIBXSMM_X86_AVX2 > LIBXSMM_STATIC_TARGET_ARCH)
  if (LIBXSMM_X86_AVX2 <= libxsmm_target_archid)
#   endif
  {
#   include "libxsmm_spmdm_begin_avx2.h"
#   include "template/libxsmm_spmdm_createSparseSlice_fp32_thread.tpl.c"
#   include "libxsmm_spmdm_end.h"
  }
#   if (LIBXSMM_X86_AVX2 > LIBXSMM_STATIC_TARGET_ARCH)
  else
#   endif
# endif
#endif /*!defined(LIBXSMM_INTRINSICS_NONE)*/
#if defined(LIBXSMM_INTRINSICS_NONE) || (LIBXSMM_X86_AVX2 > LIBXSMM_STATIC_TARGET_ARCH)
  {
# include "libxsmm_spmdm_begin.h"
# include "template/libxsmm_spmdm_createSparseSlice_fp32_thread.tpl.c"
# include "libxsmm_spmdm_end.h"
  }
#endif
}


LIBXSMM_API_DEFINITION LIBXSMM_INTRINSICS void libxsmm_spmdm_createSparseSlice_bfloat16_thread(
  const libxsmm_spmdm_handle* handle,
  char transA,
  const uint16_t* A,
  libxsmm_CSR_sparseslice* libxsmm_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
#if !defined(LIBXSMM_INTRINSICS_NONE)
# if (LIBXSMM_X86_AVX512_CORE <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
#   if (LIBXSMM_X86_AVX512_CORE > LIBXSMM_STATIC_TARGET_ARCH)
  if (LIBXSMM_X86_AVX512_CORE <= libxsmm_target_archid)
#   endif
  {
#   include "libxsmm_spmdm_begin_avx512.h"
#   include "template/libxsmm_spmdm_createSparseSlice_bfloat16_thread.tpl.c"
#   include "libxsmm_spmdm_end.h"
  }
#   if (LIBXSMM_X86_AVX512_CORE > LIBXSMM_STATIC_TARGET_ARCH)
  else
#   endif
# endif
# if (LIBXSMM_X86_AVX2 <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
#   if (LIBXSMM_X86_AVX2 > LIBXSMM_STATIC_TARGET_ARCH)
  if (LIBXSMM_X86_AVX2 <= libxsmm_target_archid)
#   endif
  {
#   include "libxsmm_spmdm_begin_avx2.h"
#   include "template/libxsmm_spmdm_createSparseSlice_bfloat16_thread.tpl.c"
#   include "libxsmm_spmdm_end.h"
  }
#   if (LIBXSMM_X86_AVX2 > LIBXSMM_STATIC_TARGET_ARCH)
  else
#   endif
# endif
#endif /*!defined(LIBXSMM_INTRINSICS_NONE)*/
#if defined(LIBXSMM_INTRINSICS_NONE) || (LIBXSMM_X86_AVX2 > LIBXSMM_STATIC_TARGET_ARCH)
  {
# include "libxsmm_spmdm_begin.h"
# include "template/libxsmm_spmdm_createSparseSlice_bfloat16_thread.tpl.c"
# include "libxsmm_spmdm_end.h"
  }
#endif
}


LIBXSMM_API_DEFINITION LIBXSMM_INTRINSICS void libxsmm_spmdm_compute_fp32_thread(
  const libxsmm_spmdm_handle* handle,
  char transA,
  char transB,
  const float* alpha,
  libxsmm_CSR_sparseslice* A_sparse,
  const float* B,
  char transC,
  const float* beta,
  float* C,
  int block_id,
  int tid, int nthreads)
{
#if !defined(LIBXSMM_INTRINSICS_NONE)
# if (LIBXSMM_X86_AVX512_CORE <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
#   if (LIBXSMM_X86_AVX512_CORE > LIBXSMM_STATIC_TARGET_ARCH)
  if (LIBXSMM_X86_AVX512_CORE <= libxsmm_target_archid)
#   endif
  {
#   include "libxsmm_spmdm_begin_avx512.h"
#   include "template/libxsmm_spmdm_compute_fp32_thread.tpl.c"
#   include "libxsmm_spmdm_end.h"
  }
#   if (LIBXSMM_X86_AVX512_CORE > LIBXSMM_STATIC_TARGET_ARCH)
  else
#   endif
# endif
# if (LIBXSMM_X86_AVX2 <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
#   if (LIBXSMM_X86_AVX2 > LIBXSMM_STATIC_TARGET_ARCH)
  if (LIBXSMM_X86_AVX2 <= libxsmm_target_archid)
#   endif
  {
#   include "libxsmm_spmdm_begin_avx2.h"
#   include "template/libxsmm_spmdm_compute_fp32_thread.tpl.c"
#   include "libxsmm_spmdm_end.h"
  }
#   if (LIBXSMM_X86_AVX2 > LIBXSMM_STATIC_TARGET_ARCH)
  else
#   endif
# endif
#endif /*!defined(LIBXSMM_INTRINSICS_NONE)*/
#if defined(LIBXSMM_INTRINSICS_NONE) || (LIBXSMM_X86_AVX2 > LIBXSMM_STATIC_TARGET_ARCH)
  {
# include "libxsmm_spmdm_begin.h"
# include "template/libxsmm_spmdm_compute_fp32_thread.tpl.c"
# include "libxsmm_spmdm_end.h"
  }
#endif
}


LIBXSMM_API_DEFINITION LIBXSMM_INTRINSICS void libxsmm_spmdm_compute_bfloat16_thread(
  const libxsmm_spmdm_handle* handle,
  char transA,
  char transB,
  const uint16_t* alpha,
  libxsmm_CSR_sparseslice* A_sparse,
  const uint16_t* B,
  char transC,
  const uint16_t* beta,
  float* C,
  int block_id,
  int tid, int nthreads)
{
#if !defined(LIBXSMM_INTRINSICS_NONE)
# if (LIBXSMM_X86_AVX512_CORE <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
#   if (LIBXSMM_X86_AVX512_CORE > LIBXSMM_STATIC_TARGET_ARCH)
  if (LIBXSMM_X86_AVX512_CORE <= libxsmm_target_archid)
#   endif
  {
#   include "libxsmm_spmdm_begin_avx512.h"
#   include "template/libxsmm_spmdm_compute_bfloat16_thread.tpl.c"
#   include "libxsmm_spmdm_end.h"
  }
#   if (LIBXSMM_X86_AVX512_CORE > LIBXSMM_STATIC_TARGET_ARCH)
  else
#   endif
# endif
# if (LIBXSMM_X86_AVX2 <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
#   if (LIBXSMM_X86_AVX2 > LIBXSMM_STATIC_TARGET_ARCH)
  if (LIBXSMM_X86_AVX2 <= libxsmm_target_archid)
#   endif
  {
#   include "libxsmm_spmdm_begin_avx2.h"
#   include "template/libxsmm_spmdm_compute_bfloat16_thread.tpl.c"
#   include "libxsmm_spmdm_end.h"
  }
#   if (LIBXSMM_X86_AVX2 > LIBXSMM_STATIC_TARGET_ARCH)
  else
#   endif
# endif
#endif /*!defined(LIBXSMM_INTRINSICS_NONE)*/
#if defined(LIBXSMM_INTRINSICS_NONE) || (LIBXSMM_X86_AVX2 > LIBXSMM_STATIC_TARGET_ARCH)
  {
# include "libxsmm_spmdm_begin.h"
# include "template/libxsmm_spmdm_compute_bfloat16_thread.tpl.c"
# include "libxsmm_spmdm_end.h"
  }
#endif
}

