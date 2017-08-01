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
#include <libxsmm.h>
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

/* Enable/disable specific code paths */
#if defined(LIBXSMM_INTRINSICS_AVX) && !defined(LIBXSMM_SPMDM_AVX)
# define LIBXSMM_SPMDM_AVX
#endif
#if defined(LIBXSMM_INTRINSICS_AVX2) && !defined(LIBXSMM_SPMDM_AVX2)
# define LIBXSMM_SPMDM_AVX2
#endif
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE) && !defined(LIBXSMM_SPMDM_AVX512_CORE)
# define LIBXSMM_SPMDM_AVX512_CORE
#endif


/* function pointer for the CPUID-dispatched implementation */
LIBXSMM_API_VARIABLE void (*internal_spmdm_createSparseSlice_fp32_thread)(const libxsmm_spmdm_handle*, char,
  const float*, libxsmm_CSR_sparseslice*, int, int, int);
LIBXSMM_API_VARIABLE void (*internal_spmdm_createSparseSlice_bfloat16_thread)(const libxsmm_spmdm_handle*, char,
  const uint16_t*, libxsmm_CSR_sparseslice*, int, int, int);
LIBXSMM_API_VARIABLE void (*internal_spmdm_compute_fp32_thread)(const libxsmm_spmdm_handle*, char, char,
  const float*, libxsmm_CSR_sparseslice*, const float*, char, const float*, float*, int, int, int);
LIBXSMM_API_VARIABLE void (*internal_spmdm_compute_bfloat16_thread)(const libxsmm_spmdm_handle*, char, char,
  const uint16_t*, libxsmm_CSR_sparseslice*, const uint16_t*, char, const uint16_t*, float*, int, int, int);

#if defined(LIBXSMM_SPMDM_AVX)
LIBXSMM_API_VARIABLE __m256i* internal_spmdm_shufmasks_32;
LIBXSMM_API_VARIABLE __m256i* internal_spmdm_shufmasks_16;
#endif


LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX)
LIBXSMM_ATTRIBUTE_UNUSED void internal_spmdm_init_shufmask_avx(void)
{
#if defined(LIBXSMM_SPMDM_AVX)
  static __m256i spmdm_shufmasks_32[256], spmdm_shufmasks_16[256];
  LIBXSMM_ALIGNED(int temp_shufmasks[8], 64);
  LIBXSMM_ALIGNED(uint16_t temp_shufmasks2[16], 64);
  unsigned int i, j, c, last_bit;
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
    spmdm_shufmasks_32[i] = _mm256_loadu_si256((const __m256i*)temp_shufmasks);
    spmdm_shufmasks_16[i] = _mm256_loadu_si256((const __m256i*)temp_shufmasks2);
  }
  internal_spmdm_shufmasks_32 = spmdm_shufmasks_32;
  internal_spmdm_shufmasks_16 = spmdm_shufmasks_16;
#endif
}


LIBXSMM_API_INLINE void internal_spmdm_allocate_csr_a(libxsmm_spmdm_handle* handle, libxsmm_CSR_sparseslice** libxsmm_output_csr)
{
  int kb, mb;
  int m_blocks = handle->mb;
  int k_blocks = handle->kb;

  size_t sz_block = ((handle->bm + 1)*sizeof(uint16_t) + (handle->bm)*(handle->bk)*sizeof(uint16_t) + (handle->bm)*(handle->bk)*sizeof(float) + sizeof(libxsmm_CSR_sparseslice));
  size_t sz_all_blocks = sz_block * handle->mb * handle->kb;
  char* memory_block = 0;

  /* use low-level scratch memory allocation since life-time of this buffer is unknown */
  if (EXIT_SUCCESS == libxsmm_xmalloc((void**)&memory_block, sz_all_blocks, 2097152,
    LIBXSMM_MALLOC_FLAG_SCRATCH, 0/*extra*/, 0/*extra_size*/))
  {
    char* memory_head  = memory_block;
    libxsmm_CSR_sparseslice* libxsmm_output_csr_a = (libxsmm_CSR_sparseslice*)(memory_head);
    memory_head += handle->mb * handle->kb * sizeof(libxsmm_CSR_sparseslice);
    assert(0 != libxsmm_output_csr_a/*sanity check*/);

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
  }
  else if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
    fprintf(stderr, "LIBXSMM ERROR: SPMDM CSR scratch memory allocation failed!\n");
  }

  handle->base_ptr_scratch_A = memory_block;
}


LIBXSMM_API_INLINE void internal_spmdm_allocate_scratch(libxsmm_spmdm_handle* handle, int max_threads)
{
  size_t sz_memory_for_scratch_per_thread = ((handle->bm)*(handle->bn)*sizeof(float) + (handle->bk)*(handle->bn)*sizeof(float)), sz_total_memory;
  sz_memory_for_scratch_per_thread = LIBXSMM_UP2(sz_memory_for_scratch_per_thread, 4096);
  sz_total_memory = sz_memory_for_scratch_per_thread * max_threads;
  handle->base_ptr_scratch_B_scratch_C = 0;

  /* use low-level scratch memory allocation since life-time of this buffer is unknown */
  if (EXIT_SUCCESS == libxsmm_xmalloc((void**)&handle->base_ptr_scratch_B_scratch_C, sz_total_memory, 2097152,
    LIBXSMM_MALLOC_FLAG_SCRATCH, 0/*extra*/, 0/*extra_size*/))
  {
    handle->memory_for_scratch_per_thread = (int)sz_memory_for_scratch_per_thread;
  }
  else {
    if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
      fprintf(stderr, "LIBXSMM ERROR: SPMDM scratch memory allocation failed!\n");
    }
    handle->memory_for_scratch_per_thread = 0;
  }
}


LIBXSMM_API_INLINE void internal_spmdm_deallocate_csr_a(libxsmm_spmdm_handle* handle)
{
  libxsmm_xfree(handle->base_ptr_scratch_A);
  handle->base_ptr_scratch_A= NULL;
  libxsmm_xfree(handle->base_ptr_scratch_B_scratch_C);
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


LIBXSMM_API_INLINE
void internal_spmdm_createSparseSlice_fp32_thread_sw(
  const libxsmm_spmdm_handle* handle,
  char transA,
  const float* A,
  libxsmm_CSR_sparseslice* libxsmm_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
# include "libxsmm_spmdm_begin.h"
# include "template/libxsmm_spmdm_createSparseSlice_fp32_thread.tpl.c"
# include "libxsmm_spmdm_end.h"
}


LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX2)
LIBXSMM_ATTRIBUTE_UNUSED void internal_spmdm_createSparseSlice_fp32_thread_avx2(
  const libxsmm_spmdm_handle* handle,
  char transA,
  const float* A,
  libxsmm_CSR_sparseslice* libxsmm_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
#if defined(LIBXSMM_SPMDM_AVX2)
# include "libxsmm_spmdm_begin_avx2.h"
# include "template/libxsmm_spmdm_createSparseSlice_fp32_thread.tpl.c"
# include "libxsmm_spmdm_end.h"
#else
  internal_spmdm_createSparseSlice_fp32_thread_sw(handle, transA, A, libxsmm_output_csr_a, block_id, tid, nthreads);
#endif
}


LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
LIBXSMM_ATTRIBUTE_UNUSED void internal_spmdm_createSparseSlice_fp32_thread_avx512_core(
  const libxsmm_spmdm_handle* handle,
  char transA,
  const float* A,
  libxsmm_CSR_sparseslice* libxsmm_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
#if defined(LIBXSMM_SPMDM_AVX512_CORE)
# include "libxsmm_spmdm_begin_avx512.h"
# include "template/libxsmm_spmdm_createSparseSlice_fp32_thread.tpl.c"
# include "libxsmm_spmdm_end.h"
#else
  internal_spmdm_createSparseSlice_fp32_thread_avx2(handle, transA, A, libxsmm_output_csr_a, block_id, tid, nthreads);
#endif
}


LIBXSMM_API_DEFINITION
void libxsmm_spmdm_createSparseSlice_fp32_thread(
  const libxsmm_spmdm_handle* handle,
  char transA,
  const float* A,
  libxsmm_CSR_sparseslice* libxsmm_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
  /* if highest implemented code path is statically present, no need for an indirect call (function pointer) */
#if (LIBXSMM_X86_AVX512_CORE <= LIBXSMM_STATIC_TARGET_ARCH)
  internal_spmdm_createSparseSlice_fp32_thread_avx512_core(handle, transA, A, libxsmm_output_csr_a, block_id, tid, nthreads);
#elif (LIBXSMM_STATIC_TARGET_ARCH == LIBXSMM_MAX_STATIC_TARGET_ARCH) /* eventually no need for an indirect call */
# if (LIBXSMM_X86_AVX2 <= LIBXSMM_STATIC_TARGET_ARCH)
  internal_spmdm_createSparseSlice_fp32_thread_avx2(handle, transA, A, libxsmm_output_csr_a, block_id, tid, nthreads);
# else /* pointer based function call */
  assert(0 != internal_spmdm_createSparseSlice_fp32_thread);
  internal_spmdm_createSparseSlice_fp32_thread(handle, transA, A, libxsmm_output_csr_a, block_id, tid, nthreads);
# endif
#else /* pointer based function call */
  assert(0 != internal_spmdm_createSparseSlice_fp32_thread);
  internal_spmdm_createSparseSlice_fp32_thread(handle, transA, A, libxsmm_output_csr_a, block_id, tid, nthreads);
#endif
}


LIBXSMM_API_INLINE
void internal_spmdm_createSparseSlice_bfloat16_thread_sw(
  const libxsmm_spmdm_handle* handle,
  char transA,
  const uint16_t* A,
  libxsmm_CSR_sparseslice* libxsmm_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
# include "libxsmm_spmdm_begin.h"
# include "template/libxsmm_spmdm_createSparseSlice_bfloat16_thread.tpl.c"
# include "libxsmm_spmdm_end.h"
}


LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX2)
LIBXSMM_ATTRIBUTE_UNUSED void internal_spmdm_createSparseSlice_bfloat16_thread_avx2(
  const libxsmm_spmdm_handle* handle,
  char transA,
  const uint16_t* A,
  libxsmm_CSR_sparseslice* libxsmm_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
#if defined(LIBXSMM_SPMDM_AVX2)
# include "libxsmm_spmdm_begin_avx2.h"
# include "template/libxsmm_spmdm_createSparseSlice_bfloat16_thread.tpl.c"
# include "libxsmm_spmdm_end.h"
#else
  internal_spmdm_createSparseSlice_bfloat16_thread_sw(handle, transA, A, libxsmm_output_csr_a, block_id, tid, nthreads);
#endif
}


LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
LIBXSMM_ATTRIBUTE_UNUSED void internal_spmdm_createSparseSlice_bfloat16_thread_avx512_core(
  const libxsmm_spmdm_handle* handle,
  char transA,
  const uint16_t* A,
  libxsmm_CSR_sparseslice* libxsmm_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
#if defined(LIBXSMM_SPMDM_AVX512_CORE)
# include "libxsmm_spmdm_begin_avx512.h"
# include "template/libxsmm_spmdm_createSparseSlice_bfloat16_thread.tpl.c"
# include "libxsmm_spmdm_end.h"
#else
  internal_spmdm_createSparseSlice_bfloat16_thread_avx2(handle, transA, A, libxsmm_output_csr_a, block_id, tid, nthreads);
#endif
}


LIBXSMM_API_DEFINITION
void libxsmm_spmdm_createSparseSlice_bfloat16_thread(
  const libxsmm_spmdm_handle* handle,
  char transA,
  const uint16_t* A,
  libxsmm_CSR_sparseslice* libxsmm_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
  /* if highest implemented code path is statically present, no need for an indirect call (function pointer) */
#if (LIBXSMM_X86_AVX512_CORE <= LIBXSMM_STATIC_TARGET_ARCH)
  internal_spmdm_createSparseSlice_bfloat16_thread_avx512_core(handle, transA, A, libxsmm_output_csr_a, block_id, tid, nthreads);
#elif (LIBXSMM_STATIC_TARGET_ARCH == LIBXSMM_MAX_STATIC_TARGET_ARCH) /* eventually no need for an indirect call */
# if (LIBXSMM_X86_AVX2 <= LIBXSMM_STATIC_TARGET_ARCH)
  internal_spmdm_createSparseSlice_bfloat16_thread_avx2(handle, transA, A, libxsmm_output_csr_a, block_id, tid, nthreads);
# else /* pointer based function call */
  assert(0 != internal_spmdm_createSparseSlice_fp32_thread);
  internal_spmdm_createSparseSlice_bfloat16_thread(handle, transA, A, libxsmm_output_csr_a, block_id, tid, nthreads);
# endif
#else /* pointer based function call */
  assert(0 != internal_spmdm_createSparseSlice_fp32_thread);
  internal_spmdm_createSparseSlice_bfloat16_thread(handle, transA, A, libxsmm_output_csr_a, block_id, tid, nthreads);
#endif
}


LIBXSMM_API_INLINE
void internal_spmdm_compute_fp32_thread_sw(
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
# include "libxsmm_spmdm_begin.h"
# include "template/libxsmm_spmdm_compute_fp32_thread.tpl.c"
# include "libxsmm_spmdm_end.h"
}


LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX2)
LIBXSMM_ATTRIBUTE_UNUSED void internal_spmdm_compute_fp32_thread_avx2(
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
#if defined(LIBXSMM_SPMDM_AVX2)
# include "libxsmm_spmdm_begin_avx2.h"
# include "template/libxsmm_spmdm_compute_fp32_thread.tpl.c"
# include "libxsmm_spmdm_end.h"
#else
  internal_spmdm_compute_fp32_thread_sw(handle, transA, transB, alpha, A_sparse, B, transC, beta, C, block_id, tid, nthreads);
#endif
}


LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
LIBXSMM_ATTRIBUTE_UNUSED void internal_spmdm_compute_fp32_thread_avx512_core(
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
#if defined(LIBXSMM_SPMDM_AVX512_CORE)
# include "libxsmm_spmdm_begin_avx512.h"
# include "template/libxsmm_spmdm_compute_fp32_thread.tpl.c"
# include "libxsmm_spmdm_end.h"
#else
  internal_spmdm_compute_fp32_thread_avx2(handle, transA, transB, alpha, A_sparse, B, transC, beta, C, block_id, tid, nthreads);
#endif
}


LIBXSMM_API_DEFINITION
void libxsmm_spmdm_compute_fp32_thread(
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
  /* if highest implemented code path is statically present, no need for an indirect call (function pointer) */
#if (LIBXSMM_X86_AVX512_CORE <= LIBXSMM_STATIC_TARGET_ARCH)
  internal_spmdm_compute_fp32_thread_avx512_core(handle, transA, transB, alpha, A_sparse, B, transC, beta, C, block_id, tid, nthreads);
#else /* pointer based function call */
  assert(0 != internal_spmdm_compute_fp32_thread);
  internal_spmdm_compute_fp32_thread(handle, transA, transB, alpha, A_sparse, B, transC, beta, C, block_id, tid, nthreads);
#endif
}


LIBXSMM_API_INLINE
void internal_spmdm_compute_bfloat16_thread_sw(
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
# include "libxsmm_spmdm_begin.h"
# include "template/libxsmm_spmdm_compute_bfloat16_thread.tpl.c"
# include "libxsmm_spmdm_end.h"
}


LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX2)
LIBXSMM_ATTRIBUTE_UNUSED void internal_spmdm_compute_bfloat16_thread_avx2(
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
#if defined(LIBXSMM_SPMDM_AVX2)
# include "libxsmm_spmdm_begin_avx2.h"
# include "template/libxsmm_spmdm_compute_bfloat16_thread.tpl.c"
# include "libxsmm_spmdm_end.h"
#else
  internal_spmdm_compute_bfloat16_thread_sw(handle, transA, transB, alpha, A_sparse, B, transC, beta, C, block_id, tid, nthreads);
#endif
}


LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
LIBXSMM_ATTRIBUTE_UNUSED void internal_spmdm_compute_bfloat16_thread_avx512_core(
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
#if defined(LIBXSMM_SPMDM_AVX512_CORE)
# include "libxsmm_spmdm_begin_avx512.h"
# include "template/libxsmm_spmdm_compute_bfloat16_thread.tpl.c"
# include "libxsmm_spmdm_end.h"
#else
  internal_spmdm_compute_bfloat16_thread_avx2(handle, transA, transB, alpha, A_sparse, B, transC, beta, C, block_id, tid, nthreads);
#endif
}


LIBXSMM_API_DEFINITION
void libxsmm_spmdm_compute_bfloat16_thread(
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
  /* if highest implemented code path is statically present, no need for an indirect call (function pointer) */
#if (LIBXSMM_X86_AVX512_CORE <= LIBXSMM_STATIC_TARGET_ARCH)
  internal_spmdm_compute_bfloat16_thread_avx512_core(handle, transA, transB, alpha, A_sparse, B, transC, beta, C, block_id, tid, nthreads);
#elif (LIBXSMM_STATIC_TARGET_ARCH == LIBXSMM_MAX_STATIC_TARGET_ARCH) /* eventually no need for an indirect call */
# if (LIBXSMM_X86_AVX2 <= LIBXSMM_STATIC_TARGET_ARCH)
  internal_spmdm_compute_bfloat16_thread_avx2(handle, transA, transB, alpha, A_sparse, B, transC, beta, C, block_id, tid, nthreads);
# else /* pointer based function call */
  assert(0 != internal_spmdm_compute_bfloat16_thread);
  internal_spmdm_compute_bfloat16_thread(handle, transA, transB, alpha, A_sparse, B, transC, beta, C, block_id, tid, nthreads);
# endif
#else /* pointer based function call */
  assert(0 != internal_spmdm_compute_bfloat16_thread);
  internal_spmdm_compute_bfloat16_thread(handle, transA, transB, alpha, A_sparse, B, transC, beta, C, block_id, tid, nthreads);
#endif
}


LIBXSMM_API_INLINE void internal_spmdm_init_check(int archid)
{
  if (archid < libxsmm_target_archid
    && 0 != libxsmm_verbosity) /* library code is expected to be mute */
  {
    static int error_once = 0;
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM ERROR: missed to enter \"%s\" code path due to the compiler used!\n", libxsmm_get_target_arch());
    }
  }
}


LIBXSMM_API_DEFINITION void libxsmm_spmdm_init(int M, int N, int K, int max_threads,
  libxsmm_spmdm_handle* handle, libxsmm_CSR_sparseslice** libxsmm_output_csr)
{
  double load_imbalance_tolerate = 1.1;
  int max_work_per_block;
  double avg_work_per_block;
  int max_blocks_per_thread;
  double avg_blocks_per_thread;
  double load_imbalance_1, load_imbalance_2, load_imbalance;

  /* initialize internal library structures */
  LIBXSMM_INIT

  handle->m  = M;
  handle->n  = N;
  handle->k  = K;
  handle->bm = (M >= 4096 || M <= 1024) ? 512 : 256;

#if defined(LIBXSMM_SPMDM_AVX512_CORE)
  if (LIBXSMM_X86_AVX512_CORE <= libxsmm_target_archid || LIBXSMM_X86_AVX512_CORE <= LIBXSMM_STATIC_TARGET_ARCH) {
    internal_spmdm_init_check(LIBXSMM_X86_AVX512_CORE);
    internal_spmdm_createSparseSlice_fp32_thread = internal_spmdm_createSparseSlice_fp32_thread_avx512_core;
    internal_spmdm_createSparseSlice_bfloat16_thread = internal_spmdm_createSparseSlice_bfloat16_thread_avx512_core;
    internal_spmdm_compute_fp32_thread = internal_spmdm_compute_fp32_thread_avx512_core;
    internal_spmdm_compute_bfloat16_thread = internal_spmdm_compute_bfloat16_thread_avx512_core;
    handle->bn = 96;
  }
  else
#endif
#if defined(LIBXSMM_SPMDM_AVX2)
  if (LIBXSMM_X86_AVX2 <= libxsmm_target_archid || LIBXSMM_X86_AVX2 <= LIBXSMM_STATIC_TARGET_ARCH) {
    internal_spmdm_init_check(LIBXSMM_X86_AVX512_MIC);
    internal_spmdm_createSparseSlice_fp32_thread = internal_spmdm_createSparseSlice_fp32_thread_avx2;
    internal_spmdm_createSparseSlice_bfloat16_thread = internal_spmdm_createSparseSlice_bfloat16_thread_avx2;
    internal_spmdm_compute_fp32_thread = internal_spmdm_compute_fp32_thread_avx2;
    internal_spmdm_compute_bfloat16_thread = internal_spmdm_compute_bfloat16_thread_avx2;
    handle->bn = 48;
  }
  else
#endif
  {
    internal_spmdm_init_check(LIBXSMM_X86_AVX);
    internal_spmdm_createSparseSlice_fp32_thread = internal_spmdm_createSparseSlice_fp32_thread_sw;
    internal_spmdm_createSparseSlice_bfloat16_thread = internal_spmdm_createSparseSlice_bfloat16_thread_sw;
    internal_spmdm_compute_fp32_thread = internal_spmdm_compute_fp32_thread_sw;
    internal_spmdm_compute_bfloat16_thread = internal_spmdm_compute_bfloat16_thread_sw;
    handle->bn = 6;
  }
  handle->bk = 128;
  handle->mb = (handle->m + handle->bm - 1) / handle->bm;
  handle->nb = (handle->n + handle->bn - 1) / handle->bn;
  handle->kb = (handle->k + handle->bk - 1) / handle->bk;

  max_work_per_block    = handle->bm * handle->bn;
  avg_work_per_block    = (double)(handle->m * handle->n) / (handle->mb * handle->nb);
  load_imbalance_1      = max_work_per_block / avg_work_per_block;
  max_blocks_per_thread = (handle->mb * handle->nb + max_threads - 1) / max_threads;
  avg_blocks_per_thread = (double)handle->mb * handle->nb / max_threads;
  load_imbalance_2      = max_blocks_per_thread / avg_blocks_per_thread;
  load_imbalance        = load_imbalance_1 * load_imbalance_2;

  while (32 < handle->bm && load_imbalance > load_imbalance_tolerate) {
    handle->bm--;
    handle->mb = (handle->m + handle->bm - 1) / handle->bm;

    max_blocks_per_thread = (handle->mb * handle->nb + max_threads - 1) / max_threads;
    avg_blocks_per_thread = (double)handle->mb * handle->nb / max_threads;
    load_imbalance_2      = max_blocks_per_thread / avg_blocks_per_thread;
    max_work_per_block    = handle->bm * handle->bn;
    avg_work_per_block    = (double)(handle->m * handle->n) / (handle->mb * handle->nb);
    load_imbalance_1      = max_work_per_block / avg_work_per_block;
    load_imbalance        = load_imbalance_1 * load_imbalance_2;
  }

  /* This is temporary space needed; allocate for each different size of A */
  internal_spmdm_allocate_csr_a(handle, libxsmm_output_csr);
  internal_spmdm_allocate_scratch(handle, max_threads);

  /* Initialize shuffle masks for the computation */
#if defined(LIBXSMM_SPMDM_AVX)
  if (LIBXSMM_X86_AVX <= libxsmm_target_archid) {
    internal_spmdm_init_shufmask_avx();
  }
  assert(0 != internal_spmdm_shufmasks_32);
  assert(0 != internal_spmdm_shufmasks_16);
#endif

  /* post-conditions */
  assert(0 != internal_spmdm_createSparseSlice_fp32_thread);
  assert(0 != internal_spmdm_createSparseSlice_bfloat16_thread);
  assert(0 != internal_spmdm_compute_fp32_thread);
  assert(0 != internal_spmdm_compute_bfloat16_thread);
}

