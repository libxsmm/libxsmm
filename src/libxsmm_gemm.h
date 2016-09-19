/******************************************************************************
** Copyright (c) 2015-2016, Intel Corporation                                **
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
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_GEMM_H
#define LIBXSMM_GEMM_H

#include <libxsmm.h>
#include <libxsmm_sync.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#if !defined(LIBXSMM_GEMM_WRAP_DYNAMIC) && defined(LIBXSMM_BUILD) && \
  (!defined(__BLAS) || (0 != __BLAS)) && defined(__GNUC__) && \
  !(defined(__APPLE__) && defined(__MACH__) && LIBXSMM_VERSION3(6, 1, 0) >= \
    LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)) && \
  !defined(_WIN32) && !defined(__CYGWIN__)
# include <dlfcn.h>
# define LIBXSMM_GEMM_WRAP_DYNAMIC
#endif
#if !defined(NDEBUG)
# include <stdio.h>
#endif
#include <math.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

/** Undefine (disarm) MKL's DIRECT_CALL macros. */
#if defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# if defined(sgemm_)
#   undef sgemm_
# endif
# if defined(dgemm_)
#   undef dgemm_
# endif
#endif

#if !defined(LIBXSMM_GEMM_COLLAPSE)
# define LIBXSMM_GEMM_COLLAPSE 2
#endif

/** Enable tiled GEMM in non-ext. library */
#if !defined(LIBXSMM_GEMM_TILED)
/*# define LIBXSMM_GEMM_TILED*/
#endif

#define LIBXSMM_GEMM_NO_BYPASS(FLAGS, ALPHA, BETA) ( \
  0 == ((FLAGS) & (LIBXSMM_GEMM_FLAG_TRANS_A | LIBXSMM_GEMM_FLAG_TRANS_B)) && \
  LIBXSMM_FEQ(1, ALPHA) && (LIBXSMM_FEQ(1, BETA) || LIBXSMM_FEQ(0, BETA)))

#define LIBXSMM_GEMM_TILED_ABOVE_THRESHOLD(M, N, K) (((LIBXSMM_MAX_M < (M)) || (LIBXSMM_MAX_N < (N)) || (LIBXSMM_MAX_K < (K))) ? 1 : 0)

#define LIBXSMM_GEMM_TILED_KERNEL(KERNEL_INNER, TYPE, FLAGS, POS_H, POS_I, TILE_M, TILE_N, TILE_K, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  const libxsmm_blasint libxsmm_gemm_tiled_kernel_mm_ = LIBXSMM_MIN(TILE_M, (M) - (POS_H)); \
  const libxsmm_blasint libxsmm_gemm_tiled_kernel_nn_ = LIBXSMM_MIN(TILE_N, (N) - (POS_I)); \
  libxsmm_blasint libxsmm_gemm_tiled_kernel_ij_ = 0, libxsmm_gemm_tiled_kernel_pj_ = TILE_K; \
  const TYPE* libxsmm_gemm_tiled_kernel_ia_ = (A) + (POS_H); \
  const TYPE* libxsmm_gemm_tiled_kernel_ib_ = (B) + (POS_I) * (LDB); \
  const TYPE* libxsmm_gemm_tiled_kernel_pa_ = libxsmm_gemm_tiled_kernel_ia_ + (TILE_K) * (LDA); \
  const TYPE* libxsmm_gemm_tiled_kernel_pb_ = libxsmm_gemm_tiled_kernel_ib_ + (TILE_K); \
  TYPE *const libxsmm_gemm_tiled_kernel_ic_ = (C) + (POS_I) * (LDC) + (POS_H); \
  libxsmm_gemm_descriptor libxsmm_tiled_gemm_kernel_desc_; \
  if (((TILE_M) == libxsmm_gemm_tiled_kernel_mm_) && ((TILE_N) == libxsmm_gemm_tiled_kernel_nn_)) { \
    for (; libxsmm_gemm_tiled_kernel_ij_ < libxsmm_tiled_gemm_max_j_; libxsmm_gemm_tiled_kernel_ij_ = libxsmm_gemm_tiled_kernel_pj_) { \
      LIBXSMM_MMCALL_PRF((KERNEL_INNER).LIBXSMM_TPREFIX(TYPE, mm), \
        libxsmm_gemm_tiled_kernel_ia_, libxsmm_gemm_tiled_kernel_ib_, libxsmm_gemm_tiled_kernel_ic_, \
        libxsmm_gemm_tiled_kernel_pa_, libxsmm_gemm_tiled_kernel_pb_, libxsmm_gemm_tiled_kernel_ic_); \
      libxsmm_gemm_tiled_kernel_ia_ = libxsmm_gemm_tiled_kernel_pa_; libxsmm_gemm_tiled_kernel_ib_ = libxsmm_gemm_tiled_kernel_pb_; \
      libxsmm_gemm_tiled_kernel_pj_ = libxsmm_gemm_tiled_kernel_ij_ + (TILE_K); \
      libxsmm_gemm_tiled_kernel_pa_ += (TILE_K) * (LDA); \
      libxsmm_gemm_tiled_kernel_pb_ += (TILE_K); \
    } \
  } \
  for (; libxsmm_gemm_tiled_kernel_ij_ < (K); libxsmm_gemm_tiled_kernel_ij_ = libxsmm_gemm_tiled_kernel_pj_) { /* remainder */ \
    libxsmm_xmmfunction libxsmm_gemm_tiled_kernel_outer_; \
    LIBXSMM_GEMM_DESCRIPTOR(libxsmm_tiled_gemm_kernel_desc_, LIBXSMM_ALIGNMENT, FLAGS, \
      libxsmm_gemm_tiled_kernel_mm_, libxsmm_gemm_tiled_kernel_nn_, LIBXSMM_MIN(TILE_K, (K) - libxsmm_gemm_tiled_kernel_ij_), \
      LDA, LDB, LDC, ALPHA, BETA, libxsmm_gemm_prefetch); \
    libxsmm_gemm_tiled_kernel_outer_ = libxsmm_xmmdispatch(&libxsmm_tiled_gemm_kernel_desc_); \
    if (0 != libxsmm_gemm_tiled_kernel_outer_.LIBXSMM_TPREFIX(TYPE, mm)) { \
      LIBXSMM_MMCALL_PRF(libxsmm_gemm_tiled_kernel_outer_.LIBXSMM_TPREFIX(TYPE, mm), \
        libxsmm_gemm_tiled_kernel_ia_, libxsmm_gemm_tiled_kernel_ib_, libxsmm_gemm_tiled_kernel_ic_, \
        libxsmm_gemm_tiled_kernel_pa_, libxsmm_gemm_tiled_kernel_pb_, libxsmm_gemm_tiled_kernel_ic_); \
    } \
    else { \
      LIBXSMM_FALLBACK0(TYPE, libxsmm_blasint, FLAGS, libxsmm_gemm_tiled_kernel_mm_, libxsmm_gemm_tiled_kernel_nn_, \
        LIBXSMM_MIN(TILE_K, (K) - libxsmm_gemm_tiled_kernel_ij_), \
        ALPHA, libxsmm_gemm_tiled_kernel_ia_, LDA, libxsmm_gemm_tiled_kernel_ib_, LDB, \
         BETA, libxsmm_gemm_tiled_kernel_ic_, LDC); \
    } \
    libxsmm_gemm_tiled_kernel_ia_ = libxsmm_gemm_tiled_kernel_pa_; libxsmm_gemm_tiled_kernel_ib_ = libxsmm_gemm_tiled_kernel_pb_; \
    libxsmm_gemm_tiled_kernel_pj_ = libxsmm_gemm_tiled_kernel_ij_ + (TILE_K); \
    libxsmm_gemm_tiled_kernel_pa_ += (TILE_K) * (LDA); \
    libxsmm_gemm_tiled_kernel_pb_ += (TILE_K); \
  } \
}

#if !defined(LIBXSMM_EXT_GEMM_BLAS)
# if !defined(__BLAS) || (0 != __BLAS)
#   define LIBXSMM_EXT_GEMM_BLAS 1
# else
#   define LIBXSMM_EXT_GEMM_BLAS 0
# endif
#endif

#define LIBXSMM_TILED_XGEMM(PARALLEL, SINGLE_OUTER, SINGLE_INNER, COLLAPSE, LOOP_START, KERNEL_START, SYNC, \
  MIN_TASKS, OVERHEAD, NT, TYPE, FLAGS, TILE_M, TILE_N, TILE_K, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
SINGLE_OUTER { \
  const int libxsmm_tiled_gemm_above_threshold_ = LIBXSMM_GEMM_TILED_ABOVE_THRESHOLD(M, N, K); \
  libxsmm_blasint libxsmm_tiled_gemm_tile_m_ = 0, libxsmm_tiled_gemm_tile_n_ = 0, libxsmm_tiled_gemm_tile_k_ = 0; \
  libxsmm_blasint libxsmm_tiled_gemm_num_m_ = 0, libxsmm_tiled_gemm_num_n_ = 0, libxsmm_tiled_gemm_num_k_ = 0; \
  libxsmm_xmmfunction libxsmm_tiled_gemm_xmm_ = { 0 }; \
  SINGLE_INNER \
  if (0 != libxsmm_tiled_gemm_above_threshold_ && LIBXSMM_GEMM_NO_BYPASS(FLAGS, ALPHA, BETA)) { \
    libxsmm_tiled_gemm_num_m_ = LIBXSMM_MAX(((M) + (TILE_M) - 1) / (TILE_M), 4); \
    libxsmm_tiled_gemm_num_n_ = LIBXSMM_MAX(((N) + (TILE_N) - 1) / (TILE_N), 2); \
    libxsmm_tiled_gemm_num_k_ = ((K) + (TILE_K) - 1) / (TILE_K); \
    { /* opening scope for additional variable declarations */ \
      const libxsmm_blasint libxsmm_tiled_gemm_num_t_ = (OVERHEAD(NT) <= libxsmm_tiled_gemm_num_k_ && 1 < (COLLAPSE)) \
        ? (libxsmm_tiled_gemm_num_m_ * libxsmm_tiled_gemm_num_n_) \
        : (libxsmm_tiled_gemm_num_n_ <= libxsmm_tiled_gemm_num_m_ ? libxsmm_tiled_gemm_num_m_ : libxsmm_tiled_gemm_num_n_); \
      const libxsmm_blasint libxsmm_tiled_gemm_min_ntasks_ = MIN_TASKS(NT); \
      libxsmm_gemm_descriptor libxsmm_tiled_gemm_desc_; \
      if (libxsmm_tiled_gemm_min_ntasks_ < libxsmm_tiled_gemm_num_t_) { /* ensure enough parallel slack */ \
        libxsmm_tiled_gemm_tile_m_ = (M) / libxsmm_tiled_gemm_num_m_; \
        libxsmm_tiled_gemm_tile_n_ = (N) / libxsmm_tiled_gemm_num_n_; \
      } \
      else if ((OVERHEAD(NT)) <= libxsmm_tiled_gemm_num_k_) { \
        const libxsmm_blasint libxsmm_tiled_gemm_ratio_ = LIBXSMM_SQRT2(libxsmm_tiled_gemm_min_ntasks_ / libxsmm_tiled_gemm_num_t_); \
        libxsmm_tiled_gemm_tile_n_ = (libxsmm_tiled_gemm_num_n_ * libxsmm_tiled_gemm_ratio_); \
        libxsmm_tiled_gemm_tile_m_ = (libxsmm_tiled_gemm_min_ntasks_ + libxsmm_tiled_gemm_tile_n_ - 1) / libxsmm_tiled_gemm_tile_n_; \
      } \
      else if (libxsmm_tiled_gemm_num_n_ <= libxsmm_tiled_gemm_num_m_) { \
        libxsmm_tiled_gemm_tile_m_ = ((M) + libxsmm_tiled_gemm_min_ntasks_ - 1) / libxsmm_tiled_gemm_min_ntasks_; \
        libxsmm_tiled_gemm_tile_n_ = TILE_N; \
      } \
      else { \
        libxsmm_tiled_gemm_tile_m_ = TILE_M; \
        libxsmm_tiled_gemm_tile_n_ = ((N) + libxsmm_tiled_gemm_min_ntasks_ - 1) / libxsmm_tiled_gemm_min_ntasks_; \
      } \
      libxsmm_tiled_gemm_tile_k_ = TILE_K; \
      { /* adjust for non-square operand shapes */ \
        float libxsmm_tiled_gemm_rm_ = 1.f, libxsmm_tiled_gemm_rn_ = ((float)(N)) / (M), libxsmm_tiled_gemm_rk_ = ((float)(K)) / (M); \
        if (1.f < libxsmm_tiled_gemm_rn_) { libxsmm_tiled_gemm_rm_ /= libxsmm_tiled_gemm_rn_; libxsmm_tiled_gemm_rn_ = 1.f; libxsmm_tiled_gemm_rk_ /= libxsmm_tiled_gemm_rn_; } \
        if (1.f < libxsmm_tiled_gemm_rk_) { libxsmm_tiled_gemm_rm_ /= libxsmm_tiled_gemm_rk_; libxsmm_tiled_gemm_rn_ /= libxsmm_tiled_gemm_rk_; libxsmm_tiled_gemm_rk_ = 1.f; } \
        libxsmm_tiled_gemm_tile_m_ = LIBXSMM_CLMP((libxsmm_blasint)(1 << LIBXSMM_LOG2(libxsmm_tiled_gemm_tile_m_ * libxsmm_tiled_gemm_rm_)/* + 0.5*/), 8, M); \
        libxsmm_tiled_gemm_tile_n_ = LIBXSMM_CLMP((libxsmm_blasint)(1 << LIBXSMM_LOG2(libxsmm_tiled_gemm_tile_n_ * libxsmm_tiled_gemm_rn_)/* + 0.5*/), 8, N); \
        libxsmm_tiled_gemm_tile_k_ = LIBXSMM_CLMP((libxsmm_blasint)(1 << LIBXSMM_LOG2(libxsmm_tiled_gemm_tile_k_ * libxsmm_tiled_gemm_rk_)/* + 0.5*/), 8, K); \
      } \
      LIBXSMM_GEMM_DESCRIPTOR(libxsmm_tiled_gemm_desc_, LIBXSMM_ALIGNMENT, FLAGS, \
        libxsmm_tiled_gemm_tile_m_, libxsmm_tiled_gemm_tile_n_, libxsmm_tiled_gemm_tile_k_, \
        LDA, LDB, LDC, ALPHA, BETA, libxsmm_gemm_prefetch); \
      libxsmm_tiled_gemm_xmm_ = libxsmm_xmmdispatch(&libxsmm_tiled_gemm_desc_); \
    } \
  } \
  if (0 != libxsmm_tiled_gemm_xmm_.LIBXSMM_TPREFIX(TYPE, mm)) { \
    const libxsmm_blasint libxsmm_tiled_gemm_max_j_ = ((K) / libxsmm_tiled_gemm_tile_k_) * libxsmm_tiled_gemm_tile_k_; \
    libxsmm_blasint libxsmm_tiled_gemm_h_ = 0, libxsmm_tiled_gemm_i_ = 0; \
    if ((OVERHEAD(NT)) <= libxsmm_tiled_gemm_num_k_) { /* amortize overhead */ \
      PARALLEL LOOP_START(COLLAPSE) \
      for (libxsmm_tiled_gemm_h_ = 0; libxsmm_tiled_gemm_h_ < (M); libxsmm_tiled_gemm_h_ += libxsmm_tiled_gemm_tile_m_) { \
        for (libxsmm_tiled_gemm_i_ = 0; libxsmm_tiled_gemm_i_ < (N); libxsmm_tiled_gemm_i_ += libxsmm_tiled_gemm_tile_n_) { \
          KERNEL_START(libxsmm_tiled_gemm_h_, libxsmm_tiled_gemm_i_) \
          LIBXSMM_GEMM_TILED_KERNEL(libxsmm_tiled_gemm_xmm_, TYPE, FLAGS, libxsmm_tiled_gemm_h_, libxsmm_tiled_gemm_i_, \
            libxsmm_tiled_gemm_tile_m_, libxsmm_tiled_gemm_tile_n_, libxsmm_tiled_gemm_tile_k_, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
        } \
      } \
    } \
    else if (libxsmm_tiled_gemm_num_n_ <= libxsmm_tiled_gemm_num_m_) { \
      PARALLEL LOOP_START(COLLAPSE) \
      for (libxsmm_tiled_gemm_h_ = 0; libxsmm_tiled_gemm_h_ < (M); libxsmm_tiled_gemm_h_ += libxsmm_tiled_gemm_tile_m_) { \
        KERNEL_START(libxsmm_tiled_gemm_h_) \
        for (libxsmm_tiled_gemm_i_ = 0; libxsmm_tiled_gemm_i_ < (N); libxsmm_tiled_gemm_i_ += libxsmm_tiled_gemm_tile_n_) { \
          LIBXSMM_GEMM_TILED_KERNEL(libxsmm_tiled_gemm_xmm_, TYPE, FLAGS, libxsmm_tiled_gemm_h_, libxsmm_tiled_gemm_i_, \
            libxsmm_tiled_gemm_tile_m_, libxsmm_tiled_gemm_tile_n_, libxsmm_tiled_gemm_tile_k_, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
        } \
      } \
    } \
    else { \
      PARALLEL LOOP_START(COLLAPSE) \
      for (libxsmm_tiled_gemm_i_ = 0; libxsmm_tiled_gemm_i_ < (N); libxsmm_tiled_gemm_i_ += libxsmm_tiled_gemm_tile_n_) { \
        KERNEL_START(libxsmm_tiled_gemm_i_) \
        for (libxsmm_tiled_gemm_h_ = 0; libxsmm_tiled_gemm_h_ < (M); libxsmm_tiled_gemm_h_ += libxsmm_tiled_gemm_tile_m_) { \
          LIBXSMM_GEMM_TILED_KERNEL(libxsmm_tiled_gemm_xmm_, TYPE, FLAGS, libxsmm_tiled_gemm_h_, libxsmm_tiled_gemm_i_, \
            libxsmm_tiled_gemm_tile_m_, libxsmm_tiled_gemm_tile_n_, libxsmm_tiled_gemm_tile_k_, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
        } \
      } \
    } \
    SYNC \
  } \
  else if (0 != libxsmm_tiled_gemm_above_threshold_ && 0 != LIBXSMM_EXT_GEMM_BLAS) { /* fall-back */ \
    LIBXSMM_FALLBACK1(TYPE, libxsmm_blasint, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
  else { /* small problem size */ \
    libxsmm_xmmfunction libxsmm_gemm_tiled_kernel_; \
    LIBXSMM_GEMM_DESCRIPTOR_TYPE(libxsmm_tiled_gemm_smalldesc_, LIBXSMM_ALIGNMENT, FLAGS, M, N, K, \
      LDA, LDB, LDC, ALPHA, BETA, LIBXSMM_PREFETCH_NONE); \
    libxsmm_gemm_tiled_kernel_ = libxsmm_xmmdispatch(&libxsmm_tiled_gemm_smalldesc_); \
    if (0 != libxsmm_gemm_tiled_kernel_.LIBXSMM_TPREFIX(TYPE, mm)) { \
      LIBXSMM_MMCALL_ABC/*no prefetch*/(libxsmm_gemm_tiled_kernel_.LIBXSMM_TPREFIX(TYPE, mm), A, B, C); \
    } \
    else { \
      LIBXSMM_FALLBACK0(TYPE, libxsmm_blasint, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
    } \
  } \
}

#if (!defined(__BLAS) || (0 != __BLAS))
# define LIBXSMM_GEMM_WRAPPER_BLAS(TYPE, ORIGINAL, CALLER, SYMBOL) if (0 == (ORIGINAL)) { \
    union { const void* pv; LIBXSMM_GEMMFUNCTION_TYPE(TYPE) pf; } libxsmm_gemm_wrapper_blas_; \
    libxsmm_gemm_wrapper_blas_.pf = (SYMBOL); \
    if (libxsmm_gemm_wrapper_blas_.pv != (CALLER)) ORIGINAL = libxsmm_gemm_wrapper_blas_.pf; \
  }
#else
# define LIBXSMM_GEMM_WRAPPER_BLAS(TYPE, ORIGINAL, CALLER, SYMBOL) LIBXSMM_UNUSED(CALLER)
#endif

#if defined(LIBXSMM_GEMM_WRAP) && defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT) && \
  !(defined(__APPLE__) && defined(__MACH__) /*&& defined(__clang__)*/) && !defined(__CYGWIN__)
# if (2 != (LIBXSMM_GEMM_WRAP)) /* SGEMM and DGEMM */
#   define LIBXSMM_GEMM_WRAPPER_STATIC(TYPE, ORIGINAL, CALLER) LIBXSMM_GEMM_WRAPPER_BLAS(TYPE, ORIGINAL, CALLER, \
      LIBXSMM_FSYMBOL(LIBXSMM_CONCATENATE(__real_, LIBXSMM_TPREFIX(TYPE, gemm))))
# else /* DGEMM only */
#   define LIBXSMM_GEMM_WRAPPER_STATIC(TYPE, ORIGINAL, CALLER) LIBXSMM_EQUAL(TYPE, double, \
      LIBXSMM_GEMM_WRAPPER_BLAS(TYPE, ORIGINAL, CALLER, LIBXSMM_FSYMBOL(__real_dgemm)))
# endif
# define LIBXSMM_GEMM_WRAP_STATIC
#else
# define LIBXSMM_GEMM_WRAPPER_STATIC(TYPE, ORIGINAL, CALLER)
#endif

#if defined(LIBXSMM_GEMM_WRAP_DYNAMIC)
# define LIBXSMM_GEMM_WRAPPER_DYNAMIC(TYPE, ORIGINAL, CALLER) \
    if (0 == (ORIGINAL)) { \
      union { const void* pv; LIBXSMM_GEMMFUNCTION_TYPE(TYPE) pf; } libxsmm_gemm_wrapper_dynamic_ = { 0 }; \
      dlerror(); /* clear an eventual error status */ \
      libxsmm_gemm_wrapper_dynamic_.pv = dlsym(RTLD_NEXT, LIBXSMM_STRINGIFY(LIBXSMM_FSYMBOL(LIBXSMM_TPREFIX(TYPE, gemm)))); \
      if (libxsmm_gemm_wrapper_dynamic_.pv != (CALLER)) ORIGINAL = libxsmm_gemm_wrapper_dynamic_.pf; \
      LIBXSMM_GEMM_WRAPPER_BLAS(TYPE, ORIGINAL, CALLER, LIBXSMM_FSYMBOL(LIBXSMM_TPREFIX(TYPE, gemm))); \
    }
#else
# define LIBXSMM_GEMM_WRAPPER_DYNAMIC(TYPE, ORIGINAL, CALLER) LIBXSMM_GEMM_WRAPPER_BLAS( \
    TYPE, ORIGINAL, CALLER, LIBXSMM_FSYMBOL(LIBXSMM_TPREFIX(TYPE, gemm)))
#endif

#if defined(NDEBUG) /* library code is expected to be mute */
# define LIBXSMM_GEMM_WRAPPER(TYPE, ORIGINAL, CALLER) if (0 == (ORIGINAL)) { \
    LIBXSMM_GEMM_WRAPPER_STATIC(TYPE, ORIGINAL, CALLER); \
    LIBXSMM_GEMM_WRAPPER_DYNAMIC(TYPE, ORIGINAL, CALLER); \
  }
#else
# define LIBXSMM_GEMM_WRAPPER(TYPE, ORIGINAL, CALLER) if (0 == (ORIGINAL)) { \
    LIBXSMM_GEMM_WRAPPER_STATIC(TYPE, ORIGINAL, CALLER); \
    LIBXSMM_GEMM_WRAPPER_DYNAMIC(TYPE, ORIGINAL, CALLER); \
    if (0 == (ORIGINAL)) { \
      static LIBXSMM_TLS int libxsmm_gemm_wrapper_error_ = 0; \
      if (0 == libxsmm_gemm_wrapper_error_) { \
        fprintf(stderr, "LIBXSMM: application must be linked against a LAPACK/BLAS implementation!\n"); \
        libxsmm_gemm_wrapper_error_ = 1; \
      } \
    } \
  }
#endif


/** Provides GEMM functions available via BLAS; NOT thread-safe. */
LIBXSMM_API void libxsmm_gemm_init(int archid, int prefetch/*default prefetch strategy*/);

/** Finalizes the GEMM facility; NOT thread-safe. */
LIBXSMM_API void libxsmm_gemm_finalize(void);

/** Helper function, which dumps all input and output data of a GEMM call. */
LIBXSMM_API void libxsmm_gemm_dump(libxsmm_gemm_xflags precision, const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
  const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc);

#if defined(LIBXSMM_GEMM_WRAP_STATIC)
LIBXSMM_EXTERN LIBXSMM_RETARGETABLE void LIBXSMM_FSYMBOL(__real_sgemm)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*, const float* b, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_EXTERN LIBXSMM_RETARGETABLE void LIBXSMM_FSYMBOL(__real_dgemm)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*, const double* b, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);
#endif /*defined(LIBXSMM_GEMM_WRAP_STATIC)*/

#if defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT)
LIBXSMM_EXTERN LIBXSMM_RETARGETABLE void LIBXSMM_FSYMBOL(__wrap_sgemm)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*, const float* b, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_EXTERN LIBXSMM_RETARGETABLE void LIBXSMM_FSYMBOL(__wrap_dgemm)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*, const double* b, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);
#endif

LIBXSMM_EXTERN LIBXSMM_RETARGETABLE void LIBXSMM_FSYMBOL(sgemm)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*, const float*, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_EXTERN LIBXSMM_RETARGETABLE void LIBXSMM_FSYMBOL(dgemm)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*, const double*, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);

/** Configuration table containing the tile sizes separate for DP and SP. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE LIBXSMM_GEMM_DESCRIPTOR_DIM_TYPE libxsmm_gemm_tile[2/*DP/SP*/][3/*TILE_M,TILE_N,TILE_K*/];
/** Prefetch strategy. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int libxsmm_gemm_prefetch;

#endif /*LIBXSMM_GEMM_H*/

