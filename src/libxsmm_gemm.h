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
#include <stdio.h>
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

#if !defined(LIBXSMM_NO_BLAS)
# if !defined(__BLAS) || (0 != __BLAS)
#   define LIBXSMM_NO_BLAS 0
# else
#   define LIBXSMM_NO_BLAS 1
# endif
#endif

#if defined(_CRAYC)
# define LIBXSMM_EXT_FOR_SINGLE LIBXSMM_NOOP
#else
# define LIBXSMM_EXT_FOR_SINGLE LIBXSMM_EXT_SINGLE
#endif

#define LIBXSMM_GEMM_TILED_ABOVE_THRESHOLD(M, N, K) \
  (((LIBXSMM_MAX_M < (M)) || \
    (LIBXSMM_MAX_N < (N)) || \
    (LIBXSMM_MAX_K < (K))) ? 1 : 0)

#define LIBXSMM_GEMM_NO_BYPASS(FLAGS, ALPHA, BETA) ( \
  0 == ((FLAGS) & (LIBXSMM_GEMM_FLAG_TRANS_A | LIBXSMM_GEMM_FLAG_TRANS_B)) && \
        (LIBXSMM_FEQ(1, ALPHA) /*|| LIBXSMM_FEQ(-1, ALPHA)*/) && \
        (LIBXSMM_FEQ(1, BETA) || LIBXSMM_FEQ(0, BETA)))

#if defined(LIBXSMM_GEMM_TILED_INNER_FALLBACK)
# define LIBXSMM_GEMM_TILED_FALLBACK_CHECK(CONDITION) if (CONDITION)
# define LIBXSMM_GEMM_TILED_FALLBACK(TYPE, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) else { \
    LIBXSMM_FALLBACK0(TYPE, libxsmm_blasint, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  }
#else
# define LIBXSMM_GEMM_TILED_FALLBACK_CHECK(CONDITION)
# define LIBXSMM_GEMM_TILED_FALLBACK(TYPE, FLAGS, TILE_M, TILE_N, TILE_K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#endif

#define LIBXSMM_GEMM_TILED_KERNEL(KERNEL_INNER_BETA1, TYPE, FLAGS, POS_I, POS_J, MAX_K, TILE_M, TILE_N, TILE_K, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  const libxsmm_blasint libxsmm_tiled_xgemm_kernel_tm_ = LIBXSMM_MIN(TILE_M, (M) - (POS_I)); \
  const libxsmm_blasint libxsmm_tiled_xgemm_kernel_tn_ = LIBXSMM_MIN(TILE_N, (N) - (POS_J)); \
  const libxsmm_blasint libxsmm_tiled_xgemm_kernel_tk_ = ((TILE_K) <= (K) ? (TILE_K) : ((K) - (MAX_K))); \
  const TYPE* libxsmm_tiled_xgemm_kernel_ia_ = (A) + (POS_I); \
  const TYPE* libxsmm_tiled_xgemm_kernel_ib_ = (B) + (POS_J) * (LDB); \
  const TYPE* libxsmm_tiled_xgemm_kernel_pa_ = libxsmm_tiled_xgemm_kernel_ia_ + (libxsmm_tiled_xgemm_kernel_tk_) * (LDA); \
  const TYPE* libxsmm_tiled_xgemm_kernel_pb_ = libxsmm_tiled_xgemm_kernel_ib_ + (libxsmm_tiled_xgemm_kernel_tk_); \
  TYPE *const libxsmm_tiled_xgemm_kernel_ic_ = (C) + (POS_J) * (LDC) + (POS_I), libxsmm_tiled_xgemm_kernel_beta_ = BETA; \
  libxsmm_gemm_descriptor libxsmm_tiled_xgemm_kernel_desc_; \
  libxsmm_xmmfunction libxsmm_gemm_tiled_kernel_ = { 0 }; \
  libxsmm_blasint libxsmm_tiled_xgemm_kernel_k_ = 0; \
  if (((TILE_M) == libxsmm_tiled_xgemm_kernel_tm_) && ((TILE_N) == libxsmm_tiled_xgemm_kernel_tn_) && ((TILE_K) == libxsmm_tiled_xgemm_kernel_tk_)) { \
    if (libxsmm_tiled_xgemm_kernel_k_ < (MAX_K)) { /* peel */ \
      LIBXSMM_GEMM_DESCRIPTOR(libxsmm_tiled_xgemm_kernel_desc_, LIBXSMM_ALIGNMENT, FLAGS, TILE_M, TILE_N, TILE_K, \
        LDA, LDB, LDC, ALPHA, BETA, libxsmm_tiled_gemm_prefetch); \
      libxsmm_gemm_tiled_kernel_ = libxsmm_xmmdispatch(&libxsmm_tiled_xgemm_kernel_desc_); \
      LIBXSMM_GEMM_TILED_FALLBACK_CHECK(0 != libxsmm_gemm_tiled_kernel_.LIBXSMM_TPREFIX(TYPE, mm)) \
      { \
        LIBXSMM_MMCALL_PRF(libxsmm_gemm_tiled_kernel_.LIBXSMM_TPREFIX(TYPE, mm), \
          libxsmm_tiled_xgemm_kernel_ia_, libxsmm_tiled_xgemm_kernel_ib_, libxsmm_tiled_xgemm_kernel_ic_, \
          libxsmm_tiled_xgemm_kernel_pa_, libxsmm_tiled_xgemm_kernel_pb_, libxsmm_tiled_xgemm_kernel_ic_); \
      } \
      LIBXSMM_GEMM_TILED_FALLBACK(TYPE, FLAGS, TILE_M, TILE_N, TILE_K, \
        ALPHA, libxsmm_tiled_xgemm_kernel_ia_, LDA, libxsmm_tiled_xgemm_kernel_ib_, LDB, \
         BETA, libxsmm_tiled_xgemm_kernel_ic_, LDC); \
      libxsmm_tiled_xgemm_kernel_ia_ = libxsmm_tiled_xgemm_kernel_pa_; \
      libxsmm_tiled_xgemm_kernel_ib_ = libxsmm_tiled_xgemm_kernel_pb_; \
      libxsmm_tiled_xgemm_kernel_pa_ += (TILE_K) * (LDA); \
      libxsmm_tiled_xgemm_kernel_pb_ += TILE_K; \
      libxsmm_tiled_xgemm_kernel_k_ = TILE_K; \
      libxsmm_tiled_xgemm_kernel_beta_ = 1; \
    } \
    for (; libxsmm_tiled_xgemm_kernel_k_ < (MAX_K); libxsmm_tiled_xgemm_kernel_k_ += TILE_K) { /* inner */ \
      LIBXSMM_MMCALL_PRF((KERNEL_INNER_BETA1).LIBXSMM_TPREFIX(TYPE, mm), \
        libxsmm_tiled_xgemm_kernel_ia_, libxsmm_tiled_xgemm_kernel_ib_, libxsmm_tiled_xgemm_kernel_ic_, \
        libxsmm_tiled_xgemm_kernel_pa_, libxsmm_tiled_xgemm_kernel_pb_, libxsmm_tiled_xgemm_kernel_ic_); \
      libxsmm_tiled_xgemm_kernel_ia_ = libxsmm_tiled_xgemm_kernel_pa_; \
      libxsmm_tiled_xgemm_kernel_ib_ = libxsmm_tiled_xgemm_kernel_pb_; \
      libxsmm_tiled_xgemm_kernel_pa_ += (TILE_K) * (LDA); \
      libxsmm_tiled_xgemm_kernel_pb_ += TILE_K; \
    } \
  } \
  if (libxsmm_tiled_xgemm_kernel_k_ < (K)) { /* remainder */ \
    LIBXSMM_GEMM_DESCRIPTOR(libxsmm_tiled_xgemm_kernel_desc_, LIBXSMM_ALIGNMENT, FLAGS, \
      libxsmm_tiled_xgemm_kernel_tm_, libxsmm_tiled_xgemm_kernel_tn_, (K) - libxsmm_tiled_xgemm_kernel_k_, \
      LDA, LDB, LDC, ALPHA, libxsmm_tiled_xgemm_kernel_beta_, libxsmm_tiled_gemm_prefetch); \
    libxsmm_gemm_tiled_kernel_ = libxsmm_xmmdispatch(&libxsmm_tiled_xgemm_kernel_desc_); \
    LIBXSMM_GEMM_TILED_FALLBACK_CHECK(0 != libxsmm_gemm_tiled_kernel_.LIBXSMM_TPREFIX(TYPE, mm)) \
    { \
      LIBXSMM_MMCALL_PRF(libxsmm_gemm_tiled_kernel_.LIBXSMM_TPREFIX(TYPE, mm), \
        libxsmm_tiled_xgemm_kernel_ia_, libxsmm_tiled_xgemm_kernel_ib_, libxsmm_tiled_xgemm_kernel_ic_, \
        libxsmm_tiled_xgemm_kernel_pa_, libxsmm_tiled_xgemm_kernel_pb_, libxsmm_tiled_xgemm_kernel_ic_); \
    } \
    LIBXSMM_GEMM_TILED_FALLBACK(TYPE, FLAGS, libxsmm_tiled_xgemm_kernel_tm_, libxsmm_tiled_xgemm_kernel_tn_, \
      LIBXSMM_MIN(TILE_K, (K) - libxsmm_tiled_xgemm_kernel_k_), \
      ALPHA, libxsmm_tiled_xgemm_kernel_ia_, LDA, libxsmm_tiled_xgemm_kernel_ib_, LDB, \
      libxsmm_tiled_xgemm_kernel_beta_, libxsmm_tiled_xgemm_kernel_ic_, LDC); \
  } \
}

#define LIBXSMM_TILED_XGEMM(PARALLEL, SINGLE_OUTER, SINGLE_INNER, COLLAPSE, LOOP_START, KERNEL_START, SYNC, \
  MIN_TASKS, OVERHEAD, NT, TYPE, FLAGS, TILE_M, TILE_N, TILE_K, MM, NN, KK, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
SINGLE_OUTER { /* use NN, etc. rather than N due to below char. constant */ \
  const int libxsmm_tiled_xgemm_above_threshold_ = LIBXSMM_GEMM_TILED_ABOVE_THRESHOLD(MM, NN, KK); \
  const int libxsmm_tiled_xgemm_no_bypass_ = LIBXSMM_GEMM_NO_BYPASS(FLAGS, ALPHA, BETA); \
  libxsmm_blasint libxsmm_tiled_xgemm_tm_ = 0, libxsmm_tiled_xgemm_tn_ = 0, libxsmm_tiled_xgemm_tk_ = 0; \
  libxsmm_blasint libxsmm_tiled_xgemm_num_m_ = 0, libxsmm_tiled_xgemm_num_n_ = 0, libxsmm_tiled_xgemm_num_k_ = 0; \
  libxsmm_xmmfunction libxsmm_tiled_xgemm_kernel_ = { 0 }; \
  SINGLE_INNER \
  if (0 != libxsmm_tiled_xgemm_above_threshold_ && 0 != libxsmm_tiled_xgemm_no_bypass_) { \
    libxsmm_tiled_xgemm_num_m_ = ((MM) + (TILE_M) - 1) / (TILE_M); \
    libxsmm_tiled_xgemm_num_n_ = ((NN) + (TILE_N) - 1) / (TILE_N); \
    libxsmm_tiled_xgemm_num_k_ = ((KK) + (TILE_K) - 1) / (TILE_K); \
    { /* opening scope for additional variable declarations */ \
      const libxsmm_blasint libxsmm_tiled_xgemm_num_t_ = (OVERHEAD(NT) < libxsmm_tiled_xgemm_num_k_ && 1 < (COLLAPSE)) \
        ? (libxsmm_tiled_xgemm_num_m_ * libxsmm_tiled_xgemm_num_n_) \
        : (libxsmm_tiled_xgemm_num_n_ <= libxsmm_tiled_xgemm_num_m_ ? libxsmm_tiled_xgemm_num_m_ : libxsmm_tiled_xgemm_num_n_); \
      const libxsmm_blasint libxsmm_tiled_xgemm_min_ntasks_ = MIN_TASKS(NT); \
      libxsmm_gemm_descriptor libxsmm_tiled_xgemm_desc_; \
      if (libxsmm_tiled_xgemm_min_ntasks_ <= libxsmm_tiled_xgemm_num_t_) { /* ensure enough parallel slack */ \
        libxsmm_tiled_xgemm_tm_ = (MM) / libxsmm_tiled_xgemm_num_m_; \
        libxsmm_tiled_xgemm_tn_ = (NN) / libxsmm_tiled_xgemm_num_n_; \
      } \
      else if (OVERHEAD(NT) < libxsmm_tiled_xgemm_num_k_) { \
        const libxsmm_blasint libxsmm_tiled_xgemm_ratio_ = LIBXSMM_SQRT2(libxsmm_tiled_xgemm_min_ntasks_ / libxsmm_tiled_xgemm_num_t_); \
        libxsmm_tiled_xgemm_tn_ = (libxsmm_tiled_xgemm_num_n_ * libxsmm_tiled_xgemm_ratio_); \
        libxsmm_tiled_xgemm_tm_ = (libxsmm_tiled_xgemm_min_ntasks_ + libxsmm_tiled_xgemm_tn_ - 1) / libxsmm_tiled_xgemm_tn_; \
      } \
      else if (libxsmm_tiled_xgemm_num_n_ <= libxsmm_tiled_xgemm_num_m_) { \
        libxsmm_tiled_xgemm_tm_ = ((MM) + libxsmm_tiled_xgemm_min_ntasks_ - 1) / libxsmm_tiled_xgemm_min_ntasks_; \
        libxsmm_tiled_xgemm_tn_ = TILE_N; \
      } \
      else { \
        libxsmm_tiled_xgemm_tm_ = TILE_M; \
        libxsmm_tiled_xgemm_tn_ = ((NN) + libxsmm_tiled_xgemm_min_ntasks_ - 1) / libxsmm_tiled_xgemm_min_ntasks_; \
      } \
      libxsmm_tiled_xgemm_tk_ = TILE_K; \
      { /* adjust for non-square operand shapes */ \
        float libxsmm_tiled_xgemm_rm_ = 1.f, libxsmm_tiled_xgemm_rn_ = ((float)(NN)) / (MM), libxsmm_tiled_xgemm_rk_ = ((float)(KK)) / (MM); \
        if (1.f < libxsmm_tiled_xgemm_rn_) { libxsmm_tiled_xgemm_rm_ /= libxsmm_tiled_xgemm_rn_; libxsmm_tiled_xgemm_rn_ = 1.f; libxsmm_tiled_xgemm_rk_ /= libxsmm_tiled_xgemm_rn_; } \
        if (1.f < libxsmm_tiled_xgemm_rk_) { libxsmm_tiled_xgemm_rm_ /= libxsmm_tiled_xgemm_rk_; libxsmm_tiled_xgemm_rn_ /= libxsmm_tiled_xgemm_rk_; libxsmm_tiled_xgemm_rk_ = 1.f; } \
        libxsmm_tiled_xgemm_tm_ = LIBXSMM_CLMP((libxsmm_blasint)(1 << LIBXSMM_LOG2(libxsmm_tiled_xgemm_tm_ * libxsmm_tiled_xgemm_rm_)/* + 0.5*/), 8, MM); \
        libxsmm_tiled_xgemm_tn_ = LIBXSMM_CLMP((libxsmm_blasint)(1 << LIBXSMM_LOG2(libxsmm_tiled_xgemm_tn_ * libxsmm_tiled_xgemm_rn_)/* + 0.5*/), 8, NN); \
        libxsmm_tiled_xgemm_tk_ = LIBXSMM_CLMP((libxsmm_blasint)(1 << LIBXSMM_LOG2(libxsmm_tiled_xgemm_tk_ * libxsmm_tiled_xgemm_rk_)/* + 0.5*/), 8, KK); \
      } \
      LIBXSMM_GEMM_DESCRIPTOR(libxsmm_tiled_xgemm_desc_, LIBXSMM_ALIGNMENT, FLAGS, \
        libxsmm_tiled_xgemm_tm_, libxsmm_tiled_xgemm_tn_, libxsmm_tiled_xgemm_tk_, \
        LDA, LDB, LDC, ALPHA, 1/*beta*/, libxsmm_tiled_gemm_prefetch); \
      libxsmm_tiled_xgemm_kernel_ = libxsmm_xmmdispatch(&libxsmm_tiled_xgemm_desc_); \
    } \
  } \
  if (0 != libxsmm_tiled_xgemm_kernel_.LIBXSMM_TPREFIX(TYPE, mm)) { \
    const int libxsmm_tiled_xgemm_amortized_ = (OVERHEAD(NT) * libxsmm_tiled_xgemm_tn_) < (KK); \
    const libxsmm_blasint libxsmm_tiled_xgemm_max_k_ = ((KK) / libxsmm_tiled_xgemm_tk_) * libxsmm_tiled_xgemm_tk_; \
    libxsmm_blasint libxsmm_tiled_xgemm_m_ = MM, libxsmm_tiled_xgemm_n_ = NN, libxsmm_tiled_xgemm_i_ = 0, libxsmm_tiled_xgemm_j_ = 0; \
    libxsmm_blasint libxsmm_tiled_xgemm_dm_ = libxsmm_tiled_xgemm_tm_, libxsmm_tiled_xgemm_dn_ = libxsmm_tiled_xgemm_tn_; \
    libxsmm_blasint libxsmm_tiled_xgemm_swap_ = 0; \
    if ((1 == (COLLAPSE) || 0 == libxsmm_tiled_xgemm_amortized_) && \
      libxsmm_tiled_xgemm_tn_ * (MM) < libxsmm_tiled_xgemm_tm_ * (NN)) /* approx. of num_m < num_n */ \
    { \
      libxsmm_tiled_xgemm_swap_ = libxsmm_tiled_xgemm_dm_; libxsmm_tiled_xgemm_dm_ = libxsmm_tiled_xgemm_dn_; libxsmm_tiled_xgemm_dn_ = libxsmm_tiled_xgemm_swap_; \
      libxsmm_tiled_xgemm_swap_ = libxsmm_tiled_xgemm_m_; libxsmm_tiled_xgemm_m_ = libxsmm_tiled_xgemm_n_; libxsmm_tiled_xgemm_n_ = libxsmm_tiled_xgemm_swap_; \
    } \
    if (0 != libxsmm_tiled_xgemm_amortized_) { /* amortized overhead */ \
      PARALLEL LOOP_START(COLLAPSE, libxsmm_tiled_xgemm_i_, libxsmm_tiled_xgemm_j_) \
      for (libxsmm_tiled_xgemm_i_ = 0; libxsmm_tiled_xgemm_i_ < libxsmm_tiled_xgemm_m_; libxsmm_tiled_xgemm_i_ += libxsmm_tiled_xgemm_dm_) { \
        for (libxsmm_tiled_xgemm_j_ = 0; libxsmm_tiled_xgemm_j_ < libxsmm_tiled_xgemm_n_; libxsmm_tiled_xgemm_j_ += libxsmm_tiled_xgemm_dn_) { \
          KERNEL_START(libxsmm_tiled_xgemm_i_, libxsmm_tiled_xgemm_j_) \
          LIBXSMM_GEMM_TILED_KERNEL(libxsmm_tiled_xgemm_kernel_, TYPE, FLAGS, \
            0 == libxsmm_tiled_xgemm_swap_ ? libxsmm_tiled_xgemm_i_ : libxsmm_tiled_xgemm_j_, \
            0 == libxsmm_tiled_xgemm_swap_ ? libxsmm_tiled_xgemm_j_ : libxsmm_tiled_xgemm_i_, \
            libxsmm_tiled_xgemm_max_k_, libxsmm_tiled_xgemm_tm_, libxsmm_tiled_xgemm_tn_, libxsmm_tiled_xgemm_tk_, \
            MM, NN, KK, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
        } \
      } \
    } \
    else { \
      PARALLEL LOOP_START(1/*COLLAPSE*/, libxsmm_tiled_xgemm_i_, libxsmm_tiled_xgemm_j_) \
      for (libxsmm_tiled_xgemm_i_ = 0; libxsmm_tiled_xgemm_i_ < libxsmm_tiled_xgemm_m_; libxsmm_tiled_xgemm_i_ += libxsmm_tiled_xgemm_dm_) { \
        KERNEL_START(libxsmm_tiled_xgemm_i_) \
        for (libxsmm_tiled_xgemm_j_ = 0; libxsmm_tiled_xgemm_j_ < libxsmm_tiled_xgemm_n_; libxsmm_tiled_xgemm_j_ += libxsmm_tiled_xgemm_dn_) { \
          LIBXSMM_GEMM_TILED_KERNEL(libxsmm_tiled_xgemm_kernel_, TYPE, FLAGS, \
            0 == libxsmm_tiled_xgemm_swap_ ? libxsmm_tiled_xgemm_i_ : libxsmm_tiled_xgemm_j_, \
            0 == libxsmm_tiled_xgemm_swap_ ? libxsmm_tiled_xgemm_j_ : libxsmm_tiled_xgemm_i_, \
            libxsmm_tiled_xgemm_max_k_, libxsmm_tiled_xgemm_tm_, libxsmm_tiled_xgemm_tn_, libxsmm_tiled_xgemm_tk_, \
            MM, NN, KK, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
        } \
      } \
    } \
    SYNC \
  } \
  else if ((0 == libxsmm_tiled_xgemm_above_threshold_ /* small problem size */ \
    && 0 != libxsmm_tiled_xgemm_no_bypass_)) \
  { \
    LIBXSMM_GEMM_DESCRIPTOR_TYPE(libxsmm_tiled_xgemm_smalldesc_, LIBXSMM_ALIGNMENT, FLAGS, MM, NN, KK, \
      LDA, LDB, LDC, ALPHA, BETA, LIBXSMM_PREFETCH_NONE); \
    libxsmm_tiled_xgemm_kernel_ = libxsmm_xmmdispatch(&libxsmm_tiled_xgemm_smalldesc_); \
    if (0 != libxsmm_tiled_xgemm_kernel_.LIBXSMM_TPREFIX(TYPE, mm)) { \
      LIBXSMM_MMCALL_ABC/*no prefetch*/(libxsmm_tiled_xgemm_kernel_.LIBXSMM_TPREFIX(TYPE, mm), A, B, C); \
    } \
    else { /* fall-back */ \
      assert(0 == LIBXSMM_NO_BLAS); \
      LIBXSMM_FALLBACK0(TYPE, libxsmm_blasint, FLAGS, MM, NN, KK, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
      if ((unsigned int)LIBXSMM_ABS(libxsmm_verbosity) > libxsmm_update_mmstatistic(FLAGS, MM, NN, KK, 1, 0)) { \
        const char libxsmm_tiled_xgemm_transa_ = (char)(0 == ((FLAGS) & LIBXSMM_GEMM_FLAG_TRANS_A) ? 'N' : 'T'); \
        const char libxsmm_tiled_xgemm_transb_ = (char)(0 == ((FLAGS) & LIBXSMM_GEMM_FLAG_TRANS_B) ? 'N' : 'T'); \
        const TYPE libxsmm_tiled_xgemm_alpha_ = (TYPE)(ALPHA), libxsmm_tiled_xgemm_beta_ = (TYPE)(BETA); \
        libxsmm_gemm_print(0 < libxsmm_verbosity ? stderr : 0, LIBXSMM_GEMM_TYPEFLAG(TYPE), \
          &libxsmm_tiled_xgemm_transa_, &libxsmm_tiled_xgemm_transb_, &(MM), &(NN), &(KK), \
          &libxsmm_tiled_xgemm_alpha_, A, &(LDA), B, &(LDB), &libxsmm_tiled_xgemm_beta_, C, &(LDC)); \
        if (0 < libxsmm_verbosity) fprintf(stderr, "\n"); \
      } \
    } \
  } \
  else { /* fall-back */ \
    assert(0 == LIBXSMM_NO_BLAS); \
    LIBXSMM_FALLBACK1(TYPE, libxsmm_blasint, FLAGS, MM, NN, KK, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
    if ((unsigned int)LIBXSMM_ABS(libxsmm_verbosity) > libxsmm_update_mmstatistic(FLAGS, MM, NN, KK, 1, 0)) { \
      const char libxsmm_tiled_xgemm_transa_ = (char)(0 == ((FLAGS) & LIBXSMM_GEMM_FLAG_TRANS_A) ? 'N' : 'T'); \
      const char libxsmm_tiled_xgemm_transb_ = (char)(0 == ((FLAGS) & LIBXSMM_GEMM_FLAG_TRANS_B) ? 'N' : 'T'); \
      const TYPE libxsmm_tiled_xgemm_alpha_ = (TYPE)(ALPHA), libxsmm_tiled_xgemm_beta_ = (TYPE)(BETA); \
      libxsmm_gemm_print(0 < libxsmm_verbosity ? stderr : 0, LIBXSMM_GEMM_TYPEFLAG(TYPE), \
        &libxsmm_tiled_xgemm_transa_, &libxsmm_tiled_xgemm_transb_, &(MM), &(NN), &(KK), \
        &libxsmm_tiled_xgemm_alpha_, A, &(LDA), B, &(LDB), &libxsmm_tiled_xgemm_beta_, C, &(LDC)); \
      if (0 < libxsmm_verbosity) fprintf(stderr, "\n"); \
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
      static int libxsmm_gemm_wrapper_error_once_ = 0; \
      if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&libxsmm_gemm_wrapper_error_once_, 1, LIBXSMM_ATOMIC_RELAXED)) { \
        fprintf(stderr, "LIBXSMM: application must be linked against a LAPACK/BLAS implementation!\n"); \
      } \
    } \
  }
#endif


/** Provides GEMM functions available via BLAS; NOT thread-safe. */
LIBXSMM_API void libxsmm_gemm_init(int archid, int prefetch/*default prefetch strategy*/);

/** Finalizes the GEMM facility; NOT thread-safe. */
LIBXSMM_API void libxsmm_gemm_finalize(void);

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
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int libxsmm_tiled_gemm_prefetch;

#endif /*LIBXSMM_GEMM_H*/

