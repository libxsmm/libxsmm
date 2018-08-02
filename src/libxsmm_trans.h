/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
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
#ifndef LIBXSMM_TRANS_H
#define LIBXSMM_TRANS_H

#include <libxsmm.h>

#if !defined(LIBXSMM_TRANS_CHECK) && !defined(NDEBUG)
# define LIBXSMM_TRANS_CHECK
#endif
#if !defined(LIBXSMM_TRANS_TASKSCALE)
# define LIBXSMM_TRANS_TASKSCALE 2
#endif

/* kernel uses consecutive stores and consecutive loads (copy) */
#define LIBXSMM_MCOPY_KERNEL(TYPE, TYPESIZE, OUT, IN, LDI, LDO, INDEX_I, INDEX_J, SRC, DST) \
  const TYPE *const SRC = (const TYPE*)(((const char*) (IN)) + (TYPESIZE) * ((INDEX_J) * (LDI) + (INDEX_I))); \
        TYPE *const DST = (      TYPE*)(((      char*)(OUT)) + (TYPESIZE) * ((INDEX_J) * (LDO) + (INDEX_I)))
/* call JIT-kernel (matrix-copy) */
#define LIBXSMM_MCOPY_CALL_NOPF(KERNEL, TYPESIZE, SRC, LDI, DST, LDO) { \
  const unsigned int libxsmm_mcopy_call_nopf_uldi_ = (unsigned int)(LDI); \
  const unsigned int libxsmm_mcopy_call_nopf_uldo_ = (unsigned int)(LDO); \
  (KERNEL)(SRC, &libxsmm_mcopy_call_nopf_uldi_, DST, &libxsmm_mcopy_call_nopf_uldo_); \
}
/* call JIT-kernel (matrix-copy with prefetch) */
#define LIBXSMM_MCOPY_CALL(PRFT_KERNEL, TYPESIZE, SRC, LDI, DST, LDO) { \
  const unsigned int libxsmm_mcopy_call_uldi_ = (unsigned int)(LDI); \
  const unsigned int libxsmm_mcopy_call_uldo_ = (unsigned int)(LDO); \
  (PRFT_KERNEL)(SRC, &libxsmm_mcopy_call_uldi_, DST, &libxsmm_mcopy_call_uldo_, \
    /*prefetch next line*/((const char*)(SRC)) + (TYPESIZE) * (LDI)); \
}
/* kernel uses consecutive stores and strided loads (transpose) */
#define LIBXSMM_TCOPY_KERNEL(TYPE, TYPESIZE, OUT, IN, LDI, LDO, INDEX_I, INDEX_J, SRC, DST) \
  const TYPE *const SRC = (const TYPE*)(((const char*) (IN)) + (TYPESIZE) * ((INDEX_J) * (LDI) + (INDEX_I))); \
        TYPE *const DST = (      TYPE*)(((      char*)(OUT)) + (TYPESIZE) * ((INDEX_I) * (LDO) + (INDEX_J)))
/* call JIT-kernel (transpose) */
#define LIBXSMM_TCOPY_CALL(KERNEL, TYPESIZE, SRC, LDI, DST, LDO) { \
  const unsigned int libxsmm_tcopy_call_uldi_ = (unsigned int)(LDI); \
  const unsigned int libxsmm_tcopy_call_uldo_ = (unsigned int)(LDO); \
  (KERNEL)(SRC, &libxsmm_tcopy_call_uldi_, DST, &libxsmm_tcopy_call_uldo_); \
}

#define LIBXSMM_XCOPY_LOOP_UNALIGNED(A)
#define LIBXSMM_XCOPY_LOOP(TYPE, TYPESIZE, XKERNEL, HINT_ALIGNED, OUT, IN, LDI, LDO, M0, M1, N0, N1) { \
  /*const*/int libxsmm_xcopy_loop_generic_ = (sizeof(TYPE) != (TYPESIZE)); /* mute warning (constant conditional) */ \
  libxsmm_blasint libxsmm_xcopy_loop_i_, libxsmm_xcopy_loop_j_; \
  if (0 == libxsmm_xcopy_loop_generic_) { /* specific type-size */ \
    for (libxsmm_xcopy_loop_i_ = M0; libxsmm_xcopy_loop_i_ < (libxsmm_blasint)(M1); ++libxsmm_xcopy_loop_i_) { \
      LIBXSMM_PRAGMA_NONTEMPORAL HINT_ALIGNED(OUT) \
      for (libxsmm_xcopy_loop_j_ = N0; libxsmm_xcopy_loop_j_ < (libxsmm_blasint)(N1); ++libxsmm_xcopy_loop_j_) { \
        XKERNEL(TYPE, TYPESIZE, OUT, IN, LDI, LDO, libxsmm_xcopy_loop_i_, libxsmm_xcopy_loop_j_, \
          libxsmm_xcopy_loop_src_, libxsmm_xcopy_loop_dst_); *libxsmm_xcopy_loop_dst_ = *libxsmm_xcopy_loop_src_; \
      } \
    } \
  } \
  else { /* generic type-size */ \
    unsigned int libxsmm_xcopy_loop_k_; \
    for (libxsmm_xcopy_loop_i_ = M0; libxsmm_xcopy_loop_i_ < (libxsmm_blasint)(M1); ++libxsmm_xcopy_loop_i_) { \
      LIBXSMM_PRAGMA_NONTEMPORAL HINT_ALIGNED(OUT) \
      for (libxsmm_xcopy_loop_j_ = N0; libxsmm_xcopy_loop_j_ < (libxsmm_blasint)(N1); ++libxsmm_xcopy_loop_j_) { \
        XKERNEL(TYPE, TYPESIZE, OUT, IN, LDI, LDO, libxsmm_xcopy_loop_i_, libxsmm_xcopy_loop_j_, \
          libxsmm_xcopy_loop_src_, libxsmm_xcopy_loop_dst_); \
        for (libxsmm_xcopy_loop_k_ = 0; libxsmm_xcopy_loop_k_ < (TYPESIZE); ++libxsmm_xcopy_loop_k_) { \
          libxsmm_xcopy_loop_dst_[libxsmm_xcopy_loop_k_] = libxsmm_xcopy_loop_src_[libxsmm_xcopy_loop_k_]; \
        } \
      } \
    } \
  } \
}

#define LIBXSMM_XALIGN_TCOPY(N0, TYPESIZE) (0 == LIBXSMM_MOD2((N0) * (TYPESIZE), LIBXSMM_ALIGNMENT))
#define LIBXSMM_XALIGN_MCOPY(N0, TYPESIZE) (1)

#define LIBXSMM_XCOPY_XALIGN(TYPE, TYPESIZE, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1, XALIGN) { \
  if (0 == LIBXSMM_MOD2((uintptr_t)(OUT), LIBXSMM_ALIGNMENT) && \
      0 == LIBXSMM_MOD2((LDO) * (TYPESIZE), LIBXSMM_ALIGNMENT) && \
      XALIGN(N0, TYPESIZE)) \
  { \
    LIBXSMM_XCOPY_LOOP(TYPE, TYPESIZE, XKERNEL, LIBXSMM_PRAGMA_VALIGNED_VAR, OUT, IN, LDI, LDO, M0, M1, N0, N1); \
  } \
  else { /* unaligned store */ \
    LIBXSMM_XCOPY_LOOP(TYPE, TYPESIZE, XKERNEL, LIBXSMM_XCOPY_LOOP_UNALIGNED, OUT, IN, LDI, LDO, M0, M1, N0, N1); \
  } \
}

#define LIBXSMM_XCOPY_NONJIT(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, M0, M1, N0, N1, XALIGN) { \
  switch(TYPESIZE) { \
    case 2: { \
      LIBXSMM_XCOPY_XALIGN(short, 2, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1, XALIGN); \
    } break; \
    case 4: { \
      LIBXSMM_XCOPY_XALIGN(float, 4, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1, XALIGN); \
    } break; \
    case 8: { \
      LIBXSMM_XCOPY_XALIGN(double, 8, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1, XALIGN); \
    } break; \
    case 16: { \
      typedef struct /*libxsmm_xcopy_nonjit_elem_t*/ { double value[2]; } libxsmm_xcopy_nonjit_elem_t; \
      LIBXSMM_XCOPY_XALIGN(libxsmm_xcopy_nonjit_elem_t, 16, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1, XALIGN); \
    } break; \
    default: { \
      LIBXSMM_XCOPY_XALIGN(char, TYPESIZE, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1, XALIGN); \
    } break; \
  } \
}

#if 1
# define LIBXSMM_XCOPY_PRECOND(COND)
#else
# define LIBXSMM_XCOPY_PRECOND(COND) COND
#endif

#define LIBXSMM_XCOPY(XKERNEL, KERNEL_CALL, KERNEL, OUT, IN, TYPESIZE, LDI, LDO, TILE_M, TILE_N, M0, M1, N0, N1, XALIGN) { \
  libxsmm_blasint libxsmm_xcopy_i_ = M0, libxsmm_xcopy_j_ = N0; \
  if (0 != (KERNEL)) { /* inner tiles with JIT */ \
    for (; libxsmm_xcopy_i_ < (libxsmm_blasint)((M1) - (TILE_M) + 1); libxsmm_xcopy_i_ += TILE_M) { \
      for (libxsmm_xcopy_j_ = N0; libxsmm_xcopy_j_ < (libxsmm_blasint)((N1) - (TILE_N) + 1); libxsmm_xcopy_j_ += TILE_N) { \
        XKERNEL(char, TYPESIZE, OUT, IN, LDI, LDO, libxsmm_xcopy_i_, libxsmm_xcopy_j_, libxsmm_xcopy_src_, libxsmm_xcopy_dst_); \
        KERNEL_CALL(KERNEL, TYPESIZE, libxsmm_xcopy_src_, LDI, libxsmm_xcopy_dst_, LDO); \
      } \
    } \
  } \
  else { /* inner tiles without JIT */ \
    for (; libxsmm_xcopy_i_ < (libxsmm_blasint)((M1) - (TILE_M) + 1); libxsmm_xcopy_i_ += TILE_M) { \
      for (libxsmm_xcopy_j_ = N0; libxsmm_xcopy_j_ < (libxsmm_blasint)((N1) - (TILE_N) + 1); libxsmm_xcopy_j_ += TILE_N) { \
        LIBXSMM_XCOPY_NONJIT(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, \
          libxsmm_xcopy_i_, libxsmm_xcopy_i_ + (TILE_M), \
          libxsmm_xcopy_j_, libxsmm_xcopy_j_ + (TILE_N), XALIGN); \
      } \
    } \
  } \
  LIBXSMM_XCOPY_PRECOND(if (libxsmm_xcopy_j_ < (N1))) { \
    for (libxsmm_xcopy_i_ = M0; libxsmm_xcopy_i_ < (libxsmm_blasint)((M1) - (TILE_M) + 1); libxsmm_xcopy_i_ += TILE_M) { \
      LIBXSMM_XCOPY_NONJIT(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, \
        libxsmm_xcopy_i_, libxsmm_xcopy_i_ + (TILE_M), \
        libxsmm_xcopy_j_, N1, XALIGN); \
    } \
  } \
  LIBXSMM_XCOPY_PRECOND(if (libxsmm_xcopy_i_ < (M1))) { \
    for (libxsmm_xcopy_j_ = N0; libxsmm_xcopy_j_ < (libxsmm_blasint)((N1) - (TILE_N)); libxsmm_xcopy_j_ += TILE_N) { \
      LIBXSMM_XCOPY_NONJIT(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, \
        libxsmm_xcopy_i_, M1, \
        libxsmm_xcopy_j_, libxsmm_xcopy_j_ + (TILE_N), XALIGN); \
    } \
  } \
  LIBXSMM_XCOPY_PRECOND(if (libxsmm_xcopy_i_ < (M1) && libxsmm_xcopy_j_ < (N1))) { \
    LIBXSMM_XCOPY_NONJIT(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, \
      libxsmm_xcopy_i_, M1, \
      libxsmm_xcopy_j_, N1, XALIGN); \
  } \
}


/** Initializes the transpose functionality; NOT thread-safe. */
LIBXSMM_API_INTERN void libxsmm_trans_init(int archid);
/** Finalizes the transpose functionality; NOT thread-safe. */
LIBXSMM_API_INTERN void libxsmm_trans_finalize(void);

LIBXSMM_API void libxsmm_matcopy_internal(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo, const int* prefetch,
  libxsmm_blasint tm, libxsmm_blasint tn, libxsmm_xmcopyfunction kernel,
  int tid, int nthreads);
LIBXSMM_API void libxsmm_otrans_thread_internal(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  libxsmm_blasint tm, libxsmm_blasint tn, libxsmm_xtransfunction kernel,
  int tid, int nthreads);
LIBXSMM_API_INTERN void libxsmm_otrans_internal(void* out, const void* in,
  unsigned int typesize, libxsmm_blasint ldi, libxsmm_blasint ldo,
  libxsmm_blasint m0, libxsmm_blasint m1, libxsmm_blasint n0, libxsmm_blasint n1,
  libxsmm_blasint tm, libxsmm_blasint tn, libxsmm_xtransfunction kernel);

/** Determines whether JIT-kernels are used or not (0: none, 1: matcopy, 2: transpose, 3: matcopy+transpose). */
LIBXSMM_APIVAR_PUBLIC(int libxsmm_trans_jit);
/** M-factor shaping the N-extent (tile shape). */
LIBXSMM_APIVAR_PUBLIC(float libxsmm_trans_tile_stretch);
/** Table of M-extents per type-size (tile shape). */
LIBXSMM_APIVAR_PUBLIC(libxsmm_blasint* libxsmm_trans_mtile);
/** Determines if OpenMP tasks are used, and scales beyond the number of threads. */
LIBXSMM_APIVAR_PUBLIC(int libxsmm_trans_taskscale);

#endif /*LIBXSMM_TRANS_H*/

