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
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_TRANS_H
#define LIBXSMM_TRANS_H

#include <libxsmm.h>

#if !defined(LIBXSMM_TRANS_M)
# define LIBXSMM_TRANS_M 192
#endif
#if !defined(LIBXSMM_TRANS_N)
# define LIBXSMM_TRANS_N LIBXSMM_TRANS_M
#endif

/* kernel uses consecutive stores and consecutive loads (copy) */
#define LIBXSMM_MCOPY_KERNEL(TYPE, TYPESIZE, OUT, IN, LDI, LDO, INDEX_I, INDEX_J, SRC, DST) \
  const TYPE *const SRC = (const TYPE*)(((const char*)(IN)) + (TYPESIZE) * ((INDEX_I) * (LDI) + (INDEX_J))); \
  TYPE *const DST = (TYPE*)(((const char*)(OUT)) + (TYPESIZE) * ((INDEX_I) * (LDO) + (INDEX_J)))

/* kernel uses consecutive stores and strided loads (transpose) */
#define LIBXSMM_TCOPY_KERNEL(TYPE, TYPESIZE, OUT, IN, LDI, LDO, INDEX_I, INDEX_J, SRC, DST) \
  const TYPE *const SRC = (const TYPE*)(((const char*)(IN)) + (TYPESIZE) * ((INDEX_J) * (LDI) + (INDEX_I))); \
  TYPE *const DST = (TYPE*)(((const char*)(OUT)) + (TYPESIZE) * ((INDEX_I) * (LDO) + (INDEX_J)))

#define LIBXSMM_XCOPY_LOOP_UNALIGNED(...)
#define LIBXSMM_XCOPY_LOOP(TYPE, TYPESIZE, XKERNEL, HINT_ALIGNED, OUT, IN, LDI, LDO, M0, M1, N0, N1, NCHUNK) { \
  /*const*/int generic_type = sizeof(TYPE) == (TYPESIZE); /* used to mute warning (conditional is constant) */ \
  unsigned int libxsmm_xcopy_loop_i_, libxsmm_xcopy_loop_j_; \
  if (generic_type) { \
    for (libxsmm_xcopy_loop_i_ = M0; libxsmm_xcopy_loop_i_ < (M1); ++libxsmm_xcopy_loop_i_) { \
      LIBXSMM_PRAGMA_NONTEMPORAL HINT_ALIGNED(libxsmm_xcopy_loop_out_) \
      for (libxsmm_xcopy_loop_j_ = N0; libxsmm_xcopy_loop_j_ < ((N0) + (NCHUNK)); ++libxsmm_xcopy_loop_j_) { \
        XKERNEL(TYPE, TYPESIZE, OUT, IN, LDI, LDO, libxsmm_xcopy_loop_i_, libxsmm_xcopy_loop_j_, \
          libxsmm_xcopy_loop_src_, libxsmm_xcopy_loop_dst_); \
        *libxsmm_xcopy_loop_dst_ = *libxsmm_xcopy_loop_src_; \
      } \
    } \
  } \
  else { \
    unsigned int libxsmm_xcopy_loop_k_; \
    for (libxsmm_xcopy_loop_i_ = M0; libxsmm_xcopy_loop_i_ < (M1); ++libxsmm_xcopy_loop_i_) { \
      LIBXSMM_PRAGMA_NONTEMPORAL HINT_ALIGNED(libxsmm_xcopy_loop_out_) \
      for (libxsmm_xcopy_loop_j_ = N0; libxsmm_xcopy_loop_j_ < ((N0) + (NCHUNK)); ++libxsmm_xcopy_loop_j_) { \
        XKERNEL(TYPE, TYPESIZE, OUT, IN, LDI, LDO, libxsmm_xcopy_loop_i_, libxsmm_xcopy_loop_j_, \
          libxsmm_xcopy_loop_src_, libxsmm_xcopy_loop_dst_); \
        for (libxsmm_xcopy_loop_k_ = 0; libxsmm_xcopy_loop_k_ < (TYPESIZE); ++libxsmm_xcopy_loop_k_) { \
          libxsmm_xcopy_loop_dst_[libxsmm_xcopy_loop_k_] = libxsmm_xcopy_loop_src_[libxsmm_xcopy_loop_k_]; \
        } \
      } \
    } \
  } \
}

#define LIBXSMM_XCOPY_AUX(TYPE, TYPESIZE, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1, N) { \
  if (0 == LIBXSMM_MOD2((LDO) * (TYPESIZE), LIBXSMM_ALIGNMENT) \
   && 0 == LIBXSMM_MOD2((uintptr_t)(OUT), LIBXSMM_ALIGNMENT)) \
  { \
    if (LIBXSMM_TRANS_N == (N)) { \
      LIBXSMM_XCOPY_LOOP(TYPE, TYPESIZE, XKERNEL, LIBXSMM_PRAGMA_VALIGNED_VARS, \
        OUT, IN, LDI, LDO, M0, M1, N0, N1, LIBXSMM_TRANS_N); \
    } \
    else { \
      LIBXSMM_XCOPY_LOOP(TYPE, TYPESIZE, XKERNEL, LIBXSMM_PRAGMA_VALIGNED_VARS, \
        OUT, IN, LDI, LDO, M0, M1, N0, N1, N); \
    } \
  } \
  else { /* unaligned store */ \
    if (LIBXSMM_TRANS_N == (N)) { \
      LIBXSMM_XCOPY_LOOP(TYPE, TYPESIZE, XKERNEL, LIBXSMM_XCOPY_LOOP_UNALIGNED, \
        OUT, IN, LDI, LDO, M0, M1, N0, N1, LIBXSMM_TRANS_N); \
    } \
    else { \
      LIBXSMM_XCOPY_LOOP(TYPE, TYPESIZE, XKERNEL, LIBXSMM_XCOPY_LOOP_UNALIGNED, \
        OUT, IN, LDI, LDO, M0, M1, N0, N1, N); \
    } \
  } \
}

/**
 * Based on the cache-oblivious transpose by Frigo et.al. with some additional
 * optimization such as using a loop with bounds, which are known at compile-time
 * due to splitting up tiles with one fixed-size extent (chunk).
 */
#define LIBXSMM_XCOPY(FN, KERNEL_START, XKERNEL, KERNEL, OUT, IN, TYPESIZE, LDI, LDO, TILE_M, TILE_N, M0, M1, N0, N1) { \
  /*const*/ unsigned int libxsmm_xcopy_m_ = (M1) - (M0), libxsmm_xcopy_n_ = (N1) - (N0); \
  if (libxsmm_xcopy_m_ <= (TILE_M) && libxsmm_xcopy_n_ <= (TILE_N)) { \
    KERNEL_START(firstprivate(libxsmm_xcopy_m_, libxsmm_xcopy_n_) untied) \
    if (0 != (KERNEL) /* check below if the current tile is an inner tile */ \
      && (TILE_M) == libxsmm_xcopy_m_ && (TILE_N) == libxsmm_xcopy_n_) \
    { \
      const unsigned int libxsmm_xcopy_ldi_ = LDI, libxsmm_xcopy_ldo_ = LDO; \
      (KERNEL)( /* call the pre-scheduled JIT-kernel */ \
        ((const char*)(IN)) + (TYPESIZE) * ((N0) * (LDI) + (M0)), &libxsmm_xcopy_ldi_, \
        ((char*)(OUT)) + (TYPESIZE) * ((M0) * (LDO) + (N0)), &libxsmm_xcopy_ldo_); \
    } \
    else { \
      switch(TYPESIZE) { \
        case 2: { \
          LIBXSMM_XCOPY_AUX(short, 2, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1, libxsmm_xcopy_n_); \
        } break; \
        case 4: { \
          LIBXSMM_XCOPY_AUX(float, 4, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1, libxsmm_xcopy_n_); \
        } break; \
        case 8: { \
          LIBXSMM_XCOPY_AUX(double, 8, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1, libxsmm_xcopy_n_); \
        } break; \
        case 16: { \
          typedef struct dvec2_t { double value[2]; } dvec2_t; \
          LIBXSMM_XCOPY_AUX(dvec2_t, 16, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1, libxsmm_xcopy_n_); \
        } break; \
        default: { \
          LIBXSMM_XCOPY_AUX(char, TYPESIZE, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1, libxsmm_xcopy_n_); \
        } break; \
      } \
    } \
  } \
  else if (libxsmm_xcopy_m_ >= libxsmm_xcopy_n_) { \
    const unsigned int libxsmm_xcopy_mi_ = (TILE_M) < libxsmm_xcopy_m_ \
      ? ((M0) + (TILE_M)) : (((M0) + (M1)) / 2); \
    (FN)(KERNEL, OUT, IN, TYPESIZE, LDI, LDO, TILE_M, TILE_N, M0, libxsmm_xcopy_mi_, N0, N1); \
    (FN)(KERNEL, OUT, IN, TYPESIZE, LDI, LDO, TILE_M, TILE_N, libxsmm_xcopy_mi_, M1, N0, N1); \
  } \
  else { \
    const unsigned int libxsmm_xcopy_ni_ = (TILE_N) < libxsmm_xcopy_n_ \
      ? ((N0) + (TILE_N)) : (((N0) + (N1)) / 2); \
    (FN)(KERNEL, OUT, IN, TYPESIZE, LDI, LDO, TILE_M, TILE_N, M0, M1, N0, libxsmm_xcopy_ni_); \
    (FN)(KERNEL, OUT, IN, TYPESIZE, LDI, LDO, TILE_M, TILE_N, M0, M1, libxsmm_xcopy_ni_, N1); \
  } \
}


/** Initializes the transpose functionality; NOT thread-safe. */
LIBXSMM_API void libxsmm_trans_init(int archid);

/** Finalizes the transpose functionality; NOT thread-safe. */
LIBXSMM_API void libxsmm_trans_finalize(void);


/** Size of peeled chunks during transposing inner tiles. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE unsigned int libxsmm_trans_tile[2/*DP/SP*/][2/*M,N*/][8/*size-range*/];

#endif /*LIBXSMM_TRANS_H*/

