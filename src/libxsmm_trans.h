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

#if !defined(LIBXSMM_TRANS_MIN_CHUNKSIZE)
# define LIBXSMM_TRANS_MIN_CHUNKSIZE 8
#endif
#if !defined(LIBXSMM_TRANS_MAX_CHUNKSIZE)
# define LIBXSMM_TRANS_MAX_CHUNKSIZE 32
#endif

#define LIBXSMM_OTRANS_KERNEL(TYPE, TYPESIZE, INDEX_I, INDEX_J, OUT, IN, LDI, LDO) \
  (OUT)[(INDEX_I)*(LDO)+(INDEX_J)] = (IN)[(INDEX_J)*(LDI)+(INDEX_I)]

#define LIBXSMM_OTRANS_KERNEL_GENERIC(TYPE, TYPESIZE, INDEX_I, INDEX_J, OUT, IN, LDI, LDO) { \
  const TYPE *const libxsmm_otrans_kernel_generic_a_ = (IN) + (TYPESIZE) * ((INDEX_J) * (LDI) + (INDEX_I)); \
  TYPE *const libxsmm_otrans_kernel_generic_b_ = (OUT) + (TYPESIZE) * ((INDEX_I) * (LDO) + (INDEX_J)); \
  unsigned int libxsmm_otrans_kernel_generic_k_; \
  for (libxsmm_otrans_kernel_generic_k_ = 0; libxsmm_otrans_kernel_generic_k_ < (TYPESIZE); ++libxsmm_otrans_kernel_generic_k_) { \
    libxsmm_otrans_kernel_generic_b_[libxsmm_otrans_kernel_generic_k_] = libxsmm_otrans_kernel_generic_a_[libxsmm_otrans_kernel_generic_k_]; \
  } \
}

#define LIBXSMM_OTRANS_LOOP_UNALIGNED(...)
#define LIBXSMM_OTRANS_LOOP(TYPE, TYPESIZE, KERNEL, HINT_ALIGNED, OUT, IN, M0, M1, N0, N1, NCHUNK, LDI, LDO) { \
  const TYPE *const libxsmm_otrans_loop_a_ = (const TYPE*)(IN); \
  TYPE *const libxsmm_otrans_loop_b_ = (TYPE*)(OUT); \
  libxsmm_blasint libxsmm_otrans_loop_i_, libxsmm_otrans_loop_j_; \
  for (libxsmm_otrans_loop_i_ = M0; libxsmm_otrans_loop_i_ < (M1); ++libxsmm_otrans_loop_i_) { \
    LIBXSMM_PRAGMA_NONTEMPORAL HINT_ALIGNED(libxsmm_otrans_loop_b_) \
    for (libxsmm_otrans_loop_j_ = N0; libxsmm_otrans_loop_j_ < ((N0) + (NCHUNK)); ++libxsmm_otrans_loop_j_) { \
      /* kernel uses consecutive stores and strided loads */ \
      KERNEL(TYPE, TYPESIZE, libxsmm_otrans_loop_i_, libxsmm_otrans_loop_j_, \
        libxsmm_otrans_loop_b_, libxsmm_otrans_loop_a_, LDI, LDO); \
    } \
  } \
}

#define LIBXSMM_OTRANS(TYPE, TYPESIZE, OUT, IN, M0, M1, N0, N1, N, LDI, LDO) { \
  if (LIBXSMM_MAX(libxsmm_trans_chunksize, LIBXSMM_TRANS_MIN_CHUNKSIZE) == (N)) { \
    if (0 == LIBXSMM_MOD2((LDO) * (TYPESIZE), LIBXSMM_ALIGNMENT) \
     && 0 == LIBXSMM_MOD2((uintptr_t)(OUT), LIBXSMM_ALIGNMENT)) \
    { \
      switch (N) { \
        case LIBXSMM_TRANS_MAX_CHUNKSIZE: { \
          LIBXSMM_OTRANS_LOOP(TYPE, TYPESIZE, LIBXSMM_OTRANS_KERNEL, LIBXSMM_PRAGMA_VALIGNED_VARS, \
            OUT, IN, M0, M1, N0, N1, LIBXSMM_TRANS_MAX_CHUNKSIZE, LDI, LDO); \
        } break; \
        case LIBXSMM_TRANS_MIN_CHUNKSIZE: { \
          LIBXSMM_OTRANS_LOOP(TYPE, TYPESIZE, LIBXSMM_OTRANS_KERNEL, LIBXSMM_PRAGMA_VALIGNED_VARS, \
            OUT, IN, M0, M1, N0, N1, LIBXSMM_TRANS_MIN_CHUNKSIZE, LDI, LDO); \
        } break; \
        default: { \
          LIBXSMM_OTRANS_LOOP(TYPE, TYPESIZE, LIBXSMM_OTRANS_KERNEL, LIBXSMM_PRAGMA_VALIGNED_VARS, \
            OUT, IN, M0, M1, N0, N1, N, LDI, LDO); \
        } \
      } \
    } \
    else { /* unaligned store */ \
      switch (N) { \
        case LIBXSMM_TRANS_MAX_CHUNKSIZE: { \
          LIBXSMM_OTRANS_LOOP(TYPE, TYPESIZE, LIBXSMM_OTRANS_KERNEL, LIBXSMM_OTRANS_LOOP_UNALIGNED, \
            OUT, IN, M0, M1, N0, N1, LIBXSMM_TRANS_MAX_CHUNKSIZE, LDI, LDO); \
        } break; \
        case LIBXSMM_TRANS_MIN_CHUNKSIZE: { \
          LIBXSMM_OTRANS_LOOP(TYPE, TYPESIZE, LIBXSMM_OTRANS_KERNEL, LIBXSMM_OTRANS_LOOP_UNALIGNED, \
            OUT, IN, M0, M1, N0, N1, LIBXSMM_TRANS_MIN_CHUNKSIZE, LDI, LDO); \
        } break; \
        default: { \
          LIBXSMM_OTRANS_LOOP(TYPE, TYPESIZE, LIBXSMM_OTRANS_KERNEL, LIBXSMM_OTRANS_LOOP_UNALIGNED, \
            OUT, IN, M0, M1, N0, N1, N, LDI, LDO); \
        } \
      } \
    } \
  } \
  else { /* remainder tile */ \
    LIBXSMM_OTRANS_LOOP(char, TYPESIZE, LIBXSMM_OTRANS_KERNEL_GENERIC, LIBXSMM_OTRANS_LOOP_UNALIGNED, \
      OUT, IN, M0, M1, N0, N1, N, LDI, LDO); \
  } \
}

/**
 * Based on the cache-oblivious transpose by Frigo et.al. with some additional
 * optimization such as using a loop with bounds, which are known at compile-time
 * due to splitting up tiles with one fixed-size extent (chunk).
 */
#define LIBXSMM_OTRANS_MAIN(FN, KERNEL_START, KERNEL, OUT, IN, TYPESIZE, M0, M1, N0, N1, LDI, LDO) { \
  const int libxsmm_otrans_main_chunksize_ = LIBXSMM_MAX(libxsmm_trans_chunksize, LIBXSMM_TRANS_MIN_CHUNKSIZE); \
  /*const*/ libxsmm_blasint libxsmm_otrans_main_m_ = (M1) - (M0), libxsmm_otrans_main_n_ = (N1) - (N0); \
  if (libxsmm_otrans_main_m_ <= libxsmm_trans_chunksize && libxsmm_otrans_main_n_ <= libxsmm_trans_chunksize) { \
    KERNEL_START(firstprivate(libxsmm_otrans_main_n_) untied) \
    if (0 != (KERNEL) /* check below if the current tile is an inner tile */ \
      && libxsmm_otrans_main_chunksize_ == libxsmm_otrans_main_m_ \
      && libxsmm_otrans_main_chunksize_ == libxsmm_otrans_main_n_) \
    { \
      const unsigned int libxsmm_otrans_main_ldi_ = LDI, libxsmm_otrans_main_ldo_ = LDO; \
      (KERNEL)( /* call the pre-scheduled JIT-kernel */ \
        ((const char*)(IN)) + (TYPESIZE) * ((N0) * (LDI) + (M0)), &libxsmm_otrans_main_ldi_, \
        ((char*)(OUT)) + (TYPESIZE) * ((M0) * (LDO) + (N0)), &libxsmm_otrans_main_ldo_); \
    } \
    else { \
      switch(TYPESIZE) { \
        case 2: { \
          LIBXSMM_OTRANS(short, 2, OUT, IN, M0, M1, N0, N1, libxsmm_otrans_main_n_, LDI, LDO); \
        } break; \
        case 4: { \
          LIBXSMM_OTRANS(float, 4, OUT, IN, M0, M1, N0, N1, libxsmm_otrans_main_n_, LDI, LDO); \
        } break; \
        case 8: { \
          LIBXSMM_OTRANS(double, 8, OUT, IN, M0, M1, N0, N1, libxsmm_otrans_main_n_, LDI, LDO); \
        } break; \
        case 16: { \
          typedef struct dvec2_t { double value[2]; } dvec2_t; \
          LIBXSMM_OTRANS(dvec2_t, 16, OUT, IN, M0, M1, N0, N1, libxsmm_otrans_main_n_, LDI, LDO); \
        } break; \
        default: { \
          LIBXSMM_OTRANS(char, TYPESIZE, OUT, IN, M0, M1, N0, N1, libxsmm_otrans_main_n_, LDI, LDO); \
        } break; \
      } \
    } \
  } \
  else if (libxsmm_otrans_main_m_ >= libxsmm_otrans_main_n_) { \
    const libxsmm_blasint libxsmm_otrans_main_mi_ = libxsmm_otrans_main_chunksize_ < libxsmm_otrans_main_m_ \
      ? ((M0) + libxsmm_otrans_main_chunksize_) : (((M0) + (M1)) / 2); \
    (FN)(KERNEL, OUT, IN, TYPESIZE, M0, libxsmm_otrans_main_mi_, N0, N1, LDI, LDO); \
    (FN)(KERNEL, OUT, IN, TYPESIZE, libxsmm_otrans_main_mi_, M1, N0, N1, LDI, LDO); \
  } \
  else { \
    const libxsmm_blasint libxsmm_otrans_main_ni_ = libxsmm_otrans_main_chunksize_ < libxsmm_otrans_main_n_ \
      ? ((N0) + libxsmm_otrans_main_chunksize_) : (((N0) + (N1)) / 2); \
    (FN)(KERNEL, OUT, IN, TYPESIZE, M0, M1, N0, libxsmm_otrans_main_ni_, LDI, LDO); \
    (FN)(KERNEL, OUT, IN, TYPESIZE, M0, M1, libxsmm_otrans_main_ni_, N1, LDI, LDO); \
  } \
}


/** Initializes the transpose functionality; NOT thread-safe. */
LIBXSMM_API void libxsmm_trans_init(int archid);

/** Finalizes the transpose functionality; NOT thread-safe. */
LIBXSMM_API void libxsmm_trans_finalize(void);


/** Size of peeled chunks during transposing inner tiles. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int libxsmm_trans_chunksize;

#endif /*LIBXSMM_TRANS_H*/

