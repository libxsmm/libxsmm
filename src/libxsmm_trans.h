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
#if !defined(LIBXSMM_TRANS_TYPEOPT)
# define LIBXSMM_TRANS_TYPEOPT
#endif

#define LIBXSMM_OTRANS_GENERIC(TYPESIZE, OUT, IN, M0, M1, N0, N1, N, LDI, LDO) { \
  const char *const libxsmm_otrans_generic_a_ = (const char*)(IN); \
  char *const libxsmm_otrans_generic_b_ = (char*)(OUT); \
  libxsmm_blasint libxsmm_otrans_generic_i_, libxsmm_otrans_generic_j_; \
  unsigned int libxsmm_otrans_generic_k_; \
  for (libxsmm_otrans_generic_i_ = M0; libxsmm_otrans_generic_i_ < (M1); ++libxsmm_otrans_generic_i_) { \
    LIBXSMM_PRAGMA_NONTEMPORAL \
    for (libxsmm_otrans_generic_j_ = N0; libxsmm_otrans_generic_j_ < (N1); ++libxsmm_otrans_generic_j_) { \
      const char *const libxsmm_otrans_generic_aji_ = libxsmm_otrans_generic_a_ + (TYPESIZE) * (libxsmm_otrans_generic_j_ * (LDI) + libxsmm_otrans_generic_i_); \
      char *const libxsmm_otrans_generic_bij_ = libxsmm_otrans_generic_b_ + (TYPESIZE) * (libxsmm_otrans_generic_i_ * (LDO) + libxsmm_otrans_generic_j_); \
      for (libxsmm_otrans_generic_k_ = 0; libxsmm_otrans_generic_k_ < (TYPESIZE); ++libxsmm_otrans_generic_k_) { \
        libxsmm_otrans_generic_bij_[libxsmm_otrans_generic_k_] = libxsmm_otrans_generic_aji_[libxsmm_otrans_generic_k_]; \
      } \
    } \
  } \
}

#define LIBXSMM_OTRANS(TYPE, OUT, IN, M0, M1, N0, N1, N, LDI, LDO) { \
  if (LIBXSMM_MAX(libxsmm_trans_chunksize, LIBXSMM_TRANS_MIN_CHUNKSIZE) == (N) \
   && LIBXSMM_MOD2((uintptr_t)(IN), LIBXSMM_ALIGNMENT) == 0) \
  { \
    const TYPE *const libxsmm_otrans_a_ = (const TYPE*)(IN); \
    TYPE *const libxsmm_otrans_b_ = (TYPE*)(OUT); \
    libxsmm_blasint libxsmm_otrans_generic_i_, libxsmm_otrans_generic_j_; \
    if (LIBXSMM_TRANS_MAX_CHUNKSIZE == (N)) { \
      for (libxsmm_otrans_generic_i_ = M0; libxsmm_otrans_generic_i_ < (M1); ++libxsmm_otrans_generic_i_) { \
        LIBXSMM_PRAGMA_NONTEMPORAL \
        LIBXSMM_PRAGMA_VALIGNED_VARS(libxsmm_otrans_b_) \
        for (libxsmm_otrans_generic_j_ = N0; libxsmm_otrans_generic_j_ < (N0) + (LIBXSMM_TRANS_MAX_CHUNKSIZE); ++libxsmm_otrans_generic_j_) { \
          /* use consecutive stores and strided loads */ \
          libxsmm_otrans_b_[libxsmm_otrans_generic_i_*(LDO)+libxsmm_otrans_generic_j_] = libxsmm_otrans_a_[libxsmm_otrans_generic_j_*(LDI)+libxsmm_otrans_generic_i_]; \
        } \
      } \
    } \
    else { \
      assert(LIBXSMM_TRANS_MIN_CHUNKSIZE == (N)); \
      for (libxsmm_otrans_generic_i_ = M0; libxsmm_otrans_generic_i_ < (M1); ++libxsmm_otrans_generic_i_) { \
        LIBXSMM_PRAGMA_NONTEMPORAL \
        LIBXSMM_PRAGMA_VALIGNED_VARS(libxsmm_otrans_b_) \
        for (libxsmm_otrans_generic_j_ = N0; libxsmm_otrans_generic_j_ < (N0) + (LIBXSMM_TRANS_MIN_CHUNKSIZE); ++libxsmm_otrans_generic_j_) { \
          /* use consecutive stores and strided loads */ \
          libxsmm_otrans_b_[libxsmm_otrans_generic_i_*(LDO)+libxsmm_otrans_generic_j_] = libxsmm_otrans_a_[libxsmm_otrans_generic_j_*(LDI)+libxsmm_otrans_generic_i_]; \
        } \
      } \
    } \
  } \
  else { /* remainder tile */ \
    LIBXSMM_OTRANS_GENERIC(sizeof(TYPE), OUT, IN, M0, M1, N0, N1, N, LDI, LDO); \
  } \
}

#if defined(LIBXSMM_TRANS_TYPEOPT)
# define LIBXSMM_OTRANS_TYPEOPT_BEGIN(OUT, IN, TYPESIZE, M0, M1, N0, N1, N, LDI, LDO) \
    switch(TYPESIZE) { \
      case 1: { \
        LIBXSMM_OTRANS(char, OUT, IN, M0, M1, N0, N1, N, LDI, LDO); \
      } break; \
      case 2: { \
        LIBXSMM_OTRANS(short, OUT, IN, M0, M1, N0, N1, N, LDI, LDO); \
      } break; \
      case 4: { \
        LIBXSMM_OTRANS(float, OUT, IN, M0, M1, N0, N1, N, LDI, LDO); \
      } break; \
      case 8: { \
        LIBXSMM_OTRANS(double, OUT, IN, M0, M1, N0, N1, N, LDI, LDO); \
      } break; \
      case 16: { \
        typedef struct dvec2_t { double value[2]; } dvec2_t; \
        LIBXSMM_OTRANS(dvec2_t, OUT, IN, M0, M1, N0, N1, N, LDI, LDO); \
      } break; \
      default:
#else
# define LIBXSMM_OTRANS_TYPEOPT_BEGIN(OUT, IN, TYPESIZE, M0, M1, N0, N1, N, LDI, LDO) {
#endif
#define LIBXSMM_OTRANS_TYPEOPT_END }

/**
 * Based on the cache-oblivious transpose by Frigo et.al. with some additional
 * optimization such as using a loop with bounds which are known at compile-time
 * due to splitting up tiles with one fixed-size extent (chunk).
 */
#define LIBXSMM_OTRANS_MAIN(KERNEL_START, FN, OUT, IN, TYPESIZE, M0, M1, N0, N1, LDI, LDO) { \
  /*const*/ libxsmm_blasint libxsmm_otrans_main_m_ = (M1) - (M0), libxsmm_otrans_main_n_ = (N1) - (N0); \
  if (libxsmm_otrans_main_m_ * libxsmm_otrans_main_n_ * (TYPESIZE) <= ((LIBXSMM_CPU_DCACHESIZE) / 2)) { \
    KERNEL_START(libxsmm_otrans_main_n_) \
    { \
      LIBXSMM_OTRANS_TYPEOPT_BEGIN(OUT, IN, TYPESIZE, M0, M1, N0, N1, libxsmm_otrans_main_n_, LDI, LDO) \
      /* fall-back code path which is generic with respect to the typesize */ \
      LIBXSMM_OTRANS_GENERIC(TYPESIZE, OUT, IN, M0, M1, N0, N1, libxsmm_otrans_main_n_, LDI, LDO); \
      LIBXSMM_OTRANS_TYPEOPT_END \
    } \
  } \
  else if (libxsmm_otrans_main_m_ >= libxsmm_otrans_main_n_) { \
    const libxsmm_blasint libxsmm_otrans_main_mi_ = ((M0) + (M1)) / 2; \
    (FN)(OUT, IN, TYPESIZE, M0, libxsmm_otrans_main_mi_, N0, N1, LDI, LDO); \
    (FN)(OUT, IN, TYPESIZE, libxsmm_otrans_main_mi_, M1, N0, N1, LDI, LDO); \
  } \
  else { \
    const int libxsmm_otrans_main_chunksize_ = LIBXSMM_MAX(libxsmm_trans_chunksize, LIBXSMM_TRANS_MIN_CHUNKSIZE); \
    if (libxsmm_otrans_main_chunksize_ < libxsmm_otrans_main_n_) { \
      const libxsmm_blasint libxsmm_otrans_main_ni_ = (N0) + libxsmm_trans_chunksize; \
      (FN)(OUT, IN, TYPESIZE, M0, M1, N0, libxsmm_otrans_main_ni_, LDI, LDO); \
      (FN)(OUT, IN, TYPESIZE, M0, M1, libxsmm_otrans_main_ni_, N1, LDI, LDO); \
    } \
    else \
    { \
      const libxsmm_blasint libxsmm_otrans_main_ni_ = ((N0) + (N1)) / 2; \
      (FN)(OUT, IN, TYPESIZE, M0, M1, N0, libxsmm_otrans_main_ni_, LDI, LDO); \
      (FN)(OUT, IN, TYPESIZE, M0, M1, libxsmm_otrans_main_ni_, N1, LDI, LDO); \
    } \
  } \
}


/** Initializes the tranpose functionality; NOT thread-safe. */
LIBXSMM_API void libxsmm_trans_init(int archid);

/** Finalizes the transpose functionality; NOT thread-safe. */
LIBXSMM_API void libxsmm_trans_finalize(void);


/** Size of peeled chunks during transposing inner tiles. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int libxsmm_trans_chunksize;

#endif /*LIBXSMM_TRANS_H*/

