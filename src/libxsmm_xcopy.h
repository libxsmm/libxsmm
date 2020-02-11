/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_XCOPY_H
#define LIBXSMM_XCOPY_H

#include <libxsmm_typedefs.h>

#if !defined(LIBXSMM_XCOPY_CHECK) && !defined(NDEBUG)
# define LIBXSMM_XCOPY_CHECK
#endif
#if !defined(LIBXSMM_XCOPY_TASKSCALE)
# define LIBXSMM_XCOPY_TASKSCALE 2
#endif

/* kernel uses consecutive stores and consecutive loads (copy) */
#define LIBXSMM_MCOPY_KERNEL(TYPE, TYPESIZE, OUT, IN, LDI, LDO, INDEX_I, INDEX_J, SRC, DST) \
  const TYPE *const SRC = (const TYPE*)(((const char*) (IN)) + (TYPESIZE) * ((size_t)(INDEX_J) * (LDI) + (INDEX_I))); \
        TYPE *const DST = (      TYPE*)(((      char*)(OUT)) + (TYPESIZE) * ((size_t)(INDEX_J) * (LDO) + (INDEX_I)))
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
    /*prefetch next line*/((const char*)(SRC)) + (TYPESIZE) * (size_t)(LDI)); \
}
/* kernel uses consecutive stores and strided loads (transpose) */
#define LIBXSMM_TCOPY_KERNEL(TYPE, TYPESIZE, OUT, IN, LDI, LDO, INDEX_I, INDEX_J, SRC, DST) \
  const TYPE *const SRC = (const TYPE*)(((const char*) (IN)) + (TYPESIZE) * ((size_t)(INDEX_J) * (LDI) + (INDEX_I))); \
        TYPE *const DST = (      TYPE*)(((      char*)(OUT)) + (TYPESIZE) * ((size_t)(INDEX_I) * (LDO) + (INDEX_J)))
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
    for (libxsmm_xcopy_loop_i_ = M0; libxsmm_xcopy_loop_i_ < (libxsmm_blasint)(M1); ++libxsmm_xcopy_loop_i_) { \
      LIBXSMM_PRAGMA_NONTEMPORAL HINT_ALIGNED(OUT) \
      for (libxsmm_xcopy_loop_j_ = N0; libxsmm_xcopy_loop_j_ < (libxsmm_blasint)(N1); ++libxsmm_xcopy_loop_j_) { \
        XKERNEL(TYPE, TYPESIZE, OUT, IN, LDI, LDO, libxsmm_xcopy_loop_i_, libxsmm_xcopy_loop_j_, \
          libxsmm_xcopy_loop_src_, libxsmm_xcopy_loop_dst_); \
        LIBXSMM_MEMCPY127(libxsmm_xcopy_loop_dst_, libxsmm_xcopy_loop_src_, TYPESIZE); \
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
  LIBXSMM_ASSERT_MSG(0 < (TILE_M) && 0 < (TILE_N), "XCOPY cannot make progress"); \
  if (0 != (KERNEL)) { /* inner tiles with JIT */ \
    for (; libxsmm_xcopy_i_ < (((libxsmm_blasint)M1) - ((libxsmm_blasint)TILE_M) + 1); libxsmm_xcopy_i_ += TILE_M) { \
      for (libxsmm_xcopy_j_ = N0; libxsmm_xcopy_j_ < (((libxsmm_blasint)N1) - ((libxsmm_blasint)TILE_N) + 1); libxsmm_xcopy_j_ += TILE_N) { \
        XKERNEL(char, TYPESIZE, OUT, IN, LDI, LDO, libxsmm_xcopy_i_, libxsmm_xcopy_j_, libxsmm_xcopy_src_, libxsmm_xcopy_dst_); \
        KERNEL_CALL(KERNEL, TYPESIZE, libxsmm_xcopy_src_, LDI, libxsmm_xcopy_dst_, LDO); \
      } \
    } \
  } \
  else { /* inner tiles without JIT */ \
    for (; libxsmm_xcopy_i_ < (((libxsmm_blasint)M1) - ((libxsmm_blasint)TILE_M) + 1); libxsmm_xcopy_i_ += TILE_M) { \
      for (libxsmm_xcopy_j_ = N0; libxsmm_xcopy_j_ < (((libxsmm_blasint)N1) - ((libxsmm_blasint)TILE_N) + 1); libxsmm_xcopy_j_ += TILE_N) { \
        LIBXSMM_XCOPY_NONJIT(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, \
          libxsmm_xcopy_i_, libxsmm_xcopy_i_ + (TILE_M), \
          libxsmm_xcopy_j_, libxsmm_xcopy_j_ + (TILE_N), XALIGN); \
      } \
    } \
  } \
  LIBXSMM_XCOPY_PRECOND(if (libxsmm_xcopy_j_ < ((libxsmm_blasint)N1))) { \
    for (libxsmm_xcopy_i_ = M0; libxsmm_xcopy_i_ < (((libxsmm_blasint)M1) - ((libxsmm_blasint)TILE_M) + 1); libxsmm_xcopy_i_ += TILE_M) { \
      LIBXSMM_XCOPY_NONJIT(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, \
        libxsmm_xcopy_i_, libxsmm_xcopy_i_ + (TILE_M), \
        libxsmm_xcopy_j_, N1, XALIGN); \
    } \
  } \
  LIBXSMM_XCOPY_PRECOND(if (libxsmm_xcopy_i_ < ((libxsmm_blasint)M1))) { \
    for (libxsmm_xcopy_j_ = N0; libxsmm_xcopy_j_ < (((libxsmm_blasint)N1) - ((libxsmm_blasint)TILE_N)); libxsmm_xcopy_j_ += TILE_N) { \
      LIBXSMM_XCOPY_NONJIT(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, \
        libxsmm_xcopy_i_, M1, \
        libxsmm_xcopy_j_, libxsmm_xcopy_j_ + (TILE_N), XALIGN); \
    } \
  } \
  LIBXSMM_XCOPY_PRECOND(if (libxsmm_xcopy_i_ < ((libxsmm_blasint)M1) && libxsmm_xcopy_j_ < ((libxsmm_blasint)N1))) { \
    LIBXSMM_XCOPY_NONJIT(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, \
      libxsmm_xcopy_i_, M1, \
      libxsmm_xcopy_j_, N1, XALIGN); \
  } \
}


/** Initializes the transpose functionality; NOT thread-safe. */
LIBXSMM_API_INTERN void libxsmm_xcopy_init(int archid);
/** Finalizes the transpose functionality; NOT thread-safe. */
LIBXSMM_API_INTERN void libxsmm_xcopy_finalize(void);

LIBXSMM_API void libxsmm_matcopy_thread_internal(void* out, const void* in, unsigned int typesize,
  unsigned int m, unsigned int n, unsigned int ldi, unsigned int ldo, const int* prefetch,
  unsigned int tm, unsigned int tn, libxsmm_xmcopyfunction kernel,
  int tid, int nthreads);
LIBXSMM_API_INTERN void libxsmm_matcopy_internal_pf(void* out, const void* in,
  unsigned int typesize, unsigned int ldi, unsigned int ldo,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1,
  unsigned int tm, unsigned int tn, libxsmm_xmcopyfunction kernel);
LIBXSMM_API_INTERN void libxsmm_matcopy_internal(void* out, const void* in,
  unsigned int typesize, unsigned int ldi, unsigned int ldo,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1,
  unsigned int tm, unsigned int tn, libxsmm_xmcopyfunction kernel);

LIBXSMM_API void libxsmm_otrans_thread_internal(void* out, const void* in, unsigned int typesize,
  unsigned int m, unsigned int n, unsigned int ldi, unsigned int ldo,
  unsigned int tm, unsigned int tn, libxsmm_xtransfunction kernel,
  int tid, int nthreads);
LIBXSMM_API_INTERN void libxsmm_otrans_internal(void* out, const void* in,
  unsigned int typesize, unsigned int ldi, unsigned int ldo,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1,
  unsigned int tm, unsigned int tn, libxsmm_xtransfunction kernel);

/** Determines whether JIT-kernels are used or not (0: none, 1: matcopy, 2: transpose, 3: matcopy+transpose). */
LIBXSMM_APIVAR_PUBLIC(int libxsmm_trans_jit);
/** M-factor shaping the N-extent (tile shape). */
LIBXSMM_APIVAR_PUBLIC(float libxsmm_trans_tile_stretch);
/** Table of M-extents per type-size (tile shape). */
LIBXSMM_APIVAR_PUBLIC(unsigned int* libxsmm_trans_mtile);
/** Determines if OpenMP tasks are used, and scales beyond the number of threads. */
LIBXSMM_APIVAR_PUBLIC(int libxsmm_trans_taskscale);

#endif /*LIBXSMM_XCOPY_H*/

