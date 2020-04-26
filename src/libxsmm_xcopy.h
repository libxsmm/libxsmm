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
/* 0: none, 1: transpose, 2: matcopy, 3: transpose+matcopy */
#if !defined(LIBXSMM_XCOPY_JIT)
# if (defined(_WIN32) || defined(__CYGWIN__))
/* only enable matcopy code generation (workaround issue with taking GP registers correctly) */
#   define LIBXSMM_XCOPY_JIT 0
# else
#   define LIBXSMM_XCOPY_JIT 1
# endif
#endif

/* kernel uses consecutive stores */
#define LIBXSMM_MZERO_KERNEL(TYPE, TYPESIZE, OUT, IN, LDI, LDO, INDEX_I, INDEX_J, SRC, DST) \
  static /*const*/ TYPE libxsmm_mzero_kernel_src_value_ /* zero */; \
  const TYPE *const SRC = &libxsmm_mzero_kernel_src_value_; \
  TYPE *const DST = (TYPE*)(((char*)(OUT)) + (TYPESIZE) * ((size_t)(INDEX_I) * (LDO) + (INDEX_J)))
/* call JIT-kernel (matrix-copy with prefetch) */
#define LIBXSMM_MZERO_CALL(KERNEL, TYPESIZE, SRC, LDI, DST, LDO) { \
  const unsigned int libxsmm_mzero_call_uldo_ = (unsigned int)(LDO); \
  (KERNEL)(SRC, &libxsmm_mzero_call_uldo_, DST, &libxsmm_mzero_call_uldo_); \
}
/* kernel uses consecutive stores and consecutive loads (copy) */
#define LIBXSMM_MCOPY_KERNEL(TYPE, TYPESIZE, OUT, IN, LDI, LDO, INDEX_I, INDEX_J, SRC, DST) \
  const TYPE *const SRC = (const TYPE*)(((const char*) (IN)) + (TYPESIZE) * ((size_t)(INDEX_I) * (LDI) + (INDEX_J))); \
        TYPE *const DST = (      TYPE*)(((      char*)(OUT)) + (TYPESIZE) * ((size_t)(INDEX_I) * (LDO) + (INDEX_J)))
/* call JIT-kernel (matrix-copy) */
#define LIBXSMM_MCOPY_CALL(KERNEL, TYPESIZE, SRC, LDI, DST, LDO) { \
  const unsigned int libxsmm_mcopy_call_nopf_uldi_ = (unsigned int)(LDI); \
  const unsigned int libxsmm_mcopy_call_nopf_uldo_ = (unsigned int)(LDO); \
  (KERNEL)(SRC, &libxsmm_mcopy_call_nopf_uldi_, DST, &libxsmm_mcopy_call_nopf_uldo_); \
}
/* call JIT-kernel (matrix-copy with prefetch) */
#define LIBXSMM_MCOPY_CALL_PF(PRFT_KERNEL, TYPESIZE, SRC, LDI, DST, LDO) { \
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

#define LIBXSMM_XCOPY_LOOP(TYPE, TYPESIZE, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1) { \
  /*const*/int libxsmm_xcopy_loop_generic_ = (sizeof(TYPE) != (TYPESIZE)); /* mute warning (constant conditional) */ \
  libxsmm_blasint libxsmm_xcopy_loop_i_, libxsmm_xcopy_loop_j_; \
  if (0 == libxsmm_xcopy_loop_generic_) { /* specific type-size */ \
    for (libxsmm_xcopy_loop_i_ = M0; libxsmm_xcopy_loop_i_ < (libxsmm_blasint)(M1); ++libxsmm_xcopy_loop_i_) { \
      LIBXSMM_PRAGMA_NONTEMPORAL \
      for (libxsmm_xcopy_loop_j_ = N0; libxsmm_xcopy_loop_j_ < (libxsmm_blasint)(N1); ++libxsmm_xcopy_loop_j_) { \
        XKERNEL(TYPE, TYPESIZE, OUT, IN, LDI, LDO, libxsmm_xcopy_loop_i_, libxsmm_xcopy_loop_j_, \
          libxsmm_xcopy_loop_src_, libxsmm_xcopy_loop_dst_); *libxsmm_xcopy_loop_dst_ = *libxsmm_xcopy_loop_src_; \
      } \
    } \
  } \
  else { /* generic type-size */ \
    for (libxsmm_xcopy_loop_i_ = M0; libxsmm_xcopy_loop_i_ < (libxsmm_blasint)(M1); ++libxsmm_xcopy_loop_i_) { \
      LIBXSMM_PRAGMA_NONTEMPORAL \
      for (libxsmm_xcopy_loop_j_ = N0; libxsmm_xcopy_loop_j_ < (libxsmm_blasint)(N1); ++libxsmm_xcopy_loop_j_) { \
        XKERNEL(TYPE, TYPESIZE, OUT, IN, LDI, LDO, libxsmm_xcopy_loop_i_, libxsmm_xcopy_loop_j_, \
          libxsmm_xcopy_loop_src_, libxsmm_xcopy_loop_dst_); \
        LIBXSMM_MEMCPY127(libxsmm_xcopy_loop_dst_, libxsmm_xcopy_loop_src_, TYPESIZE); \
      } \
    } \
  } \
}

#define LIBXSMM_MZERO_KERNEL_LOOP_NONJIT(TYPE, TYPESIZE, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1) \
  LIBXSMM_XCOPY_LOOP(TYPE, TYPESIZE, XKERNEL, OUT, IN, LDI, LDO, N0, N1, M0, M1)
#define LIBXSMM_MCOPY_KERNEL_LOOP_NONJIT(TYPE, TYPESIZE, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1) \
  LIBXSMM_XCOPY_LOOP(TYPE, TYPESIZE, XKERNEL, OUT, IN, LDI, LDO, N0, N1, M0, M1)
#define LIBXSMM_TCOPY_KERNEL_LOOP_NONJIT(TYPE, TYPESIZE, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1) \
  LIBXSMM_XCOPY_LOOP(TYPE, TYPESIZE, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1)

#define LIBXSMM_XCOPY_NONJIT(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, M0, M1, N0, N1) { \
  switch(TYPESIZE) { \
    case 2: { \
      LIBXSMM_CONCATENATE(XKERNEL,_LOOP_NONJIT)(short, 2, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1); \
    } break; \
    case 4: { \
      LIBXSMM_CONCATENATE(XKERNEL,_LOOP_NONJIT)(float, 4, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1); \
    } break; \
    case 8: { \
      LIBXSMM_CONCATENATE(XKERNEL,_LOOP_NONJIT)(double, 8, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1); \
    } break; \
    case 16: { \
      typedef struct /*libxsmm_xcopy_nonjit_elem_t*/ { double value[2]; } libxsmm_xcopy_nonjit_elem_t; \
      LIBXSMM_CONCATENATE(XKERNEL,_LOOP_NONJIT)(libxsmm_xcopy_nonjit_elem_t, 16, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1); \
    } break; \
    default: { \
      LIBXSMM_CONCATENATE(XKERNEL,_LOOP_NONJIT)(char, TYPESIZE, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1); \
    } break; \
  } \
}

#if 1
# define LIBXSMM_XCOPY_PRECOND(COND)
#else
# define LIBXSMM_XCOPY_PRECOND(COND) COND
#endif

#define LIBXSMM_MZERO_KERNEL_INDEX(I, J) (J)
#define LIBXSMM_MCOPY_KERNEL_INDEX(I, J) (J)
#define LIBXSMM_TCOPY_KERNEL_INDEX(I, J) (I)

#define LIBXSMM_XCOPY(XKERNEL, KERNEL_CALL, KERNEL, OUT, IN, TYPESIZE, LDI, LDO, TILE_M, TILE_N, M0, M1, N0, N1) { \
  libxsmm_blasint libxsmm_xcopy_i_ = M0, libxsmm_xcopy_j_ = N0; \
  LIBXSMM_ASSERT_MSG(0 < (TILE_M) && 0 < (TILE_N), "XCOPY cannot make progress"); \
  if (NULL != (KERNEL)) { /* inner tiles with JIT */ \
    for (; libxsmm_xcopy_i_ < (((libxsmm_blasint)M1) - ((libxsmm_blasint)TILE_M) + 1); libxsmm_xcopy_i_ += TILE_M) { \
      for (libxsmm_xcopy_j_ = N0; libxsmm_xcopy_j_ < (((libxsmm_blasint)N1) - ((libxsmm_blasint)TILE_N) + 1); libxsmm_xcopy_j_ += TILE_N) { \
        XKERNEL(char, TYPESIZE, OUT, IN, LDI, LDO, \
          LIBXSMM_CONCATENATE(XKERNEL,_INDEX)(libxsmm_xcopy_i_, libxsmm_xcopy_j_), \
          LIBXSMM_CONCATENATE(XKERNEL,_INDEX)(libxsmm_xcopy_j_, libxsmm_xcopy_i_), \
          libxsmm_xcopy_src_, libxsmm_xcopy_dst_); \
        KERNEL_CALL(KERNEL, TYPESIZE, libxsmm_xcopy_src_, LDI, libxsmm_xcopy_dst_, LDO); \
      } \
    } \
  } \
  else { /* inner tiles without JIT */ \
    for (; libxsmm_xcopy_i_ < (((libxsmm_blasint)M1) - ((libxsmm_blasint)TILE_M) + 1); libxsmm_xcopy_i_ += TILE_M) { \
      for (libxsmm_xcopy_j_ = N0; libxsmm_xcopy_j_ < (((libxsmm_blasint)N1) - ((libxsmm_blasint)TILE_N) + 1); libxsmm_xcopy_j_ += TILE_N) { \
        LIBXSMM_XCOPY_NONJIT(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, \
          libxsmm_xcopy_i_, libxsmm_xcopy_i_ + (TILE_M), \
          libxsmm_xcopy_j_, libxsmm_xcopy_j_ + (TILE_N)); \
      } \
    } \
  } \
  LIBXSMM_XCOPY_PRECOND(if (libxsmm_xcopy_j_ < ((libxsmm_blasint)N1))) { \
    for (libxsmm_xcopy_i_ = M0; libxsmm_xcopy_i_ < (((libxsmm_blasint)M1) - ((libxsmm_blasint)TILE_M) + 1); libxsmm_xcopy_i_ += TILE_M) { \
      LIBXSMM_XCOPY_NONJIT(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, \
        libxsmm_xcopy_i_, libxsmm_xcopy_i_ + (TILE_M), \
        libxsmm_xcopy_j_, N1); \
    } \
  } \
  LIBXSMM_XCOPY_PRECOND(if (libxsmm_xcopy_i_ < ((libxsmm_blasint)M1))) { \
    for (libxsmm_xcopy_j_ = N0; libxsmm_xcopy_j_ < (((libxsmm_blasint)N1) - ((libxsmm_blasint)TILE_N)); libxsmm_xcopy_j_ += TILE_N) { \
      LIBXSMM_XCOPY_NONJIT(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, \
        libxsmm_xcopy_i_, M1, \
        libxsmm_xcopy_j_, libxsmm_xcopy_j_ + (TILE_N)); \
    } \
  } \
  LIBXSMM_XCOPY_PRECOND(if (libxsmm_xcopy_i_ < ((libxsmm_blasint)M1) && libxsmm_xcopy_j_ < ((libxsmm_blasint)N1))) { \
    LIBXSMM_XCOPY_NONJIT(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, \
      libxsmm_xcopy_i_, M1, \
      libxsmm_xcopy_j_, N1); \
  } \
}


/** Initializes the transpose functionality; NOT thread-safe. */
LIBXSMM_API_INTERN void libxsmm_xcopy_init(int archid);
/** Finalizes the transpose functionality; NOT thread-safe. */
LIBXSMM_API_INTERN void libxsmm_xcopy_finalize(void);

LIBXSMM_API void libxsmm_matcopy_thread_internal(void* out, const void* in, unsigned int typesize,
  unsigned int m, unsigned int n, unsigned int ldi, unsigned int ldo,
  unsigned int km, unsigned int kn, libxsmm_xmcopyfunction kernel,
  int tid, int nthreads);
LIBXSMM_API void libxsmm_otrans_thread_internal(void* out, const void* in, unsigned int typesize,
  unsigned int m, unsigned int n, unsigned int ldi, unsigned int ldo,
  unsigned int km, unsigned int kn, libxsmm_xtransfunction kernel,
  int tid, int nthreads);

LIBXSMM_API_INTERN void libxsmm_matcopy_internal(void* out, const void* in,
  unsigned int typesize, unsigned int ldi, unsigned int ldo,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1,
  unsigned int tm, unsigned int tn, libxsmm_xmcopyfunction kernel);
LIBXSMM_API_INTERN void libxsmm_matzero_internal(void* out, unsigned int typesize, unsigned int ldo,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1,
  unsigned int tm, unsigned int tn, libxsmm_xmcopyfunction kernel);
LIBXSMM_API_INTERN void libxsmm_otrans_internal(void* out, const void* in,
  unsigned int typesize, unsigned int ldi, unsigned int ldo,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1,
  unsigned int tm, unsigned int tn, libxsmm_xtransfunction kernel);

#if (defined(LIBXSMM_XCOPY_JIT) && 0 != (LIBXSMM_XCOPY_JIT))
/** Determines whether JIT-kernels are used or not; values see LIBXSMM_XCOPY_JIT. */
LIBXSMM_APIVAR_PUBLIC(int libxsmm_xcopy_jit);
#endif
/** Determines if OpenMP tasks are used, and scales beyond the number of threads. */
LIBXSMM_APIVAR_PUBLIC(int libxsmm_xcopy_taskscale);
/** Targeted default prefetch */
LIBXSMM_APIVAR_PUBLIC(unsigned int libxsmm_mcopy_prefetch);
/** M-extent of type-size in Byte. */
LIBXSMM_APIVAR_PUBLIC(unsigned int libxsmm_mcopy_mbytes);
LIBXSMM_APIVAR_PUBLIC(unsigned int libxsmm_mzero_mbytes);
LIBXSMM_APIVAR_PUBLIC(unsigned int libxsmm_tcopy_mbytes);
/** M-factor shaping the N-extent. */
LIBXSMM_APIVAR_PUBLIC(float libxsmm_mcopy_nscale);
LIBXSMM_APIVAR_PUBLIC(float libxsmm_mzero_nscale);
LIBXSMM_APIVAR_PUBLIC(float libxsmm_tcopy_nscale);

#endif /*LIBXSMM_XCOPY_H*/

