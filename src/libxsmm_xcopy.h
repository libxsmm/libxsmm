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
#if !defined(LIBXSMM_ITRANS_BUFFER_MAXSIZE)
# if defined(NDEBUG)
#   define LIBXSMM_ITRANS_BUFFER_MAXSIZE (12 << 10/*12kB*/)
# else
#   define LIBXSMM_ITRANS_BUFFER_MAXSIZE 1
# endif
#endif
#if !defined(LIBXSMM_XCOPY_TASKSCALE)
# define LIBXSMM_XCOPY_TASKSCALE 2
#endif
#if !defined(LIBXSMM_XCOPY_TILE_MIN)
# define LIBXSMM_XCOPY_TILE_MIN 2
#endif
#if !defined(LIBXSMM_XCOPY_MELTW) && 1
# define LIBXSMM_XCOPY_MELTW
#endif
/* 0: none, 1: transpose, 2: matcopy, 3: transpose+matcopy */
#if defined(LIBXSMM_PLATFORM_X86)
# if !defined(LIBXSMM_XCOPY_JIT)
#   if (defined(_WIN32) || defined(__CYGWIN__))
#     define LIBXSMM_XCOPY_JIT 0
#   elif defined(NDEBUG)
#     define LIBXSMM_XCOPY_JIT 0
#   else
#     define LIBXSMM_XCOPY_JIT 3
#   endif
# endif
#else
# define LIBXSMM_XCOPY_JIT 0
#endif

/* kernel uses consecutive stores */
#define LIBXSMM_MZERO_KERNEL(TYPE, TYPESIZE, OUT, IN, LDI, LDO, INDEX_I, INDEX_J, SRC, DST) \
  static /*const*/ TYPE libxsmm_mzero_kernel_src_value_ /* zero */; \
  const TYPE *const SRC = &libxsmm_mzero_kernel_src_value_; \
  TYPE *const DST = (TYPE*)(((char*)(OUT)) + (TYPESIZE) * ((size_t)(INDEX_I) * (LDO) + (INDEX_J)))
/* kernel uses consecutive stores and consecutive loads (copy) */
#define LIBXSMM_MCOPY_KERNEL(TYPE, TYPESIZE, OUT, IN, LDI, LDO, INDEX_I, INDEX_J, SRC, DST) \
  const TYPE *const SRC = (const TYPE*)(((const char*) (IN)) + (TYPESIZE) * ((size_t)(INDEX_I) * (LDI) + (INDEX_J))); \
        TYPE *const DST = (      TYPE*)(((      char*)(OUT)) + (TYPESIZE) * ((size_t)(INDEX_I) * (LDO) + (INDEX_J)))

#if defined(LIBXSMM_XCOPY_MELTW)
# define LIBXSMM_MZERO_CALL(KERNEL, TYPESIZE, SRC, LDI, DST, LDO) { \
    libxsmm_meltw_unary_param libxsmm_mzero_call_args_; \
    libxsmm_mzero_call_args_.in.primary = (void*)(SRC); \
    libxsmm_mzero_call_args_.out.primary = (DST); \
    (KERNEL).meltw_zero(&libxsmm_mzero_call_args_); \
    LIBXSMM_UNUSED(LDO); \
  }
# define LIBXSMM_MCOPY_CALL(KERNEL, TYPESIZE, SRC, LDI, DST, LDO) { \
    libxsmm_meltw_unary_param libxsmm_mcopy_call_args_; \
    libxsmm_mcopy_call_args_.in.primary = (void*)(SRC); \
    libxsmm_mcopy_call_args_.out.primary = (DST); \
    (KERNEL).meltw_copy(&libxsmm_mcopy_call_args_); \
    LIBXSMM_UNUSED(LDO); \
  }
# define LIBXSMM_MCOPY_CALL_PF(KERNEL, TYPESIZE, SRC, LDI, DST, LDO) \
    LIBXSMM_MCOPY_CALL(KERNEL, TYPESIZE, SRC, LDI, DST, LDO)
#else
/* call JIT-kernel (matrix-copy with prefetch) */
# define LIBXSMM_MZERO_CALL(KERNEL, TYPESIZE, SRC, LDI, DST, LDO) { \
    const unsigned int libxsmm_mzero_call_uldo_ = (unsigned int)(LDO); \
    (KERNEL).xmcopy(SRC, &libxsmm_mzero_call_uldo_, DST, &libxsmm_mzero_call_uldo_); \
  }
/* call JIT-kernel (matrix-copy) */
# define LIBXSMM_MCOPY_CALL(KERNEL, TYPESIZE, SRC, LDI, DST, LDO) { \
    const unsigned int libxsmm_mcopy_call_nopf_uldi_ = (unsigned int)(LDI); \
    const unsigned int libxsmm_mcopy_call_nopf_uldo_ = (unsigned int)(LDO); \
    (KERNEL).xmcopy(SRC, &libxsmm_mcopy_call_nopf_uldi_, DST, &libxsmm_mcopy_call_nopf_uldo_); \
  }
/* call JIT-kernel (matrix-copy with prefetch) */
# define LIBXSMM_MCOPY_CALL_PF(KERNEL, TYPESIZE, SRC, LDI, DST, LDO) { \
    const unsigned int libxsmm_mcopy_call_uldi_ = (unsigned int)(LDI); \
    const unsigned int libxsmm_mcopy_call_uldo_ = (unsigned int)(LDO); \
    (KERNEL).xmcopy(SRC, &libxsmm_mcopy_call_uldi_, DST, &libxsmm_mcopy_call_uldo_, \
      /*prefetch next line*/((const char*)(SRC)) + (TYPESIZE) * (size_t)(LDI)); \
  }
#endif

/* kernel uses consecutive stores and strided loads (transpose) */
#define LIBXSMM_TCOPY_KERNEL(TYPE, TYPESIZE, OUT, IN, LDI, LDO, INDEX_I, INDEX_J, SRC, DST) \
  const TYPE *const SRC = (const TYPE*)(((const char*) (IN)) + (TYPESIZE) * ((size_t)(INDEX_J) * (LDI) + (INDEX_I))); \
        TYPE *const DST = (      TYPE*)(((      char*)(OUT)) + (TYPESIZE) * ((size_t)(INDEX_I) * (LDO) + (INDEX_J)))

/* call JIT-kernel (transpose) */
#if defined(LIBXSMM_XCOPY_MELTW)
# define LIBXSMM_TCOPY_CALL(KERNEL, TYPESIZE, SRC, LDI, DST, LDO) { \
    libxsmm_meltw_unary_param libxsmm_tcopy_call_args_; \
    libxsmm_tcopy_call_args_.in.primary = (void*)(SRC); \
    libxsmm_tcopy_call_args_.out.primary = (DST); \
    (KERNEL).meltw_trans(&libxsmm_tcopy_call_args_); \
    LIBXSMM_UNUSED(LDO); \
  }
#else
# define LIBXSMM_TCOPY_CALL(KERNEL, TYPESIZE, SRC, LDI, DST, LDO) { \
    const unsigned int libxsmm_tcopy_call_uldi_ = (unsigned int)(LDI); \
    const unsigned int libxsmm_tcopy_call_uldo_ = (unsigned int)(LDO); \
    (KERNEL).xtrans(SRC, &libxsmm_tcopy_call_uldi_, DST, &libxsmm_tcopy_call_uldo_); \
  }
#endif

#define LIBXSMM_XCOPY_LOOP(TYPE, TYPESIZE, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1) { \
  libxsmm_blasint libxsmm_xcopy_loop_i_, libxsmm_xcopy_loop_j_; \
  for (libxsmm_xcopy_loop_i_ = M0; libxsmm_xcopy_loop_i_ < (libxsmm_blasint)(M1); ++libxsmm_xcopy_loop_i_) { \
    LIBXSMM_PRAGMA_NONTEMPORAL(OUT) \
    for (libxsmm_xcopy_loop_j_ = N0; libxsmm_xcopy_loop_j_ < (libxsmm_blasint)(N1); ++libxsmm_xcopy_loop_j_) { \
      XKERNEL(TYPE, TYPESIZE, OUT, IN, LDI, LDO, libxsmm_xcopy_loop_i_, libxsmm_xcopy_loop_j_, \
        libxsmm_xcopy_loop_src_, libxsmm_xcopy_loop_dst_); *libxsmm_xcopy_loop_dst_ = *libxsmm_xcopy_loop_src_; \
    } \
  } \
}

#define LIBXSMM_XCOPY_TILE(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, M0, M1, N0, N1) { \
  switch(TYPESIZE) { \
    case 2: { \
      LIBXSMM_XCOPY_LOOP(short, 2, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1); \
    } break; \
    case 4: { \
      LIBXSMM_XCOPY_LOOP(float, 4, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1); \
    } break; \
    case 8: { \
      LIBXSMM_XCOPY_LOOP(double, 8, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1); \
    } break; \
    case 16: { \
      typedef struct /*libxsmm_xcopy_tile_elem_t*/ { double value[2]; } libxsmm_xcopy_tile_elem_t; \
      LIBXSMM_XCOPY_LOOP(libxsmm_xcopy_tile_elem_t, 16, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1); \
    } break; \
    default: { /* generic type-size */ \
      libxsmm_blasint libxsmm_xcopy_tile_i_, libxsmm_xcopy_tile_j_; \
      for (libxsmm_xcopy_tile_i_ = M0; libxsmm_xcopy_tile_i_ < (libxsmm_blasint)(M1); ++libxsmm_xcopy_tile_i_) { \
        for (libxsmm_xcopy_tile_j_ = N0; libxsmm_xcopy_tile_j_ < (libxsmm_blasint)(N1); ++libxsmm_xcopy_tile_j_) { \
          XKERNEL(char, TYPESIZE, OUT, IN, LDI, LDO, libxsmm_xcopy_tile_i_, libxsmm_xcopy_tile_j_, \
            libxsmm_xcopy_tile_src_, libxsmm_xcopy_tile_dst_); \
          LIBXSMM_MEMCPY127_LOOP(libxsmm_xcopy_tile_dst_, libxsmm_xcopy_tile_src_, TYPESIZE, LIBXSMM_PRAGMA_NONTEMPORAL); \
        } \
      } \
    } \
  } \
}

#define LIBXSMM_ITRANS_LOOP(TYPE, INOUT, LD, M) { \
  libxsmm_blasint libxsmm_itrans_loop_i_, libxsmm_itrans_loop_j_; \
  LIBXSMM_ASSERT(NULL != (INOUT) && (M) <= (LD)); \
  for (libxsmm_itrans_loop_i_ = 0; libxsmm_itrans_loop_i_ < (M); ++libxsmm_itrans_loop_i_) { \
    for (libxsmm_itrans_loop_j_ = 0; libxsmm_itrans_loop_j_ < libxsmm_itrans_loop_i_; ++libxsmm_itrans_loop_j_) { \
      TYPE *const libxsmm_itrans_loop_a_ = ((TYPE*)(INOUT)) + (size_t)(LD) * libxsmm_itrans_loop_i_ + libxsmm_itrans_loop_j_; \
      TYPE *const libxsmm_itrans_loop_b_ = ((TYPE*)(INOUT)) + (size_t)(LD) * libxsmm_itrans_loop_j_ + libxsmm_itrans_loop_i_; \
      LIBXSMM_ISWAP(*libxsmm_itrans_loop_a_, *libxsmm_itrans_loop_b_); \
    } \
  } \
}

#define LIBXSMM_ITRANS(TYPESIZE, INOUT, LD, M) { \
  switch(TYPESIZE) { \
    case 2: { \
      LIBXSMM_ITRANS_LOOP(short, INOUT, LD, M); \
    } break; \
    case 4: { \
      LIBXSMM_ITRANS_LOOP(int, INOUT, LD, M); \
    } break; \
    case 8: { \
      LIBXSMM_ITRANS_LOOP(int64_t, INOUT, LD, M); \
    } break; \
    default: { /* generic type-size */ \
      const signed char libxsmm_itrans_c_ = (signed char)(TYPESIZE); \
      libxsmm_blasint libxsmm_itrans_i_, libxsmm_itrans_j_; \
      LIBXSMM_ASSERT(NULL != (INOUT) && (M) <= (LD)); \
      LIBXSMM_ASSERT(0 < (TYPESIZE) && (TYPESIZE) <= 127); \
      for (libxsmm_itrans_i_ = 0; libxsmm_itrans_i_ < (M); ++libxsmm_itrans_i_) { \
        for (libxsmm_itrans_j_ = 0; libxsmm_itrans_j_ < libxsmm_itrans_i_; ++libxsmm_itrans_j_) { \
          char *const libxsmm_itrans_a_ = &((char*)(INOUT))[((LD)*libxsmm_itrans_i_+libxsmm_itrans_j_)*(TYPESIZE)]; \
          char *const libxsmm_itrans_b_ = &((char*)(INOUT))[((LD)*libxsmm_itrans_j_+libxsmm_itrans_i_)*(TYPESIZE)]; \
          signed char libxsmm_itrans_k_ = 0; \
          for (; libxsmm_itrans_k_ < libxsmm_itrans_c_; ++libxsmm_itrans_k_) { \
            LIBXSMM_ISWAP( \
              libxsmm_itrans_a_[libxsmm_itrans_k_], \
              libxsmm_itrans_b_[libxsmm_itrans_k_]); \
          } \
        } \
      } \
    } \
  } \
}

#define LIBXSMM_MZERO_KERNEL_TILE(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, M0, M1, N0, N1) \
  LIBXSMM_XCOPY_TILE(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, N0, N1, M0, M1)
#define LIBXSMM_MCOPY_KERNEL_TILE(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, M0, M1, N0, N1) \
  LIBXSMM_XCOPY_TILE(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, N0, N1, M0, M1)
#define LIBXSMM_TCOPY_KERNEL_TILE(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, M0, M1, N0, N1) \
  LIBXSMM_XCOPY_TILE(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, M0, M1, N0, N1)

#define LIBXSMM_XCOPY_NONJIT(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, M0, M1, N0, N1) \
  LIBXSMM_CONCATENATE(XKERNEL,_TILE)(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, M0, M1, N0, N1)

#if 1
# define LIBXSMM_XCOPY_PRECOND(COND)
#else
# define LIBXSMM_XCOPY_PRECOND(COND) COND
#endif

#define LIBXSMM_XCOPY_TILES(XKERNEL, KERNEL_CALL, KERNEL, OUT, IN, TYPESIZE, LDI, LDO, TILE_M, TILE_N, M0, M1, N0, N1) { \
  libxsmm_blasint libxsmm_xcopy_i_ = M0, libxsmm_xcopy_j_ = N0; \
  LIBXSMM_ASSERT_MSG(0 < (TILE_M) && 0 < (TILE_N), "XCOPY cannot make progress"); \
  if (NULL != (KERNEL).ptr) { /* inner tiles with JIT */ \
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
        LIBXSMM_XCOPY_TILE(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, \
          libxsmm_xcopy_i_, libxsmm_xcopy_i_ + (TILE_M), \
          libxsmm_xcopy_j_, libxsmm_xcopy_j_ + (TILE_N)); \
      } \
    } \
  } \
  { /* remainder/border tiles */ \
    LIBXSMM_XCOPY_PRECOND(if (libxsmm_xcopy_j_ < ((libxsmm_blasint)N1))) { \
      for (libxsmm_xcopy_i_ = M0; libxsmm_xcopy_i_ < (((libxsmm_blasint)M1) - ((libxsmm_blasint)TILE_M) + 1); libxsmm_xcopy_i_ += TILE_M) { \
        LIBXSMM_XCOPY_TILE(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, \
          libxsmm_xcopy_i_, libxsmm_xcopy_i_ + (TILE_M), \
          libxsmm_xcopy_j_, N1); \
      } \
    } \
    LIBXSMM_XCOPY_PRECOND(if (libxsmm_xcopy_i_ < ((libxsmm_blasint)M1))) { \
      for (libxsmm_xcopy_j_ = N0; libxsmm_xcopy_j_ < (((libxsmm_blasint)N1) - ((libxsmm_blasint)TILE_N)); libxsmm_xcopy_j_ += TILE_N) { \
        LIBXSMM_XCOPY_TILE(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, \
          libxsmm_xcopy_i_, M1, \
          libxsmm_xcopy_j_, libxsmm_xcopy_j_ + (TILE_N)); \
      } \
    } \
    LIBXSMM_XCOPY_PRECOND(if (libxsmm_xcopy_i_ < ((libxsmm_blasint)M1) && libxsmm_xcopy_j_ < ((libxsmm_blasint)N1))) { \
      LIBXSMM_XCOPY_TILE(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, \
        libxsmm_xcopy_i_, M1, \
        libxsmm_xcopy_j_, N1); \
    } \
  } \
}

#define LIBXSMM_MZERO_KERNEL_TILES(XKERNEL, KERNEL_CALL, KERNEL, OUT, IN, TYPESIZE, LDI, LDO, TILE_M, TILE_N, M0, M1, N0, N1) \
  LIBXSMM_XCOPY_TILES(XKERNEL, KERNEL_CALL, KERNEL, OUT, IN, TYPESIZE, LDI, LDO, TILE_N, TILE_M, N0, N1, M0, M1)
#define LIBXSMM_MCOPY_KERNEL_TILES(XKERNEL, KERNEL_CALL, KERNEL, OUT, IN, TYPESIZE, LDI, LDO, TILE_M, TILE_N, M0, M1, N0, N1) \
  LIBXSMM_XCOPY_TILES(XKERNEL, KERNEL_CALL, KERNEL, OUT, IN, TYPESIZE, LDI, LDO, TILE_N, TILE_M, N0, N1, M0, M1)
#define LIBXSMM_TCOPY_KERNEL_TILES(XKERNEL, KERNEL_CALL, KERNEL, OUT, IN, TYPESIZE, LDI, LDO, TILE_M, TILE_N, M0, M1, N0, N1) \
  LIBXSMM_XCOPY_TILES(XKERNEL, KERNEL_CALL, KERNEL, OUT, IN, TYPESIZE, LDI, LDO, TILE_M, TILE_N, M0, M1, N0, N1)

#define LIBXSMM_XCOPY(XKERNEL, KERNEL_CALL, KERNEL, OUT, IN, TYPESIZE, LDI, LDO, TILE_M, TILE_N, M0, M1, N0, N1) \
  LIBXSMM_CONCATENATE(XKERNEL,_TILES)(XKERNEL, KERNEL_CALL, KERNEL, OUT, IN, TYPESIZE, LDI, LDO, TILE_M, TILE_N, M0, M1, N0, N1)

LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE libxsmm_xcopykernel {
  libxsmm_meltwfunction_unary meltw_trans, meltw_copy, meltw_zero;
  libxsmm_xmcopyfunction xmcopy;
  libxsmm_xtransfunction xtrans;
  const void* ptr;
} libxsmm_xcopykernel;

/** Initializes the transpose functionality; NOT thread-safe. */
LIBXSMM_API_INTERN void libxsmm_xcopy_init(int archid);
/** Finalizes the transpose functionality; NOT thread-safe. */
LIBXSMM_API_INTERN void libxsmm_xcopy_finalize(void);

LIBXSMM_API void libxsmm_matcopy_task_internal(void* out, const void* in, unsigned int typesize,
  unsigned int m, unsigned int n, unsigned int ldi, unsigned int ldo,
  unsigned int km, unsigned int kn, libxsmm_xcopykernel kernel,
  int tid, int ntasks);
LIBXSMM_API void libxsmm_otrans_task_internal(void* out, const void* in, unsigned int typesize,
  unsigned int m, unsigned int n, unsigned int ldi, unsigned int ldo,
  unsigned int km, unsigned int kn, libxsmm_xcopykernel kernel,
  int tid, int ntasks);

LIBXSMM_API_INTERN void libxsmm_matcopy_internal(void* out, const void* in,
  unsigned int typesize, unsigned int ldi, unsigned int ldo,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1,
  unsigned int tm, unsigned int tn, libxsmm_xcopykernel kernel);
LIBXSMM_API_INTERN void libxsmm_matzero_internal(void* out,
  unsigned int typesize, unsigned int ldo,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1,
  unsigned int tm, unsigned int tn, libxsmm_xcopykernel kernel);
LIBXSMM_API_INTERN void libxsmm_otrans_internal(void* out, const void* in,
  unsigned int typesize, unsigned int ldi, unsigned int ldo,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1,
  unsigned int tm, unsigned int tn, libxsmm_xcopykernel kernel);
LIBXSMM_API void libxsmm_itrans_internal(char* inout, void* scratch, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  libxsmm_blasint index_base, libxsmm_blasint index_stride, const libxsmm_blasint stride[],
  libxsmm_xcopykernel kernel, libxsmm_blasint begin, libxsmm_blasint end);

#if (defined(LIBXSMM_XCOPY_JIT) && 0 != (LIBXSMM_XCOPY_JIT))
/** Determines whether JIT-kernels are used or not; values see LIBXSMM_XCOPY_JIT. */
LIBXSMM_APIVAR_PUBLIC(int libxsmm_xcopy_jit);
# if !defined(LIBXSMM_XCOPY_MELTW)
/** Targeted default prefetch */
LIBXSMM_APIVAR_PUBLIC(unsigned int libxsmm_mcopy_prefetch);
# endif
#endif
/** Determines if OpenMP tasks are used, and scales beyond the number of threads. */
LIBXSMM_APIVAR_PUBLIC(int libxsmm_xcopy_taskscale);
/** M-extent of type-size in Byte. */
LIBXSMM_APIVAR_PUBLIC(unsigned int libxsmm_mcopy_mbytes);
LIBXSMM_APIVAR_PUBLIC(unsigned int libxsmm_mzero_mbytes);
LIBXSMM_APIVAR_PUBLIC(unsigned int libxsmm_tcopy_mbytes);
/** M-factor shaping the N-extent. */
LIBXSMM_APIVAR_PUBLIC(float libxsmm_mcopy_nscale);
LIBXSMM_APIVAR_PUBLIC(float libxsmm_mzero_nscale);
LIBXSMM_APIVAR_PUBLIC(float libxsmm_tcopy_nscale);

#endif /*LIBXSMM_XCOPY_H*/

