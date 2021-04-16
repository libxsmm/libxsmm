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
#include "libxsmm_xcopy.h"
#include "libxsmm_ext.h"

#define LIBXSMM_MCOPY_MT(MT, NT, M, N) ((MT) <= (M) && (NT) <= (N) && (64U * 64U) <= (((unsigned int)(M)) * (N)))


LIBXSMM_APIEXT void libxsmm_matcopy_omp(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  LIBXSMM_INIT
  if (0 < typesize && 256 > typesize && m <= ldi && m <= ldo && out != in &&
    ((NULL != out && 0 < m && 0 < n) || (0 == m && 0 == n)))
  {
    if (0 < m && 0 < n) {
#if defined(_OPENMP)
      unsigned int tm, tn, ts;
      if (NULL != in) { /* mcopy */
        tm = LIBXSMM_UPDIV(libxsmm_mcopy_mbytes, typesize);
        tn = (unsigned int)(libxsmm_mcopy_nscale * tm);
        ts = libxsmm_mcopy_mbytes;
      }
      else { /* mzero */
        tm = LIBXSMM_UPDIV(libxsmm_mzero_mbytes, typesize);
        tn = (unsigned int)(libxsmm_mzero_nscale * tm);
        ts = libxsmm_mzero_mbytes;
      }
      if (0 == tm) tm = m;
      if (0 == tn) tn = LIBXSMM_MIN(LIBXSMM_XCOPY_TILE_MIN, n);
      if (0 != ts && ts < (tm * tn * typesize)) {
        tm = LIBXSMM_MAX(ts / (tn * typesize), LIBXSMM_XCOPY_TILE_MIN);
      }
      if (LIBXSMM_MCOPY_MT(tm, tn, (unsigned int)m, (unsigned int)n)) { /* consider problem-size */
        libxsmm_xcopykernel kernel;
        kernel.ptr = NULL;
# if (defined(LIBXSMM_XCOPY_JIT) && 0 != (LIBXSMM_XCOPY_JIT & 2))
        if (0 != (2 & libxsmm_xcopy_jit)) { /* JIT'ted matrix-copy permitted? */
          switch (typesize) {
            case 8: kernel.meltw_copy = libxsmm_dispatch_meltw_unary(tm, tn, &ldi, &ldo,
              LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_MELTW_FLAG_UNARY_NONE,
              NULL != in ? LIBXSMM_MELTW_TYPE_UNARY_IDENTITY/*mcopy*/ : LIBXSMM_MELTW_TYPE_UNARY_XOR/*mzero*/);
              break;
            case 4: kernel.meltw_copy = libxsmm_dispatch_meltw_unary(tm, tn, &ldi, &ldo,
              LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE,
              NULL != in ? LIBXSMM_MELTW_TYPE_UNARY_IDENTITY/*mcopy*/ : LIBXSMM_MELTW_TYPE_UNARY_XOR/*mzero*/);
              break;
            case 2: kernel.meltw_copy = libxsmm_dispatch_meltw_unary(tm, tn, &ldi, &ldo,
              LIBXSMM_DATATYPE_I16, LIBXSMM_DATATYPE_I16, LIBXSMM_DATATYPE_I16, LIBXSMM_MELTW_FLAG_UNARY_NONE,
              NULL != in ? LIBXSMM_MELTW_TYPE_UNARY_IDENTITY/*mcopy*/ : LIBXSMM_MELTW_TYPE_UNARY_XOR/*mzero*/);
              break;
            case 1: kernel.meltw_copy = libxsmm_dispatch_meltw_unary(tm, tn, &ldi, &ldo,
              LIBXSMM_DATATYPE_I8, LIBXSMM_DATATYPE_I8, LIBXSMM_DATATYPE_I8, LIBXSMM_MELTW_FLAG_UNARY_NONE,
              NULL != in ? LIBXSMM_MELTW_TYPE_UNARY_IDENTITY/*mcopy*/ : LIBXSMM_MELTW_TYPE_UNARY_XOR/*mzero*/);
              break;
          }
        }
# endif
# if defined(LIBXSMM_EXT_TASKS) && 0/* implies _OPENMP */
        if (0 == omp_get_active_level())
# else
        if (0 == omp_in_parallel())
# endif
        { /* enable internal parallelization */
          const int nthreads = omp_get_max_threads();
# if defined(LIBXSMM_EXT_TASKS)
          if (0 >= libxsmm_xcopy_taskscale)
# endif
          {
#           pragma omp parallel num_threads(nthreads)
            libxsmm_matcopy_task_internal(out, in, typesize,
              (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo,
              tm, tn, kernel, omp_get_thread_num(), nthreads);
          }
# if defined(LIBXSMM_EXT_TASKS)
          else { /* tasks requested */
            const int ntasks = nthreads * libxsmm_xcopy_taskscale;
#           pragma omp parallel num_threads(nthreads)
            { /* first thread discovering work will launch all tasks */
#             pragma omp single nowait /* anyone is good */
              { int tid;
                for (tid = 0; tid < ntasks; ++tid) {
#                 pragma omp task untied
                  libxsmm_matcopy_task_internal(out, in, typesize,
                    (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo,
                    tm, tn, kernel, tid, ntasks);
                }
              }
            }
          }
# endif
        }
        else { /* assume external parallelization */
# if defined(LIBXSMM_EXT_TASKS) /* implies _OPENMP */
          const int nthreads = omp_get_num_threads();
          const int ntasks = (0 == libxsmm_xcopy_taskscale
            ? (LIBXSMM_XCOPY_TASKSCALE)
            : libxsmm_xcopy_taskscale) * nthreads;
          int tid;
          for (tid = 0; tid < ntasks; ++tid) {
#           pragma omp task untied
            libxsmm_matcopy_task_internal(out, in, typesize,
              (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo,
              tm, tn, kernel, tid, ntasks);
          }
          if (0 == libxsmm_nosync) { /* allow to omit synchronization */
#           pragma omp taskwait
          }
# else
          libxsmm_matcopy_task_internal(out, in, typesize,
            (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo,
            tm, tn, kernel, 0/*tid*/, 1/*nthreads*/);
# endif
        }
      }
      else
#endif /*defined(_OPENMP)*/
      if (NULL != in) { /* no MT, or small problem-size */
        LIBXSMM_XCOPY_NONJIT(LIBXSMM_MCOPY_KERNEL,
          typesize, out, in, ldi, ldo, 0, m, 0, n);
      }
      else { /* no MT, or small problem-size */
        /* coverity[ptr_arith] */
        LIBXSMM_XCOPY_NONJIT(LIBXSMM_MZERO_KERNEL,
          typesize, out, in, ldi, ldo, 0, m, 0, n);
      }
    }
  }
  else {
    static int error_once = 0;
    if ( 0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      if (NULL == out) {
        fprintf(stderr, "LIBXSMM ERROR: the matrix-copy input and/or output is NULL!\n");
      }
      else if (out == in) {
        fprintf(stderr, "LIBXSMM ERROR: output and input of the matrix-copy must be different!\n");
      }
      else if (0 == typesize || 256 <= typesize) {
        fprintf(stderr, "LIBXSMM ERROR: invalid type-size for matrix-copy specified!\n");
      }
      else if (ldi < m || ldo < m) {
        fprintf(stderr, "LIBXSMM ERROR: the leading dimension(s) of the matrix-copy is/are too small!\n");
      }
      else if (0 > m || 0 > n) {
        fprintf(stderr, "LIBXSMM ERROR: the matrix extent(s) of the matrix-copy is/are negative!\n");
      }
    }
  }
}


LIBXSMM_APIEXT void libxsmm_otrans_omp(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  static int error_once = 0;
  LIBXSMM_INIT
  if (0 < typesize && 256 > typesize && m <= ldi && n <= ldo &&
    ((NULL != out && NULL != in && 0 < m && 0 < n) || (0 == m && 0 == n)))
  {
    if (0 < m && 0 < n) {
      if (out != in) {
#if defined(_OPENMP)
        unsigned int tm = LIBXSMM_UPDIV(libxsmm_tcopy_mbytes, typesize);
        unsigned int tn = (unsigned int)(libxsmm_tcopy_nscale * tm);
        if (0 == tm) tm = m;
        if (0 == tn) tn = LIBXSMM_MIN(LIBXSMM_XCOPY_TILE_MIN, n);
        if (0 != libxsmm_tcopy_mbytes && libxsmm_tcopy_mbytes < (tm * tn * typesize)) {
          tm = LIBXSMM_MAX(libxsmm_tcopy_mbytes / (tn * typesize), LIBXSMM_XCOPY_TILE_MIN);
        }
        if (tm <= (unsigned int)m && tn <= (unsigned int)n) { /* consider problem-size */
          libxsmm_xcopykernel kernel;
          kernel.ptr = NULL;
# if defined(LIBXSMM_EXT_TASKS) /* implies _OPENMP */
          if (0 == omp_get_active_level())
# else
          if (0 == omp_in_parallel())
# endif
          { /* enable internal parallelization */
            const int nthreads = omp_get_max_threads();
# if defined(LIBXSMM_EXT_TASKS)
            if (0 >= libxsmm_xcopy_taskscale)
# endif
            {
#             pragma omp parallel num_threads(nthreads)
              { /* coverity[divide_by_zero] */
                libxsmm_otrans_task_internal(out, in, typesize,
                  (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo,
                  tm, tn, kernel, omp_get_thread_num(), nthreads);
              }
            }
# if defined(LIBXSMM_EXT_TASKS)
            else { /* tasks requested */
              const int ntasks = nthreads * libxsmm_xcopy_taskscale;
#             pragma omp parallel num_threads(nthreads)
              { /* first thread discovering work will launch all tasks */
#               pragma omp single nowait /* anyone is good */
                { int tid;
                  for (tid = 0; tid < ntasks; ++tid) {
#                   pragma omp task untied
                    libxsmm_otrans_task_internal(out, in, typesize,
                      (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo,
                      tm, tn, kernel, tid, ntasks);
                  }
                }
              }
            }
# endif
          }
          else { /* assume external parallelization */
# if defined(LIBXSMM_EXT_TASKS) /* implies _OPENMP */
            const int nthreads = omp_get_num_threads();
            const int ntasks = (0 == libxsmm_xcopy_taskscale
              ? (LIBXSMM_XCOPY_TASKSCALE)
              : libxsmm_xcopy_taskscale) * nthreads;
            int tid;
            for (tid = 0; tid < ntasks; ++tid) {
#             pragma omp task untied
              libxsmm_otrans_task_internal(out, in, typesize,
                (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo,
                tm, tn, kernel, tid, ntasks);
            }
            if (0 == libxsmm_nosync) { /* allow to omit synchronization */
#             pragma omp taskwait
            }
# else    /* coverity[divide_by_zero] */
            libxsmm_otrans_task_internal(out, in, typesize,
              (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo,
              tm, tn, kernel, 0/*tid*/, 1/*nthreads*/);
# endif
          }
        }
        else
#endif /*defined(_OPENMP)*/
        { /* no MT, or small problem-size */
#if (defined(LIBXSMM_XCOPY_JIT) && 0 != (LIBXSMM_XCOPY_JIT & 1))
          libxsmm_xcopykernel kernel;
          kernel.ptr = NULL;
          if (0 != (1 & libxsmm_xcopy_jit)) { /* JIT'ted transpose permitted? */
            switch (typesize) {
              case 8: kernel.meltw_trans = libxsmm_dispatch_meltw_unary(m, n, &ldi, &ldo,
                LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64,
                LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
                break;
              case 4: kernel.meltw_trans = libxsmm_dispatch_meltw_unary(m, n, &ldi, &ldo,
                LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
                LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
                break;
              case 2: kernel.meltw_trans = libxsmm_dispatch_meltw_unary(m, n, &ldi, &ldo,
                LIBXSMM_DATATYPE_I16, LIBXSMM_DATATYPE_I16, LIBXSMM_DATATYPE_I16,
                LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
                break;
              case 1: kernel.meltw_trans = libxsmm_dispatch_meltw_unary(m, n, &ldi, &ldo,
                LIBXSMM_DATATYPE_I8, LIBXSMM_DATATYPE_I8, LIBXSMM_DATATYPE_I8,
                LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
                break;
            }
            if (NULL != kernel.ptr) { /* JIT-kernel available */
              LIBXSMM_TCOPY_CALL(kernel, typesize, in, ldi, out, ldo);
            }
          }
          else
#endif
          {
            LIBXSMM_XCOPY_NONJIT(LIBXSMM_TCOPY_KERNEL,
              typesize, out, in, ldi, ldo, 0, m, 0, n);
          }
        }
      }
      else if (ldi == ldo) {
        libxsmm_itrans/*TODO: omp*/(out, typesize, m, n, ldi, ldo);
      }
      else if (0 != libxsmm_verbosity /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM ERROR: output and input of the transpose must be different!\n");
      }
    }
  }
  else {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
     && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      if (NULL == out || NULL == in) {
        fprintf(stderr, "LIBXSMM ERROR: the transpose input and/or output is NULL!\n");
      }
      else if (out == in) {
        fprintf(stderr, "LIBXSMM ERROR: output and input of the transpose must be different!\n");
      }
      else if (0 == typesize || 256 <= typesize) {
        fprintf(stderr, "LIBXSMM ERROR: invalid type-size for matrix-transpose specified!\n");
      }
      else if (ldi < m || ldo < n) {
        fprintf(stderr, "LIBXSMM ERROR: the leading dimension(s) of the transpose is/are too small!\n");
      }
      else if (0 > m || 0 > n) {
        fprintf(stderr, "LIBXSMM ERROR: the matrix extent(s) of the transpose is/are negative!\n");
      }
    }
  }
}


LIBXSMM_APIEXT void libxsmm_itrans_batch_omp(void* inout, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride[], libxsmm_blasint batchsize)
{
#if defined(_OPENMP)
  if (1 < batchsize) { /* consider problem-size */
    const libxsmm_blasint scratchsize = m * n * typesize;
    const libxsmm_blasint size = LIBXSMM_ABS(batchsize);
    char buffer[LIBXSMM_ITRANS_BUFFER_MAXSIZE];
    char *const mat0 = (char*)inout;
    void* scratch = NULL;
    libxsmm_xcopykernel kernel = { NULL };
    if (m != n || ldi != ldo || 127 < typesize) {
      if (scratchsize <= LIBXSMM_ITRANS_BUFFER_MAXSIZE) {
        scratch = buffer;
      }
      else {
        static int error_once = 0;
        LIBXSMM_INIT
        if (EXIT_SUCCESS != libxsmm_xmalloc(&scratch, scratchsize, 0/*auto-align*/,
            LIBXSMM_MALLOC_FLAG_SCRATCH | LIBXSMM_MALLOC_FLAG_PRIVATE,
            0/*extra*/, 0/*extra_size*/)
          && 0 != libxsmm_verbosity /* library code is expected to be mute */
          && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXSMM ERROR: failed to allocate buffer for in-place transpose!\n");
        }
      }
#if (defined(LIBXSMM_XCOPY_JIT) && 0 != (LIBXSMM_XCOPY_JIT & 1))
      if (0 != (1 & libxsmm_xcopy_jit) /* JIT'ted transpose permitted? */
        /* avoid outgrown transpose kernel upfront */
        && (m <= LIBXSMM_CONFIG_MAX_DIM || n <= LIBXSMM_CONFIG_MAX_DIM))
      {
        switch (typesize) {
          case 8: kernel.meltw_trans = libxsmm_dispatch_meltw_unary(m, n, &ldi, &ldo,
            LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64,
            LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
            break;
          case 4: kernel.meltw_trans = libxsmm_dispatch_meltw_unary(m, n, &ldi, &ldo,
            LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
            break;
          case 2: kernel.meltw_trans = libxsmm_dispatch_meltw_unary(m, n, &ldi, &ldo,
            LIBXSMM_DATATYPE_I16, LIBXSMM_DATATYPE_I16, LIBXSMM_DATATYPE_I16,
            LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
            break;
          case 1: kernel.meltw_trans = libxsmm_dispatch_meltw_unary(m, n, &ldi, &ldo,
            LIBXSMM_DATATYPE_I8, LIBXSMM_DATATYPE_I8, LIBXSMM_DATATYPE_I8,
            LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
            break;
        }
      }
#endif
    }
# if defined(LIBXSMM_EXT_TASKS) && 0/* implies _OPENMP */
    if (0 == omp_get_active_level())
# else
    if (0 == omp_in_parallel())
# endif
    { /* enable internal parallelization */
      const int nthreads = omp_get_max_threads();
# if defined(LIBXSMM_EXT_TASKS)
      if (0 >= libxsmm_xcopy_taskscale)
# endif
      {
        const libxsmm_blasint tasksize = LIBXSMM_UPDIV(size, nthreads);
#       pragma omp parallel num_threads(nthreads)
        {
          const libxsmm_blasint begin = omp_get_thread_num() * tasksize;
          const libxsmm_blasint span = begin + tasksize;
          libxsmm_itrans_internal(mat0, scratch, typesize, m, n, ldi, ldo, index_base,
            index_stride, stride, kernel, begin, LIBXSMM_MIN(span, size));
        }
      }
# if defined(LIBXSMM_EXT_TASKS)
      else { /* tasks requested */
        const int ntasks = nthreads * libxsmm_xcopy_taskscale;
        const libxsmm_blasint tasksize = LIBXSMM_UPDIV(size, ntasks);
#       pragma omp parallel num_threads(nthreads)
        { /* first thread discovering work will launch all tasks */
#         pragma omp single nowait /* anyone is good */
          { int tid;
            for (tid = 0; tid < ntasks; ++tid) {
              const libxsmm_blasint begin = tid * tasksize;
              const libxsmm_blasint span = begin + tasksize;
#             pragma omp task untied
              libxsmm_itrans_internal(mat0, scratch, typesize, m, n, ldi, ldo, index_base,
                index_stride, stride, kernel, begin, LIBXSMM_MIN(span, size));
            }
          }
        }
      }
# endif
    }
    else { /* assume external parallelization */
# if defined(LIBXSMM_EXT_TASKS) /* implies _OPENMP */
      const int nthreads = omp_get_num_threads();
      const int ntasks = (0 == libxsmm_xcopy_taskscale
        ? (LIBXSMM_XCOPY_TASKSCALE)
        : libxsmm_xcopy_taskscale) * nthreads;
      const libxsmm_blasint tasksize = LIBXSMM_UPDIV(size, ntasks);
      int tid;
      for (tid = 0; tid < ntasks; ++tid) {
        const libxsmm_blasint begin = tid * tasksize;
        const libxsmm_blasint span = begin + tasksize;
#       pragma omp task untied
        libxsmm_itrans_internal(mat0, scratch, typesize, m, n, ldi, ldo, index_base,
          index_stride, stride, kernel, begin, LIBXSMM_MIN(span, size));
      }
      if (0 == libxsmm_nosync) { /* allow to omit synchronization */
#       pragma omp taskwait
      }
# else
      libxsmm_itrans_internal(mat0, scratch, typesize, m, n, ldi, ldo, index_base,
        index_stride, stride, kernel, 0, batchsize);
# endif
    }
    if (NULL != scratch && LIBXSMM_ITRANS_BUFFER_MAXSIZE < scratchsize) {
      libxsmm_xfree(scratch, 0/*no check*/);
    }
  }
  else
#endif /*defined(_OPENMP)*/
  libxsmm_itrans_batch(inout, typesize, m, n, ldi, ldo,
    index_base, index_stride, stride, batchsize,
    0/*tid*/, 1/*ntasks*/);
}


#if defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT) && (!defined(LIBXSMM_NOFORTRAN) || defined(__clang_analyzer__))

/* implementation provided for Fortran 77 compatibility */
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_matcopy_omp)(void* /*out*/, const void* /*in*/, const int* /*typesize*/,
  const libxsmm_blasint* /*m*/, const libxsmm_blasint* /*n*/, const libxsmm_blasint* /*ldi*/, const libxsmm_blasint* /*ldo*/);
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_matcopy_omp)(void* out, const void* in, const int* typesize,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* ldi, const libxsmm_blasint* ldo)
{
  libxsmm_blasint ldx;
  LIBXSMM_ASSERT(NULL != typesize && 0 < *typesize && NULL != m);
  ldx = *(NULL != ldi ? ldi : m);
  libxsmm_matcopy_omp(out, in, (unsigned int)*typesize, *m, *(NULL != n ? n : m), ldx, NULL != ldo ? *ldo : ldx);
}



/* implementation provided for Fortran 77 compatibility */
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_otrans_omp)(void* /*out*/, const void* /*in*/, const int* /*typesize*/,
  const libxsmm_blasint* /*m*/, const libxsmm_blasint* /*n*/, const libxsmm_blasint* /*ldi*/, const libxsmm_blasint* /*ldo*/);
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_otrans_omp)(void* out, const void* in, const int* typesize,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* ldi, const libxsmm_blasint* ldo)
{
  libxsmm_blasint ldx;
  LIBXSMM_ASSERT(NULL != typesize && 0 < *typesize && NULL != m);
  ldx = *(NULL != ldi ? ldi : m);
  libxsmm_otrans_omp(out, in, (unsigned int)*typesize, *m, *(NULL != n ? n : m), ldx, NULL != ldo ? *ldo : ldx);
}

#endif /*defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT) && (!defined(LIBXSMM_NOFORTRAN) || defined(__clang_analyzer__))*/

