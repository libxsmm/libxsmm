/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include <utils/libxsmm_intrinsics_x86.h>
#include <libxsmm.h>

#if !defined(CHECK_SETUP) && 1
# define CHECK_SETUP
#endif
#if !defined(CHECK_REALLOC) && 1
# define CHECK_REALLOC
#endif
#if !defined(CHECK_SCRATCH) && 1
# define CHECK_SCRATCH
#endif
#if !defined(CHECK_PMALLOC) && 1
# define CHECK_PMALLOC
#endif

#if defined(_DEBUG)
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif


int main(void)
{
  const size_t size_malloc = 2507, alignment = (2U << 20);
  libxsmm_malloc_info malloc_info;
  int avalue, nerrors = 0;
#if defined(CHECK_SCRATCH)
  const size_t size_scratch = (24U << 20);
  void *q, *r;
#endif
  void* s;

#if defined(CHECK_PMALLOC)
  { /* check pooled malloc */
    void* pool[128];
    char storage[8*sizeof(pool)/sizeof(*pool)];
    void* backup[sizeof(pool)];
    const int npool = sizeof(pool) / sizeof(*pool);
    size_t num = npool;
    int i, j;

    libxsmm_pmalloc_init(sizeof(storage) / num, &num, pool, storage);
    memcpy(backup, pool, sizeof(pool));
    for (i = 0; i < 1024; ++i) {
# if defined(_OPENMP)
#     pragma omp parallel for private(j) schedule(dynamic,1)
# endif
      for (j = 0; j < npool; ++j) {
        void *const p = libxsmm_pmalloc(pool, &num);
        libxsmm_pfree(p, pool, &num);
      }
      if (npool != (int)num) break;
    }
    if (npool == (int)num) {
      for (i = 0, num = 0; i < npool; ++i) {
        const void *const p = backup[i];
        for (j = 0; j < npool; ++j) {
          if (p == pool[j]) {
            ++num; break;
          }
        }
      }
      nerrors += LIBXSMM_DELTA(npool, (int)num);
    }
    else ++nerrors;
  }
  if (0 != nerrors) FPRINTF(stderr, "Error: incorrect pmalloc!\n");
#endif

#if defined(CHECK_SETUP)
  { /* check allocator setup */
    libxsmm_malloc_function default_malloc_fn, malloc_fn;
    libxsmm_free_function default_free_fn, free_fn;
    const void *default_context, *context;

    /* determine default allocation functions and context */
    if (EXIT_SUCCESS != libxsmm_get_default_allocator(&default_context/*context*/, &default_malloc_fn, &default_free_fn)) {
      ++nerrors;
    }
    /* set specific functions for default allocations and check adoption */
    malloc_fn.function = malloc; free_fn.function = free;
    if  (EXIT_SUCCESS != libxsmm_set_default_allocator(NULL/*context*/, malloc_fn/*malloc*/, free_fn/*free*/)
      || EXIT_SUCCESS != libxsmm_get_default_allocator(&context, &malloc_fn, &free_fn)
      || NULL != context || malloc != malloc_fn.function || free != free_fn.function)
    {
      ++nerrors;
    }
    /* check adoption/inheritance from the default allocator */
    malloc_fn.function = NULL; free_fn.function = NULL;
    if  (EXIT_SUCCESS != libxsmm_set_scratch_allocator(NULL/*context*/, malloc_fn/*NULL*/, free_fn/*NULL*/)
      || EXIT_SUCCESS != libxsmm_get_scratch_allocator(&context, &malloc_fn, &free_fn)
      || NULL != context || malloc != malloc_fn.function || free != free_fn.function)
    {
      ++nerrors;
    }
    /* reset default allocator */
    if (EXIT_SUCCESS != libxsmm_set_default_allocator(default_context, default_malloc_fn, default_free_fn)) {
      ++nerrors;
    }
  }
  if (0 != nerrors) FPRINTF(stderr, "Error: incorrect allocator setup!\n");
#endif

  /* allocate some amount of memory */
  s = libxsmm_malloc(size_malloc);

  /* query and check the size of the buffer */
  if (NULL != s && (EXIT_SUCCESS != libxsmm_get_malloc_info(s, &malloc_info) || malloc_info.size < size_malloc)) {
    FPRINTF(stderr, "Error: buffer info (1/4) failed!\n");
    ++nerrors;
  }

#if defined(CHECK_SCRATCH)
  q = libxsmm_aligned_scratch(size_scratch, 0/*auto*/);
  libxsmm_free(q);
  q = libxsmm_aligned_scratch(size_scratch / 3, 0/*auto*/);
  r = libxsmm_aligned_scratch(size_scratch / 3, 0/*auto*/);
  /* confirm malloc succeeds for an in-scratch buffer */
  if (NULL != q && NULL == r) {
    FPRINTF(stderr, "Error: in-scratch buffer allocation failed!\n");
    ++nerrors;
  }
#endif

#if defined(CHECK_REALLOC)
  if (NULL != s) { /* reallocate larger amount of memory */
    unsigned char* c = (unsigned char*)s;
    size_t i;
    int n;
    avalue = 1 << LIBXSMM_INTRINSICS_BITSCANFWD64((uintptr_t)s);
    for (i = 0; i < size_malloc; ++i) c[i] = (unsigned char)LIBXSMM_MOD2(i, 256);
    s = libxsmm_realloc(size_malloc * 2, s);
    /* check that alignment is preserved */
    if (0 != (((uintptr_t)s) % avalue)) {
      FPRINTF(stderr, "Error: buffer alignment (1/3) not preserved!\n");
      ++nerrors;
    }
    c = (unsigned char*)s;
    for (i = size_malloc; i < (size_malloc * 2); ++i) c[i] = (unsigned char)LIBXSMM_MOD2(i, 256);
    /* reallocate again with same size */
    s = libxsmm_realloc(size_malloc * 2, s);
    /* check that alignment is preserved */
    if (0 != (((uintptr_t)s) % avalue)) {
      FPRINTF(stderr, "Error: buffer alignment (2/3) not preserved!\n");
      ++nerrors;
    }
    c = (unsigned char*)s;
    for (i = n = 0; i < (size_malloc * 2); ++i) { /* check that content is preserved */
      n += (c[i] == (unsigned char)LIBXSMM_MOD2(i, 256) ? 0 : 1);
    }
    if (0 < n) {
      FPRINTF(stderr, "Error: buffer content (1/2) not preserved!\n");
      nerrors += n;
    }
    /* reallocate with smaller size */
    s = libxsmm_realloc(size_malloc / 2, s);
    /* check that alignment is preserved */
    if (0 != (((uintptr_t)s) % avalue)) {
      FPRINTF(stderr, "Error: buffer alignment (3/3) not preserved!\n");
      ++nerrors;
    }
    c = (unsigned char*)s;
    for (i = n = 0; i < size_malloc / 2; ++i) { /* check that content is preserved */
      n += (c[i] == (unsigned char)LIBXSMM_MOD2(i, 256) ? 0 : 1);
    }
    if (0 < n) {
      FPRINTF(stderr, "Error: buffer content (2/2) not preserved!\n");
      nerrors += n;
    }
  }
  /* query and check the size of the buffer */
  if (NULL != s && (EXIT_SUCCESS != libxsmm_get_malloc_info(s, &malloc_info) || malloc_info.size < (size_malloc / 2))) {
    FPRINTF(stderr, "Error: buffer info (2/4) failed!\n");
    ++nerrors;
  }
  libxsmm_free(s); /* release buffer */

  /* check degenerated reallocation */
  s = libxsmm_realloc(size_malloc, NULL/*allocation*/);
  /* query and check the size of the buffer */
  if (NULL != s && (EXIT_SUCCESS != libxsmm_get_malloc_info(s, &malloc_info) || malloc_info.size < size_malloc)) {
    FPRINTF(stderr, "Error: buffer info (3/4) failed!\n");
    ++nerrors;
  }
#endif

  /* check that a NULL-pointer yields no size */
  if (EXIT_SUCCESS != libxsmm_get_malloc_info(NULL, &malloc_info) || 0 != malloc_info.size) {
    FPRINTF(stderr, "Error: buffer info (4/4) failed!\n");
    ++nerrors;
  }

  /* release NULL pointer */
  libxsmm_free(NULL);

  /* release buffer */
  libxsmm_free(s);

  /* allocate memory with specific alignment */
  s = libxsmm_aligned_malloc(size_malloc, alignment);
  /* check function that determines alignment */
  libxsmm_aligned(s, NULL/*inc*/, &avalue);

  /* check the alignment of the allocation */
  if (0 != (((uintptr_t)s) % alignment) || ((size_t)avalue) < alignment) {
    FPRINTF(stderr, "Error: buffer alignment (1/3) incorrect!\n");
    ++nerrors;
  }

  if (libxsmm_aligned(s, NULL/*inc*/, NULL/*alignment*/)) { /* pointer is SIMD-aligned */
    if (alignment < ((size_t)libxsmm_cpuid_vlen(libxsmm_get_target_archid()))) {
      FPRINTF(stderr, "Error: buffer alignment (2/3) incorrect!\n");
      ++nerrors;
    }
  }
  else { /* pointer is not SIMD-aligned */
    if (((size_t)libxsmm_cpuid_vlen(libxsmm_get_target_archid())) <= alignment) {
      FPRINTF(stderr, "Error: buffer alignment (3/3) incorrect!\n");
      ++nerrors;
    }
  }

  /* release memory */
  libxsmm_free(s);
#if defined(CHECK_SCRATCH)
  libxsmm_free(q);
  libxsmm_free(r);
#endif

  /* check foreign memory */
  if (EXIT_SUCCESS == libxsmm_get_malloc_info(&size_malloc/*faulty pointer*/, &malloc_info)) {
    FPRINTF(stderr, "Error: uncaught faulty pointer!\n");
    ++nerrors;
  }

  if (0 == nerrors) {
    return EXIT_SUCCESS;
  }
  else {
    FPRINTF(stderr, "Errors: %i\n", nerrors);
    return EXIT_FAILURE;
  }
}
