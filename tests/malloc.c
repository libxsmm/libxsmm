/******************************************************************************
** Copyright (c) 2016-2019, Intel Corporation                                **
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
#include <libxsmm.h>
#include <stdlib.h>


int main(void)
{
  const size_t size = 2507, alignment = (2u << 20);
  const void* context;
  int nerrors = 0;
  void* p;

  libxsmm_malloc_info malloc_info;
  libxsmm_malloc_function malloc_fn;
  libxsmm_free_function free_fn;
  malloc_fn.function = malloc; free_fn.function = free;
  libxsmm_set_default_allocator(NULL/*context*/, malloc_fn/*malloc*/, free_fn/*free*/);
  malloc_fn.function = NULL; free_fn.function = NULL;
  libxsmm_set_scratch_allocator(NULL/*context*/, malloc_fn/*NULL*/, free_fn/*NULL*/);

  /* check adoption of the default allocator */
  libxsmm_get_scratch_allocator(&context, &malloc_fn, &free_fn);
  if (NULL != context || malloc != malloc_fn.function || free != free_fn.function) {
    ++nerrors;
  }

  /* allocate some amount of memory */
  p = libxsmm_malloc(size);

  /* query and check the size of the buffer */
  if (NULL != p && (EXIT_SUCCESS != libxsmm_get_malloc_info(p, &malloc_info) || malloc_info.size < size)) {
    ++nerrors;
  }

  { /* reallocate larger amount of memory */
    unsigned char* c = (unsigned char*)p;
    size_t i;
    for (i = 0; i < size; ++i) c[i] = (unsigned char)LIBXSMM_MOD2(i, 256);
    p = libxsmm_realloc(size * 2, p);
    /* reallocate again with same size */
    p = libxsmm_realloc(size * 2, p);
    c = (unsigned char*)p;
    for (i = 0; i < size; ++i) { /* check that content is preserved */
      nerrors += (c[i] == (unsigned char)LIBXSMM_MOD2(i, 256) ? 0 : 1);
    }
    for (i = size; i < 2 * size; ++i) c[i] = (unsigned char)LIBXSMM_MOD2(i, 256);
    /* reallocate with smaller size */
    p = libxsmm_realloc(size / 2, p);
    c = (unsigned char*)p;
    for (i = 0; i < size / 2; ++i) { /* check that content is preserved */
      nerrors += (c[i] == (unsigned char)LIBXSMM_MOD2(i, 256) ? 0 : 1);
    }
  }

  /* query and check the size of the buffer */
  if (NULL != p && (EXIT_SUCCESS != libxsmm_get_malloc_info(p, &malloc_info) || malloc_info.size < (size / 2))) {
    ++nerrors;
  }

  /* check that a NULL-pointer yields no size */
  if (EXIT_SUCCESS != libxsmm_get_malloc_info(NULL, &malloc_info) || 0 != malloc_info.size) {
    ++nerrors;
  }

  /* release a NULL pointer */
  libxsmm_free(NULL);

  /* release a buffer */
  libxsmm_free(p);

  /* allocate memory with specific alignment */
  p = libxsmm_aligned_malloc(size, alignment);

  /* check the alignment of the allocation */
  if (0 != (((uintptr_t)p) % alignment)) {
    ++nerrors;
  }

  /* release aligned memory */
  libxsmm_free(p);

  return 0 == nerrors ? EXIT_SUCCESS : EXIT_FAILURE;
}

