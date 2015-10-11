/******************************************************************************
** Copyright (c) 2013-2015, Intel Corporation                                **
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
/* Alexander Heinecke (Intel Corp.), Hans Pabst (Intel Corp.)
******************************************************************************/
#include "libxsmm_dispatch.h"
#include <libxsmm_generator.h>
#include <libxsmm.h>

#if defined(LIBXSMM_OFFLOAD_BUILD)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#if !defined(_WIN32)
# include <sys/mman.h>
#endif
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(LIBXSMM_OFFLOAD_BUILD)
# pragma offload_attribute(pop)
#endif

#define LIBXSMM_CODE_PAGESIZE 4096


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_build_jit(int single_precision, int m, int n, int k)
{
  /* calling libxsmm_build_jit shall imply an early/explicit initialization of the library */
  libxsmm_build_static();

#if (0 != (LIBXSMM_JIT))
#if !defined(_WIN32)
  /* build xgemm descriptor */
  libxsmm_xgemm_descriptor LIBXSMM_XGEMM_DESCRIPTOR(l_xgemm_desc, single_precision, LIBXSMM_PREFETCH, 'n', 'n', 1/*alpha*/, LIBXSMM_BETA,
    m, n, k, m, k, LIBXSMM_ALIGN_STORES(m, 0 != single_precision ? sizeof(float) : sizeof(double)));

  /* set arch string */
  char l_arch[14];
#ifdef __SSE3__
#ifndef __AVX__
#error "SSE3 instructions set extensions have no jitting support!"
#endif
#endif
#ifdef __MIC__
#error "Xeon Phi coprocessors (IMCI architecture) have no jitting support!"
#endif
#ifdef __AVX__
  strcpy ( l_arch, "snb" );
#endif
#ifdef __AVX2__
  strcpy ( l_arch, "hsw" );
#endif
#ifdef __AVX512F__
  strcpy ( l_arch, "knl" );
#endif

  /* allocate buffer for code */
  unsigned char* l_gen_code = (unsigned char*) malloc( 32768 * sizeof(unsigned char) );
  libxsmm_generated_code l_generated_code;
  l_generated_code.generated_code = (void*)l_gen_code;
  l_generated_code.buffer_size = 32768;
  l_generated_code.code_size = 0;
  l_generated_code.code_type = 2;
  l_generated_code.last_error = 0;

  /* generate kernel */
  libxsmm_generator_dense_kernel( &l_generated_code,
                                  &l_xgemm_desc,
                                  l_arch );

  /* handle an eventual error */
  if ( l_generated_code.last_error != 0 ) {
    fprintf(stderr, "%s\n", libxsmm_strerror( l_generated_code.last_error ) );
    exit(-1);
  }

  /* create executable buffer */
  const int l_code_pages = (((l_generated_code.code_size-1)*sizeof(unsigned char))/LIBXSMM_CODE_PAGESIZE)+1;
  unsigned char* l_code = 0;
#if defined(_WIN32)
  l_code = (unsigned char*)_aligned_malloc(l_code_pages * LIBXSMM_CODE_PAGESIZE, 4096);
#else
  void* p = 0;
#if !defined(NDEBUG)
  const int result =
#endif
  posix_memalign(&p, 4096, l_code_pages * LIBXSMM_CODE_PAGESIZE);
  assert(0 == result);
  l_code = (unsigned char*)p;
#endif
  memset( l_code, 0, l_code_pages*LIBXSMM_CODE_PAGESIZE );
  memcpy( l_code, l_gen_code, l_generated_code.code_size );
  /* set memory protection to R/E */
  mprotect( (void*)l_code, l_code_pages*LIBXSMM_CODE_PAGESIZE, PROT_EXEC | PROT_READ );

  /* free tmp buffer */
  free(l_gen_code);

  /* make function pointer available for dispatch */
  libxsmm_dispatch(&l_xgemm_desc, sizeof(l_xgemm_desc), single_precision % 2, (libxsmm_function)l_code);
#else
  fprintf(stderr, "LIBXSMM ERROR: JITTING IS NOT SUPPORTED ON WINDOWS RIGHT NOW!\n");
#endif
#endif
}
