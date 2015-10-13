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
/* Hans Pabst (Intel Corp.), Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_extern_typedefs.h"
#include "libxsmm_crc32.h"
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

#define LIBXSMM_BUILD_CACHESIZE (LIBXSMM_MAX_M) * (LIBXSMM_MAX_N) * (LIBXSMM_MAX_K) * 8
#define LIBXSMM_BUILD_PAGESIZE 4096
#define LIBXSMM_BUILD_SEED 0


/** Filled with zeros due to C language rule. */
LIBXSMM_RETARGETABLE libxsmm_function libxsmm_cache[2][(LIBXSMM_BUILD_CACHESIZE)];
LIBXSMM_RETARGETABLE LIBXSMM_LOCK_TYPE libxsmm_locks[] = {
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT,
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT,
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT,
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT
};


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_build_static()
{
  static int init = 0;
  if (0 == init) {
    LIBXSMM_LOCK_ACQUIRE(libxsmm_locks[0]);
    if (0 == init) {
#     include <libxsmm_build.h>
      init = 1;
    }
    LIBXSMM_LOCK_RELEASE(libxsmm_locks[0]);
  }
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_function libxsmm_build_jit(int single_precision, int m, int n, int k)
{
  libxsmm_function result = 0;

  /* calling libxsmm_build_jit shall imply an early/explicit initialization of the library */
  libxsmm_build_static();

#if (0 != (LIBXSMM_JIT))
  /* build xgemm descriptor */
  libxsmm_xgemm_descriptor LIBXSMM_XGEMM_DESCRIPTOR(l_xgemm_desc, single_precision, LIBXSMM_PREFETCH, 'n', 'n', 1/*alpha*/, LIBXSMM_BETA,
    m, n, k, m, k, LIBXSMM_ALIGN_STORES(m, 0 != single_precision ? sizeof(float) : sizeof(double)));

  /* check if the requested xGEMM is already JITted */
  const unsigned int hash = libxsmm_crc32(&l_xgemm_desc, LIBXSMM_XGEMM_DESCRIPTOR_SIZE, LIBXSMM_BUILD_SEED);
  const unsigned int indx = hash % (LIBXSMM_BUILD_CACHESIZE);
  libxsmm_function *const cache = libxsmm_cache[single_precision&1];
  /* TODO: handle collision */
  result = cache[indx];

  if (0 == result) {
#if !defined(_WIN32)
    const unsigned int lock = LIBXSMM_MOD2(indx, sizeof(libxsmm_locks) / sizeof(*libxsmm_locks));
    LIBXSMM_LOCK_ACQUIRE(libxsmm_locks[lock]);
    result = cache[indx];

    if (0 == result) {
      /* set arch string */
      char l_arch[14];
#ifdef __SSE3__
# ifndef __AVX__
#   error "SSE3 instructions set extensions have no jitting support!"
# endif
#endif
#ifdef __MIC__
# error "Xeon Phi coprocessors (IMCI architecture) have no jitting support!"
#endif
#ifdef __AVX__
      strcpy(l_arch, "snb");
#endif
#ifdef __AVX2__
      strcpy(l_arch, "hsw");
#endif
#ifdef __AVX512F__
      strcpy(l_arch, "knl");
#endif
      /* allocate buffer for code */
      unsigned char* l_gen_code = (unsigned char*) malloc(32768 * sizeof(unsigned char));
      libxsmm_generated_code l_generated_code;
      l_generated_code.generated_code = (void*)l_gen_code;
      l_generated_code.buffer_size = 32768;
      l_generated_code.code_size = 0;
      l_generated_code.code_type = 2;
      l_generated_code.last_error = 0;

      /* generate kernel */
      libxsmm_generator_dense_kernel(&l_generated_code, &l_xgemm_desc, l_arch);

      /* handle an eventual error */
      if (l_generated_code.last_error != 0) {
        fprintf(stderr, "%s\n", libxsmm_strerror(l_generated_code.last_error));
        exit(-1);
      }

      /* create executable buffer */
      const int l_code_pages = (((l_generated_code.code_size - 1) * sizeof(unsigned char)) / LIBXSMM_BUILD_PAGESIZE) + 1;
      unsigned char* l_code = 0;
#if defined(_WIN32)
      l_code = (unsigned char*)_aligned_malloc(l_code_pages * LIBXSMM_BUILD_PAGESIZE, 4096);
#else
      void* p = 0;
#if !defined(NDEBUG)
      const int error =
#endif
      posix_memalign(&p, 4096, l_code_pages * LIBXSMM_BUILD_PAGESIZE);
      assert(0 == error);
      l_code = (unsigned char*)p;
#endif
      memset(l_code, 0, l_code_pages * LIBXSMM_BUILD_PAGESIZE);
      memcpy(l_code, l_gen_code, l_generated_code.code_size);
      /* set memory protection to R/E */
      mprotect((void*)l_code, l_code_pages * LIBXSMM_BUILD_PAGESIZE, PROT_EXEC | PROT_READ);

#if !defined(NDEBUG)
      /* write buffer for manual decode as binary to a file */
      char l_objdump_name[512];
      sprintf( l_objdump_name, "kernel_prec%i_m%i_n%i_k%i_lda%i_ldb%i_ldc%i_a%i_b%i_ta%c_tb%c_pf%i.bin", 
               l_xgemm_desc.single_precision, l_xgemm_desc.m, l_xgemm_desc.n, l_xgemm_desc.k,
               l_xgemm_desc.lda, l_xgemm_desc.ldb, l_xgemm_desc.ldc, l_xgemm_desc.alpha, l_xgemm_desc.beta,
               l_xgemm_desc.trans_a, l_xgemm_desc.trans_b, l_xgemm_desc.prefetch ); 
      FILE *l_byte_code = fopen( l_objdump_name, "wb");
      if ( l_byte_code != NULL ) {
        fwrite( (const void*)l_gen_code, 1, l_generated_code.code_size, l_byte_code);
        fclose( l_byte_code );
      } else {
        /* error */
      }
#endif

      /* free temporary buffer, and prepare return value */
      free(l_gen_code);
      result = (libxsmm_function)l_code;

      /* make function pointer available for dispatch */
      cache[indx] = result;
    }

    LIBXSMM_LOCK_RELEASE(libxsmm_locks[lock]);
#else
    fprintf(stderr, "LIBXSMM ERROR: JITTING IS NOT SUPPORTED ON WINDOWS RIGHT NOW!\n");
#endif /*_WIN32*/
  }
#endif
  return result;
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_smm_function libxsmm_smm_dispatch(int m, int n, int k)
{
#if 0 != (LIBXSMM_JIT) && 1 >= (LIBXSMM_JIT) /* automatic JITting */
  return (libxsmm_smm_function)libxsmm_build_jit(1/*single precision*/, m, n, k);
#else /* explicit JITting */
  libxsmm_xgemm_descriptor LIBXSMM_XGEMM_DESCRIPTOR(desc, 1/*single precision*/, LIBXSMM_PREFETCH,
    'n', 'n', 1/*alpha*/, LIBXSMM_BETA, m, n, k, m, k, LIBXSMM_ALIGN_STORES(m, sizeof(float)));
  const unsigned int hash = libxsmm_crc32(&desc, LIBXSMM_XGEMM_DESCRIPTOR_SIZE, LIBXSMM_BUILD_SEED);
  const unsigned int indx = hash % (LIBXSMM_BUILD_CACHESIZE);
  return (libxsmm_smm_function)libxsmm_cache[1/*single precision*/][indx];
#endif
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_dmm_function libxsmm_dmm_dispatch(int m, int n, int k)
{
#if 0 != (LIBXSMM_JIT) && 1 >= (LIBXSMM_JIT) /* automatic JITting */
  return (libxsmm_dmm_function)libxsmm_build_jit(0/*double precision*/, m, n, k);
#else /* explicit JITting */
  libxsmm_xgemm_descriptor LIBXSMM_XGEMM_DESCRIPTOR(desc, 0/*double precision*/, LIBXSMM_PREFETCH,
    'n', 'n', 1/*alpha*/, LIBXSMM_BETA, m, n, k, m, k, LIBXSMM_ALIGN_STORES(m, sizeof(double)));
  const unsigned int hash = libxsmm_crc32(&desc, LIBXSMM_XGEMM_DESCRIPTOR_SIZE, LIBXSMM_BUILD_SEED);
  const unsigned int indx = hash % (LIBXSMM_BUILD_CACHESIZE);
  return (libxsmm_dmm_function)libxsmm_cache[0/*double precision*/][indx];
#endif
}
