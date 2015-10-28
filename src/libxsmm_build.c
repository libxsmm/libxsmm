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
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(_WIN32)
# include <Windows.h>
#else
# include <fcntl.h>
# include <unistd.h>
# include <sys/mman.h>
#endif
#if !defined(NDEBUG)
#include <errno.h>
#endif
#if defined(LIBXSMM_OFFLOAD_BUILD)
# pragma offload_attribute(pop)
#endif

#define LIBXSMM_BUILD_CACHESIZE ((LIBXSMM_MAX_MNK) * 8)
#if !defined(_WIN32)
#define LIBXSMM_BUILD_PAGESIZE sysconf(_SC_PAGESIZE)
#else
#define LIBXSMM_BUILD_PAGESIZE 4096
#endif
#define LIBXSMM_BUILD_SEED 0


/** Filled with zeros due to C language rule. */
LIBXSMM_RETARGETABLE libxsmm_function libxsmm_cache[2][(LIBXSMM_BUILD_CACHESIZE)];

#if !defined(_OPENMP)
LIBXSMM_RETARGETABLE LIBXSMM_LOCK_TYPE libxsmm_build_lock[] = {
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT,
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT,
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT,
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT
};
#endif


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_build_static(void)
{
  static int init = 0;

  if (0 == init) {
#if !defined(_OPENMP)
    int i;
    for (i = 0; i < 0; ++i) {
      LIBXSMM_LOCK_ACQUIRE(libxsmm_build_lock[i]);
    }
#else
#   pragma omp critical(libxsmm_build_lock)
#endif
    if (0 == init) {
#     include <libxsmm_build.h>
      init = 1;
    }
#if !defined(_OPENMP)
    for (i = 0; i < 0; ++i) {
      LIBXSMM_LOCK_RELEASE(libxsmm_build_lock[i]);
    }
#endif
  }
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_function libxsmm_build_jit(int single_precision, int m, int n, int k)
{
  libxsmm_function result = 0;

  /* calling libxsmm_build_jit shall imply an early/explicit initialization of the library, this is lazy initialization */
  libxsmm_build_static();

#if (0 != (LIBXSMM_JIT))
  {
    libxsmm_function *const cache = libxsmm_cache[single_precision&1];
    unsigned int hash, indx;

    /* build xgemm descriptor: LIBXSMM_XGEMM_DESCRIPTOR(DESCRIPTOR, M, N, K, LDA, LDB, LDC, PREFETCH, FLAGS, FALPHA, FBETA) */
    LIBXSMM_XGEMM_DESCRIPTOR_TYPE(l_xgemm_desc,
      m, n, k, m, k, LIBXSMM_ALIGN_STORES(m, 0 != single_precision ? sizeof(float) : sizeof(double)), LIBXSMM_PREFETCH,
      (0 == single_precision ? 0 : LIBXSMM_XGEMM_FLAG_F32PREC)
        | (1 < (LIBXSMM_ALIGNED_LOADS) ? LIBXSMM_XGEMM_FLAG_ALIGN_A : 0)
        | (1 < (LIBXSMM_ALIGNED_STORES) ? LIBXSMM_XGEMM_FLAG_ALIGN_C : 0),
      1/*alpha*/, LIBXSMM_BETA);

    /* check if the requested xGEMM is already JITted */
    LIBXSMM_PRAGMA_FORCEINLINE /* must precede a statement */
    hash = libxsmm_crc32(&l_xgemm_desc, LIBXSMM_XGEMM_DESCRIPTOR_SIZE, LIBXSMM_BUILD_SEED);

    indx = hash % (LIBXSMM_BUILD_CACHESIZE);
    /* TODO: handle collision */
    result = cache[indx];

    if (0 == result) {
# if !defined(_WIN32)
# if !defined(_OPENMP)
      const unsigned int lock = LIBXSMM_MOD2(indx, sizeof(libxsmm_build_lock) / sizeof(*libxsmm_build_lock));
      LIBXSMM_LOCK_ACQUIRE(libxsmm_build_lock[lock]);
# else
#     pragma omp critical(libxsmm_build_lock)
# endif
      {
        result = cache[indx];

        if (0 == result) {
          int l_code_pages, l_code_page_size, l_fd;
          libxsmm_generated_code l_generated_code;
          char l_arch[14]; /* set arch string */
          union { /* used to avoid conversion warning */
            libxsmm_function pf;
            void* pv;
          } l_code;

# ifdef __SSE3__
#   ifndef __AVX__
#         error "SSE3 instructions set extensions have no jitting support!"
#   endif
# endif
# ifdef __MIC__
#         error "Xeon Phi coprocessors (IMCI architecture) have no jitting support!"
# endif
# ifdef __AVX__
          strcpy(l_arch, "snb");
# endif
# ifdef __AVX2__
          strcpy(l_arch, "hsw");
# endif
# ifdef __AVX512F__
          strcpy(l_arch, "knl");
# endif
          /* allocate buffer for code */
          l_generated_code.generated_code = malloc(131072 * sizeof(unsigned char));
          l_generated_code.buffer_size = 0 != l_generated_code.generated_code ? 131072 : 0;
          l_generated_code.code_size = 0;
          l_generated_code.code_type = 2;
          l_generated_code.last_error = 0;

          /* generate kernel */
          libxsmm_generator_dense_kernel(&l_generated_code, &l_xgemm_desc, l_arch);

          /* handle an eventual error */
          if (l_generated_code.last_error != 0) {
# if !defined(NDEBUG) /* library code is usually expected to be mute */
            fprintf(stderr, "%s\n", libxsmm_strerror(l_generated_code.last_error));
# endif /*NDEBUG*/
            free(l_generated_code.generated_code);
            return 0;
          }

          /* create executable buffer */
          l_code_pages = (((l_generated_code.code_size-1)*sizeof(unsigned char))/(LIBXSMM_BUILD_PAGESIZE))+1;
          l_code_page_size = (LIBXSMM_BUILD_PAGESIZE)*l_code_pages;
          l_fd = open("/dev/zero", O_RDWR);
          l_code.pv = mmap(0, l_code_page_size, PROT_READ|PROT_WRITE, MAP_PRIVATE, l_fd, 0);
          close(l_fd);

          /* explicitly disable THP for this memory region, kernel 2.6.38 or higher */
# if defined(MADV_NOHUGEPAGE)
          madvise(l_code.pv, l_code_page_size, MADV_NOHUGEPAGE);
# endif /*MADV_NOHUGEPAGE*/

          if (l_code.pv == MAP_FAILED) {
# if !defined(NDEBUG) /* library code is usually expected to be mute */
            fprintf(stderr, "LIBXSMM: something bad happend in mmap, couldn't allocate code buffer!\n");
# endif /*NDEBUG*/
            free(l_generated_code.generated_code);
            return 0;
          }

          memcpy( l_code.pv, l_generated_code.generated_code, l_generated_code.code_size );
          if (-1 == mprotect(l_code.pv, l_code_page_size, PROT_EXEC | PROT_READ)) {
# if !defined(NDEBUG)
            int errsv = errno;
            if (errsv == EINVAL) {
              fprintf(stderr, "LIBXSMM: mprotect failed: addr is not a valid pointer, or not a multiple of the system page size!\n");
            } else if (errsv == ENOMEM) {
              fprintf(stderr, "LIBXSMM: mprotect failed: Internal kernel structures could not be allocated!\n");
            } else if (errsv == EACCES) {
              fprintf(stderr, "LIBXSMM: mprotect failed: The memory cannot be given the specified access!\n");
            } else {
              fprintf(stderr, "LIBXSMM: mprotect failed: Unknown Error!\n");
            }
# endif /*NDEBUG*/
            free(l_generated_code.generated_code);
            return 0;
          }

# if !defined(NDEBUG)
          /* write buffer for manual decode as binary to a file */
          char l_objdump_name[512];
          sprintf( l_objdump_name, "kernel_prec%i_m%i_n%i_k%i_lda%i_ldb%i_ldc%i_a%i_b%i_ta%c_tb%c_pf%i.bin",
                   l_xgemm_desc.single_precision, l_xgemm_desc.m, l_xgemm_desc.n, l_xgemm_desc.k,
                   l_xgemm_desc.lda, l_xgemm_desc.ldb, l_xgemm_desc.ldc, l_xgemm_desc.alpha, l_xgemm_desc.beta,
                   l_xgemm_desc.trans_a, l_xgemm_desc.trans_b, l_xgemm_desc.prefetch );
          FILE *const l_byte_code = fopen( l_objdump_name, "wb");
          if ( l_byte_code != NULL ) {
            fwrite( l_generated_code.generated_code, 1, l_generated_code.code_size, l_byte_code);
            fclose( l_byte_code );
          }
# endif /*NDEBUG*/
          /* free temporary buffer, and prepare return value */
          free(l_generated_code.generated_code);
          result = l_code.pf;

          /* make function pointer available for dispatch */
          cache[indx] = result;
        }
      }

# if !defined(_OPENMP)
      LIBXSMM_LOCK_RELEASE(libxsmm_build_lock[lock]);
# endif
# else
#     error "LIBXSMM ERROR: JITTING IS NOT SUPPORTED ON WINDOWS RIGHT NOW!"
# endif /*_WIN32*/
    }
  }
#else
  LIBXSMM_UNUSED(single_precision); LIBXSMM_UNUSED(m); LIBXSMM_UNUSED(n); LIBXSMM_UNUSED(k);
#endif /*LIBXSMM_JIT*/
  return result;
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_smm_function libxsmm_smm_dispatch(int m, int n, int k)
{
#if 1 == (LIBXSMM_JIT) || 0 > (LIBXSMM_JIT) /* automatic JITting */
  return (libxsmm_smm_function)libxsmm_build_jit(1/*single precision*/, m, n, k);
#else /* explicit JITting and static code generation */
  unsigned int hash, indx;
  /* calling libxsmm_build_jit shall imply an early/explicit initialization of the librar, this is lazy initializationy */
  libxsmm_build_static();
  {
    /* build xgemm descriptor: LIBXSMM_XGEMM_DESCRIPTOR(DESCRIPTOR, M, N, K, LDA, LDB, LDC, PREFETCH, FLAGS, FALPHA, FBETA) */
    LIBXSMM_XGEMM_DESCRIPTOR_TYPE(desc,
      m, n, k, m, k, LIBXSMM_ALIGN_STORES(m, sizeof(float)), LIBXSMM_PREFETCH, LIBXSMM_XGEMM_FLAG_F32PREC
        | (1 < (LIBXSMM_ALIGNED_LOADS) ? LIBXSMM_XGEMM_FLAG_ALIGN_A : 0)
        | (1 < (LIBXSMM_ALIGNED_STORES) ? LIBXSMM_XGEMM_FLAG_ALIGN_C : 0),
      1/*alpha*/, LIBXSMM_BETA);
    LIBXSMM_PRAGMA_FORCEINLINE /* must precede a statement */
    hash = libxsmm_crc32(&desc, LIBXSMM_XGEMM_DESCRIPTOR_SIZE, LIBXSMM_BUILD_SEED);
    indx = hash % (LIBXSMM_BUILD_CACHESIZE);
  }
  return (libxsmm_smm_function)libxsmm_cache[1/*single precision*/][indx];
#endif
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_dmm_function libxsmm_dmm_dispatch(int m, int n, int k)
{
#if 1 == (LIBXSMM_JIT) || 0 > (LIBXSMM_JIT) /* automatic JITting */
  return (libxsmm_dmm_function)libxsmm_build_jit(0/*double precision*/, m, n, k);
#else /* explicit JITting and static code generation */
  unsigned int hash, indx;
  /* calling libxsmm_build_jit shall imply an early/explicit initialization of the library, this is lazy initialization */
  libxsmm_build_static();
  {
    /* build xgemm descriptor: LIBXSMM_XGEMM_DESCRIPTOR(DESCRIPTOR, M, N, K, LDA, LDB, LDC, PREFETCH, FLAGS, FALPHA, FBETA) */
    LIBXSMM_XGEMM_DESCRIPTOR_TYPE(desc,
      m, n, k, m, k, LIBXSMM_ALIGN_STORES(m, sizeof(double)), LIBXSMM_PREFETCH, 0/*double-precision*/
        | (1 < (LIBXSMM_ALIGNED_LOADS) ? LIBXSMM_XGEMM_FLAG_ALIGN_A : 0)
        | (1 < (LIBXSMM_ALIGNED_STORES) ? LIBXSMM_XGEMM_FLAG_ALIGN_C : 0),
      1/*alpha*/, LIBXSMM_BETA);
    LIBXSMM_PRAGMA_FORCEINLINE /* must precede a statement */
    hash = libxsmm_crc32(&desc, LIBXSMM_XGEMM_DESCRIPTOR_SIZE, LIBXSMM_BUILD_SEED);
    indx = hash % (LIBXSMM_BUILD_CACHESIZE);
  }
  return (libxsmm_dmm_function)libxsmm_cache[0/*double precision*/][indx];
#endif
}
