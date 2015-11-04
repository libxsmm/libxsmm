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

#define LIBXSMM_DISPATCH_CACHESIZE ((LIBXSMM_MAX_MNK) * 8)
#if !defined(_WIN32)
#define LIBXSMM_DISPATCH_PAGESIZE sysconf(_SC_PAGESIZE)
#else
#define LIBXSMM_DISPATCH_PAGESIZE 4096
#endif
#define LIBXSMM_DISPATCH_SEED 0


typedef union LIBXSMM_RETARGETABLE libxsmm_cache_entry {
  libxsmm_smm_function smm;
  libxsmm_dmm_function dmm;
  const void* pv;
} libxsmm_cache_entry;
/** Filled with zeros due to C language rule. */
LIBXSMM_RETARGETABLE libxsmm_cache_entry libxsmm_cache[(LIBXSMM_DISPATCH_CACHESIZE)];
int libxsmm_init_check = 0;

#if !defined(_OPENMP)
LIBXSMM_RETARGETABLE LIBXSMM_LOCK_TYPE libxsmm_dispatch_lock[] = {
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT,
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT,
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT,
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT
};
#endif


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_init(void)
{
#if !defined(_OPENMP)
  /* acquire one of the locks as the master lock */
  LIBXSMM_LOCK_ACQUIRE(libxsmm_dispatch_lock[0]);
#else
# pragma omp critical(libxsmm_dispatch_lock)
#endif
  if (0 == libxsmm_init_check) {
#if !defined(_OPENMP)
    const int nlocks = sizeof(libxsmm_dispatch_lock) / sizeof(libxsmm_cache_entry);
    int i;
    /* acquire and release remaining locks to shortcut any lazy initialization later on */
    for (i = 1; i < nlocks; ++i) {
      LIBXSMM_LOCK_ACQUIRE(libxsmm_dispatch_lock[i]);
      LIBXSMM_LOCK_RELEASE(libxsmm_dispatch_lock[i]);
    }
#endif
    {
      /* setup the dispatch table for the statically generated code */
#     include <libxsmm_dispatch.h>
    }
    libxsmm_init_check = 1;
  }
#if !defined(_OPENMP)
  /* release the master lock */
  LIBXSMM_LOCK_RELEASE(libxsmm_dispatch_lock[0]);
#endif
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_init(void)
{
  if (0 == libxsmm_init_check) {
    internal_init();
  }
}


LIBXSMM_RETARGETABLE libxsmm_cache_entry internal_build(const libxsmm_gemm_descriptor* desc)
{
  libxsmm_cache_entry result;
  unsigned int hash, indx;
  assert(0 != desc);

  /* lazy initialization */
  if (0 == libxsmm_init_check) {
    internal_init();
  }

  /* check if the requested xGEMM is already JITted */
  LIBXSMM_PRAGMA_FORCEINLINE /* must precede a statement */
  hash = libxsmm_crc32(desc, LIBXSMM_GEMM_DESCRIPTOR_SIZE, LIBXSMM_DISPATCH_SEED);

  indx = hash % (LIBXSMM_DISPATCH_CACHESIZE);
  result = libxsmm_cache[indx]; /* TODO: handle collision */

#if (0 != (LIBXSMM_JIT))
  if (0 == result.pv) {
# if !defined(_WIN32) && !defined(__CYGWIN__)
# if !defined(_OPENMP)
    const unsigned int lock = LIBXSMM_MOD2(indx, sizeof(libxsmm_dispatch_lock) / sizeof(*libxsmm_dispatch_lock));
    LIBXSMM_LOCK_ACQUIRE(libxsmm_dispatch_lock[lock]);
# else
#   pragma omp critical(libxsmm_dispatch_lock)
# endif
    {
      result = libxsmm_cache[indx];

      if (0 == result.pv) {
        char l_arch[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }; /* empty initial arch string */
        int l_code_pages, l_code_page_size, l_fd;
        libxsmm_generated_code l_generated_code;
        void* l_code;

# if defined(__SSE3__)
#   if !defined(__AVX__)
#       error "SSE3 instruction set extension is not supported for JIT-code generation!"
#   endif
# elif !defined(__MIC__)
#       error "No instruction set extension found for JIT-code generation!"
# endif
# if defined(__MIC__)
#       error "IMCI architecture (Xeon Phi coprocessor) is not supported for JIT-code generation!"
# endif
# if defined(__AVX__)
        strcpy(l_arch, "snb");
# endif
# if defined(__AVX2__)
        strcpy(l_arch, "hsw");
# endif
# if defined(__AVX512F__)
        strcpy(l_arch, "knl");
# endif
        /* allocate buffer for code */
        l_generated_code.generated_code = malloc(131072 * sizeof(unsigned char));
        l_generated_code.buffer_size = 0 != l_generated_code.generated_code ? 131072 : 0;
        l_generated_code.code_size = 0;
        l_generated_code.code_type = 2;
        l_generated_code.last_error = 0;

        /* generate kernel */
        libxsmm_generator_dense_kernel(&l_generated_code, desc, l_arch);

        /* handle an eventual error */
        if (l_generated_code.last_error != 0) {
# if !defined(NDEBUG) /* library code is usually expected to be mute */
          fprintf(stderr, "%s\n", libxsmm_strerror(l_generated_code.last_error));
# endif /*NDEBUG*/
          free(l_generated_code.generated_code);
          return result;
        }

        /* create executable buffer */
        l_code_pages = (((l_generated_code.code_size-1)*sizeof(unsigned char))/(LIBXSMM_DISPATCH_PAGESIZE))+1;
        l_code_page_size = (LIBXSMM_DISPATCH_PAGESIZE)*l_code_pages;
        l_fd = open("/dev/zero", O_RDWR);
        l_code = mmap(0, l_code_page_size, PROT_READ|PROT_WRITE, MAP_PRIVATE, l_fd, 0);
        close(l_fd);

        /* explicitly disable THP for this memory region, kernel 2.6.38 or higher */
# if defined(MADV_NOHUGEPAGE)
        madvise(l_code, l_code_page_size, MADV_NOHUGEPAGE);
# endif /*MADV_NOHUGEPAGE*/

        if (MAP_FAILED == l_code) {
# if !defined(NDEBUG) /* library code is usually expected to be mute */
          fprintf(stderr, "LIBXSMM: something bad happend in mmap, couldn't allocate code buffer!\n");
# endif /*NDEBUG*/
          free(l_generated_code.generated_code);
          return result;
        }

        memcpy( l_code, l_generated_code.generated_code, l_generated_code.code_size );
        if (-1 == mprotect(l_code, l_code_page_size, PROT_EXEC | PROT_READ)) {
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
          return result;
        }

# if !defined(NDEBUG)
        { /* write buffer for manual decode as binary to a file */
          char l_objdump_name[512];
          FILE* l_byte_code;
          sprintf(l_objdump_name, "kernel_prec%i_m%u_n%u_k%u_lda%u_ldb%u_ldc%u_a%i_b%i_ta%c_tb%c_pf%i.bin",
            0 == (LIBXSMM_GEMM_FLAG_F32PREC & desc->flags) ? 0 : 1,
            desc->m, desc->n, desc->k, desc->lda, desc->ldb, desc->ldc, desc->alpha, desc->beta,
            0 == (LIBXSMM_GEMM_FLAG_TRANS_A & desc->flags) ? 'n' : 't',
            0 == (LIBXSMM_GEMM_FLAG_TRANS_B & desc->flags) ? 'n' : 't',
            desc->prefetch);
          l_byte_code = fopen(l_objdump_name, "wb");
          if (l_byte_code != NULL) {
            fwrite(l_generated_code.generated_code, 1, l_generated_code.code_size, l_byte_code);
            fclose(l_byte_code);
          }
        }
# endif /*NDEBUG*/
        /* free temporary buffer */
        free(l_generated_code.generated_code);

        /* make function pointer available for dispatch */
        libxsmm_cache[indx].pv = l_code;
        
        /* prepare return value */
        result.pv = l_code;
      }
    }

# if !defined(_OPENMP)
    LIBXSMM_LOCK_RELEASE(libxsmm_dispatch_lock[lock]);
# endif
# else
#   error "LIBXSMM ERROR: JITTING IS NOT SUPPORTED ON WINDOWS RIGHT NOW!"
# endif /*_WIN32*/
  }
#endif /*LIBXSMM_JIT*/

  return result;
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_smm_function libxsmm_smm_dispatch(float alpha, float beta,
  int m, int n, int k, int lda, int ldb, int ldc, int flags, int prefetch)
{
  LIBXSMM_GEMM_DESCRIPTOR_TYPE(desc, alpha, beta, m, n, k, LIBXSMM_MAX(lda, LIBXSMM_LD(m, n)), LIBXSMM_MAX(ldb, k),
    LIBXSMM_MAX(ldc, LIBXSMM_ALIGN_STORES(LIBXSMM_LD(m, n), sizeof(double))),
    flags | LIBXSMM_GEMM_FLAG_F32PREC, prefetch);
  return internal_build(&desc).smm;
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_dmm_function libxsmm_dmm_dispatch(double alpha, double beta,
  int m, int n, int k, int lda, int ldb, int ldc, int flags, int prefetch)
{
  LIBXSMM_GEMM_DESCRIPTOR_TYPE(desc, alpha, beta, m, n, k, LIBXSMM_MAX(lda, LIBXSMM_LD(m, n)), LIBXSMM_MAX(ldb, k),
    LIBXSMM_MAX(ldc, LIBXSMM_ALIGN_STORES(LIBXSMM_LD(m, n), sizeof(double))),
    flags, prefetch);
  return internal_build(&desc).dmm;
}
