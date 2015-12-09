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
# include <sys/mman.h>
# include <stdlib.h>
# include <unistd.h>
# include <fcntl.h>
#endif
#if !defined(NDEBUG)
#include <errno.h>
#endif
#if defined(LIBXSMM_OFFLOAD_BUILD)
# pragma offload_attribute(pop)
#endif

/* rely on a "pseudo prime" number (Mersenne) to improve cache spread */
#define LIBXSMM_DISPATCH_CACHESIZE ((2 << LIBXSMM_NBITS(LIBXSMM_MAX_MNK * (0 != LIBXSMM_JIT ? 2 : 5))) - 1)
#define LIBXSMM_DISPATCH_HASH_SEED 0


typedef union LIBXSMM_RETARGETABLE libxsmm_dispatch_entry {
  libxsmm_smmfunction smm;
  libxsmm_dmmfunction dmm;
  const void* pv;
} libxsmm_dispatch_entry;
LIBXSMM_RETARGETABLE libxsmm_dispatch_entry* libxsmm_dispatch_cache = 0;

#if !defined(_OPENMP)
LIBXSMM_RETARGETABLE LIBXSMM_LOCK_TYPE libxsmm_dispatch_lock[] = {
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT,
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT,
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT,
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT
};
#define LIBXSMM_DISPATCH_LOCKMASTER 0
#endif


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_init(void)
{
#if !defined(_OPENMP)
  /* acquire one of the locks as the master lock */
  LIBXSMM_LOCK_ACQUIRE(libxsmm_dispatch_lock[LIBXSMM_DISPATCH_LOCKMASTER]);
#else
# pragma omp critical(libxsmm_dispatch_lock)
#endif
  {
    /*const*/void* cache;
#if defined(_OPENMP)
# if (201107 <= _OPENMP)
#   pragma omp atomic read
# else
#   pragma omp flush(libxsmm_dispatch_cache)
# endif
    cache = libxsmm_dispatch_cache;
#elif defined(__GNUC__)
    __atomic_load((void**)&libxsmm_dispatch_cache, &cache, __ATOMIC_RELAXED);
#else
    cache = libxsmm_dispatch_cache;
#endif

    if (0 == cache) {
      libxsmm_dispatch_entry *const buffer = (libxsmm_dispatch_entry*)malloc(
        LIBXSMM_DISPATCH_CACHESIZE * sizeof(libxsmm_dispatch_entry));
      assert(buffer);
      if (buffer) {
        int i;
        for (i = 0; i < LIBXSMM_DISPATCH_CACHESIZE; ++i) buffer[i].pv = 0;
        { /* open scope for variable declarations */
          /* setup the dispatch table for the statically generated code */
#         include <libxsmm_dispatch.h>
        }
#if !defined(_OPENMP)
        { /* acquire and release remaining locks to shortcut any lazy initialization later on */
          const int nlocks = sizeof(libxsmm_dispatch_lock) / sizeof(*libxsmm_dispatch_lock);
          for (i = 1; i < nlocks; ++i) {
            LIBXSMM_LOCK_ACQUIRE(libxsmm_dispatch_lock[i]);
            LIBXSMM_LOCK_RELEASE(libxsmm_dispatch_lock[i]);
          }
        }
# if defined(__GNUC__)
        __atomic_store(&libxsmm_dispatch_cache, (libxsmm_dispatch_entry**)&buffer, __ATOMIC_RELAXED);
# else
        libxsmm_dispatch_cache = buffer;
# endif
#else
# if (201107 <= _OPENMP)
#       pragma omp atomic write
# endif
        libxsmm_dispatch_cache = buffer;
# if (201107 > _OPENMP)
#       pragma omp flush(libxsmm_dispatch_cache)
# endif
#endif
      }
    }
  }
#if !defined(_OPENMP)
  /* release the master lock */
  LIBXSMM_LOCK_RELEASE(libxsmm_dispatch_lock[LIBXSMM_DISPATCH_LOCKMASTER]);
#endif
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_init(void)
{
  /*const*/void* cache;
#if defined(_OPENMP)
# if (201107 <= _OPENMP)
# pragma omp atomic read
# else
# pragma omp flush(libxsmm_dispatch_cache)
# endif
  cache = libxsmm_dispatch_cache;
#elif defined(__GNUC__)
  __atomic_load((void**)&libxsmm_dispatch_cache, &cache, __ATOMIC_RELAXED);
#else
  cache = libxsmm_dispatch_cache;
#endif

  if (0 == cache) {
    internal_init();
  }
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_finalize(void)
{
  libxsmm_dispatch_entry* cache = 0;
#if defined(_OPENMP)
# if (201107 <= _OPENMP)
# pragma omp atomic read
# else
# pragma omp flush(libxsmm_dispatch_cache)
# endif
  cache = libxsmm_dispatch_cache;
#elif defined(__GNUC__)
  __atomic_load(&libxsmm_dispatch_cache, &cache, __ATOMIC_RELAXED);
#else
  cache = libxsmm_dispatch_cache;
#endif

  if (0 != cache) {
#if !defined(_OPENMP)
    /* acquire one of the locks as the master lock */
    LIBXSMM_LOCK_ACQUIRE(libxsmm_dispatch_lock[LIBXSMM_DISPATCH_LOCKMASTER]);
#else
#   pragma omp critical(libxsmm_dispatch_lock)
#endif
    {
#if defined(_OPENMP)
# if (201107 <= _OPENMP)
#     pragma omp atomic read
# else
#     pragma omp flush(libxsmm_dispatch_cache)
# endif
      cache = libxsmm_dispatch_cache;
#elif defined(__GNUC__)
      __atomic_load(&libxsmm_dispatch_cache, &cache, __ATOMIC_RELAXED);
#else
      cache = libxsmm_dispatch_cache;
#endif

      if (0 != cache) {
#if defined(_OPENMP)
# if (201107 <= _OPENMP)
#       pragma omp atomic write
# endif
        libxsmm_dispatch_cache = 0;
# if (201107 > _OPENMP)
#       pragma omp flush(libxsmm_dispatch_cache)
# endif
#elif defined(__GNUC__)
        /*const*/libxsmm_dispatch_entry* /*const*/zero = 0;
        __atomic_store(&libxsmm_dispatch_cache, &zero, __ATOMIC_RELAXED);
#else
        libxsmm_dispatch_cache = 0;
#endif
        free((void*)cache);
      }
    }
#if !defined(_OPENMP)
    /* release the master lock */
    LIBXSMM_LOCK_RELEASE(libxsmm_dispatch_lock[LIBXSMM_DISPATCH_LOCKMASTER]);
#endif
  }
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE const char* internal_supply_archid(void)
{
  unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
  const char* archid = 0;

  LIBXSMM_CPUID(0, eax, ebx, ecx, edx);
  if (7 <= eax) {
    LIBXSMM_CPUID(1, eax, ebx, ecx, edx);

    if (0x02000000 == (0x02000000 & ecx)) { /* XGETBV */
      LIBXSMM_XGETBV(0, eax, edx);

      if (0x04000000 == (0x04000000 & ecx)) { /* XSAVE */
        if (0x00000006 == (0x00000006 & eax)) { /* OS XSAVE 256-bit */
          if (0x000000E0 == (0x000000E0 & eax)) { /* OS XSAVE 512-bit */
            /* AVX512F, AVX512PF, AVX512ER, AVX512CD */
            if (0x1C010000 == (0x1C010000 & ebx)) {
              archid = "knl";
            }
            /* AVX512F, AVX512DQ, AVX512CD, AVX512BW, AVX512VL */
            else if (0xD0030000 == (0xD0030000 & ebx)) {
              archid = "skx";
            }
          }
          else if (0x08000000 == (0x08000000 & ecx)) { /* AVX */
            if (0x00001000 == (0x00001000 & ecx)) { /* FMA */
              archid = "hsw";
            }
            else {
              archid = "snb";
            }
          }
        }
      }
    }
  }

  return archid;
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE libxsmm_dispatch_entry internal_build(const libxsmm_gemm_descriptor* desc)
{
  libxsmm_dispatch_entry result, *cache;
  unsigned int hash, indx;
  assert(0 != desc);

#if defined(_OPENMP)
# if (201107 <= _OPENMP)
# pragma omp atomic read
# else
# pragma omp flush(libxsmm_dispatch_cache)
# endif
  cache = libxsmm_dispatch_cache;
#elif defined(__GNUC__)
  __atomic_load(&libxsmm_dispatch_cache, &cache, __ATOMIC_RELAXED);
#else
  cache = libxsmm_dispatch_cache;
#endif

  /* lazy initialization */
  if (0 == cache) {
    internal_init();
#if defined(_OPENMP)
# if (201107 <= _OPENMP)
#   pragma omp atomic read
# else
#   pragma omp flush(libxsmm_dispatch_cache)
# endif
    cache = libxsmm_dispatch_cache;
#elif defined(__GNUC__)
    __atomic_load(&libxsmm_dispatch_cache, &cache, __ATOMIC_RELAXED);
#else
    cache = libxsmm_dispatch_cache;
#endif
  }

  /* check if the requested xGEMM is already JITted */
  LIBXSMM_PRAGMA_FORCEINLINE /* must precede a statement */
  hash = libxsmm_crc32(desc, LIBXSMM_GEMM_DESCRIPTOR_SIZE, LIBXSMM_DISPATCH_HASH_SEED);
  indx = hash % LIBXSMM_DISPATCH_CACHESIZE;
  cache += indx;
  result = *cache; /* TODO: handle collision */

#if (0 != LIBXSMM_JIT)
  if (0 == result.pv) {
# if !defined(_WIN32) && (!defined(__CYGWIN__) || !defined(NDEBUG)/*allow code coverage with Cygwin; fails at runtime!*/)
# if !defined(_OPENMP)
    const unsigned int lock = LIBXSMM_MOD2(indx, sizeof(libxsmm_dispatch_lock) / sizeof(*libxsmm_dispatch_lock));
    LIBXSMM_LOCK_ACQUIRE(libxsmm_dispatch_lock[lock]);
# else
#   pragma omp critical(libxsmm_dispatch_lock)
# endif
    {
      result = *cache;

      if (0 == result.pv) {
        const char *const archid = internal_supply_archid();
        libxsmm_generated_code l_generated_code;
        void* l_code;

        if (0 != archid) {
          /* allocate buffer for code */
          l_generated_code.generated_code = malloc(131072 * sizeof(unsigned char));
          l_generated_code.buffer_size = 0 != l_generated_code.generated_code ? 131072 : 0;
          l_generated_code.code_size = 0;
          l_generated_code.code_type = 2;
          l_generated_code.last_error = 0;

          /* generate kernel */
          libxsmm_generator_dense_kernel(&l_generated_code, desc, archid);

          /* handle an eventual error in the else-branch */
          if (0 == l_generated_code.last_error) {
            /* create executable buffer */
            const int l_fd = open("/dev/zero", O_RDWR);
            /* must be a superset of what mprotect populates (see below) */
            const int perms = PROT_READ | PROT_WRITE | PROT_EXEC;
            l_code = mmap(0, l_generated_code.code_size, perms, MAP_PRIVATE, l_fd, 0);
            close(l_fd);

            if (MAP_FAILED != l_code) {
# if defined(MADV_NOHUGEPAGE)
#   if !defined(NDEBUG)
              const int error =
#   endif
              /* explicitly disable THP for this memory region, kernel 2.6.38 or higher */
              madvise(l_code, l_generated_code.code_size, MADV_NOHUGEPAGE);
              /* proceed even in case of an error, we then just take what we got (THP) */
#   if !defined(NDEBUG) /* library code is usually expected to be mute */
              if (-1 == error) fprintf(stderr, "LIBXSMM: failed to advise page size!\n");
#   endif
# endif /*MADV_NOHUGEPAGE*/
              /* copy temporary buffer into the prepared executable buffer */
              memcpy(l_code, l_generated_code.generated_code, l_generated_code.code_size);

              if (-1 != mprotect(l_code, l_generated_code.code_size, PROT_EXEC | PROT_READ)) {
# if !defined(NDEBUG)
                /* write buffer for manual decode as binary to a file */
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
# endif /*NDEBUG*/
                /* free temporary buffer */
                free(l_generated_code.generated_code);

                /* prepare return value */
                result.pv = l_code;

                /* make function pointer available for dispatch */
                cache->pv = l_code;
              }
              else { /* there was an error with mprotect */
# if !defined(NDEBUG) /* library code is usually expected to be mute */
                int error = errno;
                switch (error) {
                  case EINVAL: fprintf(stderr, "LIBXSMM: protecting memory failed (invalid pointer)!\n"); break;
                  case ENOMEM: fprintf(stderr, "LIBXSMM: protecting memory failed (kernel out of memory)\n"); break;
                  case EACCES: fprintf(stderr, "LIBXSMM: protecting memory failed (permission denied)!\n"); break;
                  default: fprintf(stderr, "LIBXSMM: protecting memory failed (unknown error)!\n");
                }
                error =
# endif /*NDEBUG*/
                munmap(l_code, l_generated_code.code_size);
# if !defined(NDEBUG) /* library code is usually expected to be mute */
                if (-1 == error) fprintf(stderr, "LIBXSMM: failed to unmap memory!\n");
# endif
                free(l_generated_code.generated_code);
              }
            }
            else {
# if !defined(NDEBUG) /* library code is usually expected to be mute */
              fprintf(stderr, "LIBXSMM: mapping memory failed!\n");
# endif /*NDEBUG*/
              free(l_generated_code.generated_code);
            }
          }
          else {
# if !defined(NDEBUG) /* library code is usually expected to be mute */
            fprintf(stderr, "%s\n", libxsmm_strerror(l_generated_code.last_error));
# endif /*NDEBUG*/
            free(l_generated_code.generated_code);
          }
        }
        else {
# if !defined(NDEBUG) /* library code is usually expected to be mute */
#   if defined(__SSE3__)
          fprintf(stderr, "LIBXSMM: SSE3 instruction set extension is not supported for JIT-code generation!\n");
#   elif defined(__MIC__)
          fprintf(stderr, "LIBXSMM: IMCI architecture (Xeon Phi coprocessor) is not supported for JIT-code generation!\n");
#   else
          fprintf(stderr, "LIBXSMM: no instruction set extension found for JIT-code generation!\n");
#   endif
# endif
        }
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


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_smmfunction libxsmm_smmdispatch(int m, int n, int k,
  const int* lda, const int* ldb, const int* ldc,
  const float* alpha, const float* beta,
  const int* flags, const int* prefetch)
{
  const int iflags = (0 == flags ? LIBXSMM_FLAGS : (*flags)) | LIBXSMM_GEMM_FLAG_F32PREC;
  const int ilda = (0 == lda ? m : (0 != *lda ? *lda
    /* if the value of lda was zero: make lda a multiple of LIBXSMM_ALIGNMENT */
    : LIBXSMM_ALIGN_VALUE(m, sizeof(*alpha), LIBXSMM_ALIGNMENT)));
  const int ildb = (0 == ldb ? k : *ldb);
  const int ildc = (0 == ldc ? LIBXSMM_LD(m, n) : (0 != *ldc ? LIBXSMM_LD(*ldc, n)
    /* if the value of ldc was zero: make ldc a multiple of LIBXSMM_ALIGNMENT */
    : LIBXSMM_ALIGN_VALUE(LIBXSMM_LD(m, n), sizeof(*alpha), LIBXSMM_ALIGNMENT)));

  LIBXSMM_GEMM_DESCRIPTOR_TYPE(desc, LIBXSMM_ALIGNMENT, iflags,
    LIBXSMM_LD(m, n), LIBXSMM_LD(n, ilda), LIBXSMM_LD(k, ildb),
    LIBXSMM_LD(ilda, n), LIBXSMM_LD(ildb, k), ildc,
    0 == alpha ? LIBXSMM_ALPHA : *alpha,
    0 == beta ? LIBXSMM_BETA : *beta,
    0 == prefetch ? LIBXSMM_PREFETCH : *prefetch);

  return internal_build(&desc).smm;
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_dmmfunction libxsmm_dmmdispatch(int m, int n, int k,
  const int* lda, const int* ldb, const int* ldc,
  const double* alpha, const double* beta,
  const int* flags, const int* prefetch)
{
  const int iflags = (0 == flags ? LIBXSMM_FLAGS : (*flags));
  const int ilda = (0 == lda ? m : (0 != *lda ? *lda
    /* if the value of lda was zero: make lda a multiple of LIBXSMM_ALIGNMENT */
    : LIBXSMM_ALIGN_VALUE(m, sizeof(*alpha), LIBXSMM_ALIGNMENT)));
  const int ildb = (0 == ldb ? k : *ldb);
  const int ildc = (0 == ldc ? LIBXSMM_LD(m, n) : (0 != *ldc ? LIBXSMM_LD(*ldc, n)
    /* if the value of ldc was zero: make ldc a multiple of LIBXSMM_ALIGNMENT */
    : LIBXSMM_ALIGN_VALUE(LIBXSMM_LD(m, n), sizeof(*alpha), LIBXSMM_ALIGNMENT)));

  LIBXSMM_GEMM_DESCRIPTOR_TYPE(desc, LIBXSMM_ALIGNMENT, iflags,
    LIBXSMM_LD(m, n), LIBXSMM_LD(n, ilda), LIBXSMM_LD(k, ildb),
    LIBXSMM_LD(ilda, n), LIBXSMM_LD(ildb, k), ildc,
    0 == alpha ? LIBXSMM_ALPHA : *alpha,
    0 == beta ? LIBXSMM_BETA : *beta,
    0 == prefetch ? LIBXSMM_PREFETCH : *prefetch);

  return internal_build(&desc).dmm;
}
