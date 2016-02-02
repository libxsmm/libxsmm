/******************************************************************************
** Copyright (c) 2014-2016, Intel Corporation                                **
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
#include <libxsmm.h>

#if defined(__TRACE)
# include "libxsmm_trace.h"
#endif

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
/* mute warning about target attribute; KNC/native plus JIT is disabled below! */
#include <libxsmm_generator.h>
#include <immintrin.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#if defined(_WIN32)
# include <Windows.h>
#else
# include <sys/mman.h>
# include <pthread.h>
# include <stdlib.h>
# include <unistd.h>
# include <fcntl.h>
#endif
#if !defined(NDEBUG)
#include <errno.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif
#if defined(__GNUC__)
# if !defined(LIBXSMM_GCCATOMICS)
#   if (40704 <= (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__))
#     define LIBXSMM_GCCATOMICS 1
#   else
#     define LIBXSMM_GCCATOMICS 0
#   endif
# endif
#endif

/* larger capacity of the registry lowers the probability of key collisions */
/*#define LIBXSMM_HASH_PRIME*/
#if defined(LIBXSMM_HASH_PRIME)
# define LIBXSMM_HASH_MOD(A, B) ((A) % (B))
#else
# define LIBXSMM_HASH_MOD(A, B) LIBXSMM_MOD2(A, B)
#endif

/* allow external definition to enable testing */
#if !defined(LIBXSMM_REGSIZE)
# if defined(LIBXSMM_HASH_PRIME)
#   define LIBXSMM_REGSIZE 999979
# else
#   define LIBXSMM_REGSIZE 1048576
# endif
#endif

/* flag fused into the memory address of a code version in case of collision */
#define LIBXSMM_HASH_COLLISION (1ULL << (8 * sizeof(void*) - 1))
#define LIBXSMM_HASH_SEED 0 /* CRC32 seed */

typedef union LIBXSMM_RETARGETABLE internal_code {
  libxsmm_smmfunction smm;
  libxsmm_dmmfunction dmm;
  /*const*/void* xmm;
  uintptr_t imm;
} internal_code;
typedef struct LIBXSMM_RETARGETABLE internal_regentry {
  libxsmm_gemm_descriptor descriptor;
  internal_code code;
  /* needed to distinct statically generated code and for munmap */
  unsigned int code_size;
} internal_regentry;
LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL internal_regentry* internal_registry = 0;
LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL const char* internal_jit = 0;
LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL int internal_has_crc32 = 0;

#if !defined(_OPENMP)
LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL LIBXSMM_LOCK_TYPE internal_reglock[] = {
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT,
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT,
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT,
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT
};
#endif


LIBXSMM_INLINE LIBXSMM_RETARGETABLE const char* internal_arch_name(int* is_static, int* has_crc32)
{
  unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
  const char* name = 0;

  if (is_static) *is_static = 0;

  LIBXSMM_CPUID(0, eax, ebx, ecx, edx);
  if (1 <= eax) { /* CPUID */
    LIBXSMM_CPUID(1, eax, ebx, ecx, edx);

    /* XSAVE/XGETBV(0x04000000), OSXSAVE(0x08000000) */
    if (0x0C000000 == (0x0C000000 & ecx)) {
      /* Check for CRC32 (this is not a proper test for SSE 4.2 as a whole!) */
      if (has_crc32) {
        *has_crc32 = (0x00100000 == (0x00100000 & ecx) ? 1 : 0);
#if defined(__SSE4_2__)
        assert(0 != *has_crc32); /* failed to detect CRC32 instruction */
#endif
      }
      LIBXSMM_XGETBV(0, eax, edx);

      if (0x00000006 == (0x00000006 & eax)) { /* OS XSAVE 256-bit */
        if (0x000000E0 == (0x000000E0 & eax)) { /* OS XSAVE 512-bit */
          LIBXSMM_CPUID(7, eax, ebx, ecx, edx);

          /* AVX512F(0x00010000), AVX512CD(0x10000000), AVX512PF(0x04000000),
             AVX512ER(0x08000000) */
          if (0x1C010000 == (0x1C010000 & ebx)) {
            name = "knl";
          }
          /* AVX512F(0x00010000), AVX512CD(0x10000000), AVX512DQ(0x00020000),
             AVX512BW(0x40000000), AVX512VL(0x80000000) */
          else if (0xD0030000 == (0xD0030000 & ebx)) {
            name = "skx";
          }

#if defined(__AVX512F__)
          if (is_static) *is_static = 1;
#endif
        }
        else if (0x10000000 == (0x10000000 & ecx)) { /* AVX(0x10000000) */
          if (0x00001000 == (0x00001000 & ecx)) { /* FMA(0x00001000) */
#if defined(__AVX512F__)
            assert(!"Failed to detect Intel AVX-512 extensions!");
#endif
#if defined(__AVX2__)
            if (is_static) *is_static = 1;
#endif
            name = "hsw";
          }
          else {
#if defined(__AVX2__)
            assert(!"Failed to detect Intel AVX2 extensions!");
#endif
#if defined(__AVX__)
            if (is_static) *is_static = 1;
#endif
            name = "snb";
          }
        }
      }
    }
  }
  else if (has_crc32) {
    *has_crc32 = 0;
  }

  return name;
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE internal_regentry* internal_init(void)
{
  /*const*/internal_regentry* result;
  int i;

#if !defined(_OPENMP)
  /* acquire locks and thereby shortcut lazy initialization later on */
  const int nlocks = sizeof(internal_reglock) / sizeof(*internal_reglock);
  for (i = 0; i < nlocks; ++i) LIBXSMM_LOCK_ACQUIRE(internal_reglock[i]);
#else
# pragma omp critical(internal_reglock)
#endif
  {
#if (defined(_REENTRANT) || defined(_OPENMP)) && defined(LIBXSMM_GCCATOMICS)
# if (0 != LIBXSMM_GCCATOMICS)
    result = __atomic_load_n(&internal_registry, __ATOMIC_SEQ_CST);
# else
    result = __sync_or_and_fetch(&internal_registry, 0);
# endif
#elif (defined(_REENTRANT) || defined(_OPENMP)) && defined(_WIN32)
    result = internal_registry; /*TODO*/
#else
    result = internal_registry;
#endif
    if (0 == result) {
#if defined(__TRACE)
      const char *const env_trace_init = getenv("LIBXSMM_TRACE");
      if (env_trace_init) {
        int match[] = { 0, 0 }, filter_threadid = 0, filter_mindepth = 1, filter_maxnsyms = -1;
        char buffer[32];

        if (1 == sscanf(env_trace_init, "%32[^,],", buffer)) {
          sscanf(buffer, "%i", &filter_threadid);
        }
        if (1 == sscanf(env_trace_init, "%*[^,],%32[^,],", buffer)) {
          match[0] = sscanf(buffer, "%i", &filter_mindepth);
        }
        if (1 == sscanf(env_trace_init, "%*[^,],%*[^,],%32s", buffer)) {
          match[1] = sscanf(buffer, "%i", &filter_maxnsyms);
        }
        i = (0 == filter_threadid && 0 == match[0] && 0 == match[1]) ? EXIT_SUCCESS
          : libxsmm_trace_init(filter_threadid - 1, filter_mindepth, filter_maxnsyms);
      }
      else
#endif
      {
        i = EXIT_SUCCESS;
      }
      if (EXIT_SUCCESS == i) {
        result = (internal_regentry*)malloc(LIBXSMM_REGSIZE * sizeof(internal_regentry));

        if (result) {
          int is_static = 0;
          /* decide using internal_has_crc32 instead of relying on a libxsmm_crc32_function pointer
           * to be able to inline the call instead of using an indirection (via fn. pointer)
           */
          const char *const arch_name = internal_arch_name(&is_static, &internal_has_crc32);
          for (i = 0; i < LIBXSMM_REGSIZE; ++i) result[i].code.xmm = 0;
          { /* omit registering code if JIT is enabled and if an ISA extension is found
             * which is beyond the static code path used to compile the library
             */
#if (0 != LIBXSMM_JIT) && !defined(__MIC__)
            const char *const env_jit = getenv("LIBXSMM_JIT");
            internal_jit = (0 == env_jit || 0 == *env_jit || '1' == *env_jit) ? arch_name : ('0' != *env_jit ? env_jit : 0);
            if (0 == internal_jit || 0 != is_static)
#endif
            { /* open scope for variable declarations */
              LIBXSMM_DEBUG(unsigned int csp = 0, cdp = 0);
              /* setup the dispatch table for the statically generated code */
#             include <libxsmm_dispatch.h>
#if !defined(NDEBUG) /* library code is expected to be mute */ && (0 != LIBXSMM_JIT)
              if (0 == arch_name && (0 == env_jit || '1' == *env_jit)) {
# if defined(__SSE3__)
                fprintf(stderr, "LIBXSMM: SSE3 instruction set extension is not supported for JIT-code generation!\n");
# elif defined(__MIC__)
                fprintf(stderr, "LIBXSMM: IMCI architecture (Xeon Phi coprocessor) is not supported for JIT-code generation!\n");
# else
                fprintf(stderr, "LIBXSMM: no instruction set extension found for JIT-code generation!\n");
# endif
              }
              if (0 < csp) {
                fprintf(stderr, "LIBXSMM: %u SP-kernels are not registered due to hash key collisions!\n", csp);
              }
              if (0 < cdp) {
                fprintf(stderr, "LIBXSMM: %u DP-kernels are not registered due to hash key collisions!\n", cdp);
              }
#endif
            }
          }
#if !defined(NDEBUG) && defined(__MIC__) /* library code is expected to be mute */ && (0 != LIBXSMM_JIT)
          if (0 == internal_has_crc32) {
            fprintf(stderr, "LIBXSMM: CRC32 instructions are not available!\n");
          }
#endif
          atexit(libxsmm_finalize);
#if (defined(_REENTRANT) || defined(_OPENMP)) && defined(LIBXSMM_GCCATOMICS)
# if (0 != LIBXSMM_GCCATOMICS)
          __atomic_store_n(&internal_registry, result, __ATOMIC_SEQ_CST);
# else
          {
            internal_regentry* old = internal_registry;
            while (!__sync_bool_compare_and_swap(&internal_registry, old, result)) old = internal_registry;
          }
# endif
#elif (defined(_REENTRANT) || defined(_OPENMP)) && defined(_WIN32)
          internal_registry = result; /*TODO*/
#else
          internal_registry = result;
#endif
        }
      }
#if !defined(NDEBUG) && defined(__TRACE) /* library code is expected to be mute */
      else {
        fprintf(stderr, "LIBXSMM: failed to initialize trace (%i)!\n", i);
      }
#endif
    }
  }
#if !defined(_OPENMP) /* release locks */
  for (i = 0; i < nlocks; ++i) LIBXSMM_LOCK_RELEASE(internal_reglock[i]);
#endif
  assert(result);
  return result;
}


LIBXSMM_EXTERN_C
#if defined(__GNUC__)
LIBXSMM_ATTRIBUTE(constructor)
#endif
LIBXSMM_RETARGETABLE void libxsmm_init(void)
{
#if (defined(_REENTRANT) || defined(_OPENMP)) && defined(LIBXSMM_GCCATOMICS)
# if (0 != LIBXSMM_GCCATOMICS)
  const void *const registry = __atomic_load_n(&internal_registry, __ATOMIC_RELAXED);
# else
  const void *const registry = __sync_or_and_fetch(&internal_registry, 0);
# endif
#elif (defined(_REENTRANT) || defined(_OPENMP)) && defined(_WIN32)
  const void *const registry = internal_registry; /*TODO*/
#else
  const void *const registry = internal_registry;
#endif
  if (0 == registry) {
    internal_init();
  }
}


LIBXSMM_EXTERN_C
#if defined(__GNUC__)
LIBXSMM_ATTRIBUTE(destructor)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
LIBXSMM_RETARGETABLE void libxsmm_finalize(void)
{
#if (defined(_REENTRANT) || defined(_OPENMP)) && defined(LIBXSMM_GCCATOMICS)
# if (0 != LIBXSMM_GCCATOMICS)
  internal_regentry* registry = __atomic_load_n(&internal_registry, __ATOMIC_SEQ_CST);
# else
  internal_regentry* registry = __sync_or_and_fetch(&internal_registry, 0);
# endif
#elif (defined(_REENTRANT) || defined(_OPENMP)) && defined(_WIN32)
  internal_regentry* registry = internal_registry; /*TODO*/
#else
  internal_regentry* registry = internal_registry;
#endif

  if (0 != registry) {
    int i;
#if !defined(_OPENMP)
    /* acquire locks and thereby shortcut lazy initialization later on */
    const int nlocks = sizeof(internal_reglock) / sizeof(*internal_reglock);
    for (i = 0; i < nlocks; ++i) LIBXSMM_LOCK_ACQUIRE(internal_reglock[i]);
#else
#   pragma omp critical(internal_reglock)
#endif
    {
      registry = internal_registry;

      if (0 != registry) {
#if defined(__TRACE)
        i = libxsmm_trace_finalize();
# if !defined(NDEBUG) /* library code is expected to be mute */
        if (EXIT_SUCCESS != i) {
          fprintf(stderr, "LIBXSMM: failed to finalize trace (%i)!\n", i);
        }
# endif
#endif
#if (defined(_REENTRANT) || defined(_OPENMP)) && defined(LIBXSMM_GCCATOMICS)
# if (0 != LIBXSMM_GCCATOMICS)
        __atomic_store_n(&internal_registry, 0, __ATOMIC_SEQ_CST);
# else
        { /* use store side-effect of built-in (dummy assignment to mute warning) */
          internal_regentry *const dummy = __sync_and_and_fetch(&internal_registry, 0);
          LIBXSMM_UNUSED(dummy);
        }
# endif
#elif (defined(_REENTRANT) || defined(_OPENMP)) && defined(_WIN32)
        internal_registry = 0; /*TODO*/
#else
        internal_registry = 0;
#endif
#if defined(_WIN32)
        /* TODO: to be implemented */
        LIBXSMM_UNUSED(i);
#else
        for (i = 0; i < LIBXSMM_REGSIZE; ++i) {
          const unsigned int code_size = registry[i].code_size;
          void *const code = registry[i].code.xmm;
          if (0 != code/*allocated*/ && 0 != code_size/*JIT*/) {
# if defined(NDEBUG)
            munmap(code, registry[i].code_size);
# else /* library code is expected to be mute */
            if (0 != munmap(code, code_size)) {
              static LIBXSMM_TLS int once = 0;
              if (0 == once) {
                fprintf(stderr, "LIBXSMM: %s (munmap error #%i)!\n", strerror(errno), errno);
                once = 1;
              }
            }
# endif
          }
        }
#endif /*defined(__GNUC__)*/
        free((void*)registry);
      }
    }
#if !defined(_OPENMP) /* release locks */
  for (i = 0; i < nlocks; ++i) LIBXSMM_LOCK_RELEASE(internal_reglock[i]);
#endif
  }
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_build(const libxsmm_gemm_descriptor* desc,
  void** code, unsigned int* code_size)
{
#if !defined(_WIN32) && !defined(__MIC__) && (!defined(__CYGWIN__) || !defined(NDEBUG)/*code-coverage with Cygwin; fails@runtime!*/)
  libxsmm_generated_code generated_code;
  assert(0 != desc && 0 != code && 0 != code_size);
  assert(0 != internal_jit);
  assert(0 == *code);

  /* allocate temporary buffer which is large enough to cover the generated code */
  generated_code.generated_code = malloc(131072 * sizeof(unsigned char));
  generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
  generated_code.code_size = 0;
  generated_code.code_type = 2;
  generated_code.last_error = 0;

  /* generate kernel */
  libxsmm_generator_dense_kernel(&generated_code, desc, internal_jit);

  /* handle an eventual error in the else-branch */
  if (0 == generated_code.last_error) {
#if defined(__APPLE__) && defined(__MACH__)
    const int fd = 0;
#else
    const int fd = open("/dev/zero", O_RDWR);
#endif
    if (0 <= fd) {
      /* create executable buffer */
      *code = mmap(0, generated_code.code_size,
        /* must be a superset of what mprotect populates (see below) */
        PROT_READ | PROT_WRITE | PROT_EXEC,
#if defined(__APPLE__) && defined(__MACH__)
        MAP_ANON | MAP_PRIVATE, fd, 0);
#else
        MAP_PRIVATE, fd, 0);
      close(fd);
#endif
      if (MAP_FAILED != *code) {
        /* explicitly disable THP for this memory region, kernel 2.6.38 or higher */
#if defined(MADV_NOHUGEPAGE)
# if defined(NDEBUG)
        madvise(*code, generated_code.code_size, MADV_NOHUGEPAGE);
# else /* library code is expected to be mute */
        /* proceed even in case of an error, we then just take what we got (THP) */
        if (0 != madvise(*code, generated_code.code_size, MADV_NOHUGEPAGE)) {
          static LIBXSMM_TLS int once = 0;
          if (0 == once) {
            fprintf(stderr, "LIBXSMM: %s (madvise error #%i)!\n", strerror(errno), errno);
            once = 1;
          }
        }
# endif /*defined(NDEBUG)*/
#elif !(defined(__APPLE__) && defined(__MACH__))
        LIBXSMM_MESSAGE("====================================================================")
        LIBXSMM_MESSAGE("Adjusting THP is unavailable due to C89 or kernel older than 2.6.38!")
        LIBXSMM_MESSAGE("====================================================================")
#endif /*MADV_NOHUGEPAGE*/
        /* copy temporary buffer into the prepared executable buffer */
        memcpy(*code, generated_code.generated_code, generated_code.code_size);

        if (0/*ok*/ == mprotect(*code, generated_code.code_size, PROT_EXEC | PROT_READ)) {
#if !defined(NDEBUG) && defined(_DEBUG)
          /* write buffer for manual decode as binary to a file */
          char objdump_name[512];
          FILE* byte_code;
          sprintf(objdump_name, "kernel_%s_f%i_%c%c_m%u_n%u_k%u_lda%u_ldb%u_ldc%u_a%i_b%i_pf%i.bin",
            internal_jit /* best available/supported code path */,
            0 == (LIBXSMM_GEMM_FLAG_F32PREC & desc->flags) ? 64 : 32,
            0 == (LIBXSMM_GEMM_FLAG_TRANS_A & desc->flags) ? 'n' : 't',
            0 == (LIBXSMM_GEMM_FLAG_TRANS_B & desc->flags) ? 'n' : 't',
            desc->m, desc->n, desc->k, desc->lda, desc->ldb, desc->ldc,
            desc->alpha, desc->beta, desc->prefetch);
          byte_code = fopen(objdump_name, "wb");
          if (0 != byte_code) {
            fwrite(generated_code.generated_code, 1, generated_code.code_size, byte_code);
            fclose(byte_code);
          }
#endif /*!defined(NDEBUG) && defined(_DEBUG)*/
          /* free temporary/initial code buffer */
          free(generated_code.generated_code);
          /* finalize code generation */
          *code_size = generated_code.code_size;
        }
        else { /* there was an error with mprotect */
#if defined(NDEBUG)
          munmap(*code, generated_code.code_size);
#else /* library code is expected to be mute */
          static LIBXSMM_TLS int once = 0;
          if (0 == once) {
            fprintf(stderr, "LIBXSMM: %s (mprotect error #%i)!\n", strerror(errno), errno);
            once = 1;
          }
          if (0 != munmap(*code, generated_code.code_size)) {
            static LIBXSMM_TLS int once_mmap_error = 0;
            if (0 == once_mmap_error) {
              fprintf(stderr, "LIBXSMM: %s (munmap error #%i)!\n", strerror(errno), errno);
              once_mmap_error = 1;
            }
          }
#endif
          free(generated_code.generated_code);
        }
      }
      else {
#if !defined(NDEBUG) /* library code is expected to be mute */
        static LIBXSMM_TLS int once = 0;
        if (0 == once) {
          fprintf(stderr, "LIBXSMM: %s (mmap error #%i)!\n", strerror(errno), errno);
          once = 1;
        }
#endif
        free(generated_code.generated_code);
        /* clear MAP_FAILED value */
        *code = 0;
      }
    }
#if !defined(NDEBUG)/* library code is expected to be mute */
    else {
      static LIBXSMM_TLS int once = 0;
      if (0 == once) {
        fprintf(stderr, "LIBXSMM: invalid file descriptor (%i)\n", fd);
        once = 1;
      }
    }
#endif
  }
  else {
#if !defined(NDEBUG) /* library code is expected to be mute */
    static LIBXSMM_TLS int once = 0;
    if (0 == once) {
      fprintf(stderr, "%s (error #%i)\n", libxsmm_strerror(generated_code.last_error), generated_code.last_error);
      once = 1;
    }
#endif
    free(generated_code.generated_code);
  }
#elif !defined(__MIC__)
  LIBXSMM_UNUSED(desc); LIBXSMM_UNUSED(code); LIBXSMM_UNUSED(code_size);
  LIBXSMM_MESSAGE("======================================================")
  LIBXSMM_MESSAGE("The JIT BACKEND is not supported on Windows right now!")
  LIBXSMM_MESSAGE("======================================================")
#else
  LIBXSMM_UNUSED(desc); LIBXSMM_UNUSED(code); LIBXSMM_UNUSED(code_size);
#endif /*_WIN32*/
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE unsigned int internal_gemmdiff(
  const libxsmm_gemm_descriptor* a, const libxsmm_gemm_descriptor* b)
{
  const unsigned *const ia = (const unsigned int*)a, *const ib = (const unsigned int*)b;
  unsigned int result, i;
  assert(0 != a && 0 != b && 0 == LIBXSMM_MOD2(LIBXSMM_GEMM_DESCRIPTOR_SIZE, sizeof(unsigned int)));

  result = ia[0] ^ ib[0];
  for (i = 1; i < LIBXSMM_DIV2(LIBXSMM_GEMM_DESCRIPTOR_SIZE, sizeof(unsigned int)); ++i) {
    result |= (ia[i] ^ ib[i]);
  }

  return result;
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE internal_code internal_find_code(const libxsmm_gemm_descriptor* desc)
{
  internal_code result;
  unsigned int hash, i, diff = 0;
  unsigned int diff0 = 0, i0;

#if (defined(_REENTRANT) || defined(_OPENMP)) && defined(LIBXSMM_GCCATOMICS)
# if (0 != LIBXSMM_GCCATOMICS)
  internal_regentry* entry = __atomic_load_n(&internal_registry, __ATOMIC_RELAXED);
# else
  internal_regentry* entry = __sync_or_and_fetch(&internal_registry, 0);
# endif
#elif (defined(_REENTRANT) || defined(_OPENMP)) && defined(_WIN32)
  internal_regentry* entry = internal_registry; /*TODO*/
#else
  internal_regentry* entry = internal_registry;
#endif
  assert(0 != desc);

  /* lazy initialization */
  if (0 == entry) {
    /* use init's return value to refresh local representation */
    entry = internal_init();
  }

  /* check if the requested xGEMM is already JITted */
#if defined(__SSE4_2__)
  LIBXSMM_PRAGMA_FORCEINLINE /* must precede a statement */
  hash = libxsmm_crc32_sse42(desc, LIBXSMM_GEMM_DESCRIPTOR_SIZE, LIBXSMM_HASH_SEED);
#else
  if (0 != internal_has_crc32) {
    LIBXSMM_PRAGMA_FORCEINLINE /* must precede a statement */
    hash = libxsmm_crc32_sse42(desc, LIBXSMM_GEMM_DESCRIPTOR_SIZE, LIBXSMM_HASH_SEED);
  }
  else {
    LIBXSMM_PRAGMA_FORCEINLINE /* must precede a statement */
    hash = libxsmm_crc32(desc, LIBXSMM_GEMM_DESCRIPTOR_SIZE, LIBXSMM_HASH_SEED);
  }
#endif
  i = i0 = LIBXSMM_HASH_MOD(hash, LIBXSMM_REGSIZE);
  entry += i; /* actual entry */

  do {
    /* read registered code */
#if (defined(_REENTRANT) || defined(_OPENMP)) && defined(LIBXSMM_GCCATOMICS)
# if (0 != LIBXSMM_GCCATOMICS)
    result.xmm = __atomic_load_n(&entry->code.xmm, __ATOMIC_SEQ_CST);
# else
    result.xmm = __sync_or_and_fetch(&entry->code.xmm, 0);
# endif
#elif (defined(_REENTRANT) || defined(_OPENMP)) && defined(_WIN32)
    result = entry->code; /*TODO*/
#else
    result = entry->code;
#endif

    /* entire block is conditional wrt LIBXSMM_JIT; static code currently does not have collisions */
    if (0 != result.xmm) {
      if (0 == diff0) {
        if (0 == (LIBXSMM_HASH_COLLISION & result.imm)) { /* check for no collision */
          /* calculate bitwise difference (deep check) */
          diff = internal_gemmdiff(desc, &entry->descriptor);
          if (0 != diff) { /* new collision discovered (but no code version yet) */
            /* allow to fixup current entry inside of the guarded/locked region */
            result.xmm = 0;
          }
        }
        /* collision discovered but code version exists; perform intial deep check */
        else if (0 != internal_gemmdiff(desc, &entry->descriptor)) {
          /* continue linearly searching code starting at re-hashed index position */
          const unsigned int index = LIBXSMM_HASH_MOD(LIBXSMM_HASH_VALUE(hash), LIBXSMM_REGSIZE);
          unsigned int next;
          internal_regentry *const registry = entry - i; /* recalculate base address */
          for (i0 = (index != i ? index : LIBXSMM_HASH_MOD(index + 1, LIBXSMM_REGSIZE)),
            i = i0, next = LIBXSMM_HASH_MOD(i0 + 1, LIBXSMM_REGSIZE); next != i0/*no code found*/ &&
            /* skip any (still invalid) descriptor which corresponds to no code, or continue on diff */
            (0 == (entry = registry + i)->code.xmm || 0 != (diff = internal_gemmdiff(desc, &entry->descriptor)));
            i = next, next = LIBXSMM_HASH_MOD(i + 1, LIBXSMM_REGSIZE));
          if (0 == diff) { /* found exact code version; continue with atomic load */
            continue;
          }
          else { /* no code found */
            result.xmm = 0;
            break;
          }
        }
        else { /* clear the uppermost bit of the address */
          result.imm &= ~LIBXSMM_HASH_COLLISION;
        }
      }
      else { /* new collision discovered (but no code version yet) */
        result.xmm = 0;
      }
    }

    /* check if code generation or fixup is needed, also check whether JIT is supported (CPUID) */
    if (0 == result.xmm && 0 != internal_jit) {
      /* attempt to lock the registry entry */
# if !defined(_OPENMP)
      const unsigned int lock = LIBXSMM_MOD2(i, sizeof(internal_reglock) / sizeof(*internal_reglock));
      LIBXSMM_LOCK_ACQUIRE(internal_reglock[lock]);
# else
#     pragma omp critical(internal_reglock)
# endif
      {
        /* re-read registry entry after acquiring the lock */
        if (0 == diff) result = entry->code;

        if (0 == result.xmm) { /* double-check after acquiring the lock */
          if (0 == diff) {
            /* found a conflict-free registry-slot, and attempt to build the kernel */
            internal_build(desc, &result.xmm, &entry->code_size);

            if (0 != result.xmm) { /* synchronize registry entry */
              entry->descriptor = *desc;
# if (defined(_REENTRANT) || defined(_OPENMP)) && defined(LIBXSMM_GCCATOMICS)
#   if (0 != LIBXSMM_GCCATOMICS)
              __atomic_store_n(&entry->code.xmm, result.xmm, __ATOMIC_SEQ_CST);
#   else
              {
                /*const*/void* old = entry->code.xmm;
                while (!__sync_bool_compare_and_swap(&entry->code.xmm, old, result.xmm)) old = entry->code.xmm;
              }
#   endif
# elif (defined(_REENTRANT) || defined(_OPENMP)) && defined(_WIN32)
              entry->code.xmm = result.xmm; /*TODO*/
# else
              entry->code.xmm = result.xmm;
# endif
            }
          }
          else { /* 0 != diff */
            const unsigned int base = i;

            if (0 == diff0) {
              /* flag existing entry as collision */
              /*const*/ void * /*const*/ code = (void*)(entry->code.imm | LIBXSMM_HASH_COLLISION);

              /* find new slot to store the code version */
              const unsigned int index = LIBXSMM_HASH_MOD(LIBXSMM_HASH_VALUE(hash), LIBXSMM_REGSIZE);
              i = (index != i ? index : LIBXSMM_HASH_MOD(index + 1, LIBXSMM_REGSIZE));
              i0 = i; /* keep starting point of free-slot-search in mind */

              /* fixup existing entry */
# if (defined(_REENTRANT) || defined(_OPENMP)) && defined(LIBXSMM_GCCATOMICS)
#   if (0 != LIBXSMM_GCCATOMICS)
              __atomic_store_n(&entry->code.xmm, code, __ATOMIC_SEQ_CST);
#   else
              {
                /*const*/void* old = entry->code.xmm;
                while (!__sync_bool_compare_and_swap(&entry->code.xmm, old, code)) old = entry->code.xmm;
              }
#   endif
# elif (defined(_REENTRANT) || defined(_OPENMP)) && defined(_WIN32)
              entry->code.xmm = code; /*TODO*/
# else
              entry->code.xmm = code;
# endif
              diff0 = diff; /* no more fixup */
            }
            else {
              const unsigned int next = LIBXSMM_HASH_MOD(i + 1, LIBXSMM_REGSIZE);
              if (next != i0) { /* linear search for free slot */
                i = next;
              }
              else { /* out of registry capacity (no free slot found) */
                diff = 0;
              }
            }

            entry -= base; /* recalculate base address */
            entry += i;
          }
        }
      }
# if !defined(_OPENMP)
      LIBXSMM_LOCK_RELEASE(internal_reglock[lock]);
# endif
    }
    else {
      diff = 0;
    }
  }
  while (0 != diff);

  assert(0 == result.xmm || 0 == internal_gemmdiff(desc, &entry->descriptor));
  return result;
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_smmfunction libxsmm_smmdispatch(int m, int n, int k,
  const int* lda, const int* ldb, const int* ldc,
  const float* alpha, const float* beta,
  const int* flags, const int* prefetch)
{
  const int iflags = (0 == flags ? LIBXSMM_FLAGS : (*flags)) | LIBXSMM_GEMM_FLAG_F32PREC;
  const int ilda = (0 == lda ? LIBXSMM_LD(m, k) : *lda);
  const int ildb = (0 == ldb ? LIBXSMM_LD(k, n) : *ldb);
  const int ildc = (0 == ldc ? LIBXSMM_LD(m, n) : *ldc);

  LIBXSMM_GEMM_DESCRIPTOR_TYPE(desc, LIBXSMM_ALIGNMENT, iflags,
    LIBXSMM_LD(m, n), LIBXSMM_LD(n, m), k,
    LIBXSMM_LD(ilda, ildb), LIBXSMM_LD(ildb, ilda), ildc,
    0 == alpha ? LIBXSMM_ALPHA : *alpha,
    0 == beta ? LIBXSMM_BETA : *beta,
    0 == prefetch ? LIBXSMM_PREFETCH : *prefetch);

  return internal_find_code(&desc).smm;
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_dmmfunction libxsmm_dmmdispatch(int m, int n, int k,
  const int* lda, const int* ldb, const int* ldc,
  const double* alpha, const double* beta,
  const int* flags, const int* prefetch)
{
  const int iflags = (0 == flags ? LIBXSMM_FLAGS : (*flags));
  const int ilda = (0 == lda ? LIBXSMM_LD(m, k) : *lda);
  const int ildb = (0 == ldb ? LIBXSMM_LD(k, n) : *ldb);
  const int ildc = (0 == ldc ? LIBXSMM_LD(m, n) : *ldc);

  LIBXSMM_GEMM_DESCRIPTOR_TYPE(desc, LIBXSMM_ALIGNMENT, iflags,
    LIBXSMM_LD(m, n), LIBXSMM_LD(n, m), k,
    LIBXSMM_LD(ilda, ildb), LIBXSMM_LD(ildb, ilda), ildc,
    0 == alpha ? LIBXSMM_ALPHA : *alpha,
    0 == beta ? LIBXSMM_BETA : *beta,
    0 == prefetch ? LIBXSMM_PREFETCH : *prefetch);

  return internal_find_code(&desc).dmm;
}
