/******************************************************************************
** Copyright (c) 2014-2017, Intel Corporation                                **
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
/* Hans Pabst, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "libxsmm_gemm_diff.h"
#include "libxsmm_trans.h"
#include "libxsmm_gemm.h"
#include "libxsmm_hash.h"
#include "libxsmm_main.h"
#if defined(__TRACE)
# include "libxsmm_trace.h"
#endif
#if defined(LIBXSMM_PERF)
# include "libxsmm_perf.h"
#endif
#include <libxsmm_intrinsics_x86.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
/* mute warning about target attribute; KNC/native plus JIT is disabled below! */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if !defined(NDEBUG)
# include <errno.h>
#endif
#if defined(_WIN32)
# include <Windows.h>
#else
# include <sys/mman.h>
# include <unistd.h>
# include <fcntl.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

/* alternative hash algorithm (instead of CRC32) */
#if !defined(LIBXSMM_HASH_BASIC)
# if (LIBXSMM_X86_SSE4 > LIBXSMM_MAX_STATIC_TARGET_ARCH)
/*#   define LIBXSMM_HASH_BASIC*/
# endif
#endif

/* LIBXSMM_CAPACITY_REGISTRY is POT */
/*#define LIBXSMM_HASH_MOD(N, NGEN) ((N) % (NGEN))*/
#define LIBXSMM_HASH_MOD(N, NPOT) LIBXSMM_MOD2(N, NPOT)

#if !defined(LIBXSMM_CAPACITY_CACHE)
# define LIBXSMM_CAPACITY_CACHE 4
#endif

#if defined(LIBXSMM_HASH_BASIC)
# define LIBXSMM_HASH_FUNCTION_CALL(HASH, INDX, DESCRIPTOR) \
    HASH = libxsmm_hash_npot(&(DESCRIPTOR), LIBXSMM_GEMM_DESCRIPTOR_SIZE, LIBXSMM_CAPACITY_REGISTRY); \
    assert((LIBXSMM_CAPACITY_REGISTRY) > (HASH)); \
    INDX = (HASH)
#else
# define LIBXSMM_HASH_FUNCTION_CALL(HASH, INDX, DESCRIPTOR) \
    HASH = libxsmm_crc32(&(DESCRIPTOR), LIBXSMM_GEMM_DESCRIPTOR_SIZE, 25071975/*seed*/); \
    INDX = LIBXSMM_HASH_MOD(HASH, LIBXSMM_CAPACITY_REGISTRY)
#endif

/* flag fused into the memory address of a code version in case of non-JIT */
#define LIBXSMM_CODE_STATIC (1ULL << (8 * sizeof(void*) - 1))
/* flag fused into the memory address of a code version in case of collision */
#if 0 /* disabled due to no performance advantage */
#define LIBXSMM_HASH_COLLISION (1ULL << (8 * sizeof(void*) - 2))
#endif

#if 16 >= (LIBXSMM_GEMM_DESCRIPTOR_SIZE)
# define LIBXSMM_GEMM_DESCRIPTOR_SIMD_SIZE 16
#elif 32 >= (LIBXSMM_GEMM_DESCRIPTOR_SIZE)
# define LIBXSMM_GEMM_DESCRIPTOR_SIMD_SIZE 32
#else
# define LIBXSMM_GEMM_DESCRIPTOR_SIMD_SIZE LIBXSMM_GEMM_DESCRIPTOR_SIZE
#endif

typedef union LIBXSMM_RETARGETABLE internal_regkey_type {
  char simd[LIBXSMM_GEMM_DESCRIPTOR_SIMD_SIZE];
  libxsmm_gemm_descriptor descriptor;
} internal_regkey_type;

typedef struct LIBXSMM_RETARGETABLE internal_statistic_type {
  unsigned int ntry, ncol, njit, nsta;
} internal_statistic_type;

/** Helper macro determining the default prefetch strategy which is used for statically generated kernels. */
#if (0 > LIBXSMM_PREFETCH) /* auto-prefetch (frontend) */
# define INTERNAL_PREFETCH LIBXSMM_PREFETCH_NONE
#else
# define INTERNAL_PREFETCH LIBXSMM_PREFETCH
#endif

#if defined(LIBXSMM_NO_SYNC)
# define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX, DIFF, CODE)
# define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX)
#else
# define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX, DIFF, CODE) { \
  const unsigned int LOCKINDEX = LIBXSMM_MOD2(INDEX, INTERNAL_REGLOCK_COUNT); \
  if (LIBXSMM_LOCK_ACQUIRED != LIBXSMM_LOCK_TRYLOCK(internal_reglock + (LOCKINDEX))) { \
    if (0 == libxsmm_dispatch_trylock) { /* (re-)try and get (meanwhile) generated code */ \
      continue; \
    } \
    else { /* exit dispatch and let client fall back */ \
      DIFF = 0; CODE = 0; \
      break; \
    } \
  }
# define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX) LIBXSMM_LOCK_RELEASE(internal_reglock + (LOCKINDEX)); }
#endif

#if defined(LIBXSMM_GEMM_DIFF_SW) && (2 == (LIBXSMM_GEMM_DIFF_SW)) /* most general implementation */
# define INTERNAL_FIND_CODE_CACHE_INDEX(CACHE_HIT, RESULT_INDEX) \
    RESULT_INDEX = ((CACHE_HIT) + ((LIBXSMM_CAPACITY_CACHE) - 1)) % (LIBXSMM_CAPACITY_CACHE)
#else
# define INTERNAL_FIND_CODE_CACHE_INDEX(CACHE_HIT, RESULT_INDEX) \
    assert(/*is pot*/(LIBXSMM_CAPACITY_CACHE) == (1 << LIBXSMM_LOG2(LIBXSMM_CAPACITY_CACHE))); \
    RESULT_INDEX = LIBXSMM_MOD2((CACHE_HIT) + ((LIBXSMM_CAPACITY_CACHE) - 1), LIBXSMM_CAPACITY_CACHE)
#endif

#define INTERNAL_DISPATCH_MAIN(TYPE, DESCRIPTOR_DECL, DESC, PFLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH) { \
  const int internal_dispatch_main_flags_ = (0 == (PFLAGS) ? LIBXSMM_FLAGS : *(PFLAGS)) | LIBXSMM_GEMM_TYPEFLAG(TYPE); \
  const int internal_dispatch_main_lda_ = (0 == LIBXSMM_LD(PLDA, PLDB) ? LIBXSMM_LD(M, N) : *LIBXSMM_LD(PLDA, PLDB)); \
  const int internal_dispatch_main_ldb_ = (0 == LIBXSMM_LD(PLDB, PLDA) ? (K) : *LIBXSMM_LD(PLDB, PLDA)); \
  const int internal_dispatch_main_ldc_ = (0 == (PLDC) ? LIBXSMM_LD(M, N) : *(PLDC)); \
  const TYPE internal_dispatch_main_alpha_ = (0 == (PALPHA) ? ((TYPE)LIBXSMM_ALPHA) : *(PALPHA)); \
  const TYPE internal_dispatch_main_beta_ = (0 == (PBETA) ? ((TYPE)LIBXSMM_BETA) : *(PBETA)); \
  if (LIBXSMM_GEMM_NO_BYPASS(internal_dispatch_main_flags_, internal_dispatch_main_alpha_, internal_dispatch_main_beta_) && LIBXSMM_GEMM_NO_BYPASS_DIMS(M, N, K) && \
    LIBXSMM_GEMM_NO_BYPASS_DIMS(internal_dispatch_main_lda_, internal_dispatch_main_ldb_, internal_dispatch_main_ldc_)) \
  { \
    const int internal_dispatch_main_prefetch_ = (0 == (PREFETCH) ? libxsmm_gemm_auto_prefetch : *(PREFETCH)); \
    DESCRIPTOR_DECL; LIBXSMM_GEMM_DESCRIPTOR(*(DESC), 0 != (VECTOR_WIDTH) ? (VECTOR_WIDTH): LIBXSMM_ALIGNMENT, \
      internal_dispatch_main_flags_, LIBXSMM_LD(M, N), LIBXSMM_LD(N, M), K, internal_dispatch_main_lda_, internal_dispatch_main_ldb_, internal_dispatch_main_ldc_, \
      (signed char)(internal_dispatch_main_alpha_), (signed char)(internal_dispatch_main_beta_), \
      (0 > internal_dispatch_main_prefetch_ ? internal_gemm_auto_prefetch : internal_dispatch_main_prefetch_)); \
    { \
      return internal_find_code(DESC).LIBXSMM_TPREFIX(TYPE, mm); \
    } \
  } \
  else { /* bypass (not supported) */ \
    /* libxsmm_gemm_print is not suitable here since A, B, and C are unknown at this point */ \
    libxsmm_update_mmstatistic(internal_dispatch_main_flags_, LIBXSMM_LD(M, N), LIBXSMM_LD(N, M), K, 1/*try*/, 0); \
    return 0; \
  } \
}

#if defined(LIBXSMM_GEMM_DIFF_MASK_A) /* no padding i.e., LIBXSMM_GEMM_DESCRIPTOR_SIZE */
# define INTERNAL_DISPATCH(TYPE, PFLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH) \
    INTERNAL_DISPATCH_MAIN(TYPE, libxsmm_gemm_descriptor descriptor = { 0 }, &descriptor, \
      PFLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH)
#else /* padding: LIBXSMM_GEMM_DESCRIPTOR_SIZE -> LIBXSMM_ALIGNMENT */
# define INTERNAL_DISPATCH(TYPE, PFLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH) \
    INTERNAL_DISPATCH_MAIN(TYPE, union { libxsmm_gemm_descriptor desc; char simd[LIBXSMM_ALIGNMENT]; } simd_descriptor; int i; \
      for (i = LIBXSMM_GEMM_DESCRIPTOR_SIZE; i < sizeof(simd_descriptor.simd); ++i) simd_descriptor.simd[i] = 0, &simd_descriptor.desc, \
      PFLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH)
#endif

#if !defined(LIBXSMM_NO_SYNC)
# define INTERNAL_REGLOCK_COUNT 256
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE LIBXSMM_LOCK_TYPE internal_reglock[INTERNAL_REGLOCK_COUNT];
#endif

LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE internal_regkey_type* internal_registry_keys;
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_code_pointer* internal_registry;
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE internal_statistic_type internal_statistic[2/*DP/SP*/][4/*sml/med/big/xxx*/];
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE unsigned int internal_statistic_sml;
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE unsigned int internal_statistic_med;
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE unsigned int internal_statistic_mnk;
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE unsigned int internal_teardown;
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE size_t internal_heapmem;
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int internal_dispatch_trylock_locked;
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int internal_gemm_auto_prefetch_locked;
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int internal_gemm_auto_prefetch;


LIBXSMM_API_DEFINITION unsigned int libxsmm_update_mmstatistic(int flags, int m, int n, int k, unsigned int ntry, unsigned int ncol)
{
  const unsigned long long kernel_size = LIBXSMM_MNK_SIZE(m, n, k);
  const int precision = (0 == (LIBXSMM_GEMM_FLAG_F32PREC & flags) ? 0 : 1);
  int bucket = 3/*huge*/;

  if (LIBXSMM_MNK_SIZE(internal_statistic_sml, internal_statistic_sml, internal_statistic_sml) >= kernel_size) {
    bucket = 0;
  }
  else if (LIBXSMM_MNK_SIZE(internal_statistic_med, internal_statistic_med, internal_statistic_med) >= kernel_size) {
    bucket = 1;
  }
  else if (LIBXSMM_MNK_SIZE(internal_statistic_mnk, internal_statistic_mnk, internal_statistic_mnk) >= kernel_size) {
    bucket = 2;
  }

  LIBXSMM_ATOMIC_ADD_FETCH(&internal_statistic[precision][bucket].ncol, ncol, LIBXSMM_ATOMIC_RELAXED);
  return LIBXSMM_ATOMIC_ADD_FETCH(&internal_statistic[precision][bucket].ntry, ntry, LIBXSMM_ATOMIC_RELAXED);
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE unsigned int internal_update_mmstatistic(const libxsmm_gemm_descriptor* desc,
  unsigned int ntry, unsigned int ncol)
{
  assert(0 != desc);
  return libxsmm_update_mmstatistic(desc->flags, desc->m, desc->n, desc->k, ntry, ncol);
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE const char* internal_get_target_arch(int id);
LIBXSMM_INLINE LIBXSMM_RETARGETABLE const char* internal_get_target_arch(int id)
{
  const char* target_arch = 0;
  switch (id) {
    case LIBXSMM_X86_AVX512_CORE: {
      target_arch = "skx";
    } break;
    case LIBXSMM_X86_AVX512_KNM: {
      target_arch = "knm";
    } break;
    case LIBXSMM_X86_AVX512_MIC: {
      target_arch = "knl";
    } break;
    case LIBXSMM_X86_AVX512: {
      target_arch = "avx3";
    } break;
    case LIBXSMM_X86_AVX2: {
      target_arch = "hsw";
    } break;
    case LIBXSMM_X86_AVX: {
      target_arch = "snb";
    } break;
    case LIBXSMM_X86_SSE4: {
      target_arch = "wsm";
    } break;
    case LIBXSMM_X86_SSE3: {
      target_arch = "sse3";
    } break;
    case LIBXSMM_TARGET_ARCH_GENERIC: {
      target_arch = "generic";
    } break;
    default: if (LIBXSMM_X86_GENERIC <= id) {
      target_arch = "x86";
    }
    else {
      target_arch = "unknown";
    }
  }

  assert(0 != target_arch);
  return target_arch;
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE unsigned int internal_print_number(unsigned int n, char default_unit, char* unit)
{
  unsigned int number = n;
  assert(0 != unit);
  *unit = default_unit;
  if ((1000000) <= n) {
    number = (n + 500000) / 1000000;
    *unit = 'm';
  }
  else if (9999 < n) {
    number = (n + 500) / 1000;
    *unit = 'k';
  }
  return number;
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE unsigned int internal_print_statistic(FILE* ostream,
  const char* target_arch, int precision, unsigned int linebreaks, unsigned int indent)
{
  const internal_statistic_type statistic_sml = internal_statistic[precision][0/*SML*/];
  const internal_statistic_type statistic_med = internal_statistic[precision][1/*MED*/];
  const internal_statistic_type statistic_big = internal_statistic[precision][2/*BIG*/];
  const internal_statistic_type statistic_xxx = internal_statistic[precision][3/*XXX*/];
  int printed = 0;
  assert(0 != ostream && 0 != target_arch && (0 <= precision && precision < 2));

  if (/* omit to print anything if it is superfluous */
    0 != statistic_sml.ntry || 0 != statistic_sml.njit || 0 != statistic_sml.nsta || 0 != statistic_sml.ncol ||
    0 != statistic_med.ntry || 0 != statistic_med.njit || 0 != statistic_med.nsta || 0 != statistic_med.ncol ||
    0 != statistic_big.ntry || 0 != statistic_big.njit || 0 != statistic_big.nsta || 0 != statistic_big.ncol ||
    0 != statistic_xxx.ntry || 0 != statistic_xxx.njit || 0 != statistic_xxx.nsta || 0 != statistic_xxx.ncol)
  {
    char title[256], range[256], unit[4];
    unsigned int counter[4];
    {
      unsigned int n;
      assert(strlen(target_arch) < sizeof(title));
      for (n = 0; 0 != target_arch[n] /*avoid code-gen. issue with some clang versions: && n < sizeof(title)*/; ++n) {
        const char c = target_arch[n];
        title[n] = (char)(('a' <= c && c <= 'z') ? (c - 32) : c); /* toupper */
      }
      LIBXSMM_SNPRINTF(title + n, sizeof(title) - n, "/%s", 0 == precision ? "DP" : "SP");
      for (n = 0; n < linebreaks; ++n) fprintf(ostream, "\n");
    }
    fprintf(ostream, "%*s%-8s %6s %6s %6s %6s\n", (int)indent, "", title, "TRY" ,"JIT", "STA", "COL");
    LIBXSMM_SNPRINTF(range, sizeof(range), "%u..%u", 0u, internal_statistic_sml);
    counter[0] = internal_print_number(statistic_sml.ntry, ' ', unit + 0);
    counter[1] = internal_print_number(statistic_sml.njit, ' ', unit + 1);
    counter[2] = internal_print_number(statistic_sml.nsta, ' ', unit + 2);
    counter[3] = internal_print_number(statistic_sml.ncol, ' ', unit + 3);
    fprintf(ostream, "%*s%8s %6u%c %5u%c %5u%c %5u%c\n", (int)indent, "", range,
      counter[0], unit[0], counter[1], unit[1], counter[2], unit[2], counter[3], unit[3]);
    LIBXSMM_SNPRINTF(range, sizeof(range), "%u..%u", internal_statistic_sml + 1u, internal_statistic_med);
    counter[0] = internal_print_number(statistic_med.ntry, ' ', unit + 0);
    counter[1] = internal_print_number(statistic_med.njit, ' ', unit + 1);
    counter[2] = internal_print_number(statistic_med.nsta, ' ', unit + 2);
    counter[3] = internal_print_number(statistic_med.ncol, ' ', unit + 3);
    fprintf(ostream, "%*s%8s %6u%c %5u%c %5u%c %5u%c\n", (int)indent, "", range,
      counter[0], unit[0], counter[1], unit[1], counter[2], unit[2], counter[3], unit[3]);
    LIBXSMM_SNPRINTF(range, sizeof(range), "%u..%u", internal_statistic_med + 1u, internal_statistic_mnk);
    counter[0] = internal_print_number(statistic_big.ntry, ' ', unit + 0);
    counter[1] = internal_print_number(statistic_big.njit, ' ', unit + 1);
    counter[2] = internal_print_number(statistic_big.nsta, ' ', unit + 2);
    counter[3] = internal_print_number(statistic_big.ncol, ' ', unit + 3);
    fprintf(ostream, "%*s%8s %6u%c %5u%c %5u%c %5u%c\n", (int)indent, "", range,
      counter[0], unit[0], counter[1], unit[1], counter[2], unit[2], counter[3], unit[3]);
    if (0 != statistic_xxx.ntry || 0 != statistic_xxx.njit || 0 != statistic_xxx.nsta || 0 != statistic_xxx.ncol) {
      LIBXSMM_SNPRINTF(range, sizeof(range), "> %u", internal_statistic_mnk);
      counter[0] = internal_print_number(statistic_xxx.ntry, ' ', unit + 0);
      counter[1] = internal_print_number(statistic_xxx.njit, ' ', unit + 1);
      counter[2] = internal_print_number(statistic_xxx.nsta, ' ', unit + 2);
      counter[3] = internal_print_number(statistic_xxx.ncol, ' ', unit + 3);
      fprintf(ostream, "%*s%8s %6u%c %5u%c %5u%c %5u%c\n", (int)indent, "", range,
        counter[0], unit[0], counter[1], unit[1], counter[2], unit[2], counter[3], unit[3]);
    }
    printed = 1;
  }

  return printed;
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE unsigned int internal_statistic_ntry(int precision)
{
  return internal_statistic[precision][0/*SML*/].ntry + internal_statistic[precision][1/*MED*/].ntry
       + internal_statistic[precision][2/*BIG*/].ntry + internal_statistic[precision][3/*XXX*/].ntry;
}


LIBXSMM_API void internal_register_static_code(const libxsmm_gemm_descriptor*,
  unsigned int, unsigned int, libxsmm_xmmfunction, libxsmm_code_pointer*);
LIBXSMM_API_DEFINITION void internal_register_static_code(const libxsmm_gemm_descriptor* desc,
  unsigned int index, unsigned int hash, libxsmm_xmmfunction src, libxsmm_code_pointer* registry)
{
  internal_regkey_type* dst_key = internal_registry_keys + index;
  libxsmm_code_pointer* dst_entry = registry + index;
#if !defined(NDEBUG)
  libxsmm_code_pointer code; code.xmm = src;
  assert(0 != desc && 0 != code.const_pmm && 0 != dst_key && 0 != registry);
  assert(0 == (LIBXSMM_CODE_STATIC & code.imm));
#endif

  if (0 != dst_entry->const_pmm) { /* collision? */
    /* start at a re-hashed index position */
    const unsigned int start = LIBXSMM_HASH_MOD(LIBXSMM_HASH_VALUE(hash), LIBXSMM_CAPACITY_REGISTRY);
    unsigned int i0, i, next;
#if defined(LIBXSMM_HASH_COLLISION)
    /* mark current entry as a collision (this might be already the case) */
    dst_entry->imm |= LIBXSMM_HASH_COLLISION;
#endif
    /* start linearly searching for an available slot */
    for (i = (start != index) ? start : LIBXSMM_HASH_MOD(start + 1, LIBXSMM_CAPACITY_REGISTRY), i0 = i, next = LIBXSMM_HASH_MOD(i + 1, LIBXSMM_CAPACITY_REGISTRY);
      0 != registry[i].const_pmm && next != i0; i = next, next = LIBXSMM_HASH_MOD(i + 1, LIBXSMM_CAPACITY_REGISTRY));

    /* calculate destinations */
    dst_key = internal_registry_keys + i;
    dst_entry = registry + i;

    internal_update_mmstatistic(desc, 0, 1/*collision*/);
  }

  if (0 == dst_entry->const_pmm) { /* registry not (yet) exhausted */
    dst_key->descriptor = *desc;
    dst_entry->xmm = src;
    /* mark current entry as static code (non-JIT) */
    dst_entry->imm |= LIBXSMM_CODE_STATIC;
  }

  internal_update_mmstatistic(desc, 1/*try*/, 0);
}


LIBXSMM_API_DEFINITION int libxsmm_gemm_prefetch2uid(libxsmm_gemm_prefetch_type prefetch)
{
  switch (prefetch) {
    case LIBXSMM_PREFETCH_SIGONLY:            return 2;
    case LIBXSMM_PREFETCH_BL2_VIA_C:          return 3;
    case LIBXSMM_PREFETCH_AL2_AHEAD:          return 4;
    case LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD: return 5;
    case LIBXSMM_PREFETCH_AL2:                return 6;
    case LIBXSMM_PREFETCH_AL2BL2_VIA_C:       return 7;
    case LIBXSMM_PREFETCH_AL2_JPST:           return 8;
    case LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST:  return 9;
    case LIBXSMM_PREFETCH_AL2CL2BL2_VIA_C:    return 10;
    default: {
      assert(LIBXSMM_PREFETCH_NONE == prefetch);
      return 0;
    }
  }
}


LIBXSMM_API_DEFINITION libxsmm_gemm_prefetch_type libxsmm_gemm_uid2prefetch(int uid)
{
  switch (uid) {
    case  2: return LIBXSMM_PREFETCH_SIGONLY;             /* pfsigonly */
    case  3: return LIBXSMM_PREFETCH_BL2_VIA_C;           /* BL2viaC */
    case  4: return LIBXSMM_PREFETCH_AL2_AHEAD;           /* curAL2 */
    case  5: return LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD;  /* curAL2_BL2viaC */
    case  6: return LIBXSMM_PREFETCH_AL2;                 /* AL2 */
    case  7: return LIBXSMM_PREFETCH_AL2BL2_VIA_C;        /* AL2_BL2viaC */
    case  8: return LIBXSMM_PREFETCH_AL2_JPST;            /* AL2jpst */
    case  9: return LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST;   /* AL2jpst_BL2viaC */
    case 10: return LIBXSMM_PREFETCH_AL2CL2BL2_VIA_C;     /* AL2_BL2viaC_CL2 */
    default: return LIBXSMM_PREFETCH_NONE;
  }
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_finalize(void)
{
  libxsmm_finalize();
  if (0 != libxsmm_verbosity) { /* print statistic on termination */
    fflush(stdout); /* synchronize with standard output */
    {
      const char *const target_arch = internal_get_target_arch(libxsmm_target_archid);
      const unsigned int linebreak = (0 == internal_print_statistic(stderr, target_arch, 1/*SP*/, 1, 0)) ? 1 : 0;
      if (0 == internal_print_statistic(stderr, target_arch, 0/*DP*/, linebreak, 0) && 0 != linebreak) {
        fprintf(stderr, "LIBXSMM_TARGET=%s ", target_arch);
      }
      fprintf(stderr, "HEAP: %.f MB\n", 1.0 * internal_heapmem / (1 << 20));
    }
  }
  {
    size_t n = 0;
    /* release scratch memory pool */
    libxsmm_release_scratch(&n);
    if (0 < n && 0 != libxsmm_verbosity) { /* library code is expected to be mute */
      fprintf(stderr, "LIBXSMM: pending scratch-memory allocations discovered!\n");
    }
#if !defined(LIBXSMM_NO_SYNC) /* release locks */
    for (n = 0; n < INTERNAL_REGLOCK_COUNT; ++n) LIBXSMM_LOCK_DESTROY(internal_reglock + n);
    LIBXSMM_LOCK_DESTROY(&libxsmm_lock_global);
#endif
  }
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_init(void)
{
  libxsmm_code_pointer* result;
  int init_code = EXIT_FAILURE, i;
#if !defined(LIBXSMM_NO_SYNC) /* setup the locks in a thread-safe fashion */
  for (i = 0; i < INTERNAL_REGLOCK_COUNT; ++i) LIBXSMM_LOCK_ACQUIRE(internal_reglock + i);
  LIBXSMM_LOCK_ACQUIRE(&libxsmm_lock_global);
#endif
  result = internal_registry;
  if (0 == result) {
    const libxsmm_malloc_function null_malloc_fn = { 0 };
    const libxsmm_free_function null_free_fn = { 0 };
    libxsmm_xset_default_allocator(0/*lock*/, 0/*context*/, null_malloc_fn, null_free_fn);
    libxsmm_xset_scratch_allocator(0/*lock*/, 0/*context*/, null_malloc_fn, null_free_fn);
    libxsmm_set_target_arch(getenv("LIBXSMM_TARGET")); /* set libxsmm_target_archid */
    libxsmm_mt = 2;
    { /* behavior of parallelized routines which are located in libxsmmext library
       * 0: sequential below-threshold routine (no OpenMP); may fall-back to BLAS,
       * 1: (OpenMP-)parallelized but without internal parallel region,
       * 2: (OpenMP-)parallelized with internal parallel region"
       */
      const char *const env = getenv("LIBXSMM_MT");
      if (0 != env && 0 != *env) {
        libxsmm_mt = atoi(env);
      }
    }
    { const char *const env = getenv("LIBXSMM_TASKS");
      if (0 != env && 0 != *env) {
        libxsmm_tasks = atoi(env);
      }
    }
    { const char *const env = getenv("LIBXSMM_TRYLOCK");
      if (0 != env && 0 != *env) {
        libxsmm_dispatch_trylock = atoi(env);
        internal_dispatch_trylock_locked = 1;
      }
    }
    /* clear internal counters/statistic */
    for (i = 0; i < 4/*sml/med/big/xxx*/; ++i) {
      memset(&internal_statistic[0/*DP*/][i], 0, sizeof(internal_statistic_type));
      memset(&internal_statistic[1/*SP*/][i], 0, sizeof(internal_statistic_type));
    }
    libxsmm_nt = 2;
#if !defined(__MIC__) && (LIBXSMM_X86_AVX512_MIC != LIBXSMM_STATIC_TARGET_ARCH)
    if (LIBXSMM_X86_AVX512_MIC == libxsmm_target_archid)
#endif
    {
      libxsmm_nt = 4;
    }
    {
      const char *const env = getenv("LIBXSMM_VERBOSE");
      internal_statistic_mnk = (unsigned int)(pow((double)(LIBXSMM_MAX_MNK), 0.3333333333333333) + 0.5);
      internal_statistic_sml = 13; internal_statistic_med = 23;
      if (0 != env && 0 != *env) {
        libxsmm_verbosity = atoi(env);
      }
#if !defined(NDEBUG)
      else {
        libxsmm_verbosity = INT_MAX - 1; /* quiet -> verbose */
      }
#endif
    }
#if !defined(__TRACE)
    LIBXSMM_UNUSED(init_code);
#else
    {
      int filter_threadid = 0, filter_mindepth = 1, filter_maxnsyms = 0;
      const char *const env = getenv("LIBXSMM_TRACE");
      if (0 != env && 0 != *env) {
        char buffer[32];
        if (1 == sscanf(env, "%32[^,],", buffer)) {
          sscanf(buffer, "%i", &filter_threadid);
        }
        if (1 == sscanf(env, "%*[^,],%32[^,],", buffer)) {
          sscanf(buffer, "%i", &filter_mindepth);
        }
        if (1 == sscanf(env, "%*[^,],%*[^,],%32s", buffer)) {
          sscanf(buffer, "%i", &filter_maxnsyms);
        }
        else {
          filter_maxnsyms = -1; /* all */
        }
      }
      init_code = libxsmm_trace_init(filter_threadid - 1, filter_mindepth, filter_maxnsyms);
    }
    if (EXIT_SUCCESS == init_code)
#endif
    {
      libxsmm_gemm_diff_init(libxsmm_target_archid);
      libxsmm_trans_init(libxsmm_target_archid);
      libxsmm_hash_init(libxsmm_target_archid);
#if defined(LIBXSMM_PERF)
      libxsmm_perf_init();
#endif
      assert(0 == internal_registry_keys && 0 == internal_registry); /* should never happen */
      result = (libxsmm_code_pointer*)malloc((LIBXSMM_CAPACITY_REGISTRY) * sizeof(libxsmm_code_pointer));
      internal_registry_keys = (internal_regkey_type*)malloc((LIBXSMM_CAPACITY_REGISTRY) * sizeof(internal_regkey_type));
      if (0 != result && 0 != internal_registry_keys) {
        const char *const env = getenv("LIBXSMM_GEMM_PREFETCH");
        for (i = 0; i < (LIBXSMM_CAPACITY_REGISTRY); ++i) result[i].pmm = 0;
        /* omit registering code if JIT is enabled and if an ISA extension is found
         * which is beyond the static code path used to compile the library
         */
#if defined(LIBXSMM_BUILD)
# if (0 != LIBXSMM_JIT) && !defined(__MIC__)
        /* check if target arch. permits execution (arch. may be overridden) */
        if (LIBXSMM_STATIC_TARGET_ARCH <= libxsmm_target_archid &&
           (LIBXSMM_X86_AVX > libxsmm_target_archid /* JIT code gen. is not available */
            /* condition allows to avoid JIT (if static code is good enough) */
         || LIBXSMM_STATIC_TARGET_ARCH == libxsmm_target_archid))
# endif
        { /* opening a scope for eventually declaring variables */
          /* setup the dispatch table for the statically generated code */
#           include <libxsmm_dispatch.h>
        }
#endif
        internal_gemm_auto_prefetch = (0 == internal_statistic_ntry(0/*DP*/) && 0 == internal_statistic_ntry(1/*SP*/))
          /* avoid special prefetch if static code is present, since such code uses INTERNAL_PREFETCH */
          ? (LIBXSMM_X86_AVX512_MIC != libxsmm_target_archid ? LIBXSMM_PREFETCH_AL2BL2_VIA_C : LIBXSMM_PREFETCH_BL2_VIA_C)
          : INTERNAL_PREFETCH;
        libxsmm_gemm_auto_prefetch = INTERNAL_PREFETCH;
        if (0 != env && 0 != *env) { /* user input beyond auto-prefetch is always considered */
          const int uid = atoi(env);
          if (0 <= uid) {
            internal_gemm_auto_prefetch = libxsmm_gemm_uid2prefetch(uid);
            libxsmm_gemm_auto_prefetch = internal_gemm_auto_prefetch;
            internal_gemm_auto_prefetch_locked = 1;
          }
        }
        libxsmm_gemm_init(libxsmm_target_archid, libxsmm_gemm_auto_prefetch);
        if (0 == internal_teardown) {
          atexit(internal_finalize);
        }
        {
          void *const pv_registry = &internal_registry;
          LIBXSMM_ATOMIC_STORE((void**)pv_registry, (void*)result, LIBXSMM_ATOMIC_SEQ_CST);
        }
      }
      else {
        if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
          fprintf(stderr, "LIBXSMM: failed to allocate code registry!\n");
        }
        free(internal_registry_keys);
        free(result);
      }
    }
#if defined(__TRACE)
    else if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
      fprintf(stderr, "LIBXSMM: failed to initialize TRACE (error #%i)!\n", init_code);
    }
#endif
  }
#if !defined(LIBXSMM_NO_SYNC) /* release locks */
  for (i = 0; i < INTERNAL_REGLOCK_COUNT; ++i) LIBXSMM_LOCK_RELEASE(internal_reglock + i);
  LIBXSMM_LOCK_RELEASE(&libxsmm_lock_global);
#endif
}


LIBXSMM_API_DEFINITION LIBXSMM_ATTRIBUTE_CTOR void libxsmm_init(void)
{
  const void *const registry = LIBXSMM_ATOMIC_LOAD(&internal_registry, LIBXSMM_ATOMIC_RELAXED);
  if (0 == registry) {
#if !defined(LIBXSMM_NO_SYNC) /* setup the locks in a thread-safe fashion */
    static int reglock_check = 0;
    int i;
    assert(sizeof(internal_reglock) == (INTERNAL_REGLOCK_COUNT * sizeof(*internal_reglock)));
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&reglock_check, 1, LIBXSMM_ATOMIC_SEQ_CST)) {
      for (i = 0; i < INTERNAL_REGLOCK_COUNT; ++i) LIBXSMM_LOCK_INIT(internal_reglock + i);
      LIBXSMM_LOCK_INIT(&libxsmm_lock_global);
    }
    else { /* wait until locks are initialized, or until shutdown */
      while (0 == internal_registry && 0 == internal_teardown) {
        if (0 != LIBXSMM_ATOMIC_LOAD(&internal_registry, LIBXSMM_ATOMIC_RELAXED)) break;
        if (0 != LIBXSMM_ATOMIC_LOAD(&internal_teardown, LIBXSMM_ATOMIC_RELAXED)) break;
      }
    }
#endif
    internal_init();
  }
}


LIBXSMM_API
#if defined(__GNUC__)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
void libxsmm_finalize(void);

LIBXSMM_API_DEFINITION LIBXSMM_ATTRIBUTE_DTOR void libxsmm_finalize(void)
{
  libxsmm_code_pointer* registry = LIBXSMM_ATOMIC_LOAD(&internal_registry, LIBXSMM_ATOMIC_SEQ_CST);
  if (0 != registry) {
    int i;
#if !defined(LIBXSMM_NO_SYNC)
    /* acquire locks and thereby shortcut lazy initialization later on */
    for (i = 0; i < INTERNAL_REGLOCK_COUNT; ++i) LIBXSMM_LOCK_ACQUIRE(internal_reglock + i);
#endif
    registry = internal_registry;

    if (0 != registry) {
      internal_regkey_type *const registry_keys = internal_registry_keys;
      internal_heapmem = (LIBXSMM_CAPACITY_REGISTRY) * (sizeof(libxsmm_code_pointer) + sizeof(internal_regkey_type));

      /* serves as an id to invalidate the thread-local cache; never decremented */
      ++internal_teardown;
#if defined(__TRACE)
      i = libxsmm_trace_finalize();
      if (EXIT_SUCCESS != i && 0 != libxsmm_verbosity) { /* library code is expected to be mute */
        fprintf(stderr, "LIBXSMM: failed to finalize trace (error #%i)!\n", i);
      }
#endif
      libxsmm_gemm_finalize();
      libxsmm_gemm_diff_finalize();
      libxsmm_trans_finalize();
      libxsmm_hash_finalize();
#if defined(LIBXSMM_PERF)
      libxsmm_perf_finalize();
#endif
      /* make internal registry globally unavailable */
      LIBXSMM_ATOMIC_STORE_ZERO(&internal_registry, LIBXSMM_ATOMIC_SEQ_CST);
      internal_registry_keys = 0;

      for (i = 0; i < (LIBXSMM_CAPACITY_REGISTRY); ++i) {
        libxsmm_code_pointer code = registry[i];
        if (0 != code.const_pmm) {
          const libxsmm_gemm_descriptor *const desc = &registry_keys[i].descriptor;
          const unsigned long long kernel_size = LIBXSMM_MNK_SIZE(desc->m, desc->n, desc->k);
          const int precision = (0 == (LIBXSMM_GEMM_FLAG_F32PREC & desc->flags) ? 0 : 1);
          int bucket = 3/*huge*/;
          assert(0 < kernel_size);
          if (LIBXSMM_MNK_SIZE(internal_statistic_sml, internal_statistic_sml, internal_statistic_sml) >= kernel_size) {
            bucket = 0;
          }
          else if (LIBXSMM_MNK_SIZE(internal_statistic_med, internal_statistic_med, internal_statistic_med) >= kernel_size) {
            bucket = 1;
          }
          else if (LIBXSMM_MNK_SIZE(internal_statistic_mnk, internal_statistic_mnk, internal_statistic_mnk) >= kernel_size) {
            bucket = 2;
          }
          if (0 == (LIBXSMM_CODE_STATIC & code.imm)) { /* check for allocated/generated JIT-code */
            void* buffer = 0;
            size_t size = 0;
#if defined(LIBXSMM_HASH_COLLISION)
            code.imm &= ~LIBXSMM_HASH_COLLISION; /* clear collision flag */
#endif
            if (EXIT_SUCCESS == libxsmm_malloc_info(code.const_pmm, &size, 0/*flags*/, &buffer)) {
              libxsmm_xfree(code.const_pmm);
              ++internal_statistic[precision][bucket].njit;
              internal_heapmem += (unsigned int)(size + (((char*)code.const_pmm) - (char*)buffer));
            }
          }
          else {
            ++internal_statistic[precision][bucket].nsta;
          }
        }
      }
      free(registry_keys);
      free(registry);
    }
#if !defined(LIBXSMM_NO_SYNC) /* LIBXSMM_LOCK_RELEASE, but no LIBXSMM_LOCK_DESTROY */
    for (i = 0; i < INTERNAL_REGLOCK_COUNT; ++i) LIBXSMM_LOCK_RELEASE(internal_reglock + i);
#endif
  }
  /* release scratch memory pool */
  libxsmm_release_scratch(0);
}


LIBXSMM_API_DEFINITION int libxsmm_get_target_archid(void)
{
  LIBXSMM_INIT
#if !defined(__MIC__) && (!defined(__CYGWIN__) || !defined(NDEBUG)/*code-coverage with Cygwin; fails@runtime!*/)
  return libxsmm_target_archid;
#else /* no JIT support */
  return LIBXSMM_MIN(libxsmm_target_archid, LIBXSMM_X86_SSE4);
#endif
}


LIBXSMM_API_DEFINITION void libxsmm_set_target_archid(int id)
{
  int target_archid = LIBXSMM_TARGET_ARCH_UNKNOWN;
  switch (id) {
    case LIBXSMM_X86_AVX512_KNM:
    case LIBXSMM_X86_AVX512_CORE:
    case LIBXSMM_X86_AVX512_MIC:
    case LIBXSMM_X86_AVX512:
    case LIBXSMM_X86_AVX2:
    case LIBXSMM_X86_AVX:
    case LIBXSMM_X86_SSE4:
    case LIBXSMM_X86_SSE3:
    case LIBXSMM_TARGET_ARCH_GENERIC: {
      target_archid = id;
    } break;
    default: if (LIBXSMM_X86_GENERIC <= id) {
      target_archid = LIBXSMM_X86_GENERIC;
    }
    else {
      target_archid = libxsmm_cpuid();
    }
  }
  LIBXSMM_ATOMIC_STORE(&libxsmm_target_archid, target_archid, LIBXSMM_ATOMIC_RELAXED);
  if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
    const int cpuid = libxsmm_cpuid();
    if (cpuid < libxsmm_target_archid) {
      const char *const target_arch = internal_get_target_arch(libxsmm_target_archid);
      fprintf(stderr, "LIBXSMM: \"%s\" code will fail to run on \"%s\"!\n",
        target_arch, internal_get_target_arch(cpuid));
    }
  }
}


LIBXSMM_API_DEFINITION const char* libxsmm_get_target_arch(void)
{
  LIBXSMM_INIT
  return internal_get_target_arch(libxsmm_target_archid);
}


/* function serves as a helper for implementing the Fortran interface */
LIBXSMM_API const char* get_target_arch(int* length);
LIBXSMM_API_DEFINITION const char* get_target_arch(int* length)
{
  const char *const arch = libxsmm_get_target_arch();
  /* valid here since function is not in the public interface */
  assert(0 != arch && 0 != length);
  *length = (int)strlen(arch);
  return arch;
}


LIBXSMM_API_DEFINITION void libxsmm_set_target_arch(const char* arch)
{
  int target_archid = LIBXSMM_TARGET_ARCH_UNKNOWN;
  if (0 != arch && 0 != *arch) {
    const int jit = atoi(arch);
    if (0 == strcmp("0", arch)) {
      target_archid = LIBXSMM_TARGET_ARCH_GENERIC;
    }
    else if (1 < jit) {
      target_archid = LIBXSMM_X86_GENERIC + jit;
    }
    else if (0 == strcmp("skx", arch) || 0 == strcmp("skl", arch)) {
      target_archid = LIBXSMM_X86_AVX512_CORE;
    }
    else if (0 == strcmp("knm", arch) || 0 == strcmp("mic2", arch)) {
      target_archid = LIBXSMM_X86_AVX512_KNM;
    }
    else if (0 == strcmp("knl", arch) || 0 == strcmp("mic", arch)) {
      target_archid = LIBXSMM_X86_AVX512_MIC;
    }
    else if (0 == strcmp("avx3", arch) || 0 == strcmp("avx512", arch)) {
      target_archid = LIBXSMM_X86_AVX512;
    }
    else if (0 == strcmp("hsw", arch) || 0 == strcmp("avx2", arch)) {
      target_archid = LIBXSMM_X86_AVX2;
    }
    else if (0 == strcmp("snb", arch) || 0 == strcmp("avx", arch)) {
      target_archid = LIBXSMM_X86_AVX;
    }
    else if (0 == strcmp("wsm", arch) || 0 == strcmp("nhm", arch) || 0 == strcmp("sse4", arch) || 0 == strcmp("sse4_2", arch) || 0 == strcmp("sse4.2", arch)) {
      target_archid = LIBXSMM_X86_SSE4;
    }
    else if (0 == strcmp("sse3", arch) || 0 == strcmp("sse", arch)) {
      target_archid = LIBXSMM_X86_SSE3;
    }
    else if (0 == strcmp("x86", arch) || 0 == strcmp("sse2", arch)) {
      target_archid = LIBXSMM_X86_GENERIC;
    }
    else if (0 == strcmp("generic", arch) || 0 == strcmp("none", arch)) {
      target_archid = LIBXSMM_TARGET_ARCH_GENERIC;
    }
  }

  if (LIBXSMM_TARGET_ARCH_UNKNOWN == target_archid || LIBXSMM_X86_AVX512_KNM < target_archid) {
    target_archid = libxsmm_cpuid();
  }
  else if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
    const int cpuid = libxsmm_cpuid();
    if (cpuid < target_archid) {
      const char *const target_arch = internal_get_target_arch(target_archid);
      fprintf(stderr, "LIBXSMM: \"%s\" code will fail to run on \"%s\"!\n",
        target_arch, internal_get_target_arch(cpuid));
    }
  }
  LIBXSMM_ATOMIC_STORE(&libxsmm_target_archid, target_archid, LIBXSMM_ATOMIC_RELAXED);
}


LIBXSMM_API_DEFINITION int libxsmm_get_verbosity(void)
{
  LIBXSMM_INIT
  return libxsmm_verbosity;
}


LIBXSMM_API_DEFINITION void libxsmm_set_verbosity(int level)
{
  LIBXSMM_INIT
  LIBXSMM_ATOMIC_STORE(&libxsmm_verbosity, level, LIBXSMM_ATOMIC_RELAXED);
}


LIBXSMM_API_DEFINITION int libxsmm_get_dispatch_trylock(void)
{
  LIBXSMM_INIT
  return libxsmm_dispatch_trylock;
}


LIBXSMM_API_DEFINITION void libxsmm_set_dispatch_trylock(int trylock)
{
  LIBXSMM_INIT
  if (0 == internal_dispatch_trylock_locked) { /* LIBXSMM_TRYLOCK environment takes precedence */
    LIBXSMM_ATOMIC_STORE(&libxsmm_dispatch_trylock, trylock, LIBXSMM_ATOMIC_RELAXED);
  }
}


LIBXSMM_API_DEFINITION libxsmm_gemm_prefetch_type libxsmm_get_gemm_auto_prefetch(void)
{
  return (libxsmm_gemm_prefetch_type)libxsmm_gemm_auto_prefetch;
}


LIBXSMM_API_DEFINITION void libxsmm_set_gemm_auto_prefetch(libxsmm_gemm_prefetch_type strategy)
{
  if (0 == internal_gemm_auto_prefetch_locked) { /* LIBXSMM_GEMM_PREFETCH environment takes precedence */
    LIBXSMM_ATOMIC_STORE(&internal_gemm_auto_prefetch, strategy, LIBXSMM_ATOMIC_RELAXED);
    LIBXSMM_ATOMIC_STORE(&libxsmm_gemm_auto_prefetch, strategy, LIBXSMM_ATOMIC_RELAXED);
  }
}


LIBXSMM_API const char* internal_get_precision_string(libxsmm_dnn_datatype);
LIBXSMM_API_DEFINITION const char* internal_get_precision_string(libxsmm_dnn_datatype datatype)
{
  const char* result = "unk"; /* unknown */
  switch (datatype) {
    case LIBXSMM_DNN_DATATYPE_F32: result = "f32"; break;
    case LIBXSMM_DNN_DATATYPE_I32: result = "i32"; break;
    case LIBXSMM_DNN_DATATYPE_I16: result = "i16"; break;
    case LIBXSMM_DNN_DATATYPE_I8:  result = "i8";  break;
  }
  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_build(const libxsmm_build_request* request, unsigned int regindex, libxsmm_code_pointer* code)
{
  int result = EXIT_SUCCESS;
#if !defined(__MIC__) && (!defined(__CYGWIN__) || !defined(NDEBUG)/*code-coverage with Cygwin; fails@runtime!*/)
  const char *const target_arch = internal_get_target_arch(libxsmm_target_archid);
  libxsmm_generated_code generated_code;
  char jit_name[256] = { 0 };

  assert(0 != request && 0 != libxsmm_target_archid);
  assert(0 != code && 0 == code->const_pmm);
  /* setup code generation */
  memset(&generated_code, 0, sizeof(generated_code));
  generated_code.code_type = 2;

  switch (request->kind) { /* generate kernel */
    case LIBXSMM_BUILD_KIND_GEMM: { /* small MxM kernel */
      assert(0 != request->descriptor.gemm);
      if (0 < request->descriptor.gemm->m   && 0 < request->descriptor.gemm->n   && 0 < request->descriptor.gemm->k &&
          0 < request->descriptor.gemm->lda && 0 < request->descriptor.gemm->ldb && 0 < request->descriptor.gemm->ldc)
      {
        generated_code.generated_code = malloc(131072); /* large enough temporary buffer for generated code */
        generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
        LIBXSMM_NO_OFFLOAD(void, libxsmm_generator_gemm_kernel, &generated_code, request->descriptor.gemm, target_arch);
# if !defined(LIBXSMM_VTUNE)
        if (0 > libxsmm_verbosity)
# endif
        {
          const int uid = libxsmm_gemm_prefetch2uid((libxsmm_gemm_prefetch_type)request->descriptor.gemm->prefetch);
          /* adopt scheme which allows kernel names of LIBXSMM to appear in order (Intel VTune, etc.) */
          LIBXSMM_SNPRINTF(jit_name, sizeof(jit_name), "libxsmm_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_a%i_b%i_p%i.mxm", target_arch/*code path name*/,
            0 == (LIBXSMM_GEMM_FLAG_F32PREC & request->descriptor.gemm->flags) ? "f64" : "f32",
            0 == (LIBXSMM_GEMM_FLAG_TRANS_A & request->descriptor.gemm->flags) ? 'n' : 't',
            0 == (LIBXSMM_GEMM_FLAG_TRANS_B & request->descriptor.gemm->flags) ? 'n' : 't',
            (unsigned int)request->descriptor.gemm->m,   (unsigned int)request->descriptor.gemm->n,   (unsigned int)request->descriptor.gemm->k,
            (unsigned int)request->descriptor.gemm->lda, (unsigned int)request->descriptor.gemm->ldb, (unsigned int)request->descriptor.gemm->ldc,
            request->descriptor.gemm->alpha, request->descriptor.gemm->beta, uid);
        }
      }
      else { /* this case is not an actual error */
        return result;
      }
    } break;
    case LIBXSMM_BUILD_KIND_SSOA: { /* sparse SOA kernel */
      assert(0 != request->descriptor.ssoa && 0 != request->descriptor.ssoa->gemm);
      assert(0 != request->descriptor.ssoa->row_ptr && 0 != request->descriptor.ssoa->column_idx && 0 != request->descriptor.ssoa->values);
      if (0 == (LIBXSMM_GEMM_FLAG_F32PREC & (request->descriptor.ssoa->gemm->flags))/*only double-precision*/) {
        generated_code.generated_code = malloc(131072); /* large enough temporary buffer for generated code */
        generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
        LIBXSMM_NO_OFFLOAD(void, libxsmm_generator_spgemm_csr_soa_kernel, &generated_code, request->descriptor.ssoa->gemm, target_arch,
          request->descriptor.ssoa->row_ptr, request->descriptor.ssoa->column_idx,
          (const double*)request->descriptor.ssoa->values);
# if !defined(LIBXSMM_VTUNE)
        if (0 > libxsmm_verbosity)
# endif
        {
          const int uid = libxsmm_gemm_prefetch2uid((libxsmm_gemm_prefetch_type)request->descriptor.ssoa->gemm->prefetch);
          /* adopt scheme which allows kernel names of LIBXSMM to appear in order (Intel VTune, etc.) */
          LIBXSMM_SNPRINTF(jit_name, sizeof(jit_name), "libxsmm_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_a%i_b%i_p%i.ssoa", target_arch/*code path name*/,
            0 == (LIBXSMM_GEMM_FLAG_F32PREC & request->descriptor.ssoa->gemm->flags) ? "f64" : "f32",
            0 == (LIBXSMM_GEMM_FLAG_TRANS_A & request->descriptor.ssoa->gemm->flags) ? 'n' : 't',
            0 == (LIBXSMM_GEMM_FLAG_TRANS_B & request->descriptor.ssoa->gemm->flags) ? 'n' : 't',
            (unsigned int)request->descriptor.ssoa->gemm->m,   (unsigned int)request->descriptor.ssoa->gemm->n,   (unsigned int)request->descriptor.ssoa->gemm->k,
            (unsigned int)request->descriptor.ssoa->gemm->lda, (unsigned int)request->descriptor.ssoa->gemm->ldb, (unsigned int)request->descriptor.ssoa->gemm->ldc,
            request->descriptor.ssoa->gemm->alpha, request->descriptor.ssoa->gemm->beta, uid);
        }
      }
      else { /* this case is not an actual error */
        return result;
      }
    } break;
    case LIBXSMM_BUILD_KIND_SREG: { /* sparse register kernel */
      assert(0 != request->descriptor.sreg && 0 != request->descriptor.ssoa->gemm);
      assert(0 != request->descriptor.sreg->row_ptr && 0 != request->descriptor.sreg->column_idx && 0 != request->descriptor.sreg->values);
#if 1
      if (0 == (LIBXSMM_GEMM_FLAG_F32PREC & (request->descriptor.sreg->gemm->flags))/*only double-precision*/) {
#endif
        generated_code.generated_code = malloc(131072); /* large enough temporary buffer for generated code */
        generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
        LIBXSMM_NO_OFFLOAD(void, libxsmm_generator_spgemm_csr_reg_kernel, &generated_code, request->descriptor.sreg->gemm, target_arch,
          request->descriptor.sreg->row_ptr, request->descriptor.sreg->column_idx,
          (const double*)request->descriptor.sreg->values);
# if !defined(LIBXSMM_VTUNE)
        if (0 > libxsmm_verbosity)
# endif
        {
          const int uid = libxsmm_gemm_prefetch2uid((libxsmm_gemm_prefetch_type)request->descriptor.ssoa->gemm->prefetch);
          /* adopt scheme which allows kernel names of LIBXSMM to appear in order (Intel VTune, etc.) */
          LIBXSMM_SNPRINTF(jit_name, sizeof(jit_name), "libxsmm_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_a%i_b%i_p%i.sreg", target_arch/*code path name*/,
            0 == (LIBXSMM_GEMM_FLAG_F32PREC & request->descriptor.sreg->gemm->flags) ? "f64" : "f32",
            0 == (LIBXSMM_GEMM_FLAG_TRANS_A & request->descriptor.sreg->gemm->flags) ? 'n' : 't',
            0 == (LIBXSMM_GEMM_FLAG_TRANS_B & request->descriptor.sreg->gemm->flags) ? 'n' : 't',
            (unsigned int)request->descriptor.sreg->gemm->m,   (unsigned int)request->descriptor.sreg->gemm->n,   (unsigned int)request->descriptor.sreg->gemm->k,
            (unsigned int)request->descriptor.sreg->gemm->lda, (unsigned int)request->descriptor.sreg->gemm->ldb, (unsigned int)request->descriptor.sreg->gemm->ldc,
            request->descriptor.sreg->gemm->alpha, request->descriptor.sreg->gemm->beta, uid);
        }
#if 1
      }
      else { /* this case is not an actual error */
        return result;
      }
#endif
    } break;
    case LIBXSMM_BUILD_KIND_CFWD: { /* forward convolution */
      assert(0 != request->descriptor.cfwd);
      if (0 < request->descriptor.cfwd->kw && 0 < request->descriptor.cfwd->kh &&
          0 != request->descriptor.cfwd->stride_w && 0 != request->descriptor.cfwd->stride_h)
      {
        generated_code.generated_code = malloc(131072); /* large enough temporary buffer for generated code */
        generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
        LIBXSMM_NO_OFFLOAD(void, libxsmm_generator_convolution_forward_kernel, &generated_code, request->descriptor.cfwd, target_arch);
# if !defined(LIBXSMM_VTUNE)
        if (0 > libxsmm_verbosity)
# endif
        {
          const char *const precision_in = internal_get_precision_string(request->descriptor.cfwd->datatype);
          const char *const precision_out = internal_get_precision_string(request->descriptor.cfwd->datatype_itm);
          /* adopt scheme which allows kernel names of LIBXSMM to appear in order (Intel VTune, etc.) */
          LIBXSMM_SNPRINTF(jit_name, sizeof(jit_name), "libxsmm_%s_fwd_%s_%s_%ux%u_%ux%uu_s%ii%io_vl%ui%uo_ri%ux%u_ro%ux%u_r%ux%u_p%i_f%i.conv",
            target_arch/*code path name*/, precision_in, precision_out,
            (unsigned int)request->descriptor.cfwd->kw/*kernel width*/, (unsigned int)request->descriptor.cfwd->kh/*kernel height*/,
            (unsigned int)request->descriptor.cfwd->unroll_kw/*width*/, (unsigned int)request->descriptor.cfwd->unroll_kh/*height*/,
            (int)request->descriptor.cfwd->stride_w/*input offset*/, (int)request->descriptor.cfwd->stride_h/*output offsets*/,
            (unsigned int)request->descriptor.cfwd->ifm_block/*VLEN*/, (unsigned int)request->descriptor.cfwd->ofm_block/*VLEN*/,
            (unsigned int)request->descriptor.cfwd->ifw_padded, (unsigned int)request->descriptor.cfwd->ifh_padded,
            (unsigned int)request->descriptor.cfwd->ofw_padded/*1D and 2D register block*/,
            (unsigned int)request->descriptor.cfwd->ofh_padded/*2D register block*/,
            (unsigned int)request->descriptor.cfwd->ofw_rb/*register block ofw*/,
            (unsigned int)request->descriptor.cfwd->ofh_rb/*register block ofh*/,
            (int)request->descriptor.cfwd->prefetch/*binary OR'd prefetch flags*/,
            (int)request->descriptor.cfwd->format/*binary OR'd format flags*/);
        }
      }
    } break;
    case LIBXSMM_BUILD_KIND_CBWD: { /* backward convolution */
      assert(0 != request->descriptor.cbwd);
      if (0 < request->descriptor.cbwd->kw && 0 < request->descriptor.cbwd->kh &&
          0 != request->descriptor.cbwd->stride_w && 0 != request->descriptor.cbwd->stride_h)
      {
        generated_code.generated_code = malloc(131072); /* large enough temporary buffer for generated code */
        generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
        LIBXSMM_NO_OFFLOAD(void, libxsmm_generator_convolution_backward_kernel, &generated_code, request->descriptor.cbwd, target_arch);
# if !defined(LIBXSMM_VTUNE)
        if (0 > libxsmm_verbosity)
# endif
        {
          const char *const precision_in = internal_get_precision_string(request->descriptor.cbwd->datatype);
          const char *const precision_out = internal_get_precision_string(request->descriptor.cbwd->datatype_itm);
          /* adopt scheme which allows kernel names of LIBXSMM to appear in order (Intel VTune, etc.) */
          LIBXSMM_SNPRINTF(jit_name, sizeof(jit_name), "libxsmm_%s_bwd_%s_%s_%ux%u_%ux%uu_s%ii%io_vl%ui%uo_ri%ux%u_ro%ux%u_r%ux%u_of%uu%u_v%u_pa%u_p%i_f%i.conv",
            target_arch/*code path name*/, precision_in, precision_out,
            (unsigned int)request->descriptor.cbwd->kw/*kernel width*/, (unsigned int)request->descriptor.cbwd->kh/*kernel height*/,
            (unsigned int)request->descriptor.cbwd->unroll_kw/*width*/, (unsigned int)request->descriptor.cbwd->unroll_kh/*height*/,
            (int)request->descriptor.cbwd->stride_w/*input offset*/, (int)request->descriptor.cbwd->stride_h/*output offsets*/,
            (unsigned int)request->descriptor.cbwd->ifm_block/*VLEN*/, (unsigned int)request->descriptor.cbwd->ofm_block/*VLEN*/,
            (unsigned int)request->descriptor.cbwd->ifw_padded, (unsigned int)request->descriptor.cbwd->ifh_padded,
            (unsigned int)request->descriptor.cbwd->ofw_padded/*1D and 2D register block*/,
            (unsigned int)request->descriptor.cbwd->ofh_padded/*2D register block*/,
            (unsigned int)request->descriptor.cbwd->ofw_rb/*register block ofw*/,
            (unsigned int)request->descriptor.cbwd->ofh_rb/*register block ofh*/,
            (unsigned int)request->descriptor.cbwd->ofw/*ofw*/, (unsigned int)request->descriptor.cbwd->ofw_unroll/*ofw_unroll*/,
            (unsigned int)request->descriptor.cbwd->peeled/*peeled version*/,
            (unsigned int)request->descriptor.cbwd->prefetch_output_ahead/*prefetch kj outputs for jumping from non-peel to peel version*/,
            (int)request->descriptor.cbwd->prefetch/*binary OR'd prefetch flags*/,
            (int)request->descriptor.cbwd->format/*binary OR'd format flags*/);
        }
      }
    } break;
    case LIBXSMM_BUILD_KIND_CUPD: { /* convolution update weights */
      assert(0 != request->descriptor.cupd);
      if (0 < request->descriptor.cupd->kw &&
          0 != request->descriptor.cupd->stride_w && 0 != request->descriptor.cupd->stride_h)
      {
        generated_code.generated_code = malloc(131072); /* large enough temporary buffer for generated code */
        generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
        LIBXSMM_NO_OFFLOAD(void, libxsmm_generator_convolution_weight_update_kernel, &generated_code, request->descriptor.cupd, target_arch);
# if !defined(LIBXSMM_VTUNE)
        if (0 > libxsmm_verbosity)
# endif
        {
          const char *const precision_in = internal_get_precision_string(request->descriptor.cupd->datatype);
          const char *const precision_out = internal_get_precision_string(request->descriptor.cupd->datatype_itm);
          /* adopt scheme which allows kernel names of LIBXSMM to appear in order (Intel VTune, etc.) */
          LIBXSMM_SNPRINTF(jit_name, sizeof(jit_name), "libxsmm_%s_upd_%s_%s_%ux%u_%uu_s%ii%io_vl%ui%uo_ri%ux%u_ro%ux%u_r%ux%u_of%uu%ux%uu%u_if%uu_t%u_p%i_f%i.conv",
            target_arch/*code path name*/, precision_in, precision_out,
            (unsigned int)request->descriptor.cupd->kw/*kernel width*/, (unsigned int)request->descriptor.cupd->kh/*kernel height*/,
            (unsigned int)request->descriptor.cupd->unroll_kw/*width*/,
            (int)request->descriptor.cupd->stride_w/*input offset*/, (int)request->descriptor.cupd->stride_h/*output offsets*/,
            (unsigned int)request->descriptor.cupd->ifm_block/*VLEN*/, (unsigned int)request->descriptor.cupd->ofm_block/*VLEN*/,
            (unsigned int)request->descriptor.cupd->ifw_padded, (unsigned int)request->descriptor.cupd->ifh_padded,
            (unsigned int)request->descriptor.cupd->ofw_padded/*1D and 2D register block*/,
            (unsigned int)request->descriptor.cupd->ofh_padded/*2D register block*/,
            (unsigned int)request->descriptor.cupd->ofw_rb/*register block ofw*/,
            (unsigned int)request->descriptor.cupd->ofh_rb/*register block ofh*/,
            (unsigned int)request->descriptor.cupd->ofw/*ofw*/, (unsigned int)request->descriptor.cupd->ofw_unroll/*ofw_unroll*/,
            (unsigned int)request->descriptor.cupd->ofh/*ofh*/, (unsigned int)request->descriptor.cupd->ofh_unroll/*ofh_unroll*/,
            (unsigned int)request->descriptor.cupd->ifm_unroll/*ifm unroll*/,
            (unsigned int)request->descriptor.cupd->transpose_ofw_ifm/*transpose_ofw_ifm*/,
            (int)request->descriptor.cupd->prefetch/*binary OR'd prefetch flags*/,
            (int)request->descriptor.cupd->format/*binary OR'd format flags*/);
        }
      }
    } break;
    case LIBXSMM_BUILD_KIND_CWFWD: { /* convolution winograd forward  */
      assert(0 != request->descriptor.cwino);
      if (0 < request->descriptor.cwino->itiles && 0 < request->descriptor.cwino->jtiles && 0 < request->descriptor.cwino->bimg &&
          0 < request->descriptor.cwino->ur_i && 0 < request->descriptor.cwino->ur_j && 0 < request->descriptor.cwino->ur_m)
      {
        generated_code.generated_code = malloc(131072); /* large enough temporary buffer for generated code */
        generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
        LIBXSMM_NO_OFFLOAD(void, libxsmm_generator_convolution_winograd_forward_kernel, &generated_code, request->descriptor.cwino, target_arch);
# if !defined(LIBXSMM_VTUNE)
        if (0 > libxsmm_verbosity)
# endif
        {
          const char *const precision_in = internal_get_precision_string(LIBXSMM_DNN_DATATYPE_F32);
          const char *const precision_out = internal_get_precision_string(LIBXSMM_DNN_DATATYPE_F32);
          /* adopt scheme which allows kernel names of LIBXSMM to appear in order (Intel VTune, etc.) */
          LIBXSMM_SNPRINTF(jit_name, sizeof(jit_name), "libxsmm_%s_wfwd_%s_%s_t%ux%u_mb%u_ut%ux%u_umb%u_v%u_p%i.convwino",
            target_arch/*code path name*/, precision_in, precision_out,
            (unsigned int)request->descriptor.cwino->itiles/*itiles*/, (unsigned int)request->descriptor.cwino->jtiles/*jtiles*/,
            (unsigned int)request->descriptor.cwino->bimg/*image block*/,
            (unsigned int)request->descriptor.cwino->ur_i/*unrolliing of itiles*/, (unsigned int)request->descriptor.cwino->ur_j/* unrolling jtiles*/,
            (unsigned int)request->descriptor.cwino->ur_m/* unrolling image block*/,
            (unsigned int)request->descriptor.cwino->vratio/*vratio*/,
            (int)request->descriptor.cwino->prefetch/*binary OR'd prefetch flags*/);
        }
      }
    } break;
    case LIBXSMM_BUILD_KIND_CWBWD: { /* convolution winograd forward  */
      assert(0 != request->descriptor.cwino);
      if (0 < request->descriptor.cwino->itiles && 0 < request->descriptor.cwino->jtiles && 0 < request->descriptor.cwino->bimg &&
          0 < request->descriptor.cwino->ur_i && 0 < request->descriptor.cwino->ur_j && 0 < request->descriptor.cwino->ur_m)
      {
        generated_code.generated_code = malloc(131072); /* large enough temporary buffer for generated code */
        generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
        LIBXSMM_NO_OFFLOAD(void, libxsmm_generator_convolution_winograd_forward_kernel, &generated_code, request->descriptor.cwino, target_arch);
# if !defined(LIBXSMM_VTUNE)
        if (0 > libxsmm_verbosity)
# endif
        {
          const char *const precision_in = internal_get_precision_string(LIBXSMM_DNN_DATATYPE_F32);
          const char *const precision_out = internal_get_precision_string(LIBXSMM_DNN_DATATYPE_F32);
          /* adopt scheme which allows kernel names of LIBXSMM to appear in order (Intel VTune, etc.) */
          LIBXSMM_SNPRINTF(jit_name, sizeof(jit_name), "libxsmm_%s_wbwd_%s_%s_t%ux%u_mb%u_ut%ux%u_umb%u_v%u_p%i.convwino",
            target_arch/*code path name*/, precision_in, precision_out,
            (unsigned int)request->descriptor.cwino->itiles/*itiles*/, (unsigned int)request->descriptor.cwino->jtiles/*jtiles*/,
            (unsigned int)request->descriptor.cwino->bimg/*image block*/,
            (unsigned int)request->descriptor.cwino->ur_i/*unrolliing of itiles*/, (unsigned int)request->descriptor.cwino->ur_j/* unrolling jtiles*/,
            (unsigned int)request->descriptor.cwino->ur_m/* unrolling image block*/,
            (unsigned int)request->descriptor.cwino->vratio/*vratio*/,
            (int)request->descriptor.cwino->prefetch/*binary OR'd prefetch flags*/);
        }
      }
    } break;
    case LIBXSMM_BUILD_KIND_CWUPD: { /* convolution winograd forward  */
      assert(0 != request->descriptor.cwino);
      if (0 < request->descriptor.cwino->itiles && 0 < request->descriptor.cwino->jtiles && 0 < request->descriptor.cwino->bimg &&
          0 < request->descriptor.cwino->ur_i && 0 < request->descriptor.cwino->ur_j && 0 < request->descriptor.cwino->ur_m)
      {
        generated_code.generated_code = malloc(131072); /* large enough temporary buffer for generated code */
        generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
        LIBXSMM_NO_OFFLOAD(void, libxsmm_generator_convolution_winograd_weight_update_kernel, &generated_code, request->descriptor.cwino, target_arch);
# if !defined(LIBXSMM_VTUNE)
        if (0 > libxsmm_verbosity)
# endif
        {
          const char *const precision_in = internal_get_precision_string(LIBXSMM_DNN_DATATYPE_F32);
          const char *const precision_out = internal_get_precision_string(LIBXSMM_DNN_DATATYPE_F32);
          /* adopt scheme which allows kernel names of LIBXSMM to appear in order (Intel VTune, etc.) */
          LIBXSMM_SNPRINTF(jit_name, sizeof(jit_name), "libxsmm_%s_wupd_%s_%s_t%ux%u_mb%u_ut%ux%u_umb%u_v%u_p%i.convwino",
            target_arch/*code path name*/, precision_in, precision_out,
            (unsigned int)request->descriptor.cwino->itiles/*itiles*/, (unsigned int)request->descriptor.cwino->jtiles/*jtiles*/,
            (unsigned int)request->descriptor.cwino->bimg/*image block*/,
            (unsigned int)request->descriptor.cwino->ur_i/*unrolliing of itiles*/, (unsigned int)request->descriptor.cwino->ur_j/* unrolling jtiles*/,
            (unsigned int)request->descriptor.cwino->ur_m/* unrolling image block*/,
            (unsigned int)request->descriptor.cwino->vratio/*vratio*/,
            (int)request->descriptor.cwino->prefetch/*binary OR'd prefetch flags*/);
        }
      }
    } break;
# if !defined(NDEBUG) /* library code is expected to be mute */
    default: { /* unknown kind */
      static int error_once = 0;
      if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
        fprintf(stderr, "LIBXSMM: invalid build request discovered!\n");
      }
      result = EXIT_FAILURE;
    }
# endif
  }

  /* handle an eventual error in the else-branch */
  if (0 == generated_code.last_error) {
    assert(0 < generated_code.code_size/*sanity check*/);
    /* attempt to create executable buffer */
    result = libxsmm_xmalloc(&code->pmm, generated_code.code_size, 0/*auto*/,
      /* flag must be a superset of what's populated by libxsmm_malloc_attrib */
      LIBXSMM_MALLOC_FLAG_RWX, &regindex, sizeof(regindex));
    if (EXIT_SUCCESS == result) { /* check for success */
      assert(0 != code->const_pmm && 0 == (LIBXSMM_CODE_STATIC & code->imm));
      assert(0 != generated_code.generated_code/*sanity check*/);
      /* copy temporary buffer into the prepared executable buffer */
      memcpy(code->pmm, generated_code.generated_code, generated_code.code_size);
      /* attribute/protect buffer and revoke unnecessary flags */
      result = libxsmm_malloc_attrib(&code->pmm, LIBXSMM_MALLOC_FLAG_X, jit_name);
    }
  }
  else {
    if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
      static int error_once = 0;
      if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
        LIBXSMM_NO_OFFLOAD(int, fprintf, stderr, "%s (error #%u)\n",
          LIBXSMM_NO_OFFLOAD(const char*, libxsmm_strerror, generated_code.last_error),
          generated_code.last_error);
      }
    }
    result = EXIT_FAILURE;
  }
  free(generated_code.generated_code); /* free temporary/initial code buffer */
#else /* unsupported platform */
  LIBXSMM_UNUSED(request); LIBXSMM_UNUSED(regindex); LIBXSMM_UNUSED(code);
  /* libxsmm_get_target_arch also serves as a runtime check whether JIT is available or not */
  if (LIBXSMM_X86_AVX <= libxsmm_target_archid) result = EXIT_FAILURE;
#endif
  return result;
}


/** This function only works for JIT-generated code! */
LIBXSMM_API const libxsmm_gemm_descriptor* internal_get_gemm_descriptor(const void* gemm_jit);
LIBXSMM_API_DEFINITION const libxsmm_gemm_descriptor* internal_get_gemm_descriptor(const void* gemm_jit)
{
  const libxsmm_gemm_descriptor* result = 0;
  void* extra = 0;
  if (EXIT_SUCCESS == libxsmm_malloc_info(gemm_jit, 0/*size*/, 0/*flags*/, &extra) && 0 != extra) {
    const unsigned int i = *((const unsigned int*)extra);
    result = &internal_registry_keys[i].descriptor;
  }
  return result;
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE libxsmm_xmmfunction internal_find_code(const libxsmm_gemm_descriptor* descriptor)
{
  libxsmm_code_pointer flux_entry = { 0 };
  unsigned int hash, i0, i = 0, mode = 0, diff = 1;
#if !defined(NDEBUG)
  const libxsmm_gemm_descriptor* refdesc = 0;
#endif
#if defined(LIBXSMM_CAPACITY_CACHE) && (0 < (LIBXSMM_CAPACITY_CACHE))
  static LIBXSMM_TLS struct {
    union { char padding[LIBXSMM_GEMM_DESCRIPTOR_SIMD_SIZE]; libxsmm_gemm_descriptor desc; } keys[LIBXSMM_CAPACITY_CACHE];
    libxsmm_code_pointer code[LIBXSMM_CAPACITY_CACHE];
    unsigned int hit, id;
  } cache;
  unsigned int cache_index;
  assert(0 != descriptor && LIBXSMM_GEMM_DESCRIPTOR_SIMD_SIZE >= LIBXSMM_GEMM_DESCRIPTOR_SIZE);
  /* search small cache starting with the last hit on record */
  cache_index = libxsmm_gemm_diffn(descriptor, &cache.keys->desc, cache.hit, LIBXSMM_CAPACITY_CACHE, LIBXSMM_GEMM_DESCRIPTOR_SIMD_SIZE);
  if ((LIBXSMM_CAPACITY_CACHE) > cache_index && cache.id == internal_teardown) { /* cache hit, and valid */
    flux_entry = cache.code[cache_index];
    cache.hit = cache_index;
#if !defined(NDEBUG)
    if (0 == (LIBXSMM_CODE_STATIC & flux_entry.imm)) { /* JIT only */
# if defined(LIBXSMM_HASH_COLLISION)
      flux_entry.imm &= ~LIBXSMM_HASH_COLLISION; /* clear collision flag */
# endif
      refdesc = internal_get_gemm_descriptor(flux_entry.const_pmm);
    }
#endif
  }
  else
#else
  assert(0 != descriptor);
#endif
  {
    assert(0 != internal_registry);
    /* check if the requested xGEMM is already JITted */
    LIBXSMM_HASH_FUNCTION_CALL(hash, i = i0, *descriptor);
    while (0 != diff) {
      flux_entry.pmm = LIBXSMM_ATOMIC_LOAD(&internal_registry[i].pmm, LIBXSMM_ATOMIC_RELAXED); /* read registered code */
      if ((0 != flux_entry.const_pmm || 1 == mode) && 2 > mode) { /* check existing entry further */
        diff = 0 != flux_entry.const_pmm ? libxsmm_gemm_diff(descriptor, &internal_registry_keys[i].descriptor) : 1;
        if (0 != diff) { /* search for code version */
          if (0 == mode) { /* transition to higher mode */
            i0 = i; /* keep current position on record */
#if defined(LIBXSMM_HASH_COLLISION)
            /* enter code generation, and collision fix-up */
            if (0 == (LIBXSMM_HASH_COLLISION & flux_entry.imm)) {
              assert(0 != flux_entry.const_pmm); /* collision */
              mode = 3;
            }
            else
#endif      /* search for an existing code version */
            {
              mode = 1;
            }
          }
          i = LIBXSMM_HASH_MOD(i + 1, LIBXSMM_CAPACITY_REGISTRY);
          if (i == i0) { /* search finished, no code version exists */
#if defined(LIBXSMM_HASH_COLLISION)
            mode = 3; /* enter code generation, and collision fix-up */
#else
            mode = 2; /* enter code generation */
#endif
          }
          assert(0 != diff); /* continue */
        }
      }
      else { /* enter code generation (there is no code version yet) */
        assert(0 == mode || 1 < mode);
#if (0 != LIBXSMM_JIT)
        if (LIBXSMM_X86_AVX <= libxsmm_target_archid) { /* check if JIT is supported (CPUID) */
          assert(0 != mode || 0 == flux_entry.const_pmm/*code version does not exist*/);
          INTERNAL_FIND_CODE_LOCK(lock, i, diff, flux_entry.pmm); /* lock the registry entry */
          if (0 == internal_registry[i].const_pmm) { /* double-check registry after acquiring the lock */
            libxsmm_build_request request; /* setup the code build request */
            request.descriptor.gemm = descriptor; request.kind = LIBXSMM_BUILD_KIND_GEMM;
            internal_update_mmstatistic(descriptor, 1/*try*/, 0); /* count attempt */
            if (EXIT_SUCCESS == libxsmm_build(&request, i, &flux_entry) && 0 != flux_entry.const_pmm) {
              internal_registry_keys[i].descriptor = *descriptor;
              LIBXSMM_ATOMIC_STORE(&internal_registry[i].pmm, flux_entry.pmm, LIBXSMM_ATOMIC_RELAXED); /* sync */
# if defined(LIBXSMM_HASH_COLLISION)
              if (2 < mode) { /* arrived from collision state; now mark as collision */
                libxsmm_code_pointer fix_entry;
                fix_entry.pmm = LIBXSMM_ATOMIC_LOAD(&internal_registry[i0].pmm, LIBXSMM_ATOMIC_RELAXED);
                assert(0 != fix_entry.const_pmm);
                if (0 == (LIBXSMM_HASH_COLLISION & fix_entry.imm)) {
                  fix_entry.imm |= LIBXSMM_HASH_COLLISION; /* mark current entry as collision */
                  LIBXSMM_ATOMIC_STORE(&internal_registry[i0].pmm, fix_entry.pmm, LIBXSMM_ATOMIC_RELAXED);
                }
              }
# endif
            }
            diff = 0; /* inside of locked region (do not use break!) */
          }
          INTERNAL_FIND_CODE_UNLOCK(lock);
          if (0 != diff) { /* acquire registry slot */
            if (0 == mode) { /* initial condition */
              mode = 2; /* continue to linearly search for an empty slot */
              i0 = i; /* keep current position on record */
            }
            for (i = LIBXSMM_HASH_MOD(i + 1, LIBXSMM_CAPACITY_REGISTRY); i != i0 && 0 != internal_registry[i].const_pmm;
                 i = LIBXSMM_HASH_MOD(i + 1, LIBXSMM_CAPACITY_REGISTRY)); /* continue to linearly search code */
            if (i == i0) { /* out of capacity (no registry slot available) */
              diff = 0; /* inside of locked region (do not use break!) */
            }
            flux_entry.pmm = 0; /* no result */
          }
        }
        else
#endif
        { /* leave the dispatch loop */
          flux_entry.pmm = 0;
          diff = 0;
        }
      }
    }
#if defined(LIBXSMM_CAPACITY_CACHE) && (0 < (LIBXSMM_CAPACITY_CACHE))
    if (0 != flux_entry.const_pmm) { /* keep code version on record (cache) */
      INTERNAL_FIND_CODE_CACHE_INDEX(cache.hit, cache_index);
      cache.keys[cache_index].desc = *descriptor;
      cache.code[cache_index] = flux_entry;
      cache.hit = cache_index;
      assert(0 == diff);
    }
    if (cache.id != internal_teardown) {
      memset(cache.keys, 0, sizeof(cache.keys));
      cache.id = internal_teardown;
    }
#endif
#if !defined(NDEBUG)
    refdesc = &internal_registry_keys[i].descriptor;
#endif
  }
  assert(0 == flux_entry.const_pmm || 0 == refdesc || 0 == memcmp(refdesc, descriptor, LIBXSMM_GEMM_DESCRIPTOR_SIZE));
#if defined(LIBXSMM_HASH_COLLISION)
  flux_entry.imm &= ~(LIBXSMM_CODE_STATIC | LIBXSMM_HASH_COLLISION); /* clear non-JIT and collision flag */
#else
  flux_entry.imm &= ~LIBXSMM_CODE_STATIC; /* clear non-JIT flag */
#endif
  return flux_entry.xmm;
}


LIBXSMM_API_DEFINITION int libxsmm_get_registry_info(libxsmm_registry_info* info)
{
  int result = EXIT_SUCCESS;
  if (0 != info) {
    LIBXSMM_INIT
    if (0 != internal_registry) {
      size_t i;
      memset(info, 0, sizeof(libxsmm_registry_info)); /* info->nstatic = 0; info->size = 0; */
      info->nbytes = (LIBXSMM_CAPACITY_REGISTRY) * (sizeof(libxsmm_code_pointer) + sizeof(internal_regkey_type));
      info->capacity = LIBXSMM_CAPACITY_REGISTRY;
      info->ncache = LIBXSMM_CAPACITY_CACHE;
      for (i = 0; i < (LIBXSMM_CAPACITY_REGISTRY); ++i) {
        libxsmm_code_pointer code = internal_registry[i];
        if (0 != code.const_pmm && EXIT_SUCCESS == result) {
          if (0 == (LIBXSMM_CODE_STATIC & code.imm)) { /* check for allocated/generated JIT-code */
            size_t buffer_size = 0;
            void* buffer = 0;
#if defined(LIBXSMM_HASH_COLLISION)
            code.imm &= ~LIBXSMM_HASH_COLLISION; /* clear collision flag */
#endif
            result = libxsmm_malloc_info(code.const_pmm, &buffer_size, 0/*flags*/, &buffer);
            if (EXIT_SUCCESS == result) {
              info->nbytes += (unsigned int)(buffer_size + (((char*)code.const_pmm) - (char*)buffer));
            }
          }
          else {
            ++info->nstatic;
          }
          ++info->size;
        }
      }
    }
    else {
      result = EXIT_FAILURE;
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API_DEFINITION libxsmm_gemm_descriptor* libxsmm_create_dgemm_descriptor(char transa, char transb,
  int m, int n, int k, int lda, int ldb, int ldc, double alpha, double beta,
  libxsmm_gemm_prefetch_type strategy)
{
  libxsmm_gemm_descriptor *const result = (libxsmm_gemm_descriptor*)malloc(sizeof(libxsmm_gemm_descriptor));
  assert(0 != transa && 0 != transb && 0 != strchr("NnTt", transa) && 0 != strchr("NnTt", transb));
  /* filter alpha and beta values since the descriptor cannot store general real values */
  if (0 != result && 0 != LIBXSMM_GEMM_NO_BYPASS(0, alpha, beta)) {
    LIBXSMM_GEMM_DESCRIPTOR(*result, 1, LIBXSMM_GEMM_FLAG_F64PREC |
      (('T' == transa || 't' == transa) ? LIBXSMM_GEMM_FLAG_TRANS_A : 0) |
      (('T' == transb || 't' == transb) ? LIBXSMM_GEMM_FLAG_TRANS_B : 0),
      m, n, k, lda, ldb, ldc, alpha, beta, strategy);
  }
  return result;
}


LIBXSMM_API_DEFINITION void libxsmm_release_gemm_descriptor(const libxsmm_gemm_descriptor* descriptor)
{
  free((void*)descriptor);
}


LIBXSMM_API_DEFINITION libxsmm_xmmfunction libxsmm_xmmdispatch(const libxsmm_gemm_descriptor* descriptor)
{
  libxsmm_xmmfunction result = { 0 };
  /* there is no need to check LIBXSMM_GEMM_NO_BYPASS_DIMS (M, N, K, LDx) since we already got a descriptor */
  if (0 != descriptor && LIBXSMM_GEMM_NO_BYPASS(descriptor->flags, descriptor->alpha, descriptor->beta)) {
    libxsmm_gemm_descriptor backend_descriptor;
    LIBXSMM_INIT
    if (0 > (int)descriptor->prefetch) {
      backend_descriptor = *descriptor;
      backend_descriptor.prefetch = (unsigned char)libxsmm_gemm_auto_prefetch;
      descriptor = &backend_descriptor;
    }
    result = internal_find_code(descriptor);
  }
  else { /* bypass (not supported) */
    internal_update_mmstatistic(descriptor, 1/*try*/, 0);
  }
  return result;
}

#if !defined(LIBXSMM_BUILD) && defined(__APPLE__) && defined(__MACH__)
LIBXSMM_PRAGMA_OPTIMIZE_OFF
#endif

LIBXSMM_API_DEFINITION libxsmm_smmfunction libxsmm_smmdispatch(int m, int n, int k,
  const int* lda, const int* ldb, const int* ldc,
  const float* alpha, const float* beta,
  const int* flags, const int* prefetch)
{
  LIBXSMM_INIT
  INTERNAL_DISPATCH(float, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch);
}


LIBXSMM_API_DEFINITION libxsmm_dmmfunction libxsmm_dmmdispatch(int m, int n, int k,
  const int* lda, const int* ldb, const int* ldc,
  const double* alpha, const double* beta,
  const int* flags, const int* prefetch)
{
  LIBXSMM_INIT
  INTERNAL_DISPATCH(double, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch);
}

#if !defined(LIBXSMM_BUILD) && defined(__APPLE__) && defined(__MACH__)
LIBXSMM_PRAGMA_OPTIMIZE_ON
#endif

LIBXSMM_API_DEFINITION libxsmm_xmmfunction libxsmm_create_dcsr_soa(const libxsmm_gemm_descriptor* descriptor,
  const unsigned int* row_ptr, const unsigned int* column_idx, const double* values)
{
  libxsmm_code_pointer code = { 0 };
  libxsmm_csr_soa_descriptor ssoa;
  libxsmm_build_request request;
  LIBXSMM_INIT
  ssoa.gemm = descriptor;
  ssoa.row_ptr = row_ptr;
  ssoa.column_idx = column_idx;
  ssoa.values = values;
  request.descriptor.ssoa = &ssoa;
  request.kind = LIBXSMM_BUILD_KIND_SSOA;
  libxsmm_build(&request, LIBXSMM_CAPACITY_REGISTRY/*not managed*/, &code);
  return code.xmm;
}


LIBXSMM_API_DEFINITION libxsmm_dmmfunction libxsmm_create_dcsr_reg(const libxsmm_gemm_descriptor* descriptor,
  const unsigned int* row_ptr, const unsigned int* column_idx, const double* values)
{
  libxsmm_code_pointer code = { 0 };
  libxsmm_csr_reg_descriptor sreg;
  libxsmm_build_request request;
  LIBXSMM_INIT
  sreg.gemm = descriptor;
  sreg.row_ptr = row_ptr;
  sreg.column_idx = column_idx;
  sreg.values = values;
  request.descriptor.sreg = &sreg;
  request.kind = LIBXSMM_BUILD_KIND_SREG;
  libxsmm_build(&request, LIBXSMM_CAPACITY_REGISTRY/*not managed*/, &code);
  return code.xmm.dmm;
}


LIBXSMM_API_DEFINITION libxsmm_smmfunction libxsmm_create_scsr_reg(const libxsmm_gemm_descriptor* descriptor,
  const unsigned int* row_ptr, const unsigned int* column_idx, const float* values)
{
  libxsmm_code_pointer code = { 0 };
  libxsmm_csr_reg_descriptor sreg;
  libxsmm_build_request request;
  double* d_values;
  unsigned int i;
  LIBXSMM_INIT
  /* we need to copy the values into a double precision buffer */
  d_values = (double*)malloc(row_ptr[descriptor->m]*sizeof(double));
  for ( i = 0; i < row_ptr[descriptor->m]; i++) {
    d_values[i] = (double)values[i];
  }
  sreg.gemm = descriptor;
  sreg.row_ptr = row_ptr;
  sreg.column_idx = column_idx;
  sreg.values = d_values;
  request.descriptor.sreg = &sreg;
  request.kind = LIBXSMM_BUILD_KIND_SREG;
  libxsmm_build(&request, LIBXSMM_CAPACITY_REGISTRY/*not managed*/, &code);
  free(d_values);
  return code.xmm.smm;
}


LIBXSMM_API_DEFINITION void libxsmm_release_kernel(const void* jit_code)
{
  void* extra = 0;
  LIBXSMM_INIT
  if (EXIT_SUCCESS == libxsmm_malloc_info(jit_code, 0/*size*/, 0/*flags*/, &extra) && 0 != extra) {
    const unsigned int regindex = *((const unsigned int*)extra);
    if ((LIBXSMM_CAPACITY_REGISTRY) <= regindex) {
      libxsmm_xfree(jit_code);
    }
    /* TODO: implement to unregister GEMM kernels */
  }
  else if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
    static int error_once = 0;
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM: failed to release kernel!\n");
    }
  }
}

