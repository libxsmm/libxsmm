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
#ifndef LIBXSMM_SYNC_H
#define LIBXSMM_SYNC_H

#include "libxsmm_intrinsics_x86.h"

#if !defined(LIBXSMM_TLS)
# if (0 != LIBXSMM_SYNC) && !defined(LIBXSMM_NO_TLS)
#   if defined(__CYGWIN__) && defined(__clang__)
#     define LIBXSMM_NO_TLS
#     define LIBXSMM_TLS
#   else
#     if (defined(_WIN32) && !defined(__GNUC__) && !defined(__clang__)) || (defined(__PGI) && !defined(__PGLLVM__))
#       define LIBXSMM_TLS LIBXSMM_ATTRIBUTE(thread)
#     elif defined(__GNUC__) || defined(__clang__) || defined(__PGLLVM__) || defined(_CRAYC)
#       define LIBXSMM_TLS __thread
#     elif defined(__cplusplus)
#       define LIBXSMM_TLS thread_local
#     else
#       error Missing TLS support!
#     endif
#   endif
# else
#   if !defined(LIBXSMM_NO_TLS)
#     define LIBXSMM_NO_TLS
#   endif
#   define LIBXSMM_TLS
# endif
#endif

#if !defined(LIBXSMM_GCC_BASELINE) && !defined(LIBXSMM_SYNC_LEGACY) && ((defined(_WIN32) && defined(__clang__)) || \
    (defined(__GNUC__) && LIBXSMM_VERSION2(4, 7) <= LIBXSMM_VERSION2(__GNUC__, __GNUC_MINOR__)))
# define LIBXSMM_GCC_BASELINE
#endif

#if defined(__MIC__)
# define LIBXSMM_SYNC_PAUSE _mm_delay_32(8/*delay*/)
#elif !defined(LIBXSMM_INTRINSICS_NONE)
# if defined(LIBXSMM_GCC_BASELINE) && !defined(__INTEL_COMPILER)
#   define LIBXSMM_SYNC_PAUSE __builtin_ia32_pause()
# else
#   define LIBXSMM_SYNC_PAUSE _mm_pause()
# endif
#elif (LIBXSMM_X86_GENERIC <= LIBXSMM_STATIC_TARGET_ARCH) && defined(__GNUC__)
# define LIBXSMM_SYNC_PAUSE __asm__ __volatile__("pause" ::: "memory")
#else
# define LIBXSMM_SYNC_PAUSE
#endif

/* permit thread-unsafe */
#if !defined(LIBXSMM_SYNC_NONE) && ( \
  (defined(__PGI) && !defined(LIBXSMM_LIBATOMIC)) || \
  (defined(_CRAYC) && !defined(__GNUC__)))
# define LIBXSMM_SYNC_NONE
#endif

#if !defined(LIBXSMM_ATOMIC_TRYLOCK_CMPSWP) && 0
# define LIBXSMM_ATOMIC_TRYLOCK_CMPSWP
#endif
#if !defined(LIBXSMM_ATOMIC_ZERO_STORE) && defined(_CRAYC)
# define LIBXSMM_ATOMIC_ZERO_STORE
#endif
#if !defined(LIBXSMM_ATOMIC_LOCKTYPE)
# if defined(_WIN32) || 1/*alignment*/
#   define LIBXSMM_ATOMIC_LOCKTYPE int
# else
#   define LIBXSMM_ATOMIC_LOCKTYPE char
# endif
#endif

typedef enum libxsmm_atomic_kind {
#if defined(__ATOMIC_SEQ_CST)
  LIBXSMM_ATOMIC_SEQ_CST = __ATOMIC_SEQ_CST,
#else
  LIBXSMM_ATOMIC_SEQ_CST = 0,
#endif
#if defined(__ATOMIC_RELAXED)
  LIBXSMM_ATOMIC_RELAXED = __ATOMIC_RELAXED
#else
  LIBXSMM_ATOMIC_RELAXED = LIBXSMM_ATOMIC_SEQ_CST
#endif
} libxsmm_atomic_kind;

#define LIBXSMM_NONATOMIC_LOCKTYPE LIBXSMM_ATOMIC_LOCKTYPE
#define LIBXSMM_NONATOMIC_LOAD(SRC_PTR, KIND) (*(SRC_PTR))
#define LIBXSMM_NONATOMIC_STORE(DST_PTR, VALUE, KIND) { LIBXSMM_UNUSED(KIND); *(DST_PTR) = (VALUE); }
#define LIBXSMM_NONATOMIC_STORE_ZERO(DST_PTR, KIND) LIBXSMM_NONATOMIC_STORE(DST_PTR, 0, KIND)
#define LIBXSMM_NONATOMIC_FETCH_OR(DST_PTR, VALUE/*side-effect*/, KIND) (/* 1st step: swap(dst, val) */ \
  ((*DST_PTR) = (*DST_PTR) ^ (VALUE)), (VALUE = (VALUE) ^ (*DST_PTR)), ((*DST_PTR) = (*DST_PTR) ^ (VALUE)), \
  (*(DST_PTR) |= VALUE), (VALUE) /* 2nd step: or, and 3rd/last step: original dst-value */)
#define LIBXSMM_NONATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND) (*(DST_PTR) += VALUE)
#define LIBXSMM_NONATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND) (*(DST_PTR) -= VALUE)
#define LIBXSMM_NONATOMIC_FETCH_ADD(DST_PTR, VALUE, KIND) (LIBXSMM_NONATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND), (*(DST_PTR) - (VALUE)))
#define LIBXSMM_NONATOMIC_FETCH_SUB(DST_PTR, VALUE, KIND) (LIBXSMM_NONATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND), (*(DST_PTR) + (VALUE)))
#define LIBXSMM_NONATOMIC_CMPSWP(DST_PTR, OLDVAL, NEWVAL, KIND) ((NEWVAL) == (*(DST_PTR) == (OLDVAL) ? (*(DST_PTR) = (NEWVAL)) : (OLDVAL)))
#define LIBXSMM_NONATOMIC_TRYLOCK(DST_PTR, KIND) LIBXSMM_NONATOMIC_CMPSWP(DST_PTR, 0, 1, KIND)
#define LIBXSMM_NONATOMIC_ACQUIRE(DST_PTR, NPAUSE, KIND) { LIBXSMM_UNUSED(NPAUSE); \
          LIBXSMM_ASSERT_MSG(0 == *(DST_PTR), "LIBXSMM_NONATOMIC_ACQUIRE"); LIBXSMM_NONATOMIC_STORE(DST_PTR, 1, KIND); \
          LIBXSMM_ASSERT_MSG(0 != *(DST_PTR), "LIBXSMM_NONATOMIC_ACQUIRE"); }
#define LIBXSMM_NONATOMIC_RELEASE(DST_PTR, KIND) { LIBXSMM_UNUSED(DST_PTR); LIBXSMM_UNUSED(KIND); \
          LIBXSMM_ASSERT_MSG(0 != *(DST_PTR), "LIBXSMM_NONATOMIC_RELEASE"); LIBXSMM_NONATOMIC_STORE(DST_PTR, 0, KIND); \
          LIBXSMM_ASSERT_MSG(0 == *(DST_PTR), "LIBXSMM_NONATOMIC_RELEASE"); }
#define LIBXSMM_NONATOMIC_SYNC(KIND) LIBXSMM_UNUSED(KIND)

#if (0 == LIBXSMM_SYNC) || defined(LIBXSMM_SYNC_NONE)
# define LIBXSMM_ATOMIC(FN, BITS) FN
# define LIBXSMM_ATOMIC_LOAD LIBXSMM_NONATOMIC_LOAD
# define LIBXSMM_ATOMIC_STORE LIBXSMM_NONATOMIC_STORE
# define LIBXSMM_ATOMIC_STORE_ZERO LIBXSMM_NONATOMIC_STORE_ZERO
# define LIBXSMM_ATOMIC_FETCH_OR LIBXSMM_NONATOMIC_FETCH_OR
# define LIBXSMM_ATOMIC_ADD_FETCH LIBXSMM_NONATOMIC_ADD_FETCH
# define LIBXSMM_ATOMIC_SUB_FETCH LIBXSMM_NONATOMIC_SUB_FETCH
# define LIBXSMM_ATOMIC_FETCH_ADD LIBXSMM_NONATOMIC_FETCH_ADD
# define LIBXSMM_ATOMIC_FETCH_SUB LIBXSMM_NONATOMIC_FETCH_SUB
# define LIBXSMM_ATOMIC_CMPSWP LIBXSMM_NONATOMIC_CMPSWP
# define LIBXSMM_ATOMIC_TRYLOCK LIBXSMM_NONATOMIC_TRYLOCK
# define LIBXSMM_ATOMIC_ACQUIRE LIBXSMM_NONATOMIC_ACQUIRE
# define LIBXSMM_ATOMIC_RELEASE LIBXSMM_NONATOMIC_RELEASE
# define LIBXSMM_ATOMIC_SYNC LIBXSMM_NONATOMIC_SYNC
# if !defined(LIBXSMM_SYNC_NPAUSE)
#   define LIBXSMM_SYNC_NPAUSE 0
# endif
#elif (defined(LIBXSMM_GCC_BASELINE) || defined(LIBXSMM_LIBATOMIC) /* GNU's libatomic required */ || \
      (defined(__GNUC__) && LIBXSMM_VERSION2(4, 1) <= LIBXSMM_VERSION2(__GNUC__, __GNUC_MINOR__)))
# if defined(LIBXSMM_LIBATOMIC)
#   define LIBXSMM_ATOMIC(FN, BITS) LIBXSMM_CONCATENATE(LIBXSMM_ATOMIC, BITS)(FN)
#   define LIBXSMM_ATOMIC8(FN) LIBXSMM_CONCATENATE(FN, 8)
#   define LIBXSMM_ATOMIC16(FN) LIBXSMM_CONCATENATE(FN, 16)
#   define LIBXSMM_ATOMIC32(FN) FN/*default*/
#   define LIBXSMM_ATOMIC64(FN) LIBXSMM_CONCATENATE(FN, 64)
#   if defined(__PGI)
#     define LIBXSMM_ATOMIC_LOAD(SRC_PTR, KIND) LIBXSMM_NONATOMIC_LOAD(SRC_PTR, KIND)
#     define LIBXSMM_ATOMIC_LOAD8(SRC_PTR, KIND) LIBXSMM_NONATOMIC_LOAD(SRC_PTR, KIND)
#     define LIBXSMM_ATOMIC_LOAD16(SRC_PTR, KIND) LIBXSMM_NONATOMIC_LOAD(SRC_PTR, KIND)
#     define LIBXSMM_ATOMIC_LOAD64(SRC_PTR, KIND) LIBXSMM_NONATOMIC_LOAD(SRC_PTR, KIND)
#     define LIBXSMM_ATOMIC_STORE(DST_PTR, VALUE, KIND) LIBXSMM_NONATOMIC_STORE(DST_PTR, VALUE, KIND)
#     define LIBXSMM_ATOMIC_STORE8(DST_PTR, VALUE, KIND) LIBXSMM_NONATOMIC_STORE(DST_PTR, VALUE, KIND)
#     define LIBXSMM_ATOMIC_STORE16(DST_PTR, VALUE, KIND) LIBXSMM_NONATOMIC_STORE(DST_PTR, VALUE, KIND)
#     define LIBXSMM_ATOMIC_STORE64(DST_PTR, VALUE, KIND) LIBXSMM_NONATOMIC_STORE(DST_PTR, VALUE, KIND)
#   else
#     define LIBXSMM_ATOMIC_LOAD(SRC_PTR, KIND) __atomic_load_4(SRC_PTR, KIND)
#     define LIBXSMM_ATOMIC_LOAD8(SRC_PTR, KIND) __atomic_load_1(SRC_PTR, KIND)
#     define LIBXSMM_ATOMIC_LOAD16(SRC_PTR, KIND) __atomic_load_2(SRC_PTR, KIND)
#     define LIBXSMM_ATOMIC_LOAD64(SRC_PTR, KIND) __atomic_load_8(SRC_PTR, KIND)
#     define LIBXSMM_ATOMIC_STORE(DST_PTR, VALUE, KIND) __atomic_store_4(DST_PTR, (unsigned int)(VALUE), KIND)
#     define LIBXSMM_ATOMIC_STORE8(DST_PTR, VALUE, KIND) __atomic_store_1(DST_PTR, (unsigned char)(VALUE), KIND)
#     define LIBXSMM_ATOMIC_STORE16(DST_PTR, VALUE, KIND) __atomic_store_2(DST_PTR, (unsigned short)(VALUE), KIND)
#     define LIBXSMM_ATOMIC_STORE64(DST_PTR, VALUE, KIND) __atomic_store_8(DST_PTR, (unsigned long long)(VALUE), KIND)
#   endif
#   define LIBXSMM_ATOMIC_FETCH_OR(DST_PTR, VALUE, KIND) __atomic_fetch_or_4(DST_PTR, (unsigned int)(VALUE), KIND)
#   define LIBXSMM_ATOMIC_FETCH_OR8(DST_PTR, VALUE, KIND) __atomic_fetch_or_1(DST_PTR, (unsigned char)(VALUE), KIND)
#   define LIBXSMM_ATOMIC_FETCH_OR16(DST_PTR, VALUE, KIND) __atomic_fetch_or_2(DST_PTR, (unsigned short)(VALUE), KIND)
#   define LIBXSMM_ATOMIC_FETCH_OR64(DST_PTR, VALUE, KIND) __atomic_fetch_or_8(DST_PTR, (unsigned long long)(VALUE), KIND)
#   define LIBXSMM_ATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND) ((int)__atomic_add_fetch_4(DST_PTR, (int)(VALUE), KIND))
#   define LIBXSMM_ATOMIC_ADD_FETCH8(DST_PTR, VALUE, KIND) ((signed char)__atomic_add_fetch_1(DST_PTR, (signed char)(VALUE), KIND))
#   define LIBXSMM_ATOMIC_ADD_FETCH16(DST_PTR, VALUE, KIND) ((short)__atomic_add_fetch_2(DST_PTR, (short)(VALUE), KIND))
#   define LIBXSMM_ATOMIC_ADD_FETCH64(DST_PTR, VALUE, KIND) ((long long)__atomic_add_fetch_8(DST_PTR, (long long)(VALUE), KIND))
#   define LIBXSMM_ATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND) ((int)__atomic_sub_fetch_4(DST_PTR, (int)(VALUE), KIND))
#   define LIBXSMM_ATOMIC_SUB_FETCH8(DST_PTR, VALUE, KIND) ((signed char)__atomic_sub_fetch_1(DST_PTR, (signed char)(VALUE), KIND))
#   define LIBXSMM_ATOMIC_SUB_FETCH16(DST_PTR, VALUE, KIND) ((short)__atomic_sub_fetch_2(DST_PTR, (short)(VALUE), KIND))
#   define LIBXSMM_ATOMIC_SUB_FETCH64(DST_PTR, VALUE, KIND) ((long long)__atomic_sub_fetch_8(DST_PTR, (long long)(VALUE), KIND))
#   define LIBXSMM_ATOMIC_FETCH_ADD(DST_PTR, VALUE, KIND) ((int)__atomic_fetch_add_4(DST_PTR, (int)(VALUE), KIND))
#   define LIBXSMM_ATOMIC_FETCH_ADD8(DST_PTR, VALUE, KIND) ((signed char)__atomic_fetch_add_1(DST_PTR, (signed char)(VALUE), KIND))
#   define LIBXSMM_ATOMIC_FETCH_ADD16(DST_PTR, VALUE, KIND) ((short)__atomic_fetch_add_2(DST_PTR, (short)(VALUE), KIND))
#   define LIBXSMM_ATOMIC_FETCH_ADD64(DST_PTR, VALUE, KIND) ((long long)__atomic_fetch_add_8(DST_PTR, (long long)(VALUE), KIND))
#   define LIBXSMM_ATOMIC_FETCH_SUB(DST_PTR, VALUE, KIND) ((int)__atomic_fetch_sub_4(DST_PTR, (int)(VALUE), KIND))
#   define LIBXSMM_ATOMIC_FETCH_SUB8(DST_PTR, VALUE, KIND) ((signed char)__atomic_fetch_sub_1(DST_PTR, (signed char)(VALUE), KIND))
#   define LIBXSMM_ATOMIC_FETCH_SUB16(DST_PTR, VALUE, KIND) ((short)__atomic_fetch_sub_2(DST_PTR, (short)(VALUE), KIND))
#   define LIBXSMM_ATOMIC_FETCH_SUB64(DST_PTR, VALUE, KIND) ((long long)__atomic_fetch_sub_8(DST_PTR, (long long)(VALUE), KIND))
#   define LIBXSMM_ATOMIC_CMPSWP(DST_PTR, OLDVAL, NEWVAL, KIND) \
            __atomic_compare_exchange_4(DST_PTR, &(OLDVAL), (NEWVAL), 0/*false*/, KIND, LIBXSMM_ATOMIC_RELAXED)
#   define LIBXSMM_ATOMIC_CMPSWP8(DST_PTR, OLDVAL, NEWVAL, KIND) \
            __atomic_compare_exchange_1(DST_PTR, &(OLDVAL), (NEWVAL), 0/*false*/, KIND, LIBXSMM_ATOMIC_RELAXED)
#   define LIBXSMM_ATOMIC_CMPSWP16(DST_PTR, OLDVAL, NEWVAL, KIND) \
            __atomic_compare_exchange_2(DST_PTR, &(OLDVAL), (NEWVAL), 0/*false*/, KIND, LIBXSMM_ATOMIC_RELAXED)
#   define LIBXSMM_ATOMIC_CMPSWP64(DST_PTR, OLDVAL, NEWVAL, KIND) \
            __atomic_compare_exchange_8(DST_PTR, &(OLDVAL), (NEWVAL), 0/*false*/, KIND, LIBXSMM_ATOMIC_RELAXED)
#   if defined(LIBXSMM_ATOMIC_TRYLOCK_CMPSWP)
#     define LIBXSMM_ATOMIC_TRYLOCK(DST_PTR, KIND) (!__atomic_test_and_set(DST_PTR, KIND))
#   endif
#   if defined(__PGI)
#     define LIBXSMM_ATOMIC_RELEASE(DST_PTR, KIND) { LIBXSMM_ASSERT_MSG(0 != *(DST_PTR), "LIBXSMM_ATOMIC_RELEASE"); \
              LIBXSMM_ATOMIC_STORE_ZERO8(DST_PTR, KIND); } /* matches bit-width of LIBXSMM_ATOMIC_LOCKTYPE */
#   else
#     define LIBXSMM_ATOMIC_RELEASE(DST_PTR, KIND) { LIBXSMM_ASSERT_MSG(0 != *(DST_PTR), "LIBXSMM_ATOMIC_RELEASE"); \
              __atomic_clear(DST_PTR, KIND); }
#   endif
#   define LIBXSMM_ATOMIC_SYNC(KIND) __sync_synchronize()
#   if !defined(LIBXSMM_ATOMIC_ZERO_STORE)
#     define LIBXSMM_ATOMIC_ZERO_STORE
#   endif
# elif defined(LIBXSMM_GCC_BASELINE)
#   define LIBXSMM_ATOMIC(FN, BITS) FN
#   define LIBXSMM_ATOMIC_LOAD(SRC_PTR, KIND) __atomic_load_n(SRC_PTR, KIND)
#   define LIBXSMM_ATOMIC_STORE(DST_PTR, VALUE, KIND) __atomic_store_n(DST_PTR, VALUE, KIND)
#   if !defined(LIBXSMM_ATOMIC_ZERO_STORE)
#     define LIBXSMM_ATOMIC_STORE_ZERO(DST_PTR, KIND) do {} while (__atomic_and_fetch(DST_PTR, 0, KIND))
#   endif
#   define LIBXSMM_ATOMIC_FETCH_OR(DST_PTR, VALUE, KIND) __atomic_fetch_or(DST_PTR, VALUE, KIND)
#   define LIBXSMM_ATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND) __atomic_add_fetch(DST_PTR, VALUE, KIND)
#   define LIBXSMM_ATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND) __atomic_sub_fetch(DST_PTR, VALUE, KIND)
#   define LIBXSMM_ATOMIC_FETCH_ADD(DST_PTR, VALUE, KIND) __atomic_fetch_add(DST_PTR, VALUE, KIND)
#   define LIBXSMM_ATOMIC_FETCH_SUB(DST_PTR, VALUE, KIND) __atomic_fetch_sub(DST_PTR, VALUE, KIND)
#   define LIBXSMM_ATOMIC_CMPSWP(DST_PTR, OLDVAL, NEWVAL, KIND) __sync_bool_compare_and_swap(DST_PTR, OLDVAL, NEWVAL)
#   if defined(LIBXSMM_ATOMIC_TRYLOCK_CMPSWP)
#     define LIBXSMM_ATOMIC_TRYLOCK(DST_PTR, KIND) (!__atomic_test_and_set(DST_PTR, KIND))
#   endif
#   define LIBXSMM_ATOMIC_RELEASE(DST_PTR, KIND) { LIBXSMM_ASSERT_MSG(0 != *(DST_PTR), "LIBXSMM_ATOMIC_RELEASE"); \
            __atomic_clear(DST_PTR, KIND); }
#   if 0 /* __atomic_thread_fence: incorrect behavior in libxsmm_barrier (even with LIBXSMM_ATOMIC_SEQ_CST) */
#     define LIBXSMM_ATOMIC_SYNC(KIND) __atomic_thread_fence(KIND)
#   else
#     define LIBXSMM_ATOMIC_SYNC(KIND) __sync_synchronize()
#   endif
# else /* GCC legacy atomics */
#   define LIBXSMM_ATOMIC(FN, BITS) FN
#   define LIBXSMM_ATOMIC_LOAD(SRC_PTR, KIND) __sync_or_and_fetch(SRC_PTR, 0)
#   if (LIBXSMM_X86_GENERIC <= LIBXSMM_STATIC_TARGET_ARCH)
#     define LIBXSMM_ATOMIC_STORE(DST_PTR, VALUE, KIND) do { \
              __asm__ __volatile__("" ::: "memory"); *(DST_PTR) = (VALUE); \
              __asm__ __volatile__("" ::: "memory"); } while(0)
#   else
#     define LIBXSMM_ATOMIC_SYNC_NOFENCE(KIND)
#     define LIBXSMM_ATOMIC_STORE(DST_PTR, VALUE, KIND) *(DST_PTR) = (VALUE)
#   endif
#   if !defined(LIBXSMM_ATOMIC_ZERO_STORE)
#     define LIBXSMM_ATOMIC_STORE_ZERO(DST_PTR, KIND) do {} while (__sync_and_and_fetch(DST_PTR, 0))
#   endif
#   define LIBXSMM_ATOMIC_FETCH_OR(DST_PTR, VALUE, KIND) __sync_fetch_and_or(DST_PTR, VALUE)
#   define LIBXSMM_ATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND) __sync_add_and_fetch(DST_PTR, VALUE)
#   define LIBXSMM_ATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND) __sync_sub_and_fetch(DST_PTR, VALUE)
#   define LIBXSMM_ATOMIC_FETCH_ADD(DST_PTR, VALUE, KIND) __sync_fetch_and_add(DST_PTR, VALUE)
#   define LIBXSMM_ATOMIC_FETCH_SUB(DST_PTR, VALUE, KIND) __sync_fetch_and_sub(DST_PTR, VALUE)
#   define LIBXSMM_ATOMIC_CMPSWP(DST_PTR, OLDVAL, NEWVAL, KIND) __sync_bool_compare_and_swap(DST_PTR, OLDVAL, NEWVAL)
#   if defined(LIBXSMM_ATOMIC_TRYLOCK_CMPSWP)
#     define LIBXSMM_ATOMIC_TRYLOCK(DST_PTR, KIND) (0 == __sync_lock_test_and_set(DST_PTR, 1))
#   endif
#   define LIBXSMM_ATOMIC_RELEASE(DST_PTR, KIND) { LIBXSMM_ASSERT_MSG(0 != *(DST_PTR), "LIBXSMM_ATOMIC_RELEASE"); \
            __sync_lock_release(DST_PTR); }
#   define LIBXSMM_ATOMIC_SYNC(KIND) __sync_synchronize()
# endif
# if defined(LIBXSMM_ATOMIC_ZERO_STORE)
#   define LIBXSMM_ATOMIC_STORE_ZERO(DST_PTR, KIND) LIBXSMM_ATOMIC_STORE(DST_PTR, 0, KIND)
#   define LIBXSMM_ATOMIC_STORE_ZERO8(DST_PTR, KIND) LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_STORE, 8)(DST_PTR, 0, KIND)
#   define LIBXSMM_ATOMIC_STORE_ZERO16(DST_PTR, KIND) LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_STORE, 16)(DST_PTR, 0, KIND)
#   define LIBXSMM_ATOMIC_STORE_ZERO64(DST_PTR, KIND) LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_STORE, 64)(DST_PTR, 0, KIND)
# endif
# if !defined(LIBXSMM_ATOMIC_TRYLOCK_CMPSWP)
#   define LIBXSMM_ATOMIC_TRYLOCK(DST_PTR, KIND) /* matches bit-width of LIBXSMM_ATOMIC_LOCKTYPE */ \
            (0 == LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_FETCH_OR, 8)(DST_PTR, 1, KIND))
# endif
# define LIBXSMM_ATOMIC_ACQUIRE(DST_PTR, NPAUSE, KIND) \
          LIBXSMM_ASSERT(0 == LIBXSMM_MOD2((uintptr_t)(DST_PTR), 4)); \
          while (!LIBXSMM_ATOMIC_TRYLOCK(DST_PTR, KIND)) LIBXSMM_SYNC_CYCLE(DST_PTR, 0/*free*/, NPAUSE); \
          LIBXSMM_ASSERT_MSG(0 != *(DST_PTR), "LIBXSMM_ATOMIC_ACQUIRE")
# if !defined(LIBXSMM_SYNC_NPAUSE)
#   define LIBXSMM_SYNC_NPAUSE 4096
# endif
#elif defined(_WIN32)
# define LIBXSMM_ATOMIC(FN, BITS) LIBXSMM_CONCATENATE(LIBXSMM_ATOMIC, BITS)(FN)
# define LIBXSMM_ATOMIC8(FN) LIBXSMM_CONCATENATE(FN, 8)
# define LIBXSMM_ATOMIC16(FN) LIBXSMM_CONCATENATE(FN, 16)
# define LIBXSMM_ATOMIC32(FN) FN/*default*/
# define LIBXSMM_ATOMIC64(FN) LIBXSMM_CONCATENATE(FN, 64)
# define LIBXSMM_ATOMIC_LOAD(SRC_PTR, KIND) InterlockedOr((volatile LONG*)(SRC_PTR), 0)
# define LIBXSMM_ATOMIC_LOAD8(SRC_PTR, KIND) _InterlockedOr8((volatile char*)(SRC_PTR), 0)
# define LIBXSMM_ATOMIC_LOAD64(SRC_PTR, KIND) InterlockedOr64((volatile LONGLONG*)(SRC_PTR), 0)
# define LIBXSMM_ATOMIC_STORE(DST_PTR, VALUE, KIND) InterlockedExchange((volatile LONG*)(DST_PTR), (LONG)(VALUE))
# define LIBXSMM_ATOMIC_STORE8(DST_PTR, VALUE, KIND) InterlockedExchange8((volatile char*)(DST_PTR), (LONGLONG)(VALUE))
# define LIBXSMM_ATOMIC_STORE64(DST_PTR, VALUE, KIND) InterlockedExchange64((volatile LONGLONG*)(DST_PTR), (LONGLONG)(VALUE))
# if defined(LIBXSMM_ATOMIC_ZERO_STORE)
#   define LIBXSMM_ATOMIC_STORE_ZERO(DST_PTR, KIND) LIBXSMM_ATOMIC_STORE(DST_PTR, 0, KIND)
#   define LIBXSMM_ATOMIC_STORE_ZERO8(DST_PTR, KIND) LIBXSMM_ATOMIC_STORE8(DST_PTR, 0, KIND)
#   define LIBXSMM_ATOMIC_STORE_ZERO64(DST_PTR, KIND) LIBXSMM_ATOMIC_STORE64(DST_PTR, 0, KIND)
# else
#   define LIBXSMM_ATOMIC_STORE_ZERO(DST_PTR, KIND) InterlockedAnd((volatile LONG*)(DST_PTR), 0)
#   define LIBXSMM_ATOMIC_STORE_ZERO8(DST_PTR, KIND) InterlockedAnd8((volatile char*)(DST_PTR), 0)
#   define LIBXSMM_ATOMIC_STORE_ZERO64(DST_PTR, KIND) InterlockedAnd64((volatile LONGLONG*)(DST_PTR), 0)
# endif
# define LIBXSMM_ATOMIC_FETCH_OR(DST_PTR, VALUE, KIND) InterlockedOr((volatile LONG*)(DST_PTR), (LONG)VALUE)
# define LIBXSMM_ATOMIC_FETCH_OR8(DST_PTR, VALUE, KIND) _InterlockedOr8((volatile char*)(DST_PTR), VALUE)
# define LIBXSMM_ATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND) (LIBXSMM_ATOMIC_FETCH_ADD(DST_PTR, VALUE, KIND) + (VALUE))
# define LIBXSMM_ATOMIC_ADD_FETCH16(DST_PTR, VALUE, KIND) (LIBXSMM_ATOMIC_FETCH_ADD16(DST_PTR, VALUE, KIND) + (VALUE))
# define LIBXSMM_ATOMIC_ADD_FETCH64(DST_PTR, VALUE, KIND) (LIBXSMM_ATOMIC_FETCH_ADD64(DST_PTR, VALUE, KIND) + (VALUE))
# define LIBXSMM_ATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND) ((LONG)LIBXSMM_ATOMIC_FETCH_SUB(DST_PTR, VALUE, KIND) - ((LONG)VALUE))
# define LIBXSMM_ATOMIC_SUB_FETCH16(DST_PTR, VALUE, KIND) (LIBXSMM_ATOMIC_FETCH_SUB16(DST_PTR, VALUE, KIND) - (VALUE))
# define LIBXSMM_ATOMIC_SUB_FETCH64(DST_PTR, VALUE, KIND) (LIBXSMM_ATOMIC_FETCH_SUB64(DST_PTR, VALUE, KIND) - (VALUE))
# define LIBXSMM_ATOMIC_FETCH_ADD(DST_PTR, VALUE, KIND) InterlockedExchangeAdd((volatile LONG*)(DST_PTR), (LONG)VALUE)
# define LIBXSMM_ATOMIC_FETCH_ADD16(DST_PTR, VALUE, KIND) _InterlockedExchangeAdd16((volatile SHORT*)(DST_PTR), (SHORT)VALUE)
# define LIBXSMM_ATOMIC_FETCH_ADD64(DST_PTR, VALUE, KIND) InterlockedExchangeAdd64((volatile LONGLONG*)(DST_PTR), (LONGLONG)VALUE)
# define LIBXSMM_ATOMIC_FETCH_SUB(DST_PTR, VALUE, KIND) LIBXSMM_ATOMIC_FETCH_ADD(DST_PTR, -1 * (VALUE), KIND)
# define LIBXSMM_ATOMIC_FETCH_SUB16(DST_PTR, VALUE, KIND) LIBXSMM_ATOMIC_FETCH_ADD16(DST_PTR, -1 * (VALUE), KIND)
# define LIBXSMM_ATOMIC_FETCH_SUB64(DST_PTR, VALUE, KIND) LIBXSMM_ATOMIC_FETCH_ADD64(DST_PTR, -1 * (VALUE), KIND)
# define LIBXSMM_ATOMIC_CMPSWP(DST_PTR, OLDVAL, NEWVAL, KIND) \
    (((LONG)(OLDVAL)) == InterlockedCompareExchange((volatile LONG*)(DST_PTR), (LONG)NEWVAL, (LONG)OLDVAL))
# define LIBXSMM_ATOMIC_CMPSWP8(DST_PTR, OLDVAL, NEWVAL, KIND) \
    ((OLDVAL) == _InterlockedCompareExchange8((volatile char*)(DST_PTR), NEWVAL, OLDVAL))
# define LIBXSMM_ATOMIC_CMPSWP64(DST_PTR, OLDVAL, NEWVAL, KIND) \
    (((LONG64)(OLDVAL)) == InterlockedCompareExchange64((volatile LONG64*)(DST_PTR), NEWVAL, OLDVAL))
# if defined(LIBXSMM_ATOMIC_TRYLOCK_CMPSWP)
#   define LIBXSMM_ATOMIC_TRYLOCK(DST_PTR, KIND) LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_CMPSWP, 8)(DST_PTR, 0, 1, KIND)
# else
#   define LIBXSMM_ATOMIC_TRYLOCK(DST_PTR, KIND) (0 == LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_FETCH_OR, 8)(DST_PTR, 1, KIND))
# endif
# define LIBXSMM_ATOMIC_ACQUIRE(DST_PTR, NPAUSE, KIND) do { \
          LIBXSMM_ASSERT(0 == LIBXSMM_MOD2((uintptr_t)(DST_PTR), 4)); \
          while (!LIBXSMM_ATOMIC_TRYLOCK(DST_PTR, KIND)) LIBXSMM_SYNC_CYCLE(DST_PTR, 0/*free*/, NPAUSE); \
          LIBXSMM_ASSERT_MSG(0 != *(DST_PTR), "LIBXSMM_ATOMIC_ACQUIRE"); } while(0)
# define LIBXSMM_ATOMIC_RELEASE(DST_PTR, KIND) do { \
          LIBXSMM_ASSERT_MSG(0 != *(DST_PTR), "LIBXSMM_ATOMIC_RELEASE"); \
          LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_STORE_ZERO, 8)(DST_PTR, KIND); } while(0)
# define LIBXSMM_ATOMIC_SYNC(KIND) _ReadWriteBarrier()
# if !defined(LIBXSMM_SYNC_NPAUSE)
#   define LIBXSMM_SYNC_NPAUSE 4096
# endif
#else /* consider to permit LIBXSMM_SYNC_NONE */
# error LIBXSMM is missing atomic compiler builtins!
#endif

#if !defined(LIBXSMM_SYNC_CYCLE)
# if (0 < LIBXSMM_SYNC_NPAUSE)
#   define LIBXSMM_SYNC_CYCLE_ELSE(DST_PTR, EXP_STATE, NPAUSE, ELSE) do { int libxsmm_sync_cycle_npause_ = 1; \
      do { int libxsmm_sync_cycle_counter_ = 0; \
        for (; libxsmm_sync_cycle_counter_ < libxsmm_sync_cycle_npause_; ++libxsmm_sync_cycle_counter_) LIBXSMM_SYNC_PAUSE; \
        if (libxsmm_sync_cycle_npause_ < (NPAUSE)) { \
          libxsmm_sync_cycle_npause_ *= 2; \
        } \
        else { \
          libxsmm_sync_cycle_npause_ = (NPAUSE); \
          LIBXSMM_SYNC_YIELD; \
          ELSE \
        } \
      } while(((EXP_STATE) & 1) != (*(DST_PTR) & 1)); \
    } while(0)
# else
#   define LIBXSMM_SYNC_CYCLE_ELSE(DST_PTR, EXP_STATE, NPAUSE, ELSE) LIBXSMM_SYNC_PAUSE
# endif
# define LIBXSMM_SYNC_CYCLE(DST_PTR, EXP_STATE, NPAUSE) \
    LIBXSMM_SYNC_CYCLE_ELSE(DST_PTR, EXP_STATE, NPAUSE, /*else*/;)
#endif

#if (0 != LIBXSMM_SYNC)
# define LIBXSMM_LOCK_DEFAULT LIBXSMM_LOCK_SPINLOCK
# if !defined(LIBXSMM_LOCK_SYSTEM_SPINLOCK) && !(defined(_OPENMP) && defined(LIBXSMM_SYNC_OMP)) && \
    (!defined(__linux__) || defined(__USE_XOPEN2K)) && 0/*disabled*/
#   define LIBXSMM_LOCK_SYSTEM_SPINLOCK
# endif
# if !defined(LIBXSMM_LOCK_SYSTEM_MUTEX) && !(defined(_OPENMP) && defined(LIBXSMM_SYNC_OMP))
#   define LIBXSMM_LOCK_SYSTEM_MUTEX
# endif
# if !defined(LIBXSMM_LOCK_SYSTEM_RWLOCK) && !(defined(_OPENMP) && defined(LIBXSMM_SYNC_OMP)) && \
    (!defined(__linux__) || defined(__USE_XOPEN2K) || defined(__USE_UNIX98))
#   define LIBXSMM_LOCK_SYSTEM_RWLOCK
# endif
  /* Lock type, initialization, destruction, (try-)lock, unlock, etc */
# define LIBXSMM_LOCK_ACQUIRED(KIND) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_ACQUIRED_, KIND)
# define LIBXSMM_LOCK_TYPE_ISPOD(KIND) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_TYPE_ISPOD_, KIND)
# define LIBXSMM_LOCK_TYPE_ISRW(KIND) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_TYPE_ISRW_, KIND)
# define LIBXSMM_LOCK_TYPE(KIND) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_TYPE_, KIND)
# define LIBXSMM_LOCK_INIT(KIND, LOCK, ATTR) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_INIT_, KIND)(LOCK, ATTR)
# define LIBXSMM_LOCK_DESTROY(KIND, LOCK) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_DESTROY_, KIND)(LOCK)
# define LIBXSMM_LOCK_TRYLOCK(KIND, LOCK) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_TRYLOCK_, KIND)(LOCK)
# define LIBXSMM_LOCK_ACQUIRE(KIND, LOCK) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_ACQUIRE_, KIND)(LOCK)
# define LIBXSMM_LOCK_RELEASE(KIND, LOCK) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_RELEASE_, KIND)(LOCK)
# define LIBXSMM_LOCK_TRYREAD(KIND, LOCK) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_TRYREAD_, KIND)(LOCK)
# define LIBXSMM_LOCK_ACQREAD(KIND, LOCK) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_ACQREAD_, KIND)(LOCK)
# define LIBXSMM_LOCK_RELREAD(KIND, LOCK) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_RELREAD_, KIND)(LOCK)
  /* Attribute type, initialization, destruction */
# define LIBXSMM_LOCK_ATTR_TYPE(KIND) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_ATTR_TYPE_, KIND)
# define LIBXSMM_LOCK_ATTR_INIT(KIND, ATTR) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_ATTR_INIT_, KIND)(ATTR)
# define LIBXSMM_LOCK_ATTR_DESTROY(KIND, ATTR) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_ATTR_DESTROY_, KIND)(ATTR)
  /* Cygwin's Pthread implementation appears to be broken; use Win32 */
# if !defined(LIBXSMM_WIN32_THREADS) && (defined(_WIN32) || defined(__CYGWIN__))
#   define LIBXSMM_WIN32_THREADS _WIN32_WINNT
#   if defined(__CYGWIN__) || defined(__MINGW32__) /* hack: make SRW-locks available */
#     if defined(_WIN32_WINNT)
#       undef _WIN32_WINNT
#       if !defined(NTDDI_VERSION)
#         define NTDDI_VERSION 0x0600
#       endif
#       define _WIN32_WINNT ((LIBXSMM_WIN32_THREADS) | 0x0600)
#     else
#       define _WIN32_WINNT 0x0600
#     endif
#   endif
# endif
# if defined(LIBXSMM_WIN32_THREADS)
#   define LIBXSMM_TLS_TYPE DWORD
#   define LIBXSMM_TLS_CREATE(KEYPTR) *(KEYPTR) = TlsAlloc()
#   define LIBXSMM_TLS_DESTROY(KEY) TlsFree(KEY)
#   define LIBXSMM_TLS_SETVALUE(KEY, PTR) TlsSetValue(KEY, PTR)
#   define LIBXSMM_TLS_GETVALUE(KEY) TlsGetValue(KEY)
#   define LIBXSMM_LOCK_SPINLOCK spin
#   if ((LIBXSMM_WIN32_THREADS) & 0x0600)
#     define LIBXSMM_LOCK_MUTEX rwlock
#     define LIBXSMM_LOCK_RWLOCK rwlock
#   else /* mutex exposes high latency */
#     define LIBXSMM_LOCK_MUTEX mutex
#     define LIBXSMM_LOCK_RWLOCK mutex
#   endif
#   if defined(LIBXSMM_LOCK_SYSTEM_SPINLOCK)
#     define LIBXSMM_LOCK_ACQUIRED_spin TRUE
#     define LIBXSMM_LOCK_TYPE_ISPOD_spin 0
#     define LIBXSMM_LOCK_TYPE_spin CRITICAL_SECTION
#     define LIBXSMM_LOCK_INIT_spin(LOCK, ATTR) { LIBXSMM_UNUSED(ATTR); InitializeCriticalSection(LOCK); }
#     define LIBXSMM_LOCK_DESTROY_spin(LOCK) DeleteCriticalSection((LIBXSMM_LOCK_TYPE_spin*)(LOCK))
#     define LIBXSMM_LOCK_TRYLOCK_spin(LOCK) TryEnterCriticalSection(LOCK)
#     define LIBXSMM_LOCK_ACQUIRE_spin(LOCK) EnterCriticalSection(LOCK)
#     define LIBXSMM_LOCK_RELEASE_spin(LOCK) LeaveCriticalSection(LOCK)
#     define LIBXSMM_LOCK_TRYREAD_spin(LOCK) LIBXSMM_LOCK_TRYLOCK_spin(LOCK)
#     define LIBXSMM_LOCK_ACQREAD_spin(LOCK) LIBXSMM_LOCK_ACQUIRE_spin(LOCK)
#     define LIBXSMM_LOCK_RELREAD_spin(LOCK) LIBXSMM_LOCK_RELEASE_spin(LOCK)
#     define LIBXSMM_LOCK_ATTR_TYPE_spin int
#     define LIBXSMM_LOCK_ATTR_INIT_spin(ATTR) LIBXSMM_UNUSED(ATTR)
#     define LIBXSMM_LOCK_ATTR_DESTROY_spin(ATTR) LIBXSMM_UNUSED(ATTR)
#   endif
#   if defined(LIBXSMM_LOCK_SYSTEM_MUTEX)
#     define LIBXSMM_LOCK_ACQUIRED_mutex WAIT_OBJECT_0
#     define LIBXSMM_LOCK_TYPE_ISPOD_mutex 0
#     define LIBXSMM_LOCK_TYPE_ISRW_mutex 0
#     define LIBXSMM_LOCK_TYPE_mutex HANDLE
#     define LIBXSMM_LOCK_INIT_mutex(LOCK, ATTR) (*(LOCK) = CreateMutex(*(ATTR), FALSE, NULL))
#     define LIBXSMM_LOCK_DESTROY_mutex(LOCK) CloseHandle(*(LOCK))
#     define LIBXSMM_LOCK_TRYLOCK_mutex(LOCK) WaitForSingleObject(*(LOCK), 0)
#     define LIBXSMM_LOCK_ACQUIRE_mutex(LOCK) WaitForSingleObject(*(LOCK), INFINITE)
#     define LIBXSMM_LOCK_RELEASE_mutex(LOCK) ReleaseMutex(*(LOCK))
#     define LIBXSMM_LOCK_TRYREAD_mutex(LOCK) LIBXSMM_LOCK_TRYLOCK_mutex(LOCK)
#     define LIBXSMM_LOCK_ACQREAD_mutex(LOCK) LIBXSMM_LOCK_ACQUIRE_mutex(LOCK)
#     define LIBXSMM_LOCK_RELREAD_mutex(LOCK) LIBXSMM_LOCK_RELEASE_mutex(LOCK)
#     define LIBXSMM_LOCK_ATTR_TYPE_mutex LPSECURITY_ATTRIBUTES
#     define LIBXSMM_LOCK_ATTR_INIT_mutex(ATTR) (*(ATTR) = NULL)
#     define LIBXSMM_LOCK_ATTR_DESTROY_mutex(ATTR) LIBXSMM_UNUSED(ATTR)
#   endif
#   if defined(LIBXSMM_LOCK_SYSTEM_RWLOCK)
#     define LIBXSMM_LOCK_ACQUIRED_rwlock TRUE
#     define LIBXSMM_LOCK_TYPE_ISPOD_rwlock 1
#     define LIBXSMM_LOCK_TYPE_ISRW_rwlock 1
#     define LIBXSMM_LOCK_TYPE_rwlock SRWLOCK
#     define LIBXSMM_LOCK_INIT_rwlock(LOCK, ATTR) { LIBXSMM_UNUSED(ATTR); InitializeSRWLock(LOCK); }
#     define LIBXSMM_LOCK_DESTROY_rwlock(LOCK) LIBXSMM_UNUSED(LOCK)
#     define LIBXSMM_LOCK_TRYLOCK_rwlock(LOCK) TryAcquireSRWLockExclusive(LOCK)
#     define LIBXSMM_LOCK_ACQUIRE_rwlock(LOCK) AcquireSRWLockExclusive(LOCK)
#     define LIBXSMM_LOCK_RELEASE_rwlock(LOCK) ReleaseSRWLockExclusive(LOCK)
#     define LIBXSMM_LOCK_TRYREAD_rwlock(LOCK) TryAcquireSRWLockShared(LOCK)
#     define LIBXSMM_LOCK_ACQREAD_rwlock(LOCK) AcquireSRWLockShared(LOCK)
#     define LIBXSMM_LOCK_RELREAD_rwlock(LOCK) ReleaseSRWLockShared(LOCK)
#     define LIBXSMM_LOCK_ATTR_TYPE_rwlock int
#     define LIBXSMM_LOCK_ATTR_INIT_rwlock(ATTR) LIBXSMM_UNUSED(ATTR)
#     define LIBXSMM_LOCK_ATTR_DESTROY_rwlock(ATTR) LIBXSMM_UNUSED(ATTR)
#   endif
#   define LIBXSMM_SYNC_YIELD YieldProcessor()
# else
#   define LIBXSMM_TLS_TYPE pthread_key_t
#   define LIBXSMM_TLS_CREATE(KEYPTR) pthread_key_create(KEYPTR, NULL)
#   define LIBXSMM_TLS_DESTROY(KEY) pthread_key_delete(KEY)
#   define LIBXSMM_TLS_SETVALUE(KEY, PTR) pthread_setspecific(KEY, PTR)
#   define LIBXSMM_TLS_GETVALUE(KEY) pthread_getspecific(KEY)
#   if defined(__APPLE__) && defined(__MACH__)
      LIBXSMM_EXTERN void pthread_jit_write_protect_np(int) LIBXSMM_NOTHROW;
      LIBXSMM_EXTERN void pthread_yield_np(void) LIBXSMM_NOTHROW;
#     define LIBXSMM_SYNC_YIELD pthread_yield_np()
#   elif defined(_POSIX_PRIORITY_SCHEDULING) || (defined(__GLIBC__) && defined(__GLIBC_MINOR__) \
      && LIBXSMM_VERSION2(2, 34) <= LIBXSMM_VERSION2(__GLIBC__, __GLIBC_MINOR__))
      LIBXSMM_EXTERN int sched_yield(void) LIBXSMM_NOTHROW;
#     define LIBXSMM_SYNC_YIELD sched_yield()
#   else
#     if defined(__USE_GNU) || !defined(__BSD_VISIBLE)
      LIBXSMM_EXTERN int pthread_yield(void) LIBXSMM_NOTHROW;
#     else
      LIBXSMM_EXTERN void pthread_yield(void) LIBXSMM_NOTHROW;
#     endif
#     define LIBXSMM_SYNC_YIELD pthread_yield()
#   endif
#   if defined(LIBXSMM_LOCK_SYSTEM_SPINLOCK) && defined(__APPLE__) && defined(__MACH__)
#     define LIBXSMM_LOCK_SPINLOCK mutex
#   else
#     define LIBXSMM_LOCK_SPINLOCK spin
#   endif
#   define LIBXSMM_LOCK_MUTEX mutex
#   define LIBXSMM_LOCK_RWLOCK rwlock
#   if defined(LIBXSMM_LOCK_SYSTEM_SPINLOCK)
#     define LIBXSMM_LOCK_ACQUIRED_spin 0
#     define LIBXSMM_LOCK_TYPE_ISPOD_spin 0
#     define LIBXSMM_LOCK_TYPE_ISRW_spin 0
#     define LIBXSMM_LOCK_TYPE_spin pthread_spinlock_t
#     define LIBXSMM_LOCK_INIT_spin(LOCK, ATTR) LIBXSMM_EXPECT(0 == pthread_spin_init(LOCK, *(ATTR)))
#     define LIBXSMM_LOCK_DESTROY_spin(LOCK) LIBXSMM_EXPECT(0 == pthread_spin_destroy(LOCK))
#     define LIBXSMM_LOCK_TRYLOCK_spin(LOCK) pthread_spin_trylock(LOCK)
#     define LIBXSMM_LOCK_ACQUIRE_spin(LOCK) LIBXSMM_EXPECT(0 == pthread_spin_lock(LOCK))
#     define LIBXSMM_LOCK_RELEASE_spin(LOCK) LIBXSMM_EXPECT(0 == pthread_spin_unlock(LOCK))
#     define LIBXSMM_LOCK_TRYREAD_spin(LOCK) LIBXSMM_LOCK_TRYLOCK_spin(LOCK)
#     define LIBXSMM_LOCK_ACQREAD_spin(LOCK) LIBXSMM_LOCK_ACQUIRE_spin(LOCK)
#     define LIBXSMM_LOCK_RELREAD_spin(LOCK) LIBXSMM_LOCK_RELEASE_spin(LOCK)
#     define LIBXSMM_LOCK_ATTR_TYPE_spin int
#     define LIBXSMM_LOCK_ATTR_INIT_spin(ATTR) (*(ATTR) = 0)
#     define LIBXSMM_LOCK_ATTR_DESTROY_spin(ATTR) LIBXSMM_UNUSED(ATTR)
#   endif
#   if defined(LIBXSMM_LOCK_SYSTEM_MUTEX)
#     define LIBXSMM_LOCK_ACQUIRED_mutex 0
#     define LIBXSMM_LOCK_TYPE_ISPOD_mutex 0
#     define LIBXSMM_LOCK_TYPE_ISRW_mutex 0
#     define LIBXSMM_LOCK_TYPE_mutex pthread_mutex_t
#     define LIBXSMM_LOCK_INIT_mutex(LOCK, ATTR) LIBXSMM_EXPECT(0 == pthread_mutex_init(LOCK, ATTR))
#     define LIBXSMM_LOCK_DESTROY_mutex(LOCK) LIBXSMM_EXPECT_DEBUG(0 == pthread_mutex_destroy(LOCK))
#     define LIBXSMM_LOCK_TRYLOCK_mutex(LOCK) pthread_mutex_trylock(LOCK) /*!LIBXSMM_EXPECT*/
#     define LIBXSMM_LOCK_ACQUIRE_mutex(LOCK) LIBXSMM_EXPECT(0 == pthread_mutex_lock(LOCK))
#     define LIBXSMM_LOCK_RELEASE_mutex(LOCK) LIBXSMM_EXPECT(0 == pthread_mutex_unlock(LOCK))
#     define LIBXSMM_LOCK_TRYREAD_mutex(LOCK) LIBXSMM_LOCK_TRYLOCK_mutex(LOCK)
#     define LIBXSMM_LOCK_ACQREAD_mutex(LOCK) LIBXSMM_LOCK_ACQUIRE_mutex(LOCK)
#     define LIBXSMM_LOCK_RELREAD_mutex(LOCK) LIBXSMM_LOCK_RELEASE_mutex(LOCK)
#     define LIBXSMM_LOCK_ATTR_TYPE_mutex pthread_mutexattr_t
#     if !defined(__linux__) || defined(__USE_UNIX98) || defined(__USE_XOPEN2K8)
#       if defined(_DEBUG)
#         define LIBXSMM_LOCK_ATTR_INIT_mutex(ATTR) (LIBXSMM_EXPECT(0 == pthread_mutexattr_init(ATTR)), \
                 LIBXSMM_EXPECT(0 == pthread_mutexattr_settype(ATTR, PTHREAD_MUTEX_ERRORCHECK)))
#       else
#         define LIBXSMM_LOCK_ATTR_INIT_mutex(ATTR) (pthread_mutexattr_init(ATTR), \
                 pthread_mutexattr_settype(ATTR, PTHREAD_MUTEX_NORMAL))
#       endif
#     else
#       define LIBXSMM_LOCK_ATTR_INIT_mutex(ATTR) pthread_mutexattr_init(ATTR)
#     endif
#     define LIBXSMM_LOCK_ATTR_DESTROY_mutex(ATTR) LIBXSMM_EXPECT(0 == pthread_mutexattr_destroy(ATTR))
#   endif
#   if defined(LIBXSMM_LOCK_SYSTEM_RWLOCK)
#     define LIBXSMM_LOCK_ACQUIRED_rwlock 0
#     define LIBXSMM_LOCK_TYPE_ISPOD_rwlock 0
#     define LIBXSMM_LOCK_TYPE_ISRW_rwlock 1
#     define LIBXSMM_LOCK_TYPE_rwlock pthread_rwlock_t
#     define LIBXSMM_LOCK_INIT_rwlock(LOCK, ATTR) LIBXSMM_EXPECT(0 == pthread_rwlock_init(LOCK, ATTR))
#     define LIBXSMM_LOCK_DESTROY_rwlock(LOCK) LIBXSMM_EXPECT(0 == pthread_rwlock_destroy(LOCK))
#     define LIBXSMM_LOCK_TRYLOCK_rwlock(LOCK) pthread_rwlock_trywrlock(LOCK)
#     define LIBXSMM_LOCK_ACQUIRE_rwlock(LOCK) LIBXSMM_EXPECT(0 == pthread_rwlock_wrlock(LOCK))
#     define LIBXSMM_LOCK_RELEASE_rwlock(LOCK) LIBXSMM_EXPECT(0 == pthread_rwlock_unlock(LOCK))
#     define LIBXSMM_LOCK_TRYREAD_rwlock(LOCK) pthread_rwlock_tryrdlock(LOCK)
#     define LIBXSMM_LOCK_ACQREAD_rwlock(LOCK) LIBXSMM_EXPECT(0 == pthread_rwlock_rdlock(LOCK))
#     define LIBXSMM_LOCK_RELREAD_rwlock(LOCK) LIBXSMM_LOCK_RELEASE_rwlock(LOCK)
#     define LIBXSMM_LOCK_ATTR_TYPE_rwlock pthread_rwlockattr_t
#     define LIBXSMM_LOCK_ATTR_INIT_rwlock(ATTR) LIBXSMM_EXPECT(0 == pthread_rwlockattr_init(ATTR))
#     define LIBXSMM_LOCK_ATTR_DESTROY_rwlock(ATTR) LIBXSMM_EXPECT(0 == pthread_rwlockattr_destroy(ATTR))
#   endif
# endif
/* OpenMP based locks need to stay disabled unless both
 * libxsmm and libxsmmext are built with OpenMP support.
 */
# if defined(_OPENMP) && defined(LIBXSMM_SYNC_OMP)
#   include <omp.h>
#   if !defined(LIBXSMM_LOCK_SYSTEM_SPINLOCK)
#     define LIBXSMM_LOCK_ACQUIRED_spin 1
#     define LIBXSMM_LOCK_TYPE_ISPOD_spin 0
#     define LIBXSMM_LOCK_TYPE_ISRW_spin 0
#     define LIBXSMM_LOCK_TYPE_spin omp_lock_t
#     define LIBXSMM_LOCK_DESTROY_spin(LOCK) omp_destroy_lock(LOCK)
#     define LIBXSMM_LOCK_TRYLOCK_spin(LOCK) omp_test_lock(LOCK)
#     define LIBXSMM_LOCK_ACQUIRE_spin(LOCK) omp_set_lock(LOCK)
#     define LIBXSMM_LOCK_RELEASE_spin(LOCK) omp_unset_lock(LOCK)
#     define LIBXSMM_LOCK_TRYREAD_spin(LOCK) LIBXSMM_LOCK_TRYLOCK_spin(LOCK)
#     define LIBXSMM_LOCK_ACQREAD_spin(LOCK) LIBXSMM_LOCK_ACQUIRE_spin(LOCK)
#     define LIBXSMM_LOCK_RELREAD_spin(LOCK) LIBXSMM_LOCK_RELEASE_spin(LOCK)
#     if (201811 <= _OPENMP/*v5.0*/)
#       define LIBXSMM_LOCK_INIT_spin(LOCK, ATTR) omp_init_lock_with_hint(LOCK, *(ATTR))
#       define LIBXSMM_LOCK_ATTR_TYPE_spin omp_lock_hint_t
#       define LIBXSMM_LOCK_ATTR_INIT_spin(ATTR) (*(ATTR) = omp_lock_hint_none)
#     else
#       define LIBXSMM_LOCK_INIT_spin(LOCK, ATTR) { LIBXSMM_UNUSED(ATTR); omp_init_lock(LOCK); }
#       define LIBXSMM_LOCK_ATTR_TYPE_spin const void*
#       define LIBXSMM_LOCK_ATTR_INIT_spin(ATTR) LIBXSMM_UNUSED(ATTR)
#     endif
#     define LIBXSMM_LOCK_ATTR_DESTROY_spin(ATTR) LIBXSMM_UNUSED(ATTR)
#   endif
#   if !defined(LIBXSMM_LOCK_SYSTEM_MUTEX)
#     define LIBXSMM_LOCK_ACQUIRED_mutex 1
#     define LIBXSMM_LOCK_TYPE_ISPOD_mutex 0
#     define LIBXSMM_LOCK_TYPE_ISRW_mutex 0
#     define LIBXSMM_LOCK_TYPE_mutex omp_lock_t
#     define LIBXSMM_LOCK_DESTROY_mutex(LOCK) omp_destroy_lock(LOCK)
#     define LIBXSMM_LOCK_TRYLOCK_mutex(LOCK) omp_test_lock(LOCK)
#     define LIBXSMM_LOCK_ACQUIRE_mutex(LOCK) omp_set_lock(LOCK)
#     define LIBXSMM_LOCK_RELEASE_mutex(LOCK) omp_unset_lock(LOCK)
#     define LIBXSMM_LOCK_TRYREAD_mutex(LOCK) LIBXSMM_LOCK_TRYLOCK_mutex(LOCK)
#     define LIBXSMM_LOCK_ACQREAD_mutex(LOCK) LIBXSMM_LOCK_ACQUIRE_mutex(LOCK)
#     define LIBXSMM_LOCK_RELREAD_mutex(LOCK) LIBXSMM_LOCK_RELEASE_mutex(LOCK)
#     if (201811 <= _OPENMP/*v5.0*/)
#       define LIBXSMM_LOCK_INIT_mutex(LOCK, ATTR) omp_init_lock_with_hint(LOCK, *(ATTR))
#       define LIBXSMM_LOCK_ATTR_TYPE_mutex omp_lock_hint_t
#       define LIBXSMM_LOCK_ATTR_INIT_mutex(ATTR) (*(ATTR) = omp_lock_hint_none)
#     else
#       define LIBXSMM_LOCK_INIT_mutex(LOCK, ATTR) { LIBXSMM_UNUSED(ATTR); omp_init_lock(LOCK); }
#       define LIBXSMM_LOCK_ATTR_TYPE_mutex const void*
#       define LIBXSMM_LOCK_ATTR_INIT_mutex(ATTR) LIBXSMM_UNUSED(ATTR)
#     endif
#     define LIBXSMM_LOCK_ATTR_DESTROY_mutex(ATTR) LIBXSMM_UNUSED(ATTR)
#   endif
#   if !defined(LIBXSMM_LOCK_SYSTEM_RWLOCK)
#     define LIBXSMM_LOCK_ACQUIRED_rwlock 1
#     define LIBXSMM_LOCK_TYPE_ISPOD_rwlock 0
#     define LIBXSMM_LOCK_TYPE_ISRW_rwlock 0
#     define LIBXSMM_LOCK_TYPE_rwlock omp_lock_t
#     define LIBXSMM_LOCK_DESTROY_rwlock(LOCK) omp_destroy_lock(LOCK)
#     define LIBXSMM_LOCK_TRYLOCK_rwlock(LOCK) omp_test_lock(LOCK)
#     define LIBXSMM_LOCK_ACQUIRE_rwlock(LOCK) omp_set_lock(LOCK)
#     define LIBXSMM_LOCK_RELEASE_rwlock(LOCK) omp_unset_lock(LOCK)
#     define LIBXSMM_LOCK_TRYREAD_rwlock(LOCK) LIBXSMM_LOCK_TRYLOCK_rwlock(LOCK)
#     define LIBXSMM_LOCK_ACQREAD_rwlock(LOCK) LIBXSMM_LOCK_ACQUIRE_rwlock(LOCK)
#     define LIBXSMM_LOCK_RELREAD_rwlock(LOCK) LIBXSMM_LOCK_RELEASE_rwlock(LOCK)
#     if (201811 <= _OPENMP/*v5.0*/)
#       define LIBXSMM_LOCK_INIT_rwlock(LOCK, ATTR) omp_init_lock_with_hint(LOCK, *(ATTR))
#       define LIBXSMM_LOCK_ATTR_TYPE_rwlock omp_lock_hint_t
#       define LIBXSMM_LOCK_ATTR_INIT_rwlock(ATTR) (*(ATTR) = omp_lock_hint_none)
#     else
#       define LIBXSMM_LOCK_INIT_rwlock(LOCK, ATTR) { LIBXSMM_UNUSED(ATTR); omp_init_lock(LOCK); }
#       define LIBXSMM_LOCK_ATTR_TYPE_rwlock const void*
#       define LIBXSMM_LOCK_ATTR_INIT_rwlock(ATTR) LIBXSMM_UNUSED(ATTR)
#     endif
#     define LIBXSMM_LOCK_ATTR_DESTROY_rwlock(ATTR) LIBXSMM_UNUSED(ATTR)
#   endif
# else /* based on atomic primitives */
#   if !defined(LIBXSMM_LOCK_SYSTEM_SPINLOCK)
#     define LIBXSMM_LOCK_ACQUIRED_spin 0
#     define LIBXSMM_LOCK_TYPE_ISPOD_spin 1
#     define LIBXSMM_LOCK_TYPE_ISRW_spin 0
#     define LIBXSMM_LOCK_TYPE_spin volatile LIBXSMM_ATOMIC_LOCKTYPE
#     define LIBXSMM_LOCK_INIT_spin(LOCK, ATTR) { LIBXSMM_UNUSED(ATTR); (*(LOCK) = 0); }
#     define LIBXSMM_LOCK_DESTROY_spin(LOCK) LIBXSMM_UNUSED(LOCK)
#     define LIBXSMM_LOCK_TRYLOCK_spin(LOCK) (LIBXSMM_LOCK_ACQUIRED_spin + !LIBXSMM_ATOMIC_TRYLOCK(LOCK, LIBXSMM_ATOMIC_RELAXED))
#     define LIBXSMM_LOCK_ACQUIRE_spin(LOCK) LIBXSMM_ATOMIC_ACQUIRE(LOCK, LIBXSMM_SYNC_NPAUSE, LIBXSMM_ATOMIC_RELAXED)
#     define LIBXSMM_LOCK_RELEASE_spin(LOCK) LIBXSMM_ATOMIC_RELEASE(LOCK, LIBXSMM_ATOMIC_RELAXED)
#     define LIBXSMM_LOCK_TRYREAD_spin(LOCK) LIBXSMM_LOCK_TRYLOCK_spin(LOCK)
#     define LIBXSMM_LOCK_ACQREAD_spin(LOCK) LIBXSMM_LOCK_ACQUIRE_spin(LOCK)
#     define LIBXSMM_LOCK_RELREAD_spin(LOCK) LIBXSMM_LOCK_RELEASE_spin(LOCK)
#     define LIBXSMM_LOCK_ATTR_TYPE_spin int
#     define LIBXSMM_LOCK_ATTR_INIT_spin(ATTR) LIBXSMM_UNUSED(ATTR)
#     define LIBXSMM_LOCK_ATTR_DESTROY_spin(ATTR) LIBXSMM_UNUSED(ATTR)
#   endif
#   if !defined(LIBXSMM_LOCK_SYSTEM_MUTEX)
#     define LIBXSMM_LOCK_ACQUIRED_mutex 0
#     define LIBXSMM_LOCK_TYPE_ISPOD_mutex 1
#     define LIBXSMM_LOCK_TYPE_ISRW_mutex 0
#     define LIBXSMM_LOCK_TYPE_mutex volatile LIBXSMM_ATOMIC_LOCKTYPE
#     define LIBXSMM_LOCK_INIT_mutex(LOCK, ATTR) { LIBXSMM_UNUSED(ATTR); (*(LOCK) = 0); }
#     define LIBXSMM_LOCK_DESTROY_mutex(LOCK) LIBXSMM_UNUSED(LOCK)
#     define LIBXSMM_LOCK_TRYLOCK_mutex(LOCK) (LIBXSMM_LOCK_ACQUIRED_mutex + !LIBXSMM_ATOMIC_TRYLOCK(LOCK, LIBXSMM_ATOMIC_RELAXED))
#     define LIBXSMM_LOCK_ACQUIRE_mutex(LOCK) LIBXSMM_ATOMIC_ACQUIRE(LOCK, LIBXSMM_SYNC_NPAUSE, LIBXSMM_ATOMIC_RELAXED)
#     define LIBXSMM_LOCK_RELEASE_mutex(LOCK) LIBXSMM_ATOMIC_RELEASE(LOCK, LIBXSMM_ATOMIC_RELAXED)
#     define LIBXSMM_LOCK_TRYREAD_mutex(LOCK) LIBXSMM_LOCK_TRYLOCK_mutex(LOCK)
#     define LIBXSMM_LOCK_ACQREAD_mutex(LOCK) LIBXSMM_LOCK_ACQUIRE_mutex(LOCK)
#     define LIBXSMM_LOCK_RELREAD_mutex(LOCK) LIBXSMM_LOCK_RELEASE_mutex(LOCK)
#     define LIBXSMM_LOCK_ATTR_TYPE_mutex int
#     define LIBXSMM_LOCK_ATTR_INIT_mutex(ATTR) LIBXSMM_UNUSED(ATTR)
#     define LIBXSMM_LOCK_ATTR_DESTROY_mutex(ATTR) LIBXSMM_UNUSED(ATTR)
#   endif
#   if !defined(LIBXSMM_LOCK_SYSTEM_RWLOCK)
#     define LIBXSMM_LOCK_ACQUIRED_rwlock 0
#     define LIBXSMM_LOCK_TYPE_ISPOD_rwlock 1
#     define LIBXSMM_LOCK_TYPE_ISRW_rwlock 0
#     define LIBXSMM_LOCK_TYPE_rwlock volatile LIBXSMM_ATOMIC_LOCKTYPE
#     define LIBXSMM_LOCK_INIT_rwlock(LOCK, ATTR) { LIBXSMM_UNUSED(ATTR); (*(LOCK) = 0); }
#     define LIBXSMM_LOCK_DESTROY_rwlock(LOCK) LIBXSMM_UNUSED(LOCK)
#     define LIBXSMM_LOCK_TRYLOCK_rwlock(LOCK) (LIBXSMM_LOCK_ACQUIRED_rwlock + !LIBXSMM_ATOMIC_TRYLOCK(LOCK, LIBXSMM_ATOMIC_RELAXED))
#     define LIBXSMM_LOCK_ACQUIRE_rwlock(LOCK) LIBXSMM_ATOMIC_ACQUIRE(LOCK, LIBXSMM_SYNC_NPAUSE, LIBXSMM_ATOMIC_RELAXED)
#     define LIBXSMM_LOCK_RELEASE_rwlock(LOCK) LIBXSMM_ATOMIC_RELEASE(LOCK, LIBXSMM_ATOMIC_RELAXED)
#     define LIBXSMM_LOCK_TRYREAD_rwlock(LOCK) LIBXSMM_LOCK_TRYLOCK_rwlock(LOCK)
#     define LIBXSMM_LOCK_ACQREAD_rwlock(LOCK) LIBXSMM_LOCK_ACQUIRE_rwlock(LOCK)
#     define LIBXSMM_LOCK_RELREAD_rwlock(LOCK) LIBXSMM_LOCK_RELEASE_rwlock(LOCK)
#     define LIBXSMM_LOCK_ATTR_TYPE_rwlock int
#     define LIBXSMM_LOCK_ATTR_INIT_rwlock(ATTR) LIBXSMM_UNUSED(ATTR)
#     define LIBXSMM_LOCK_ATTR_DESTROY_rwlock(ATTR) LIBXSMM_UNUSED(ATTR)
#   endif
# endif
#else /* no synchronization */
# define LIBXSMM_LOCK_DEFAULT int
# define LIBXSMM_SYNC_YIELD LIBXSMM_SYNC_PAUSE
# define LIBXSMM_LOCK_SPINLOCK spinlock_dummy
# define LIBXSMM_LOCK_MUTEX mutex_dummy
# define LIBXSMM_LOCK_RWLOCK rwlock_dummy
# define LIBXSMM_LOCK_ACQUIRED(KIND) 0
# define LIBXSMM_LOCK_TYPE_ISPOD(KIND) 1
# define LIBXSMM_LOCK_TYPE_ISRW(KIND) 0
# define LIBXSMM_LOCK_ATTR_TYPE(KIND) int
# define LIBXSMM_LOCK_ATTR_INIT(KIND, ATTR) LIBXSMM_UNUSED(ATTR)
# define LIBXSMM_LOCK_ATTR_DESTROY(KIND, ATTR) LIBXSMM_UNUSED(ATTR)
# define LIBXSMM_LOCK_TYPE(KIND) LIBXSMM_LOCK_DEFAULT
# define LIBXSMM_LOCK_INIT(KIND, LOCK, ATTR) { LIBXSMM_UNUSED(LOCK); LIBXSMM_UNUSED(ATTR); }
# define LIBXSMM_LOCK_DESTROY(KIND, LOCK) LIBXSMM_UNUSED(LOCK)
# define LIBXSMM_LOCK_TRYLOCK(KIND, LOCK) LIBXSMM_LOCK_ACQUIRED(KIND)
# define LIBXSMM_LOCK_ACQUIRE(KIND, LOCK) LIBXSMM_UNUSED(LOCK)
# define LIBXSMM_LOCK_RELEASE(KIND, LOCK) LIBXSMM_UNUSED(LOCK)
# define LIBXSMM_LOCK_TRYREAD(KIND, LOCK) LIBXSMM_LOCK_TRYLOCK(KIND, LOCK)
# define LIBXSMM_LOCK_ACQREAD(KIND, LOCK) LIBXSMM_LOCK_ACQUIRE(KIND, LOCK)
# define LIBXSMM_LOCK_RELREAD(KIND, LOCK) LIBXSMM_LOCK_RELEASE(KIND, LOCK)
#endif

#if (0 == LIBXSMM_SYNC)
# define LIBXSMM_FLOCK(FILE)
# define LIBXSMM_FUNLOCK(FILE)
#elif defined(_WIN32)
# define LIBXSMM_FLOCK(FILE) _lock_file(FILE)
# define LIBXSMM_FUNLOCK(FILE) _unlock_file(FILE)
#else
# if !defined(__CYGWIN__)
#   define LIBXSMM_FLOCK(FILE) flockfile(FILE)
#   define LIBXSMM_FUNLOCK(FILE) funlockfile(FILE)
LIBXSMM_EXTERN void flockfile(FILE*) LIBXSMM_NOTHROW;
LIBXSMM_EXTERN void funlockfile(FILE*) LIBXSMM_NOTHROW;
# else /* Only available with __CYGWIN__ *and* C++0x. */
#   define LIBXSMM_FLOCK(FILE)
#   define LIBXSMM_FUNLOCK(FILE)
# endif
#endif

/** Synchronize console output */
#define LIBXSMM_STDIO_ACQUIRE() libxsmm_stdio_acquire()
#define LIBXSMM_STDIO_RELEASE() libxsmm_stdio_release()


/** Utility function to receive the process ID of the calling process. */
LIBXSMM_API unsigned int libxsmm_get_pid(void);
/**
 * Utility function to receive a Thread-ID (TID) for the calling thread.
 * The TID is not related to a specific threading runtime. TID=0 may not
 * represent the main thread. TIDs are zero-based and consecutive numbers.
 */
LIBXSMM_API unsigned int libxsmm_get_tid(void);

/** Synchronize console output (lock). */
LIBXSMM_API void libxsmm_stdio_acquire(void);
/** Synchronize console output (unlock). */
LIBXSMM_API void libxsmm_stdio_release(void);

#endif /*LIBXSMM_SYNC_H*/
