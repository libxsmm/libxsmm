/******************************************************************************
** Copyright (c) 2014-2018, Intel Corporation                                **
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
#ifndef LIBXSMM_SYNC_H
#define LIBXSMM_SYNC_H

#include "libxsmm_intrinsics_x86.h"

#if !defined(LIBXSMM_TLS)
# if !defined(LIBXSMM_NO_SYNC) && !defined(LIBXSMM_NO_TLS)
#   if defined(__CYGWIN__) && defined(__clang__)
#     define LIBXSMM_NO_TLS
#     define LIBXSMM_TLS
#   else
#     if (defined(_WIN32) && !defined(__GNUC__)) || (defined(__PGI) && !defined(__cplusplus))
#       define LIBXSMM_TLS LIBXSMM_ATTRIBUTE(thread)
#     elif defined(__GNUC__) || defined(_CRAYC)
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

#if !defined(LIBXSMM_GCC_BASELINE) && defined(__GNUC__) && \
  LIBXSMM_VERSION3(4, 7, 0) <= LIBXSMM_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)
# define LIBXSMM_GCC_BASELINE
#endif

#if defined(__MIC__)
# define LIBXSMM_SYNC_PAUSE _mm_delay_32(8/*delay*/)
#elif !defined(LIBXSMM_INTRINSICS_NONE) && !defined(LIBXSMM_INTRINSICS_STATIC)
# define LIBXSMM_SYNC_PAUSE _mm_pause()
#elif defined(LIBXSMM_GCC_BASELINE) && !defined(__PGI)
# define LIBXSMM_SYNC_PAUSE __builtin_ia32_pause()
#else
# define LIBXSMM_SYNC_PAUSE
#endif

#if !defined(LIBXSMM_SYNC_SYSTEM) && (defined(__MINGW32__) || defined(__PGI))
# define LIBXSMM_SYNC_SYSTEM
#endif
#if !defined(LIBXSMM_ATOMIC_TRYLOCK_CMPSWP) && 0
# define LIBXSMM_ATOMIC_TRYLOCK_CMPSWP
#endif
#if !defined(LIBXSMM_ATOMIC_ZERO_STORE) && defined(_CRAYC)
# define LIBXSMM_ATOMIC_ZERO_STORE
#endif
#if defined(__ATOMIC_RELAXED)
# define LIBXSMM_ATOMIC_RELAXED __ATOMIC_RELAXED
#else
# define LIBXSMM_ATOMIC_RELAXED 0
#endif
#if defined(__ATOMIC_SEQ_CST)
# define LIBXSMM_ATOMIC_SEQ_CST __ATOMIC_SEQ_CST
#else
# define LIBXSMM_ATOMIC_SEQ_CST 0
#endif
#if !defined(LIBXSMM_ATOMIC_LOCKTYPE)
# define LIBXSMM_ATOMIC_LOCKTYPE char
#endif

#define LIBXSMM_NONATOMIC_LOCKTYPE LIBXSMM_ATOMIC_LOCKTYPE
#define LIBXSMM_NONATOMIC_LOAD(SRC_PTR, KIND) (*(SRC_PTR))
#define LIBXSMM_NONATOMIC_STORE(DST_PTR, VALUE, KIND) { LIBXSMM_UNUSED(KIND); *(DST_PTR) = VALUE; }
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

#if defined(LIBXSMM_NO_SYNC)
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
# define LIBXSMM_SYNC_CYCLE(COUNTER, NPAUSE)
# if !defined(LIBXSMM_SYNC_NPAUSE)
#   define LIBXSMM_SYNC_NPAUSE 0
# endif
#else
# if defined(__GNUC__) && !defined(LIBXSMM_SYNC_SYSTEM) && /* check for legacy GCC (no atomics) */ \
  LIBXSMM_VERSION3(4, 1, 0) <= LIBXSMM_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)
#   define LIBXSMM_ATOMIC(FN, BITS) FN
#   if defined(LIBXSMM_GCC_BASELINE)
#     define LIBXSMM_ATOMIC_LOAD(SRC_PTR, KIND) __atomic_load_n(SRC_PTR, KIND)
#     define LIBXSMM_ATOMIC_STORE(DST_PTR, VALUE, KIND) __atomic_store_n(DST_PTR, VALUE, KIND)
#     if !defined(LIBXSMM_ATOMIC_ZERO_STORE)
#       define LIBXSMM_ATOMIC_STORE_ZERO(DST_PTR, KIND) do {} while (__atomic_and_fetch(DST_PTR, 0, KIND))
#     endif
#     define LIBXSMM_ATOMIC_FETCH_OR(DST_PTR, VALUE, KIND) __atomic_fetch_or(DST_PTR, VALUE, KIND)
#     define LIBXSMM_ATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND) __atomic_add_fetch(DST_PTR, VALUE, KIND)
#     define LIBXSMM_ATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND) __atomic_sub_fetch(DST_PTR, VALUE, KIND)
#     define LIBXSMM_ATOMIC_FETCH_ADD(DST_PTR, VALUE, KIND) __atomic_fetch_add(DST_PTR, VALUE, KIND)
#     define LIBXSMM_ATOMIC_FETCH_SUB(DST_PTR, VALUE, KIND) __atomic_fetch_sub(DST_PTR, VALUE, KIND)
#     if 0 /* avoid to manually prevent the side-effect of the atomic when inside of a loop. */
#     define LIBXSMM_ATOMIC_CMPSWP(DST_PTR, OLDVAL, NEWVAL, KIND) __atomic_compare_exchange_n(DST_PTR, &(OLDVAL), NEWVAL, \
                                                                                0/*false*/, KIND, LIBXSMM_ATOMIC_RELAXED)
#     else /* GCC legacy atomics */
#     define LIBXSMM_ATOMIC_CMPSWP(DST_PTR, OLDVAL, NEWVAL, KIND) __sync_bool_compare_and_swap(DST_PTR, OLDVAL, NEWVAL)
#     endif
#     if defined(LIBXSMM_ATOMIC_TRYLOCK_CMPSWP)
#       define LIBXSMM_ATOMIC_TRYLOCK(DST_PTR, KIND) (!__atomic_test_and_set(DST_PTR, KIND))
#     endif
#     if defined(__PGI) /* ICE: __atomic_clear not implemented */
#       define LIBXSMM_ATOMIC_RELEASE(DST_PTR, KIND) { LIBXSMM_ASSERT_MSG(0 != *(DST_PTR), "LIBXSMM_ATOMIC_RELEASE"); \
                LIBXSMM_ATOMIC_STORE_ZERO(DST_PTR, KIND); }
#     else
#       define LIBXSMM_ATOMIC_RELEASE(DST_PTR, KIND) { LIBXSMM_ASSERT_MSG(0 != *(DST_PTR), "LIBXSMM_ATOMIC_RELEASE"); \
                __atomic_clear(DST_PTR, KIND); }
#     endif
#     if 0 /* __atomic_thread_fence: incorrect behavior in libxsmm_barrier (even with LIBXSMM_ATOMIC_SEQ_CST) */
#     define LIBXSMM_ATOMIC_SYNC(KIND) __atomic_thread_fence(KIND)
#     else
#     define LIBXSMM_ATOMIC_SYNC(KIND) __sync_synchronize()
#     endif
#   else /* GCC legacy atomics */
#     define LIBXSMM_ATOMIC_LOAD(SRC_PTR, KIND) __sync_or_and_fetch(SRC_PTR, 0)
#     define LIBXSMM_ATOMIC_STORE(DST_PTR, VALUE, KIND) { \
              LIBXSMM_ATOMIC_SYNC_NOFENCE(KIND); *(DST_PTR) = VALUE; \
              LIBXSMM_ATOMIC_SYNC_NOFENCE(KIND); }
#     if !defined(LIBXSMM_ATOMIC_ZERO_STORE)
#       define LIBXSMM_ATOMIC_STORE_ZERO(DST_PTR, KIND) do {} while (__sync_and_and_fetch(DST_PTR, 0))
#     endif
#     define LIBXSMM_ATOMIC_FETCH_OR(DST_PTR, VALUE, KIND) __sync_fetch_and_or(DST_PTR, VALUE)
#     define LIBXSMM_ATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND) __sync_add_and_fetch(DST_PTR, VALUE)
#     define LIBXSMM_ATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND) __sync_sub_and_fetch(DST_PTR, VALUE)
#     define LIBXSMM_ATOMIC_FETCH_ADD(DST_PTR, VALUE, KIND) __sync_fetch_and_add(DST_PTR, VALUE)
#     define LIBXSMM_ATOMIC_FETCH_SUB(DST_PTR, VALUE, KIND) __sync_fetch_and_sub(DST_PTR, VALUE)
#     define LIBXSMM_ATOMIC_CMPSWP(DST_PTR, OLDVAL, NEWVAL, KIND) __sync_bool_compare_and_swap(DST_PTR, OLDVAL, NEWVAL)
#     if defined(LIBXSMM_ATOMIC_TRYLOCK_CMPSWP)
#       define LIBXSMM_ATOMIC_TRYLOCK(DST_PTR, KIND) (0 == __sync_lock_test_and_set(DST_PTR, 1))
#     endif
#     define LIBXSMM_ATOMIC_RELEASE(DST_PTR, KIND) { LIBXSMM_ASSERT_MSG(0 != *(DST_PTR), "LIBXSMM_ATOMIC_RELEASE"); \
              __sync_lock_release(DST_PTR); }
#     define LIBXSMM_ATOMIC_SYNC(KIND) __sync_synchronize()
#   endif
#   if defined(LIBXSMM_ATOMIC_ZERO_STORE)
#     define LIBXSMM_ATOMIC_STORE_ZERO(DST_PTR, KIND) LIBXSMM_ATOMIC_STORE(DST_PTR, 0, KIND)
#   endif
#   if !defined(LIBXSMM_ATOMIC_TRYLOCK_CMPSWP)
#     define LIBXSMM_ATOMIC_TRYLOCK(DST_PTR, KIND) (0 == LIBXSMM_ATOMIC_FETCH_OR(DST_PTR, 1, KIND))
#   endif
#   define LIBXSMM_ATOMIC_ACQUIRE(DST_PTR, NPAUSE, KIND) { LIBXSMM_SYNC_CYCLE_DECL(libxsmm_atomic_acquire_counter_); \
            LIBXSMM_ASSERT(1 == sizeof(LIBXSMM_ATOMIC_LOCKTYPE)); LIBXSMM_ASSERT(0 == LIBXSMM_MOD2((uintptr_t)(DST_PTR), 4)); \
            while (!LIBXSMM_ATOMIC_TRYLOCK(DST_PTR, KIND)) LIBXSMM_SYNC_CYCLE(libxsmm_atomic_acquire_counter_, NPAUSE); \
            LIBXSMM_ASSERT_MSG(0 != *(DST_PTR), "LIBXSMM_ATOMIC_ACQUIRE"); }
#   define LIBXSMM_ATOMIC_SYNC_NOFENCE(KIND) __asm__ __volatile__ ("" ::: "memory")
#   if !defined(LIBXSMM_SYNC_NPAUSE)
#     define LIBXSMM_SYNC_NPAUSE 4096
#   endif
# elif defined(_WIN32) && !defined(LIBXSMM_SYNC_SYSTEM)
#   define LIBXSMM_ATOMIC(FN, BITS) LIBXSMM_CONCATENATE(LIBXSMM_ATOMIC, BITS)(FN)
#   define LIBXSMM_ATOMIC8(FN) LIBXSMM_CONCATENATE(FN, 8)
#   define LIBXSMM_ATOMIC16(FN) LIBXSMM_CONCATENATE(FN, 16)
#   define LIBXSMM_ATOMIC32(FN) FN/*default*/
#   define LIBXSMM_ATOMIC64(FN) LIBXSMM_CONCATENATE(FN, 64)
#   define LIBXSMM_ATOMIC_LOAD(SRC_PTR, KIND) InterlockedOr((volatile LONG*)(SRC_PTR), 0)
#   define LIBXSMM_ATOMIC_LOAD8(SRC_PTR, KIND) _InterlockedOr8((volatile char*)(SRC_PTR), 0)
#   define LIBXSMM_ATOMIC_LOAD64(SRC_PTR, KIND) InterlockedOr64((volatile LONGLONG*)(SRC_PTR), 0)
#   define LIBXSMM_ATOMIC_STORE(DST_PTR, VALUE, KIND) InterlockedExchange((volatile LONG*)(DST_PTR), (LONG)(VALUE))
#   define LIBXSMM_ATOMIC_STORE8(DST_PTR, VALUE, KIND) InterlockedExchange8((volatile char*)(DST_PTR), (LONGLONG)(VALUE))
#   define LIBXSMM_ATOMIC_STORE64(DST_PTR, VALUE, KIND) InterlockedExchange64((volatile LONGLONG*)(DST_PTR), (LONGLONG)(VALUE))
#   if defined(LIBXSMM_ATOMIC_ZERO_STORE)
#     define LIBXSMM_ATOMIC_STORE_ZERO(DST_PTR, KIND) LIBXSMM_ATOMIC_STORE(DST_PTR, 0, KIND)
#     define LIBXSMM_ATOMIC_STORE_ZERO8(DST_PTR, KIND) LIBXSMM_ATOMIC_STORE8(DST_PTR, 0, KIND)
#     define LIBXSMM_ATOMIC_STORE_ZERO64(DST_PTR, KIND) LIBXSMM_ATOMIC_STORE64(DST_PTR, 0, KIND)
#   else
#     define LIBXSMM_ATOMIC_STORE_ZERO(DST_PTR, KIND) InterlockedAnd((volatile LONG*)(DST_PTR), 0)
#     define LIBXSMM_ATOMIC_STORE_ZERO8(DST_PTR, KIND) InterlockedAnd8((volatile char*)(DST_PTR), 0)
#     define LIBXSMM_ATOMIC_STORE_ZERO64(DST_PTR, KIND) InterlockedAnd64((volatile LONGLONG*)(DST_PTR), 0)
#   endif
#   define LIBXSMM_ATOMIC_FETCH_OR(DST_PTR, VALUE, KIND) InterlockedOr((volatile LONG*)(DST_PTR), VALUE)
#   define LIBXSMM_ATOMIC_FETCH_OR8(DST_PTR, VALUE, KIND) _InterlockedOr8((volatile char*)(DST_PTR), VALUE)
#   define LIBXSMM_ATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND) (LIBXSMM_ATOMIC_FETCH_ADD(DST_PTR, VALUE, KIND) + (VALUE))
#   define LIBXSMM_ATOMIC_ADD_FETCH64(DST_PTR, VALUE, KIND) (LIBXSMM_ATOMIC_FETCH_ADD64(DST_PTR, VALUE, KIND) + (VALUE))
#   define LIBXSMM_ATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND) (LIBXSMM_ATOMIC_FETCH_SUB(DST_PTR, VALUE, KIND) - (VALUE))
#   define LIBXSMM_ATOMIC_SUB_FETCH64(DST_PTR, VALUE, KIND) (LIBXSMM_ATOMIC_FETCH_SUB64(DST_PTR, VALUE, KIND) - (VALUE))
#   define LIBXSMM_ATOMIC_FETCH_ADD(DST_PTR, VALUE, KIND) InterlockedExchangeAdd((volatile LONG*)(DST_PTR), VALUE)
#   define LIBXSMM_ATOMIC_FETCH_ADD64(DST_PTR, VALUE, KIND) InterlockedExchangeAdd64((volatile LONGLONG*)(DST_PTR), VALUE)
#   define LIBXSMM_ATOMIC_FETCH_SUB(DST_PTR, VALUE, KIND) LIBXSMM_ATOMIC_FETCH_ADD(DST_PTR, -1 * (VALUE), KIND)
#   define LIBXSMM_ATOMIC_CMPSWP(DST_PTR, OLDVAL, NEWVAL, KIND) (((LONG)(OLDVAL)) == InterlockedCompareExchange((volatile LONG*)(DST_PTR), NEWVAL, OLDVAL))
#   define LIBXSMM_ATOMIC_CMPSWP8(DST_PTR, OLDVAL, NEWVAL, KIND) ((OLDVAL) == _InterlockedCompareExchange8((volatile char*)(DST_PTR), NEWVAL, OLDVAL))
#   if defined(LIBXSMM_ATOMIC_TRYLOCK_CMPSWP)
#     define LIBXSMM_ATOMIC_TRYLOCK(DST_PTR, KIND) LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_CMPSWP, 8)(DST_PTR, 0, 1, KIND)
#   else
#     define LIBXSMM_ATOMIC_TRYLOCK(DST_PTR, KIND) (0 == LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_FETCH_OR, 8)(DST_PTR, 1, KIND))
#   endif
#   define LIBXSMM_ATOMIC_ACQUIRE(DST_PTR, NPAUSE, KIND) { LIBXSMM_SYNC_CYCLE_DECL(libxsmm_atomic_acquire_counter_); \
            LIBXSMM_ASSERT(1 == sizeof(LIBXSMM_ATOMIC_LOCKTYPE)); LIBXSMM_ASSERT(0 == LIBXSMM_MOD2((uintptr_t)(DST_PTR), 4)); \
            while (!LIBXSMM_ATOMIC_TRYLOCK(DST_PTR, KIND)) LIBXSMM_SYNC_CYCLE(libxsmm_atomic_acquire_counter_, NPAUSE); \
            LIBXSMM_ASSERT_MSG(0 != *(DST_PTR), "LIBXSMM_ATOMIC_ACQUIRE"); }
#   define LIBXSMM_ATOMIC_RELEASE(DST_PTR, KIND) { \
            LIBXSMM_ASSERT_MSG(0 != *(DST_PTR), "LIBXSMM_ATOMIC_RELEASE"); \
            LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_STORE_ZERO, 8)(DST_PTR, KIND); }
#   define LIBXSMM_ATOMIC_SYNC(KIND) _ReadWriteBarrier()
#   if !defined(LIBXSMM_SYNC_NPAUSE)
#     define LIBXSMM_SYNC_NPAUSE 4096
#   endif
# elif defined(LIBXSMM_SYNC_SYSTEM) /* fall-back */
#   define LIBXSMM_ATOMIC(FN, BITS) FN
#   define LIBXSMM_ATOMIC_LOAD LIBXSMM_NONATOMIC_LOAD
#   define LIBXSMM_ATOMIC_STORE LIBXSMM_NONATOMIC_STORE
#   define LIBXSMM_ATOMIC_STORE_ZERO LIBXSMM_NONATOMIC_STORE_ZERO
#   define LIBXSMM_ATOMIC_FETCH_OR LIBXSMM_NONATOMIC_FETCH_OR
#   define LIBXSMM_ATOMIC_ADD_FETCH LIBXSMM_NONATOMIC_ADD_FETCH
#   define LIBXSMM_ATOMIC_SUB_FETCH LIBXSMM_NONATOMIC_SUB_FETCH
#   define LIBXSMM_ATOMIC_FETCH_ADD LIBXSMM_NONATOMIC_FETCH_ADD
#   define LIBXSMM_ATOMIC_FETCH_SUB LIBXSMM_NONATOMIC_FETCH_SUB
#   define LIBXSMM_ATOMIC_CMPSWP LIBXSMM_NONATOMIC_CMPSWP
#   define LIBXSMM_ATOMIC_TRYLOCK LIBXSMM_NONATOMIC_TRYLOCK
#   define LIBXSMM_ATOMIC_ACQUIRE LIBXSMM_NONATOMIC_ACQUIRE
#   define LIBXSMM_ATOMIC_RELEASE LIBXSMM_NONATOMIC_RELEASE
#   define LIBXSMM_ATOMIC_SYNC LIBXSMM_NONATOMIC_SYNC
#   if !defined(LIBXSMM_SYNC_NPAUSE)
#     define LIBXSMM_SYNC_NPAUSE 0
#   endif
# else /* consider to enable LIBXSMM_SYNC_SYSTEM */
#   error LIBXSMM is missing atomic compiler builtins!
# endif
# if (0 < LIBXSMM_SYNC_NPAUSE)
#   define LIBXSMM_SYNC_CYCLE_ELSE(COUNTER, NPAUSE, ELSE) if (0 <= ((NPAUSE) - (++(COUNTER)))) { \
      LIBXSMM_SYNC_PAUSE; \
    } \
    else { \
      LIBXSMM_SYNC_YIELD(); ELSE \
    }
#   define LIBXSMM_SYNC_CYCLE_DECL(NAME) int NAME = 0
# else
#   define LIBXSMM_SYNC_CYCLE_ELSE(COUNTER, NPAUSE, ELSE) LIBXSMM_SYNC_PAUSE
#   define LIBXSMM_SYNC_CYCLE_DECL(NAME)
# endif
# define LIBXSMM_SYNC_CYCLE(COUNTER, NPAUSE) LIBXSMM_SYNC_CYCLE_ELSE(COUNTER, NPAUSE, ;)
#endif

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#if !defined(LIBXSMM_NO_SYNC) /** Default lock-kind */
# define LIBXSMM_LOCK_DEFAULT LIBXSMM_LOCK_SPINLOCK
# if !defined(LIBXSMM_LOCK_SYSTEM_SPINLOCK) && (defined(LIBXSMM_SYNC_SYSTEM) || 1)
#   define LIBXSMM_LOCK_SYSTEM_SPINLOCK
# endif
# if !defined(LIBXSMM_LOCK_SYSTEM_MUTEX) && (defined(LIBXSMM_SYNC_SYSTEM) || !defined(_MSC_VER))
#   define LIBXSMM_LOCK_SYSTEM_MUTEX
# endif
# if !defined(LIBXSMM_LOCK_SYSTEM_RWLOCK) && (defined(LIBXSMM_SYNC_SYSTEM) || 1)
#   define LIBXSMM_LOCK_SYSTEM_RWLOCK
# endif
  /* Lock type, initialization, destruction, (try-)lock, unlock, etc */
# define LIBXSMM_LOCK_ACQUIRED(KIND) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_ACQUIRED_, KIND)
# define LIBXSMM_LOCK_TYPE_ISPOD(KIND) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_TYPE_ISPOD_, KIND)
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
#   include <windows.h>
#   define LIBXSMM_LOCK_SPINLOCK spin
#   define LIBXSMM_LOCK_MUTEX mutex
#   define LIBXSMM_LOCK_RWLOCK rwlock
#   if !defined(LIBXSMM_LOCK_SYSTEM_SPINLOCK) || !(defined(_OPENMP) && defined(LIBXSMM_OMP))
#     define LIBXSMM_LOCK_ACQUIRED_spin TRUE
#   endif
#   if defined(LIBXSMM_LOCK_SYSTEM_SPINLOCK) && !(defined(_OPENMP) && defined(LIBXSMM_OMP))
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
#   if !defined(LIBXSMM_LOCK_SYSTEM_MUTEX) || !(defined(_OPENMP) && defined(LIBXSMM_OMP))
#     define LIBXSMM_LOCK_ACQUIRED_mutex WAIT_OBJECT_0
#   endif
#   if defined(LIBXSMM_LOCK_SYSTEM_MUTEX) && !(defined(_OPENMP) && defined(LIBXSMM_OMP))
#     define LIBXSMM_LOCK_TYPE_ISPOD_mutex 0
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
#   if !defined(LIBXSMM_LOCK_SYSTEM_RWLOCK) || !(defined(_OPENMP) && defined(LIBXSMM_OMP))
#     define LIBXSMM_LOCK_ACQUIRED_rwlock TRUE
#   endif
#   if defined(LIBXSMM_LOCK_SYSTEM_RWLOCK) && !(defined(_OPENMP) && defined(LIBXSMM_OMP))
#     define LIBXSMM_LOCK_TYPE_ISPOD_rwlock 1
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
#   define LIBXSMM_SYNC_YIELD YieldProcessor
# else
#   include <pthread.h>
#   if defined(__APPLE__) && defined(__MACH__)
#     define LIBXSMM_PTHREAD_FN(FN) LIBXSMM_CONCATENATE(FN, _np)
#   else
#     define LIBXSMM_PTHREAD_FN(FN) FN
#   endif
#   define LIBXSMM_SYNC_YIELD LIBXSMM_PTHREAD_FN(pthread_yield)
#   if defined(__APPLE__) && defined(__MACH__) && \
       defined(LIBXSMM_LOCK_SYSTEM_SPINLOCK) && \
     !(defined(_OPENMP) && defined(LIBXSMM_OMP))
#     define LIBXSMM_LOCK_SPINLOCK mutex
#   else
#     define LIBXSMM_LOCK_SPINLOCK spin
#   endif
#   define LIBXSMM_LOCK_MUTEX mutex
#   define LIBXSMM_LOCK_RWLOCK rwlock
#   if !defined(LIBXSMM_LOCK_SYSTEM_SPINLOCK) || !(defined(_OPENMP) && defined(LIBXSMM_OMP))
#     define LIBXSMM_LOCK_ACQUIRED_spin 0
#   endif
#   if defined(LIBXSMM_LOCK_SYSTEM_SPINLOCK) && !(defined(_OPENMP) && defined(LIBXSMM_OMP))
#     define LIBXSMM_LOCK_TYPE_ISPOD_spin 0
#     define LIBXSMM_LOCK_TYPE_spin pthread_spinlock_t
#     define LIBXSMM_LOCK_INIT_spin(LOCK, ATTR) LIBXSMM_EXPECT(0, pthread_spin_init(LOCK, *(ATTR)))
#     define LIBXSMM_LOCK_DESTROY_spin(LOCK) LIBXSMM_EXPECT(0, pthread_spin_destroy(LOCK))
#     define LIBXSMM_LOCK_TRYLOCK_spin(LOCK) pthread_spin_trylock(LOCK)
#     define LIBXSMM_LOCK_ACQUIRE_spin(LOCK) LIBXSMM_EXPECT(0, pthread_spin_lock(LOCK))
#     define LIBXSMM_LOCK_RELEASE_spin(LOCK) LIBXSMM_EXPECT(0, pthread_spin_unlock(LOCK))
#     define LIBXSMM_LOCK_TRYREAD_spin(LOCK) LIBXSMM_LOCK_TRYLOCK_spin(LOCK)
#     define LIBXSMM_LOCK_ACQREAD_spin(LOCK) LIBXSMM_LOCK_ACQUIRE_spin(LOCK)
#     define LIBXSMM_LOCK_RELREAD_spin(LOCK) LIBXSMM_LOCK_RELEASE_spin(LOCK)
#     define LIBXSMM_LOCK_ATTR_TYPE_spin int
#     define LIBXSMM_LOCK_ATTR_INIT_spin(ATTR) (*(ATTR) = 0)
#     define LIBXSMM_LOCK_ATTR_DESTROY_spin(ATTR) LIBXSMM_UNUSED(ATTR)
#   endif
#   if !defined(LIBXSMM_LOCK_SYSTEM_MUTEX) || !(defined(_OPENMP) && defined(LIBXSMM_OMP))
#     define LIBXSMM_LOCK_ACQUIRED_mutex 0
#   endif
#   if defined(LIBXSMM_LOCK_SYSTEM_MUTEX) && !(defined(_OPENMP) && defined(LIBXSMM_OMP))
#     define LIBXSMM_LOCK_TYPE_ISPOD_mutex 0
#     define LIBXSMM_LOCK_TYPE_mutex pthread_mutex_t
#     define LIBXSMM_LOCK_INIT_mutex(LOCK, ATTR) LIBXSMM_EXPECT(0, pthread_mutex_init(LOCK, ATTR))
#     define LIBXSMM_LOCK_DESTROY_mutex(LOCK) LIBXSMM_EXPECT(0, pthread_mutex_destroy(LOCK))
#     define LIBXSMM_LOCK_TRYLOCK_mutex(LOCK) pthread_mutex_trylock(LOCK)
#     define LIBXSMM_LOCK_ACQUIRE_mutex(LOCK) LIBXSMM_EXPECT(0, pthread_mutex_lock(LOCK))
#     define LIBXSMM_LOCK_RELEASE_mutex(LOCK) LIBXSMM_EXPECT(0, pthread_mutex_unlock(LOCK))
#     define LIBXSMM_LOCK_TRYREAD_mutex(LOCK) LIBXSMM_LOCK_TRYLOCK_mutex(LOCK)
#     define LIBXSMM_LOCK_ACQREAD_mutex(LOCK) LIBXSMM_LOCK_ACQUIRE_mutex(LOCK)
#     define LIBXSMM_LOCK_RELREAD_mutex(LOCK) LIBXSMM_LOCK_RELEASE_mutex(LOCK)
#     define LIBXSMM_LOCK_ATTR_TYPE_mutex pthread_mutexattr_t
#     if defined(NDEBUG)
#       define LIBXSMM_LOCK_ATTR_INIT_mutex(ATTR) (pthread_mutexattr_init(ATTR), \
                          pthread_mutexattr_settype(ATTR, PTHREAD_MUTEX_NORMAL))
#     else
#       define LIBXSMM_LOCK_ATTR_INIT_mutex(ATTR) (LIBXSMM_EXPECT(0, pthread_mutexattr_init(ATTR)), \
                      LIBXSMM_EXPECT(0, pthread_mutexattr_settype(ATTR, PTHREAD_MUTEX_ERRORCHECK)))
#     endif
#     define LIBXSMM_LOCK_ATTR_DESTROY_mutex(ATTR) LIBXSMM_EXPECT(0, pthread_mutexattr_destroy(ATTR))
#   endif
#   if !defined(LIBXSMM_LOCK_SYSTEM_RWLOCK) || !(defined(_OPENMP) && defined(LIBXSMM_OMP))
#     define LIBXSMM_LOCK_ACQUIRED_rwlock 0
#   endif
#   if defined(LIBXSMM_LOCK_SYSTEM_RWLOCK) && !(defined(_OPENMP) && defined(LIBXSMM_OMP))
#     define LIBXSMM_LOCK_TYPE_ISPOD_rwlock 0
#     define LIBXSMM_LOCK_TYPE_rwlock pthread_rwlock_t
#     define LIBXSMM_LOCK_INIT_rwlock(LOCK, ATTR) LIBXSMM_EXPECT(0, pthread_rwlock_init(LOCK, ATTR))
#     define LIBXSMM_LOCK_DESTROY_rwlock(LOCK) LIBXSMM_EXPECT(0, pthread_rwlock_destroy(LOCK))
#     define LIBXSMM_LOCK_TRYLOCK_rwlock(LOCK) pthread_rwlock_trywrlock(LOCK)
#     define LIBXSMM_LOCK_ACQUIRE_rwlock(LOCK) LIBXSMM_EXPECT(0, pthread_rwlock_wrlock(LOCK))
#     define LIBXSMM_LOCK_RELEASE_rwlock(LOCK) LIBXSMM_EXPECT(0, pthread_rwlock_unlock(LOCK))
#     define LIBXSMM_LOCK_TRYREAD_rwlock(LOCK) pthread_rwlock_tryrdlock(LOCK)
#     define LIBXSMM_LOCK_ACQREAD_rwlock(LOCK) LIBXSMM_EXPECT(0, pthread_rwlock_rdlock(LOCK))
#     define LIBXSMM_LOCK_RELREAD_rwlock(LOCK) LIBXSMM_LOCK_RELEASE_rwlock(LOCK)
#     define LIBXSMM_LOCK_ATTR_TYPE_rwlock pthread_rwlockattr_t
#     define LIBXSMM_LOCK_ATTR_INIT_rwlock(ATTR) LIBXSMM_EXPECT(0, pthread_rwlockattr_init(ATTR))
#     define LIBXSMM_LOCK_ATTR_DESTROY_rwlock(ATTR) LIBXSMM_EXPECT(0, pthread_rwlockattr_destroy(ATTR))
#   endif
# endif
/* OpenMP based locks need to stay disabled unless both
 * libxsmm and libxsmmext are built with OpenMP support.
 */
# if defined(_OPENMP) && defined(LIBXSMM_OMP)
#   include <omp.h>
# endif
# if !defined(LIBXSMM_LOCK_SYSTEM_SPINLOCK)
#   if 1 /* directly based on atomic primitives */
#     define LIBXSMM_LOCK_TYPE_ISPOD_spin 1
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
#   else /* rely on LIBXSMM's portable locks */
#     define LIBXSMM_LOCK_TYPE_ISPOD_spin 0
#     define LIBXSMM_LOCK_TYPE_spin libxsmm_spinlock*
#     define LIBXSMM_LOCK_INIT_spin(LOCK, ATTR) { LIBXSMM_UNUSED(ATTR); (*(LOCK) = libxsmm_spinlock_create()); }
#     define LIBXSMM_LOCK_DESTROY_spin(LOCK) libxsmm_spinlock_destroy(*(LOCK))
#     define LIBXSMM_LOCK_TRYLOCK_spin(LOCK) libxsmm_spinlock_trylock(*(LOCK))
#     define LIBXSMM_LOCK_ACQUIRE_spin(LOCK) libxsmm_spinlock_acquire(*(LOCK))
#     define LIBXSMM_LOCK_RELEASE_spin(LOCK) libxsmm_spinlock_release(*(LOCK))
#     define LIBXSMM_LOCK_TRYREAD_spin(LOCK) LIBXSMM_LOCK_TRYLOCK_spin(LOCK)
#     define LIBXSMM_LOCK_ACQREAD_spin(LOCK) LIBXSMM_LOCK_ACQUIRE_spin(LOCK)
#     define LIBXSMM_LOCK_RELREAD_spin(LOCK) LIBXSMM_LOCK_RELEASE_spin(LOCK)
#     define LIBXSMM_LOCK_ATTR_TYPE_spin int
#     define LIBXSMM_LOCK_ATTR_INIT_spin(ATTR) LIBXSMM_UNUSED(ATTR)
#     define LIBXSMM_LOCK_ATTR_DESTROY_spin(ATTR) LIBXSMM_UNUSED(ATTR)
#   endif
# elif defined(_OPENMP) && defined(LIBXSMM_OMP)
#   define LIBXSMM_LOCK_ACQUIRED_spin 1
#   define LIBXSMM_LOCK_TYPE_ISPOD_spin 0
#   define LIBXSMM_LOCK_TYPE_spin omp_lock_t
#   define LIBXSMM_LOCK_INIT_spin(LOCK, ATTR) { LIBXSMM_UNUSED(ATTR); omp_init_lock(LOCK); }
#   define LIBXSMM_LOCK_DESTROY_spin(LOCK) omp_destroy_lock(LOCK)
#   define LIBXSMM_LOCK_TRYLOCK_spin(LOCK) omp_test_lock(LOCK)
#   define LIBXSMM_LOCK_ACQUIRE_spin(LOCK) omp_set_lock(LOCK)
#   define LIBXSMM_LOCK_RELEASE_spin(LOCK) omp_unset_lock(LOCK)
#   define LIBXSMM_LOCK_TRYREAD_spin(LOCK) LIBXSMM_LOCK_TRYLOCK_spin(LOCK)
#   define LIBXSMM_LOCK_ACQREAD_spin(LOCK) LIBXSMM_LOCK_ACQUIRE_spin(LOCK)
#   define LIBXSMM_LOCK_RELREAD_spin(LOCK) LIBXSMM_LOCK_RELEASE_spin(LOCK)
#   define LIBXSMM_LOCK_ATTR_TYPE_spin const void*
#   define LIBXSMM_LOCK_ATTR_INIT_spin(ATTR) LIBXSMM_UNUSED(ATTR)
#   define LIBXSMM_LOCK_ATTR_DESTROY_spin(ATTR) LIBXSMM_UNUSED(ATTR)
# endif
# if !defined(LIBXSMM_LOCK_SYSTEM_MUTEX)
#   define LIBXSMM_LOCK_TYPE_ISPOD_mutex 0
#   define LIBXSMM_LOCK_TYPE_mutex libxsmm_mutex*
#   define LIBXSMM_LOCK_INIT_mutex(LOCK, ATTR) { LIBXSMM_UNUSED(ATTR); (*(LOCK) = libxsmm_mutex_create()); }
#   define LIBXSMM_LOCK_DESTROY_mutex(LOCK) libxsmm_mutex_destroy(*(LOCK))
#   define LIBXSMM_LOCK_TRYLOCK_mutex(LOCK) libxsmm_mutex_trylock(*(LOCK))
#   define LIBXSMM_LOCK_ACQUIRE_mutex(LOCK) libxsmm_mutex_acquire(*(LOCK))
#   define LIBXSMM_LOCK_RELEASE_mutex(LOCK) libxsmm_mutex_release(*(LOCK))
#   define LIBXSMM_LOCK_TRYREAD_mutex(LOCK) LIBXSMM_LOCK_TRYLOCK_mutex(LOCK)
#   define LIBXSMM_LOCK_ACQREAD_mutex(LOCK) LIBXSMM_LOCK_ACQUIRE_mutex(LOCK)
#   define LIBXSMM_LOCK_RELREAD_mutex(LOCK) LIBXSMM_LOCK_RELEASE_mutex(LOCK)
#   define LIBXSMM_LOCK_ATTR_TYPE_mutex int
#   define LIBXSMM_LOCK_ATTR_INIT_mutex(ATTR) LIBXSMM_UNUSED(ATTR)
#   define LIBXSMM_LOCK_ATTR_DESTROY_mutex(ATTR) LIBXSMM_UNUSED(ATTR)
# elif defined(_OPENMP) && defined(LIBXSMM_OMP)
#   define LIBXSMM_LOCK_ACQUIRED_mutex 1
#   define LIBXSMM_LOCK_TYPE_ISPOD_mutex 0
#   define LIBXSMM_LOCK_TYPE_mutex omp_lock_t
#   define LIBXSMM_LOCK_INIT_mutex(LOCK, ATTR) { LIBXSMM_UNUSED(ATTR); omp_init_lock(LOCK); }
#   define LIBXSMM_LOCK_DESTROY_mutex(LOCK) omp_destroy_lock(LOCK)
#   define LIBXSMM_LOCK_TRYLOCK_mutex(LOCK) omp_test_lock(LOCK)
#   define LIBXSMM_LOCK_ACQUIRE_mutex(LOCK) omp_set_lock(LOCK)
#   define LIBXSMM_LOCK_RELEASE_mutex(LOCK) omp_unset_lock(LOCK)
#   define LIBXSMM_LOCK_TRYREAD_mutex(LOCK) LIBXSMM_LOCK_TRYLOCK_mutex(LOCK)
#   define LIBXSMM_LOCK_ACQREAD_mutex(LOCK) LIBXSMM_LOCK_ACQUIRE_mutex(LOCK)
#   define LIBXSMM_LOCK_RELREAD_mutex(LOCK) LIBXSMM_LOCK_RELEASE_mutex(LOCK)
#   define LIBXSMM_LOCK_ATTR_TYPE_mutex const void*
#   define LIBXSMM_LOCK_ATTR_INIT_mutex(ATTR) LIBXSMM_UNUSED(ATTR)
#   define LIBXSMM_LOCK_ATTR_DESTROY_mutex(ATTR) LIBXSMM_UNUSED(ATTR)
# endif
# if !defined(LIBXSMM_LOCK_SYSTEM_RWLOCK)
#   define LIBXSMM_LOCK_TYPE_ISPOD_rwlock 0
#   define LIBXSMM_LOCK_TYPE_rwlock libxsmm_rwlock*
#   define LIBXSMM_LOCK_INIT_rwlock(LOCK, ATTR) { LIBXSMM_UNUSED(ATTR); (*(LOCK) = libxsmm_rwlock_create()); }
#   define LIBXSMM_LOCK_DESTROY_rwlock(LOCK) libxsmm_rwlock_destroy(*(LOCK))
#   define LIBXSMM_LOCK_TRYLOCK_rwlock(LOCK) libxsmm_rwlock_trylock(*(LOCK))
#   define LIBXSMM_LOCK_ACQUIRE_rwlock(LOCK) libxsmm_rwlock_acquire(*(LOCK))
#   define LIBXSMM_LOCK_RELEASE_rwlock(LOCK) libxsmm_rwlock_release(*(LOCK))
#   define LIBXSMM_LOCK_TRYREAD_rwlock(LOCK) libxsmm_rwlock_tryread(*(LOCK))
#   define LIBXSMM_LOCK_ACQREAD_rwlock(LOCK) libxsmm_rwlock_acqread(*(LOCK))
#   define LIBXSMM_LOCK_RELREAD_rwlock(LOCK) libxsmm_rwlock_relread(*(LOCK))
#   define LIBXSMM_LOCK_ATTR_TYPE_rwlock int
#   define LIBXSMM_LOCK_ATTR_INIT_rwlock(ATTR) LIBXSMM_UNUSED(ATTR)
#   define LIBXSMM_LOCK_ATTR_DESTROY_rwlock(ATTR) LIBXSMM_UNUSED(ATTR)
# elif defined(_OPENMP) && defined(LIBXSMM_OMP)
#   define LIBXSMM_LOCK_ACQUIRED_rwlock 1
#   define LIBXSMM_LOCK_TYPE_ISPOD_rwlock 0
#   define LIBXSMM_LOCK_TYPE_rwlock omp_lock_t
#   define LIBXSMM_LOCK_INIT_rwlock(LOCK, ATTR) { LIBXSMM_UNUSED(ATTR); omp_init_lock(LOCK); }
#   define LIBXSMM_LOCK_DESTROY_rwlock(LOCK) omp_destroy_lock(LOCK)
#   define LIBXSMM_LOCK_TRYLOCK_rwlock(LOCK) omp_test_lock(LOCK)
#   define LIBXSMM_LOCK_ACQUIRE_rwlock(LOCK) omp_set_lock(LOCK)
#   define LIBXSMM_LOCK_RELEASE_rwlock(LOCK) omp_unset_lock(LOCK)
#   define LIBXSMM_LOCK_TRYREAD_rwlock(LOCK) LIBXSMM_LOCK_TRYLOCK_rwlock(LOCK)
#   define LIBXSMM_LOCK_ACQREAD_rwlock(LOCK) LIBXSMM_LOCK_ACQUIRE_rwlock(LOCK)
#   define LIBXSMM_LOCK_RELREAD_rwlock(LOCK) LIBXSMM_LOCK_RELEASE_rwlock(LOCK)
#   define LIBXSMM_LOCK_ATTR_TYPE_rwlock const void*
#   define LIBXSMM_LOCK_ATTR_INIT_rwlock(ATTR) LIBXSMM_UNUSED(ATTR)
#   define LIBXSMM_LOCK_ATTR_DESTROY_rwlock(ATTR) LIBXSMM_UNUSED(ATTR)
# endif
#else
# define LIBXSMM_LOCK_SPINLOCK
# define LIBXSMM_LOCK_MUTEX
# define LIBXSMM_LOCK_RWLOCK
# define LIBXSMM_LOCK_ACQUIRED(KIND) 0
# define LIBXSMM_LOCK_ATTR_TYPE(KIND) int
# define LIBXSMM_LOCK_ATTR_INIT(KIND, ATTR) LIBXSMM_UNUSED(ATTR)
# define LIBXSMM_LOCK_ATTR_DESTROY(KIND, ATTR) LIBXSMM_UNUSED(ATTR)
# define LIBXSMM_LOCK_TYPE(KIND) int
# define LIBXSMM_LOCK_INIT(KIND, LOCK, ATTR) { LIBXSMM_UNUSED(LOCK); LIBXSMM_UNUSED(ATTR); }
# define LIBXSMM_LOCK_DESTROY(KIND, LOCK) LIBXSMM_UNUSED(LOCK)
# define LIBXSMM_LOCK_TRYLOCK(KIND, LOCK) LIBXSMM_LOCK_ACQUIRED(KIND)
# define LIBXSMM_LOCK_ACQUIRE(KIND, LOCK) LIBXSMM_UNUSED(LOCK)
# define LIBXSMM_LOCK_RELEASE(KIND, LOCK) LIBXSMM_UNUSED(LOCK)
# define LIBXSMM_LOCK_TRYREAD(KIND, LOCK) LIBXSMM_LOCK_TRYLOCK(KIND, LOCK)
# define LIBXSMM_LOCK_ACQREAD(KIND, LOCK) LIBXSMM_LOCK_ACQUIRE(KIND, LOCK)
# define LIBXSMM_LOCK_RELREAD(KIND, LOCK) LIBXSMM_LOCK_RELEASE(KIND, LOCK)
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


/** Opaque type which represents a barrier. */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_barrier libxsmm_barrier;

/** Create barrier from one of the threads. */
LIBXSMM_API libxsmm_barrier* libxsmm_barrier_create(int ncores, int nthreads_per_core);
/** Initialize the barrier from each thread of the team. */
LIBXSMM_API void libxsmm_barrier_init(libxsmm_barrier* barrier, int tid);
/** Wait for the entire team to arrive. */
LIBXSMM_API void libxsmm_barrier_wait(libxsmm_barrier* barrier, int tid);
/** Destroy the resources associated with this barrier. */
LIBXSMM_API void libxsmm_barrier_destroy(const libxsmm_barrier* barrier);
/** DEPRECATED: use libxsmm_barrier_destroy instead. */
#define libxsmm_barrier_release libxsmm_barrier_destroy

/** Spin-lock, which eventually differs from LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK_SPINLOCK). */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_spinlock libxsmm_spinlock;
LIBXSMM_API libxsmm_spinlock* libxsmm_spinlock_create(void);
LIBXSMM_API void libxsmm_spinlock_destroy(const libxsmm_spinlock* spinlock);
LIBXSMM_API int libxsmm_spinlock_trylock(libxsmm_spinlock* spinlock);
LIBXSMM_API void libxsmm_spinlock_acquire(libxsmm_spinlock* spinlock);
LIBXSMM_API void libxsmm_spinlock_release(libxsmm_spinlock* spinlock);

/** Mutual-exclusive lock (Mutex), which eventually differs from LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK_MUTEX). */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_mutex libxsmm_mutex;
LIBXSMM_API libxsmm_mutex* libxsmm_mutex_create(void);
LIBXSMM_API void libxsmm_mutex_destroy(const libxsmm_mutex* mutex);
LIBXSMM_API int libxsmm_mutex_trylock(libxsmm_mutex* mutex);
LIBXSMM_API void libxsmm_mutex_acquire(libxsmm_mutex* mutex);
LIBXSMM_API void libxsmm_mutex_release(libxsmm_mutex* mutex);

/** Reader-Writer lock (RW-lock), which eventually differs from LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK_RWLOCK). */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_rwlock libxsmm_rwlock;
LIBXSMM_API libxsmm_rwlock* libxsmm_rwlock_create(void);
LIBXSMM_API void libxsmm_rwlock_destroy(const libxsmm_rwlock* rwlock);
LIBXSMM_API int libxsmm_rwlock_trylock(libxsmm_rwlock* rwlock);
LIBXSMM_API void libxsmm_rwlock_acquire(libxsmm_rwlock* rwlock);
LIBXSMM_API void libxsmm_rwlock_release(libxsmm_rwlock* rwlock);
LIBXSMM_API int libxsmm_rwlock_tryread(libxsmm_rwlock* rwlock);
LIBXSMM_API void libxsmm_rwlock_acqread(libxsmm_rwlock* rwlock);
LIBXSMM_API void libxsmm_rwlock_relread(libxsmm_rwlock* rwlock);

/** Utility function to receive the process ID of the calling process. */
LIBXSMM_API unsigned int libxsmm_get_pid(void);
/**
 * Utility function to receive a Thread-ID (TID) for the calling thread.
 * The TID is not related to a specific threading runtime. TID=0 may not
 * represent the main thread. TIDs are zero-based and consecutive numbers.
 */
LIBXSMM_API unsigned int libxsmm_get_tid(void);

#endif /*LIBXSMM_SYNC_H*/

