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
#     if (defined(_WIN32) && !defined(__GNUC__)) || defined(__PGI)
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

#if defined(__MIC__)
# define LIBXSMM_SYNC_PAUSE _mm_delay_32(8/*delay*/)
#elif !defined(LIBXSMM_INTRINSICS_NONE) && !defined(LIBXSMM_INTRINSICS_LEGACY)
# define LIBXSMM_SYNC_PAUSE _mm_pause()
#else
# define LIBXSMM_SYNC_PAUSE LIBXSMM_FLOCK(stdout); LIBXSMM_FUNLOCK(stdout)
#endif

#if defined(__GNUC__)
# if !defined(LIBXSMM_GCCATOMICS)
    /* note: the following version check does *not* prevent non-GNU compilers to adopt GCC's atomics */
#   if LIBXSMM_VERSION3(4, 7, 4) <= LIBXSMM_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)
#     define LIBXSMM_GCCATOMICS 1
#   else
#     define LIBXSMM_GCCATOMICS 0
#   endif
# endif
#endif

#define LIBXSMM_ATOMIC_RELAXED __ATOMIC_RELAXED
#define LIBXSMM_ATOMIC_SEQ_CST __ATOMIC_SEQ_CST

#define LIBXSMM_NONATOMIC_LOAD(SRC_PTR, KIND) (*(SRC_PTR))
#define LIBXSMM_NONATOMIC_STORE(DST_PTR, VALUE, KIND) (*(DST_PTR) = VALUE)
#define LIBXSMM_NONATOMIC_STORE_ZERO(DST_PTR, KIND) LIBXSMM_NONATOMIC_STORE(DST_PTR, 0, KIND)
#define LIBXSMM_NONATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND) (*(DST_PTR) += VALUE)
#define LIBXSMM_NONATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND) (*(DST_PTR) -= VALUE)
#define LIBXSMM_NONATOMIC_FETCH_ADD(DST_PTR, VALUE, KIND) (LIBXSMM_NONATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND), (*(DST_PTR) - (VALUE)))
#define LIBXSMM_NONATOMIC_FETCH_SUB(DST_PTR, VALUE, KIND) (LIBXSMM_NONATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND), (*(DST_PTR) + (VALUE)))
#define LIBXSMM_NONATOMIC_CMPSWP(DST_PTR, OLDVAL, NEWVAL, KIND) ((NEWVAL) == (*(DST_PTR) == (OLDVAL) ? (*(DST_PTR) = (NEWVAL)) : (OLDVAL)))
#define LIBXSMM_NONATOMIC_SET(DST_PTR, VALUE) (*(DST_PTR) = (VALUE))

#if !defined(LIBXSMM_NO_SYNC) && defined(LIBXSMM_GCCATOMICS)
# if (0 != LIBXSMM_GCCATOMICS)
#   define LIBXSMM_ATOMIC_LOAD(SRC_PTR, KIND) __atomic_load_n(SRC_PTR, KIND)
#   define LIBXSMM_ATOMIC_STORE(DST_PTR, VALUE, KIND) __atomic_store_n(DST_PTR, VALUE, KIND)
#   define LIBXSMM_ATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND) /**(DST_PTR) =*/ __atomic_add_fetch(DST_PTR, VALUE, KIND)
#   define LIBXSMM_ATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND) /**(DST_PTR) =*/ __atomic_sub_fetch(DST_PTR, VALUE, KIND)
#   define LIBXSMM_ATOMIC_FETCH_ADD(DST_PTR, VALUE, KIND) /**(DST_PTR) =*/ __atomic_fetch_add(DST_PTR, VALUE, KIND)
#   define LIBXSMM_ATOMIC_FETCH_SUB(DST_PTR, VALUE, KIND) /**(DST_PTR) =*/ __atomic_fetch_sub(DST_PTR, VALUE, KIND)
#   define LIBXSMM_ATOMIC_CMPSWP(DST_PTR, OLDVAL, NEWVAL, KIND) __atomic_compare_exchange_n(DST_PTR, &(OLDVAL), NEWVAL, \
                                                                              0/*false*/, KIND, LIBXSMM_ATOMIC_RELAXED)
# else
#   define LIBXSMM_ATOMIC_LOAD(SRC_PTR, KIND) __sync_or_and_fetch(SRC_PTR, 0)
#   define LIBXSMM_ATOMIC_STORE(DST_PTR, VALUE, KIND) while (*(DST_PTR) != (VALUE)) \
      if (0/*false*/ != __sync_bool_compare_and_swap(DST_PTR, *(DST_PTR), VALUE)) break
    /* use store side-effect of built-in (dummy assignment to mute warning) */
#   if 0 /* disabled as it appears to hang on some systems; fall-back is below */
#   define LIBXSMM_ATOMIC_STORE_ZERO(DST_PTR, KIND) { \
      const int libxsmm_store_zero_ = (0 != __sync_and_and_fetch(DST_PTR, 0)) ? 1 : 0; \
      LIBXSMM_UNUSED(libxsmm_store_zero_); \
    }
#   endif
#   define LIBXSMM_ATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND) /**(DST_PTR) = */__sync_add_and_fetch(DST_PTR, VALUE)
#   define LIBXSMM_ATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND) /**(DST_PTR) = */__sync_sub_and_fetch(DST_PTR, VALUE)
#   define LIBXSMM_ATOMIC_FETCH_ADD(DST_PTR, VALUE, KIND) /**(DST_PTR) = */__sync_fetch_and_add(DST_PTR, VALUE)
#   define LIBXSMM_ATOMIC_FETCH_SUB(DST_PTR, VALUE, KIND) /**(DST_PTR) = */__sync_fetch_and_sub(DST_PTR, VALUE)
#   define LIBXSMM_ATOMIC_CMPSWP(DST_PTR, OLDVAL, NEWVAL, KIND) __sync_bool_compare_and_swap(DST_PTR, OLDVAL, NEWVAL)
# endif
# define LIBXSMM_SYNC_BARRIER __asm__ __volatile__ ("" ::: "memory")
# define LIBXSMM_SYNCHRONIZE __sync_synchronize()
    /* TODO: distinct implementation of LIBXSMM_ATOMIC_SYNC_* wrt LIBXSMM_GCCATOMICS */
# define LIBXSMM_ATOMIC_SYNC_CHECK(LOCK, VALUE) while ((VALUE) == (LOCK)); LIBXSMM_SYNC_PAUSE
# define LIBXSMM_ATOMIC_SYNC_SET(LOCK) do { LIBXSMM_ATOMIC_SYNC_CHECK(LOCK, 1); } while(0 != __sync_lock_test_and_set(&(LOCK), 1))
# define LIBXSMM_ATOMIC_SYNC_UNSET(LOCK) __sync_lock_release(&(LOCK))
#elif !defined(LIBXSMM_NO_SYNC) && defined(_WIN32) /* TODO: atomics */
# define LIBXSMM_ATOMIC_LOAD(SRC_PTR, KIND) (*((SRC_PTR) /*+ InterlockedOr((LONG volatile*)(SRC_PTR), 0) * 0*/))
# define LIBXSMM_ATOMIC_STORE(DST_PTR, VALUE, KIND) (*(DST_PTR) = VALUE)
# define LIBXSMM_ATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND) (*(DST_PTR) += VALUE)
# define LIBXSMM_ATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND) (*(DST_PTR) -= VALUE)
# define LIBXSMM_ATOMIC_FETCH_ADD(DST_PTR, VALUE, KIND) *(DST_PTR); LIBXSMM_ATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND)
# define LIBXSMM_ATOMIC_FETCH_SUB(DST_PTR, VALUE, KIND) *(DST_PTR); LIBXSMM_ATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND)
# define LIBXSMM_ATOMIC_CMPSWP(DST_PTR, OLDVAL, NEWVAL, KIND) (((LONG)(NEWVAL)) == InterlockedCompareExchange((volatile LONG*)(DST_PTR), NEWVAL, OLDVAL))
# define LIBXSMM_ATOMIC_SYNC_CHECK(LOCK, VALUE) while ((VALUE) == (LOCK)); LIBXSMM_SYNC_PAUSE
# define LIBXSMM_ATOMIC_SYNC_SET(LOCK) { int libxsmm_sync_set_i_; \
    do { LIBXSMM_ATOMIC_SYNC_CHECK(LOCK, 1); \
      libxsmm_sync_set_i_ = LOCK; LOCK = 1; \
    } while(0 != libxsmm_sync_set_i_); \
  }
# define LIBXSMM_ATOMIC_SYNC_UNSET(LOCK) (LOCK) = 0
# define LIBXSMM_SYNC_BARRIER _ReadWriteBarrier()
# define LIBXSMM_SYNCHRONIZE /* TODO */
#else
# define LIBXSMM_ATOMIC_LOAD LIBXSMM_NONATOMIC_LOAD
# define LIBXSMM_ATOMIC_STORE LIBXSMM_NONATOMIC_STORE
# define LIBXSMM_ATOMIC_ADD_FETCH LIBXSMM_NONATOMIC_ADD_FETCH
# define LIBXSMM_ATOMIC_SUB_FETCH LIBXSMM_NONATOMIC_SUB_FETCH
# define LIBXSMM_ATOMIC_FETCH_ADD LIBXSMM_NONATOMIC_FETCH_ADD
# define LIBXSMM_ATOMIC_FETCH_SUB LIBXSMM_NONATOMIC_FETCH_SUB
# define LIBXSMM_ATOMIC_CMPSWP LIBXSMM_NONATOMIC_CMPSWP
# define LIBXSMM_ATOMIC_SYNC_CHECK(LOCK, VALUE) LIBXSMM_UNUSED(LOCK)
# define LIBXSMM_ATOMIC_SYNC_SET(LOCK) LIBXSMM_UNUSED(LOCK)
# define LIBXSMM_ATOMIC_SYNC_UNSET(LOCK) LIBXSMM_UNUSED(LOCK)
# define LIBXSMM_SYNC_BARRIER
# define LIBXSMM_SYNCHRONIZE
#endif
#if !defined(LIBXSMM_ATOMIC_STORE_ZERO)
# define LIBXSMM_ATOMIC_STORE_ZERO(DST_PTR, KIND) LIBXSMM_ATOMIC_STORE(DST_PTR, 0, KIND)
#endif
#if !defined(LIBXSMM_ATOMIC_SET) /* TODO */
# define LIBXSMM_ATOMIC_SET(DST_PTR, VALUE) LIBXSMM_NONATOMIC_SET(DST_PTR, VALUE)
#endif

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#if !defined(LIBXSMM_NO_SYNC)
  /** Default lock-kind */
# define LIBXSMM_LOCK_DEFAULT LIBXSMM_LOCK_SPINLOCK
# if !defined(LIBXSMM_LOCK_SYSTEM)
#   define LIBXSMM_LOCK_SYSTEM
# endif
  /* OpenMP based locks need to stay disabled unless both
   * libxsmm and libxsmmext are built with OpenMP support.
   */
# if defined(_OPENMP) && defined(LIBXSMM_OMP)
#   include <omp.h>
#   define LIBXSMM_LOCK_SPINLOCK
#   define LIBXSMM_LOCK_MUTEX
#   define LIBXSMM_LOCK_RWLOCK
#   define LIBXSMM_LOCK_ACQUIRED(KIND) 1
#   define LIBXSMM_LOCK_ATTR_TYPE(KIND) const void*
#   define LIBXSMM_LOCK_ATTR_INIT(KIND, ATTR) LIBXSMM_UNUSED(ATTR)
#   define LIBXSMM_LOCK_ATTR_DESTROY(KIND, ATTR) LIBXSMM_UNUSED(ATTR)
#   define LIBXSMM_LOCK_TYPE(KIND) omp_lock_t
#   define LIBXSMM_LOCK_INIT(KIND, LOCK, ATTR) omp_init_lock(LOCK)
#   define LIBXSMM_LOCK_DESTROY(KIND, LOCK) omp_destroy_lock(LOCK)
#   define LIBXSMM_LOCK_TRYLOCK(KIND, LOCK) omp_test_lock(LOCK)
#   define LIBXSMM_LOCK_ACQUIRE(KIND, LOCK) omp_set_lock(LOCK)
#   define LIBXSMM_LOCK_RELEASE(KIND, LOCK) omp_unset_lock(LOCK)
#   define LIBXSMM_LOCK_TRYREAD(KIND, LOCK) LIBXSMM_LOCK_TRYLOCK(KIND, LOCK)
#   define LIBXSMM_LOCK_ACQREAD(KIND, LOCK) LIBXSMM_LOCK_ACQUIRE(KIND, LOCK)
#   define LIBXSMM_LOCK_RELREAD(KIND, LOCK) LIBXSMM_LOCK_RELEASE(KIND, LOCK)
# else
    /* Lock type, initialization, destruction, (try-)lock, unlock, etc */
#   define LIBXSMM_LOCK_TYPE(KIND) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_TYPE_, KIND)
#   define LIBXSMM_LOCK_INIT(KIND, LOCK, ATTR) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_INIT_, KIND)(LOCK, ATTR)
#   define LIBXSMM_LOCK_DESTROY(KIND, LOCK) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_DESTROY_, KIND)(LOCK)
#   define LIBXSMM_LOCK_TRYLOCK(KIND, LOCK) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_TRYLOCK_, KIND)(LOCK)
#   define LIBXSMM_LOCK_ACQUIRE(KIND, LOCK) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_ACQUIRE_, KIND)(LOCK)
#   define LIBXSMM_LOCK_RELEASE(KIND, LOCK) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_RELEASE_, KIND)(LOCK)
#   define LIBXSMM_LOCK_TRYREAD(KIND, LOCK) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_TRYREAD_, KIND)(LOCK)
#   define LIBXSMM_LOCK_ACQREAD(KIND, LOCK) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_ACQREAD_, KIND)(LOCK)
#   define LIBXSMM_LOCK_RELREAD(KIND, LOCK) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_RELREAD_, KIND)(LOCK)
    /* Attribute type, initialization, destruction */
#   define LIBXSMM_LOCK_ATTR_TYPE(KIND) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_ATTR_TYPE_, KIND)
#   define LIBXSMM_LOCK_ATTR_INIT(KIND, ATTR) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_ATTR_INIT_, KIND)(ATTR)
#   define LIBXSMM_LOCK_ATTR_DESTROY(KIND, ATTR) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_ATTR_DESTROY_, KIND)(ATTR)
    /* implementation */
#   if defined(_WIN32) \
    /* Cygwin's Pthread implementation appears to be broken; use Win32 */ \
    || defined(__CYGWIN__)
#     if defined(__CYGWIN__) /* hack: make SRW-locks available */
#       if defined(_WIN32_WINNT)
#         define LIBXSMM_WIN32_WINNT _WIN32_WINNT
#         undef _WIN32_WINNT
#         define _WIN32_WINNT (LIBXSMM_WIN32_WINNT | 0x0600)
#       else
#         define _WIN32_WINNT 0x0600
#       endif
#     endif
#     include <windows.h>
#     define LIBXSMM_LOCK_SPINLOCK spin
#     define LIBXSMM_LOCK_MUTEX mutex
#     define LIBXSMM_LOCK_RWLOCK rwlock
#     define LIBXSMM_LOCK_ACQUIRED(KIND) LIBXSMM_CONCATENATE(LIBXSMM_LOCK_ACQUIRED_, KIND)
      /* implementation spinlock */
#     define LIBXSMM_LOCK_ACQUIRED_spin TRUE
#     define LIBXSMM_LOCK_TYPE_spin CRITICAL_SECTION
#     define LIBXSMM_LOCK_INIT_spin(LOCK, ATTR) InitializeCriticalSection(LOCK)
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
      /* implementation mutex */
#     define LIBXSMM_LOCK_ACQUIRED_mutex WAIT_OBJECT_0
#     if defined(LIBXSMM_LOCK_SYSTEM)
#       define LIBXSMM_LOCK_TYPE_mutex HANDLE
#       define LIBXSMM_LOCK_INIT_mutex(LOCK, ATTR) (*(LOCK) = CreateMutex(*(ATTR), FALSE, NULL))
#       define LIBXSMM_LOCK_DESTROY_mutex(LOCK) CloseHandle(*(LOCK))
#       define LIBXSMM_LOCK_TRYLOCK_mutex(LOCK) WaitForSingleObject(*(LOCK), 0)
#       define LIBXSMM_LOCK_ACQUIRE_mutex(LOCK) WaitForSingleObject(*(LOCK), INFINITE)
#       define LIBXSMM_LOCK_RELEASE_mutex(LOCK) ReleaseMutex(*(LOCK))
#       define LIBXSMM_LOCK_TRYREAD_mutex(LOCK) LIBXSMM_LOCK_TRYLOCK_mutex(LOCK)
#       define LIBXSMM_LOCK_ACQREAD_mutex(LOCK) LIBXSMM_LOCK_ACQUIRE_mutex(LOCK)
#       define LIBXSMM_LOCK_RELREAD_mutex(LOCK) LIBXSMM_LOCK_RELEASE_mutex(LOCK)
#       define LIBXSMM_LOCK_ATTR_TYPE_mutex LPSECURITY_ATTRIBUTES
#       define LIBXSMM_LOCK_ATTR_INIT_mutex(ATTR) (*(ATTR) = NULL)
#       define LIBXSMM_LOCK_ATTR_DESTROY_mutex(ATTR) LIBXSMM_UNUSED(ATTR)
#     endif
      /* implementation rwlock */
#     define LIBXSMM_LOCK_ACQUIRED_rwlock TRUE
#     if defined(LIBXSMM_LOCK_SYSTEM)
#       define LIBXSMM_LOCK_TYPE_rwlock SRWLOCK
#       define LIBXSMM_LOCK_INIT_rwlock(LOCK, ATTR) InitializeSRWLock(LOCK)
#       define LIBXSMM_LOCK_DESTROY_rwlock(LOCK) LIBXSMM_UNUSED(LOCK)
#       define LIBXSMM_LOCK_TRYLOCK_rwlock(LOCK) TryAcquireSRWLockExclusive(LOCK)
#       define LIBXSMM_LOCK_ACQUIRE_rwlock(LOCK) AcquireSRWLockExclusive(LOCK)
#       define LIBXSMM_LOCK_RELEASE_rwlock(LOCK) ReleaseSRWLockExclusive(LOCK)
#       define LIBXSMM_LOCK_TRYREAD_rwlock(LOCK) TryAcquireSRWLockShared(LOCK)
#       define LIBXSMM_LOCK_ACQREAD_rwlock(LOCK) AcquireSRWLockShared(LOCK)
#       define LIBXSMM_LOCK_RELREAD_rwlock(LOCK) ReleaseSRWLockShared(LOCK)
#       define LIBXSMM_LOCK_ATTR_TYPE_rwlock int
#       define LIBXSMM_LOCK_ATTR_INIT_rwlock(ATTR) LIBXSMM_UNUSED(ATTR)
#       define LIBXSMM_LOCK_ATTR_DESTROY_rwlock(ATTR) LIBXSMM_UNUSED(ATTR)
#     endif
#   else
#     include <pthread.h>
#     if defined(__APPLE__) && defined(__MACH__)
#       define LIBXSMM_LOCK_SPINLOCK mutex
#     else
#       define LIBXSMM_LOCK_SPINLOCK spin
#     endif
#     define LIBXSMM_LOCK_MUTEX mutex
#     define LIBXSMM_LOCK_RWLOCK rwlock
#     define LIBXSMM_LOCK_ACQUIRED(KIND) 0
      /* implementation spinlock */
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
#     if defined(LIBXSMM_LOCK_SYSTEM)
        /* implementation mutex */
#       define LIBXSMM_LOCK_TYPE_mutex pthread_mutex_t
#       define LIBXSMM_LOCK_INIT_mutex(LOCK, ATTR) LIBXSMM_EXPECT(0, pthread_mutex_init(LOCK, ATTR))
#       define LIBXSMM_LOCK_DESTROY_mutex(LOCK) LIBXSMM_EXPECT(0, pthread_mutex_destroy(LOCK))
#       define LIBXSMM_LOCK_TRYLOCK_mutex(LOCK) pthread_mutex_trylock(LOCK)
#       define LIBXSMM_LOCK_ACQUIRE_mutex(LOCK) LIBXSMM_EXPECT(0, pthread_mutex_lock(LOCK))
#       define LIBXSMM_LOCK_RELEASE_mutex(LOCK) LIBXSMM_EXPECT(0, pthread_mutex_unlock(LOCK))
#       define LIBXSMM_LOCK_TRYREAD_mutex(LOCK) LIBXSMM_LOCK_TRYLOCK_mutex(LOCK)
#       define LIBXSMM_LOCK_ACQREAD_mutex(LOCK) LIBXSMM_LOCK_ACQUIRE_mutex(LOCK)
#       define LIBXSMM_LOCK_RELREAD_mutex(LOCK) LIBXSMM_LOCK_RELEASE_mutex(LOCK)
#       define LIBXSMM_LOCK_ATTR_TYPE_mutex pthread_mutexattr_t
#       if defined(NDEBUG)
#         define LIBXSMM_LOCK_ATTR_INIT_mutex(ATTR) pthread_mutexattr_init(ATTR); \
                            pthread_mutexattr_settype(ATTR, PTHREAD_MUTEX_NORMAL)
#       else
#         define LIBXSMM_LOCK_ATTR_INIT_mutex(ATTR) LIBXSMM_EXPECT(0, pthread_mutexattr_init(ATTR)); \
                        LIBXSMM_EXPECT(0, pthread_mutexattr_settype(ATTR, PTHREAD_MUTEX_ERRORCHECK))
#       endif
#       define LIBXSMM_LOCK_ATTR_DESTROY_mutex(ATTR) LIBXSMM_EXPECT(0, pthread_mutexattr_destroy(ATTR))
        /* implementation rwlock */
#       define LIBXSMM_LOCK_TYPE_rwlock pthread_rwlock_t
#       define LIBXSMM_LOCK_INIT_rwlock(LOCK, ATTR) LIBXSMM_EXPECT(0, pthread_rwlock_init(LOCK, ATTR))
#       define LIBXSMM_LOCK_DESTROY_rwlock(LOCK) LIBXSMM_EXPECT(0, pthread_rwlock_destroy(LOCK))
#       define LIBXSMM_LOCK_TRYLOCK_rwlock(LOCK) pthread_rwlock_trywrlock(LOCK)
#       define LIBXSMM_LOCK_ACQUIRE_rwlock(LOCK) LIBXSMM_EXPECT(0, pthread_rwlock_wrlock(LOCK))
#       define LIBXSMM_LOCK_RELEASE_rwlock(LOCK) LIBXSMM_EXPECT(0, pthread_rwlock_unlock(LOCK))
#       define LIBXSMM_LOCK_TRYREAD_rwlock(LOCK) pthread_rwlock_tryrdlock(LOCK)
#       define LIBXSMM_LOCK_ACQREAD_rwlock(LOCK) LIBXSMM_EXPECT(0, pthread_rwlock_rdlock(LOCK))
#       define LIBXSMM_LOCK_RELREAD_rwlock(LOCK) LIBXSMM_LOCK_RELEASE_rwlock(LOCK)
#       define LIBXSMM_LOCK_ATTR_TYPE_rwlock pthread_rwlockattr_t
#       define LIBXSMM_LOCK_ATTR_INIT_rwlock(ATTR) LIBXSMM_EXPECT(0, pthread_rwlockattr_init(ATTR))
#       define LIBXSMM_LOCK_ATTR_DESTROY_rwlock(ATTR) LIBXSMM_EXPECT(0, pthread_rwlockattr_destroy(ATTR))
#     endif
#   endif
# endif
# if !defined(LIBXSMM_LOCK_SYSTEM)
#   define LIBXSMM_LOCK_TYPE_mutex libxsmm_mutex*
#   define LIBXSMM_LOCK_INIT_mutex(LOCK, ATTR) (*(LOCK) = libxsmm_mutex_create())
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

#   define LIBXSMM_LOCK_TYPE_rwlock libxsmm_rwlock*
#   define LIBXSMM_LOCK_INIT_rwlock(LOCK, ATTR) (*(LOCK) = libxsmm_rwlock_create())
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
# endif
#else
# define LIBXSMM_LOCK_SPINLOCK spin
# define LIBXSMM_LOCK_MUTEX mutex
# define LIBXSMM_LOCK_RWLOCK rwlock
# define LIBXSMM_LOCK_ACQUIRED(KIND) 0
# define LIBXSMM_LOCK_ATTR_TYPE(KIND) int
# define LIBXSMM_LOCK_ATTR_INIT(KIND, ATTR) LIBXSMM_UNUSED(ATTR)
# define LIBXSMM_LOCK_ATTR_DESTROY(KIND, ATTR) LIBXSMM_UNUSED(ATTR)
# define LIBXSMM_LOCK_TYPE(KIND) int
# define LIBXSMM_LOCK_INIT(KIND, LOCK, ATTR) LIBXSMM_UNUSED(LOCK); LIBXSMM_UNUSED(ATTR)
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
typedef struct LIBXSMM_RETARGETABLE libxsmm_barrier libxsmm_barrier;

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

/** Mutex, which is eventually not based on LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK_MUTEX). */
typedef struct LIBXSMM_RETARGETABLE libxsmm_mutex libxsmm_mutex;
LIBXSMM_API libxsmm_mutex* libxsmm_mutex_create(void);
LIBXSMM_API void libxsmm_mutex_destroy(const libxsmm_mutex* mutex);
LIBXSMM_API int libxsmm_mutex_trylock(libxsmm_mutex* mutex);
LIBXSMM_API void libxsmm_mutex_acquire(libxsmm_mutex* mutex);
LIBXSMM_API void libxsmm_mutex_release(libxsmm_mutex* mutex);

/** RW-lock, which is eventually not based on LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK_RWLOCK). */
typedef struct LIBXSMM_RETARGETABLE libxsmm_rwlock libxsmm_rwlock;
LIBXSMM_API libxsmm_rwlock* libxsmm_rwlock_create(void);
LIBXSMM_API void libxsmm_rwlock_destroy(const libxsmm_rwlock* rwlock);
LIBXSMM_API int libxsmm_rwlock_trylock(libxsmm_rwlock* mutex);
LIBXSMM_API void libxsmm_rwlock_acquire(libxsmm_rwlock* rwlock);
LIBXSMM_API void libxsmm_rwlock_release(libxsmm_rwlock* rwlock);
LIBXSMM_API int libxsmm_rwlock_tryread(libxsmm_rwlock* mutex);
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

