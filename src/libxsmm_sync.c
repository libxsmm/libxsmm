/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/*  Hans Pabst, Alexander Heinecke (Intel Corp.)
******************************************************************************/
/* Lock primitives inspired by Karl Malbrain, Concurrency Kit, and TF/sync.
******************************************************************************/
#include "libxsmm_main.h"

#if !defined(LIBXSMM_SYNC_FUTEX) && defined(__linux__) && defined(__USE_GNU)
# define LIBXSMM_SYNC_FUTEX
#endif

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdint.h>
#if defined(_WIN32)
# include <process.h>
#else
# if defined(LIBXSMM_SYNC_FUTEX) && defined(__linux__) && defined(__USE_GNU)
#   include <linux/futex.h>
# endif
# include <unistd.h>
# include <time.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXSMM_SYNC_RWLOCK_BITS)
# if defined(__MINGW32__)
#   define LIBXSMM_SYNC_RWLOCK_BITS 32
# else
#   define LIBXSMM_SYNC_RWLOCK_BITS 16
# endif
#endif


LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE internal_sync_core_tag { /* per-core */
  uint8_t id;
  volatile uint8_t core_sense;
  volatile uint8_t* thread_senses;
  volatile uint8_t* my_flags[2];
  uint8_t** partner_flags[2];
  uint8_t parity;
  uint8_t sense;
} internal_sync_core_tag;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE internal_sync_thread_tag { /* per-thread */
  int core_tid;
  internal_sync_core_tag *core;
} internal_sync_thread_tag;

struct LIBXSMM_RETARGETABLE libxsmm_barrier {
  internal_sync_core_tag** cores;
  internal_sync_thread_tag** threads;
  int ncores, nthreads_per_core;
  int nthreads, ncores_nbits; /* nbits(ncores) != log2(ncores) */
  /* internal counter type which is guaranteed to be atomic when using certain methods */
  volatile int threads_waiting;
  /* thread-safety during initialization */
  volatile uint8_t init_done;
};


LIBXSMM_API libxsmm_barrier* libxsmm_barrier_create(int ncores, int nthreads_per_core)
{
  libxsmm_barrier *const barrier = (libxsmm_barrier*)malloc(sizeof(libxsmm_barrier));
#if (0 == LIBXSMM_SYNC)
  LIBXSMM_UNUSED(ncores); LIBXSMM_UNUSED(nthreads_per_core);
#else
  if (NULL != barrier && 1 < ncores && 1 <= nthreads_per_core) {
    barrier->ncores = ncores;
    barrier->ncores_nbits = (int)LIBXSMM_NBITS(ncores);
    barrier->nthreads_per_core = nthreads_per_core;
    barrier->nthreads = ncores * nthreads_per_core;
    barrier->threads = (internal_sync_thread_tag**)libxsmm_aligned_malloc(
      barrier->nthreads * sizeof(internal_sync_thread_tag*), LIBXSMM_CACHELINE);
    barrier->cores = (internal_sync_core_tag**)libxsmm_aligned_malloc(
      barrier->ncores * sizeof(internal_sync_core_tag*), LIBXSMM_CACHELINE);
    barrier->threads_waiting = barrier->nthreads; /* atomic */
    barrier->init_done = 0; /* false */
  }
  else
#endif
  if (NULL != barrier) {
    barrier->nthreads = 1;
  }
  return barrier;
}


LIBXSMM_API void libxsmm_barrier_init(libxsmm_barrier* barrier, int tid)
{
#if (0 == LIBXSMM_SYNC)
  LIBXSMM_UNUSED(barrier); LIBXSMM_UNUSED(tid);
#else
  if (NULL != barrier && 1 < barrier->nthreads) {
    const int cid = tid / barrier->nthreads_per_core; /* this thread's core ID */
    internal_sync_core_tag* core = 0;
    int i;
    internal_sync_thread_tag* thread;

    /* we only initialize the barrier once */
    if (barrier->init_done == 2) {
      return;
    }

    /* allocate per-thread structure */
    thread = (internal_sync_thread_tag*)libxsmm_aligned_malloc(
      sizeof(internal_sync_thread_tag), LIBXSMM_CACHELINE);
    barrier->threads[tid] = thread;
    thread->core_tid = tid - (barrier->nthreads_per_core * cid); /* mod */

    /* each core's thread 0 does all the allocations */
    if (0 == thread->core_tid) {
      core = (internal_sync_core_tag*)libxsmm_aligned_malloc(
        sizeof(internal_sync_core_tag), LIBXSMM_CACHELINE);
      core->id = (uint8_t)cid;
      core->core_sense = 1;

      core->thread_senses = (uint8_t*)libxsmm_aligned_malloc(
        barrier->nthreads_per_core * sizeof(uint8_t), LIBXSMM_CACHELINE);
      for (i = 0; i < barrier->nthreads_per_core; ++i) core->thread_senses[i] = 1;

      for (i = 0; i < 2; ++i) {
        core->my_flags[i] = (uint8_t*)libxsmm_aligned_malloc(
          barrier->ncores_nbits * sizeof(uint8_t) * LIBXSMM_CACHELINE,
          LIBXSMM_CACHELINE);
        core->partner_flags[i] = (uint8_t**)libxsmm_aligned_malloc(
          barrier->ncores_nbits * sizeof(uint8_t*),
          LIBXSMM_CACHELINE);
      }

      core->parity = 0;
      core->sense = 1;
      barrier->cores[cid] = core;
    }

    /* barrier to let all the allocations complete */
    if (0 == LIBXSMM_ATOMIC_SUB_FETCH(&barrier->threads_waiting, 1, LIBXSMM_ATOMIC_RELAXED)) {
      barrier->threads_waiting = barrier->nthreads; /* atomic */
      barrier->init_done = 1; /* true */
    }
    else {
      while (0/*false*/ == barrier->init_done);
    }

    /* set required per-thread information */
    thread->core = barrier->cores[cid];

    /* each core's thread 0 completes setup */
    if (0 == thread->core_tid) {
      int di;
      for (i = di = 0; i < barrier->ncores_nbits; ++i, di += LIBXSMM_CACHELINE) {
        /* find dissemination partner and link to it */
        const int dissem_cid = (cid + (1 << i)) % barrier->ncores;
        assert(0 != core); /* initialized under the same condition; see above */
        core->my_flags[0][di] = core->my_flags[1][di] = 0;
        core->partner_flags[0][i] = (uint8_t*)&barrier->cores[dissem_cid]->my_flags[0][di];
        core->partner_flags[1][i] = (uint8_t*)&barrier->cores[dissem_cid]->my_flags[1][di];
      }
    }

    /* barrier to let initialization complete */
    if (0 == LIBXSMM_ATOMIC_SUB_FETCH(&barrier->threads_waiting, 1, LIBXSMM_ATOMIC_RELAXED)) {
      barrier->threads_waiting = barrier->nthreads; /* atomic */
      barrier->init_done = 2;
    }
    else {
      while (2 != barrier->init_done);
    }
  }
#endif
}


LIBXSMM_API LIBXSMM_INTRINSICS(LIBXSMM_X86_GENERIC)
void libxsmm_barrier_wait(libxsmm_barrier* barrier, int tid)
{
#if (0 == LIBXSMM_SYNC)
  LIBXSMM_UNUSED(barrier); LIBXSMM_UNUSED(tid);
#else
  if (NULL != barrier && 1 < barrier->nthreads) {
    internal_sync_thread_tag *const thread = barrier->threads[tid];
    internal_sync_core_tag *const core = thread->core;

    /* first let's execute a memory fence */
    LIBXSMM_ATOMIC_SYNC(LIBXSMM_ATOMIC_SEQ_CST);

    /* first signal this thread's arrival */
    core->thread_senses[thread->core_tid] = (uint8_t)(0 == core->thread_senses[thread->core_tid] ? 1 : 0);

    /* each core's thread 0 syncs across cores */
    if (0 == thread->core_tid) {
      int i;
      /* wait for the core's remaining threads */
      for (i = 1; i < barrier->nthreads_per_core; ++i) {
        uint8_t core_sense = core->core_sense, thread_sense = core->thread_senses[i];
        while (core_sense == thread_sense) { /* avoid evaluation in unspecified order */
          LIBXSMM_SYNC_PAUSE;
          core_sense = core->core_sense;
          thread_sense = core->thread_senses[i];
        }
      }

      if (1 < barrier->ncores) {
        int di;
# if defined(__MIC__)
        /* cannot use LIBXSMM_ALIGNED since attribute may not apply to local non-static arrays */
        uint8_t sendbuffer[LIBXSMM_CACHELINE+LIBXSMM_CACHELINE-1];
        uint8_t *const sendbuf = LIBXSMM_ALIGN(sendbuffer, LIBXSMM_CACHELINE);
        __m512d m512d;
        _mm_prefetch((const char*)core->partner_flags[core->parity][0], _MM_HINT_ET1);
        sendbuf[0] = core->sense;
        m512d = LIBXSMM_INTRINSICS_MM512_LOAD_PD(sendbuf);
# endif

        for (i = di = 0; i < barrier->ncores_nbits - 1; ++i, di += LIBXSMM_CACHELINE) {
# if defined(__MIC__)
          _mm_prefetch((const char*)core->partner_flags[core->parity][i+1], _MM_HINT_ET1);
          _mm512_storenrngo_pd(core->partner_flags[core->parity][i], m512d);
# else
          *core->partner_flags[core->parity][i] = core->sense;
# endif
          while (core->my_flags[core->parity][di] != core->sense) LIBXSMM_SYNC_PAUSE;
        }

# if defined(__MIC__)
        _mm512_storenrngo_pd(core->partner_flags[core->parity][i], m512d);
# else
        *core->partner_flags[core->parity][i] = core->sense;
# endif
        while (core->my_flags[core->parity][di] != core->sense) LIBXSMM_SYNC_PAUSE;
        if (1 == core->parity) {
          core->sense = (uint8_t)(0 == core->sense ? 1 : 0);
        }
        core->parity = (uint8_t)(1 - core->parity);
      }

      /* wake up the core's remaining threads */
      core->core_sense = core->thread_senses[0];
    }
    else { /* other threads wait for cross-core sync to complete */
      uint8_t core_sense = core->core_sense, thread_sense = core->thread_senses[thread->core_tid];
      while (core_sense != thread_sense) { /* avoid evaluation in unspecified order */
        LIBXSMM_SYNC_PAUSE;
        core_sense = core->core_sense;
        thread_sense = core->thread_senses[thread->core_tid];
      }
    }
  }
#endif
}


LIBXSMM_API void libxsmm_barrier_destroy(const libxsmm_barrier* barrier)
{
#if (0 != LIBXSMM_SYNC)
  if (NULL != barrier && 1 < barrier->nthreads) {
    if (2 == barrier->init_done) {
      int i;
      for (i = 0; i < barrier->ncores; ++i) {
        int j;
        libxsmm_free((const void*)barrier->cores[i]->thread_senses);
        for (j = 0; j < 2; ++j) {
          libxsmm_free((const void*)barrier->cores[i]->my_flags[j]);
          libxsmm_free(barrier->cores[i]->partner_flags[j]);
        }
        libxsmm_free(barrier->cores[i]);
      }
      for (i = 0; i < barrier->nthreads; ++i) {
        libxsmm_free(barrier->threads[i]);
      }
    }
    libxsmm_free(barrier->threads);
    libxsmm_free(barrier->cores);
  }
#endif
  free((libxsmm_barrier*)barrier);
}


#if (0 != LIBXSMM_SYNC)
enum {
  INTERNAL_SYNC_LOCK_FREE = 0,
  INTERNAL_SYNC_LOCK_LOCKED = 1,
  INTERNAL_SYNC_LOCK_CONTESTED = 2,
  INTERNAL_SYNC_RWLOCK_READINC = 0x10000/*(USHRT_MAX+1)*/,
  INTERNAL_SYNC_FUTEX = 202
};
#endif


typedef unsigned int libxsmm_spinlock_state;
struct LIBXSMM_RETARGETABLE libxsmm_spinlock {
  volatile libxsmm_spinlock_state state;
};


LIBXSMM_API libxsmm_spinlock* libxsmm_spinlock_create(void)
{
  libxsmm_spinlock *const result = (libxsmm_spinlock*)malloc(sizeof(libxsmm_spinlock));
#if (0 != LIBXSMM_SYNC)
  if (0 != result) {
    result->state = INTERNAL_SYNC_LOCK_FREE;
  }
#endif
  return result;
}


LIBXSMM_API void libxsmm_spinlock_destroy(const libxsmm_spinlock* spinlock)
{
  free((libxsmm_spinlock*)spinlock);
}


LIBXSMM_API int libxsmm_spinlock_trylock(libxsmm_spinlock* spinlock)
{
#if (0 != LIBXSMM_SYNC)
# if 0
  /*const*/ libxsmm_spinlock_state lock_free = INTERNAL_SYNC_LOCK_FREE;
  assert(0 != spinlock);
  return 0/*false*/ == LIBXSMM_ATOMIC_CMPSWP(&spinlock->state, lock_free, INTERNAL_SYNC_LOCK_LOCKED, LIBXSMM_ATOMIC_RELAXED)
    ? (LIBXSMM_LOCK_ACQUIRED(LIBXSMM_LOCK_SPINLOCK) + 1) /* not acquired */
    : (LIBXSMM_LOCK_ACQUIRED(LIBXSMM_LOCK_SPINLOCK));
# else
  return LIBXSMM_LOCK_ACQUIRED(LIBXSMM_LOCK_SPINLOCK) + !LIBXSMM_ATOMIC_TRYLOCK(&spinlock->state, LIBXSMM_ATOMIC_RELAXED);
# endif
#else
  LIBXSMM_UNUSED(spinlock);
  return LIBXSMM_LOCK_ACQUIRED(LIBXSMM_LOCK_SPINLOCK);
#endif
}


LIBXSMM_API void libxsmm_spinlock_acquire(libxsmm_spinlock* spinlock)
{
#if (0 != LIBXSMM_SYNC)
  assert(0 != spinlock);
  for (;;) {
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&spinlock->state, 1, LIBXSMM_ATOMIC_RELAXED)) break;
    LIBXSMM_SYNC_CYCLE(&spinlock->state, INTERNAL_SYNC_LOCK_FREE, LIBXSMM_SYNC_NPAUSE);
  }
  LIBXSMM_ATOMIC_SYNC(LIBXSMM_ATOMIC_SEQ_CST);
#else
  LIBXSMM_UNUSED(spinlock);
#endif
}


LIBXSMM_API void libxsmm_spinlock_release(libxsmm_spinlock* spinlock)
{
#if (0 != LIBXSMM_SYNC)
  assert(0 != spinlock);
  LIBXSMM_ATOMIC_SYNC(LIBXSMM_ATOMIC_SEQ_CST);
  spinlock->state = INTERNAL_SYNC_LOCK_FREE;
#else
  LIBXSMM_UNUSED(spinlock);
#endif
}


#if defined(LIBXSMM_SYNC_FUTEX) && defined(__linux__) && defined(__USE_GNU)
typedef int libxsmm_mutex_state;
#else
typedef char libxsmm_mutex_state;
#endif
struct LIBXSMM_RETARGETABLE libxsmm_mutex {
  volatile libxsmm_mutex_state state;
};


LIBXSMM_API libxsmm_mutex* libxsmm_mutex_create(void)
{
  libxsmm_mutex *const result = (libxsmm_mutex*)malloc(sizeof(libxsmm_mutex));
#if (0 != LIBXSMM_SYNC)
  if (0 != result) {
    result->state = INTERNAL_SYNC_LOCK_FREE;
  }
#endif
  return result;
}


LIBXSMM_API void libxsmm_mutex_destroy(const libxsmm_mutex* mutex)
{
  free((libxsmm_mutex*)mutex);
}


LIBXSMM_API int libxsmm_mutex_trylock(libxsmm_mutex* mutex)
{
#if (0 != LIBXSMM_SYNC)
  assert(0 != mutex);
  return LIBXSMM_LOCK_ACQUIRED(LIBXSMM_LOCK_MUTEX) + !LIBXSMM_ATOMIC_TRYLOCK(&mutex->state, LIBXSMM_ATOMIC_RELAXED);
#else
  LIBXSMM_UNUSED(mutex);
  return LIBXSMM_LOCK_ACQUIRED(LIBXSMM_LOCK_MUTEX);
#endif
}


LIBXSMM_API void libxsmm_mutex_acquire(libxsmm_mutex* mutex)
{
#if (0 != LIBXSMM_SYNC)
# if defined(_WIN32)
  assert(0 != mutex);
  while (LIBXSMM_LOCK_ACQUIRED(LIBXSMM_LOCK_MUTEX) != libxsmm_mutex_trylock(mutex)) {
    LIBXSMM_SYNC_CYCLE(&mutex->state, 0/*free*/, LIBXSMM_SYNC_NPAUSE);
  }
# else
  libxsmm_mutex_state lock_free = INTERNAL_SYNC_LOCK_FREE, lock_state = INTERNAL_SYNC_LOCK_LOCKED;
  assert(0 != mutex);
  while (0/*false*/ == LIBXSMM_ATOMIC_CMPSWP(&mutex->state, lock_free, lock_state, LIBXSMM_ATOMIC_RELAXED)) {
    libxsmm_mutex_state state;
    /* coverity[unreachable] may be reachable more than once due to volatile state */
    for (state = mutex->state; INTERNAL_SYNC_LOCK_FREE != state; state = mutex->state) {
#     if defined(LIBXSMM_SYNC_FUTEX) && defined(__linux__)
      LIBXSMM_SYNC_CYCLE_ELSE(&mutex->state, INTERNAL_SYNC_LOCK_FREE, LIBXSMM_SYNC_NPAUSE, {
        /*const*/ libxsmm_mutex_state state_locked = INTERNAL_SYNC_LOCK_LOCKED;
        if (INTERNAL_SYNC_LOCK_LOCKED != state || LIBXSMM_ATOMIC_CMPSWP(&mutex->state,
          state_locked, INTERNAL_SYNC_LOCK_CONTESTED, LIBXSMM_ATOMIC_RELAXED))
        {
          syscall(INTERNAL_SYNC_FUTEX, &mutex->state, FUTEX_WAIT, INTERNAL_SYNC_LOCK_CONTESTED, NULL, NULL, 0);
          lock_state = INTERNAL_SYNC_LOCK_CONTESTED;
        }}
      );
      break;
#     else
      LIBXSMM_SYNC_CYCLE(&mutex->state, INTERNAL_SYNC_LOCK_FREE, LIBXSMM_SYNC_NPAUSE);
#     endif
    }
    lock_free = INTERNAL_SYNC_LOCK_FREE;
  }
# endif
#else
  LIBXSMM_UNUSED(mutex);
#endif
}


LIBXSMM_API void libxsmm_mutex_release(libxsmm_mutex* mutex)
{
#if (0 != LIBXSMM_SYNC)
  assert(0 != mutex);
  LIBXSMM_ATOMIC_SYNC(LIBXSMM_ATOMIC_SEQ_CST);
# if defined(LIBXSMM_SYNC_FUTEX) && defined(__linux__) && defined(__USE_GNU)
  if (INTERNAL_SYNC_LOCK_CONTESTED == LIBXSMM_ATOMIC_FETCH_SUB(&mutex->state, 1, LIBXSMM_ATOMIC_RELAXED)) {
    mutex->state = INTERNAL_SYNC_LOCK_FREE;
    syscall(INTERNAL_SYNC_FUTEX, &mutex->state, FUTEX_WAKE, 1, NULL, NULL, 0);
  }
# else
  mutex->state = INTERNAL_SYNC_LOCK_FREE;
# endif
#else
  LIBXSMM_UNUSED(mutex);
#endif
}


#if (0 != LIBXSMM_SYNC)
typedef LIBXSMM_CONCATENATE3(uint,LIBXSMM_SYNC_RWLOCK_BITS,_t) internal_sync_uint_t;
typedef LIBXSMM_CONCATENATE3(int,LIBXSMM_SYNC_RWLOCK_BITS,_t) internal_sync_int_t;
LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE internal_sync_counter {
  struct { internal_sync_uint_t writer, reader; } kind;
  uint32_t bits;
} internal_sync_counter;
#endif
LIBXSMM_EXTERN_C struct LIBXSMM_RETARGETABLE libxsmm_rwlock {
#if (0 != LIBXSMM_SYNC)
  volatile internal_sync_counter completions;
  volatile internal_sync_counter requests;
#else
  int dummy;
#endif
};


LIBXSMM_API libxsmm_rwlock* libxsmm_rwlock_create(void)
{
  libxsmm_rwlock *const result = (libxsmm_rwlock*)malloc(sizeof(libxsmm_rwlock));
  if (0 != result) {
#if (0 != LIBXSMM_SYNC)
    LIBXSMM_MEMZERO127(&result->completions);
    LIBXSMM_MEMZERO127(&result->requests);
#else
    LIBXSMM_MEMZERO127(result);
#endif
  }
  return result;
}


LIBXSMM_API void libxsmm_rwlock_destroy(const libxsmm_rwlock* rwlock)
{
  free((libxsmm_rwlock*)rwlock);
}


#if (0 != LIBXSMM_SYNC)
LIBXSMM_API_INLINE int internal_rwlock_trylock(libxsmm_rwlock* rwlock, internal_sync_counter* prev)
{
  internal_sync_counter next = { 0 };
  assert(0 != rwlock && 0 != prev);
  do {
    prev->bits = rwlock->requests.bits;
    next.bits = prev->bits;
    ++next.kind.writer;
  }
  while (0/*false*/ == LIBXSMM_ATOMIC_CMPSWP(&rwlock->requests.bits, prev->bits, next.bits, LIBXSMM_ATOMIC_RELAXED));
  return rwlock->completions.bits != prev->bits
    ? (LIBXSMM_LOCK_ACQUIRED(LIBXSMM_LOCK_RWLOCK) + 1) /* not acquired */
    : (LIBXSMM_LOCK_ACQUIRED(LIBXSMM_LOCK_RWLOCK));
}
#endif


LIBXSMM_API int libxsmm_rwlock_trylock(libxsmm_rwlock* rwlock)
{
#if (0 != LIBXSMM_SYNC)
  internal_sync_counter prev;
  return internal_rwlock_trylock(rwlock, &prev);
#else
  LIBXSMM_UNUSED(rwlock);
  return LIBXSMM_LOCK_ACQUIRED(LIBXSMM_LOCK_RWLOCK);
#endif
}


LIBXSMM_API void libxsmm_rwlock_acquire(libxsmm_rwlock* rwlock)
{
#if (0 != LIBXSMM_SYNC)
  internal_sync_counter prev;
  if (LIBXSMM_LOCK_ACQUIRED(LIBXSMM_LOCK_RWLOCK) != internal_rwlock_trylock(rwlock, &prev)) {
    while (rwlock->completions.bits != prev.bits) {
      LIBXSMM_SYNC_CYCLE(&rwlock->completions.bits, prev.bits, LIBXSMM_SYNC_NPAUSE);
    }
  }
#else
  LIBXSMM_UNUSED(rwlock);
#endif
}


LIBXSMM_API void libxsmm_rwlock_release(libxsmm_rwlock* rwlock)
{
#if (0 != LIBXSMM_SYNC)
  assert(0 != rwlock);
  LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_FETCH_ADD, LIBXSMM_SYNC_RWLOCK_BITS)(&rwlock->completions.kind.writer, 1, LIBXSMM_ATOMIC_SEQ_CST);
#else
  LIBXSMM_UNUSED(rwlock);
#endif
}


#if (0 != LIBXSMM_SYNC)
LIBXSMM_API_INLINE int internal_rwlock_tryread(libxsmm_rwlock* rwlock, internal_sync_counter* prev)
{
#if (0 != LIBXSMM_SYNC)
  assert(0 != rwlock && 0 != prev);
  prev->bits = LIBXSMM_ATOMIC_FETCH_ADD(&rwlock->requests.bits, INTERNAL_SYNC_RWLOCK_READINC, LIBXSMM_ATOMIC_SEQ_CST);
  return rwlock->completions.kind.writer != prev->kind.writer
    ? (LIBXSMM_LOCK_ACQUIRED(LIBXSMM_LOCK_RWLOCK) + 1) /* not acquired */
    : (LIBXSMM_LOCK_ACQUIRED(LIBXSMM_LOCK_RWLOCK));
#else
  LIBXSMM_UNUSED(rwlock); LIBXSMM_UNUSED(prev);
  return LIBXSMM_LOCK_ACQUIRED(LIBXSMM_LOCK_RWLOCK);
#endif
}
#endif


LIBXSMM_API int libxsmm_rwlock_tryread(libxsmm_rwlock* rwlock)
{
#if (0 != LIBXSMM_SYNC)
  internal_sync_counter prev;
  return internal_rwlock_tryread(rwlock, &prev);
#else
  LIBXSMM_UNUSED(rwlock);
  return LIBXSMM_LOCK_ACQUIRED(LIBXSMM_LOCK_RWLOCK);
#endif
}


LIBXSMM_API void libxsmm_rwlock_acqread(libxsmm_rwlock* rwlock)
{
#if (0 != LIBXSMM_SYNC)
  internal_sync_counter prev;
  if (LIBXSMM_LOCK_ACQUIRED(LIBXSMM_LOCK_RWLOCK) != internal_rwlock_tryread(rwlock, &prev)) {
    while (rwlock->completions.kind.writer != prev.kind.writer) {
      LIBXSMM_SYNC_CYCLE(&rwlock->completions.kind.writer, prev.kind.writer, LIBXSMM_SYNC_NPAUSE);
    }
  }
#else
  LIBXSMM_UNUSED(rwlock);
#endif
}


LIBXSMM_API void libxsmm_rwlock_relread(libxsmm_rwlock* rwlock)
{
#if (0 != LIBXSMM_SYNC)
  assert(0 != rwlock);
  LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_FETCH_ADD, LIBXSMM_SYNC_RWLOCK_BITS)(&rwlock->completions.kind.reader, 1, LIBXSMM_ATOMIC_SEQ_CST);
#else
  LIBXSMM_UNUSED(rwlock);
#endif
}


LIBXSMM_API unsigned int libxsmm_get_pid(void)
{
#if defined(_WIN32)
  return (unsigned int)_getpid();
#else
  return (unsigned int)getpid();
#endif
}


LIBXSMM_API_INTERN unsigned int internal_get_tid(void);
LIBXSMM_API_INTERN unsigned int internal_get_tid(void)
{
  const unsigned int nthreads = LIBXSMM_ATOMIC_ADD_FETCH(&libxsmm_thread_count, 1, LIBXSMM_ATOMIC_RELAXED);
#if !defined(NDEBUG)
  static int error_once = 0;
  if (LIBXSMM_NTHREADS_MAX < nthreads
    && 0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: maximum number of threads is exhausted!\n");
  }
#endif
  LIBXSMM_ASSERT(LIBXSMM_ISPOT(LIBXSMM_NTHREADS_MAX));
  return LIBXSMM_MOD2(nthreads - 1, LIBXSMM_NTHREADS_MAX);
}


LIBXSMM_API unsigned int libxsmm_get_tid(void)
{
#if (0 != LIBXSMM_SYNC)
# if defined(_OPENMP) && defined(LIBXSMM_SYNC_OMP)
  return (unsigned int)omp_get_thread_num();
# else
  static LIBXSMM_TLS unsigned int tid = 0xFFFFFFFF;
  if (0xFFFFFFFF == tid) tid = internal_get_tid();
  return tid;
# endif
#else
  return 0;
#endif
}
