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
/* Alexander Heinecke (Intel Corp.), Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm_sync.h>
#include <libxsmm_intrinsics_x86.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <assert.h>
#include <stdint.h>
#include <math.h>
#if defined(__linux__)
# include <syscall.h>
#endif
#if defined(_WIN32)
# include <process.h>
#else
# include <unistd.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXSMM_SYNC_CACHELINE_SIZE)
# define LIBXSMM_SYNC_CACHELINE_SIZE 64
#endif
#if !defined(LIBXSMM_SYNC_DELAY)
# define LIBXSMM_SYNC_DELAY 8
#endif
#if !defined(LIBXSMM_SYNC_ATOMIC_SET)
# define LIBXSMM_SYNC_ATOMIC_SET(DST, VALUE) ((DST) = (VALUE))
#endif
#if !defined(LIBXSMM_SYNC_MALLOC_INTRINSIC)
# define LIBXSMM_SYNC_MALLOC_INTRINSIC
#endif
#if defined(LIBXSMM_SYNC_MALLOC_INTRINSIC)
# define LIBXSMM_SYNC_MALLOC(SIZE, ALIGNMENT) _mm_malloc(SIZE, ALIGNMENT)
# define LIBXSMM_SYNC_FREE(BUFFER) _mm_free((void*)(BUFFER))
#else
# define LIBXSMM_SYNC_MALLOC(SIZE, ALIGNMENT) libxsmm_aligned_malloc(SIZE, -(ALIGNMENT))
# define LIBXSMM_SYNC_FREE(BUFFER) libxsmm_free(BUFFER)
#endif
#if defined(__MIC__)
# define LIBXSMM_SYNC_PAUSE(DELAY) _mm_delay_32(DELAY)
#elif !defined(LIBXSMM_INTRINSICS_NONE)
# define LIBXSMM_SYNC_PAUSE(DELAY) _mm_pause()
#else
# define LIBXSMM_SYNC_PAUSE(DELAY)
#endif


/* internal counter type which is guaranteed to be atomic when using certain methods */
typedef struct LIBXSMM_RETARGETABLE internal_sync_counter {
  volatile int counter;
} internal_sync_counter;

typedef struct LIBXSMM_RETARGETABLE internal_sync_core_tag { /* per-core */
  uint8_t id;
  volatile uint8_t core_sense;
  volatile uint8_t* thread_senses;
  volatile uint8_t* my_flags[2];
  uint8_t** partner_flags[2];
  uint8_t parity;
  uint8_t sense;
} internal_sync_core_tag;

typedef struct LIBXSMM_RETARGETABLE internal_sync_thread_tag { /* per-thread */
  int core_tid;
  internal_sync_core_tag *core;
} internal_sync_thread_tag;

struct LIBXSMM_RETARGETABLE libxsmm_barrier {
  internal_sync_core_tag** cores;
  internal_sync_thread_tag** threads;
  int ncores, nthreads_per_core;
  int nthreads, ncores_log2;
  /* thread-safety during initialization */
  internal_sync_counter threads_waiting;
  volatile uint8_t init_done;
};


LIBXSMM_API_DEFINITION libxsmm_barrier* libxsmm_barrier_create(int ncores, int nthreads_per_core)
{
  libxsmm_barrier *const barrier = (libxsmm_barrier*)LIBXSMM_SYNC_MALLOC(
    sizeof(libxsmm_barrier), LIBXSMM_SYNC_CACHELINE_SIZE);
#if defined(_REENTRANT)
  barrier->ncores = ncores;
  barrier->ncores_log2 = (int)ceil(log2(ncores));
  barrier->nthreads_per_core = nthreads_per_core;
  barrier->nthreads = ncores * nthreads_per_core;

  barrier->threads = (internal_sync_thread_tag**)LIBXSMM_SYNC_MALLOC(
    barrier->nthreads * sizeof(internal_sync_thread_tag*), LIBXSMM_SYNC_CACHELINE_SIZE);
  barrier->cores = (internal_sync_core_tag**)LIBXSMM_SYNC_MALLOC(
    barrier->ncores * sizeof(internal_sync_core_tag*), LIBXSMM_SYNC_CACHELINE_SIZE);

  LIBXSMM_SYNC_ATOMIC_SET(barrier->threads_waiting.counter, barrier->nthreads);
  barrier->init_done = 0;
#else
  LIBXSMM_UNUSED(ncores); LIBXSMM_UNUSED(nthreads_per_core);
#endif
  return barrier;
}


LIBXSMM_API_DEFINITION void libxsmm_barrier_init(libxsmm_barrier* barrier, int tid)
{
#if defined(_REENTRANT)
  const int cid = tid / barrier->nthreads_per_core; /* this thread's core ID */
  internal_sync_core_tag* core = 0;
  int i;
  internal_sync_thread_tag* thread;

  /* we only initialize the barrier once */
  if (barrier->init_done == 2) {
    return;
  }

  /* allocate per-thread structure */
  thread = (internal_sync_thread_tag*)LIBXSMM_SYNC_MALLOC(
    sizeof(internal_sync_thread_tag), LIBXSMM_SYNC_CACHELINE_SIZE);
  barrier->threads[tid] = thread;
  thread->core_tid = tid - (barrier->nthreads_per_core * cid); /* mod */

  /* each core's thread 0 does all the allocations */
  if (0 == thread->core_tid) {
    core = (internal_sync_core_tag*)LIBXSMM_SYNC_MALLOC(
      sizeof(internal_sync_core_tag), LIBXSMM_SYNC_CACHELINE_SIZE);
    core->id = (uint8_t)cid;
    core->core_sense = 1;

    core->thread_senses = (uint8_t*)LIBXSMM_SYNC_MALLOC(
      barrier->nthreads_per_core * sizeof(uint8_t), LIBXSMM_SYNC_CACHELINE_SIZE);
    for (i = 0; i < barrier->nthreads_per_core; ++i) core->thread_senses[i] = 1;

    for (i = 0; i < 2;  ++i) {
      core->my_flags[i] = (uint8_t*)LIBXSMM_SYNC_MALLOC(
        barrier->ncores_log2 * sizeof(uint8_t) * LIBXSMM_SYNC_CACHELINE_SIZE,
        LIBXSMM_SYNC_CACHELINE_SIZE);
      core->partner_flags[i] = (uint8_t**)LIBXSMM_SYNC_MALLOC(
        barrier->ncores_log2 * sizeof(uint8_t*),
        LIBXSMM_SYNC_CACHELINE_SIZE);
    }

    core->parity = 0;
    core->sense = 1;
    barrier->cores[cid] = core;
  }

  /* barrier to let all the allocations complete */
  if (0 == LIBXSMM_ATOMIC_SUB_FETCH(&barrier->threads_waiting.counter, 1, LIBXSMM_ATOMIC_RELAXED)) {
    LIBXSMM_SYNC_ATOMIC_SET(barrier->threads_waiting.counter, barrier->nthreads);
    barrier->init_done = 1;
  }
  else {
    while (0 == barrier->init_done);
  }

  /* set required per-thread information */
  thread->core = barrier->cores[cid];

  /* each core's thread 0 completes setup */
  if (0 == thread->core_tid) {
    int di;
    for (i = di = 0; i < barrier->ncores_log2; ++i, di += LIBXSMM_SYNC_CACHELINE_SIZE) {
      /* find dissemination partner and link to it */
      const int dissem_cid = (cid + (1 << i)) % barrier->ncores;
      assert(0 != core); /* initialized under the same condition; see above */
      core->my_flags[0][di] = core->my_flags[1][di] = 0;
      core->partner_flags[0][i] = (uint8_t*)&barrier->cores[dissem_cid]->my_flags[0][di];
      core->partner_flags[1][i] = (uint8_t*)&barrier->cores[dissem_cid]->my_flags[1][di];
    }
  }

  /* barrier to let initialization complete */
  if (0 == LIBXSMM_ATOMIC_SUB_FETCH(&barrier->threads_waiting.counter, 1, LIBXSMM_ATOMIC_RELAXED)) {
    LIBXSMM_SYNC_ATOMIC_SET(barrier->threads_waiting.counter, barrier->nthreads);
    barrier->init_done = 2;
  }
  else {
    while (2 != barrier->init_done);
  }
#else
  LIBXSMM_UNUSED(barrier); LIBXSMM_UNUSED(tid);
#endif
}


LIBXSMM_API_DEFINITION LIBXSMM_INTRINSICS void libxsmm_barrier_wait(libxsmm_barrier* barrier, int tid)
{
#if defined(_REENTRANT)
  internal_sync_thread_tag *const thread = barrier->threads[tid];
  internal_sync_core_tag *const core = thread->core;

  /* first signal this thread's arrival */
  core->thread_senses[thread->core_tid] = (uint8_t)(0 == core->thread_senses[thread->core_tid] ? 1 : 0);

  /* each core's thread 0 syncs across cores */
  if (0 == thread->core_tid) {
    int i;
    /* wait for the core's remaining threads */
    for (i = 1; i < barrier->nthreads_per_core; ++i) {
      uint8_t core_sense = core->core_sense, thread_sense = core->thread_senses[i];
      while (core_sense == thread_sense) { /* avoid evaluation in unspecified order */
        LIBXSMM_SYNC_PAUSE(LIBXSMM_SYNC_DELAY);
        core_sense = core->core_sense;
        thread_sense = core->thread_senses[i];
      }
    }

    if (1 < barrier->ncores) {
      int di;
#if defined(__MIC__)
      /* cannot use LIBXSMM_ALIGNED since attribute may not apply to local non-static arrays */
      uint8_t sendbuffer[LIBXSMM_SYNC_CACHELINE_SIZE+LIBXSMM_SYNC_CACHELINE_SIZE-1];
      uint8_t *const sendbuf = LIBXSMM_ALIGN2(sendbuffer, LIBXSMM_SYNC_CACHELINE_SIZE);
      __m512d m512d;
      _mm_prefetch((const char*)core->partner_flags[core->parity][0], _MM_HINT_ET1);
      sendbuf[0] = core->sense;
      m512d = _mm512_load_pd(sendbuf);
#endif

      for (i = di = 0; i < barrier->ncores_log2 - 1; ++i, di += LIBXSMM_SYNC_CACHELINE_SIZE) {
#if defined(__MIC__)
        _mm_prefetch((const char*)core->partner_flags[core->parity][i+1], _MM_HINT_ET1);
        _mm512_storenrngo_pd(core->partner_flags[core->parity][i], m512d);
#else
        *core->partner_flags[core->parity][i] = core->sense;
#endif
        while (core->my_flags[core->parity][di] != core->sense) LIBXSMM_SYNC_PAUSE(LIBXSMM_SYNC_DELAY);
      }

#if defined(__MIC__)
      _mm512_storenrngo_pd(core->partner_flags[core->parity][i], m512d);
#else
      *core->partner_flags[core->parity][i] = core->sense;
#endif
      while (core->my_flags[core->parity][di] != core->sense) LIBXSMM_SYNC_PAUSE(LIBXSMM_SYNC_DELAY);
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
      LIBXSMM_SYNC_PAUSE(LIBXSMM_SYNC_DELAY);
      core_sense = core->core_sense;
      thread_sense = core->thread_senses[thread->core_tid];
    }
  }
#else
  LIBXSMM_UNUSED(barrier); LIBXSMM_UNUSED(tid);
#endif
}


LIBXSMM_API_DEFINITION void libxsmm_barrier_release(const libxsmm_barrier* barrier)
{
#if defined(_REENTRANT)
  int i;
  if ( barrier->init_done == 2 ) {
    for (i = 0; i < barrier->ncores; ++i) {
      int j;
      LIBXSMM_SYNC_FREE(barrier->cores[i]->thread_senses);
      for (j = 0; j < 2; ++j) {
        LIBXSMM_SYNC_FREE(barrier->cores[i]->partner_flags[j]);
        LIBXSMM_SYNC_FREE(barrier->cores[i]->my_flags[j]);
      }
      LIBXSMM_SYNC_FREE(barrier->cores[i]);
    }
    for (i = 0; i < barrier->nthreads; ++i) {
      LIBXSMM_SYNC_FREE(barrier->threads[i]);
    }
  }
  LIBXSMM_SYNC_FREE(barrier->cores);
  LIBXSMM_SYNC_FREE(barrier->threads);
#endif
  LIBXSMM_SYNC_FREE(barrier);
}


LIBXSMM_API_DEFINITION unsigned int libxsmm_get_pid(void)
{
#if defined(_WIN32)
  return (unsigned int)_getpid();
#else
  return (unsigned int)getpid();
#endif
}


LIBXSMM_API_DEFINITION unsigned int libxsmm_get_tid(void)
{
#if defined(__linux__)
  return (unsigned int)syscall(__NR_gettid);
#else /* fallback */
  static LIBXSMM_TLS unsigned int tid = (unsigned int)(-1);
  if ((unsigned int)(-1) == tid) {
    static unsigned int tc = 0; tid = tc;
    LIBXSMM_ATOMIC_ADD_FETCH(&tc, 1, LIBXSMM_ATOMIC_RELAXED);
  }
  return tid;
#endif
}

