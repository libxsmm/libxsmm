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
#include <utils/libxsmm_barrier.h>
#include <libxsmm_malloc.h>
#include <libxsmm_sync.h>


LIBXSMM_EXTERN_C typedef struct internal_sync_core_tag { /* per-core */
  uint8_t id;
  volatile uint8_t core_sense;
  volatile uint8_t* thread_senses;
  volatile uint8_t* my_flags[2];
  uint8_t** partner_flags[2];
  uint8_t parity;
  uint8_t sense;
} internal_sync_core_tag;

LIBXSMM_EXTERN_C typedef struct internal_sync_thread_tag { /* per-thread */
  int core_tid;
  internal_sync_core_tag *core;
} internal_sync_thread_tag;

struct libxsmm_barrier {
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


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
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
      while (0/*false*/ == barrier->init_done) {} /* empty block instead of semicolon */
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
      while (2 != barrier->init_done) {} /* empty block instead of semicolon */
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
#pragma GCC diagnostic pop
