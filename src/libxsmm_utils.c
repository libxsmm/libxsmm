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
#include <utils/libxsmm_utils.h>
#include "libxsmm_main.h"


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


LIBXSMM_API void libxsmm_gemm_print(void* ostream,
  libxsmm_datatype precision, const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
  const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc)
{
  libxsmm_gemm_print2(ostream, precision, precision, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


LIBXSMM_API void libxsmm_gemm_print2(void* ostream,
  libxsmm_datatype iprec, libxsmm_datatype oprec, const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
  const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc)
{
  const libxsmm_blasint nn = *(n ? n : m), kk = *(k ? k : m);
  const int itransa = LIBXSMM_FLAGS & LIBXSMM_GEMM_FLAG_TRANS_A, itransb = LIBXSMM_FLAGS & LIBXSMM_GEMM_FLAG_TRANS_B;
  const char ctransa = (char)(NULL != transa ? (*transa) : (0 == itransa ? 'n' : 't'));
  const char ctransb = (char)(NULL != transb ? (*transb) : (0 == itransb ? 'n' : 't'));
  const libxsmm_blasint ilda = (NULL != lda ? *lda : (('n' == ctransa || 'N' == ctransa) ? *m : kk));
  const libxsmm_blasint ildb = (NULL != ldb ? *ldb : (('n' == ctransb || 'N' == ctransb) ? kk : nn));
  const libxsmm_blasint ildc = *(NULL != ldc ? ldc : m);
  libxsmm_mhd_elemtype mhd_elemtype = LIBXSMM_MHD_ELEMTYPE_UNKNOWN;
  char string_a[128] = "", string_b[128] = "", typeprefix = 0;

  switch (iprec | oprec) {
    case LIBXSMM_DATATYPE_F64: {
      LIBXSMM_ASSERT(iprec == oprec);
      LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "%g", NULL != alpha ? *((const double*)alpha) : LIBXSMM_ALPHA);
      LIBXSMM_SNPRINTF(string_b, sizeof(string_b), "%g", NULL != beta  ? *((const double*)beta)  : LIBXSMM_BETA);
      mhd_elemtype = LIBXSMM_MHD_ELEMTYPE_F64;
      typeprefix = 'd';
    } break;
    case LIBXSMM_DATATYPE_F32: {
      LIBXSMM_ASSERT(iprec == oprec);
      LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "%g", NULL != alpha ? *((const float*)alpha) : LIBXSMM_ALPHA);
      LIBXSMM_SNPRINTF(string_b, sizeof(string_b), "%g", NULL != beta  ? *((const float*)beta)  : LIBXSMM_BETA);
      mhd_elemtype = LIBXSMM_MHD_ELEMTYPE_F32;
      typeprefix = 's';
    } break;
    default: if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
      static int error_once = 0;
      if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) { /* TODO: support I16, etc. */
        fprintf(stderr, "LIBXSMM ERROR: unsupported data-type requested!\n");
      }
    }
  }

  if (0 != typeprefix) {
    if (NULL != ostream) { /* print information about GEMM call */
      if (NULL != a && NULL != b && NULL != c) {
        fprintf((FILE*)ostream, "%cgemm('%c', '%c', %" PRIuPTR "/*m*/, %" PRIuPTR "/*n*/, %" PRIuPTR "/*k*/,\n"
                                "  %s/*alpha*/, %p/*a*/, %" PRIuPTR "/*lda*/,\n"
                                "              %p/*b*/, %" PRIuPTR "/*ldb*/,\n"
                                "   %s/*beta*/, %p/*c*/, %" PRIuPTR "/*ldc*/)",
          typeprefix, ctransa, ctransb, (uintptr_t)*m, (uintptr_t)nn, (uintptr_t)kk,
          string_a, a, (uintptr_t)ilda, b, (uintptr_t)ildb, string_b, c, (uintptr_t)ildc);
      }
      else {
        fprintf((FILE*)ostream, "%cgemm(trans=%c%c mnk=%" PRIuPTR ",%" PRIuPTR ",%" PRIuPTR
                                                 " ldx=%" PRIuPTR ",%" PRIuPTR ",%" PRIuPTR " a,b=%s,%s)",
          typeprefix, ctransa, ctransb, (uintptr_t)*m, (uintptr_t)nn, (uintptr_t)kk,
          (uintptr_t)ilda, (uintptr_t)ildb, (uintptr_t)ildc, string_a, string_b);
      }
    }
    else { /* dump A, B, and C matrices into MHD files */
      char extension_header[256] = "";
      size_t data_size[2] = { 0 }, size[2] = { 0 };

      if (NULL != a) {
        LIBXSMM_SNPRINTF(extension_header, sizeof(extension_header), "TRANS = %c\nALPHA = %s", ctransa, string_a);
        LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "libxsmm_a_%p.mhd", a);
        data_size[0] = (size_t)ilda; data_size[1] = (size_t)kk; size[0] = (size_t)(*m); size[1] = (size_t)kk;
        libxsmm_mhd_write(string_a, NULL/*offset*/, size, data_size, 2/*ndims*/, 1/*ncomponents*/, mhd_elemtype,
          NULL/*conversion*/, a, NULL/*header_size*/, extension_header, NULL/*extension*/, 0/*extension_size*/);
      }
      if (NULL != b) {
        LIBXSMM_SNPRINTF(extension_header, sizeof(extension_header), "\nTRANS = %c", ctransb);
        LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "libxsmm_b_%p.mhd", b);
        data_size[0] = (size_t)ildb; data_size[1] = (size_t)nn; size[0] = (size_t)kk; size[1] = (size_t)nn;
        libxsmm_mhd_write(string_a, NULL/*offset*/, size, data_size, 2/*ndims*/, 1/*ncomponents*/, mhd_elemtype,
          NULL/*conversion*/, b, NULL/*header_size*/, extension_header, NULL/*extension*/, 0/*extension_size*/);
      }
      if (NULL != c) {
        LIBXSMM_SNPRINTF(extension_header, sizeof(extension_header), "BETA = %s", string_b);
        LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "libxsmm_c_%p.mhd", c);
        data_size[0] = (size_t)ildc; data_size[1] = (size_t)nn; size[0] = (size_t)(*m); size[1] = (size_t)nn;
        libxsmm_mhd_write(string_a, NULL/*offset*/, size, data_size, 2/*ndims*/, 1/*ncomponents*/, mhd_elemtype,
          NULL/*conversion*/, c, NULL/*header_size*/, extension_header, NULL/*extension*/, 0/*extension_size*/);
      }
    }
  }
}


LIBXSMM_API void libxsmm_gemm_dprint(
  void* ostream, libxsmm_datatype precision, char transa, char transb,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, double dalpha, const void* a, libxsmm_blasint lda,
  const void* b, libxsmm_blasint ldb, double dbeta, void* c, libxsmm_blasint ldc)
{
  libxsmm_gemm_dprint2(ostream, precision, precision, transa, transb, m, n, k, dalpha, a, lda, b, ldb, dbeta, c, ldc);
}


LIBXSMM_API void libxsmm_gemm_dprint2(
  void* ostream, libxsmm_datatype iprec, libxsmm_datatype oprec, char transa, char transb,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, double dalpha, const void* a, libxsmm_blasint lda,
  const void* b, libxsmm_blasint ldb, double dbeta, void* c, libxsmm_blasint ldc)
{
  switch ((int)iprec) {
    case LIBXSMM_DATATYPE_F64: {
      libxsmm_gemm_print2(ostream, LIBXSMM_DATATYPE_F64, oprec, &transa, &transb,
        &m, &n, &k, &dalpha, a, &lda, b, &ldb, &dbeta, c, &ldc);
    } break;
    case LIBXSMM_DATATYPE_F32: {
      const float alpha = (float)dalpha, beta = (float)dbeta;
      libxsmm_gemm_print2(ostream, LIBXSMM_DATATYPE_F32, oprec, &transa, &transb,
        &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    } break;
    default: {
      libxsmm_gemm_print2(ostream, iprec, oprec, &transa, &transb,
        &m, &n, &k, &dalpha, a, &lda, b, &ldb, &dbeta, c, &ldc);
    }
  }
}


LIBXSMM_API void libxsmm_gemm_xprint(void* ostream,
  libxsmm_xmmfunction kernel, const void* a, const void* b, void* c)
{
  const libxsmm_descriptor* desc;
  libxsmm_code_pointer code = { NULL };
  size_t code_size;
  code.xgemm = kernel;
  if (NULL != libxsmm_get_kernel_xinfo(code, &desc, &code_size) &&
      NULL != desc && LIBXSMM_KERNEL_KIND_MATMUL == LIBXSMM_DESCRIPTOR_KIND(desc->kind))
  {
    libxsmm_gemm_dprint2(ostream,
      (libxsmm_datatype)LIBXSMM_GETENUM_INP(desc->gemm.desc.datatype),
      (libxsmm_datatype)LIBXSMM_GETENUM_OUT(desc->gemm.desc.datatype),
      (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_A & desc->gemm.desc.flags) ? 'N' : 'T'),
      (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_B & desc->gemm.desc.flags) ? 'N' : 'T'),
      (libxsmm_blasint)desc->gemm.desc.m, (libxsmm_blasint)desc->gemm.desc.n, (libxsmm_blasint)desc->gemm.desc.k,
      1, a, (libxsmm_blasint)desc->gemm.desc.lda, b, (libxsmm_blasint)desc->gemm.desc.ldb,
      0 != (LIBXSMM_GEMM_FLAG_BETA_0 & desc->gemm.desc.flags) ? 0 : 1, c, (libxsmm_blasint)desc->gemm.desc.ldc);
    fprintf((FILE*)ostream, " = %p+%u", code.ptr_const, (unsigned int)code_size);
  }
}
