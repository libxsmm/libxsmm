/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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
/* Kunal Banerjee (Intel Corp.), Dheevatsa Mudigere (Intel Corp.)
******************************************************************************/

#ifndef BLOCK_XGEMM_H
#define BLOCK_XGEMM_H

#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <libxsmm.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

#define _USE_LIBXSMM_PREFETCH
#define BG_type 1 /* 1-float, 2-double */
#if (BG_type==2)
  #define BG_TYPE double
  #define _KERNEL libxsmm_dmmfunction
  #define _KERNEL_JIT libxsmm_dmmdispatch
#else
  #define BG_TYPE float
  #define _KERNEL libxsmm_smmfunction
  #define _KERNEL_JIT libxsmm_smmdispatch
#endif

// ORDER: 0-jik, 1-ijk, 2-jki, 3-ikj, 4-kji, 5-kij 
#define jik 0
#define ijk 1
#define jki 2
#define ikj 3
#define kji 4
#define kij 5

#define _alloc(size,alignment)  \
  mmap(NULL, size, PROT_READ | PROT_WRITE, \
      MAP_ANONYMOUS | MAP_SHARED | MAP_HUGETLB| MAP_POPULATE, -1, 0);
#define _free(addr) {munmap(addr, 0);}

#if defined( __MIC__) || defined (__AVX512F__)
#define VLEN 16
#elif defined (__AVX__)
#define VLEN 8
#else
#error "Either AVX or MIC must be used"
#endif

//#define OMP_LOCK 
#ifdef OMP_LOCK
#define LOCK_T omp_lock_t
#define LOCK_INIT(x) {omp_init_lock(x);}
#define LOCK_SET(x) {omp_set_lock(x);}
#define LOCK_UNSET(x) {omp_unset_lock(x);}
#else
typedef struct{volatile int var[16];}lock_var;
#define LOCK_T lock_var
#define LOCK_INIT(x) {(x)->var[0] = 0;}
#define LOCK_SET(x) {do { \
        while ((x)->var[0] ==1); \
        _mm_pause(); \
      } while(__sync_lock_test_and_set(&((x)->var[0]), 1) != 0); \
}
#define LOCK_CHECK(x) {while ((x)->var[0] ==0); _mm_pause();}
#define LOCK_UNSET(x) {__sync_lock_release(&((x)->var[0]));}
#endif

#if defined(_OPENMP)
#define BG_BARRIER_INIT(x) { \
  int nthrds; \
_Pragma("omp parallel")\
  { \
    nthrds = omp_get_num_threads(); \
  } \
  int Cores = Cores = nthrds > 68 ? nthrds > 136 ? nthrds/4 : nthrds/2 : nthrds; \
  /*printf ("Cores=%d, Threads=%d\n", Cores, nthrds);*/ \
  /* create a new barrier */ \
  x = libxsmm_barrier_create(Cores, nthrds/Cores); \
  assert(x!= NULL); \
  /* each thread must initialize with the barrier */ \
_Pragma("omp parallel") \
  { \
    libxsmm_barrier_init(x, omp_get_thread_num()); \
  } \ 
}
#define BG_BARRIER(x, y) {libxsmm_barrier_wait((libxsmm_barrier*)x, y);}
#define BG_BARRIER_DEL(x) {libxsmm_barrier_release((libxsmm_barrier*)x);}
#else
#define BG_BARRIER_INIT(x) {}
#define BG_BARRIER(x, y) {}
#define BG_BARRIER_DEL(x) {}
#endif

#ifdef __cplusplus
  extern "C"{
#endif

struct BGEMM_HANDLE_s {
  int _B_M1, _B_N1, _B_K1, _B_K2, _ORDER;
  int C_pre_init;
  LOCK_T* _wlock;
  void* bar;
  _KERNEL _l_kernel;
#ifdef _USE_LIBXSMM_PREFETCH
  _KERNEL _l_kernel_pf;
#endif
  libxsmm_barrier* barrier;
};
typedef struct BGEMM_HANDLE_s BGEMM_HANDLE;
void BGEMM_HANDLE_init (BGEMM_HANDLE* handle);

struct BGEMM_GROUP_HANDLE_s {
  void* bar;
  int* work_part;
  int* work_id;
};
typedef struct BGEMM_GROUP_HANDLE_s BGEMM_GROUP_HANDLE;
void BGEMM_GROUP_HANDLE_init (BGEMM_GROUP_HANDLE* grp_handle);

/* The following functions are to be added in a later version */
/*void BGEMM_HANDLE_alloc (BGEMM_HANDLE* handle, const int M, const int MB, const int N, const int NB);*/ /*KB*/
void BGEMM_HANDLE_delete (BGEMM_HANDLE* handle);

/******************************************************************************
  Fine grain parallelized version(s) of BGEMM - BGEMM2_7
    - Requires block structure layout for A,B matrices
    - Parallelized across all three dims - M, N, K
    - Uses fine-grain on-demand locks for write to C and fast barrier
    - Allows for calling multiple GEMMs, specified by 'count'
******************************************************************************/
/*void bgemm2_7(const int _M, const int _N, const int _K, const int B_M, const int B_N, const int B_K, BG_TYPE *Ap, BG_TYPE *Bp, BG_TYPE *Cp, int tid, int nthrds, BGEMM_HANDLE* handle);*/ /*KB*/
void mbgemm2_7_A(const int _M, const int _N, const int _K, const int B_M, const int B_N, const int B_K, BG_TYPE *Ap, BG_TYPE *Bp, BG_TYPE *Cp, int tid, int nthrds, BGEMM_HANDLE* handle);
void mbgemm2_7_B(const int _M, const int _N, const int _K, const int B_M, const int B_N, const int B_K, BG_TYPE *Ap, BG_TYPE *Bp, BG_TYPE *Cp, int tid, int nthrds, BGEMM_HANDLE* handle);

int BGEMM( int transA, int transB, const int M, const int N, const int K, const int MB, const int NB, const int KB, BG_TYPE ALPHA, BG_TYPE* A, BG_TYPE * B, BG_TYPE BETA, BG_TYPE *C, const int count, BGEMM_HANDLE* handle);

// Init function - for one-time setup and book-keeping, the locak aray can be placed in a MKL-like handle/struct instead of explictly passing around as global variaible
int BGEMM_init( int transA, int transB, const int M, const int N, const int K, const int MB, const int NB, const int KB, BG_TYPE ALPHA, BG_TYPE* A, BG_TYPE* B, BG_TYPE BETA, BG_TYPE* C, const int count, BGEMM_HANDLE* handle);

/******************************************************************************
  Explicitly threaded interface to BGEMM - BGEMM_THR
    - Allows for calling within threaded region
    - Assumes user take care of work-partitioning and synchronization
******************************************************************************/
int BGEMM_THR( int transA, int transB, const int M, const int N, const int K, const int MB, const int NB, const int KB, BG_TYPE ALPHA, BG_TYPE* A, BG_TYPE* B, BG_TYPE BETA, BG_TYPE* C, int tid, int nthrds, BGEMM_HANDLE* handle);

/******************************************************************************
  Group interface for BGEMM
    - Allows for multiple GEMM, each running on a subset of the machine
    - Requries using seperate BGEMM_group_init routine to initiale the kernel
******************************************************************************/
int BGEMM_group( int* transA, int* transB, const int* M, const int* N, const int* K, const int* MB, const int* NB, const int* KB, BG_TYPE* ALPHA, BG_TYPE** A, BG_TYPE** B, BG_TYPE* BETA, BG_TYPE** C, const int count, const int ngroups, BGEMM_HANDLE* handle, BGEMM_GROUP_HANDLE* grp_handle);

int BGEMM_group_init( int* transA, int* transB, const int* M, const int* N, const int* K, const int* MB, const int* NB, const int* KB, BG_TYPE* ALPHA, BG_TYPE** A, BG_TYPE** B, BG_TYPE* BETA, BG_TYPE** C, const int count, const int ngrooups, BGEMM_HANDLE* handle, BGEMM_GROUP_HANDLE* grp_handle);



/******************************************************************************
  Stride variant of BGEMM, using huge pages HPBGEMM
    - Works with non-block strucred data layout
******************************************************************************/
int HPBGEMM (int transA, int transB, const int M, const int N, const int K, const int MB, const int NB, const int KB, BG_TYPE ALPHA, BG_TYPE* A, const int LDA, BG_TYPE* B, const int LDB, BG_TYPE BETA, BG_TYPE* C, const int LDC, const int count, BGEMM_HANDLE* handle);

int HPBGEMM_init (int transA, int transB, const int M, const int N, const int K, const int MB, const int NB, const int KB, BG_TYPE ALPHA, BG_TYPE* A, const int LDA, BG_TYPE* B, const int LDB, BG_TYPE BETA, BG_TYPE* C, const int LDC, const int count, BGEMM_HANDLE* handle);



/******************************************************************************
  Mixed data-layout variant MBGEMM
    - Allows for A and/or B to non block structred
******************************************************************************/
int MBGEMM (int transA, int transB, const int M, const int N, const int K, const int MB, const int NB, const int KB, BG_TYPE ALPHA, BG_TYPE* A, const int LDA, BG_TYPE* B, const int LDB, BG_TYPE BETA, BG_TYPE* C, const int LDC, const int count, BGEMM_HANDLE* handle);

int MBGEMM_init (int transA, int transB, const int M, const int N, const int K, const int MB, const int NB, const int KB, BG_TYPE ALPHA, BG_TYPE* A, const int LDA, BG_TYPE* B, const int LDB, BG_TYPE BETA, BG_TYPE* C, const int LDC, const int count, BGEMM_HANDLE* handle);



#ifdef __cplusplus
}
#endif
#endif /* BLOCK_XGEMM_H */
