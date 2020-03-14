/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "edge_proxy_common.h"
#include <libxsmm_intrinsics_x86.h>
#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#if defined(_OPENMP)
# include <omp.h>
#endif

/*#define EDGE_HP_1G*/
/*#define HANDLE_AMOK*/

#if defined(EDGE_HP_1G) || defined(EDGE_HP_2M)
#include <sys/mman.h>
#include <linux/mman.h>
#endif

LIBXSMM_INLINE void* edge_hp_malloc( size_t nbytes, size_t alignment ) {
  void* ret_ptr = NULL;
#if defined(EDGE_HP_1G)
  size_t num_large_pages = nbytes / (1073741824L);
  if ( nbytes > num_large_pages*1073741824L ) {
    num_large_pages++;
  }
  nbytes = (size_t) num_large_pages * 1073741824L;
  printf("trying to allocate %ld 1G pages\n", num_large_pages);
  /*ret_ptr = mmap( NULL, nbytes, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB | MAP_HUGE_1GB, -1, 0 );*/
  ret_ptr = mmap( NULL, nbytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB | MAP_HUGE_1GB, -1, 0 );
  if ( (ret_ptr == (void *)(-1)) ) {
    fprintf(stderr,"1G mmap call failed\n");
    exit(1);
  }
#elif defined(EDGE_HP_2M)
  size_t num_large_pages = nbytes / (2097152UL);
  if ( nbytes > num_large_pages*2097152UL ) {
    num_large_pages++;
  }
  nbytes = (size_t) num_large_pages * 2097152UL;
  printf("trying to allocate %ld 2M pages\n", num_large_pages);
  /*ret_ptr = mmap( NULL, nbytes, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB, -1, 0 );*/
  ret_ptr = mmap( NULL, nbytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB, -1, 0 );
  if ( (ret_ptr == (void *)(-1)) ) {
    fprintf(stderr,"2M mmap call failed\n");
    exit(1);
  }
#else
  ret_ptr = libxsmm_aligned_malloc( nbytes, alignment );
#endif
  return ret_ptr;
}

LIBXSMM_INLINE void edge_hp_free( void* ptr,  size_t nbytes ) {
  LIBXSMM_UNUSED( nbytes );
#if defined(EDGE_HP_1G)
  /* to be implemented */
#elif defined(EDGE_HP_2M)
  /* to be implemented */
#else
  libxsmm_free( ptr );
#endif
}

#if defined(__AVX512F__)
LIBXSMM_INLINE void matMulFusedAC(  unsigned short  i_r,
                                    unsigned int    i_m,
                                    unsigned int    i_n,
                                    unsigned int    i_k,
                                    unsigned int    i_ldA,
                                    unsigned int    i_ldB,
                                    unsigned int    i_ldC,
                                          double    i_beta,
                                    const double   *i_a,
                                    const double   *i_b,
                                          double   *o_c ) {
  unsigned int l_m, l_n, l_k;
  const __m512d beta = _mm512_set1_pd( i_beta );
  LIBXSMM_UNUSED(i_r);
  for( l_m = 0; l_m < i_m; l_m++ ) {
    for( l_n = 0; l_n < i_n; l_n++ ) {
      __m512d vc = (i_beta != 0.0) ? _mm512_mul_pd( _mm512_loadu_pd( &(o_c[l_m*i_ldC*8 + l_n*8 + 0]) ), beta ) : _mm512_setzero_pd();
      _mm512_storeu_pd(&(o_c[l_m*i_ldC*8 + l_n*8 + 0]), vc);
    }
  }

  for( l_m = 0; l_m < i_m; l_m++ ) {
    for( l_n = 0; l_n < i_n; l_n++ ) {
      __m512d vc = _mm512_loadu_pd( &(o_c[l_m*i_ldC*8 + l_n*8 + 0]) );
      for( l_k = 0; l_k < i_k; l_k++ ) {
        const __m512d alpha = _mm512_set1_pd( i_b[l_k*i_ldB + l_n] );
        vc = _mm512_fmadd_pd( alpha, _mm512_loadu_pd( &(i_a[l_m*i_ldA*8 + l_k*8 + 0]) ), vc);
      }
      _mm512_storeu_pd( &(o_c[l_m*i_ldC*8 + l_n*8 + 0]), vc );
    }
  }
}


LIBXSMM_INLINE void matMulFusedBC(  unsigned short  i_r,
                                    unsigned int    i_m,
                                    unsigned int    i_n,
                                    unsigned int    i_k,
                                    unsigned int    i_ldA,
                                    unsigned int    i_ldB,
                                    unsigned int    i_ldC,
                                    double          i_beta,
                              const double         *i_a,
                              const double         *i_b,
                                    double         *o_c ) {
  unsigned int l_m, l_n, l_k;
  const __m512d beta = _mm512_set1_pd( i_beta );
  LIBXSMM_UNUSED(i_r);
  for( l_m = 0; l_m < i_m; l_m++ ) {
    for( l_n = 0; l_n < i_n; l_n++ ) {
      __m512d vc = (i_beta != 0.0) ? _mm512_mul_pd( _mm512_loadu_pd( &(o_c[l_m*i_ldC*8 + l_n*8 + 0]) ), beta ) : _mm512_setzero_pd();
      _mm512_storeu_pd(&(o_c[l_m*i_ldC*8 + l_n*8 + 0]), vc);
    }
  }

  for( l_m = 0; l_m < i_m; l_m++ ) {
    for( l_n = 0; l_n < i_n; l_n++ ) {
      __m512d vc = _mm512_loadu_pd( &(o_c[l_m*i_ldC*8 + l_n*8 + 0]) );
      for( l_k = 0; l_k < i_k; l_k++ ) {
        const __m512d alpha = _mm512_set1_pd( i_a[l_m*i_ldA + l_k] );
        vc = _mm512_fmadd_pd( alpha, _mm512_loadu_pd( &(i_b[l_k*i_ldB*8 + l_n*8 + 0]) ), vc);
      }
      _mm512_storeu_pd( &(o_c[l_m*i_ldC*8 + l_n*8 + 0]), vc );
    }
  }
}
#endif

LIBXSMM_INLINE void amok_detect( const double* i_runtimes, size_t* io_amoks, const size_t i_workers ) {
  double time_avg;
  size_t i;
  time_avg = 0.0;
  for (i = 0; i < i_workers; i++) {
    if ( io_amoks[8*i] == 0 ) {
      time_avg += i_runtimes[8*i];
    }
  }
  time_avg = time_avg/((double)(i_workers-io_amoks[8*i_workers]));
  /* let detect amoks */
  for (i = 0; i < i_workers; i++) {
    if ( io_amoks[8*i] == 0 ) {
      if ( i_runtimes[8*i] > time_avg*1.07 ) { /* this is the amok condition */
        io_amoks[8*i_workers]++;
        io_amoks[8*i] = 1;
      }
    }
  }
}

LIBXSMM_INLINE void amok_balance( const size_t* i_amoks, const size_t i_workers, const size_t i_worksize, const size_t i_mytid, size_t* io_chunk, size_t* io_mystart, size_t* io_myend ) {
  size_t l_chunk, l_start, l_end;
  size_t l_cur_amoks = i_amoks[8*i_workers];
  size_t l_non_amoks = i_workers - l_cur_amoks;

  l_chunk = (i_worksize % l_non_amoks == 0) ? (i_worksize / l_non_amoks) : ((i_worksize / l_non_amoks) + 1);
  if (i_amoks[8*i_mytid] != 0) {
    l_start = 0;
    l_end = 0;
  } else {
    size_t l_tid_offset = 0;
    size_t l_z;
    for ( l_z = 0; l_z < i_mytid; l_z++) {
      if ( i_amoks[8*l_z] != 0 ) {
        l_tid_offset++;
      }
    }
    l_tid_offset = i_mytid - l_tid_offset;
    l_start = (l_tid_offset * l_chunk < i_worksize) ? (l_tid_offset * l_chunk) : i_worksize;
    l_end   = ((l_tid_offset+1) * l_chunk < i_worksize) ? ((l_tid_offset+1) * l_chunk) : i_worksize;
  }

  *io_chunk   = l_chunk;
  *io_mystart = l_start;
  *io_myend   = l_end;
}

int main(int argc, char* argv[])
{
  char* mat_a = 0;
  unsigned int *mat_a_rowptr, *mat_a_colidx;
  unsigned int mat_a_rowcount, mat_a_colcount, mat_a_nnz;
  double* mat_a_values;
  libxsmm_dmmfunction a_kernel;

  char* mat_b = 0;
  unsigned int *mat_b_rowptr, *mat_b_colidx;
  unsigned int mat_b_rowcount, mat_b_colcount, mat_b_nnz;
  double* mat_b_values;
  libxsmm_dmmfunction b_kernel;

  char* mat_c = 0;
  unsigned int *mat_c_rowptr, *mat_c_colidx;
  unsigned int mat_c_rowcount, mat_c_colcount, mat_c_nnz;
  double* mat_c_values;
  libxsmm_dmmfunction c_kernel;

  char* mat_st = 0;
  unsigned int *mat_st_rowptr, *mat_st_colidx;
  unsigned int mat_st_rowcount, mat_st_colcount, mat_st_nnz;
  double* mat_st_values;
  libxsmm_dmmfunction st_kernel;

  int num_modes = 9;
  int num_quants = 9;
  size_t num_elems = 0;
  size_t num_cfr = 8;
  size_t num_reps = 1;
  size_t elem_size;
  /* OpenMP: signed induction variables */
  int i, j;

  const libxsmm_gemm_descriptor *l_xgemm_desc_stiff = 0, *l_xgemm_desc_star = 0;
  libxsmm_descriptor_blob l_xgemm_blob_stiff, l_xgemm_blob_star;
  const libxsmm_gemm_prefetch_type prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  const int flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  const double alpha = 1, beta = 1;
  double flops_vol;

  double* q;
  double* qt;
  double* qs;
  double* star;
  double* global;

  unsigned long long l_start, l_end;
  double l_total;
  unsigned int l_num_threads;
  unsigned int l_star_ent = num_quants*num_quants;
  double* l_total_thread;
  double* l_cur_thread_time;
  double time_max;
  double time_min;
  double time_avg;
  size_t* amoks;

  /* read cmd */
  if ((argc > 1 && !strncmp(argv[1], "-h", 3)) || (argc != 8)) {
    printf("Usage: %s stif1 stif2 stif3 star nModes nElems nReps\n", argv[0]);
    return 0;
  }
  libxsmm_rng_set_seed(1);
  /* some empty lines at the beginning */
  printf("\n");

  i = 1;
  if (argc > (int)i) mat_a = argv[i++];
  if (argc > (int)i) mat_b = argv[i++];
  if (argc > (int)i) mat_c = argv[i++];
  if (argc > (int)i) mat_st = argv[i++];
  if (argc > (int)i) num_modes = atoi(argv[i++]);
  if (argc > (int)i) num_elems = atoi(argv[i++]);
  if (argc > (int)i) num_reps = atoi(argv[i++]);
  elem_size = num_modes*num_quants*num_cfr;

#if defined(_OPENMP)
  #pragma omp parallel
  {
    #pragma omp master
    {
      l_num_threads = omp_get_num_threads();
    }
  }
#else
  l_num_threads = 1;
#endif
  l_total_thread = (double*)malloc(8*l_num_threads*sizeof(double));
  l_cur_thread_time = (double*)malloc(8*l_num_threads*sizeof(double));
  amoks = (size_t*)malloc(8*(l_num_threads+1)*sizeof(size_t));
  for ( i = 0; i < 8*((int)l_num_threads+1); i++ ) {
    amoks[i] = 0;
  }

  /* read matrices */
  printf("reading sparse matrices... ");
  edge_sparse_csr_reader_double( mat_a, &mat_a_rowptr, &mat_a_colidx, &mat_a_values, &mat_a_rowcount, &mat_a_colcount, &mat_a_nnz );
  edge_sparse_csr_reader_double( mat_b, &mat_b_rowptr, &mat_b_colidx, &mat_b_values, &mat_b_rowcount, &mat_b_colcount, &mat_b_nnz );
  edge_sparse_csr_reader_double( mat_c, &mat_c_rowptr, &mat_c_colidx, &mat_c_values, &mat_c_rowcount, &mat_c_colcount, &mat_c_nnz );
  edge_sparse_csr_reader_double( mat_st, &mat_st_rowptr, &mat_st_colidx, &mat_st_values, &mat_st_rowcount, &mat_st_colcount, &mat_st_nnz );
  printf("done!\n\n");

  /* generate kernels */
  printf("generating code... ");
  l_xgemm_desc_stiff = libxsmm_dgemm_descriptor_init(&l_xgemm_blob_stiff,
    num_quants, num_modes, num_modes, num_modes, 0, num_modes, alpha, beta, flags, prefetch);
  l_xgemm_desc_star = libxsmm_dgemm_descriptor_init(&l_xgemm_blob_star,
    num_quants, num_modes, num_quants, 0, num_modes, num_modes, alpha, beta, flags, prefetch);
  a_kernel =  libxsmm_create_xcsr_soa( l_xgemm_desc_stiff, mat_a_rowptr,  mat_a_colidx,  (const void*)mat_a_values, (unsigned int)num_cfr ).dmm;
  b_kernel =  libxsmm_create_xcsr_soa( l_xgemm_desc_stiff, mat_b_rowptr,  mat_b_colidx,  (const void*)mat_b_values, (unsigned int)num_cfr ).dmm;
  c_kernel =  libxsmm_create_xcsr_soa( l_xgemm_desc_stiff, mat_c_rowptr,  mat_c_colidx,  (const void*)mat_c_values, (unsigned int)num_cfr ).dmm;
  st_kernel = libxsmm_create_xcsr_soa( l_xgemm_desc_star, mat_st_rowptr, mat_st_colidx, (const void*)mat_st_values, (unsigned int)num_cfr ).dmm;
  if ( a_kernel == 0 ) {
    printf("a kernel could not be built -> exit!");
    exit(-1);
  }
  if ( b_kernel == 0 ) {
    printf("b kernel could not be built -> exit!");
    exit(-1);
  }
  if ( b_kernel == 0 ) {
    printf("c kernel could not be built -> exit!");
    exit(-1);
  }
  if ( st_kernel == 0 ) {
    printf("st kernel could not be built -> exit!");
    exit(-1);
  }
  printf("done!\n\n");

  /* copying code to 1 GB page */
#if 0
#if defined(EDGE_HP_1G) || defined(EDGE_HP_2M)
  printf("copying code to 1GB page...\n");
  onegcode = (void*)edge_hp_malloc( 5*1024*1024, 2097152 );
  memcpy( onegcode,               (void*) a_kernel,  1505 );
  memcpy( onegcode+(1*1024*1024)+64, (void*) b_kernel,  2892 );
  memcpy( onegcode+(2*1024*1024)+128, (void*) c_kernel,  3249 );
  memcpy( onegcode+(3*1024*1024)+196, (void*)st_kernel, 11010 );
  a_kernel  = (libxsmm_dmmfunction)onegcode;
  b_kernel  = (libxsmm_dmmfunction)(onegcode+(1*1024*1024)+64);
  c_kernel  = (libxsmm_dmmfunction)(onegcode+(2*1024*1024)+128);
  st_kernel = (libxsmm_dmmfunction)(onegcode+(3*1024*1024)+196);
  printf("...done\n\n");
#endif
#endif

  /* create unknowns and t-unknowns */
  printf("allocating and initializing fake data... \n");
  /* DoFs */
  printf("     q: %f MiB\n", ((double)(num_elems*num_modes*num_quants*num_cfr*sizeof(double))) / ( 1024.0*1024.0) );
  q = (double*)edge_hp_malloc( num_elems*num_modes*num_quants*num_cfr*sizeof(double), 2097152);
  /* tDofs */
  printf("    qt: %f MiB\n", ((double)(num_elems*num_modes*num_quants*num_cfr*sizeof(double))) / ( 1024.0*1024.0) );
  qt = (double*)edge_hp_malloc( num_elems*num_modes*num_quants*num_cfr*sizeof(double), 2097152);
  /* star matrices */
  printf("  star: %f MiB\n", ((double)(num_elems*3*l_star_ent*sizeof(double))) / ( 1024.0*1024.0 ) );
  star = (double*)edge_hp_malloc( num_elems*3*l_star_ent*sizeof(double), 2097152);
  /* stiffness matrices */
  printf("global: %f MiB\n", ((double)(3*num_modes*num_modes*sizeof(double))) / ( 1024.0*1024 ) );
  global = (double*)edge_hp_malloc( 3*num_modes*num_modes*sizeof(double), 2097152);
  /* per thread scratch */
  printf("     t: %f MiB\n", ((double)(l_num_threads*num_modes*num_quants*num_cfr*sizeof(double)))/ ( 1024.0*1024.0) );
  qs = (double*)edge_hp_malloc( l_num_threads*num_modes*num_quants*num_cfr*sizeof(double), 2097152);

  for (i = 0; i < (int)num_elems; i++) {
    for (j = 0; j < (int)elem_size; j++) {
      q[i*elem_size + j] = libxsmm_rng_f64();
    }
  }
  for (i = 0; i < (int)num_elems; i++) {
    for (j = 0; j < (int)elem_size; j++) {
      qt[i*elem_size + j] = libxsmm_rng_f64();
    }
  }
  for (i = 0; i < (int)l_num_threads; i++) {
    for (j = 0; j < (int)elem_size; j++) {
      qs[i*elem_size + j] = libxsmm_rng_f64();
    }
  }
  for (i = 0; i < (int)num_elems; i++) {
    for (j = 0; j < (int)mat_st_nnz*3; j++) {
      star[(i*3*mat_st_nnz)+j] = libxsmm_rng_f64();
    }
  }
  for (i = 0; i < 3; i++) {
    for (j = 0; j < num_modes*num_modes; j++) {
      global[(i*num_modes*num_modes)+j] = libxsmm_rng_f64();
    }
  }
  printf("allocation done!\n\n");

  printf("running benchmark...\n");
  l_start = libxsmm_timer_tick();
#if defined(_OPENMP)
# pragma omp parallel private(i, j)
#endif
  {
#if defined(_OPENMP)
    int mytid = omp_get_thread_num();
#else
    int mytid = 0;
#endif
    libxsmm_timer_tickint mystart, myend;
#if defined(HANDLE_AMOK)
    size_t cur_amoks = 0;
    size_t non_amoks = l_num_threads;
#endif
    size_t l_el_chunk = 0;
    size_t l_el_start = 0;
    size_t l_el_end   = 0;
    /* initial work distribution */
    amok_balance( amoks, l_num_threads, num_elems, mytid, &l_el_chunk, &l_el_start, &l_el_end );
    for (i = 0; i < (int)num_reps; i++) {
#if defined(HANDLE_AMOK)
      /* did we had an amok? */
      if (cur_amoks != amoks[8*l_num_threads]) {
        cur_amoks = amoks[8*l_num_threads];
        non_amoks = l_num_threads - cur_amoks;
        /* re-balance work */
        amok_balance( amoks, l_num_threads, num_elems, mytid, &l_el_chunk, &l_el_start, &l_el_end );
      }
#endif
      mystart = libxsmm_timer_tick();
      for (j = (int)l_el_start; j < (int)l_el_end; j++) {
#if 1
        st_kernel( star+(j*3*mat_st_nnz)               , qt+(j*elem_size), qs+(mytid*elem_size) );
        a_kernel( qs+(mytid*elem_size), global                        , q+(j*elem_size) );

        st_kernel( star+(j*3*mat_st_nnz)+mat_st_nnz    , qt+(j*elem_size), qs+(mytid*elem_size) );
        b_kernel( qs+(mytid*elem_size), global+(num_modes*num_modes)  , q+(j*elem_size) );

        st_kernel( star+(j*3*mat_st_nnz)+(2*mat_st_nnz), qt+(j*elem_size), qs+(mytid*elem_size) );
        c_kernel( qs+(mytid*elem_size), global+(2*num_modes*num_modes), q+(j*elem_size) );
#else
        matMulFusedBC( 8, num_quants, num_modes, num_quants, num_quants, num_modes, num_modes, 1.0, star+(j*3*mat_st_nnz), qt+(j*elem_size), qs+(mytid*elem_size) );
        matMulFusedAC( 8, num_quants, num_modes, num_modes,  num_modes,  num_modes, num_modes, 1.0, qs+(mytid*elem_size), global, q+(j*elem_size) );

        matMulFusedBC( 8, num_quants, num_modes, num_quants, num_quants, num_modes, num_modes, 1.0, star+(j*3*mat_st_nnz)+mat_st_nnz, qt+(j*elem_size), qs+(mytid*elem_size) );
        matMulFusedAC( 8, num_quants, num_modes, num_modes,  num_modes,  num_modes, num_modes, 1.0, qs+(mytid*elem_size), global+(num_modes*num_modes)  , q+(j*elem_size) );

        matMulFusedBC( 8, num_quants, num_modes, num_quants, num_quants, num_modes, num_modes, 1.0, star+(j*3*mat_st_nnz)+(2*mat_st_nnz), qt+(j*elem_size), qs+(mytid*elem_size) );
        matMulFusedAC( 8, num_quants, num_modes, num_modes,  num_modes,  num_modes, num_modes, 1.0, qs+(mytid*elem_size), global+(2*num_modes*num_modes), q+(j*elem_size) );

#endif
      }
      myend = libxsmm_timer_tick();
      l_cur_thread_time[8*mytid] = libxsmm_timer_duration( mystart, myend );
      l_total_thread[8*mytid] += libxsmm_timer_duration( mystart, myend );
#if defined(_OPENMP)
      #pragma omp barrier
#endif
#if defined(HANDLE_AMOK)
      /* checking for amoks is centralized business */
      if (mytid == 0) {
        /* amok check */
        amok_detect( l_cur_thread_time, amoks, l_num_threads );
      }
#if defined(_OPENMP)
      #pragma omp barrier
#endif
#endif
    }
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("...done!\n\n");
  /* some timing stats */
  time_max = 0.0;
  time_min = 80000000;
  time_avg = 0.0;
  for (i = 0; i < (int)l_num_threads; i++) {
    if( amoks[8*i] == 0 ) {
      if( l_total_thread[8*i] > time_max) time_max = l_total_thread[8*i];
      if( l_total_thread[8*i] < time_min) time_min = l_total_thread[8*i];
      time_avg += l_total_thread[8*i];
    }
  }
  time_avg = time_avg/((double)(l_num_threads-amoks[8*l_num_threads]));

  flops_vol  = (double)num_quants * (double)mat_a_nnz * (double)num_cfr * 2.0;
  flops_vol += (double)num_quants * (double)mat_b_nnz * (double)num_cfr * 2.0;
  flops_vol += (double)num_quants * (double)mat_c_nnz * (double)num_cfr * 2.0;
  flops_vol += (double)num_modes * (double)mat_st_nnz * (double)num_cfr * 6.0; /* 3 star matrix mul */
  printf("%fs time for vol (asm), min %f, max %f, avg %f, #amoks %llu, amok-threads ", l_total, time_min, time_max, time_avg, (unsigned long long)amoks[8*l_num_threads]);
  for ( i = 0; i < (int)l_num_threads; i++ ) {
    if ( amoks[8*i] != 0 ) {
      printf("%i,", i);
    }
  }
  printf("\n");
  printf("%f GFLOPS for vol (asm)\n", ((double)num_elems * (double)num_reps * flops_vol) / (l_total * 1.0e9));
  printf("%f GiB/s for vol (asm)\n", (double)((double)num_elems * (double)elem_size * 8.0 * 3.0 * (double)num_reps) / (l_total * 1024.0*1024.0*1024.0) );
  printf("done!\n\n");

  /* some empty lines at the end */
  printf("\n\n");

  return 0;
}

