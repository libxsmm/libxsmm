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
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <libxsmm.h>

#if defined(_OPENMP)
# include <omp.h>
#endif

#include <edge_proxy_common.h>

#if defined(_WIN32) || defined(__CYGWIN__)
/* note: later on, this leads to (correct but) different than expected norm-values */
# define drand48() ((double)rand() / RAND_MAX)
# define srand48 srand
#endif

int main(int argc, char* argv[])
{
  char* mat_a;
  unsigned int *mat_a_rowptr, *mat_a_colidx;
  unsigned int mat_a_rowcount, mat_a_colcount, mat_a_nnz;
  double* mat_a_values;
  libxsmm_dmmfunction a_kernel;

  char* mat_b;
  unsigned int *mat_b_rowptr, *mat_b_colidx;
  unsigned int mat_b_rowcount, mat_b_colcount, mat_b_nnz;
  double* mat_b_values;
  libxsmm_dmmfunction b_kernel;

  char* mat_c;
  unsigned int *mat_c_rowptr, *mat_c_colidx;
  unsigned int mat_c_rowcount, mat_c_colcount, mat_c_nnz;
  double* mat_c_values;
  libxsmm_dmmfunction c_kernel;

  char* mat_st;
  unsigned int *mat_st_rowptr, *mat_st_colidx;
  unsigned int mat_st_rowcount, mat_st_colcount, mat_st_nnz;
  double* mat_st_values;
  libxsmm_dmmfunction st_kernel;

  size_t num_elems;
  size_t num_modes;
  size_t num_quants = 9;
  size_t num_cfr = 8;
  size_t num_reps;
  size_t elem_size;
  size_t i, j;

  libxsmm_gemm_descriptor l_xgemm_desc_stiff;
  libxsmm_gemm_descriptor l_xgemm_desc_star;

  double* q;
  double* qt;

  unsigned long long l_start, l_end;
  double l_total;

  /* read cmd */
  if ((argc > 1 && !strncmp(argv[1], "-h", 3)) || (argc != 8)) {
    printf("Usage: %s stif1 stif2 stif3 star nModes nElems nReps\n", argv[0]);
    return 0;
  }
  srand48(1);
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

  /* read matrices */
  printf("reading sparse matrices... ");
  edge_sparse_csr_reader_double( mat_a, &mat_a_rowptr, &mat_a_colidx, &mat_a_values, &mat_a_rowcount, &mat_a_colcount, &mat_a_nnz );
  edge_sparse_csr_reader_double( mat_b, &mat_b_rowptr, &mat_b_colidx, &mat_b_values, &mat_b_rowcount, &mat_b_colcount, &mat_b_nnz );
  edge_sparse_csr_reader_double( mat_c, &mat_c_rowptr, &mat_c_colidx, &mat_c_values, &mat_c_rowcount, &mat_c_colcount, &mat_c_nnz );
  edge_sparse_csr_reader_double( mat_st, &mat_st_rowptr, &mat_st_colidx, &mat_st_values, &mat_st_rowcount, &mat_st_colcount, &mat_st_nnz );
  printf("done!\n\n");

  /* generate kernels */
  printf("generating code... ");
  LIBXSMM_GEMM_DESCRIPTOR(l_xgemm_desc_stiff, LIBXSMM_GEMM_PRECISION_F64, 0/*flags*/, num_quants, num_modes, num_modes,  num_modes, 0, num_modes, 1.0, 1.0, LIBXSMM_PREFETCH_NONE);
  LIBXSMM_GEMM_DESCRIPTOR(l_xgemm_desc_star,  LIBXSMM_GEMM_PRECISION_F64, 0/*flags*/, num_quants, num_modes, num_quants, 0, num_modes, num_modes, 1.0, 1.0, LIBXSMM_PREFETCH_NONE);
  a_kernel = libxsmm_create_xcsr_soa( &l_xgemm_desc_stiff, mat_a_rowptr, mat_a_colidx, (const void*)mat_a_values ).dmm;
  b_kernel = libxsmm_create_xcsr_soa( &l_xgemm_desc_stiff, mat_b_rowptr, mat_b_colidx, (const void*)mat_b_values ).dmm;
  c_kernel = libxsmm_create_xcsr_soa( &l_xgemm_desc_stiff, mat_c_rowptr, mat_c_colidx, (const void*)mat_c_values ).dmm;
  st_kernel = libxsmm_create_xcsr_soa( &l_xgemm_desc_star, mat_st_rowptr, mat_st_colidx, (const void*)mat_st_values ).dmm;
  if ( a_kernel == 0 || b_kernel == 0 || c_kernel == 0 || st_kernel == 0 ) {
    printf("one of the kernels could not be built -> exit!");
    exit(-1);
  }
  printf("done!\n\n");

  /* create unkowns and tunkowns */
  printf("allocating and initializing fake data... \n");
  printf("   q: %f MiB\n", ((double)(num_elems*num_modes*num_quants*num_cfr*sizeof(double)))/ ( 1024.0*1024.0) );
  printf("  qt: %f MiB\n", ((double)(num_elems*num_modes*num_quants*num_cfr*sizeof(double)))/ ( 1024.0*1024.0) );
#ifdef _OPENMP
  printf("   t: %f MiB\n", ((double)(omp_get_max_threads()*num_modes*num_quants*num_cfr*sizeof(double)))/ ( 1024.0*1024.0) );
#else
  printf("   t: %f MiB\n", ((double)(num_modes*num_quants*num_cfr*sizeof(double)))/ ( 1024.0*1024.0) );
#endif
  q = (double*)libxsmm_aligned_malloc( num_elems*num_modes*num_quants*num_cfr*sizeof(double), 2097152);
  qt = (double*)libxsmm_aligned_malloc( num_elems*num_modes*num_quants*num_cfr*sizeof(double), 2097152);

  #pragma omp parallel for private(i,j)
  for ( i = 0; i < num_elems; i++ ) {
    for ( j = 0; j < elem_size; j++) {
      q[i*elem_size + j] = drand48();
    }
  }
  #pragma omp parallel for private(i,j)
  for ( i = 0; i < num_elems; i++ ) {
    for ( j = 0; j < elem_size; j++) {
      qt[i*elem_size + j] = drand48();
    }
  }

  printf("done!\n\n");

  /* benchmark single core all kernels */
  printf("benchmarking kernels... \n");
  l_start = libxsmm_timer_tick();
  for ( i = 0; i < num_reps; i++) {
    a_kernel( qt, mat_a_values, q );
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("%fs for stiff1 (asm)\n", l_total);
  printf("%f GFLOPS for stiff1 (asm)\n", ((double)((double)num_reps * (double)num_quants * (double)mat_a_nnz * (double)num_cfr) * 2.0) / (l_total * 1.0e9));

  l_start = libxsmm_timer_tick();
  for ( i = 0; i < num_reps; i++) {
    b_kernel( qt, mat_b_values, q );
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("%fs for stiff2 (asm)\n", l_total);
  printf("%f GFLOPS for stiff2 (asm)\n", ((double)((double)num_reps * (double)num_quants * (double)mat_b_nnz * (double)num_cfr) * 2.0) / (l_total * 1.0e9));

  l_start = libxsmm_timer_tick();
  for ( i = 0; i < num_reps; i++) {
    c_kernel( qt, mat_c_values, q );
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("%fs for stiff3 (asm)\n", l_total);
  printf("%f GFLOPS for stiff3 (asm)\n", ((double)((double)num_reps * (double)num_quants * (double)mat_c_nnz * (double)num_cfr) * 2.0) / (l_total * 1.0e9));

  l_start = libxsmm_timer_tick();
  for ( i = 0; i < num_reps; i++) {
    st_kernel( mat_st_values, qt, q );
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("%fs for star (asm)\n", l_total);
  printf("%f GFLOPS for star (asm)\n", ((double)((double)num_reps * (double)num_modes * (double)mat_st_nnz * (double)num_cfr) * 2.0) / (l_total * 1.0e9));
  printf("done!\n\n");

  /* benchmark volumne integration */
  #pragma omp parallel for private(i,j)
  for ( i = 0; i < num_elems; i++ ) {
    for ( j = 0; j < elem_size; j++) {
      q[i*elem_size + j] = drand48();
    }
  }
  #pragma omp parallel for private(i,j)
  for ( i = 0; i < num_elems; i++ ) {
    for ( j = 0; j < elem_size; j++) {
      qt[i*elem_size + j] = drand48();
    }
  }

  l_start = libxsmm_timer_tick();
  for ( i = 0; i < num_reps; i++) {
    #pragma omp parallel private(i, j)
    {
      __attribute__((aligned(64))) double tp[20*8*9];

      #pragma omp for private(j)
      for ( j = 0; j < num_elems; j++ ) {
        st_kernel( mat_st_values, qt+(j*elem_size), tp );
        a_kernel( tp, mat_a_values, q+(j*elem_size) );

        st_kernel( mat_st_values, qt+(j*elem_size), tp );
        b_kernel( tp, mat_b_values, q+(j*elem_size) );

        st_kernel( mat_st_values, qt+(j*elem_size), tp );
        c_kernel( tp, mat_c_values, q+(j*elem_size) );
      }
    }
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("%fs for vol (asm)\n", l_total);
  printf("%f GFLOPS for vol (asm)\n", ((double)((double)num_elems * (double)num_reps * 3.0 * ((double)num_quants + (double)num_modes) * (double)mat_st_nnz * (double)num_cfr) * 2.0) / (l_total * 1.0e9));
  printf("%f GiB/s for vol (asm)\n", (double)((double)num_elems * (double)elem_size * 8.0 * 3.0 * (double)num_reps) / (l_total * 1024.0*1024.0*1024.0) );
  printf("done!\n\n");



  /* some empty lines at the end */
  printf("\n\n");

  return 0;
}

