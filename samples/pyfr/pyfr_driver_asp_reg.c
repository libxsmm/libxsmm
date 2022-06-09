/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>

#define REALTYPE double
#define REPS 100

#if !defined(FSSPMDM)
#define FSSPMDM LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_TPREFIX(REALTYPE, fsspmdm))
#endif

#if !defined(GEMM)
# if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
#   include <mkl.h>
# else
LIBXSMM_BLAS_SYMBOL_DECL(REALTYPE, gemm)
# endif
# define GEMM LIBXSMM_GEMM_SYMBOL(REALTYPE)
#endif


int my_csr_reader( const char*           i_csr_file_in,
                   unsigned int**        o_row_idx,
                   unsigned int**        o_column_idx,
                   REALTYPE**            o_values,
                   unsigned int*         o_row_count,
                   unsigned int*         o_column_count,
                   unsigned int*         o_element_count ) {
  FILE *l_csr_file_handle;
  const unsigned int l_line_length = 512;
  char l_line[512/*l_line_length*/+1];
  unsigned int l_header_read = 0;
  unsigned int* l_row_idx_id = NULL;
  unsigned int l_i = 0;

  l_csr_file_handle = fopen( i_csr_file_in, "r" );
  if ( l_csr_file_handle == NULL ) {
    fprintf( stderr, "cannot open CSR file!\n" );
    return -1;
  }

  while (fgets(l_line, l_line_length, l_csr_file_handle) != NULL) {
    if ( strlen(l_line) == l_line_length ) {
      fprintf( stderr, "could not read file length!\n" );
      return -1;
    }
    /* check if we are still reading comments header */
    if ( l_line[0] == '%' ) {
      continue;
    } else {
      /* if we are the first line after comment header, we allocate our data structures */
      if ( l_header_read == 0 ) {
        if (3 == sscanf(l_line, "%u %u %u", o_row_count, o_column_count, o_element_count) &&
            0 != *o_row_count && 0 != *o_column_count && 0 != *o_element_count)
        {
          /* allocate CSC datastructure matching mtx file */
          *o_column_idx = (unsigned int*) malloc(sizeof(unsigned int) * ((size_t)*o_element_count));
          *o_row_idx = (unsigned int*) malloc(sizeof(unsigned int) * ((size_t)*o_row_count + 1));
          *o_values = (REALTYPE*) malloc(sizeof(REALTYPE) * ((size_t)*o_element_count));
          l_row_idx_id = (unsigned int*) malloc(sizeof(unsigned int) * ((size_t)*o_row_count));

          /* check if mallocs were successful */
          if ( ( *o_row_idx == NULL )      ||
               ( *o_column_idx == NULL )   ||
               ( *o_values == NULL )       ||
               ( l_row_idx_id == NULL ) ) {
            fprintf( stderr, "could not allocate sp data!\n" );
            return -1;
          }

          /* set everything to zero for init */
          memset(*o_row_idx, 0, sizeof(unsigned int)*((size_t)*o_row_count + 1));
          memset(*o_column_idx, 0, sizeof(unsigned int)*((size_t)*o_element_count));
          memset(*o_values, 0, sizeof(REALTYPE)*((size_t)*o_element_count));
          memset(l_row_idx_id, 0, sizeof(unsigned int)*((size_t)*o_row_count));

          /* init column idx */
          for ( l_i = 0; l_i < (*o_row_count + 1); l_i++)
            (*o_row_idx)[l_i] = (*o_element_count);

          /* init */
          (*o_row_idx)[0] = 0;
          l_i = 0;
          l_header_read = 1;
        } else {
          fprintf( stderr, "could not csr description!\n" );
          return -1;
        }
      /* now we read the actual content */
      } else {
        unsigned int l_row, l_column;
        double l_value;
        /* read a line of content */
        if ( sscanf(l_line, "%u %u %lf", &l_row, &l_column, &l_value) != 3 ) {
          fprintf( stderr, "could not read element!\n" );
          return -1;
        }
        /* adjust numbers to zero termination */
        l_row--;
        l_column--;
        /* add these values to row and value structure */
        (*o_column_idx)[l_i] = l_column;
        (*o_values)[l_i] = (REALTYPE)l_value;
        l_i++;
        /* handle columns, set id to own for this column, yeah we need to handle empty columns */
        l_row_idx_id[l_row] = 1;
        (*o_row_idx)[l_row+1] = l_i;
      }
    }
  }

  /* close mtx file */
  fclose( l_csr_file_handle );

  /* check if we read a file which was consistent */
  if ( l_i != (*o_element_count) ) {
    fprintf( stderr, "we were not able to read all elements!\n" );
    return -1;
  }

  /* let's handle empty rows */
  for ( l_i = 0; l_i < (*o_row_count); l_i++) {
    assert(NULL != l_row_idx_id);
    if ( l_row_idx_id[l_i] == 0 ) {
      (*o_row_idx)[l_i+1] = (*o_row_idx)[l_i];
    }
  }

  /* free helper data structure */
  if ( l_row_idx_id != NULL ) {
    free( l_row_idx_id );
  }
  return 0;
}

int main(int argc, char* argv[]) {
  int ret = 0;

  char* l_csr_file;
  REALTYPE* l_a_sp;
  unsigned int* l_rowptr;
  unsigned int* l_colidx;
  unsigned int l_rowcount, l_colcount, l_elements;

  REALTYPE* l_a_dense;
  REALTYPE* l_b;
  REALTYPE* l_c_betaone;
  REALTYPE* l_c_betazero;
  REALTYPE* l_c_gold_betaone;
  REALTYPE* l_c_gold_betazero;
  REALTYPE* l_c_dense_betaone;
  REALTYPE* l_c_dense_betazero;
  libxsmm_matdiff_info diff;
  int l_m;
  int l_n;
  int l_k;

  int l_i;
  int l_j;
  int l_z;
  int l_elems;
  int l_reps;
  int l_n_block;

  libxsmm_timer_tickint l_start, l_end;
  double l_total;

  REALTYPE alpha = 1.0;
  REALTYPE beta = 1.0;
  char trans = 'N';

  FSSPMDM* gemm_op_betazero = NULL;
  FSSPMDM* gemm_op_betaone = NULL;

  if (argc != 4) {
    fprintf( stderr, "need csr-filename N reps!\n" );
    exit(-1);
  }


  /* read sparse A */
  l_csr_file = argv[1];
  l_n = atoi(argv[2]);
  l_reps = atoi(argv[3]);
  if (my_csr_reader(  l_csr_file,
                 &l_rowptr,
                 &l_colidx,
                 &l_a_sp,
                 &l_rowcount, &l_colcount, &l_elements ) != 0 )
  {
    exit(-1);
  }
  l_m = l_rowcount;
  l_k = l_colcount;
  printf("CSR matrix data structure we just read:\n");
  printf("rows: %u, columns: %u, elements: %u\n", l_rowcount, l_colcount, l_elements);

  /* allocate dense matrices */
  l_a_dense = (REALTYPE*)libxsmm_aligned_malloc(sizeof(REALTYPE) * l_k * l_m, 64);
  l_b = (REALTYPE*)libxsmm_aligned_malloc(sizeof(REALTYPE) * l_k * l_n, 64);
  l_c_betazero = (REALTYPE*)libxsmm_aligned_malloc(sizeof(REALTYPE) * l_m * l_n, 64);
  l_c_betaone = (REALTYPE*)libxsmm_aligned_malloc(sizeof(REALTYPE) * l_m * l_n, 64);
  l_c_gold_betazero = (REALTYPE*)libxsmm_aligned_malloc(sizeof(REALTYPE) * l_m * l_n, 64);
  l_c_gold_betaone = (REALTYPE*)libxsmm_aligned_malloc(sizeof(REALTYPE) * l_m * l_n, 64);
  l_c_dense_betazero = (REALTYPE*)libxsmm_aligned_malloc(sizeof(REALTYPE) * l_m * l_n, 64);
  l_c_dense_betaone = (REALTYPE*)libxsmm_aligned_malloc(sizeof(REALTYPE) * l_m * l_n, 64);

  /* touch B */
  for ( l_i = 0; l_i < l_k*l_n; l_i++) {
    l_b[l_i] = (REALTYPE)libxsmm_rng_f64();
  }

  /* touch dense A */
  for ( l_i = 0; l_i < l_k*l_m; l_i++) {
    l_a_dense[l_i] = 0;
  }
  /* init dense A using sparse A */
  for ( l_i = 0; l_i < l_m; l_i++ ) {
    l_elems = l_rowptr[l_i+1] - l_rowptr[l_i];
    for ( l_z = 0; l_z < l_elems; l_z++ ) {
      l_a_dense[(l_i*l_k)+l_colidx[l_rowptr[l_i]+l_z]] = l_a_sp[l_rowptr[l_i]+l_z];
    }
  }

  /* touch C */
  for ( l_i = 0; l_i < l_m*l_n; l_i++) {
    l_c_gold_betaone[l_i] = (REALTYPE)libxsmm_rng_f64();
  }
  for ( l_i = 0; l_i < l_m*l_n; l_i++) {
    l_c_betaone[l_i] = l_c_gold_betaone[l_i];
  }
  for ( l_i = 0; l_i < l_m*l_n; l_i++) {
    l_c_dense_betaone[l_i] = l_c_gold_betaone[l_i];
  }
  for ( l_i = 0; l_i < l_m*l_n; l_i++) {
    l_c_betazero[l_i] = l_c_betaone[l_i];
  }
  for ( l_i = 0; l_i < l_m*l_n; l_i++) {
    l_c_gold_betazero[l_i] = l_c_gold_betaone[l_i];
  }
  for ( l_i = 0; l_i < l_m*l_n; l_i++) {
    l_c_dense_betazero[l_i] = l_c_dense_betaone[l_i];
  }

  /* setting up fsspmdm */
  l_n_block = 48;
  beta = 0.0;
  gemm_op_betazero = LIBXSMM_CONCATENATE(FSSPMDM,_create)( l_m, l_n_block, l_k, l_k, l_n, l_n, 1.0, beta, 1, l_a_dense );
  beta = 1.0;
  gemm_op_betaone = LIBXSMM_CONCATENATE(FSSPMDM,_create)( l_m, l_n_block, l_k, l_k, l_n, l_n, 1.0, beta, 0, l_a_dense );

  /* compute golden results */
  printf("computing golden solution...\n");
  for ( l_j = 0; l_j < l_n; l_j++ ) {
    for (l_i = 0; l_i < l_m; l_i++ ) {
      l_elems = l_rowptr[l_i+1] - l_rowptr[l_i];
      l_c_gold_betazero[(l_n*l_i) + l_j] = 0.0;
      for (l_z = 0; l_z < l_elems; l_z++) {
        l_c_gold_betazero[(l_n*l_i) + l_j] +=  l_a_sp[l_rowptr[l_i]+l_z] * l_b[(l_n*l_colidx[l_rowptr[l_i]+l_z])+l_j];
      }
    }
  }
  for ( l_j = 0; l_j < l_n; l_j++ ) {
    for (l_i = 0; l_i < l_m; l_i++ ) {
      l_elems = l_rowptr[l_i+1] - l_rowptr[l_i];
      for (l_z = 0; l_z < l_elems; l_z++) {
        l_c_gold_betaone[(l_n*l_i) + l_j] +=  l_a_sp[l_rowptr[l_i]+l_z] * l_b[(l_n*l_colidx[l_rowptr[l_i]+l_z])+l_j];
      }
    }
  }
  printf("...done!\n");

  /* libxsmm generated code */
  printf("computing libxsmm (A sparse) solution...\n");
#if defined(_OPENMP)
  #pragma omp parallel for private(l_z)
#endif
  for (l_z = 0; l_z < l_n; l_z+=l_n_block) {
    LIBXSMM_CONCATENATE(FSSPMDM,_execute)( gemm_op_betazero, l_b+l_z, l_c_betazero+l_z );
  }
#if defined(_OPENMP)
  #pragma omp parallel for private(l_z)
#endif
  for (l_z = 0; l_z < l_n; l_z+=l_n_block) {
    LIBXSMM_CONCATENATE(FSSPMDM,_execute)( gemm_op_betaone, l_b+l_z, l_c_betaone+l_z );
  }
  printf("...done!\n");

  /* BLAS code */
  printf("computing BLAS (A dense) solution...\n");
  beta = 0.0;
  GEMM( &trans, &trans, &l_n, &l_m, &l_k, &alpha, l_b, &l_n, l_a_dense, &l_k, &beta, l_c_dense_betazero, &l_n );
  beta = 1.0;
  GEMM( &trans, &trans, &l_n, &l_m, &l_k, &alpha, l_b, &l_n, l_a_dense, &l_k, &beta, l_c_dense_betaone, &l_n );
  printf("...done!\n");

  /* check for errors */
  libxsmm_matdiff_clear(&diff);
  libxsmm_matdiff(&diff, LIBXSMM_DATATYPE(REALTYPE), l_m, l_n,
    l_c_gold_betazero, l_c_betazero, NULL/*ldref*/, NULL/*ldtst*/);
  ret |= diff.linf_abs > 1e-4;
  printf("max error beta=0 (libxmm vs. gold): %f (%f != %f)\n", diff.linf_abs, diff.v_ref, diff.v_tst);

  libxsmm_matdiff_clear(&diff);
  libxsmm_matdiff(&diff, LIBXSMM_DATATYPE(REALTYPE), l_m, l_n,
    l_c_gold_betaone, l_c_betaone, NULL/*ldref*/, NULL/*ldtst*/);
  ret |= diff.linf_abs > 1e-4;
  printf("max error beta=1 (libxmm vs. gold): %f (%f != %f)\n", diff.linf_abs, diff.v_ref, diff.v_tst);

  libxsmm_matdiff_clear(&diff);
  libxsmm_matdiff(&diff, LIBXSMM_DATATYPE(REALTYPE), l_m, l_n,
    l_c_gold_betazero, l_c_dense_betazero, NULL/*ldref*/, NULL/*ldtst*/);
  printf("max error beta=0 (dense vs. gold): %f (%f != %f)\n", diff.linf_abs, diff.v_ref, diff.v_tst);

  libxsmm_matdiff_clear(&diff);
  libxsmm_matdiff(&diff, LIBXSMM_DATATYPE(REALTYPE), l_m, l_n,
    l_c_gold_betaone, l_c_dense_betaone, NULL/*ldref*/, NULL/*ldtst*/);
  printf("max error beta=1 (dense vs. gold): %f (%f != %f)\n", diff.linf_abs, diff.v_ref, diff.v_tst);

  /* Let's measure performance */
  l_start = libxsmm_timer_tick();
  for ( l_j = 0; l_j < l_reps; l_j++ ) {
#if defined(_OPENMP)
    #pragma omp parallel for private(l_z)
#endif
    for (l_z = 0; l_z < l_n; l_z+=l_n_block) {
      LIBXSMM_CONCATENATE(FSSPMDM,_execute)( gemm_op_betazero, l_b+l_z, l_c_betazero+l_z );
    }
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  fprintf(stdout, "time[s] LIBXSMM (RM, M=%i, N=%i, K=%i, beta=0): %f\n", l_m, l_n, l_k, l_total/(double)l_reps );
  fprintf(stdout, "GFLOPS  LIBXSMM (RM, M=%i, N=%i, K=%i, beta=0): %f (sparse)\n", l_m, l_n, l_k, (2.0 * (double)l_elements * (double)l_n * (double)l_reps * 1.0e-9) / l_total );
  fprintf(stdout, "GFLOPS  LIBXSMM (RM, M=%i, N=%i, K=%i, beta=0): %f (dense)\n", l_m, l_n, l_k, (2.0 * (double)l_m * (double)l_n * (double)l_k * (double)l_reps * 1.0e-9) / l_total );
  fprintf(stdout, "GB/s    LIBXSMM (RM, M=%i, N=%i, K=%i, beta=0): %f\n", l_m, l_n, l_k, ((double)sizeof(REALTYPE) * (((double)l_m * (double)l_n) + ((double)l_k * (double)l_n)) * (double)l_reps * 1.0e-9) / l_total );

  l_start = libxsmm_timer_tick();
  for ( l_j = 0; l_j < l_reps; l_j++ ) {
#if defined(_OPENMP)
    #pragma omp parallel for private(l_z)
#endif
    for (l_z = 0; l_z < l_n; l_z+=l_n_block) {
      LIBXSMM_CONCATENATE(FSSPMDM,_execute)( gemm_op_betaone, l_b+l_z, l_c_betaone+l_z );
    }
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  fprintf(stdout, "time[s] LIBXSMM (RM, M=%i, N=%i, K=%i, beta=1): %f\n", l_m, l_n, l_k, l_total/(double)l_reps );
  fprintf(stdout, "GFLOPS  LIBXSMM (RM, M=%i, N=%i, K=%i, beta=1): %f (sparse)\n", l_m, l_n, l_k, (2.0 * (double)l_elements * (double)l_n * (double)l_reps * 1.0e-9) / l_total );
  fprintf(stdout, "GFLOPS  LIBXSMM (RM, M=%i, N=%i, K=%i, beta=1): %f (dense)\n", l_m, l_n, l_k, (2.0 * (double)l_m * (double)l_n * (double)l_k * (double)l_reps * 1.0e-9) / l_total );
  fprintf(stdout, "GB/s    LIBXSMM (RM, M=%i, N=%i, K=%i, beta=1): %f\n", l_m, l_n, l_k, ((double)sizeof(REALTYPE) * ((2.0*(double)l_m * (double)l_n) + ((double)l_k * (double)l_n)) * (double)l_reps * 1.0e-9) / l_total );

  l_start = libxsmm_timer_tick();
  beta = 0.0;
  for ( l_j = 0; l_j < l_reps; l_j++ ) {
    GEMM( &trans, &trans, &l_n, &l_m, &l_k, &alpha, l_b, &l_n, l_a_dense, &l_k, &beta, l_c_dense_betazero, &l_n );
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  fprintf(stdout, "time[s] BLAS    (RM, M=%i, N=%i, K=%i, beta=0): %f\n", l_m, l_n, l_k, l_total/(double)l_reps );
  fprintf(stdout, "GFLOPS  BLAS    (RM, M=%i, N=%i, K=%i, beta=0): %f\n", l_m, l_n, l_k, (2.0 * (double)l_m * (double)l_n * (double)l_k * (double)l_reps * 1.0e-9) / l_total );
  fprintf(stdout, "GB/s    BLAS    (RM, M=%i, N=%i, K=%i, beta=0): %f\n", l_m, l_n, l_k, ((double)sizeof(REALTYPE) * ((2.0*(double)l_m * (double)l_n) + ((double)l_k * (double)l_n)) * (double)l_reps * 1.0e-9) / l_total );

  l_start = libxsmm_timer_tick();
  beta = 1.0;
  for ( l_j = 0; l_j < l_reps; l_j++ ) {
    GEMM( &trans, &trans, &l_n, &l_m, &l_k, &alpha, l_b, &l_n, l_a_dense, &l_k, &beta, l_c_dense_betaone, &l_n );
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  fprintf(stdout, "time[s] BLAS    (RM, M=%i, N=%i, K=%i, beta=1): %f\n", l_m, l_n, l_k, l_total/(double)l_reps );
  fprintf(stdout, "GFLOPS  BLAS    (RM, M=%i, N=%i, K=%i, beta=1): %f\n", l_m, l_n, l_k, (2.0 * (double)l_m * (double)l_n * (double)l_k * (double)l_reps * 1.0e-9) / l_total );
  fprintf(stdout, "GB/s    BLAS    (RM, M=%i, N=%i, K=%i, beta=1): %f\n", l_m, l_n, l_k, ((double)sizeof(REALTYPE) * ((2.0*(double)l_m * (double)l_n) + ((double)l_k * (double)l_n)) * (double)l_reps * 1.0e-9) / l_total );

  /* free */
  LIBXSMM_CONCATENATE(FSSPMDM,_destroy)( gemm_op_betazero );
  LIBXSMM_CONCATENATE(FSSPMDM,_destroy)( gemm_op_betaone );

  return ret;
}
