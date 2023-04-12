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
#include <utils/libxsmm_utils.h>
#include <libxsmm.h>

#if !defined(REALTYPE)
# define REALTYPE double
#endif

#define EPSILON(T) LIBXSMM_CONCATENATE(EPSILON_, T)
#define EPSILON_double 1e-8
#define EPSILON_float 1e-4

#if !defined(GEMM)
# if (defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)) && \
     (defined(LIBXSMM_PLATFORM_X86))
#   include <mkl.h>
# else
LIBXSMM_BLAS_SYMBOL_DECL(REALTYPE, gemm)
# endif
# define GEMM LIBXSMM_GEMM_SYMBOL(REALTYPE)
#endif


LIBXSMM_INLINE int my_csr_reader(const char* i_csr_file_in,
  unsigned int** o_row_idx, unsigned int** o_column_idx, REALTYPE** o_values,
  unsigned int* o_row_count, unsigned int* o_column_count, unsigned int* o_element_count)
{
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
          /* allocate CSC datastructure matching mtx file, and set everything to zero */
          *o_column_idx = (unsigned int*) calloc(*o_element_count, sizeof(unsigned int));
          *o_row_idx = (unsigned int*)calloc((size_t)*o_row_count + 1, sizeof(unsigned int));
          *o_values = (REALTYPE*) calloc(*o_element_count, sizeof(REALTYPE));
          l_row_idx_id = (unsigned int*) calloc(*o_row_count, sizeof(unsigned int));

          /* check if mallocs were successful */
          if ( ( *o_row_idx == NULL )      ||
               ( *o_column_idx == NULL )   ||
               ( *o_values == NULL )       ||
               ( l_row_idx_id == NULL ) ) {
            fprintf( stderr, "could not allocate sp data!\n" );
            return -1;
          }

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
  free( l_row_idx_id );

  return 0;
}

int main(int argc, char* argv[]) {
  int ret = 0;

  const char* l_csr_file = NULL;
  unsigned int* l_rowptr = NULL;
  unsigned int* l_colidx = NULL;
  unsigned int l_rowcount, l_colcount, l_elements;

  REALTYPE* l_a_dense = NULL;
  REALTYPE* l_a_sp = NULL;
  REALTYPE* l_b = NULL;

  REALTYPE *l_c_betazero = NULL, *l_c_gold_betazero = NULL, *l_c_dense_betazero = NULL;
  REALTYPE *l_c_betaone = NULL, *l_c_gold_betaone = NULL, *l_c_dense_betaone = NULL;

  const char *const env_fsspmdm_nblock = getenv("FSSPMDM_NBLOCK");
  int l_n_block = ((NULL == env_fsspmdm_nblock || '\0' == *env_fsspmdm_nblock)
    ? 48 : atoi(env_fsspmdm_nblock));

  libxsmm_matdiff_info diff;
  int l_m, l_n, l_k;
  int l_i, l_j, l_z;
  int l_elems;
  int l_reps;
  int l_beta;

  libxsmm_timer_tickint l_start, l_end;
  double l_total;

  REALTYPE alpha = 1;
  REALTYPE beta = 1;
  char trans = 'N';

  libxsmm_fsspmdm* gemm_op_betazero = NULL;
  libxsmm_fsspmdm* gemm_op_betaone = NULL;

  if (argc < 4) {
    fprintf( stderr, "need csr-filename N reps [beta=0|1]!\n" );
    exit(-1);
  }

  /* read sparse A */
  l_csr_file = argv[1];
  l_n = atoi(argv[2]);
  /* sanitize blocksize */
  if (l_n < l_n_block) l_n_block = l_n;
  l_reps = atoi(argv[3]);
  l_beta = (4 < argc ? atoi(argv[4]) : -1);

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
  printf("CSR matrix data structure we just read (%s):\n", l_csr_file);
  printf("rows: %u, columns: %u, elements: %u\n", l_rowcount, l_colcount, l_elements);

  /* allocate dense matrices */
  l_a_dense = (REALTYPE*)libxsmm_aligned_malloc(sizeof(REALTYPE) * l_k * l_m, 64);
  l_b = (REALTYPE*)libxsmm_aligned_malloc(sizeof(REALTYPE) * l_k * l_n, 64);

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

  if (0 >= l_beta) {
    const char *const env_fsspmdm_nts = getenv("FSSPMDM_NTS");
    /* allocate C */
    l_c_betazero = (REALTYPE*)libxsmm_aligned_malloc(sizeof(REALTYPE) * l_m * l_n, 64);
    l_c_gold_betazero = (REALTYPE*)libxsmm_aligned_malloc(sizeof(REALTYPE) * l_m * l_n, 64);
    l_c_dense_betazero = (REALTYPE*)libxsmm_aligned_malloc(sizeof(REALTYPE) * l_m * l_n, 64);
    assert(NULL != l_c_betazero && NULL != l_c_gold_betazero && NULL != l_c_dense_betazero);
    libxsmm_rng_set_seed(25071975);
    /* touch C */
    for (l_i = 0; l_i < (l_m*l_n); l_i++) {
      l_c_gold_betazero[l_i] = (REALTYPE)libxsmm_rng_f64();
    }
    for ( l_i = 0; l_i < (l_m*l_n); l_i++) {
      l_c_betazero[l_i] = l_c_gold_betazero[l_i];
    }
    for ( l_i = 0; l_i < (l_m*l_n); l_i++) {
      l_c_dense_betazero[l_i] = l_c_gold_betazero[l_i];
    }
    /* setting up fsspmdm */
    beta = 0;
    gemm_op_betazero = libxsmm_fsspmdm_create(LIBXSMM_DATATYPE(REALTYPE), l_m, l_n_block, l_k, l_k, l_n, l_n, &alpha, &beta,
      (NULL == env_fsspmdm_nts || '\0' == *env_fsspmdm_nts || 0 != atoi(env_fsspmdm_nts)) ? 1 : 0, l_a_dense);
  }

  if (0 > l_beta || 0 < l_beta) {
    /* allocate C */
    l_c_betaone = (REALTYPE*)libxsmm_aligned_malloc(sizeof(REALTYPE) * l_m * l_n, 64);
    l_c_gold_betaone = (REALTYPE*)libxsmm_aligned_malloc(sizeof(REALTYPE) * l_m * l_n, 64);
    l_c_dense_betaone = (REALTYPE*)libxsmm_aligned_malloc(sizeof(REALTYPE) * l_m * l_n, 64);
    assert(NULL != l_c_betaone && NULL != l_c_gold_betaone && NULL != l_c_dense_betaone);
    libxsmm_rng_set_seed(25071975);
    /* touch C */
    for (l_i = 0; l_i < l_m * l_n; l_i++) {
      l_c_gold_betaone[l_i] = (REALTYPE)libxsmm_rng_f64();
    }
    for ( l_i = 0; l_i < (l_m*l_n); l_i++) {
      l_c_betaone[l_i] = l_c_gold_betaone[l_i];
    }
    for ( l_i = 0; l_i < (l_m*l_n); l_i++) {
      l_c_dense_betaone[l_i] = l_c_gold_betaone[l_i];
    }
    /* setting up fsspmdm */
    beta = 1;
    gemm_op_betaone = libxsmm_fsspmdm_create(LIBXSMM_DATATYPE(REALTYPE),
      l_m, LIBXSMM_MIN(l_n_block, l_n), l_k, l_k, l_n, l_n,
      &alpha, &beta, 0, l_a_dense);
  }

  /* compute golden results */
  printf("\ncomputing golden solution...\n");
  if (0 >= l_beta) {
    for (l_j = 0; l_j < l_n; l_j++) {
      for (l_i = 0; l_i < l_m; l_i++) {
        l_elems = l_rowptr[l_i + 1] - l_rowptr[l_i];
        l_c_gold_betazero[(l_n * l_i) + l_j] = 0;
        for (l_z = 0; l_z < l_elems; l_z++) {
          l_c_gold_betazero[(l_n * l_i) + l_j] += l_a_sp[l_rowptr[l_i] + l_z] * l_b[(l_n * l_colidx[l_rowptr[l_i] + l_z]) + l_j];
        }
      }
    }
  }
  if (0 > l_beta || 0 < l_beta) {
    for (l_j = 0; l_j < l_n; l_j++) {
      for (l_i = 0; l_i < l_m; l_i++) {
        l_elems = l_rowptr[l_i + 1] - l_rowptr[l_i];
        for (l_z = 0; l_z < l_elems; l_z++) {
          l_c_gold_betaone[(l_n * l_i) + l_j] += l_a_sp[l_rowptr[l_i] + l_z] * l_b[(l_n * l_colidx[l_rowptr[l_i] + l_z]) + l_j];
        }
      }
    }
  }
  printf("\tdone!\n");

  /* libxsmm generated code */
  printf("computing libxsmm (A sparse) solution...\n");
  if (0 >= l_beta) {
#if defined(_OPENMP)
#   pragma omp parallel for private(l_z)
#endif
    for (l_z = 0; l_z < l_n; l_z += l_n_block) {
      libxsmm_fsspmdm_execute(gemm_op_betazero, l_b + l_z, l_c_betazero + l_z);
    }
  }
  if (0 > l_beta || 0 < l_beta) {
#if defined(_OPENMP)
#   pragma omp parallel for private(l_z)
#endif
    for (l_z = 0; l_z < l_n; l_z += l_n_block) {
      libxsmm_fsspmdm_execute(gemm_op_betaone, l_b + l_z, l_c_betaone + l_z);
    }
  }
  printf("\tdone!\n");

  /* BLAS code */
  printf("computing BLAS (A dense) solution...\n");
  if (0 >= l_beta) {
    beta = 0;
    GEMM(&trans, &trans, &l_n, &l_m, &l_k, &alpha, l_b, &l_n, l_a_dense, &l_k, &beta, l_c_dense_betazero, &l_n);
  }
  if (0 > l_beta || 0 < l_beta) {
    beta = 1;
    GEMM(&trans, &trans, &l_n, &l_m, &l_k, &alpha, l_b, &l_n, l_a_dense, &l_k, &beta, l_c_dense_betaone, &l_n);
  }
  printf("\tdone!\n");

  printf("\nvalidating results...\n"); /* check for errors */
  if (0 >= l_beta) {
    libxsmm_matdiff(&diff, LIBXSMM_DATATYPE(REALTYPE), l_m, l_n,
      l_c_gold_betazero, l_c_betazero, NULL/*ldref*/, NULL/*ldtst*/);
    printf("\tmax error beta=0 (libxmm vs. gold): %f", diff.linf_abs);
    if (EPSILON(REALTYPE) < libxsmm_matdiff_epsilon(&diff)) {
      printf(" (%f != %f)\n", diff.v_ref, diff.v_tst);
      ret |= 1;
    }
    else printf("\n");
    libxsmm_matdiff(&diff, LIBXSMM_DATATYPE(REALTYPE), l_m, l_n,
      l_c_gold_betazero, l_c_dense_betazero, NULL/*ldref*/, NULL/*ldtst*/);
    printf("\tmax error beta=0 (dense vs. gold): %f", diff.linf_abs);
    if (EPSILON(REALTYPE) < libxsmm_matdiff_epsilon(&diff)) {
      printf(" (%f != %f)\n", diff.v_ref, diff.v_tst);
    }
    else printf("\n");
  }
  if (0 > l_beta || 0 < l_beta) {
    libxsmm_matdiff(&diff, LIBXSMM_DATATYPE(REALTYPE), l_m, l_n,
      l_c_gold_betaone, l_c_betaone, NULL/*ldref*/, NULL/*ldtst*/);
    printf("\tmax error beta=1 (libxmm vs. gold): %f", diff.linf_abs);
    if (EPSILON(REALTYPE) < libxsmm_matdiff_epsilon(&diff)) {
      printf(" (%f != %f)\n", diff.v_ref, diff.v_tst);
      ret |= 1;
    }
    else printf("\n");
    libxsmm_matdiff(&diff, LIBXSMM_DATATYPE(REALTYPE), l_m, l_n,
      l_c_gold_betaone, l_c_dense_betaone, NULL/*ldref*/, NULL/*ldtst*/);
    printf("\tmax error beta=1 (dense vs. gold): %f", diff.linf_abs);
    if (EPSILON(REALTYPE) < libxsmm_matdiff_epsilon(&diff)) {
      printf(" (%f != %f)\n", diff.v_ref, diff.v_tst);
    }
    else printf("\n");
  }

  /* Let's measure performance */
  if (0 >= l_beta) {
    l_start = libxsmm_timer_tick();
    for (l_j = 0; l_j < l_reps; l_j++) {
#if defined(_OPENMP)
#     pragma omp parallel for private(l_z)
#endif
      for (l_z = 0; l_z < l_n; l_z += l_n_block) {
        libxsmm_fsspmdm_execute(gemm_op_betazero, l_b + l_z, l_c_betazero + l_z);
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    printf("\nperformance: RM, M=%i, N=%i, K=%i, beta=0\n", l_m, l_n, l_k);
    printf("\tLIBXSMM time[s]: %f\n", l_total / (double)l_reps);
    printf("\tLIBXSMM GFLOPS : %f (sparse)\n", (2.0 * (double)l_elements * (double)l_n * (double)l_reps * 1.0e-9) / l_total);
    printf("\tLIBXSMM GFLOPS : %f (dense)\n", (2.0 * (double)l_m * (double)l_n * (double)l_k * (double)l_reps * 1.0e-9) / l_total);
    printf("\tLIBXSMM GB/s   : %f\n", ((double)sizeof(REALTYPE) * (((double)l_m * (double)l_n) + ((double)l_k * (double)l_n)) * (double)l_reps * 1.0e-9) / l_total);

    l_start = libxsmm_timer_tick();
    beta = 0;
    for (l_j = 0; l_j < l_reps; l_j++) {
      GEMM(&trans, &trans, &l_n, &l_m, &l_k, &alpha, l_b, &l_n, l_a_dense, &l_k, &beta, l_c_dense_betazero, &l_n);
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    printf("\tBLAS time[s]   : %f\n", l_total / (double)l_reps);
    printf("\tBLAS GFLOPS    : %f\n", (2.0 * (double)l_m * (double)l_n * (double)l_k * (double)l_reps * 1.0e-9) / l_total);
    printf("\tBLAS GB/s      : %f\n", ((double)sizeof(REALTYPE) * ((2.0 * (double)l_m * (double)l_n) + ((double)l_k * (double)l_n)) * (double)l_reps * 1.0e-9) / l_total);
  }

  /* Let's measure performance */
  if (0 > l_beta || 0 < l_beta) {
    l_start = libxsmm_timer_tick();
    for (l_j = 0; l_j < l_reps; l_j++) {
#if defined(_OPENMP)
#     pragma omp parallel for private(l_z)
#endif
      for (l_z = 0; l_z < l_n; l_z += l_n_block) {
        libxsmm_fsspmdm_execute(gemm_op_betaone, l_b + l_z, l_c_betaone + l_z);
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    printf("\nperformance: RM, M=%i, N=%i, K=%i, beta=1\n", l_m, l_n, l_k);
    printf("\tLIBXSMM time[s]: %f\n", l_total / (double)l_reps);
    printf("\tLIBXSMM GFLOPS : %f (sparse)\n", (2.0 * (double)l_elements * (double)l_n * (double)l_reps * 1.0e-9) / l_total);
    printf("\tLIBXSMM GFLOPS : %f (dense)\n", (2.0 * (double)l_m * (double)l_n * (double)l_k * (double)l_reps * 1.0e-9) / l_total);
    printf("\tLIBXSMM GB/s   : %f\n", ((double)sizeof(REALTYPE) * ((2.0 * (double)l_m * (double)l_n) + ((double)l_k * (double)l_n)) * (double)l_reps * 1.0e-9) / l_total);

    l_start = libxsmm_timer_tick();
    beta = 1;
    for (l_j = 0; l_j < l_reps; l_j++) {
      GEMM(&trans, &trans, &l_n, &l_m, &l_k, &alpha, l_b, &l_n, l_a_dense, &l_k, &beta, l_c_dense_betaone, &l_n);
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    printf("\tBLAS time[s]   : %f\n", l_total / (double)l_reps);
    printf("\tBLAS GFLOPS    : %f\n", (2.0 * (double)l_m * (double)l_n * (double)l_k * (double)l_reps * 1.0e-9) / l_total);
    printf("\tBLAS GB/s      : %f\n", ((double)sizeof(REALTYPE) * ((2.0 * (double)l_m * (double)l_n) + ((double)l_k * (double)l_n)) * (double)l_reps * 1.0e-9) / l_total);
  }

  /* free */
  libxsmm_fsspmdm_destroy(gemm_op_betazero);
  libxsmm_fsspmdm_destroy(gemm_op_betaone);

  libxsmm_free(l_c_dense_betazero);
  libxsmm_free(l_c_gold_betazero);
  libxsmm_free(l_c_betazero);

  libxsmm_free(l_c_dense_betaone);
  libxsmm_free(l_c_gold_betaone);
  libxsmm_free(l_c_betaone);

  libxsmm_free(l_a_dense);
  libxsmm_free(l_b);

  free(l_rowptr);
  free(l_colidx);
  free(l_a_sp);

  return ret;
}
