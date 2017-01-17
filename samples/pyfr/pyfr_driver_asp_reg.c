/******************************************************************************
** Copyright (c) 2014-2017, Intel Corporation                                **
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

#include <libxsmm.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>

#define REPS 100

#define REALTYPE double

/* forward decelration of generated code */
void libxsmm_code(const double* A, const double* B, double* C);
libxsmm_dmmfunction libxsmm_jit;

void libxsmm_kernel(const double* A, const double* B, double* C, const unsigned int N, const unsigned int vlen) {
  unsigned int n;

#ifdef _OPENMP
  #pragma omp parallel for private(n)
#endif
  for (n = 0; n < N; n+=vlen) {
#if 0
    libxsmm_code(A, B+n, C+n);
#else
    libxsmm_jit(A, B+n, C+n);
#endif
  }
}

static double sec(struct timeval start, struct timeval end) {
  return ((double)(((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)))) / 1.0e6;
}

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
        if ( sscanf(l_line, "%u %u %u", o_row_count, o_column_count, o_element_count) == 3 ) {
          /* allocate CSC datastructue matching mtx file */
          *o_column_idx = (unsigned int*) malloc(sizeof(unsigned int) * (*o_element_count));
          *o_row_idx = (unsigned int*) malloc(sizeof(unsigned int) * (*o_row_count + 1));
          *o_values = (REALTYPE*) malloc(sizeof(double) * (*o_element_count));
          l_row_idx_id = (unsigned int*) malloc(sizeof(unsigned int) * (*o_row_count));

          /* check if mallocs were successful */
          if ( ( *o_row_idx == NULL )      ||
               ( *o_column_idx == NULL )   ||
               ( *o_values == NULL )       ||
               ( l_row_idx_id == NULL )    ) {
            fprintf( stderr, "could not allocate sp data!\n" );
            return -1;
          }

          /* set everything to zero for init */
          memset(*o_row_idx, 0, sizeof(unsigned int)*(*o_row_count + 1));
          memset(*o_column_idx, 0, sizeof(unsigned int)*(*o_element_count));
          memset(*o_values, 0, sizeof(double)*(*o_element_count));
          memset(l_row_idx_id, 0, sizeof(unsigned int)*(*o_row_count));

          /* init column idx */
          for ( l_i = 0; l_i < (*o_row_count + 1); l_i++)
            (*o_row_idx)[l_i] = (*o_element_count);

          /* init */
          (*o_row_idx)[0] = 0;
          l_i = 0;
          l_header_read = 1;
        } else {
          fprintf( stderr, "could not csr descripton!\n" );
          return -1;
        }
      /* now we read the actual content */
      } else {
        unsigned int l_row, l_column;
        REALTYPE l_value;
        /* read a line of content */
        if ( sscanf(l_line, "%u %u %lf", &l_row, &l_column, &l_value) != 3 ) {
          fprintf( stderr, "could not read element!\n" );
          return -1;
        }
        /* adjust numbers to zero termination */
        l_row--;
        l_column--;
        /* add these values to row and value strucuture */
        (*o_column_idx)[l_i] = l_column;
        (*o_values)[l_i] = l_value;
        l_i++;
        /* handle columns, set id to onw for this column, yeah we need to hanle empty columns */
        l_row_idx_id[l_row] = 1;
        (*o_row_idx)[l_row+1] = l_i;
      }
    }
  }

  /* close mtx file */
  fclose( l_csr_file_handle );

  /* check if we read a file which was consitent */
  if ( l_i != (*o_element_count) ) {
    fprintf( stderr, "we were not able to read all elements!\n" );
    return -1;
  }

  /* let's handle empty rows */
  for ( l_i = 0; l_i < (*o_row_count); l_i++) {
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
  if (argc != 4 ) {
    fprintf( stderr, "need csr-filename N reps!\n" );
    exit(-1);
  }

  char* l_csr_file;
  REALTYPE* l_a_sp;
  unsigned int* l_rowptr;
  unsigned int* l_colidx;
  unsigned int l_rowcount, l_colcount, l_elements;

  REALTYPE* l_a_dense;
  REALTYPE* l_b;
  REALTYPE* l_c;
  REALTYPE* l_c_gold;
  REALTYPE* l_c_dense;
  REALTYPE l_max_error = 0.0;
  unsigned int l_m;
  unsigned int l_n;
  unsigned int l_k;

  unsigned int l_i;
  unsigned int l_j;
  unsigned int l_z;
  unsigned int l_elems;
  unsigned int l_reps;
  unsigned int l_vlen;

  struct timeval l_start, l_end;
  double l_total;

  libxsmm_gemm_descriptor* l_xgemm_desc = NULL;

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

  l_vlen = 8;
  l_xgemm_desc = libxsmm_create_dgemm_descriptor('n', 'n', l_m, l_vlen, l_k, 0, l_n, l_n, 1.0, 1.0, LIBXSMM_PREFETCH_NONE);
  libxsmm_jit = libxsmm_create_dcsr_reg( l_xgemm_desc, l_rowptr, l_colidx, l_a_sp ).dmm;

  /* allocate dense matrices */
  l_a_dense = (REALTYPE*)_mm_malloc(l_k * l_m * sizeof(REALTYPE), 64);
  l_b = (REALTYPE*)_mm_malloc(l_k * l_n * sizeof(REALTYPE), 64);
  l_c = (REALTYPE*)_mm_malloc(l_m * l_n * sizeof(REALTYPE), 64);
  l_c_gold = (REALTYPE*)_mm_malloc(l_m * l_n * sizeof(REALTYPE), 64);
  l_c_dense = (REALTYPE*)_mm_malloc(l_m * l_n * sizeof(REALTYPE), 64);

  /* touch B */
  for ( l_i = 0; l_i < l_k*l_n; l_i++) {
    l_b[l_i] = (REALTYPE)drand48();
  }

  /* touch dense A */
  for ( l_i = 0; l_i < l_k*l_m; l_i++) {
    l_a_dense[l_i] = (REALTYPE)0.0;
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
    l_c[l_i] = (REALTYPE)0.0;
    l_c_gold[l_i] = (REALTYPE)0.0;
    l_c_dense[l_i] = (REALTYPE)0.0;
  }

  /* compute golden results */
  printf("computing golden solution...\n");
  for ( l_j = 0; l_j < l_n; l_j++ ) {
    for (l_i = 0; l_i < l_m; l_i++ ) {
      l_elems = l_rowptr[l_i+1] - l_rowptr[l_i];
      for (l_z = 0; l_z < l_elems; l_z++) {
        l_c_gold[(l_n*l_i) + l_j] +=  l_a_sp[l_rowptr[l_i]+l_z] * l_b[(l_n*l_colidx[l_rowptr[l_i]+l_z])+l_j];
      }
    }
  }
  printf("...done!\n");

  /* libxsmm generated code */
  printf("computing libxsmm (A sparse) solution...\n");
  libxsmm_kernel(NULL, l_b, l_c, l_n, l_vlen);
  printf("...done!\n");

  /* BLAS code */
  printf("computing BLAS (A dense) solution...\n");
  double alpha = 1.0;
  double beta = 1.0;
  char trans = 'N';
  dgemm(&trans, &trans, &l_n, &l_m, &l_k, &alpha, l_b, &l_n, l_a_dense, &l_k, &beta, l_c_dense, &l_n );
  printf("...done!\n");

  /* check for errors */
  l_max_error = (REALTYPE)0.0;
  for ( l_i = 0; l_i < l_m*l_n; l_i++) {
    if (fabs(l_c[l_i]-l_c_gold[l_i]) > l_max_error ) {
      l_max_error = fabs(l_c[l_i]-l_c_gold[l_i]);
    }
  }
  printf("max error (libxmm vs. gold): %f\n", l_max_error);
  l_max_error = (REALTYPE)0.0;
  for ( l_i = 0; l_i < l_m*l_n; l_i++) {
    if (fabs(l_c_dense[l_i]-l_c_gold[l_i]) > l_max_error ) {
      l_max_error = fabs(l_c_dense[l_i]-l_c_gold[l_i]);
    }
  }
  printf("max error (dense vs. gold): %f\n", l_max_error);

  /* Let's measure performance */
  gettimeofday(&l_start, NULL);
  for ( l_j = 0; l_j < l_reps; l_j++ ) {
    libxsmm_kernel(NULL, l_b, l_c, l_n, l_vlen);
  }
  gettimeofday(&l_end, NULL);
  l_total = sec(l_start, l_end);
  fprintf(stdout, "time[s] LIBXSMM (RM, M=%i, N=%i, K=%i): %f\n", l_m, l_n, l_k, l_total/(double)l_reps );
  fprintf(stdout, "GFLOPS  LIBXSMM (RM, M=%i, N=%i, K=%i): %f (sparse)\n", l_m, l_n, l_k, (2.0 * (double)l_elements * (double)l_n * (double)l_reps * 1.0e-9) / l_total );
  fprintf(stdout, "GFLOPS  LIBXSMM (RM, M=%i, N=%i, K=%i): %f (dense)\n", l_m, l_n, l_k, (2.0 * (double)l_m * (double)l_n * (double)l_k * (double)l_reps * 1.0e-9) / l_total );
  fprintf(stdout, "GB/s    LIBXSMM (RM, M=%i, N=%i, K=%i): %f\n", l_m, l_n, l_k, ((double)sizeof(double) * ((2.0*(double)l_m * (double)l_n) + ((double)l_k * (double)l_n)) * (double)l_reps * 1.0e-9) / l_total );

  gettimeofday(&l_start, NULL);
  for ( l_j = 0; l_j < l_reps; l_j++ ) {
    dgemm(&trans, &trans, &l_n, &l_m, &l_k, &alpha, l_b, &l_n, l_a_dense, &l_k, &beta, l_c_dense, &l_n );
  }
  gettimeofday(&l_end, NULL);
  l_total = sec(l_start, l_end);
  fprintf(stdout, "time[s] MKL     (RM, M=%i, N=%i, K=%i): %f\n", l_m, l_n, l_k, l_total/(double)l_reps );
  fprintf(stdout, "GFLOPS  MKL     (RM, M=%i, N=%i, K=%i): %f\n", l_m, l_n, l_k, (2.0 * (double)l_m * (double)l_n * (double)l_k * (double)l_reps * 1.0e-9) / l_total );
  fprintf(stdout, "GB/s    MKL     (RM, M=%i, N=%i, K=%i): %f\n", l_m, l_n, l_k, ((double)sizeof(double) * ((2.0*(double)l_m * (double)l_n) + ((double)l_k * (double)l_n)) * (double)l_reps * 1.0e-9) / l_total );

  /* free */
  libxsmm_release_kernel(libxsmm_jit);
  libxsmm_release_gemm_descriptor(l_xgemm_desc);
  /* @TODO */
}
