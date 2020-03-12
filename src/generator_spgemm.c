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
#include "generator_common.h"
#include "generator_spgemm_csc_reader.h"
#include "generator_spgemm_csr_reader.h"
#include "generator_spgemm_csc_asparse.h"
#include "generator_spgemm_csc_bsparse.h"
#include "generator_spgemm_csr_asparse.h"
#include "generator_spgemm_csr_asparse_reg.h"
#include "generator_spgemm_csr_bsparse_soa.h"
#include "generator_spgemm_csr_asparse_soa.h"
#include "generator_spgemm_csc_bsparse_soa.h"
#include "generator_spgemm_csc_csparse_soa.h"
#include "libxsmm_main.h"

LIBXSMM_API
void libxsmm_generator_spgemm_csc_kernel( libxsmm_generated_code*        io_generated_code,
                                          const libxsmm_gemm_descriptor* i_xgemm_desc,
                                          const char*                    i_arch,
                                          const unsigned int*            i_row_idx,
                                          const unsigned int*            i_column_idx,
                                          const double*                  i_values ) {
  /* A matrix is sparse */
  if ( (i_xgemm_desc->lda == 0) && (i_xgemm_desc->ldb > 0) && (i_xgemm_desc->ldc > 0) ) {
    /* check LDB */
    if ( i_xgemm_desc->ldb < i_xgemm_desc->k ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB );
      return;
    }
    /* check LDC */
    if ( i_xgemm_desc->ldc < i_xgemm_desc->m ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDC );
      return;
    }
    libxsmm_generator_spgemm_csc_asparse( io_generated_code, i_xgemm_desc, i_arch, i_row_idx, i_column_idx, i_values );
  /* B matrix is sparse */
  } else if ( (i_xgemm_desc->lda > 0) && (i_xgemm_desc->ldb == 0) && (i_xgemm_desc->ldc > 0) ) {
    /* check LDA */
    if ( i_xgemm_desc->lda < i_xgemm_desc->m ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
      return;
    }
    /* check LDC */
    if ( i_xgemm_desc->ldc < i_xgemm_desc->m ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDC );
      return;
    }
    libxsmm_generator_spgemm_csc_bsparse( io_generated_code, i_xgemm_desc, i_arch, i_row_idx, i_column_idx, i_values );
  } else {
    /* something bad happened... */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_SPGEMM_GEN );
    return;
  }
}

LIBXSMM_API
void libxsmm_generator_spgemm_csr_kernel( libxsmm_generated_code*        io_generated_code,
                                          const libxsmm_gemm_descriptor* i_xgemm_desc,
                                          const char*                    i_arch,
                                          const unsigned int*            i_row_idx,
                                          const unsigned int*            i_column_idx,
                                          const double*                  i_values ) {
  /* A matrix is sparse */
  if ( (i_xgemm_desc->lda == 0) && (i_xgemm_desc->ldb > 0) && (i_xgemm_desc->ldc > 0) ) {
    /* check LDB */
    if ( i_xgemm_desc->ldb < i_xgemm_desc->n ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB );
      return;
    }
    /* check LDC */
    if ( i_xgemm_desc->ldc < i_xgemm_desc->n ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDC );
      return;
    }
    libxsmm_generator_spgemm_csr_asparse( io_generated_code, i_xgemm_desc, i_arch, i_row_idx, i_column_idx, i_values );
  /* B matrix is sparse */
  } else if ( (i_xgemm_desc->lda > 0) && (i_xgemm_desc->ldb == 0) && (i_xgemm_desc->ldc > 0) ) {
    /* check LDA */
    if ( i_xgemm_desc->lda < i_xgemm_desc->k ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
      return;
    }
    /* check LDC */
    if ( i_xgemm_desc->ldc < i_xgemm_desc->n ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDC );
      return;
    }
    /* something bad happened... */
    fprintf(stderr, "LIBXSMM fatal error: B sparse for CSR data structure is not yet available!\n");
    exit(-1);
  } else {
    /* something bad happened... */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_SPGEMM_GEN );
    return;
  }
}

LIBXSMM_API
void libxsmm_generator_spgemm_csr_reg_kernel( libxsmm_generated_code*        io_generated_code,
                                              const libxsmm_gemm_descriptor* i_xgemm_desc,
                                              const char*                    i_arch,
                                              const unsigned int*            i_row_idx,
                                              const unsigned int*            i_column_idx,
                                              const double*                  i_values ) {
  /* A matrix is sparse */
  if ( (i_xgemm_desc->lda == 0) && (i_xgemm_desc->ldb > 0) && (i_xgemm_desc->ldc > 0) ) {
    /* check LDB */
    if ( i_xgemm_desc->ldb < i_xgemm_desc->n ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB );
      return;
    }
    /* check LDC */
    if ( i_xgemm_desc->ldc < i_xgemm_desc->n ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDC );
      return;
    }
    libxsmm_generator_spgemm_csr_asparse_reg( io_generated_code, i_xgemm_desc, i_arch, i_row_idx, i_column_idx, i_values );
  /* B matrix is sparse */
  } else if ( (i_xgemm_desc->lda > 0) && (i_xgemm_desc->ldb == 0) && (i_xgemm_desc->ldc > 0) ) {
    /* check LDA */
    if ( i_xgemm_desc->lda < i_xgemm_desc->k ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
      return;
    }
    /* check LDC */
    if ( i_xgemm_desc->ldc < i_xgemm_desc->n ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDC );
      return;
    }
    /* something bad happened... */
    fprintf(stderr, "LIBXSMM fatal error:B sparse for CSR data structure is not yet available!\n");
    exit(-1);
  } else {
    /* something bad happened... */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_SPGEMM_GEN );
    return;
  }
}

LIBXSMM_API
void libxsmm_generator_spgemm_csr_soa_kernel( libxsmm_generated_code*        io_generated_code,
                                              const libxsmm_gemm_descriptor* i_xgemm_desc,
                                              const char*                    i_arch,
                                              const unsigned int*            i_row_idx,
                                              const unsigned int*            i_column_idx,
                                              const void*                    i_values,
                                              const unsigned int             i_packed_width ) {
  /* A matrix is sparse */
  if ( (i_xgemm_desc->lda == 0) && (i_xgemm_desc->ldb > 0) && (i_xgemm_desc->ldc > 0) ) {
    /* check LDB */
    if ( i_xgemm_desc->ldb < i_xgemm_desc->n ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB );
      return;
    }
    /* check LDC */
    if ( i_xgemm_desc->ldc < i_xgemm_desc->n ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDC );
      return;
    }
    libxsmm_generator_spgemm_csr_asparse_soa( io_generated_code, i_xgemm_desc, i_arch, i_row_idx, i_column_idx, i_values, i_packed_width );
  /* B matrix is sparse */
  } else if ( (i_xgemm_desc->lda > 0) && (i_xgemm_desc->ldb == 0) && (i_xgemm_desc->ldc > 0) ) {
    /* check LDA */
    if ( i_xgemm_desc->lda < i_xgemm_desc->k ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
      return;
    }
    /* check LDC */
    /* coverity[copy_paste_error] */
    if ( i_xgemm_desc->ldc < i_xgemm_desc->n ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDC );
      return;
    }
    libxsmm_generator_spgemm_csr_bsparse_soa( io_generated_code, i_xgemm_desc, i_arch, i_row_idx, i_column_idx, i_values, i_packed_width );
  } else {
    /* something bad happened... */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_SPGEMM_GEN );
    return;
  }
}

LIBXSMM_API
void libxsmm_generator_spgemm_csc_soa_kernel( libxsmm_generated_code*        io_generated_code,
                                              const libxsmm_gemm_descriptor* i_xgemm_desc,
                                              const char*                    i_arch,
                                              const unsigned int*            i_row_idx,
                                              const unsigned int*            i_column_idx,
                                              const void*                    i_values,
                                              const unsigned int             i_packed_width ) {
  /* B matrix is sparse */
  if ( (i_xgemm_desc->lda > 0) && (i_xgemm_desc->ldb == 0) && (i_xgemm_desc->ldc > 0) ) {
    /* check LDA */
    if ( i_xgemm_desc->lda < i_xgemm_desc->k ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
      return;
    }
    /* check LDC */
    if ( i_xgemm_desc->ldc < i_xgemm_desc->n ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDC );
      return;
    }
    libxsmm_generator_spgemm_csc_bsparse_soa( io_generated_code, i_xgemm_desc, i_arch, i_row_idx, i_column_idx, i_values, i_packed_width );
  /* C matrix is sparse */
  } else if ( (i_xgemm_desc->lda > 0) && (i_xgemm_desc->ldb > 0) && (i_xgemm_desc->ldc == 0) ) {
#if 0
    /* check LDA */
    if ( i_xgemm_desc->lda < i_xgemm_desc->k ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
      return;
    }
#endif
    /* check LDB */
    if ( i_xgemm_desc->ldb < i_xgemm_desc->n ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB );
      return;
    }
    libxsmm_generator_spgemm_csc_csparse_soa( io_generated_code, i_xgemm_desc, i_arch, i_row_idx, i_column_idx, i_values, i_packed_width );
  } else {
    /* something bad happened... */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_SPGEMM_GEN );
    return;
  }
}

LIBXSMM_API
void libxsmm_generator_spgemm( const char*                    i_file_out,
                               const char*                    i_routine_name,
                               const libxsmm_gemm_descriptor* i_xgemm_desc,
                               const char*                    i_arch,
                               const char*                    i_file_in,
                               const int                      i_is_csr ) {
  /* CSC/CSR structure */
  unsigned int* l_row_idx = NULL;
  unsigned int* l_column_idx = NULL;
  double* l_values = NULL;
  unsigned int l_row_count;
  unsigned int l_column_count;
  unsigned int l_element_count = 0;

  /* init generated code object */
  libxsmm_generated_code l_generated_code;
  l_generated_code.generated_code = NULL;
  l_generated_code.buffer_size = 0;
  l_generated_code.code_size = 0;
  l_generated_code.code_type = 0;
  l_generated_code.last_error = 0;
  l_generated_code.sf_size = 0;

  /* add signature to code string */
  if (i_is_csr == 3) {
    libxsmm_mmfunction_signature_asparse_reg( &l_generated_code, i_routine_name, i_xgemm_desc );
  } else {
    libxsmm_mmfunction_signature( &l_generated_code, i_routine_name, i_xgemm_desc );
  }

  /* check if generate to CSC */
  /* @TODO, this i_is_csr is very hacky.... change it in future */
  if ( (i_is_csr == 0) || (i_is_csr > 9) ) {
    /* read CSC file and construct CSC data structure */
    libxsmm_sparse_csc_reader( &l_generated_code, i_file_in, &l_row_idx, &l_column_idx, &l_values, &l_row_count, &l_column_count, &l_element_count );

    if (0 != l_row_idx && 0 != l_column_idx && 0 != l_values) {
#if !defined(NDEBUG)
      double *const l_tmp = (double*)malloc((size_t)l_row_count * l_column_count * sizeof(double));
      unsigned int l_n;
      unsigned int l_m;

      /* mute static analysis about garbage content */
      memset(l_tmp, 0, (size_t)l_row_count * l_column_count * sizeof(double));

      printf("CSC matrix data structure we just read:\n");
      printf("rows: %u, columns: %u, elements: %u\n", l_row_count, l_column_count, l_element_count);

      if (l_tmp == NULL) {
        fprintf(stderr, "LIBXSMM fatal error:Could allocate dense value array to test CSC data structure!\n");
        exit(-1);
      }

      for ( l_n = 0; l_n < (l_row_count * l_column_count); l_n++) {
        l_tmp[l_n] = 0.0;
      }

      for ( l_n = 0; l_n < l_row_count+1; l_n++) {
         printf("%u ", l_column_idx[l_n]);
      }
      printf("\n");
      for ( l_n = 0; l_n < l_element_count; l_n++) {
         printf("%u ", l_row_idx[l_n]);
      }
      printf("\n");
      for ( l_n = 0; l_n < l_element_count; l_n++) {
         printf("%f ", l_values[l_n]);
      }
      printf("\n");

      for ( l_n = 0; l_n < l_column_count; l_n++) {
        const unsigned int l_column_elems = l_column_idx[l_n+1] - l_column_idx[l_n];
        assert(l_column_idx[l_n+1] >= l_column_idx[l_n]);

        for ( l_m = 0; l_m < l_column_elems; l_m++) {
          l_tmp[(l_row_idx[l_column_idx[l_n] + l_m]*l_column_count) + l_n] = l_values[l_column_idx[l_n] + l_m];
        }
      }

      assert(0 != l_tmp);
      for ( l_n = 0; l_n < l_row_count; l_n++) {
        for ( l_m = 0; l_m < l_column_count; l_m++) {
          printf("%f ", l_tmp[(l_n * l_column_count) + l_m]);
        }
        printf("\n");
      }

      free( l_tmp );
#endif
      /* generate the actual kernel code for current description depending on the architecture */
      if (i_is_csr == 0) {
        libxsmm_generator_spgemm_csc_kernel( &l_generated_code, i_xgemm_desc, i_arch, l_row_idx, l_column_idx, l_values );
      } else if (i_is_csr == 10) {
        assert(0/*should not happen*/);
        /*libxsmm_generator_spgemm_csc_soa_kernel( &l_generated_code, i_xgemm_desc, i_arch, l_row_idx, l_column_idx, l_values, 16 );*/
      } else {
        assert(0/*should not happen*/);
      }
    }
  } else {
    /* read CSR file and construct CSR data structure */
    libxsmm_sparse_csr_reader( &l_generated_code, i_file_in, &l_row_idx, &l_column_idx, &l_values, &l_row_count, &l_column_count, &l_element_count );

    if (0 != l_row_idx && 0 != l_column_idx && 0 != l_values) { /* libxsmm_sparse_*_reader may have deallocated l_values */
#if !defined(NDEBUG)
      double *const l_tmp = (double*)malloc((size_t)l_row_count * l_column_count * sizeof(double));
      unsigned int l_n;
      unsigned int l_m;

      /* mute static analysis about garbage content */
      memset(l_tmp, 0, (size_t)l_row_count * l_column_count * sizeof(double));

      printf("CSR matrix data structure we just read:\n");
      printf("rows: %u, columns: %u, elements: %u\n", l_row_count, l_column_count, l_element_count);

      if (l_tmp == NULL) {
        fprintf(stderr, "LIBXSMM fatal error:Could allocate dense value array to test CSR data structure!\n");
        exit(-1);
      }

      for ( l_n = 0; l_n < (l_row_count * l_column_count); l_n++) {
        l_tmp[l_n] = 0.0;
      }

      for ( l_n = 0; l_n < l_row_count+1; l_n++) {
         printf("%u ", l_row_idx[l_n]);
      }
      printf("\n");
      for ( l_n = 0; l_n < l_element_count; l_n++) {
         printf("%u ", l_column_idx[l_n]);
      }
      printf("\n");
      for ( l_n = 0; l_n < l_element_count; l_n++) {
         printf("%f ", l_values[l_n]);
      }
      printf("\n");

      for ( l_n = 0; l_n < l_row_count; l_n++) {
        const unsigned int l_row_elems = l_row_idx[l_n+1] - l_row_idx[l_n];
        assert(l_row_idx[l_n+1] >= l_row_idx[l_n]);

        for ( l_m = 0; l_m < l_row_elems; l_m++) {
          l_tmp[(l_n * l_column_count) + l_column_idx[l_row_idx[l_n] + l_m]] = l_values[l_row_idx[l_n] + l_m];
        }
      }

      assert(0 != l_tmp);
      for ( l_n = 0; l_n < l_row_count; l_n++) {
        for ( l_m = 0; l_m < l_column_count; l_m++) {
          printf("%f ", l_tmp[(l_n * l_column_count) + l_m]);
        }
        printf("\n");
      }

      free( l_tmp );
#endif
      if (i_is_csr == 1) {
        /* generate the actual kernel code for current description depending on the architecture */
        libxsmm_generator_spgemm_csr_kernel( &l_generated_code, i_xgemm_desc, i_arch, l_row_idx, l_column_idx, l_values );
      } else if (i_is_csr == 2) {
        /* generate the actual kernel code for current description depending on the architecture */
        assert(0/*should not happen*/);
        /*libxsmm_generator_spgemm_csr_soa_kernel( &l_generated_code, i_xgemm_desc, i_arch, l_row_idx, l_column_idx, l_values, 16 );*/
      } else if (i_is_csr == 3) {
        /* generate the actual kernel code for current description depending on the architecture */
        libxsmm_generator_spgemm_csr_reg_kernel( &l_generated_code, i_xgemm_desc, i_arch, l_row_idx, l_column_idx, l_values );
      } else {
        assert(0/*should not happen*/);
      }
    }
  }

  /* close current function */
  libxsmm_close_function( &l_generated_code );

  /* free if not NULL */
  if ( l_row_idx != NULL ) {
    free( l_row_idx );
  }
  if ( l_column_idx != NULL ) {
    free( l_column_idx );
  }
  if ( l_values != NULL ) {
    free( l_values );
  }

  /* check for errors during code generation */
  if ( l_generated_code.last_error != 0 ) {
    LIBXSMM_HANDLE_ERROR_VERBOSE( &l_generated_code, l_generated_code.last_error );
    exit(-1);
  }

  /* append code to source file */
  if ( l_generated_code.generated_code != NULL ) {
    FILE *const l_file_handle = fopen( i_file_out, "a" );
    if ( l_file_handle != NULL ) {
      fputs( (const char*)l_generated_code.generated_code, l_file_handle );
      fclose( l_file_handle );
    } else {
      fprintf(stderr, "LIBXSMM ERROR: libxsmm_generator_spgemm could not write to into destination source file\n");
      exit(-1);
    }
  }

  /* free code memory */
  free( l_generated_code.generated_code );
}

