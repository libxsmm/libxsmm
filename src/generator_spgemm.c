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
#include "generator_common.h"
#include "generator_spgemm_csc_reader.h"
#include "generator_spgemm_csr_reader.h"
#include "generator_spgemm_csc_asparse.h"
#include "generator_spgemm_csc_bsparse.h"
#include "generator_spgemm_csr_asparse.h"
#include "generator_spgemm_csr_asparse_reg.h"


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
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  } else {
    /* something bad happened... */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_SPGEMM_GEN );
    return;
  }
}

LIBXSMM_API
void libxsmm_generator_spgemm_csr_reg_kernel( libxsmm_generated_code*        io_generated_code,
                                              const libxsmm_gemm_descriptor* i_xgemm_desc,
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

    /* x86 */
    if ( io_generated_code->arch >= LIBXSMM_X86_GENERIC &&
         io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) {
      libxsmm_generator_spgemm_csr_asparse_reg_x86( io_generated_code, i_xgemm_desc,
                                                    i_row_idx, i_column_idx, i_values );
    /* aarch64 without SVE */
    } else if ( io_generated_code->arch >= LIBXSMM_AARCH64_V81 &&
                io_generated_code->arch < LIBXSMM_AARCH64_SVE128 ) {
      libxsmm_generator_spgemm_csr_asparse_reg_aarch64_neon( io_generated_code, i_xgemm_desc,
                                                             i_row_idx, i_column_idx, i_values );
    /* aarch64 with SVE */
    }  else if ( io_generated_code->arch >= LIBXSMM_AARCH64_SVE128 &&
                 io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT ) {
      libxsmm_generator_spgemm_csr_asparse_reg_aarch64_sve( io_generated_code, i_xgemm_desc,
                                                            i_row_idx, i_column_idx, i_values );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
      return;
    }
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
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
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
  LIBXSMM_MEMZERO127(&l_generated_code);

  /* add signature to code string */
  libxsmm_mmfunction_signature( &l_generated_code, i_routine_name, i_xgemm_desc );

  /* account for cases where requested shape does not match sparse data */
  l_column_count = i_xgemm_desc->n;
  l_row_count = i_xgemm_desc->m;

  /* check if generate to CSC */
  /* @TODO, this i_is_csr is very hacky.... change it in future */
  if ( (i_is_csr == 0) || (i_is_csr > 9) ) {
    /* read CSC file and construct CSC data structure */
    libxsmm_sparse_csc_reader( &l_generated_code, i_file_in, &l_row_idx, &l_column_idx, &l_values, &l_row_count, &l_column_count, &l_element_count );

    if (NULL != l_row_idx && NULL != l_column_idx && NULL != l_values) {
#if !defined(NDEBUG)
      /* mute static analysis about garbage content */
      double *const l_tmp = (double*)calloc((size_t)l_row_count * l_column_count, sizeof(double));
      unsigned int l_n;
      unsigned int l_m;

      printf("CSC matrix data structure we just read:\n");
      printf("rows: %u, columns: %u, elements: %u\n", l_row_count, l_column_count, l_element_count);

      if (l_tmp == NULL) {
        fprintf(stderr, "LIBXSMM fatal error: Could allocate dense value array to test CSC data structure!\n");
        LIBXSMM_EXIT_ERROR(&l_generated_code);
        return;
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

      assert(NULL != l_tmp);
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
      } else {
        assert(0/*should not happen*/);
      }
    }
  } else {
    /* read CSR file and construct CSR data structure */
    libxsmm_sparse_csr_reader( &l_generated_code, i_file_in, &l_row_idx, &l_column_idx, &l_values, &l_row_count, &l_column_count, &l_element_count );

    if (NULL != l_row_idx && NULL != l_column_idx && NULL != l_values) { /* libxsmm_sparse_*_reader may have deallocated l_values */
#if !defined(NDEBUG)
      /* mute static analysis about garbage content */
      double *const l_tmp = (double*)calloc((size_t)l_row_count * l_column_count, sizeof(double));
      unsigned int l_n;
      unsigned int l_m;

      printf("CSR matrix data structure we just read:\n");
      printf("rows: %u, columns: %u, elements: %u\n", l_row_count, l_column_count, l_element_count);

      if (l_tmp == NULL) {
        fprintf(stderr, "LIBXSMM fatal error:Could allocate dense value array to test CSR data structure!\n");
        LIBXSMM_EXIT_ERROR(&l_generated_code);
        return;
      }

      for ( l_n = 0; l_n < (l_row_count * l_column_count); l_n++) {
        l_tmp[l_n] = 0.0;
      }

      /* coverity[tainted_data] */
      for ( l_n = 0; l_n <= l_row_count; l_n++) {
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

      assert(NULL != l_tmp);
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
    LIBXSMM_EXIT_ERROR(&l_generated_code);
    return;
  }

  /* append code to source file */
  if ( l_generated_code.generated_code != NULL ) {
    FILE *const l_file_handle = fopen( i_file_out, "a" );
    if ( l_file_handle != NULL ) {
      fputs( (const char*)l_generated_code.generated_code, l_file_handle );
      fclose( l_file_handle );
    } else {
      fprintf(stderr, "LIBXSMM ERROR: libxsmm_generator_spgemm could not write to into destination source file\n");
      LIBXSMM_EXIT_ERROR(&l_generated_code);
      return;
    }
  }

  /* free code memory */
  free( l_generated_code.generated_code );
}
