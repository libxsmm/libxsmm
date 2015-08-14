/******************************************************************************
** Copyright (c) 2015, Intel Corporation                                     **
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>

#include "generator_common.h"
#include "generator_sparse.h"
#include "generator_sparse_csc_reader.h"
#include "generator_sparse_asparse.h"
#include "generator_sparse_bsparse.h"

void libxsmm_generator_sparse_kernel( libxsmm_generated_code*         io_generated_code,
                                      const libxsmm_xgemm_descriptor* i_xgemm_desc,
                                      const char*                     i_arch, 
                                      const unsigned int*             i_row_idx,
                                      const unsigned int*             i_column_idx,
                                      const double*                   i_values ) {
  /* A matrix is sparse */
  if ( (i_xgemm_desc->lda == 0) && (i_xgemm_desc->ldb > 0) && (i_xgemm_desc->ldc > 0) ) {
    /* check LDB */
    if ( i_xgemm_desc->ldb < i_xgemm_desc->k ) {
      libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_LDB );
      return;
    }  
    /* check LDC */
    if ( i_xgemm_desc->ldc < i_xgemm_desc->m ) {
      libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_LDC );
      return;
    }  
    libxsmm_generator_sparse_asparse( io_generated_code, i_xgemm_desc, i_arch, i_row_idx, i_column_idx, i_values );
  /* B matrix is sparse */
  } else if ( (i_xgemm_desc->lda > 0) && (i_xgemm_desc->ldb == 0) && (i_xgemm_desc->ldc > 0) ) {
    /* check LDA */
    if ( i_xgemm_desc->lda < i_xgemm_desc->m ) {
      libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_LDA );
      return;
    }  
    /* check LDC */
    if ( i_xgemm_desc->ldc < i_xgemm_desc->m ) {
      libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_LDC );
      return;
    }  
    libxsmm_generator_sparse_bsparse( io_generated_code, i_xgemm_desc, i_arch, i_row_idx, i_column_idx, i_values );
  } else {
    /* something bad happened... */
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_SPARSE_GEN );
    return;
  }
}

void libxsmm_generator_sparse( const char*                     i_file_out,
                               const char*                     i_routine_name,
                               const libxsmm_xgemm_descriptor* i_xgemm_desc,
                               const char*                     i_arch, 
                               const char*                     i_csc_file_in ) {
  /* init generated code object */
  libxsmm_generated_code l_generated_code;
  l_generated_code.generated_code = NULL;
  l_generated_code.buffer_size = 0;
  l_generated_code.code_size = 0;
  l_generated_code.code_type = 0;
  l_generated_code.last_error = 0;
  
  /* CSC structure */
  unsigned int* l_row_idx = NULL;
  unsigned int* l_column_idx = NULL;
  double* l_values = NULL;
  unsigned int l_row_count;
  unsigned int l_column_count;
  unsigned int l_element_count;

  /* add signature to code string */
  libxsmm_function_signature( &l_generated_code, i_routine_name, i_xgemm_desc );

  /* read CSC file and consturct CSC datastructure */
  libxsmm_sparse_csc_reader( &l_generated_code, i_csc_file_in, &l_row_idx, &l_column_idx, &l_values, &l_row_count, &l_column_count, &l_element_count );

#ifndef NDEBUG
  printf("CSC matrix data structure we just read:\n");
  printf("rows: %u, columns: %u, elements: %u\n", l_row_count, l_column_count, l_element_count);

  double* l_tmp = (double*)malloc(l_row_count * l_column_count * sizeof(double));
  unsigned int l_n;
  unsigned int l_m;

  for ( l_n = 0; l_n < (l_row_count * l_column_count); l_n++) {
    l_tmp[l_n] = 0.0;
  }

  for ( l_n = 0; l_n < l_column_count; l_n++) {
    int l_column_elems = l_column_idx[l_n+1] - l_column_idx[l_n];

    for ( l_m = 0; l_m < l_column_elems; l_m++) {
      l_tmp[(l_n * l_row_count) + l_row_idx[l_column_idx[l_n] + l_m]] = l_values[l_column_idx[l_n] + l_m];
    }
  }

  for ( l_n = 0; l_n < l_row_count; l_n++) {
    for ( l_m = 0; l_m < l_column_count; l_m++) {
      printf("%lf ", l_tmp[(l_m * l_row_count) + l_n]);
    }
    printf("\n");
  }
  
  free( l_tmp );
#endif  

  /* generate the actual kernel code for current description depending on the architecture */
  libxsmm_generator_sparse_kernel( &l_generated_code, i_xgemm_desc, i_arch, l_row_idx, l_column_idx, l_values );

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
    fprintf(stderr, "LIBXSMM ERROR there was an error generating code. Last known error is:\n");
    fprintf(stderr, libxsmm_strerror(l_generated_code.last_error) );
    exit(-1);
  }

  /* append code to source file */
  FILE *l_file_handle = fopen( i_file_out, "a" );
  if ( l_file_handle != NULL ) {
    fputs( l_generated_code.generated_code, l_file_handle );
    fclose( l_file_handle );
  } else {
    fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_sparse: could not write to into destination source file\n");
    exit(-1);
  }

  /* free code memory */
  free( l_generated_code.generated_code );
}

