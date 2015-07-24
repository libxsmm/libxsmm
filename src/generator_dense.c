/******************************************************************************
** Copyright (c) 2014-2015, Intel Corporation                                **
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
#include "generator_dense_common.h"
#include "generator_dense_avx.h"

void libxsmm_generator_dense_signature(char**             io_generated_code,
                                       const char*        i_routine_name,
                                       const unsigned int i_single_precision,
                                       const char*        i_prefetch) {
  char l_new_code_line[512];
  l_new_code_line[0] = '\0';
  
  /* selecting the correct signature */
  if (i_single_precision == 1) {
    if ( strcmp(i_prefetch, "nopf") == 0) {
      sprintf(l_new_code_line, "void %s(const float* A, const float* B, float* C) {\n", i_routine_name);
    } else {
      sprintf(l_new_code_line, "void %s(const float* A, const float* B, float* C, const float* A_prefetch = NULL, const float* B_prefetch = NULL, const float* C_prefetch = NULL) {\n", i_routine_name);
    }
  } else {
    if ( strcmp(i_prefetch, "nopf") == 0) {
      sprintf(l_new_code_line, "void %s(const double* A, const double* B, double* C) {\n", i_routine_name);
    } else {
      sprintf(l_new_code_line, "void %s(const double* A, const double* B, double* C, const double* A_prefetch = NULL, const double* B_prefetch = NULL, const double* C_prefetch = NULL) {\n", i_routine_name);
    }
  }

  libxsmm_append_string( io_generated_code, l_new_code_line );
}

void libxsmm_generator_dense_kernel(char**             io_generated_code,
                                    const unsigned int i_m,
                                    const unsigned int i_n,
                                    const unsigned int i_k,
                                    const unsigned int i_lda,
                                    const unsigned int i_ldb,
                                    const unsigned int i_ldc, 
                                    const int          i_alpha,
                                    const int          i_beta,
                                    const unsigned int i_aligned_a,
                                    const unsigned int i_aligned_c,
                                    const char*        i_arch,
                                    const char*        i_prefetch,
                                    const unsigned int i_single_precision) {
  /* add instruction set mismatch check to code, header */
  libxsmm_generator_dense_add_isa_check_header( io_generated_code, i_arch );

  if ( (strcmp(i_arch, "wsm") == 0) ) {
    /*libxsmm_generator_dense_sse(io_generated_code,
                                i_m, i_n, i_k,
                                i_lda, i_ldb, i_ldc, 
                                i_alpha, i_beta,
                                i_aligned_a, i_aligned_c,
                                i_arch,
                                i_prefetch,
                                i_single_precision);*/
  }

  if ( (strcmp(i_arch, "snb") == 0) ||
       (strcmp(i_arch, "hsw") == 0)    ) {
    libxsmm_generator_dense_avx(io_generated_code,
                                i_m, i_n, i_k,
                                i_lda, i_ldb, i_ldc, 
                                i_alpha, i_beta,
                                i_aligned_a, i_aligned_c,
                                i_arch,
                                i_prefetch,
                                i_single_precision);
  }

  if ( strcmp(i_arch, "knc") == 0 ) {
    /* libxsmm_generator_dense_knc(); */
  }

  if ( (strcmp(i_arch, "knl") == 0) || 
       (strcmp(i_arch, "skx") == 0)    ) {
    /* libxsmm_generator_dense_avx512(); */
  }

  /* add instruction set mismatch check to code, footer */
  libxsmm_generator_dense_add_isa_check_footer( io_generated_code, i_arch );

  /* add flop counter for debug compilation */
  libxsmm_generator_dense_add_flop_counter( io_generated_code, i_m, i_n, i_k );
}

void libxsmm_generator_dense(const char*        i_file_out,
                             const char*        i_routine_name,
                             const unsigned int i_m,
                             const unsigned int i_n,
                             const unsigned int i_k,
                             const unsigned int i_lda,
                             const unsigned int i_ldb,
                             const unsigned int i_ldc, 
                             const int          i_alpha,
                             const int          i_beta,
                             const unsigned int i_aligned_a,
                             const unsigned int i_aligned_c,
                             const char*        i_arch,
                             const char*        i_prefetch,
                             const unsigned int i_single_precision) {
  char* l_generated_code = NULL;
  
  /* add signature to code string */
  libxsmm_generator_dense_signature( &l_generated_code, i_routine_name, i_single_precision, i_prefetch );

  /* generate the actual kernel code depending on the architecture */
  libxsmm_generator_dense_kernel( &l_generated_code,
                                  i_m, i_n, i_k,
                                  i_lda, i_ldb, i_ldc, 
                                  i_alpha, i_beta,
                                  i_aligned_a, i_aligned_c,
                                  i_arch,
                                  i_prefetch,
                                  i_single_precision);

  /* close current function */
  libxsmm_close_function( &l_generated_code );

  /* append code to source file */
  FILE *l_file_handle = fopen( i_file_out, "a" );
  if ( l_file_handle != NULL ) {
    fputs( l_generated_code, l_file_handle );
    fclose( l_file_handle );
  } else {
    fprintf(stderr, "LIBXSMM ERROR libxsmm_generator_dense could not write to into destination source file\n");
    exit(-1);
  }

#ifndef NDEBUG
  printf("code was generated and exported to %s \n", i_file_out);
  /*printf("generated code:\n%s", l_generated_code);*/
#endif
 
  free(l_generated_code);
}

