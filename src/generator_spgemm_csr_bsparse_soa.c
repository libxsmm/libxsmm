/******************************************************************************
** Copyright (c) 2015-2016, Intel Corporation                                **
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

#include "generator_spgemm_csr_bsparse_soa.h"
#include "generator_common.h"
#include <libxsmm_macros.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void libxsmm_generator_spgemm_csr_bsparse_soa( libxsmm_generated_code*         io_generated_code,
                                               const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                               const char*                     i_arch,
                                               const unsigned int*             i_row_idx,
                                               const unsigned int*             i_column_idx,
                                               const double*                   i_values ) {
  if ( strcmp(i_arch, "knl") == 0 ) {
    libxsmm_generator_spgemm_csr_bsparse_soa_avx512( io_generated_code,
                                                     i_xgemm_desc,
                                                     i_arch,
                                                     i_row_idx,
                                                     i_column_idx,
                                                     i_values );
  } else {
    fprintf( stderr, "CSR + SOA is only available for AVX512 at this point" );
    exit(-1);
  }
}


void libxsmm_generator_spgemm_csr_bsparse_soa_avx512( libxsmm_generated_code*         io_generated_code,
                                                      const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                      const char*                     i_arch,
                                                      const unsigned int*             i_row_idx,
                                                      const unsigned int*             i_column_idx,
                                                      const double*                   i_values ) {
  unsigned int l_m;
  unsigned int l_n;
  unsigned int l_k;
  unsigned int l_z;
  unsigned int l_row_elements;
  unsigned int l_flop_count = 0;

  char l_new_code[512];
  int l_max_code_length = 511;
  int l_code_length = 0;

  unsigned int l_soa_width;

  LIBXSMM_UNUSED(i_values);

  /* check that we have enough registers (N=20) for now */
  if (i_xgemm_desc->n > 20 ) {
    fprintf( stderr, "CSR + SOA is limited to N<=20 for the time being!" );
    exit(-1);
  }

  /* select soa width */
  if ( (LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) == 0 ) {
    l_soa_width = 8;
  } else {
    l_soa_width = 16;
  }

  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  unsigned int l_m = 0;\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

  /* reset C if beta is zero */
  if ( i_xgemm_desc->beta == 0 ) {
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  unsigned int l_n = 0;\n");
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  for ( l_m = 0; l_m < %u; l_m++ ) {\n", (unsigned int)i_xgemm_desc->m);
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    for ( l_n = 0; l_n < %u; l_n++ ) {\n", (unsigned int)i_xgemm_desc->n);
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    if ( (LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) == 0 ) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "      _mm512_storeu_pd( C[(l_m*%u)+(l_n*%u)], _mm512_setzero_pd() );\n", (unsigned int)i_xgemm_desc->ldc*l_soa_width, l_soa_width);
    } else {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "      _mm512_storeu_ps( C[(l_m*%u)+(l_n*%u)], _mm512_setzero_ps() );\n", (unsigned int)i_xgemm_desc->ldc*l_soa_width, l_soa_width);
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    }\n");
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  }\n");
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

  /* generate the actuel kernel */
  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  for ( l_m = 0; l_m < %u; l_m++ ) {\n", (unsigned int)i_xgemm_desc->m);
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  /* load C accumulator */
  for ( l_n = 0; l_n < i_xgemm_desc->n; l_n++ ) {
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    __m512d acc%u = _mm512_loadu_pd( &C[(l_m*%u)+%u] );\n", l_n, (unsigned int)i_xgemm_desc->ldc*l_soa_width, l_soa_width*l_n);
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }

  for ( l_k = 0; l_k < i_xgemm_desc->k; l_k++ ) {
    l_row_elements = i_row_idx[l_k+1] - i_row_idx[l_k];
    if (l_row_elements > 0) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    __m512d src%u = _mm512_loadu_pd( &A[(l_m*%u)+%u] );\n", l_k, (unsigned int)i_xgemm_desc->ldc*l_soa_width, l_soa_width*l_k);
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    }
    for ( l_z = 0; l_z < l_row_elements; l_z++ ) {
      /* check k such that we just use columns which actually need to be multiplied */
      if ( i_column_idx[i_row_idx[l_k] + l_z] < i_xgemm_desc->n ) {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    acc%u = _mm512_fmadd_pd( _mm512_set1_pd( B[%u] ), src%u, acc%u );\n", i_column_idx[i_row_idx[l_k] + l_z], i_row_idx[l_k] + l_z, l_k, i_column_idx[i_row_idx[l_k] + l_z] );
        libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
        l_flop_count += 16;
      }
    }
  }

  /* store C accumulator */
  for ( l_n = 0; l_n < i_xgemm_desc->n; l_n++ ) {
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    _mm512_storeu_pd( &C[(l_m*%u)+%u], acc%u );\n", (unsigned int)i_xgemm_desc->ldc*l_soa_width, l_soa_width*l_n, l_n);
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }

#if 0

#endif

  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  }\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

#if 0
  /* add flop counter */
  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "\n#ifndef NDEBUG\n#ifdef _OPENMP\n#pragma omp atomic\n#endif\nlibxsmm_num_total_flops += %u;\n#endif\n", l_flop_count * (unsigned int)i_xgemm_desc->m);
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
#endif
}

