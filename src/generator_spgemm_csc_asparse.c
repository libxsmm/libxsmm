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
/**
 * @file
 * This file is part of GemmCodeGenerator.
 *
 * @author Alexander Heinecke (alexander.heinecke AT mytum.de, http://www5.in.tum.de/wiki/index.php/Alexander_Heinecke,_M.Sc.,_M.Sc._with_honors)
 *
 * @section LICENSE
 * Copyright (c) 2012-2014, Technische Universitaet Muenchen
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * @section DESCRIPTION
 * <DESCRIPTION>
 */
#include "generator_spgemm_csc_asparse.h"
#include "generator_common.h"
#include <libxsmm_macros.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_sparse_csc_asparse_innerloop_scalar( libxsmm_generated_code*        io_generated_code,
                                                  const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                  const unsigned int             i_k,
                                                  const unsigned int             i_z,
                                                  const unsigned int*            i_row_idx,
                                                  const unsigned int*            i_column_idx ) {
  char l_new_code[512];
  int l_max_code_length = 511;
  int l_code_length = 0;

  if ( (LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) == 0 ) {
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    __m128d c%u_%u = _mm_load_sd(&C[(l_n*%u)+%u]);\n", i_k, i_z, (unsigned int)i_xgemm_desc->ldc, i_row_idx[i_column_idx[i_k] + i_z] );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    __m128d a%u_%u = _mm_load_sd(&A[%u]);\n", i_k, i_z, i_column_idx[i_k] + i_z );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "#if defined(__SSE3__) && defined(__AVX__)\n");
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    c%u_%u = _mm_add_sd(c%u_%u, _mm_mul_sd(a%u_%u, _mm256_castpd256_pd128(b%u)));\n", i_k, i_z, i_k, i_z, i_k, i_z, i_k );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "#endif\n");
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "#if defined(__SSE3__) && !defined(__AVX__)\n");
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    c%u_%u = _mm_add_sd(c%u_%u, _mm_mul_sd(a%u_%u, b%u));\n", i_k, i_z, i_k, i_z, i_k, i_z, i_k );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "#endif\n");
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    _mm_store_sd(&C[(l_n*%u)+%u], c%u_%u);\n", (unsigned int)i_xgemm_desc->ldc, i_row_idx[i_column_idx[i_k] + i_z], i_k, i_z );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  } else {
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    __m128 c%u_%u = _mm_load_ss(&C[(l_n*%u)+%u]);\n", i_k, i_z, (unsigned int)i_xgemm_desc->ldc, i_row_idx[i_column_idx[i_k] + i_z] );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    __m128 a%u_%u = _mm_load_ss(&A[%u]);\n", i_k, i_z, i_column_idx[i_k] + i_z );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    c%u_%u = _mm_add_ss(c%u_%u, _mm_mul_ss(a%u_%u, b%u));\n", i_k, i_z, i_k, i_z, i_k, i_z, i_k );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    _mm_store_ss(&C[(l_n*%u)+%u], c%u_%u);\n", (unsigned int)i_xgemm_desc->ldc, i_row_idx[i_column_idx[i_k] + i_z], i_k, i_z );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_sparse_csc_asparse_innerloop_two_vector( libxsmm_generated_code*        io_generated_code,
                                                      const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                      const unsigned int             i_k,
                                                      const unsigned int             i_z,
                                                      const unsigned int*            i_row_idx,
                                                      const unsigned int*            i_column_idx ) {
  char l_new_code[512];
  int l_max_code_length = 511;
  int l_code_length = 0;

  if ( (LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) == 0 ) {
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    __m128d c%u_%u = _mm_loadu_pd(&C[(l_n*%u)+%u]);\n", i_k, i_z, (unsigned int)i_xgemm_desc->ldc, i_row_idx[i_column_idx[i_k] + i_z] );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    __m128d a%u_%u = _mm_loadu_pd(&A[%u]);\n", i_k, i_z, i_column_idx[i_k] + i_z );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "#if defined(__SSE3__) && defined(__AVX__)\n");
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    c%u_%u = _mm_add_pd(c%u_%u, _mm_mul_pd(a%u_%u, _mm256_castpd256_pd128(b%u)));\n", i_k, i_z, i_k, i_z, i_k, i_z, i_k );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "#endif\n");
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "#if defined(__SSE3__) && !defined(__AVX__)\n");
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    c%u_%u = _mm_add_pd(c%u_%u, _mm_mul_pd(a%u_%u, b%u));\n", i_k, i_z, i_k, i_z, i_k, i_z, i_k );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "#endif\n");
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    _mm_storeu_pd(&C[(l_n*%u)+%u], c%u_%u);\n", (unsigned int)i_xgemm_desc->ldc, i_row_idx[i_column_idx[i_k] + i_z], i_k, i_z );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  } else {
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    __m128 c%u_%u = _mm_castpd_ps(_mm_load_sd((const double*)&C[(l_n*%u)+%u]));\n", i_k, i_z, (unsigned int)i_xgemm_desc->ldc, i_row_idx[i_column_idx[i_k] + i_z] );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    __m128 a%u_%u = _mm_castpd_ps(_mm_load_sd((const double*)&A[%u]));\n", i_k, i_z, i_column_idx[i_k] + i_z );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    c%u_%u = _mm_add_ps(c%u_%u, _mm_mul_ps(a%u_%u, b%u));\n", i_k, i_z, i_k, i_z, i_k, i_z, i_k );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    _mm_store_sd((double*)&C[(l_n*%u)+%u], _mm_castps_pd(c%u_%u));\n", (unsigned int)i_xgemm_desc->ldc, i_row_idx[i_column_idx[i_k] + i_z], i_k, i_z );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_sparse_csc_asparse_innerloop_four_vector( libxsmm_generated_code*        io_generated_code,
                                                       const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                       const unsigned int             i_k,
                                                       const unsigned int             i_z,
                                                       const unsigned int*            i_row_idx,
                                                       const unsigned int*            i_column_idx ) {
  char l_new_code[512];
  int l_max_code_length = 511;
  int l_code_length = 0;

  if ( (LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) == 0 ) {
    unsigned int l_i;
    unsigned int l_z = i_z;
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "#if defined(__SSE3__) && defined(__AVX__)\n");
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    __m256d c%u_%u = _mm256_loadu_pd(&C[(l_n*%u)+%u]);\n", i_k, i_z, (unsigned int)i_xgemm_desc->ldc, i_row_idx[i_column_idx[i_k] + i_z] );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    __m256d a%u_%u = _mm256_loadu_pd(&A[%u]);\n", i_k, i_z, i_column_idx[i_k] + i_z );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    c%u_%u = _mm256_add_pd(c%u_%u, _mm256_mul_pd(a%u_%u, b%u));\n", i_k, i_z, i_k, i_z, i_k, i_z, i_k );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    _mm256_storeu_pd(&C[(l_n*%u)+%u], c%u_%u);\n", (unsigned int)i_xgemm_desc->ldc, i_row_idx[i_column_idx[i_k] + i_z], i_k, i_z );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "#endif\n");
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "#if defined(__SSE3__) && !defined(__AVX__)\n");
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    for ( l_i = 0; l_i < 2; l_i++ ) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    __m128d c%u_%u = _mm_loadu_pd(&C[(l_n*%u)+%u]);\n", i_k, l_z, (unsigned int)i_xgemm_desc->ldc, i_row_idx[i_column_idx[i_k] + l_z] );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    __m128d a%u_%u = _mm_loadu_pd(&A[%u]);\n", i_k, l_z, i_column_idx[i_k] + l_z );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    c%u_%u = _mm_add_pd(c%u_%u, _mm_mul_pd(a%u_%u, b%u));\n", i_k, l_z, i_k, l_z, i_k, l_z, i_k );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    _mm_storeu_pd(&C[(l_n*%u)+%u], c%u_%u);\n", (unsigned int)i_xgemm_desc->ldc, i_row_idx[i_column_idx[i_k] + l_z], i_k, l_z );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_z += 2;
    }
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "#endif\n");
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  } else {
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    __m128 c%u_%u = _mm_loadu_ps(&C[(l_n*%u)+%u]);\n", i_k, i_z, (unsigned int)i_xgemm_desc->ldc, i_row_idx[i_column_idx[i_k] + i_z] );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    __m128 a%u_%u = _mm_loadu_ps(&A[%u]);\n", i_k, i_z, i_column_idx[i_k] + i_z );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    c%u_%u = _mm_add_ps(c%u_%u, _mm_mul_ps(a%u_%u, b%u));\n", i_k, i_z, i_k, i_z, i_k, i_z, i_k );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    _mm_storeu_ps(&C[(l_n*%u)+%u], c%u_%u);\n", (unsigned int)i_xgemm_desc->ldc, i_row_idx[i_column_idx[i_k] + i_z], i_k, i_z );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_spgemm_csc_asparse( libxsmm_generated_code*        io_generated_code,
                                           const libxsmm_gemm_descriptor* i_xgemm_desc,
                                           const char*                    i_arch,
                                           const unsigned int*            i_row_idx,
                                           const unsigned int*            i_column_idx,
                                           const double*                  i_values ) {
  char l_new_code[512];
  int l_max_code_length = 511;
  int l_code_length = 0;
  unsigned int l_k;
  unsigned int l_flop_count = 0;

  LIBXSMM_UNUSED(i_arch);
  LIBXSMM_UNUSED(i_values);

  /* loop over columns in C in generated code, we fully unroll inside each column */
  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  unsigned int l_n = 0;\n  #pragma nounroll_and_jam\n  for ( l_n = 0; l_n < %u; l_n++) {\n", (unsigned int)i_xgemm_desc->n);
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

  /* reset the current column in C if needed */
  if ( i_xgemm_desc->beta == 0 ) {
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    unsigned int l_m = 0;\n");
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    if ( i_xgemm_desc->m > 1 ) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "   #pragma simd\n");
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    }
    if ( (LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) == 0 ) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    for ( l_m = 0; l_m < %u; l_m++) {\n      C[(l_n*%u)+l_m] = 0.0;\n    }\n", (unsigned int)i_xgemm_desc->m, (unsigned int)i_xgemm_desc->ldc);
    } else {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    for ( l_m = 0; l_m < %u; l_m++) {\n      C[(l_n*%u)+l_m] = 0.0f;\n    }\n", (unsigned int)i_xgemm_desc->m, (unsigned int)i_xgemm_desc->ldc);
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }

  /* loop over columns in A, rows in B and fully unroll */
  for ( l_k = 0; l_k < (unsigned int)i_xgemm_desc->k; l_k++ ) {
    unsigned int l_column_elements = i_column_idx[l_k + 1] - i_column_idx[l_k];
    unsigned int l_z = 0;

    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "#if defined(__SSE3__) || defined(__AVX__)\n");
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

    if ( l_column_elements > 0 ) {
      if ( (LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) == 0 ) {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "#if defined(__SSE3__) && defined(__AVX__)\n    __m256d b%u = _mm256_broadcast_sd(&B[(l_n*%u)+%u]);\n#endif\n", l_k, (unsigned int)i_xgemm_desc->ldb, l_k);
        libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "#if defined(__SSE3__) && !defined(__AVX__)\n    __m128d b%u = _mm_loaddup_pd(&B[(l_n*%u)+%u]);\n#endif\n", l_k, (unsigned int)i_xgemm_desc->ldb, l_k);
        libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      } else {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "#if defined(__SSE3__) && defined(__AVX__)\n    __m128 b%u = _mm_broadcast_ss(&B[(l_n*%u)+%u]);\n#endif\n", l_k, (unsigned int)i_xgemm_desc->ldb, l_k);
        libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "#if defined(__SSE3__) && !defined(__AVX__)\n    __m128 b%u = _mm_load_ss(&B[(l_n*%u)+%u]);    b%u = _mm_shuffle_ps(b%u, b%u, 0x00);\n#endif\n", l_k, (unsigned int)i_xgemm_desc->ldb, l_k, l_k, l_k, l_k);
        libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      }
    }

    /* loop over the columns of A and look for vectorization potential */
    for ( l_z = 0; l_z < l_column_elements; l_z++ ) {
      /* 4 element vector might be possible */
      if ( (l_z < (l_column_elements - 3)) && (l_column_elements > 3) ) {
        /* check for 256bit vector instruction */
        if ((i_row_idx[i_column_idx[l_k] + l_z] + 1 == i_row_idx[i_column_idx[l_k] + l_z + 1]) &&
            (i_row_idx[i_column_idx[l_k] + l_z] + 2 == i_row_idx[i_column_idx[l_k] + l_z + 2]) &&
            (i_row_idx[i_column_idx[l_k] + l_z] + 3 == i_row_idx[i_column_idx[l_k] + l_z + 3]) &&
            (i_row_idx[i_column_idx[l_k] + l_z + 3] < (unsigned int)i_xgemm_desc->m)) {
          libxsmm_sparse_csc_asparse_innerloop_four_vector(io_generated_code, i_xgemm_desc, l_k, l_z, i_row_idx, i_column_idx);
          l_z += 3;
        /* check for 128bit vector instruction */
        } else if ((i_row_idx[i_column_idx[l_k] + l_z] + 1 == i_row_idx[i_column_idx[l_k] + l_z + 1]) &&
                   (i_row_idx[i_column_idx[l_k] + l_z + 1] < (unsigned int)i_xgemm_desc->m) ) {
          libxsmm_sparse_csc_asparse_innerloop_two_vector(io_generated_code, i_xgemm_desc, l_k, l_z, i_row_idx, i_column_idx);
          l_z++;
        /* scalare instruction */
        } else {
          if ( (i_row_idx[i_column_idx[l_k] + l_z] < (unsigned int)i_xgemm_desc->m) ) {
            libxsmm_sparse_csc_asparse_innerloop_scalar(io_generated_code, i_xgemm_desc, l_k, l_z, i_row_idx, i_column_idx);
          }
        }
      /* 2 element vector might be possible */
      } else if ( (l_z < (l_column_elements - 1)) && (l_column_elements > 1)) {
        /* check for 128bit vector instruction */
        if ((i_row_idx[i_column_idx[l_k] + l_z] + 1 == i_row_idx[i_column_idx[l_k] + l_z + 1]) &&
            (i_row_idx[i_column_idx[l_k] + l_z + 1] < (unsigned int)i_xgemm_desc->m) ) {
          libxsmm_sparse_csc_asparse_innerloop_two_vector(io_generated_code, i_xgemm_desc, l_k, l_z, i_row_idx, i_column_idx);
          l_z++;
        /* scalare instruction */
        } else {
          if ( (i_row_idx[i_column_idx[l_k] + l_z] < (unsigned int)i_xgemm_desc->m) ) {
            libxsmm_sparse_csc_asparse_innerloop_scalar(io_generated_code, i_xgemm_desc, l_k, l_z, i_row_idx, i_column_idx);
          }
        }
      /* scalar anayways */
      } else {
        if ( (i_row_idx[i_column_idx[l_k] + l_z] < (unsigned int)i_xgemm_desc->m) ) {
          libxsmm_sparse_csc_asparse_innerloop_scalar(io_generated_code, i_xgemm_desc, l_k, l_z, i_row_idx, i_column_idx);
        }
      }
    }

    /* C fallback code */
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "#else\n");
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

    /* loop over the columns of A */
    for ( l_z = 0; l_z < l_column_elements; l_z++ ) {
      if ( (i_row_idx[i_column_idx[l_k] + l_z] < (unsigned int)i_xgemm_desc->m) ) {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    C[(l_n*%u)+%u] += A[%u] * B[(l_n*%u)+%u];\n", (unsigned int)i_xgemm_desc->ldc, i_row_idx[i_column_idx[l_k] + l_z], i_column_idx[l_k] + l_z, (unsigned int)i_xgemm_desc->ldb, l_k );
        libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
        l_flop_count += 2;
      }
    }

    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "#endif\n\n");
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }

  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  }\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

  /* add flop counter */
  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "\n#ifndef NDEBUG\n#ifdef _OPENMP\n#pragma omp atomic\n#endif\nlibxsmm_num_total_flops += %u;\n#endif\n", l_flop_count * i_xgemm_desc->n);
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
}

