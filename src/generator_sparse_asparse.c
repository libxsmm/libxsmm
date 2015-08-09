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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>

#include "generator_common.h"
#include "generator_sparse_asparse.h"

void libxsmm_generator_sparse_asparse( libxsmm_generated_code*         io_generated_code,
                                       const libxsmm_xgemm_descriptor* i_xgemm_desc,
                                       const char*                     i_arch, 
                                       const unsigned int*             i_row_idx,
                                       const unsigned int*             i_column_idx,
                                       const double*                   i_values ) {
  char l_new_code[512];
  unsigned int l_k;
  unsigned int l_flop_count = 0;

  /* loop over columns in C in generated code, we fully unroll inside each column */
  sprintf(l_new_code, "  #pragma nounroll_and_jam\n  unsigned int l_n = 0;\n  for ( l_n = 0; l_n < %u; l_n++) {", i_xgemm_desc->n);
  libxsmm_append_code_as_string( io_generated_code, l_new_code );

  /* reset the current column in C if needed */
  if ( i_xgemm_desc->beta == 0 ) {
    sprintf(l_new_code, "    unsigned int l_m = 0;\n");
    libxsmm_append_code_as_string( io_generated_code, l_new_code );
    if ( i_xgemm_desc->m > 1 ) {
      sprintf(l_new_code, "   #pragma simd;\n");
      libxsmm_append_code_as_string( io_generated_code, l_new_code );
    }
    if ( i_xgemm_desc->single_precision == 0 ) {  
      sprintf(l_new_code, "    for ( l_m = 0; l_m < %u; l_m++) {\n      C[(l_n*%u)+l_m] = 0.0;\n    }\n", i_xgemm_desc->m, i_xgemm_desc->ldc);
    } else {
      sprintf(l_new_code, "    for ( l_m = 0; l_m < %u; l_m++) {\n      C[(l_n*%u)+l_m] = 0.0f;\n    }\n", i_xgemm_desc->m, i_xgemm_desc->ldc);
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code );
  }
  
  /* loop over columns in A, rows in B and fully unroll */
  for ( l_k = 0; l_k < i_xgemm_desc->k; l_k++ ) {
    unsigned int l_column_elements = i_column_idx[l_k + 1] - i_column_idx[l_k];
    unsigned int l_z = 0;

    sprintf(l_new_code, "#if defined(__SSE3__) && defined(__AVX__)\n");
    libxsmm_append_code_as_string( io_generated_code, l_new_code );
    
    if ( l_column_elements > 0 ) {
      if ( i_xgemm_desc->single_precision == 0 ) {
        sprintf(l_new_code, "#if defined(__SSE3__) && defined(__AVX__)\n    __m256d b%u = _mm256_broadcast_sd(&B[(l_n*%u)+%u]);\n#endif\n", l_k, i_xgemm_desc->ldb, l_k);
        libxsmm_append_code_as_string( io_generated_code, l_new_code );
        sprintf(l_new_code, "#if defined(__SSE3__) && !defined(__AVX__)\n    __m128d b%u = _mm_loaddup_pd(&B[(l_n*%u)+%u]);\n#endif\n", l_k, i_xgemm_desc->ldb, l_k);
        libxsmm_append_code_as_string( io_generated_code, l_new_code );
      } else {
        sprintf(l_new_code, "#if defined(__SSE3__) && defined(__AVX__)\n    __m256 b%u = _mm256_broadcast_ss(&B[(l_n*%u)+%u]);\n#endif\n", l_k, i_xgemm_desc->ldb, l_k);
        libxsmm_append_code_as_string( io_generated_code, l_new_code );
        sprintf(l_new_code, "#if defined(__SSE3__) && !defined(__AVX__)\n    __m128 b%u = _mm_load_ss(&B[(l_n*%u)+%u]);    b%u = _mm_shuffle_ps(b%u, b%u, 0x00);\n#endif\n", l_k, i_xgemm_desc->ldb, l_k, l_k, l_k, l_k);
        libxsmm_append_code_as_string( io_generated_code, l_new_code );
      }
    }

    /* loop over the columns of A and look for vectorization potential */
    for ( l_z = 0; l_z < l_column_elements; l_z++ ) {
      /* 4 element vector might be possible */
      if ( l_z < (l_column_elements - 3) ) {
        /* check for 256bit vector instruction */
        if ((i_row_idx[i_column_idx[l_k] + l_z] + 1 == i_row_idx[i_column_idx[l_k] + l_z + 1]) &&
            (i_row_idx[i_column_idx[l_k] + l_z] + 2 == i_row_idx[i_column_idx[l_k] + l_z + 2]) &&
            (i_row_idx[i_column_idx[l_k] + l_z] + 3 == i_row_idx[i_column_idx[l_k] + l_z + 3]) && 
            (i_row_idx[i_column_idx[l_k] + l_z + 3] < i_xgemm_desc->m)) {
          /*generate_code_left_innerloop_4vector(codestream, ldc, l, z, rowidx, colidx);*/
          l_z += 3;
        /* check for 128bit vector instruction */
        } else if ((i_row_idx[i_column_idx[l_k] + l_z] + 1 == i_row_idx[i_column_idx[l_k] + l_z + 1]) &&
                   (i_row_idx[i_column_idx[l_k] + l_z + 1] < i_xgemm_desc->m) ) {
          /* generate_code_left_innerloop_2vector(codestream, ldc, l, z, rowidx, colidx);*/
          l_z++;
        /* scalare instruction */
        } else {
          if ( (i_row_idx[i_column_idx[l_k] + l_z] < i_xgemm_desc->m) ) {
            /* generate_code_left_innerloop_scalar(codestream, ldc, l, z, rowidx, colidx); */
          }
        }
      /* 2 element vector might be possible */
      } else if ( l_z < (l_column_elements - 1) ) {
        /* check for 128bit vector instruction */
        if ((i_row_idx[i_column_idx[l_k] + l_z] + 1 == i_row_idx[i_column_idx[l_k] + l_z + 1]) &&
            (i_row_idx[i_column_idx[l_k] + l_z + 1] < i_xgemm_desc->m) ) {
          /* generate_code_left_innerloop_2vector(codestream, ldc, l, z, rowidx, colidx);*/
          l_z++;
        /* scalare instruction */
        } else {
          if ( (i_row_idx[i_column_idx[l_k] + l_z] < i_xgemm_desc->m) ) {
            /* generate_code_left_innerloop_scalar(codestream, ldc, l, z, rowidx, colidx); */
          }
        }
      /* scalar anayways */
      } else {
        if ( (i_row_idx[i_column_idx[l_k] + l_z] < i_xgemm_desc->m) ) {
           /* generate_code_left_innerloop_scalar(codestream, ldc, l, z, rowidx, colidx); */
        }
      }
    }

    /* C fallback code */
    sprintf(l_new_code, "#else\n");
    libxsmm_append_code_as_string( io_generated_code, l_new_code );

    /* loop over the columns of A */
    for ( l_z = 0; l_z < l_column_elements; l_z++ ) {
      if ( (i_row_idx[i_column_idx[l_k] + l_z] < i_xgemm_desc->m) ) {
        sprintf(l_new_code, "    C[(l_n*%u)+%u] += A[%u] * B[(l_n*%u)+%u];\n", i_xgemm_desc->ldc, i_row_idx[i_column_idx[l_k] + l_z], i_column_idx[l_k] + l_z, i_xgemm_desc->ldb, l_k );
        libxsmm_append_code_as_string( io_generated_code, l_new_code );
        l_flop_count += 2;
      }
    }

    sprintf(l_new_code, "#endif\n\n");
    libxsmm_append_code_as_string( io_generated_code, l_new_code );
  }

  sprintf(l_new_code, "  }\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code );

  /* add flop counter */
  sprintf(l_new_code, "\n#ifndef NDEBUG\n#ifdef _OPENMP\n#pragma omp atomic\n#endif\nlibxsmm_num_total_flops += %u;\n#endif\n", l_flop_count * i_xgemm_desc->n);
  libxsmm_append_code_as_string( io_generated_code, l_new_code );
}

