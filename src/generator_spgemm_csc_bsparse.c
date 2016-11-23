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
#include "generator_spgemm_csc_bsparse.h"
#include "generator_common.h"
#include <libxsmm_macros.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_spgemm_csc_bsparse( libxsmm_generated_code*         io_generated_code,
                                           const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                           const char*                     i_arch,
                                           const unsigned int*             i_row_idx,
                                           const unsigned int*             i_column_idx,
                                           const double*                   i_values ) {
  unsigned int l_n;
  unsigned int l_z;
  unsigned int l_column_elements;
  unsigned int l_flop_count = 0;

  char l_new_code[512];
  int l_max_code_length = 511;
  int l_code_length = 0;

  LIBXSMM_UNUSED(i_values);

  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  unsigned int l_m = 0;\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

  /* reset C if beta is zero */
  if ( i_xgemm_desc->beta == 0 ) {
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  unsigned int l_n = 0;\n");
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  for ( l_n = 0; l_n < %u; l_n++) {\n", (unsigned int)i_xgemm_desc->n);
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    if ( i_xgemm_desc->m > 1 ) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    #pragma simd\n");
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    #pragma vector aligned\n");
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    }
    if ( (LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) == 0 ) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    for ( l_m = 0; l_m < %u; l_m++) { C[(l_n*%u)+l_m] = 0.0; }\n", (unsigned int)i_xgemm_desc->m, (unsigned int)i_xgemm_desc->ldc);
    } else {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    for ( l_m = 0; l_m < %u; l_m++) { C[(l_n*%u)+l_m] = 0.0f; }\n", (unsigned int)i_xgemm_desc->m, (unsigned int)i_xgemm_desc->ldc);
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  }\n");
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

  /* determine the correct simd pragma for each architecture */
  if ( ( strcmp( i_arch, "noarch" ) == 0 ) ||
       ( strcmp( i_arch, "wsm" ) == 0 )    ||
       ( strcmp( i_arch, "snb" ) == 0 )    ||
       ( strcmp( i_arch, "hsw" ) == 0 )       ) {
    if ( i_xgemm_desc->m > 7 ) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  #pragma simd vectorlength(8)\n");
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    } else if ( i_xgemm_desc->m > 3 ) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  #pragma simd vectorlength(4)\n");
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    } else if ( i_xgemm_desc->m > 1 ) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  #pragma simd vectorlength(2)\n");
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    } else {}

    if ( (i_xgemm_desc->m > 1)          &&
         ((LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0) &&
         ((LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0)    ) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  #pragma vector aligned\n");
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    }
  } else if ( ( strcmp( i_arch, "knc" ) == 0 ) ||
              ( strcmp( i_arch, "knl" ) == 0 ) ||
              ( strcmp( i_arch, "skx" ) == 0 )    ) {
    if ( (i_xgemm_desc->m > 1)          &&
         ((LIBXSMM_GEMM_FLAG_ALIGN_A & i_xgemm_desc->flags) != 0) &&
         ((LIBXSMM_GEMM_FLAG_ALIGN_C & i_xgemm_desc->flags) != 0)    ) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  #pragma simd vectorlength(32)\n  #pragma vector aligned\n");
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    }
  } else {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }

  /* generate the actuel kernel */
  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  for ( l_m = 0; l_m < %u; l_m++) {\n", (unsigned int)i_xgemm_desc->m);
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

  for ( l_n = 0; l_n < (unsigned int)i_xgemm_desc->n; l_n++ ) {
    l_column_elements = i_column_idx[l_n+1] - i_column_idx[l_n];
    for ( l_z = 0; l_z < l_column_elements; l_z++ ) {
      /* check k such that we just use rows which actually need to be multiplied */
      if ( i_row_idx[i_column_idx[l_n] + l_z] < (unsigned int)i_xgemm_desc->k ) {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    C[%u+l_m] += A[%u+l_m] * B[%u];\n", l_n * i_xgemm_desc->ldc, i_row_idx[i_column_idx[l_n] + l_z]*i_xgemm_desc->lda, i_column_idx[l_n] + l_z);
        libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
        l_flop_count += 2;
      }
    }
  }

  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  }\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

  /* add flop counter */
  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "\n#ifndef NDEBUG\n#ifdef _OPENMP\n#pragma omp atomic\n#endif\nlibxsmm_num_total_flops += %u;\n#endif\n", l_flop_count * (unsigned int)i_xgemm_desc->m);
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
}

