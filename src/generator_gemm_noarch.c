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
#include "generator_gemm_noarch.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_noarch_kernel( libxsmm_generated_code*        io_generated_code,
                                           const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  char l_new_code[512];
  int l_max_code_length = 511;
  int l_code_length = 0;

  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  unsigned int l_m = 0;\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  unsigned int l_n = 0;\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  unsigned int l_k = 0;\n\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  for ( l_n = 0; l_n < %u; l_n++ ) {\n", (unsigned int)i_xgemm_desc->n);
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  if (0 != (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) { /* Beta=0 */
    if ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )  ) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    for ( l_m = 0; l_m < %u; l_m++ ) { C[(l_n*%u)+l_m] = 0.0; }\n\n", (unsigned int)i_xgemm_desc->m, (unsigned int)i_xgemm_desc->ldc);
    } else {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    for ( l_m = 0; l_m < %u; l_m++ ) { C[(l_n*%u)+l_m] = 0.0f; }\n\n", (unsigned int)i_xgemm_desc->m, (unsigned int)i_xgemm_desc->ldc);
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    for ( l_k = 0; l_k < %u; l_k++ ) {\n", (unsigned int)i_xgemm_desc->k);
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "      #pragma simd\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "      for ( l_m = 0; l_m < %u; l_m++ ) {\n", (unsigned int)i_xgemm_desc->m);
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "        C[(l_n*%u)+l_m] += A[(l_k*%u)+l_m] * B[(l_n*%u)+l_k];\n", (unsigned int)i_xgemm_desc->ldc, (unsigned int)i_xgemm_desc->lda, (unsigned int)i_xgemm_desc->ldb);
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "      }\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    }\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  }\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
}

