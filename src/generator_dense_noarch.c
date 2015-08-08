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

#include "generator_dense_noarch.h"

void libxsmm_generator_dense_noarch_kernel( libxsmm_generated_code*         io_generated_code,
                                            const libxsmm_xgemm_descriptor* i_xgemm_desc,
                                            const char*                     i_arch ) {
  char l_new_code[512];

  sprintf(l_new_code, "  unsigned int l_m = 0;\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code );
  sprintf(l_new_code, "  unsigned int l_n = 0;\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code );
  sprintf(l_new_code, "  unsigned int l_k = 0;\n\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code );

  sprintf(l_new_code, "  for ( l_n = 0; l_n < %i; l_n++ ) {\n", i_xgemm_desc->n);
  libxsmm_append_code_as_string( io_generated_code, l_new_code );
  if ( i_xgemm_desc->beta == 0 ) {
    sprintf(l_new_code, "    for ( l_m = 0; l_m < %i; l_m++ ) { C[(l_n*%i)+l_m] = 0.0; }\n\n", i_xgemm_desc->m, i_xgemm_desc->ldc);
    libxsmm_append_code_as_string( io_generated_code, l_new_code );
  }
  sprintf(l_new_code, "    for ( l_k = 0; l_k < %i; l_k++ ) {\n", i_xgemm_desc->k);
  libxsmm_append_code_as_string( io_generated_code, l_new_code );
  sprintf(l_new_code, "      #pragma simd\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code );
  sprintf(l_new_code, "      for ( l_m = 0; l_m < %i; l_m++ ) {\n", i_xgemm_desc->m);
  libxsmm_append_code_as_string( io_generated_code, l_new_code );
  sprintf(l_new_code, "        C[(l_n*%i)+l_m] += A[(l_k*%i)+l_m] * B[(l_n*%i)+l_k];\n", i_xgemm_desc->ldc, i_xgemm_desc->lda, i_xgemm_desc->ldb);
  libxsmm_append_code_as_string( io_generated_code, l_new_code );
  sprintf(l_new_code, "      }\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code );
  sprintf(l_new_code, "    }\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code );
  sprintf(l_new_code, "  }\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code );
}

