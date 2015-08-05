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

#include "generator_dense_instructions.h"
#include "generator_dense_avx512_microkernel.h"

void libxsmm_generator_dense_avx512_microkernel( libxsmm_generated_code*             io_generated_code,
                                                 const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
                                                 const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                                 const libxsmm_xgemm_descriptor*     i_xgemm_desc,
                                                 const unsigned int                  i_m_blocking,
                                                 const unsigned int                  i_n_blocking,
                                                 const int                           i_offset ) {
#ifndef NDEBUG
  if ( i_n_blocking > 30 ) {
    fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_dense_avx512_microkernel: i_n_blocking exceeds 30\n");
    exit(-1); 
  }
  if ( i_m_blocking != i_micro_kernel_config->vector_length ) {
    fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_dense_avx512_microkernel: i_m_blocking doesn't match the vector_length\n");
    exit(-1); 
  }
#endif

  if (i_offset != (-1)) {
    libxsmm_instruction_vec_move( io_generated_code,
                                  i_micro_kernel_config->instruction_set,
                                  i_micro_kernel_config->a_vmove_instruction, 
                                  i_gp_reg_mapping->gp_reg_a, 
                                  i_xgemm_desc->lda * i_offset * i_micro_kernel_config->datatype_size, 
                                  i_micro_kernel_config->vector_name, 
                                  0, 
                                  i_micro_kernel_config->use_masking_a_c, 0 );

#if 0
    // current A prefetch, next 8 rows for the current column
    if (    (tPrefetch.compare("curAL2") == 0) 
         || (tPrefetch.compare("curAL2_BL2viaC") == 0) ) {
      codestream << "                         \"prefetcht1 " << (lda * 8 * call) + 64 << "(%%r9)\\n\\t\"" << std::endl;
    }
    // next A prefetch "same" rows in "same" column, but in a different matrix 
    if (    (tPrefetch.compare("AL2jpst") == 0)
         || (tPrefetch.compare("AL2jpst_BL2viaC") == 0)
         || (tPrefetch.compare("AL2") == 0)
         || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
      codestream << "                         \"prefetcht1 " << (lda * 8 * call) << "(%%r11)\\n\\t\"" << std::endl;
    }
    for (int n_local = 0; n_local < max_local_N; n_local++) {
      codestream << "                         \"vfmadd231pd " << (8 * call) + (ldb * 8 * n_local) << "(%%r8){{1to8}}, %%zmm0, %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
    }
#endif
  } else {
    libxsmm_instruction_vec_move( io_generated_code,
                                  i_micro_kernel_config->instruction_set,
                                  i_micro_kernel_config->a_vmove_instruction, 
                                  i_gp_reg_mapping->gp_reg_a, 
                                  i_xgemm_desc->lda * i_offset * i_micro_kernel_config->datatype_size, 
                                  i_micro_kernel_config->vector_name, 
                                  0, 
                                  i_micro_kernel_config->use_masking_a_c, 0 );
#if 0
    // current A prefetch, next 8 rows for the current column
    if (    (tPrefetch.compare("curAL2") == 0) 
         || (tPrefetch.compare("curAL2_BL2viaC") == 0) ) {
      codestream << "                         \"prefetcht1 64(%%r9)\\n\\t\"" << std::endl;
    }
    // next A prefetch "same" rows in "same" column, but in a different matrix 
    if (    (tPrefetch.compare("AL2jpst") == 0) 
         || (tPrefetch.compare("AL2jpst_BL2viaC") == 0)
         || (tPrefetch.compare("AL2") == 0)
         || (tPrefetch.compare("AL2_BL2viaC") == 0) ) {
      codestream << "                         \"prefetcht1 (%%r11)\\n\\t\"" << std::endl;
      codestream << "                         \"addq $" << lda * 8 << ", %%r11\\n\\t\"" << std::endl;
    }
    codestream << "                         \"addq $" << lda * 8 << ", %%r9\\n\\t\"" << std::endl;
    for (int n_local = 0; n_local < max_local_N; n_local++) {
      codestream << "                         \"vfmadd231pd " << (ldb * 8 * n_local) << "(%%r8){{1to8}}, %%zmm0, %%zmm" << 31-n_local << "\\n\\t\"" << std::endl;
    }
    codestream << "                         \"addq $8, %%r8\\n\\t\"" << std::endl;
#endif
  }
}

