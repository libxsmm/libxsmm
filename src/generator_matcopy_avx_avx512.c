/******************************************************************************
** Copyright (c) 2017, Intel Corporation                                     **
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

#include "generator_matcopy_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"

#include <libxsmm_intrinsics_x86.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_matcopy_avx_avx512_kernel( libxsmm_generated_code*             io_generated_code,
                                                  const libxsmm_matcopy_descriptor*   i_matcopy_desc,
                                                  const char*                         i_arch ) {
  libxsmm_matcopy_kernel_config l_kernel_config = { 0/*avoid warning "maybe used uninitialized" */ };
  libxsmm_matcopy_gp_reg_mapping l_gp_reg_mapping = { 0/*avoid warning "maybe used uninitialized" */ };
  libxsmm_loop_label_tracker l_loop_label_tracker = { 0/*avoid warning "maybe used uninitialized" */ };

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define gp register mapping */
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_lda = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_ldb = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_a_pf = LIBXSMM_X86_GP_REG_R8;
  l_gp_reg_mapping.gp_reg_b_pf = LIBXSMM_X86_GP_REG_R9;
  l_gp_reg_mapping.gp_reg_m_loop = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_n_loop = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_3 = LIBXSMM_X86_GP_REG_UNDEF;

  /* define convolution kernel config */
  if ( strcmp( i_arch, "snb" ) == 0 ) {
    l_kernel_config.instruction_set = LIBXSMM_X86_AVX;
    l_kernel_config.vector_reg_count = 16;
    l_kernel_config.vector_name = 'y';
  } else if ( strcmp( i_arch, "hsw" ) == 0 ) {
    l_kernel_config.instruction_set = LIBXSMM_X86_AVX2;
    l_kernel_config.vector_reg_count = 16;
    l_kernel_config.vector_name = 'y';
  } else if ( strcmp( i_arch, "skx" ) == 0 ) {
    l_kernel_config.instruction_set = LIBXSMM_X86_AVX512_CORE;
    l_kernel_config.vector_reg_count = 32;
    l_kernel_config.vector_name = 'z';
  } else if ( strcmp( i_arch, "knl" ) == 0 ) {
    l_kernel_config.instruction_set = LIBXSMM_X86_AVX512_MIC;
    l_kernel_config.vector_reg_count = 32;
    l_kernel_config.vector_name = 'z';
  } else if ( strcmp( i_arch, "knm" ) == 0 ) {
    l_kernel_config.instruction_set = LIBXSMM_X86_AVX512_KNM;
    l_kernel_config.vector_reg_count = 32;
    l_kernel_config.vector_name = 'z';
  } else {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
    return;
  }
  /* @Evangelos add more fields here */

  /* open asm */
  libxsmm_x86_instruction_open_stream_matcopy( io_generated_code, l_gp_reg_mapping.gp_reg_a,
                                               l_gp_reg_mapping.gp_reg_lda, l_gp_reg_mapping.gp_reg_b,
                                               l_gp_reg_mapping.gp_reg_ldb, l_gp_reg_mapping.gp_reg_a_pf,
                                               l_gp_reg_mapping.gp_reg_b_pf, i_arch );

  /* @Evangelos add generator code here, please functions defined in generator_x86_instructions.h */

  /* close asm */
  libxsmm_x86_instruction_close_stream_matcopy( io_generated_code, i_arch );
}

