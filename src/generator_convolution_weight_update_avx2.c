/******************************************************************************
** Copyright (c) 2016, Intel Corporation                                     **
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
/* Rajkishore Barik (Intel Corp.), Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include "generator_convolution_weight_update_avx2.h"
#include "generator_convolution_common.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"

#include <libxsmm_cpuid.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_weight_update_avx2_kernel( libxsmm_generated_code*                             io_generated_code,
                                                              const libxsmm_convolution_weight_update_descriptor* i_conv_desc,
                                                              const char*                                         i_arch ) {
/***** Code to generate
Example for ofm_block = 32, ifm_block%2 == 0

  for(ifm2 = 0; ifm2 < handle->ifmblock; ifm2+=2) {
    __m256 acc00 = _mm256_load_ps(&weight[ofm1][ifm1][kj][ki][ifm2][0]);
    __m256 acc01 = _mm256_load_ps(&weight[ofm1][ifm1][kj][ki][ifm2][8]);
    __m256 acc02 = _mm256_load_ps(&weight[ofm1][ifm1][kj][ki][ifm2][16]);
    __m256 acc03 = _mm256_load_ps(&weight[ofm1][ifm1][kj][ki][ifm2][24]);
    __m256 acc10 = _mm256_load_ps(&weight[ofm1][ifm1][kj][ki][ifm2+1][0]);
    __m256 acc11 = _mm256_load_ps(&weight[ofm1][ifm1][kj][ki][ifm2+1][8]);
    __m256 acc12 = _mm256_load_ps(&weight[ofm1][ifm1][kj][ki][ifm2+1][16]);
    __m256 acc13 = _mm256_load_ps(&weight[ofm1][ifm1][kj][ki][ifm2+1][24]);
    for(ij=0, oj=0; oj < handle->ofh; ij+=stride_h, oj++) {
      for(ii=0, oi=0; oi < handle->ofw; ii+=stride_w, oi++) {
        __m256 out0 = _mm256_load_ps(&output[img][ofm1][oj][oi][0]);
        __m256 out1 = _mm256_load_ps(&output[img][ofm1][oj][oi][8]);
        __m256 out2 = _mm256_load_ps(&output[img][ofm1][oj][oi][16]);
        __m256 out3 = _mm256_load_ps(&output[img][ofm1][oj][oi][24]);
        __m256 in0 = _mm256_set1_ps(input[img][ifm1][ij+kj][ii+ki][ifm2]);
        acc00 = _mm256_fmadd_ps(in0, out0, acc00);
        acc01 = _mm256_fmadd_ps(in0, out1, acc01);
        acc02 = _mm256_fmadd_ps(in0, out2, acc02);
        acc03 = _mm256_fmadd_ps(in0, out3, acc03);
        __m256 in1 = _mm256_set1_ps(input[img][ifm1][ij+kj][ii+ki][ifm2+1]);
        acc10 = _mm256_fmadd_ps(in1, out0, acc10);
        acc11 = _mm256_fmadd_ps(in1, out1, acc11);
        acc12 = _mm256_fmadd_ps(in1, out2, acc12);
        acc13 = _mm256_fmadd_ps(in1, out3, acc13);
      }
    }
    _mm256_store_ps( &weight[ofm1][ifm1][kj][ki][ifm2][0], acc00 );
    _mm256_store_ps( &weight[ofm1][ifm1][kj][ki][ifm2][8], acc01 );
    _mm256_store_ps( &weight[ofm1][ifm1][kj][ki][ifm2][16], acc02 );
    _mm256_store_ps( &weight[ofm1][ifm1][kj][ki][ifm2][24], acc03 );
    _mm256_store_ps( &weight[ofm1][ifm1][kj][ki][ifm2+1][0], acc10 );
    _mm256_store_ps( &weight[ofm1][ifm1][kj][ki][ifm2+1][8], acc11 );
    _mm256_store_ps( &weight[ofm1][ifm1][kj][ki][ifm2+1][16], acc12 );
    _mm256_store_ps( &weight[ofm1][ifm1][kj][ki][ifm2+1][24], acc13 );
  }
*****/
  libxsmm_convolution_kernel_config l_conv_kernel_config = { 0 };
  libxsmm_convolution_weight_update_gp_reg_mapping l_gp_reg_mapping;
  libxsmm_loop_label_tracker l_loop_label_tracker;

  unsigned int l_ifm_blocking = 1;
  unsigned int l_ofm_blocking = 0;
  unsigned int l_vec_reg_acc_start = 16;
  unsigned int l_found_act_format = 0;
  unsigned int l_found_fil_format = 0;
  unsigned int l_m, l_n;

  /* define gp register mapping */
  /* NOTE: do not use RSP, RBP,
     do not use don't use R12 and R13 for addresses will add 4 bytes to the instructions as they are in the same line as rsp and rbp */
  libxsmm_reset_x86_convolution_weight_update_gp_reg_mapping( &l_gp_reg_mapping );
  l_gp_reg_mapping.gp_reg_weight = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_input = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_output = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_weight_pf = LIBXSMM_X86_GP_REG_R8;
  l_gp_reg_mapping.gp_reg_output_pf = LIBXSMM_X86_GP_REG_R9;
  l_gp_reg_mapping.gp_reg_input_pf = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_oi_loop = LIBXSMM_X86_GP_REG_R15;
  l_gp_reg_mapping.gp_reg_oj_loop = LIBXSMM_X86_GP_REG_R12;
  l_gp_reg_mapping.gp_reg_ifmInner_loop = LIBXSMM_X86_GP_REG_RAX;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_3 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_4 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_5 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_6 = LIBXSMM_X86_GP_REG_UNDEF;

  /* define convolution kernel config */
  libxsmm_generator_init_convolution_kernel_config( &l_conv_kernel_config );
  if ( strcmp( i_arch, "knl" ) == 0 ||
       strcmp( i_arch, "skx" ) == 0 ||
       strcmp( i_arch, "hsw" ) == 0  ) {
    l_conv_kernel_config.instruction_set = LIBXSMM_X86_AVX2;
  } else {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
    return;
  }
  l_conv_kernel_config.vector_reg_count = 16;
  l_conv_kernel_config.vector_length_in = 8;
  l_conv_kernel_config.datatype_size_in = 4;
  l_conv_kernel_config.vector_length_out = 8;
  l_conv_kernel_config.datatype_size_out = 4;
  l_conv_kernel_config.vector_length_wt = 8;
  l_conv_kernel_config.datatype_size_wt = 4;
  l_conv_kernel_config.vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
  l_conv_kernel_config.vfma_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
  l_conv_kernel_config.vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
  l_conv_kernel_config.vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
  l_conv_kernel_config.prefetch_instruction = LIBXSMM_X86_INSTR_PREFETCHT2;
  l_conv_kernel_config.alu_add_instruction = LIBXSMM_X86_INSTR_ADDQ;
  l_conv_kernel_config.alu_sub_instruction = LIBXSMM_X86_INSTR_SUBQ;
  l_conv_kernel_config.alu_cmp_instruction = LIBXSMM_X86_INSTR_CMPQ;
  l_conv_kernel_config.alu_jmp_instruction = LIBXSMM_X86_INSTR_JL;
  l_conv_kernel_config.alu_mov_instruction = LIBXSMM_X86_INSTR_MOVQ;
  l_conv_kernel_config.vector_name = 'y';
  /* calculate leading dimension depending on format */
  if ( (i_conv_desc->format & LIBXSMM_DNN_CONV_FORMAT_LIBXSMM) > 0 ) {
    l_conv_kernel_config.l_ld_ifm_act = i_conv_desc->ifm_block;
    l_conv_kernel_config.l_ld_ofm_act = i_conv_desc->ofm_block;
    l_conv_kernel_config.l_ld_ifm_fil = i_conv_desc->ifm_block;
    l_conv_kernel_config.l_ld_ofm_fil = i_conv_desc->ofm_block;
    l_found_act_format = 1;
    l_found_fil_format = 1;
  }
  if ( (i_conv_desc->format & LIBXSMM_DNN_CONV_FORMAT_NHWC) > 0 ) {
    l_conv_kernel_config.l_ld_ifm_act = i_conv_desc->ifm_block * i_conv_desc->blocks_ifm;
    l_conv_kernel_config.l_ld_ofm_act = i_conv_desc->ofm_block * i_conv_desc->blocks_ofm;
    l_found_act_format = 1;
  }
  if ( (i_conv_desc->format & LIBXSMM_DNN_CONV_FORMAT_RSCK) > 0 ) {
    l_conv_kernel_config.l_ld_ifm_fil = i_conv_desc->ifm_block * i_conv_desc->blocks_ifm;
    l_conv_kernel_config.l_ld_ofm_fil = i_conv_desc->ofm_block * i_conv_desc->blocks_ofm;
    l_found_fil_format = 1;
  }
  if ( (l_found_act_format == 0) || (l_found_fil_format == 0) ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_UNSUP_CONV_FORMAT );
    return;
  }

  /* check if we have full vectors */
  if ( i_conv_desc->ofm_block % l_conv_kernel_config.vector_length_out != 0 ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_CONV_OFM_VEC );
    return;
  }

  /* caclulate the ifm unrolling */
  for (l_m = 3; l_m > 0; l_m--) {
    if ( i_conv_desc->ifm_block%l_m == 0 ) {
      l_ifm_blocking = l_m;
      break;
    }
  }

  /* calculate blocking */
  l_ofm_blocking = i_conv_desc->ofm_block / l_conv_kernel_config.vector_length_out;
  l_vec_reg_acc_start = 16 - (l_ofm_blocking * l_ifm_blocking);

  /* check accumulator size */
  if ( l_ofm_blocking*l_ifm_blocking > l_conv_kernel_config.vector_reg_count-4 ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_INVALID_CONV_ACC );
    return;
  }

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* open asm */
  libxsmm_x86_instruction_open_stream_convolution( io_generated_code, l_gp_reg_mapping.gp_reg_input,
                                                   l_gp_reg_mapping.gp_reg_weight, l_gp_reg_mapping.gp_reg_output,
                                                   l_gp_reg_mapping.gp_reg_input_pf, l_gp_reg_mapping.gp_reg_weight_pf,
                                                   l_gp_reg_mapping.gp_reg_output_pf, i_arch );

  /* start ifm loop */
  libxsmm_generator_convolution_header_ifm_loop( io_generated_code, &l_loop_label_tracker, &l_conv_kernel_config,
                                                   l_gp_reg_mapping.gp_reg_ifmInner_loop, l_ifm_blocking );

  /* load weights */
  for ( l_n = 0; l_n < l_ifm_blocking; l_n++ ) {
    for ( l_m = 0; l_m < l_ofm_blocking; l_m++ ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        l_conv_kernel_config.instruction_set,
                                        l_conv_kernel_config.vmove_instruction,
                                        l_gp_reg_mapping.gp_reg_weight,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        (l_m * l_conv_kernel_config.vector_length_wt * l_conv_kernel_config.datatype_size_wt) +
                                          (l_n * l_conv_kernel_config.l_ld_ofm_fil * l_conv_kernel_config.datatype_size_wt),
                                        l_conv_kernel_config.vector_name,
                                        l_vec_reg_acc_start + l_m + (l_n * l_ofm_blocking), 0, 0 );
    }
  }

  /* loop over ofh and ofw and compute weight update for blocked weights */
  libxsmm_generator_convolution_weight_update_avx2_ofhofwloops( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_conv_kernel_config,
                                                                  i_conv_desc, l_ifm_blocking, l_ofm_blocking, l_vec_reg_acc_start );

  /* store weights */
  for ( l_n = 0; l_n < l_ifm_blocking; l_n++ ) {
    for ( l_m = 0; l_m < l_ofm_blocking; l_m++ ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        l_conv_kernel_config.instruction_set,
                                        l_conv_kernel_config.vmove_instruction,
                                        l_gp_reg_mapping.gp_reg_weight,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        (l_m * l_conv_kernel_config.vector_length_wt * l_conv_kernel_config.datatype_size_wt) +
                                          (l_n * l_conv_kernel_config.l_ld_ofm_fil * l_conv_kernel_config.datatype_size_wt),
                                        l_conv_kernel_config.vector_name,
                                        l_vec_reg_acc_start + l_m + (l_n * l_ofm_blocking), 0, 1 );
    }
  }

  /* advance weight and input pointers */
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_add_instruction,
                                     l_gp_reg_mapping.gp_reg_weight, l_ifm_blocking * l_conv_kernel_config.l_ld_ofm_fil * l_conv_kernel_config.datatype_size_wt );

  libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_add_instruction,
                                     l_gp_reg_mapping.gp_reg_input, l_ifm_blocking * l_conv_kernel_config.datatype_size_in );

  /* close ifm loop */
  libxsmm_generator_convolution_footer_ifm_loop( io_generated_code, &l_loop_label_tracker, &l_conv_kernel_config,
                                                   l_gp_reg_mapping.gp_reg_ifmInner_loop, i_conv_desc->ifm_block );

  /* close asm */
  libxsmm_x86_instruction_close_stream_convolution( io_generated_code, i_arch );
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_weight_update_avx2_ofhofwloops( libxsmm_generated_code*                                 io_generated_code,
                                                                   libxsmm_loop_label_tracker*                             io_loop_label_tracker,
                                                                   const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
                                                                   const libxsmm_convolution_kernel_config*                i_conv_kernel_config,
                                                                   const libxsmm_convolution_weight_update_descriptor*     i_conv_desc,
                                                                   const unsigned int                                      i_ifm_blocking,
                                                                   const unsigned int                                      i_ofm_blocking,
                                                                   const unsigned int                                      i_vec_reg_acc_start ) {
  unsigned int l_n, l_m;

  /* start ofh loop */
  libxsmm_generator_convolution_header_oj_loop( io_generated_code, io_loop_label_tracker, i_conv_kernel_config, i_gp_reg_mapping->gp_reg_oj_loop );

  /* start ofw loop */
  libxsmm_generator_convolution_header_oi_loop( io_generated_code, io_loop_label_tracker, i_conv_kernel_config, i_gp_reg_mapping->gp_reg_oi_loop );


  /* broadcast input values into registers 0 -> l_ifm_blocking-1 */
  for ( l_n = 0; l_n < i_ifm_blocking; l_n++ ) {
    libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_conv_kernel_config->instruction_set,
                                      LIBXSMM_X86_INSTR_VBROADCASTSS,
                                      i_gp_reg_mapping->gp_reg_input,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      l_n * i_conv_kernel_config->datatype_size_in,
                                      i_conv_kernel_config->vector_name,
                                      l_n, 0, 0 );
  }

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_input, i_conv_desc->stride_w * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in );

  for ( l_m = 0; l_m < i_ofm_blocking; l_m++ ) {
    /* load weights */
    libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_conv_kernel_config->instruction_set,
                                      i_conv_kernel_config->vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_output,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      l_m*i_conv_kernel_config->vector_length_out*i_conv_kernel_config->datatype_size_out,
                                      i_conv_kernel_config->vector_name, i_ifm_blocking,
                                      0, 0  );

    /* advance output pointer */
    if ( l_m == (i_ofm_blocking-1) ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                         i_gp_reg_mapping->gp_reg_output, i_conv_kernel_config->l_ld_ofm_act * i_conv_kernel_config->datatype_size_out );
    }

    /* compute */
    for ( l_n = 0; l_n < i_ifm_blocking; l_n++ ) {
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_conv_kernel_config->instruction_set,
                                               i_conv_kernel_config->vfma_instruction,
                                               i_conv_kernel_config->vector_name,
                                               i_ifm_blocking,
                                               l_n,
                                               i_vec_reg_acc_start + l_m + (l_n * i_ofm_blocking) );
    }
  }

  /* close ofw loop */
  libxsmm_generator_convolution_footer_oi_loop( io_generated_code, io_loop_label_tracker, i_conv_kernel_config, i_gp_reg_mapping->gp_reg_oi_loop, i_conv_desc->ofw );

  /* advance input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_input,
                                     (i_conv_desc->stride_h * i_conv_desc->ifw_padded * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in)
                                        - (i_conv_desc->ofw * i_conv_desc->stride_w * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in) );

  /* advance output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_output,
                                     (i_conv_desc->ofw_padded * i_conv_kernel_config->l_ld_ofm_act * i_conv_kernel_config->datatype_size_out)
                                        - (i_conv_desc->ofw * i_conv_kernel_config->l_ld_ofm_act * i_conv_kernel_config->datatype_size_out) );

  /* close ofh loop */
  libxsmm_generator_convolution_footer_oj_loop( io_generated_code, io_loop_label_tracker, i_conv_kernel_config, i_gp_reg_mapping->gp_reg_oj_loop, i_conv_desc->ofh );

  /* reset input pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_input,
                                     i_conv_desc->ofh * i_conv_desc->stride_h * i_conv_desc->ifw_padded
                                       * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in );

  /* reset output pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_output,
                                     i_conv_desc->ofh * i_conv_desc->ofw_padded * i_conv_kernel_config->l_ld_ofm_act * i_conv_kernel_config->datatype_size_out );
}
