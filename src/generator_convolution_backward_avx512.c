/******************************************************************************
** Copyright (c) 2015-2017, Intel Corporation                                **
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
/* Ankush Mandal, Rajkishore Barik, Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include "generator_convolution_backward_avx512.h"
#include "generator_convolution_common.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"

#include <libxsmm_intrinsics_x86.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_backward_avx512_kernel( libxsmm_generated_code*           io_generated_code,
                                                           const libxsmm_convolution_backward_descriptor* i_conv_desc,
                                                           const char*                       i_arch ) {
  libxsmm_convolution_kernel_config l_conv_kernel_config = { 0/*avoid warning "maybe used uninitialized" */ };
  libxsmm_convolution_backward_gp_reg_mapping l_gp_reg_mapping;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  unsigned int l_kw_trips = 1;
  unsigned int l_kh_trips = 1;
  unsigned int l_kh = 0;
  unsigned int num_output_prefetch; /* used for distributing output prefetch in over kh loops */
  unsigned int l_found_act_format = 0;
  unsigned int l_found_fil_format = 0;
  /*unsigned int l_ofw_trips = 1;
  unsigned int oi = 0;*/
  /* define gp register mapping */
  /* NOTE: do not use RSP, RBP,
     do not use don't use R12 and R13 for addresses will add 4 bytes to the instructions as they are in the same line as rsp and rbp */
  libxsmm_reset_x86_convolution_backward_gp_reg_mapping( &l_gp_reg_mapping );
  l_gp_reg_mapping.gp_reg_input = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_weight = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_output = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_input_pf = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_weight_pf = LIBXSMM_X86_GP_REG_R8;
  l_gp_reg_mapping.gp_reg_output_pf = LIBXSMM_X86_GP_REG_R9;

  l_gp_reg_mapping.gp_reg_kw_loop = LIBXSMM_X86_GP_REG_R15;
  l_gp_reg_mapping.gp_reg_kh_loop = LIBXSMM_X86_GP_REG_R12;
  l_gp_reg_mapping.gp_reg_oi_loop = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_ofmInner_loop = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_X86_GP_REG_RAX;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_X86_GP_REG_RBX;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_X86_GP_REG_R10;
  l_gp_reg_mapping.gp_reg_help_3 = LIBXSMM_X86_GP_REG_R11;
  l_gp_reg_mapping.gp_reg_help_4 = LIBXSMM_X86_GP_REG_R13;
  l_gp_reg_mapping.gp_reg_help_5 = LIBXSMM_X86_GP_REG_R14;
  l_gp_reg_mapping.gp_reg_help_6 = LIBXSMM_X86_GP_REG_R15;

  /* define convolution kernel config */
  libxsmm_generator_init_convolution_kernel_config( &l_conv_kernel_config );
  if ( strcmp( i_arch, "knm" ) == 0 ) {
    l_conv_kernel_config.instruction_set = LIBXSMM_X86_AVX512_KNM;
  } else if ( strcmp( i_arch, "knl" ) == 0 ) {
    l_conv_kernel_config.instruction_set = LIBXSMM_X86_AVX512_MIC;
  } else if ( strcmp( i_arch, "skx" ) == 0 ) {
    l_conv_kernel_config.instruction_set = LIBXSMM_X86_AVX512_CORE;
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
    return;
  }
  l_conv_kernel_config.vector_reg_count = 32;
  if (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_F32 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32) {
    l_conv_kernel_config.vector_length_in = 16;
    l_conv_kernel_config.datatype_size_in = 4;
    l_conv_kernel_config.vector_length_out = 16;
    l_conv_kernel_config.datatype_size_out = 4;
    l_conv_kernel_config.vector_length_wt = 16;
    l_conv_kernel_config.datatype_size_wt = 4;
    l_conv_kernel_config.vfma_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
    l_conv_kernel_config.vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
    l_conv_kernel_config.vbcst_instruction = LIBXSMM_X86_INSTR_VBROADCASTSS;
  } else if (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32) {
    l_conv_kernel_config.vector_length_in = 16;
    l_conv_kernel_config.datatype_size_in = 4;
    l_conv_kernel_config.vector_length_out = 32;
    l_conv_kernel_config.datatype_size_out = 2;
    l_conv_kernel_config.vector_length_wt = 32;
    l_conv_kernel_config.datatype_size_wt = 2;
    l_conv_kernel_config.vfma_instruction = LIBXSMM_X86_INSTR_VPMADDWD;
    l_conv_kernel_config.vadd_instruction = LIBXSMM_X86_INSTR_VPADDD;
    l_conv_kernel_config.vbcst_instruction = LIBXSMM_X86_INSTR_VPBROADCASTD;
  } else if (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I8 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I16) {
    l_conv_kernel_config.vector_length_in = 32;
    l_conv_kernel_config.datatype_size_in = 2;
    l_conv_kernel_config.vector_length_out = 64;
    l_conv_kernel_config.datatype_size_out = 1;
    l_conv_kernel_config.vector_length_wt = 64;
    l_conv_kernel_config.datatype_size_wt = 1;
    l_conv_kernel_config.vfma_instruction = LIBXSMM_X86_INSTR_VPMADDUBSW;
    l_conv_kernel_config.vadd_instruction = LIBXSMM_X86_INSTR_VPADDW;
    l_conv_kernel_config.vbcst_instruction = LIBXSMM_X86_INSTR_VPBROADCASTW;
  } else if (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I8 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32) {
    l_conv_kernel_config.vector_length_in = 16;
    l_conv_kernel_config.datatype_size_in = 4;
    l_conv_kernel_config.vector_length_out = 64;
    l_conv_kernel_config.datatype_size_out = 1;
    l_conv_kernel_config.vector_length_wt = 64;
    l_conv_kernel_config.datatype_size_wt = 1;
    l_conv_kernel_config.vfma_instruction = LIBXSMM_X86_INSTR_VPMADDUBSW;
    l_conv_kernel_config.vadd_instruction = LIBXSMM_X86_INSTR_VPADDD;
    l_conv_kernel_config.vbcst_instruction = LIBXSMM_X86_INSTR_VPBROADCASTD;
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }
  l_conv_kernel_config.vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
  l_conv_kernel_config.vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
  l_conv_kernel_config.prefetch_instruction = LIBXSMM_X86_INSTR_PREFETCHT2;
  l_conv_kernel_config.alu_add_instruction = LIBXSMM_X86_INSTR_ADDQ;
  l_conv_kernel_config.alu_sub_instruction = LIBXSMM_X86_INSTR_SUBQ;
  l_conv_kernel_config.alu_cmp_instruction = LIBXSMM_X86_INSTR_CMPQ;
  l_conv_kernel_config.alu_jmp_instruction = LIBXSMM_X86_INSTR_JL;
  l_conv_kernel_config.alu_mov_instruction = LIBXSMM_X86_INSTR_MOVQ;
  l_conv_kernel_config.vector_name = 'z';

  /* calculate leading dimension depending on format */
  if ( (i_conv_desc->format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0 ) {
    l_conv_kernel_config.l_ld_ifm_act = i_conv_desc->ifm_block;
    l_conv_kernel_config.l_ld_ofm_act = i_conv_desc->ofm_block;
    l_conv_kernel_config.l_ld_ifm_fil = i_conv_desc->ifm_block;
    l_conv_kernel_config.l_ld_ofm_fil = i_conv_desc->ofm_block;
    l_found_act_format = 1;
    l_found_fil_format = 1;
  }
  if ( (i_conv_desc->format & LIBXSMM_DNN_TENSOR_FORMAT_NHWC) > 0 ) {
    l_conv_kernel_config.l_ld_ifm_act = i_conv_desc->ifm_block * i_conv_desc->blocks_ifm;
    l_conv_kernel_config.l_ld_ofm_act = i_conv_desc->ofm_block * i_conv_desc->blocks_ofm;
    l_found_act_format = 1;
    if (i_conv_desc->datatype != LIBXSMM_DNN_DATATYPE_F32) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DT_FORMAT );
      return;
    }
  }
  if ( (i_conv_desc->format & LIBXSMM_DNN_TENSOR_FORMAT_RSCK) > 0 ) {
#if 0
    l_conv_kernel_config.l_ld_ifm_fil = i_conv_desc->ifm_block * i_conv_desc->blocks_ifm;
    l_conv_kernel_config.l_ld_ofm_fil = i_conv_desc->ofm_block * i_conv_desc->blocks_ofm;
    l_found_fil_format = 1;
    if (i_conv_desc->datatype != LIBXSMM_DNN_DATATYPE_F32) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DT_FORMAT );
      return;
    }
#endif
    l_conv_kernel_config.l_ld_ifm_fil = i_conv_desc->ifm_block;
    l_conv_kernel_config.l_ld_ofm_fil = i_conv_desc->ofm_block;
    l_found_fil_format = 1;
    if (i_conv_desc->datatype != LIBXSMM_DNN_DATATYPE_F32) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DT_FORMAT );
      return;
    }
  }
  if ( (l_found_act_format == 0) || (l_found_fil_format == 0) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_CONV_FORMAT );
    return;
  }

  /* check if we have full vectors */
  if ( i_conv_desc->ifm_block % l_conv_kernel_config.vector_length_in != 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CONV_IFM_VEC );
    return;
  }

#if 0
  /* check if we have  stride of 1 */
  if ( i_conv_desc->stride_h != 1 || i_conv_desc->stride_w != 1 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CONV_CONT_STRIDE );
    return;
  }
#endif

  /* initialize KW unrolling */
  if (i_conv_desc->unroll_kw != 0) {
    l_kw_trips = i_conv_desc->kw;
  }
  /* initialize KH unrolling */
  if (i_conv_desc->unroll_kh != 0) {
    l_kh_trips = i_conv_desc->kh;
  }

  /* Setting out the number of output prefetches per kh iterations */
  num_output_prefetch = i_conv_desc->ofw_rb / i_conv_desc->kh;
  if ((i_conv_desc->ofw_rb % i_conv_desc->kh) != 0) {
    num_output_prefetch += 1;
  }

#define ENABLE_INPUT_PREFETCH
#define ENABLE_OUTPUT_PREFETCH
#define ENABLE_WEIGHT_PREFETCH

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* open asm */
  libxsmm_x86_instruction_open_stream_convolution( io_generated_code, l_gp_reg_mapping.gp_reg_input,
                                                   l_gp_reg_mapping.gp_reg_weight, l_gp_reg_mapping.gp_reg_output,
                                                   l_gp_reg_mapping.gp_reg_input_pf, l_gp_reg_mapping.gp_reg_weight_pf,
                                                   l_gp_reg_mapping.gp_reg_output_pf, i_arch );

  /* load an additional temp register with 32 16bit 1s */
  if (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I8 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_help_0, 65537 );
    libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_reg_mapping.gp_reg_help_0 );

    libxsmm_x86_instruction_vec_move( io_generated_code,
                                      l_conv_kernel_config.instruction_set,
                                      l_conv_kernel_config.vbcst_instruction,
                                      LIBXSMM_X86_GP_REG_RSP ,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                      l_conv_kernel_config.vector_name, 6, 0, 0 );

    libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_reg_mapping.gp_reg_help_0 );
  }


  if ( i_conv_desc->unroll_kh == 0 ) {
    /* open KH loop, kj */
    libxsmm_generator_convolution_header_kh_loop(  io_generated_code, &l_loop_label_tracker,
                                                  &l_conv_kernel_config, l_gp_reg_mapping.gp_reg_kh_loop );
  }
  /* unroll KH */
  for ( l_kh = 0; l_kh < l_kh_trips; l_kh++) {
    if ( i_conv_desc->unroll_kw == 0 ) {
      /* open KW loop, ki */
      libxsmm_generator_convolution_header_kw_loop(  io_generated_code, &l_loop_label_tracker,
                                                    &l_conv_kernel_config, l_gp_reg_mapping.gp_reg_kw_loop );
    }

    /* ifmInner loop, VLEN, ifm2, fully unrolled blocked by ofw_rb * ofw_rb */
    libxsmm_generator_convolution_backward_avx512_ofmloop(io_generated_code,
                                                         &l_gp_reg_mapping,
                                                         &l_conv_kernel_config,
                                                         i_conv_desc,
                                                         i_conv_desc->unroll_kw == 0 ? 1 : l_kw_trips,
                                                         num_output_prefetch );

    if (l_kw_trips == 1) {
      /* Adjust weight pointer */
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
                                        l_conv_kernel_config.alu_add_instruction,
                                        l_gp_reg_mapping.gp_reg_weight,
                                        l_conv_kernel_config.l_ld_ifm_fil * i_conv_desc->fm_lp_block * l_conv_kernel_config.l_ld_ofm_fil * l_conv_kernel_config.datatype_size_wt );
#ifdef ENABLE_WEIGHT_PREFETCH
      /* Adjust weight prefetch pointer */
      if ( (i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2 ) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
                                          l_conv_kernel_config.alu_add_instruction,
                                          l_gp_reg_mapping.gp_reg_weight_pf,
                                          l_conv_kernel_config.l_ld_ifm_fil * i_conv_desc->fm_lp_block * l_conv_kernel_config.l_ld_ofm_fil * l_conv_kernel_config.datatype_size_wt );
      }
#endif
      /* Adjust input pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                       l_conv_kernel_config.alu_add_instruction,
                                       l_gp_reg_mapping.gp_reg_input,
                                       l_conv_kernel_config.l_ld_ifm_act * l_conv_kernel_config.datatype_size_in );
    }

    if ( i_conv_desc->unroll_kw == 0 ) {
      /* close KW loop, ki */
      libxsmm_generator_convolution_footer_kw_loop(  io_generated_code, &l_loop_label_tracker,
                                                    &l_conv_kernel_config, l_gp_reg_mapping.gp_reg_kw_loop, i_conv_desc->kw );
    }

    if (l_kw_trips == 1) {
      /* Adjust input pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                       l_conv_kernel_config.alu_sub_instruction,
                                       l_gp_reg_mapping.gp_reg_input,
                                       i_conv_desc->kw * l_conv_kernel_config.l_ld_ifm_act * l_conv_kernel_config.datatype_size_in );
    } else {
      /* Adjust weight pointer */
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
                                        l_conv_kernel_config.alu_add_instruction,
                                        l_gp_reg_mapping.gp_reg_weight,
                                        i_conv_desc->kw * l_conv_kernel_config.l_ld_ifm_fil * i_conv_desc->fm_lp_block * l_conv_kernel_config.l_ld_ofm_fil * l_conv_kernel_config.datatype_size_wt );
#ifdef ENABLE_WEIGHT_PREFETCH
      /* Adjust weight prefetch pointer */
      if ( (i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2 ) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
                                          l_conv_kernel_config.alu_add_instruction,
                                          l_gp_reg_mapping.gp_reg_weight_pf,
                                          i_conv_desc->kw * l_conv_kernel_config.l_ld_ifm_fil * i_conv_desc->fm_lp_block * l_conv_kernel_config.l_ld_ofm_fil * l_conv_kernel_config.datatype_size_wt );
      }
#endif
    }

    /* Increment input pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     l_conv_kernel_config.alu_add_instruction,
                                     l_gp_reg_mapping.gp_reg_input,
                                     i_conv_desc->ifw_padded * l_conv_kernel_config.l_ld_ifm_act * l_conv_kernel_config.datatype_size_in );
#ifdef ENABLE_INPUT_PREFETCH
    /* Adjust input prefetch pointer */
    if ( (i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1 ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                       l_conv_kernel_config.alu_add_instruction,
                                       l_gp_reg_mapping.gp_reg_input_pf,
                                       i_conv_desc->ifw_padded * l_conv_kernel_config.l_ld_ifm_act * l_conv_kernel_config.datatype_size_in );
    }
#endif

#ifdef ENABLE_OUTPUT_PREFETCH
    /* Adjust output prefetch pointer */
    if ( (i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1 ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                       l_conv_kernel_config.alu_add_instruction,
                                       l_gp_reg_mapping.gp_reg_output_pf,
                                       num_output_prefetch * l_conv_kernel_config.l_ld_ofm_act * i_conv_desc->fm_lp_block * l_conv_kernel_config.datatype_size_out );
    }
#endif

  } /* end of for l_kh_trips */

  if ( i_conv_desc->unroll_kh == 0 ) {
    /* close KH loop, kj */
    libxsmm_generator_convolution_footer_kh_loop(  io_generated_code, &l_loop_label_tracker,
                                                  &l_conv_kernel_config, l_gp_reg_mapping.gp_reg_kh_loop, i_conv_desc->kh );
  }


  /* close asm */
  libxsmm_x86_instruction_close_stream_convolution( io_generated_code, i_arch );
}


LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_backward_avx512_ofmloop( libxsmm_generated_code*                           io_generated_code,
                                                           const libxsmm_convolution_backward_gp_reg_mapping* i_gp_reg_mapping,
                                                           const libxsmm_convolution_kernel_config*           i_conv_kernel_config,
                                                           const libxsmm_convolution_backward_descriptor*     i_conv_desc,
                                                           const unsigned int                                 i_kw_unroll,
                                                           const unsigned int                                 num_output_prefetch )
{
  if (i_conv_desc->ofh_rb == 2) {
    /* setup input strides */
    libxsmm_generator_convolution_backward_avx512_init_output_strides_two_rows( io_generated_code, i_gp_reg_mapping, i_conv_kernel_config, i_conv_desc );

    /* select architecture */
    if ( i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_KNM ) {
      libxsmm_generator_convolution_backward_avx512_ofmloop_qfma_two_rows( io_generated_code, i_gp_reg_mapping,
                                                                        i_conv_kernel_config, i_conv_desc, i_kw_unroll, num_output_prefetch );
    } else if ( i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_MIC  ||
                i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_CORE ) {
      libxsmm_generator_convolution_backward_avx512_ofmloop_sfma_two_rows( io_generated_code, i_gp_reg_mapping,
                                                                          i_conv_kernel_config, i_conv_desc, i_kw_unroll, num_output_prefetch );
    }
  } else {
    /* setup output strides */
    libxsmm_generator_convolution_backward_avx512_init_output_strides( io_generated_code, i_gp_reg_mapping, i_conv_kernel_config, i_conv_desc );
    /* select architecture */
    if ( i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_KNM ) {
      libxsmm_generator_convolution_backward_avx512_ofmloop_qfma( io_generated_code, i_gp_reg_mapping,
                                                                        i_conv_kernel_config, i_conv_desc, i_kw_unroll, num_output_prefetch );
    } else if ( i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_MIC  ||
                i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_CORE ) {
      libxsmm_generator_convolution_backward_avx512_ofmloop_sfma( io_generated_code, i_gp_reg_mapping,
                                                                          i_conv_kernel_config, i_conv_desc, i_kw_unroll, num_output_prefetch );
    }
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_backward_avx512_init_output_strides( libxsmm_generated_code*                           io_generated_code,
                                                                      const libxsmm_convolution_backward_gp_reg_mapping* i_gp_reg_mapping,
                                                                      const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                                      const libxsmm_convolution_backward_descriptor*     i_conv_desc ) {
  /* Initialize helper registers for SIB addressing */
  /* helper 0: Index register holding ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_help_0, i_conv_kernel_config->datatype_size_out * i_conv_kernel_config->l_ld_ofm_act * i_conv_desc->fm_lp_block);
  /* helper 1: Index register holding 3*ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_help_1, i_conv_kernel_config->datatype_size_out * i_conv_kernel_config->l_ld_ofm_act * i_conv_desc->fm_lp_block * 3 );
  /* helper 2: Index register holding 5*ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_help_2, i_conv_kernel_config->datatype_size_out * i_conv_kernel_config->l_ld_ofm_act * i_conv_desc->fm_lp_block * 5 );
  /* helper 3: Index register holding 7*ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_help_3, i_conv_kernel_config->datatype_size_out * i_conv_kernel_config->l_ld_ofm_act * i_conv_desc->fm_lp_block * 7 );

  /* helper 4: B + 9*ldb, additional base address
     helper 5: B + 18*ldb, additional base address */
  if ( i_conv_desc->ofw_rb > 9 ) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_output, i_gp_reg_mapping->gp_reg_help_4);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_help_4,  9 * i_conv_kernel_config->datatype_size_out * i_conv_kernel_config->l_ld_ofm_act * i_conv_desc->fm_lp_block);
  }
  if ( i_conv_desc->ofw_rb > 18 ) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_output, i_gp_reg_mapping->gp_reg_help_5);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_help_5, 18 *  i_conv_kernel_config->datatype_size_out * i_conv_kernel_config->l_ld_ofm_act * i_conv_desc->fm_lp_block);
  }
  if ( i_conv_desc->ofw_rb > 27 ) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_output, i_gp_reg_mapping->gp_reg_help_6);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_help_6, 27 *  i_conv_kernel_config->datatype_size_out * i_conv_kernel_config->l_ld_ofm_act * i_conv_desc->fm_lp_block);
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_backward_avx512_init_output_strides_two_rows( libxsmm_generated_code*                           io_generated_code,
                                                                      const libxsmm_convolution_backward_gp_reg_mapping* i_gp_reg_mapping,
                                                                      const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                                      const libxsmm_convolution_backward_descriptor*     i_conv_desc ) {
  /* Initialize helper registers for SIB addressing */
  /* helper 0: Index register holding ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_help_0, i_conv_kernel_config->datatype_size_out * i_conv_kernel_config->l_ld_ofm_act * i_conv_desc->fm_lp_block);
  /* helper 1: Index register holding 3*ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_help_1, i_conv_kernel_config->datatype_size_out * i_conv_kernel_config->l_ld_ofm_act * i_conv_desc->fm_lp_block * 3 );
  /* helper 2: Index register holding 5*ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_help_2, i_conv_kernel_config->datatype_size_out * i_conv_kernel_config->l_ld_ofm_act * i_conv_desc->fm_lp_block * 5 );
  /* helper 3: Index register holding 7*ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_help_3, i_conv_kernel_config->datatype_size_out * i_conv_kernel_config->l_ld_ofm_act * i_conv_desc->fm_lp_block * 7 );

  /* helper 4: B+9*ldb,            additional base address
     helper 5: B+ofw_padded,              additional base address
     helper 6: B+ofw_padded+9*ldb,        additional base address */
  libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_output, i_gp_reg_mapping->gp_reg_help_5);
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_help_5, i_conv_desc->ofw_padded * i_conv_kernel_config->l_ld_ofm_act * i_conv_desc->fm_lp_block *  i_conv_kernel_config->datatype_size_out);
  if ( i_conv_desc->ofw_rb > 9 ) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_output, i_gp_reg_mapping->gp_reg_help_4);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_help_4, 9 * i_conv_kernel_config->datatype_size_out * i_conv_kernel_config->l_ld_ofm_act * i_conv_desc->fm_lp_block);

    libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_help_5, i_gp_reg_mapping->gp_reg_help_6);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_help_6, 9 * i_conv_kernel_config->datatype_size_out * i_conv_kernel_config->l_ld_ofm_act * i_conv_desc->fm_lp_block);
  }
}


LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_backward_avx512_calc_sib_output_strides( const libxsmm_convolution_backward_gp_reg_mapping* i_gp_reg_mapping,
                                                                            const int                                         i_stride_num,
                                                                            unsigned int*                                     o_output_reg,
                                                                            unsigned int*                                     o_output_idx,
                                                                            unsigned int*                                     o_scale ) {
 /* init output variables */
  *o_output_reg = i_gp_reg_mapping->gp_reg_output;
  *o_output_idx = LIBXSMM_X86_GP_REG_UNDEF;
  *o_scale = 0;

  /* select the base register */
  if ( i_stride_num > 26 ) {
    *o_output_reg = i_gp_reg_mapping->gp_reg_help_6;
  } else if ( i_stride_num > 17 ) {
    *o_output_reg = i_gp_reg_mapping->gp_reg_help_5;
  } else if ( i_stride_num > 8 ) {
    *o_output_reg = i_gp_reg_mapping->gp_reg_help_4;
  } else {
    *o_output_reg = i_gp_reg_mapping->gp_reg_output;
  }

  /* select scale and index */
  if ( i_stride_num % 9 == 0 ) {
    *o_output_idx = LIBXSMM_X86_GP_REG_UNDEF;
    *o_scale = 0;
  } else if ( i_stride_num % 9 == 1 ) {
    *o_output_idx = i_gp_reg_mapping->gp_reg_help_0;
    *o_scale = 1;
  } else if ( i_stride_num % 9 == 2 ) {
    *o_output_idx = i_gp_reg_mapping->gp_reg_help_0;
    *o_scale = 2;
  } else if ( i_stride_num % 9 == 3 ) {
    *o_output_idx = i_gp_reg_mapping->gp_reg_help_1;
    *o_scale = 1;
  } else if ( i_stride_num % 9 == 4 ) {
    *o_output_idx = i_gp_reg_mapping->gp_reg_help_0;
    *o_scale = 4;
  } else if ( i_stride_num % 9 == 5 ) {
    *o_output_idx = i_gp_reg_mapping->gp_reg_help_2;
    *o_scale = 1;
  } else if ( i_stride_num % 9 == 6 ) {
    *o_output_idx = i_gp_reg_mapping->gp_reg_help_1;
    *o_scale = 2;
  } else if ( i_stride_num % 9 == 7 ) {
    *o_output_idx = i_gp_reg_mapping->gp_reg_help_3;
    *o_scale = 1;
  } else if ( i_stride_num % 9 == 8 ) {
    *o_output_idx = i_gp_reg_mapping->gp_reg_help_0;
    *o_scale = 8;
  } else {
    assert(0/*should not happen*/);
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_backward_avx512_calc_sib_output_strides_two_rows( const libxsmm_convolution_backward_gp_reg_mapping* i_gp_reg_mapping,
                                                                            const int                                         i_row_num,
                                                                            const int                                         i_stride_num,
                                                                            unsigned int*                                     o_output_reg,
                                                                            unsigned int*                                     o_output_idx,
                                                                            unsigned int*                                     o_scale ) {
 /* init output variables */
  *o_output_reg = i_gp_reg_mapping->gp_reg_output;
  *o_output_idx = LIBXSMM_X86_GP_REG_UNDEF;
  *o_scale = 0;

  /* select the base register */
  if ( i_row_num == 1 ) {
    if ( i_stride_num > 8 ) {
      *o_output_reg = i_gp_reg_mapping->gp_reg_help_6;
    } else {
      *o_output_reg = i_gp_reg_mapping->gp_reg_help_5;
    }
  } else {
    if ( i_stride_num > 8 ) {
      *o_output_reg = i_gp_reg_mapping->gp_reg_help_4;
    } else {
      *o_output_reg = i_gp_reg_mapping->gp_reg_output;
    }
  }

  /* select scale and index */
  if ( i_stride_num % 9 == 0 ) {
    *o_output_idx = LIBXSMM_X86_GP_REG_UNDEF;
    *o_scale = 0;
  } else if ( i_stride_num % 9 == 1 ) {
    *o_output_idx = i_gp_reg_mapping->gp_reg_help_0;
    *o_scale = 1;
  } else if ( i_stride_num % 9 == 2 ) {
    *o_output_idx = i_gp_reg_mapping->gp_reg_help_0;
    *o_scale = 2;
  } else if ( i_stride_num % 9 == 3 ) {
    *o_output_idx = i_gp_reg_mapping->gp_reg_help_1;
    *o_scale = 1;
  } else if ( i_stride_num % 9 == 4 ) {
    *o_output_idx = i_gp_reg_mapping->gp_reg_help_0;
    *o_scale = 4;
  } else if ( i_stride_num % 9 == 5 ) {
    *o_output_idx = i_gp_reg_mapping->gp_reg_help_2;
    *o_scale = 1;
  } else if ( i_stride_num % 9 == 6 ) {
    *o_output_idx = i_gp_reg_mapping->gp_reg_help_1;
    *o_scale = 2;
  } else if ( i_stride_num % 9 == 7 ) {
    *o_output_idx = i_gp_reg_mapping->gp_reg_help_3;
    *o_scale = 1;
  } else if ( i_stride_num % 9 == 8 ) {
    *o_output_idx = i_gp_reg_mapping->gp_reg_help_0;
    *o_scale = 8;
  } else {
    assert(0/*should not happen*/);
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_backward_avx512_load_input( libxsmm_generated_code*                            io_generated_code,
                                                               const libxsmm_convolution_backward_gp_reg_mapping* i_gp_reg_mapping,
                                                               const libxsmm_convolution_kernel_config*           i_conv_kernel_config,
                                                               const libxsmm_convolution_backward_descriptor*     i_conv_desc,
                                                               const unsigned int                                 l_vec_reg_acc_start,
                                                               const unsigned int                                 l_accs,
                                                               const unsigned int                                 l_k_2 ) {
  unsigned int j,k;

  if (i_conv_desc->stride_w == 1) {
    /* Loading (ofw_rb + kw) VLEN inputs in a pipelined fashion to exploit register reuse */
    if (l_k_2 == 0) {
      /* load ofw_rb VLEN inputs in the beginning*/
      if ((i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_KNM && i_conv_desc->ofw_rb <= 14 && i_conv_desc->ofh_rb == 1) || (i_conv_desc->ofw_rb < 12 && i_conv_desc->ofh_rb == 1 )) {
        /*Use extra accumlators to hide FMA latencies when ofw_rb is too small */
        for ( j = 0; j < i_conv_desc->ofw_rb; j++ ) {
          for (k = l_accs; k > 1; k--) {
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                     i_conv_kernel_config->instruction_set,
                                                     i_conv_kernel_config->vxor_instruction,
                                                     i_conv_kernel_config->vector_name,
                                                     i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*k) + j,
                                                     i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*k) + j,
                                                     i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*k) + j );
          }
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                  i_conv_kernel_config->instruction_set,
                                  i_conv_kernel_config->vmove_instruction,
                                  i_gp_reg_mapping->gp_reg_input,
                                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                                  j * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in,
                                  i_conv_kernel_config->vector_name,
                                  l_vec_reg_acc_start + j , 0, 0 );
#ifdef ENABLE_INPUT_PREFETCH
          if (((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1)) {
              libxsmm_x86_instruction_prefetch( io_generated_code,
                                             LIBXSMM_X86_INSTR_PREFETCHT0,
                                             i_gp_reg_mapping->gp_reg_input_pf,
                                             LIBXSMM_X86_GP_REG_UNDEF,
                                             0,
                                             (j * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in));
          }
#endif
        } /* end of for loop over ofw_rb */
      } else { /* ofw_rb is big enough to hide FMA latencies */
        for ( j = 0; j < i_conv_desc->ofw_rb; j++ ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_conv_kernel_config->instruction_set,
                                    i_conv_kernel_config->vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_input,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    j * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in,
                                    i_conv_kernel_config->vector_name,
                                    l_vec_reg_acc_start + j , 0, 0 );
#ifdef ENABLE_INPUT_PREFETCH
            if (((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1)) {
                libxsmm_x86_instruction_prefetch( io_generated_code,
                                               LIBXSMM_X86_INSTR_PREFETCHT0,
                                               i_gp_reg_mapping->gp_reg_input_pf,
                                               LIBXSMM_X86_GP_REG_UNDEF,
                                               0,
                                               (j * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in));
            }
#endif
        } /* end of for ofw_rb loop */
      } /* end of if extra accumulator is used */
    } else { /* If not first iteration */
    /* load one VLEN input */
      if ((i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_KNM && i_conv_desc->ofw_rb <= 14 && i_conv_desc->ofh_rb == 1) || (i_conv_desc->ofw_rb < 12 && i_conv_desc->ofh_rb == 1)) {
        /*Use extra accumulators to hide FMA latencies when ofw_rb is too small */
        for (k = 0; k < (l_accs-1); k--) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                   i_conv_kernel_config->instruction_set,
                                                   i_conv_kernel_config->vxor_instruction,
                                                   i_conv_kernel_config->vector_name,
                                                   i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*l_accs) + ((l_k_2 - 1)%i_conv_desc->ofw_rb),
                                                   i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*l_accs) + ((l_k_2 - 1)%i_conv_desc->ofw_rb),
                                                   i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*l_accs) + ((l_k_2 - 1)%i_conv_desc->ofw_rb) );
#if 0
/* AH this was the old code with inverse order on k, but it falls apart when ofw_rb = 1 */
                                                   i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*k) + ((l_k_2-1)%(i_conv_desc->ofw_rb * k)),
                                                   i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*k) + ((l_k_2-1)%(i_conv_desc->ofw_rb * k)),
                                                   i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*k) + ((l_k_2-1)%(i_conv_desc->ofw_rb * k)) );
#endif
        }
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_conv_kernel_config->instruction_set,
                                      i_conv_kernel_config->vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_input,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      (i_conv_desc->ofw_rb - 1 + l_k_2)*i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in,
                                      i_conv_kernel_config->vector_name,
                                      l_vec_reg_acc_start + ((l_k_2 - 1)%i_conv_desc->ofw_rb),
                                      0, 0 );
#ifdef ENABLE_INPUT_PREFETCH
        if (((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1)) {
            libxsmm_x86_instruction_prefetch( io_generated_code,
                                           LIBXSMM_X86_INSTR_PREFETCHT0,
                                           i_gp_reg_mapping->gp_reg_input_pf,
                                           LIBXSMM_X86_GP_REG_UNDEF,
                                           0,
                                           (i_conv_desc->ofw_rb - 1 + l_k_2)*i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in);
        }
#endif
      } else { /* ofw_rb is big enough to hide FMA latencies */
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_conv_kernel_config->instruction_set,
                                      i_conv_kernel_config->vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_input,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      (i_conv_desc->ofw_rb - 1 + l_k_2)*i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in,
                                      i_conv_kernel_config->vector_name,
                                      l_vec_reg_acc_start + ((l_k_2 - 1)%i_conv_desc->ofw_rb),
                                        0, 0 );
#ifdef ENABLE_INPUT_PREFETCH
        if (((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1)) {
            libxsmm_x86_instruction_prefetch( io_generated_code,
                                           LIBXSMM_X86_INSTR_PREFETCHT0,
                                           i_gp_reg_mapping->gp_reg_input_pf,
                                           LIBXSMM_X86_GP_REG_UNDEF,
                                           0,
                                           (i_conv_desc->ofw_rb - 1 + l_k_2)*i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in);
        }
#endif
      } /* end of if extra accumulator is used*/
    } /* end of if l_k_2 == 0 */
  } else { /* If stride_w != 1 */
    /* load ofw_rb VLEN inputs in each iteration*/
    if ( ((i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_KNM && i_conv_desc->ofw_rb <= 14 && i_conv_desc->ofh_rb == 1) || (i_conv_desc->ofw_rb < 12 && i_conv_desc->ofh_rb == 1)) ) {
      /*Use extra accumlators to hide FMA latencies when ofw_rb is too small */
      for ( j = 0; j < i_conv_desc->ofw_rb; j++ ) {
        for (k = l_accs; k > 1; k--) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                   i_conv_kernel_config->instruction_set,
                                                   i_conv_kernel_config->vxor_instruction,
                                                   i_conv_kernel_config->vector_name,
                                                   i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*k) + j,
                                                   i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*k) + j,
                                                   i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*k) + j );
        }
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                i_conv_kernel_config->instruction_set,
                                i_conv_kernel_config->vmove_instruction,
                                i_gp_reg_mapping->gp_reg_input,
                                LIBXSMM_X86_GP_REG_UNDEF, 0,
                                (j * i_conv_desc->stride_w + l_k_2) * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in,
                                i_conv_kernel_config->vector_name,
                                l_vec_reg_acc_start + j , 0, 0 );
#ifdef ENABLE_INPUT_PREFETCH
        if (((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1)) {
            libxsmm_x86_instruction_prefetch( io_generated_code,
                                           LIBXSMM_X86_INSTR_PREFETCHT0,
                                           i_gp_reg_mapping->gp_reg_input_pf,
                                           LIBXSMM_X86_GP_REG_UNDEF,
                                           0,
                                           ((j * i_conv_desc->stride_w + l_k_2) * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in));
        }
#endif
      } /* end of for loop over ofw_rb */
    } else { /* ofw_rb is big enough to hide FMA latencies */
      for ( j = 0; j < i_conv_desc->ofw_rb; j++ ) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                  i_conv_kernel_config->instruction_set,
                                  i_conv_kernel_config->vmove_instruction,
                                  i_gp_reg_mapping->gp_reg_input,
                                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                                  (j * i_conv_desc->stride_w + l_k_2) * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in,
                                  i_conv_kernel_config->vector_name,
                                  l_vec_reg_acc_start + j , 0, 0 );
#ifdef ENABLE_INPUT_PREFETCH
          if (((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1)) {
              libxsmm_x86_instruction_prefetch( io_generated_code,
                                             LIBXSMM_X86_INSTR_PREFETCHT0,
                                             i_gp_reg_mapping->gp_reg_input_pf,
                                             LIBXSMM_X86_GP_REG_UNDEF,
                                             0,
                                             ((j * i_conv_desc->stride_w + l_k_2)* i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in));
          }
#endif
      } /* end of for ofw_rb loop */
    } /* end of if extra accumulator is used */
  } /* end of if stride_w == 1 */
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_backward_avx512_load_input_two_rows( libxsmm_generated_code*                            io_generated_code,
                                                                        const libxsmm_convolution_backward_gp_reg_mapping* i_gp_reg_mapping,
                                                                        const libxsmm_convolution_kernel_config*           i_conv_kernel_config,
                                                                        const libxsmm_convolution_backward_descriptor*     i_conv_desc,
                                                                        const unsigned int                                 l_vec_reg_acc_start,
                                                                        const unsigned int                                 l_k_2 ) {
  unsigned int i, j;
  /* Assuming (ofh_rb * ofw_rb) is good enough to  hide FMA latencies. So, no extra accumulator is used here */

  if (i_conv_desc->stride_w == 1) {
    /* Loading (ofh_rb * (ofw_rb + kw)) VLEN inputs in a pipelined fashion to exploit register reuse */
    if(l_k_2 == 0) {
      /* load (ofh_rb * ofw_rb) VLEN inputs in the beginning*/
      for (i = 0; i < i_conv_desc->ofh_rb; i++) {
        for ( j = 0; j < i_conv_desc->ofw_rb; j++ ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_conv_kernel_config->instruction_set,
                                    i_conv_kernel_config->vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_input,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    (j + i * i_conv_desc->ifw_padded) * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in,
                                    i_conv_kernel_config->vector_name,
                                    l_vec_reg_acc_start + (i * i_conv_desc->ofw_rb) + j , 0, 0 );
#ifdef ENABLE_INPUT_PREFETCH
            if (((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1)) {
                libxsmm_x86_instruction_prefetch( io_generated_code,
                                               LIBXSMM_X86_INSTR_PREFETCHT0,
                                               i_gp_reg_mapping->gp_reg_input_pf,
                                               LIBXSMM_X86_GP_REG_UNDEF,
                                               0,
                                               ((j + i * i_conv_desc->ifw_padded) * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in));
            }
#endif
        } /* end of for ofw_rb loop */
      } /* end of for ofh_rb loop */
    } else { /* If not first iteraion */
    /* load  ofh_rb  VLEN inputs */
      for (i = 0; i < i_conv_desc->ofh_rb; ++i) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_conv_kernel_config->instruction_set,
                                      i_conv_kernel_config->vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_input,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      (i * i_conv_desc->ifw_padded + (i_conv_desc->ofw_rb - 1 + l_k_2)) * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in,
                                      i_conv_kernel_config->vector_name,
                                      l_vec_reg_acc_start + (i * i_conv_desc->ofw_rb) + ((l_k_2 - 1)%i_conv_desc->ofw_rb),
                                        0, 0 );
#ifdef ENABLE_INPUT_PREFETCH
        if (((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1)) {
            libxsmm_x86_instruction_prefetch( io_generated_code,
                                           LIBXSMM_X86_INSTR_PREFETCHT0,
                                           i_gp_reg_mapping->gp_reg_input_pf,
                                           LIBXSMM_X86_GP_REG_UNDEF,
                                           0,
                                           (i * i_conv_desc->ifw_padded + (i_conv_desc->ofw_rb - 1 + l_k_2)) * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in);
        }
#endif
      } /* end of for ofh_rb loop */
    } /* end of if l_k_2 == 0 */
  } else { /* If stride_w != 1 */
    /* load (ofh_rb * ofw_rb) VLEN inputs in each iteration */
    for (i = 0; i < i_conv_desc->ofh_rb; ++i) {
      for ( j = 0; j < i_conv_desc->ofw_rb; j++ ) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                  i_conv_kernel_config->instruction_set,
                                  i_conv_kernel_config->vmove_instruction,
                                  i_gp_reg_mapping->gp_reg_input,
                                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                                  ((i * i_conv_desc->stride_h) * i_conv_desc->ifw_padded + j * i_conv_desc->stride_w + l_k_2) * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in,
                                  i_conv_kernel_config->vector_name,
                                  l_vec_reg_acc_start + (i * i_conv_desc->ofw_rb) + j , 0, 0 );
#ifdef ENABLE_INPUT_PREFETCH
          if (((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1)) {
              libxsmm_x86_instruction_prefetch( io_generated_code,
                                             LIBXSMM_X86_INSTR_PREFETCHT0,
                                             i_gp_reg_mapping->gp_reg_input_pf,
                                             LIBXSMM_X86_GP_REG_UNDEF,
                                             0,
                                             (((i * i_conv_desc->stride_h) * i_conv_desc->ifw_padded + j * i_conv_desc->stride_w + l_k_2)* i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in));
          }
#endif
      } /* end of for ofw_rb loop */
    } /* end of for ofh_rb loop */
  } /* end of if stride_w == 1 */
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_backward_avx512_store_input( libxsmm_generated_code*                            io_generated_code,
                                                                const libxsmm_convolution_backward_gp_reg_mapping* i_gp_reg_mapping,
                                                                const libxsmm_convolution_kernel_config*           i_conv_kernel_config,
                                                                const libxsmm_convolution_backward_descriptor*     i_conv_desc,
                                                                const unsigned int                                 l_vec_reg_acc_start,
                                                                const unsigned int                                 i_kw_unroll,
                                                                const unsigned int                                 l_accs,
                                                                const unsigned int                                 l_k_2 ) {
  unsigned int j,k;

  if (i_conv_desc->stride_w == 1) {
    /* pipelined store of (ofw_rb + kw) VLEN inputs */
    if (l_k_2 == i_kw_unroll-1) {
      /* Store ofw_rb VLEN inputs in the end*/
      if ((i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_KNM && i_conv_desc->ofw_rb <= 14 && i_conv_desc->ofh_rb == 1) || (i_conv_desc->ofw_rb < 12 && i_conv_desc->ofh_rb == 1)) {
        /* adding up accumulators, adding different order to avoid stalls to some extent.... */
        for ( k = l_accs; k > 1; k-- ) {
          for ( j = 0; j < i_conv_desc->ofw_rb; j++ ) {
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                     i_conv_kernel_config->instruction_set,
                                                     i_conv_kernel_config->vadd_instruction,
                                                     i_conv_kernel_config->vector_name,
                                                     i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*k) + ((j+l_k_2) % i_conv_desc->ofw_rb),
                                                     i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb) + ((j+l_k_2) % i_conv_desc->ofw_rb),
                                                     i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb) + ((j+l_k_2) % i_conv_desc->ofw_rb) );
          }
        }
        for ( j = 0; j < i_conv_desc->ofw_rb; j++ ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_conv_kernel_config->instruction_set,
                                        i_conv_kernel_config->vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_input,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        (j+l_k_2)*i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in,
                                        i_conv_kernel_config->vector_name,
                                        l_vec_reg_acc_start + ((j+l_k_2) % i_conv_desc->ofw_rb) , 0, 1 );

        }
      } else { /* If no extra accumulator is used */
        for (j = 0; j < i_conv_desc->ofw_rb; j++ ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_conv_kernel_config->instruction_set,
                                        i_conv_kernel_config->vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_input,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        (j+l_k_2)*i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in,
                                        i_conv_kernel_config->vector_name,
                                        l_vec_reg_acc_start + ((j+l_k_2) % i_conv_desc->ofw_rb) , 0, 1 );
        }
      } /* end of if extra accumulator is used */
    } else { /* If not last iteration */
      /* Store one VLEN input */
      if ((i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_KNM && i_conv_desc->ofw_rb <= 14 && i_conv_desc->ofh_rb == 1) || (i_conv_desc->ofw_rb < 12 && i_conv_desc->ofh_rb == 1)) {
        /* adding up accumulators, adding different order to avoid stalls to some extent.... */
        for ( k = l_accs; k > 1; k-- ) {
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                     i_conv_kernel_config->instruction_set,
                                                     i_conv_kernel_config->vadd_instruction,
                                                     i_conv_kernel_config->vector_name,
                                                     i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*k) + (l_k_2 % i_conv_desc->ofw_rb),
                                                     i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb) + (l_k_2 % i_conv_desc->ofw_rb),
                                                     i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb) + (l_k_2 % i_conv_desc->ofw_rb) );
        }
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_conv_kernel_config->instruction_set,
                                        i_conv_kernel_config->vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_input,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        l_k_2 * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in,
                                        i_conv_kernel_config->vector_name,
                                        l_vec_reg_acc_start + (l_k_2 % i_conv_desc->ofw_rb) , 0, 1 );

      } else { /* If no extra accumulator is used */
        assert(0 != i_conv_desc->ofw_rb);
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_conv_kernel_config->instruction_set,
                                        i_conv_kernel_config->vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_input,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        l_k_2 * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in,
                                        i_conv_kernel_config->vector_name,
                                        l_vec_reg_acc_start + (l_k_2 % i_conv_desc->ofw_rb) , 0, 1 );
      } /* end of if extra accumulator is used */
    } /* end of pipelined store of inputs */
  } else { /* If stride_w != 1 */
    /* store ofw_rb VLEN inputs in each iteration*/
    if ( ((i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_KNM && i_conv_desc->ofw_rb <= 14 && i_conv_desc->ofh_rb == 1) || (i_conv_desc->ofw_rb < 12 && i_conv_desc->ofh_rb == 1)) ) {
      /* adding up accumulators, adding different order to avoid stalls to some extent.... */
      for ( k = l_accs; k > 1; k-- ) {
        for ( j = 0; j < i_conv_desc->ofw_rb; j++ ) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                   i_conv_kernel_config->instruction_set,
                                                   i_conv_kernel_config->vadd_instruction,
                                                   i_conv_kernel_config->vector_name,
                                                   i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*k) + ((j+l_k_2) % i_conv_desc->ofw_rb),
                                                   i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb) + ((j+l_k_2) % i_conv_desc->ofw_rb),
                                                   i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb) + ((j+l_k_2) % i_conv_desc->ofw_rb) );
        }
      }
      for ( j = 0; j < i_conv_desc->ofw_rb; j++ ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                i_conv_kernel_config->instruction_set,
                                i_conv_kernel_config->vmove_instruction,
                                i_gp_reg_mapping->gp_reg_input,
                                LIBXSMM_X86_GP_REG_UNDEF, 0,
                                (j * i_conv_desc->stride_w + l_k_2) * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in,
                                i_conv_kernel_config->vector_name,
                                l_vec_reg_acc_start + j , 0, 1 );
      } /* end of for loop over ofw_rb */
    } else { /* If no extra accumulator is used */
      for ( j = 0; j < i_conv_desc->ofw_rb; j++ ) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                  i_conv_kernel_config->instruction_set,
                                  i_conv_kernel_config->vmove_instruction,
                                  i_gp_reg_mapping->gp_reg_input,
                                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                                  (j * i_conv_desc->stride_w + l_k_2) * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in,
                                  i_conv_kernel_config->vector_name,
                                  l_vec_reg_acc_start + j , 0, 1 );
      } /* end of for ofw_rb loop */
    } /* end of if extra accumulator is used */
  } /* end of if stride == 1 */
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_backward_avx512_store_input_two_rows( libxsmm_generated_code*                            io_generated_code,
                                                                         const libxsmm_convolution_backward_gp_reg_mapping* i_gp_reg_mapping,
                                                                         const libxsmm_convolution_kernel_config*           i_conv_kernel_config,
                                                                         const libxsmm_convolution_backward_descriptor*     i_conv_desc,
                                                                         const unsigned int                                 l_vec_reg_acc_start,
                                                                         const unsigned int                                 i_kw_unroll,
                                                                         const unsigned int                                 l_k_2 ) {
  unsigned int i, j;
  /* Assuming (ofh_rb * ofw_rb) is good enough to  hide FMA latencies. So, no extra accumulator is used here */

  if (i_conv_desc->stride_w == 1) {
    /* pipelined store of (ofh_rb * (ofw_rb + kw)) VLEN inputs */
    if (l_k_2 == i_kw_unroll-1) {
      /* Store (ofh_rb * ofw_rb) VLEN inputs in the end*/
      for (i = 0; i < i_conv_desc->ofh_rb; i++) {
        for (j = 0; j < i_conv_desc->ofw_rb; j++ ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_conv_kernel_config->instruction_set,
                                        i_conv_kernel_config->vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_input,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        ((i * i_conv_desc->ifw_padded) + j + l_k_2)*i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in,
                                        i_conv_kernel_config->vector_name,
                                        l_vec_reg_acc_start + (i * i_conv_desc->ofw_rb) + ((j+l_k_2) % i_conv_desc->ofw_rb) , 0, 1 );
        } /* end of for ofw_rb loop */
      } /* end of for ofh_rb loop */
    } else { /* If not last iteration */
      /* Store ofh_rb VLEN input */
      for (i = 0; i < i_conv_desc->ofh_rb; i++) {
        assert(0 != i_conv_desc->ofw_rb);
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_conv_kernel_config->instruction_set,
                                        i_conv_kernel_config->vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_input,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        (i * i_conv_desc->ifw_padded + l_k_2) * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in,
                                        i_conv_kernel_config->vector_name,
                                        l_vec_reg_acc_start + (i * i_conv_desc->ofw_rb) + (l_k_2 % i_conv_desc->ofw_rb) , 0, 1 );
      } /* end of for ofh_rb loop */
    } /* end of pipelined store of inputs */
  } else { /* If stride_w != 1 */
    /* store (ofh_rb * ofw_rb) VLEN inputs in each iteration*/
    for (i = 0; i < i_conv_desc->ofh_rb; i++) {
      for ( j = 0; j < i_conv_desc->ofw_rb; j++ ) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                  i_conv_kernel_config->instruction_set,
                                  i_conv_kernel_config->vmove_instruction,
                                  i_gp_reg_mapping->gp_reg_input,
                                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                                  ((i * i_conv_desc->stride_h) * i_conv_desc->ifw_padded + j * i_conv_desc->stride_w + l_k_2) * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in,
                                  i_conv_kernel_config->vector_name,
                                  l_vec_reg_acc_start + (i * i_conv_desc->ofw_rb) + j , 0, 1 );
      } /* end of for ofw_rb loop */
    } /* end of for ofh_rb loop */
  } /* end of if stride == 1 */
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_backward_avx512_ofmloop_sfma( libxsmm_generated_code*                            io_generated_code,
                                                                 const libxsmm_convolution_backward_gp_reg_mapping* i_gp_reg_mapping,
                                                                 const libxsmm_convolution_kernel_config*           i_conv_kernel_config,
                                                                 const libxsmm_convolution_backward_descriptor*     i_conv_desc,
                                                                 const unsigned int                                 i_kw_unroll,
                                                                 const unsigned int                                 num_output_prefetch ) {
  unsigned int l_n;
  unsigned int l_k_2, l_k_3;
  unsigned int l_output_reg;
  unsigned int l_output_idx;
  unsigned int l_scale;
  unsigned int l_disp;
  unsigned int l_displacement_k = 0;
  unsigned int l_w;
  unsigned int input_reg;
  unsigned int l_k = 0;
  unsigned int weight_counter = 0;
  unsigned int current_acc_start;

  /****************************************/
  /***Assuming ifmblock is same as VLEN ***/
  /****************************************/

  const unsigned int l_accs = 2; /* Number of extra accumulator used when required to hide FMA latencies */
  /* start register of accumulator */
  const unsigned int l_vec_reg_acc_start = i_conv_kernel_config->vector_reg_count - i_conv_desc->ofw_rb;

  for ( l_k_2 = 0; l_k_2 < i_kw_unroll; l_k_2++) {
    /* load inputs */
    libxsmm_generator_convolution_backward_avx512_load_input(io_generated_code, i_gp_reg_mapping, i_conv_kernel_config, i_conv_desc, l_vec_reg_acc_start, l_accs, l_k_2);

    l_displacement_k = 0;
    for ( l_k_3 = 0; l_k_3 < i_conv_desc->ofm_block; l_k_3++, l_k++) {
      if ( l_k == 0) {
        /* load weights */
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_conv_kernel_config->instruction_set,
                                      i_conv_kernel_config->vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_weight,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      ((l_k_2 * i_conv_kernel_config->l_ld_ofm_fil + l_k_3) * i_conv_kernel_config->l_ld_ifm_fil * i_conv_desc->fm_lp_block)*(i_conv_kernel_config->datatype_size_wt),
                                      i_conv_kernel_config->vector_name, 0,
                                      0, 0 );
        if ( i_conv_desc->ofm_block *i_kw_unroll > 1 ) {
           for ( l_w = 1; l_w < 4; l_w++ ) {
            /* second weight loaded in first iteration, in case of large blockings -> hiding L1 latencies */
              libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_conv_kernel_config->instruction_set,
                                            i_conv_kernel_config->vmove_instruction,
                                            i_gp_reg_mapping->gp_reg_weight,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            ((l_k_2 * i_conv_kernel_config->l_ld_ofm_fil + l_k_3 + l_w) * i_conv_kernel_config->l_ld_ifm_fil * i_conv_desc->fm_lp_block) *(i_conv_kernel_config->datatype_size_wt),
                                            i_conv_kernel_config->vector_name, l_w,
                                            0, 0 );
          }
          weight_counter += 3;
        }
      } else if ((l_k < ((i_conv_desc->ofm_block * i_kw_unroll) - 3)) && (l_k_3 >= (i_conv_desc->ofm_block -3)) && (l_k_3 < i_conv_desc->ofm_block) ) {
        /* At the end of l_k_3, weight for next l_k_2 is loaded */
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_conv_kernel_config->instruction_set,
                                          i_conv_kernel_config->vmove_instruction,
                                          i_gp_reg_mapping->gp_reg_weight,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          ((weight_counter)* (i_conv_kernel_config->l_ld_ifm_fil * i_conv_desc->fm_lp_block)*(i_conv_kernel_config->datatype_size_wt) )
                                          + ((l_k_2 + 1) * (i_conv_kernel_config->l_ld_ofm_fil) * (i_conv_kernel_config->l_ld_ifm_fil) * (i_conv_desc->fm_lp_block) * (i_conv_kernel_config->datatype_size_wt)),
                                          i_conv_kernel_config->vector_name, (l_k+3)%4,
                                          0, 0 );
      } else if ( l_k < ((i_conv_desc->ofm_block * i_kw_unroll) - 3) ) {
        /* pipelined load of weight, one k iteration ahead */
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_conv_kernel_config->instruction_set,
                                          i_conv_kernel_config->vmove_instruction,
                                          i_gp_reg_mapping->gp_reg_weight,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          ((weight_counter) * (i_conv_kernel_config->l_ld_ifm_fil * i_conv_desc->fm_lp_block)*(i_conv_kernel_config->datatype_size_wt))
                                          + ((l_k_2) * (i_conv_kernel_config->l_ld_ofm_fil) * (i_conv_kernel_config->l_ld_ifm_fil) * (i_conv_desc->fm_lp_block) * (i_conv_kernel_config->datatype_size_wt)),
                                          i_conv_kernel_config->vector_name, (l_k+3)%4,
                                          0, 0 );

      }

      if (weight_counter >= (i_conv_desc->ofm_block-1)) {
        weight_counter = 0;
      } else weight_counter++;

      /* if required, apply additional register block to hide FMA latencies */
      if ( i_conv_desc->ofw_rb < 12 ) {
        current_acc_start = i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb* ((l_k_3 % l_accs) + 1));
      } else {
        current_acc_start = l_vec_reg_acc_start;
      }

      assert(0 != i_conv_desc->ofw_rb);
      /* compute vectorwidth (A) * column broadcast (B) */
      for ( l_n = 0; l_n < i_conv_desc->ofw_rb; l_n++) {
        /* determining base, idx and scale values */
        libxsmm_generator_convolution_backward_avx512_calc_sib_output_strides( i_gp_reg_mapping, l_n, &l_output_reg, &l_output_idx, &l_scale );
        /* set displacement */
        l_disp = l_displacement_k*i_conv_kernel_config->datatype_size_out*i_conv_desc->fm_lp_block;
        if (i_conv_desc->stride_w == 1) {
          input_reg = current_acc_start + ((l_n+l_k_2)%i_conv_desc->ofw_rb);
        } else {
          input_reg = current_acc_start + l_n;
        }

        /* depending on datatype emit the needed FMA(-sequence) */
        if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_F32 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
          libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                                   i_conv_kernel_config->instruction_set,
                                                   i_conv_kernel_config->vfma_instruction,
                                                   1,
                                                   l_output_reg,
                                                   l_output_idx,
                                                   l_scale,
                                                   l_disp,
                                                   i_conv_kernel_config->vector_name,
                                                   (l_k%4),
                                                   input_reg);

        } else if ( (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32) ||
                    (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I8  && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I16
                      && (i_conv_desc->option & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0) ) {
          /* broadcast in pairs of 8/16 bit values */
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_conv_kernel_config->instruction_set,
                                            i_conv_kernel_config->vbcst_instruction,
                                            l_output_reg,
                                            l_output_idx, l_scale,
                                            l_disp,
                                            i_conv_kernel_config->vector_name,
                                            4, 0, 0 );
          /* 8/16bit integer MADD with horizontal add */
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                   i_conv_kernel_config->instruction_set,
                                                   i_conv_kernel_config->vfma_instruction,
                                                   i_conv_kernel_config->vector_name,
                                                   l_k%4,
                                                   4,
                                                   5 );
          /* 16/32bit integer accumulation without saturation into running result buffer */
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                   i_conv_kernel_config->instruction_set,
                                                   i_conv_kernel_config->vadd_instruction,
                                                   i_conv_kernel_config->vector_name,
                                                   5,
                                                   input_reg,
                                                   input_reg );

        } else if ( (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I8  && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32
                       && (i_conv_desc->option & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0) ) {
          /* broadcast in quadruples of 8 bit values */
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_conv_kernel_config->instruction_set,
                                            i_conv_kernel_config->vbcst_instruction,
                                            l_output_reg,
                                            l_output_idx, l_scale,
                                            l_disp,
                                            i_conv_kernel_config->vector_name,
                                            4, 0, 0 );
          /* 8/16bit integer MADD with horizontal add */
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                   i_conv_kernel_config->instruction_set,
                                                   i_conv_kernel_config->vfma_instruction,
                                                   i_conv_kernel_config->vector_name,
                                                   l_k%4,
                                                   4,
                                                   5 );
          /* 16/32bit integer MADD with horizontal add */
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                   i_conv_kernel_config->instruction_set,
                                                   LIBXSMM_X86_INSTR_VPMADDWD,
                                                   i_conv_kernel_config->vector_name,
                                                   5,
                                                   6,
                                                   5 );
          /* 16/32bit integer accumulation without saturation into running result buffer */
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                   i_conv_kernel_config->instruction_set,
                                                   i_conv_kernel_config->vadd_instruction,
                                                   i_conv_kernel_config->vector_name,
                                                   5,
                                                   input_reg,
                                                   input_reg );

        } else {
          assert(0/*should not happen */);
        } /* End of FMA stream for different data types */

#ifdef ENABLE_OUTPUT_PREFETCH
        if ( (l_n == 2) && (l_k < num_output_prefetch) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            LIBXSMM_X86_INSTR_PREFETCHT0 ,
                                            i_gp_reg_mapping->gp_reg_output_pf,
                                            LIBXSMM_X86_GP_REG_UNDEF,
                                            0,
                                            (l_k * i_conv_kernel_config->l_ld_ofm_act * i_conv_desc->fm_lp_block * i_conv_kernel_config->datatype_size_out) );
        }
#endif
#ifdef ENABLE_WEIGHT_PREFETCH
        if ( (l_n == 4) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            i_conv_kernel_config->prefetch_instruction,
                                            i_gp_reg_mapping->gp_reg_weight_pf,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            ((l_k_2 * i_conv_kernel_config->l_ld_ofm_fil) + l_k_3) * i_conv_kernel_config->l_ld_ifm_fil * i_conv_desc->fm_lp_block * i_conv_kernel_config->datatype_size_wt);
        }
#endif
      } /* end of l_n over ofw_rb */
      l_displacement_k++;
    } /* end of l_k_3 over ofmblock*/
    /* Store inputs */
    libxsmm_generator_convolution_backward_avx512_store_input(io_generated_code, i_gp_reg_mapping, i_conv_kernel_config, i_conv_desc, l_vec_reg_acc_start, i_kw_unroll, l_accs, l_k_2);
  } /* end of l_k_2 over i_kw_unroll */
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_backward_avx512_ofmloop_sfma_two_rows( libxsmm_generated_code*                            io_generated_code,
                                                                 const libxsmm_convolution_backward_gp_reg_mapping* i_gp_reg_mapping,
                                                                 const libxsmm_convolution_kernel_config*           i_conv_kernel_config,
                                                                 const libxsmm_convolution_backward_descriptor*     i_conv_desc,
                                                                 const unsigned int                                 i_kw_unroll,
                                                                 const unsigned int                                 num_output_prefetch ) {
  unsigned int l_m, l_n;
  unsigned int l_k_2, l_k_3;
  unsigned int l_output_reg;
  unsigned int l_output_idx;
  unsigned int l_scale;
  unsigned int l_disp;
  unsigned int l_displacement_k = 0;
  unsigned int l_w;
  unsigned int input_reg;
  unsigned int l_k = 0;
  unsigned int weight_counter = 0;

  /****************************************/
  /***Assuming ifmblock is same as VLEN ***/
  /****************************************/

  /* start register of accumulator */
  const unsigned int l_vec_reg_acc_start = i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofh_rb * i_conv_desc->ofw_rb);

  /* if kw loop is not unrolled, we are running out of GPRs */
  if ( i_kw_unroll == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_INVALID_KW_UNROLL );
    return;
  }


  for ( l_k_2 = 0; l_k_2 < i_kw_unroll; l_k_2++) {
    /* load inputs */
    libxsmm_generator_convolution_backward_avx512_load_input_two_rows(io_generated_code, i_gp_reg_mapping, i_conv_kernel_config, i_conv_desc, l_vec_reg_acc_start, l_k_2);

    l_displacement_k = 0;
    for ( l_k_3 = 0; l_k_3 < i_conv_desc->ofm_block; l_k_3++, l_k++) {
      if ( l_k == 0) {
        /* load weights */
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_conv_kernel_config->instruction_set,
                                      i_conv_kernel_config->vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_weight,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      ((l_k_2 * i_conv_kernel_config->l_ld_ofm_fil + l_k_3) * i_conv_kernel_config->l_ld_ifm_fil * i_conv_desc->fm_lp_block)*(i_conv_kernel_config->datatype_size_wt),
                                      i_conv_kernel_config->vector_name, 0,
                                      0, 0 );
        if ( i_conv_desc->ofm_block *i_kw_unroll > 1 ) {
           for ( l_w = 1; l_w < 4; l_w++ ) {
            /* second weight loaded in first iteration, in case of large blockings -> hiding L1 latencies */
              libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_conv_kernel_config->instruction_set,
                                            i_conv_kernel_config->vmove_instruction,
                                            i_gp_reg_mapping->gp_reg_weight,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            ((l_k_2 * i_conv_kernel_config->l_ld_ofm_fil + l_k_3 + l_w) * i_conv_kernel_config->l_ld_ifm_fil * i_conv_desc->fm_lp_block) *(i_conv_kernel_config->datatype_size_wt),
                                            i_conv_kernel_config->vector_name, l_w,
                                            0, 0 );
          }
          weight_counter += 3;
        }
      } else if ((l_k < ((i_conv_desc->ofm_block * i_kw_unroll) - 3)) && (l_k_3 >= (i_conv_desc->ofm_block -3)) && (l_k_3 < i_conv_desc->ofm_block) ) {
        /* At the end of l_k_3, weight for next l_k_2 is loaded */
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_conv_kernel_config->instruction_set,
                                          i_conv_kernel_config->vmove_instruction,
                                          i_gp_reg_mapping->gp_reg_weight,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          ((weight_counter) * (i_conv_kernel_config->l_ld_ifm_fil) * (i_conv_desc->fm_lp_block) * (i_conv_kernel_config->datatype_size_wt) )
                                          + ((l_k_2 + 1) * (i_conv_kernel_config->l_ld_ofm_fil) * (i_conv_kernel_config->l_ld_ifm_fil) * (i_conv_desc->fm_lp_block) * (i_conv_kernel_config->datatype_size_wt)),
                                          i_conv_kernel_config->vector_name, (l_k+3)%4,
                                          0, 0 );
      } else if ( l_k < ((i_conv_desc->ofm_block * i_kw_unroll) - 3) ) {
        /* pipelined load of weight, one k iteration ahead */
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_conv_kernel_config->instruction_set,
                                          i_conv_kernel_config->vmove_instruction,
                                          i_gp_reg_mapping->gp_reg_weight,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          ((weight_counter) * (i_conv_kernel_config->l_ld_ifm_fil) * (i_conv_desc->fm_lp_block) * (i_conv_kernel_config->datatype_size_wt))
                                          + ((l_k_2) * (i_conv_kernel_config->l_ld_ofm_fil) * (i_conv_kernel_config->l_ld_ifm_fil) * (i_conv_desc->fm_lp_block) * (i_conv_kernel_config->datatype_size_wt)),
                                          i_conv_kernel_config->vector_name, (l_k+3)%4,
                                          0, 0 );

      }

      if (weight_counter >= (i_conv_desc->ofm_block-1)) {
        weight_counter = 0;
      } else weight_counter++;

      assert(0 != i_conv_desc->ofw_rb);
      /* compute vectorwidth (A) * column broadcast (B) */
      for (l_m = 0; l_m < i_conv_desc->ofh_rb; l_m++) {
        for ( l_n = 0; l_n < i_conv_desc->ofw_rb; l_n++) {
          /* determining base, idx and scale values */
          libxsmm_generator_convolution_backward_avx512_calc_sib_output_strides_two_rows( i_gp_reg_mapping, l_m, l_n, &l_output_reg, &l_output_idx, &l_scale );
          /* set displacement */
          l_disp = l_displacement_k*i_conv_kernel_config->datatype_size_out*i_conv_desc->fm_lp_block;
          if (i_conv_desc->stride_w == 1) {
            input_reg = l_vec_reg_acc_start + (l_m * i_conv_desc->ofw_rb) + ((l_n+l_k_2) % i_conv_desc->ofw_rb);
          } else {
            input_reg = l_vec_reg_acc_start + (l_m * i_conv_desc->ofw_rb) + l_n;
          }

          /* depending on datatype emit the needed FMA(-sequence) */
          if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_F32 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
            libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                                     i_conv_kernel_config->instruction_set,
                                                     i_conv_kernel_config->vfma_instruction,
                                                     1,
                                                     l_output_reg,
                                                     l_output_idx,
                                                     l_scale,
                                                     l_disp,
                                                     i_conv_kernel_config->vector_name,
                                                     (l_k%4),
                                                     input_reg);
          } else if ( (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32) ||
                      (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I8  && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I16
                       && (i_conv_desc->option & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0) ) {
            /* broadcast in pairs of 8/16 bit values */
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_conv_kernel_config->instruction_set,
                                              i_conv_kernel_config->vbcst_instruction,
                                              l_output_reg,
                                              l_output_idx, l_scale,
                                              l_disp,
                                              i_conv_kernel_config->vector_name,
                                              4, 0, 0 );
            /* 8/16bit integer MADD with horizontal add */
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                     i_conv_kernel_config->instruction_set,
                                                     i_conv_kernel_config->vfma_instruction,
                                                     i_conv_kernel_config->vector_name,
                                                     l_k%4,
                                                     4,
                                                     5 );
            /* 16/32bit integer accumulation without saturation into running result buffer */
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                     i_conv_kernel_config->instruction_set,
                                                     i_conv_kernel_config->vadd_instruction,
                                                     i_conv_kernel_config->vector_name,
                                                     5,
                                                     input_reg,
                                                     input_reg );

          } else if ( (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I8  && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32
                         && (i_conv_desc->option & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0) ) {
            /* broadcast in quadruples of 8 bit values */
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              i_conv_kernel_config->instruction_set,
                                              i_conv_kernel_config->vbcst_instruction,
                                              l_output_reg,
                                              l_output_idx, l_scale,
                                              l_disp,
                                              i_conv_kernel_config->vector_name,
                                              4, 0, 0 );
            /* 8/16bit integer MADD with horizontal add */
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                     i_conv_kernel_config->instruction_set,
                                                     i_conv_kernel_config->vfma_instruction,
                                                     i_conv_kernel_config->vector_name,
                                                     l_k%4,
                                                     4,
                                                     5 );
            /* 16/32bit integer MADD with horizontal add */
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                     i_conv_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VPMADDWD,
                                                     i_conv_kernel_config->vector_name,
                                                     5,
                                                     6,
                                                     5 );
            /* 16/32bit integer accumulation without saturation into running result buffer */
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                     i_conv_kernel_config->instruction_set,
                                                     i_conv_kernel_config->vadd_instruction,
                                                     i_conv_kernel_config->vector_name,
                                                     5,
                                                     input_reg,
                                                     input_reg );

          } else {
            assert(0/*should not happen */);
          } /* End of FMA stream for different data types */


#ifdef ENABLE_OUTPUT_PREFETCH
          if ( (l_n == 2) && (l_k < num_output_prefetch) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1) ) {
            libxsmm_x86_instruction_prefetch( io_generated_code,
                                              LIBXSMM_X86_INSTR_PREFETCHT0 ,
                                              i_gp_reg_mapping->gp_reg_output_pf,
                                              LIBXSMM_X86_GP_REG_UNDEF,
                                              0,
                                              (l_m * i_conv_desc->ofw_padded + l_k) * i_conv_kernel_config->l_ld_ofm_act * i_conv_desc->fm_lp_block * i_conv_kernel_config->datatype_size_out);
          }
#endif
#ifdef ENABLE_WEIGHT_PREFETCH
          if ( (l_n == 4) && (l_m == 0) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) ) {
            libxsmm_x86_instruction_prefetch( io_generated_code,
                                              i_conv_kernel_config->prefetch_instruction,
                                              i_gp_reg_mapping->gp_reg_weight_pf,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              ((l_k_2 * i_conv_kernel_config->l_ld_ofm_fil) + l_k_3) * i_conv_kernel_config->l_ld_ifm_fil * i_conv_desc->fm_lp_block * i_conv_kernel_config->datatype_size_wt);
          }
#endif
        } /* end of l_n over ofw_rb */
      } /* end of l_m over ofh_rb */
      l_displacement_k++;
    } /* end of l_k_3 over ofmblock*/
    /* Store inputs */
    libxsmm_generator_convolution_backward_avx512_store_input_two_rows(io_generated_code, i_gp_reg_mapping, i_conv_kernel_config, i_conv_desc, l_vec_reg_acc_start, i_kw_unroll, l_k_2);
  } /* end of l_k_2 over i_kw_unroll */
}


LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_backward_avx512_ofmloop_qfma( libxsmm_generated_code*                            io_generated_code,
                                                                 const libxsmm_convolution_backward_gp_reg_mapping* i_gp_reg_mapping,
                                                                 const libxsmm_convolution_kernel_config*           i_conv_kernel_config,
                                                                 const libxsmm_convolution_backward_descriptor*     i_conv_desc,
                                                                 const unsigned int                                 i_kw_unroll,
                                                                 const unsigned int                                 num_output_prefetch ) {
  unsigned int l_n;
  unsigned int l_k_2, l_k_3;
  unsigned int l_output_reg;
  unsigned int l_output_idx;
  unsigned int l_scale;
  unsigned int l_disp;
  unsigned int l_displacement_k = 0;
  unsigned int l_w;
  unsigned int input_reg;
  unsigned int l_k = 0;
  unsigned int current_acc_start;
  unsigned int l_qinstr = 0;

  /****************************************/
  /***Assuming ifmblock is same as VLEN ***/
  /****************************************/

  const unsigned int l_accs = 2; /* Number of extra accumulator used when required to hide FMA latencies */
  /* start register of accumulator */
  const unsigned int l_vec_reg_acc_start = i_conv_kernel_config->vector_reg_count - i_conv_desc->ofw_rb;

  if ((i_conv_desc->ofm_block < 4) || (i_conv_desc->ofm_block % 4 != 0)) {
    libxsmm_generator_convolution_backward_avx512_ofmloop_sfma( io_generated_code, i_gp_reg_mapping,
                                                                         i_conv_kernel_config, i_conv_desc, i_kw_unroll, num_output_prefetch );
    return;
  }
  for ( l_k_2 = 0; l_k_2 < i_kw_unroll; l_k_2++) {
    /* load inputs */
    libxsmm_generator_convolution_backward_avx512_load_input(io_generated_code, i_gp_reg_mapping, i_conv_kernel_config, i_conv_desc, l_vec_reg_acc_start, l_accs, l_k_2);

    l_displacement_k = 0;
    /* perform fused 4 FMAs for 4 consecutive values of ofm_block */
    for ( l_k_3 = 0; l_k_3 < i_conv_desc->ofm_block; l_k_3 += 4, l_k += 4) {
      /*load four source registers, we cannot perform a pipeline as in case of sfma */
      for ( l_w = 0; l_w < 4; l_w++ ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_conv_kernel_config->instruction_set,
                                      i_conv_kernel_config->vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_weight,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      (l_k_2 * i_conv_kernel_config->l_ld_ofm_fil + l_k_3 + l_w)*  (i_conv_kernel_config->l_ld_ifm_fil) * (i_conv_desc->fm_lp_block) * (i_conv_kernel_config->datatype_size_wt),
                                      i_conv_kernel_config->vector_name, l_w,
                                      0, 0 );
      }

      /* if required, apply additional register block to hide FMA latencies */
      if ( i_conv_desc->ofw_rb <= 14 ) {
        current_acc_start = i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb* ((l_k_3 % l_accs) + 1));
      } else {
        current_acc_start = l_vec_reg_acc_start;
      }

      assert(0 != i_conv_desc->ofw_rb);
      /* compute vectorwidth (A) * column broadcast (B) */
      for ( l_n = 0; l_n < i_conv_desc->ofw_rb; l_n++) {
        /* determining base, idx and scale values */
        libxsmm_generator_convolution_backward_avx512_calc_sib_output_strides( i_gp_reg_mapping, l_n, &l_output_reg, &l_output_idx, &l_scale );
        /* set displacement */
        l_disp = l_displacement_k*i_conv_kernel_config->datatype_size_out*i_conv_desc->fm_lp_block;
        if (i_conv_desc->stride_w == 1) {
          input_reg = current_acc_start + ((l_n+l_k_2)%i_conv_desc->ofw_rb);
        } else {
          input_reg = current_acc_start + l_n;
        }
        /* depending on datatype emit the needed FMA(-sequence) */
        if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_F32 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
          l_qinstr = LIBXSMM_X86_INSTR_V4FMADDPS;
        } else if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32 ) {
          l_qinstr = LIBXSMM_X86_INSTR_VP4DPWSSD;
        } else {
          assert(0/* should not happen */);
        }

        libxsmm_x86_instruction_vec_compute_qfma( io_generated_code,
                                                 i_conv_kernel_config->instruction_set,
                                                 l_qinstr,
                                                 l_output_reg,
                                                 l_output_idx,
                                                 l_scale,
                                                 l_disp,
                                                 i_conv_kernel_config->vector_name,
                                                 0,
                                                 input_reg);
#ifdef ENABLE_OUTPUT_PREFETCH
        if ( (l_n == 2) && ((l_k/2) < num_output_prefetch) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            LIBXSMM_X86_INSTR_PREFETCHT0 ,
                                            i_gp_reg_mapping->gp_reg_output_pf,
                                            LIBXSMM_X86_GP_REG_UNDEF,
                                            0,
                                            ((l_k/2)) * i_conv_kernel_config->l_ld_ofm_act * i_conv_desc->fm_lp_block * i_conv_kernel_config->datatype_size_out );
        }
        if ( (l_n == 4) && ((l_k/2) < num_output_prefetch) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            LIBXSMM_X86_INSTR_PREFETCHT0 ,
                                            i_gp_reg_mapping->gp_reg_output_pf,
                                            LIBXSMM_X86_GP_REG_UNDEF,
                                            0,
                                            ((l_k/2)+1) * i_conv_kernel_config->l_ld_ofm_act * i_conv_desc->fm_lp_block * i_conv_kernel_config->datatype_size_out );
        }
#endif
#ifdef ENABLE_WEIGHT_PREFETCH
        if ( (l_n == 1) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            i_conv_kernel_config->prefetch_instruction,
                                            i_gp_reg_mapping->gp_reg_weight_pf,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            ((l_k_2 * i_conv_kernel_config->l_ld_ofm_fil) + l_k_3) * i_conv_kernel_config->l_ld_ifm_fil * i_conv_desc->fm_lp_block * i_conv_kernel_config->datatype_size_wt);
        }
        if ( (l_n == 3) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            i_conv_kernel_config->prefetch_instruction,
                                            i_gp_reg_mapping->gp_reg_weight_pf,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            ((l_k_2 * i_conv_kernel_config->l_ld_ofm_fil) + l_k_3 + 1) * i_conv_kernel_config->l_ld_ifm_fil * i_conv_desc->fm_lp_block * i_conv_kernel_config->datatype_size_wt);
        }
        if ( (l_n == 5) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            i_conv_kernel_config->prefetch_instruction,
                                            i_gp_reg_mapping->gp_reg_weight_pf,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            ((l_k_2 * i_conv_kernel_config->l_ld_ofm_fil) + l_k_3 + 2) * i_conv_kernel_config->l_ld_ifm_fil * i_conv_desc->fm_lp_block * i_conv_kernel_config->datatype_size_wt);
        }
        if ( (l_n == 6) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            i_conv_kernel_config->prefetch_instruction,
                                            i_gp_reg_mapping->gp_reg_weight_pf,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            ((l_k_2 * i_conv_kernel_config->l_ld_ofm_fil) + l_k_3 + 3) * i_conv_kernel_config->l_ld_ifm_fil * i_conv_desc->fm_lp_block * i_conv_kernel_config->datatype_size_wt);
        }
#endif
      } /* end of l_n over ofw_rb */
      l_displacement_k += 4;
    } /* end of l_k_3 over ofmblock*/
    /* Store inputs */
    libxsmm_generator_convolution_backward_avx512_store_input(io_generated_code, i_gp_reg_mapping, i_conv_kernel_config, i_conv_desc, l_vec_reg_acc_start, i_kw_unroll, l_accs, l_k_2);
  } /* end of l_k_2 over i_kw_unroll */
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_backward_avx512_ofmloop_qfma_two_rows( libxsmm_generated_code*                            io_generated_code,
                                                                 const libxsmm_convolution_backward_gp_reg_mapping* i_gp_reg_mapping,
                                                                 const libxsmm_convolution_kernel_config*           i_conv_kernel_config,
                                                                 const libxsmm_convolution_backward_descriptor*     i_conv_desc,
                                                                 const unsigned int                                 i_kw_unroll,
                                                                 const unsigned int                                 num_output_prefetch ) {
  unsigned int l_m, l_n;
  unsigned int l_k_2, l_k_3;
  unsigned int l_output_reg;
  unsigned int l_output_idx;
  unsigned int l_scale;
  unsigned int l_disp;
  unsigned int l_displacement_k = 0;
  unsigned int l_w;
  unsigned int input_reg;
  unsigned int l_k = 0;
  unsigned int l_qinstr = 0;

  /****************************************/
  /***Assuming ifmblock is same as VLEN ***/
  /****************************************/

  /* start register of accumulator */
  const unsigned int l_vec_reg_acc_start = i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofh_rb * i_conv_desc->ofw_rb);

  /* if kw loop is not unrolled, we are running out of GPRs */
  if ( i_kw_unroll == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_INVALID_KW_UNROLL );
    return;
  }

  if ((i_conv_desc->ofm_block < 4) || (i_conv_desc->ofm_block % 4 != 0)) {
    libxsmm_generator_convolution_backward_avx512_ofmloop_sfma_two_rows( io_generated_code, i_gp_reg_mapping,
                                                                          i_conv_kernel_config, i_conv_desc, i_kw_unroll, num_output_prefetch );
    return;
  }


  for ( l_k_2 = 0; l_k_2 < i_kw_unroll; l_k_2++) {
    /* load inputs */
    libxsmm_generator_convolution_backward_avx512_load_input_two_rows(io_generated_code, i_gp_reg_mapping, i_conv_kernel_config, i_conv_desc, l_vec_reg_acc_start, l_k_2);

    l_displacement_k = 0;
    for ( l_k_3 = 0; l_k_3 < i_conv_desc->ofm_block; l_k_3+=4, l_k+=4) {
      /*load four source registers, we cannot perform a pipeline as in case of sfma */
      for ( l_w = 0; l_w < 4; l_w++ ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_conv_kernel_config->instruction_set,
                                    i_conv_kernel_config->vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_weight,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    (l_k_2 * i_conv_kernel_config->l_ld_ofm_fil + l_k_3 + l_w)*(i_conv_kernel_config->l_ld_ifm_fil)*(i_conv_desc->fm_lp_block)*(i_conv_kernel_config->datatype_size_wt),
                                    i_conv_kernel_config->vector_name, l_w,
                                    0, 0 );
      }

      /* compute vectorwidth (A) * column broadcast (B) */
      for (l_m = 0; l_m < i_conv_desc->ofh_rb; l_m++) {
        for ( l_n = 0; l_n < i_conv_desc->ofw_rb; l_n++) {
          /* determining base, idx and scale values */
          libxsmm_generator_convolution_backward_avx512_calc_sib_output_strides_two_rows( i_gp_reg_mapping, l_m, l_n, &l_output_reg, &l_output_idx, &l_scale );
          /* set displacement */
          l_disp = l_displacement_k*i_conv_kernel_config->datatype_size_out*i_conv_desc->fm_lp_block;
          if (i_conv_desc->stride_w == 1) {
            input_reg = l_vec_reg_acc_start + (l_m * i_conv_desc->ofw_rb) + ((l_n+l_k_2) % i_conv_desc->ofw_rb);
          } else {
            input_reg = l_vec_reg_acc_start + (l_m * i_conv_desc->ofw_rb) + l_n;
          }
          /* depending on datatype emit the needed FMA(-sequence) */
          if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_F32 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
            l_qinstr = LIBXSMM_X86_INSTR_V4FMADDPS;
          } else if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32 ) {
            l_qinstr = LIBXSMM_X86_INSTR_VP4DPWSSD;
          } else {
            assert(0/* should not happen */);
          }

          libxsmm_x86_instruction_vec_compute_qfma( io_generated_code,
                                                   i_conv_kernel_config->instruction_set,
                                                   l_qinstr,
                                                   l_output_reg,
                                                   l_output_idx,
                                                   l_scale,
                                                   l_disp,
                                                   i_conv_kernel_config->vector_name,
                                                   0,
                                                   input_reg );

#ifdef ENABLE_OUTPUT_PREFETCH
        if ( (l_n == 2) && ((l_k/2) < num_output_prefetch) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            LIBXSMM_X86_INSTR_PREFETCHT0 ,
                                            i_gp_reg_mapping->gp_reg_output_pf,
                                            LIBXSMM_X86_GP_REG_UNDEF,
                                            0,
                                            (l_m * i_conv_desc->ofw_padded + (l_k/2)) * i_conv_kernel_config->l_ld_ofm_act * i_conv_desc->fm_lp_block * i_conv_kernel_config->datatype_size_out );
        }
        if ( (l_n == 4) && ((l_k/2) < num_output_prefetch) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            LIBXSMM_X86_INSTR_PREFETCHT0 ,
                                            i_gp_reg_mapping->gp_reg_output_pf,
                                            LIBXSMM_X86_GP_REG_UNDEF,
                                            0,
                                            (l_m * i_conv_desc->ofw_padded + (l_k/2)+1) * i_conv_kernel_config->l_ld_ofm_act * i_conv_desc->fm_lp_block * i_conv_kernel_config->datatype_size_out );
        }
#endif
#ifdef ENABLE_WEIGHT_PREFETCH
        if ( (l_n == 1) && (l_m == 0) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            i_conv_kernel_config->prefetch_instruction,
                                            i_gp_reg_mapping->gp_reg_weight_pf,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            ((l_k_2 * i_conv_kernel_config->l_ld_ofm_fil) + l_k_3) * i_conv_kernel_config->l_ld_ifm_fil * i_conv_desc->fm_lp_block * i_conv_kernel_config->datatype_size_wt);
        }
        if ( (l_n == 3) && (l_m == 0) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            i_conv_kernel_config->prefetch_instruction,
                                            i_gp_reg_mapping->gp_reg_weight_pf,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            ((l_k_2 * i_conv_kernel_config->l_ld_ofm_fil) + l_k_3 + 1) * i_conv_kernel_config->l_ld_ifm_fil * i_conv_desc->fm_lp_block * i_conv_kernel_config->datatype_size_wt);
        }
        if ( (l_n == 5) && (l_m == 0) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            i_conv_kernel_config->prefetch_instruction,
                                            i_gp_reg_mapping->gp_reg_weight_pf,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            ((l_k_2 * i_conv_kernel_config->l_ld_ofm_fil) + l_k_3 + 2) * i_conv_kernel_config->l_ld_ifm_fil * i_conv_desc->fm_lp_block * i_conv_kernel_config->datatype_size_wt);
        }
        if ( (l_n == 6) && (l_m == 0) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            i_conv_kernel_config->prefetch_instruction,
                                            i_gp_reg_mapping->gp_reg_weight_pf,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            ((l_k_2 * i_conv_kernel_config->l_ld_ofm_fil) + l_k_3 + 3) * i_conv_kernel_config->l_ld_ifm_fil * i_conv_desc->fm_lp_block * i_conv_kernel_config->datatype_size_wt);
        }
#endif
        } /* end of l_n over ofw_rb */
      } /* end of l_m over ofh_rb */
      l_displacement_k+=4;
    } /* end of l_k_3 over ofmblock*/
    /* Store inputs */
    libxsmm_generator_convolution_backward_avx512_store_input_two_rows(io_generated_code, i_gp_reg_mapping, i_conv_kernel_config, i_conv_desc, l_vec_reg_acc_start, i_kw_unroll, l_k_2);
  } /* end of l_k_2 over i_kw_unroll */
}



#ifdef ENABLE_INPUT_PREFETCH
#undef ENABLE_INPUT_PREFETCH
#endif

#ifdef ENABLE_OUTPUT_PREFETCH
#undef ENABLE_OUTPUT_PREFETCH
#endif

#ifdef ENABLE_WEIGHT_PREFETCH
#undef ENABLE_WEIGHT_PREFETCH
#endif
