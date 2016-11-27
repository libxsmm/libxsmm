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
/* Alexander Heinecke (Intel Corp.), Naveen Mellempudi (Intel Corp.)
******************************************************************************/

#include "generator_convolution_forward_avx512.h"
#include "generator_convolution_common.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"

#include <libxsmm_cpuid.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_forward_avx512_kernel( libxsmm_generated_code*                       io_generated_code,
                                                          const libxsmm_convolution_forward_descriptor* i_conv_desc,
                                                          const char*                                   i_arch ) {
  libxsmm_convolution_kernel_config l_conv_kernel_config;
  libxsmm_convolution_forward_gp_reg_mapping l_gp_reg_mapping;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  unsigned int l_kw_trips = 1;
  unsigned int l_kh_trips = 1;
  unsigned int l_kh = 0;
  unsigned int l_found_act_format = 0;
  unsigned int l_found_fil_format = 0;
#if 0
  unsigned int l_kw = 0;
#endif
  /* define gp register mapping */
  libxsmm_reset_x86_convolution_forward_gp_reg_mapping( &l_gp_reg_mapping );
  l_gp_reg_mapping.gp_reg_input = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_weight = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_output = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_input_pf = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_weight_pf = LIBXSMM_X86_GP_REG_R8;
  l_gp_reg_mapping.gp_reg_output_pf = LIBXSMM_X86_GP_REG_R9;
  l_gp_reg_mapping.gp_reg_kw_loop = LIBXSMM_X86_GP_REG_R15;
  l_gp_reg_mapping.gp_reg_kh_loop = LIBXSMM_X86_GP_REG_R12;
  l_gp_reg_mapping.gp_reg_ifmInner_loop = LIBXSMM_X86_GP_REG_UNDEF;
  /*l_gp_reg_mapping.gp_reg_ifmInner_loop = LIBXSMM_X86_GP_REG_RDX;*/  /* this is reuse of the output pointer GPR */
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_X86_GP_REG_RAX;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_X86_GP_REG_RBX;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_X86_GP_REG_R10;
  l_gp_reg_mapping.gp_reg_help_3 = LIBXSMM_X86_GP_REG_R11;
  l_gp_reg_mapping.gp_reg_help_4 = LIBXSMM_X86_GP_REG_R13;
  l_gp_reg_mapping.gp_reg_help_5 = LIBXSMM_X86_GP_REG_R14;
  l_gp_reg_mapping.gp_reg_help_6 = LIBXSMM_X86_GP_REG_R15;

  /* define convolution kernel config */
  libxsmm_generator_init_convolution_kernel_config( &l_conv_kernel_config );
  if ( strcmp( i_arch, "knl" ) == 0 ) {
    l_conv_kernel_config.instruction_set = LIBXSMM_X86_AVX512_MIC;
  } else if (strcmp( i_arch, "skx" ) == 0 ) {
    l_conv_kernel_config.instruction_set = LIBXSMM_X86_AVX512_CORE;
  } else {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
    return;
  }
  l_conv_kernel_config.vector_reg_count = 32;
  if (i_conv_desc->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && i_conv_desc->datatype_out == LIBXSMM_DNN_DATATYPE_F32) {
    l_conv_kernel_config.vector_length_in = 16;
    l_conv_kernel_config.datatype_size_in = 4;
    l_conv_kernel_config.vector_length_out = 16;
    l_conv_kernel_config.datatype_size_out = 4;
    l_conv_kernel_config.vector_length_wt = 16;
    l_conv_kernel_config.datatype_size_wt = 4;
    l_conv_kernel_config.vfma_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
    l_conv_kernel_config.vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
    l_conv_kernel_config.vbcst_instruction = LIBXSMM_X86_INSTR_VBROADCASTSS;
  } else if (i_conv_desc->datatype_in == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_out == LIBXSMM_DNN_DATATYPE_I32) {
    l_conv_kernel_config.vector_length_in = 32;
    l_conv_kernel_config.datatype_size_in = 2;
    l_conv_kernel_config.vector_length_out = 16;
    l_conv_kernel_config.datatype_size_out = 4;
    l_conv_kernel_config.vector_length_wt = 32;
    l_conv_kernel_config.datatype_size_wt = 2;
    l_conv_kernel_config.vfma_instruction = LIBXSMM_X86_INSTR_VPMADDWD;
    l_conv_kernel_config.vadd_instruction = LIBXSMM_X86_INSTR_VPADDD;
    l_conv_kernel_config.vbcst_instruction = LIBXSMM_X86_INSTR_VPBROADCASTD;
  } else if (i_conv_desc->datatype_in == LIBXSMM_DNN_DATATYPE_I8 && i_conv_desc->datatype_out == LIBXSMM_DNN_DATATYPE_I16) {
    l_conv_kernel_config.vector_length_in = 64;
    l_conv_kernel_config.datatype_size_in = 1;
    l_conv_kernel_config.vector_length_out = 32;
    l_conv_kernel_config.datatype_size_out = 2;
    l_conv_kernel_config.vector_length_wt = 64;
    l_conv_kernel_config.datatype_size_wt = 1;
    l_conv_kernel_config.vfma_instruction = LIBXSMM_X86_INSTR_VPMADDUBSW;
    l_conv_kernel_config.vadd_instruction = LIBXSMM_X86_INSTR_VPADDW;
    l_conv_kernel_config.vbcst_instruction = LIBXSMM_X86_INSTR_VPBROADCASTW;
  } else if (i_conv_desc->datatype_in == LIBXSMM_DNN_DATATYPE_I8 && i_conv_desc->datatype_out == LIBXSMM_DNN_DATATYPE_I32) {
    l_conv_kernel_config.vector_length_in = 64;
    l_conv_kernel_config.datatype_size_in = 1;
    l_conv_kernel_config.vector_length_out = 16;
    l_conv_kernel_config.datatype_size_out = 4;
    l_conv_kernel_config.vector_length_wt = 64;
    l_conv_kernel_config.datatype_size_wt = 1;
    l_conv_kernel_config.vfma_instruction = LIBXSMM_X86_INSTR_VPMADDUBSW;
    l_conv_kernel_config.vadd_instruction = LIBXSMM_X86_INSTR_VPADDD;
    l_conv_kernel_config.vbcst_instruction = LIBXSMM_X86_INSTR_VPBROADCASTD;
  } else {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }
  l_conv_kernel_config.vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
  l_conv_kernel_config.vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
  l_conv_kernel_config.prefetch_instruction = LIBXSMM_X86_INSTR_PREFETCHT2;
  /*l_conv_kernel_config.alu_mul_instruction = LIBXSMM_X86_INSTR_IMULQ;*/
  l_conv_kernel_config.alu_add_instruction = LIBXSMM_X86_INSTR_ADDQ;
  l_conv_kernel_config.alu_sub_instruction = LIBXSMM_X86_INSTR_SUBQ;
  l_conv_kernel_config.alu_cmp_instruction = LIBXSMM_X86_INSTR_CMPQ;
  l_conv_kernel_config.alu_jmp_instruction = LIBXSMM_X86_INSTR_JL;
  l_conv_kernel_config.alu_mov_instruction = LIBXSMM_X86_INSTR_MOVQ;
  l_conv_kernel_config.vector_name = 'z';
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
    if (i_conv_desc->datatype_in != LIBXSMM_DNN_DATATYPE_F32) {
      libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_UNSUP_DT_FORMAT );
      return;
    }
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

  /* initilize KW and KH unrolling */
  if (i_conv_desc->unroll_kw != 0) {
    l_kw_trips = i_conv_desc->kw;
  }
  if (i_conv_desc->unroll_kh != 0) {
    l_kh_trips = i_conv_desc->kh;
  }

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* open asm */
  libxsmm_x86_instruction_open_stream_convolution( io_generated_code, l_gp_reg_mapping.gp_reg_input,
                                                   l_gp_reg_mapping.gp_reg_weight, l_gp_reg_mapping.gp_reg_output,
                                                   l_gp_reg_mapping.gp_reg_input_pf, l_gp_reg_mapping.gp_reg_weight_pf,
                                                   l_gp_reg_mapping.gp_reg_output_pf, i_arch );

  /* load an additoinal temp register with 32 16bit 1s */
  if (i_conv_desc->datatype_in == LIBXSMM_DNN_DATATYPE_I8 && i_conv_desc->datatype_out == LIBXSMM_DNN_DATATYPE_I32) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_help_0, 65537 );
    libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_reg_mapping.gp_reg_help_0 );

    libxsmm_x86_instruction_vec_move( io_generated_code,
                                      l_conv_kernel_config.instruction_set,
                                      l_conv_kernel_config.vbcst_instruction,
                                      LIBXSMM_X86_GP_REG_RSP ,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                      l_conv_kernel_config.vector_name, 6, 0, 0 );
/*
    libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             l_conv_kernel_config.instruction_set,
                                             l_conv_kernel_config.vxor_instruction,
                                             l_conv_kernel_config.vector_name,
                                             6,
                                             6,
                                             6 );
*/

    libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_reg_mapping.gp_reg_help_0 );
  }

  /* oj loop */
  /*libxsmm_generator_convolution_header_oj_loop(  io_generated_code, &l_loop_label_tracker,
                                                  &l_gp_reg_mapping,  &l_conv_kernel_config, i_conv_desc );*/

  /* oi loop */
  /*libxsmm_generator_convolution_header_oi_loop(  io_generated_code, &l_loop_label_tracker,
                                                  &l_gp_reg_mapping,  &l_conv_kernel_config, i_conv_desc );*/

  /* load outputs */
  /* @TODO we just go with 1D blocking */
#if 1
  libxsmm_generator_convolution_forward_load_output( io_generated_code, &l_gp_reg_mapping, &l_conv_kernel_config, i_conv_desc );
#endif
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

    if (l_kw_trips == 1) {
      /* ifmInner loop, VLEN, ifm2, fully unrolled blocked by ofw_rb * ofw_rb */
      libxsmm_generator_convolution_forward_avx512_ifmloop(  io_generated_code,
                                                            &l_gp_reg_mapping,
                                                            &l_conv_kernel_config,
                                                             i_conv_desc,
                                                             l_kw_trips );

      /* adjust weight pointer by ifmBlock times datatype size */
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
                                        l_conv_kernel_config.alu_add_instruction,
                                        l_gp_reg_mapping.gp_reg_weight,
                                        l_conv_kernel_config.l_ld_ifm_fil * i_conv_desc->fm_lp_block * l_conv_kernel_config.l_ld_ofm_fil * l_conv_kernel_config.datatype_size_wt );

      if ( (i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2 ) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
                                          l_conv_kernel_config.alu_add_instruction,
                                          l_gp_reg_mapping.gp_reg_weight_pf,
                                          l_conv_kernel_config.l_ld_ifm_fil * i_conv_desc->fm_lp_block * l_conv_kernel_config.l_ld_ofm_fil * l_conv_kernel_config.datatype_size_wt );
      }

      /* adjust innput pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                       l_conv_kernel_config.alu_add_instruction,
                                       l_gp_reg_mapping.gp_reg_input,
                                       l_conv_kernel_config.l_ld_ifm_act * i_conv_desc->fm_lp_block * l_conv_kernel_config.datatype_size_in );
    } else {
      /* ifmInner loop, VLEN, ifm2, fully unrolled blocked by ofw_rb * ofw_rb */
      libxsmm_generator_convolution_forward_avx512_ifmloop(  io_generated_code,
                                                            &l_gp_reg_mapping,
                                                            &l_conv_kernel_config,
                                                             i_conv_desc,
                                                             l_kw_trips );
    }

    if ( i_conv_desc->unroll_kw == 0 ) {
      /* close KW loop, ki */
      libxsmm_generator_convolution_footer_kw_loop(  io_generated_code, &l_loop_label_tracker,
                                                    &l_conv_kernel_config, l_gp_reg_mapping.gp_reg_kw_loop, i_conv_desc->kw );
    }

    if (l_kw_trips == 1) {
      /* adjust input pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                       l_conv_kernel_config.alu_sub_instruction,
                                       l_gp_reg_mapping.gp_reg_input,
                                       i_conv_desc->kw * l_conv_kernel_config.l_ld_ifm_act * i_conv_desc->fm_lp_block * l_conv_kernel_config.datatype_size_in );
    } else {
      /* adjust weight pointer by ifmBlock times datatype size */
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
                                        l_conv_kernel_config.alu_add_instruction,
                                        l_gp_reg_mapping.gp_reg_weight,
                                        i_conv_desc->kw * l_conv_kernel_config.l_ld_ifm_fil * i_conv_desc->fm_lp_block * l_conv_kernel_config.l_ld_ofm_fil * l_conv_kernel_config.datatype_size_wt );

      if ( (i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2 ) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
                                          l_conv_kernel_config.alu_add_instruction,
                                          l_gp_reg_mapping.gp_reg_weight_pf,
                                          i_conv_desc->kw * l_conv_kernel_config.l_ld_ifm_fil * i_conv_desc->fm_lp_block * l_conv_kernel_config.l_ld_ofm_fil * l_conv_kernel_config.datatype_size_wt );
      }
    }

    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     l_conv_kernel_config.alu_add_instruction,
                                     l_gp_reg_mapping.gp_reg_input,
                                     i_conv_desc->ifw_padded * l_conv_kernel_config.l_ld_ifm_act * i_conv_desc->fm_lp_block * l_conv_kernel_config.datatype_size_in );

    if ( (i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1 ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                       l_conv_kernel_config.alu_add_instruction,
                                       l_gp_reg_mapping.gp_reg_input_pf,
                                       i_conv_desc->ifw_padded * l_conv_kernel_config.l_ld_ifm_act * i_conv_desc->fm_lp_block * l_conv_kernel_config.datatype_size_in );
    }
  }

  if ( i_conv_desc->unroll_kh == 0 ) {
    /* close KH loop, kj */
    libxsmm_generator_convolution_footer_kh_loop(  io_generated_code, &l_loop_label_tracker,
                                                  &l_conv_kernel_config, l_gp_reg_mapping.gp_reg_kh_loop, i_conv_desc->kh );
  }

    /* store outputs */
  /* @TODO we just go with 1D blocking */
#if 1
  libxsmm_generator_convolution_forward_store_output( io_generated_code, &l_gp_reg_mapping, &l_conv_kernel_config, i_conv_desc );
#endif
  /* oi loop */
  /*libxsmm_generator_convolution_footer_oi_loop(  io_generated_code, &l_loop_label_tracker,
                                                  &l_gp_reg_mapping,  &l_conv_kernel_config, i_conv_desc );*/

  /* oj loop */
  /*libxsmm_generator_convolution_footer_oj_loop(  io_generated_code, &l_loop_label_tracker,
                                                  &l_gp_reg_mapping,  &l_conv_kernel_config, i_conv_desc );*/

  /* close asm */
  libxsmm_x86_instruction_close_stream_convolution( io_generated_code, i_arch );
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_forward_avx512_ifmloop( libxsmm_generated_code*                           io_generated_code,
                                                           const libxsmm_convolution_forward_gp_reg_mapping* i_gp_reg_mapping,
                                                           const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                           const libxsmm_convolution_forward_descriptor*     i_conv_desc,
                                                           const unsigned int                                i_kw_unroll )
{
  if (i_conv_desc->ofh_rb == 2) {
    /* setup input strides */
    libxsmm_generator_convolution_forward_avx512_init_input_strides_two_rows( io_generated_code, i_gp_reg_mapping, i_conv_kernel_config, i_conv_desc );

    /* select architecture */
    if ( i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_MIC  ||
         i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_CORE    ) {
      libxsmm_generator_convolution_forward_avx512_ifmloop_sfma_two_rows( io_generated_code, i_gp_reg_mapping,
                                                                          i_conv_kernel_config, i_conv_desc, i_kw_unroll );
    }
  } else {
    /* setup input strides */
    libxsmm_generator_convolution_forward_avx512_init_input_strides( io_generated_code, i_gp_reg_mapping, i_conv_kernel_config, i_conv_desc );

    /* select architecture */
    if ( i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_MIC  ||
         i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_CORE    ) {
      libxsmm_generator_convolution_forward_avx512_ifmloop_sfma( io_generated_code, i_gp_reg_mapping,
                                                                 i_conv_kernel_config, i_conv_desc, i_kw_unroll );
    }
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_forward_avx512_init_input_strides( libxsmm_generated_code*                           io_generated_code,
                                                                      const libxsmm_convolution_forward_gp_reg_mapping* i_gp_reg_mapping,
                                                                      const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                                      const libxsmm_convolution_forward_descriptor*     i_conv_desc ) {
  /* Intialize helper registers for SIB addressing */
  /* helper 0: Index register holding ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_help_0, i_conv_kernel_config->datatype_size_in * i_conv_desc->stride_w
                                     * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block );
  /* helper 1: Index register holding 3*ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_help_1, i_conv_kernel_config->datatype_size_in * i_conv_desc->stride_w
                                     * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block * 3 );
  /* helper 2: Index register holding 5*ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_help_2, i_conv_kernel_config->datatype_size_in * i_conv_desc->stride_w
                                     * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block * 5 );
  /* helper 3: Index register holding 7*ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_help_3, i_conv_kernel_config->datatype_size_in * i_conv_desc->stride_w
                                     * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block * 7 );

  /* helper 4: B + 9*ldb, additional base address
     helper 5: B + 18*ldb, additional base adrress */
  if ( i_conv_desc->ofw_rb > 9 ) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_input, i_gp_reg_mapping->gp_reg_help_4);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_help_4,  9 * i_conv_kernel_config->datatype_size_in
                                       * i_conv_desc->stride_w * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block);
  }
  if ( i_conv_desc->ofw_rb > 18 ) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_input, i_gp_reg_mapping->gp_reg_help_5);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_help_5, 18 *  i_conv_kernel_config->datatype_size_in
                                       * i_conv_desc->stride_w * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block );
  }
  if ( i_conv_desc->ofw_rb > 27 ) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_input, i_gp_reg_mapping->gp_reg_help_6);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_help_6, 27 *  i_conv_kernel_config->datatype_size_in
                                       * i_conv_desc->stride_w * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block );
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_forward_avx512_init_input_strides_two_rows( libxsmm_generated_code*                           io_generated_code,
                                                                               const libxsmm_convolution_forward_gp_reg_mapping* i_gp_reg_mapping,
                                                                               const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                                               const libxsmm_convolution_forward_descriptor*     i_conv_desc ) {
  /* if kw loop is not unrolled, we are running out of GPRs */
  if ( i_conv_desc->unroll_kw == 0 ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_INVALID_KW_UNROLL );
    return;
  }

  /* Intialize helper registers for SIB addressing */
  /* helper 0: Index register holding ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_help_0, i_conv_kernel_config->datatype_size_in * i_conv_desc->stride_w
                                     * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block );
  /* helper 1: Index register holding 3*ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_help_1, i_conv_kernel_config->datatype_size_in * i_conv_desc->stride_w
                                     * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block * 3 );
  /* helper 2: Index register holding 5*ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_help_2, i_conv_kernel_config->datatype_size_in * i_conv_desc->stride_w
                                     * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block * 5 );
  /* helper 3: Index register holding 7*ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction,
                                   i_gp_reg_mapping->gp_reg_help_3, i_conv_kernel_config->datatype_size_in * i_conv_desc->stride_w
                                     * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block * 7 );

  /* helper 4: B+9*ldb,            additional base address
     helper 5: B+ifw_padded,       additional base adrress
     helper 6: B+ifw_padded+9*ldb, additional base adrress */
  libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_input, i_gp_reg_mapping->gp_reg_help_5);
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                   i_gp_reg_mapping->gp_reg_help_5, i_conv_kernel_config->datatype_size_in * i_conv_desc->ifw_padded
                                     * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block );
  if ( i_conv_desc->ofw_rb > 9 ) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_input, i_gp_reg_mapping->gp_reg_help_4);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_help_4,  9 * i_conv_kernel_config->datatype_size_in * i_conv_desc->stride_w
                                       * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block );
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_input, i_gp_reg_mapping->gp_reg_help_6);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_help_6, (i_conv_kernel_config->datatype_size_in * i_conv_desc->ifw_padded
                                       * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block )
                                       + (9 * i_conv_kernel_config->datatype_size_in * i_conv_desc->stride_w
                                          * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block )    );
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_forward_avx512_advance_input_strides( libxsmm_generated_code*                           io_generated_code,
                                                                         const libxsmm_convolution_forward_gp_reg_mapping* i_gp_reg_mapping,
                                                                         const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                                         const libxsmm_convolution_forward_descriptor*     i_conv_desc,
                                                                         const int                                         i_advance_offset ) {
  /* first base pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_input, i_advance_offset * i_conv_kernel_config->datatype_size_in * i_conv_desc->fm_lp_block );
  /* advance the second base pointer only if it's needed */
  if ( i_conv_desc->ofw_rb > 8 ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                       i_gp_reg_mapping->gp_reg_help_4, i_advance_offset * i_conv_kernel_config->datatype_size_in * i_conv_desc->fm_lp_block );
  }
  /* advance the third base pointer only if it's needed */
  if ( i_conv_desc->ofw_rb > 17 ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                       i_gp_reg_mapping->gp_reg_help_5, i_advance_offset * i_conv_kernel_config->datatype_size_in * i_conv_desc->fm_lp_block );
  }
  /* advance the fourth base pointer only if it's needed */
  if ( i_conv_desc->ofw_rb > 26 ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                       i_gp_reg_mapping->gp_reg_help_6, i_advance_offset * i_conv_kernel_config->datatype_size_in * i_conv_desc->fm_lp_block );
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_forward_avx512_advance_input_strides_two_rows( libxsmm_generated_code*                           io_generated_code,
                                                                                  const libxsmm_convolution_forward_gp_reg_mapping* i_gp_reg_mapping,
                                                                                  const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                                                  const libxsmm_convolution_forward_descriptor*     i_conv_desc,
                                                                                  const int                                         i_advance_offset ) {
  /* base pointers for frist 9 image columns */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_input, i_advance_offset * i_conv_kernel_config->datatype_size_in * i_conv_desc->fm_lp_block );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_help_5, i_advance_offset * i_conv_kernel_config->datatype_size_in * i_conv_desc->fm_lp_block );

  /* advance the second base pointer only if it's needed */
  if ( i_conv_desc->ofw_rb > 8 ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                       i_gp_reg_mapping->gp_reg_help_4, i_advance_offset * i_conv_kernel_config->datatype_size_in * i_conv_desc->fm_lp_block );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                       i_gp_reg_mapping->gp_reg_help_6, i_advance_offset * i_conv_kernel_config->datatype_size_in * i_conv_desc->fm_lp_block );
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_forward_avx512_calc_sib_input_strides( const libxsmm_convolution_forward_gp_reg_mapping* i_gp_reg_mapping,
                                                                          const int                                         i_stride_num,
                                                                                unsigned int*                               o_input_reg,
                                                                                unsigned int*                               o_input_idx,
                                                                                unsigned int*                               o_scale ) {
  /* init output variables */
  *o_input_reg = i_gp_reg_mapping->gp_reg_input;
  *o_input_idx = LIBXSMM_X86_GP_REG_UNDEF;
  *o_scale = 0;

  /* select the base register */
  if ( i_stride_num > 26 ) {
    *o_input_reg = i_gp_reg_mapping->gp_reg_help_6;
  } else if ( i_stride_num > 17 ) {
    *o_input_reg = i_gp_reg_mapping->gp_reg_help_5;
  } else if ( i_stride_num > 8 ) {
    *o_input_reg = i_gp_reg_mapping->gp_reg_help_4;
  } else {
    *o_input_reg = i_gp_reg_mapping->gp_reg_input;
  }

  /* select scale and index */
  if ( i_stride_num % 9 == 0 ) {
    *o_input_idx = LIBXSMM_X86_GP_REG_UNDEF;
    *o_scale = 0;
  } else if ( i_stride_num % 9 == 1 ) {
    *o_input_idx = i_gp_reg_mapping->gp_reg_help_0;
    *o_scale = 1;
  } else if ( i_stride_num % 9 == 2 ) {
    *o_input_idx = i_gp_reg_mapping->gp_reg_help_0;
    *o_scale = 2;
  } else if ( i_stride_num % 9 == 3 ) {
    *o_input_idx = i_gp_reg_mapping->gp_reg_help_1;
    *o_scale = 1;
  } else if ( i_stride_num % 9 == 4 ) {
    *o_input_idx = i_gp_reg_mapping->gp_reg_help_0;
    *o_scale = 4;
  } else if ( i_stride_num % 9 == 5 ) {
    *o_input_idx = i_gp_reg_mapping->gp_reg_help_2;
    *o_scale = 1;
  } else if ( i_stride_num % 9 == 6 ) {
    *o_input_idx = i_gp_reg_mapping->gp_reg_help_1;
    *o_scale = 2;
  } else if ( i_stride_num % 9 == 7 ) {
    *o_input_idx = i_gp_reg_mapping->gp_reg_help_3;
    *o_scale = 1;
  } else if ( i_stride_num % 9 == 8 ) {
    *o_input_idx = i_gp_reg_mapping->gp_reg_help_0;
    *o_scale = 8;
  } else {
    /* shouldn't happen.... */
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_forward_avx512_calc_sib_input_strides_two_rows( const libxsmm_convolution_forward_gp_reg_mapping* i_gp_reg_mapping,
                                                                                   const int                                         i_row_num,
                                                                                   const int                                         i_stride_num,
                                                                                         unsigned int*                               o_input_reg,
                                                                                         unsigned int*                               o_input_idx,
                                                                                         unsigned int*                               o_scale ) {
  /* init output variables */
  *o_input_reg = i_gp_reg_mapping->gp_reg_input;
  *o_input_idx = LIBXSMM_X86_GP_REG_UNDEF;
  *o_scale = 0;

  /* select the base register */
  if ( i_row_num == 1 ) {
    if ( i_stride_num > 8 ) {
      *o_input_reg = i_gp_reg_mapping->gp_reg_help_6;
    } else {
      *o_input_reg = i_gp_reg_mapping->gp_reg_help_5;
    }
  } else {
    if ( i_stride_num > 8 ) {
      *o_input_reg = i_gp_reg_mapping->gp_reg_help_4;
    } else {
      *o_input_reg = i_gp_reg_mapping->gp_reg_input;
    }
  }

  /* select scale and index */
  if ( i_stride_num % 9 == 0 ) {
    *o_input_idx = LIBXSMM_X86_GP_REG_UNDEF;
    *o_scale = 0;
  } else if ( i_stride_num % 9 == 1 ) {
    *o_input_idx = i_gp_reg_mapping->gp_reg_help_0;
    *o_scale = 1;
  } else if ( i_stride_num % 9 == 2 ) {
    *o_input_idx = i_gp_reg_mapping->gp_reg_help_0;
    *o_scale = 2;
  } else if ( i_stride_num % 9 == 3 ) {
    *o_input_idx = i_gp_reg_mapping->gp_reg_help_1;
    *o_scale = 1;
  } else if ( i_stride_num % 9 == 4 ) {
    *o_input_idx = i_gp_reg_mapping->gp_reg_help_0;
    *o_scale = 4;
  } else if ( i_stride_num % 9 == 5 ) {
    *o_input_idx = i_gp_reg_mapping->gp_reg_help_2;
    *o_scale = 1;
  } else if ( i_stride_num % 9 == 6 ) {
    *o_input_idx = i_gp_reg_mapping->gp_reg_help_1;
    *o_scale = 2;
  } else if ( i_stride_num % 9 == 7 ) {
    *o_input_idx = i_gp_reg_mapping->gp_reg_help_3;
    *o_scale = 1;
  } else if ( i_stride_num % 9 == 8 ) {
    *o_input_idx = i_gp_reg_mapping->gp_reg_help_0;
    *o_scale = 8;
  } else {
    /* shouldn't happen.... */
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_forward_avx512_ifmloop_sfma( libxsmm_generated_code*                           io_generated_code,
                                                                const libxsmm_convolution_forward_gp_reg_mapping* i_gp_reg_mapping,
                                                                const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                                const libxsmm_convolution_forward_descriptor*     i_conv_desc,
                                                                const unsigned int                                i_kw_unroll ) {
  unsigned int l_n;
  unsigned int l_k;
  unsigned int l_input_reg;
  unsigned int l_input_idx;
  unsigned int l_scale;
  unsigned int l_disp;
  unsigned int l_displacement_k = 0;
  unsigned int l_k_updates = 0;
  unsigned int l_w;
  unsigned int l_accs;
  unsigned int l_reg_block;
  unsigned int l_filter_pos = 0;
  unsigned int l_input_offset = 0;

  /* apply k blocking */
  for ( l_k = 0; l_k < i_conv_desc->ifm_block*i_kw_unroll ; l_k++ ) {
    /* if we are not in LIBXSMM storage format, there are jumps */
    if ( (l_k > 0) && (l_k % i_conv_desc->ifm_block == 0) ) {
      /* input pointer advance */
      if ( (i_conv_desc->format & LIBXSMM_DNN_CONV_FORMAT_NHWC) > 0 ) {
        l_displacement_k += (i_conv_kernel_config->l_ld_ifm_act - i_conv_desc->ifm_block) * i_conv_desc->fm_lp_block;
      }
    }

    /* advance b pointer if needed */
    if ( l_displacement_k >= 128 ) {
      l_input_offset = 0;
      while ( l_displacement_k >= 128 ) {
        l_input_offset += 128;
        l_displacement_k -= 128;
        l_k_updates++;
      }
      libxsmm_generator_convolution_forward_avx512_advance_input_strides( io_generated_code, i_gp_reg_mapping, i_conv_kernel_config,
                                                                            i_conv_desc, l_input_offset );
    }

    if ( l_k == 0 ) {
      /* load weights */
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_conv_kernel_config->instruction_set,
                                        i_conv_kernel_config->vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_weight,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        l_filter_pos * i_conv_kernel_config->l_ld_ofm_fil * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block,
                                        i_conv_kernel_config->vector_name, 0,
                                        0, 0 );
      l_filter_pos++;

      if ( i_conv_desc->ifm_block*i_kw_unroll > 1 ) {
        for ( l_w = 1; l_w < 4; l_w++ ) {
          if ((l_k+l_w)%i_conv_desc->ifm_block == 0) {
            l_filter_pos += (i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block) * i_conv_desc->fm_lp_block;
          }
          /* second weight loaded in first iteration, in case of large blockings -> hiding L1 latencies */
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_conv_kernel_config->instruction_set,
                                            i_conv_kernel_config->vmove_instruction,
                                            i_gp_reg_mapping->gp_reg_weight,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            l_filter_pos * i_conv_kernel_config->l_ld_ofm_fil * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block,
                                            i_conv_kernel_config->vector_name, l_w,
                                            0, 0 );
          l_filter_pos++;
        }
      }
    } else if ( l_k < ((i_conv_desc->ifm_block*i_kw_unroll) - 3) ) {
      if ((l_k+3)%i_conv_desc->ifm_block == 0) {
        l_filter_pos += (i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block) * i_conv_desc->fm_lp_block;
      }
      /* pipelined load of weight, one k iteration ahead */
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_conv_kernel_config->instruction_set,
                                        i_conv_kernel_config->vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_weight,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        l_filter_pos * i_conv_kernel_config->l_ld_ofm_fil * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block,
                                        i_conv_kernel_config->vector_name, (l_k+3)%4,
                                        0, 0 );
      l_filter_pos++;
    }

    /* apply additional register block to hide FMA latencies */
    l_reg_block = i_conv_desc->ofw_rb;
    if ( i_conv_desc->ofw_rb < 12 ) {
      l_accs = (i_conv_desc->ofw_rb < 9) ? 3 : 2;
      l_reg_block = ((l_k%l_accs)+1)*i_conv_desc->ofw_rb;
    }

    /* compute vectorwidth (A) * column broadcast (B) */
    for ( l_n = 0; l_n < i_conv_desc->ofw_rb; l_n++) {
      /* determining base, idx and scale values */
      libxsmm_generator_convolution_forward_avx512_calc_sib_input_strides( i_gp_reg_mapping, l_n, &l_input_reg, &l_input_idx, &l_scale );
      /* set displacement */
      l_disp = l_displacement_k * i_conv_kernel_config->datatype_size_in * i_conv_desc->fm_lp_block;

      /* depending on datatype emit the needed FMA(-sequence) */
      if ( i_conv_desc->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && i_conv_desc->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
        libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                                 i_conv_kernel_config->instruction_set,
                                                 i_conv_kernel_config->vfma_instruction,
                                                 1,
                                                 l_input_reg,
                                                 l_input_idx,
                                                 l_scale,
                                                 l_disp,
                                                 i_conv_kernel_config->vector_name,
                                                 l_k%4,
                                                 i_conv_kernel_config->vector_reg_count - l_reg_block + l_n );
      } else if ( (i_conv_desc->datatype_in == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_out == LIBXSMM_DNN_DATATYPE_I32) ||
                  (i_conv_desc->datatype_in == LIBXSMM_DNN_DATATYPE_I8  && i_conv_desc->datatype_out == LIBXSMM_DNN_DATATYPE_I16
                     && (i_conv_desc->option & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0)  ) {
#if 0
        l_input_reg = i_gp_reg_mapping->gp_reg_input;
        l_input_idx = LIBXSMM_X86_GP_REG_UNDEF;
        l_scale = 0;
        l_disp = (l_k * i_conv_kernel_config->datatype_size_in * i_conv_desc->fm_lp_block)
                   + (l_n * i_conv_kernel_config->datatype_size_in * i_conv_desc->stride_w * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block);
#endif
        /* broadcast in pairs of 8/16 bit values */
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_conv_kernel_config->instruction_set,
                                          i_conv_kernel_config->vbcst_instruction,
                                          l_input_reg,
                                          l_input_idx, l_scale,
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
                                                 i_conv_kernel_config->vector_reg_count - l_reg_block + l_n,
                                                 i_conv_kernel_config->vector_reg_count - l_reg_block + l_n  );
      } else if ( (i_conv_desc->datatype_in == LIBXSMM_DNN_DATATYPE_I8  && i_conv_desc->datatype_out == LIBXSMM_DNN_DATATYPE_I32
                     && (i_conv_desc->option & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0)                                     ) {
        /* broadcast in quadruples of 8 bit values */
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_conv_kernel_config->instruction_set,
                                          i_conv_kernel_config->vbcst_instruction,
                                          l_input_reg,
                                          l_input_idx, l_scale,
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

        /* 32bit integer accumulation without saturation into running result buffer */
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_conv_kernel_config->instruction_set,
                                                 i_conv_kernel_config->vadd_instruction,
                                                 i_conv_kernel_config->vector_name,
                                                 5,
                                                 i_conv_kernel_config->vector_reg_count - l_reg_block + l_n,
                                                 i_conv_kernel_config->vector_reg_count - l_reg_block + l_n  );
      } else {
        /* shouldn't happen as error was thrown above */
      }

      /* handle prefetches for input and weights */
      if ( (l_n == 2) && (l_k < i_conv_desc->ifw_padded) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) ) {
        libxsmm_x86_instruction_prefetch( io_generated_code,
                                          LIBXSMM_X86_INSTR_PREFETCHT0 /*i_conv_kernel_config->prefetch_instruction*/,
                                          i_gp_reg_mapping->gp_reg_input_pf,
                                          LIBXSMM_X86_GP_REG_UNDEF,
                                          0,
                                          l_k * i_conv_kernel_config->datatype_size_in * i_conv_desc->stride_w
                                            * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block );
      }
      if ( (l_n == 4) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) ) {
        int l_filter_weight_pos = l_k + ((l_k/i_conv_desc->ifm_block*i_conv_desc->fm_lp_block)
                                    *(i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block)* i_conv_desc->fm_lp_block);
        libxsmm_x86_instruction_prefetch( io_generated_code,
                                          i_conv_kernel_config->prefetch_instruction,
                                          i_gp_reg_mapping->gp_reg_weight_pf,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          l_filter_weight_pos * i_conv_kernel_config->l_ld_ofm_fil * i_conv_desc->fm_lp_block* i_conv_kernel_config->datatype_size_wt );
      }
    }
    l_displacement_k++;
  }

  /* we have to make sure that we are reseting the pointer to its original value in case a full unroll */
  if ( l_k_updates > 0 ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_conv_kernel_config->alu_sub_instruction,
                                     i_gp_reg_mapping->gp_reg_input, 128 * l_k_updates * i_conv_kernel_config->datatype_size_in * i_conv_desc->fm_lp_block );
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_forward_avx512_ifmloop_sfma_two_rows( libxsmm_generated_code*                           io_generated_code,
                                                                         const libxsmm_convolution_forward_gp_reg_mapping* i_gp_reg_mapping,
                                                                         const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                                         const libxsmm_convolution_forward_descriptor*     i_conv_desc,
                                                                         const unsigned int                                i_kw_unroll ) {
  unsigned int l_n;
  unsigned int l_k;
  unsigned int l_input_reg;
  unsigned int l_input_idx;
  unsigned int l_scale;
  unsigned int l_disp;
  unsigned int l_displacement_k = 0;
  unsigned int l_k_updates = 0;
  unsigned int l_w;
  unsigned int l_m;
  unsigned int l_filter_pos = 0;
  unsigned int l_input_offset = 0;

  /* if kw loop is not unrolled, we are running out of GPRs */
  if ( i_kw_unroll == 0 ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_INVALID_KW_UNROLL );
    return;
  }

  /* apply k blocking */
  for ( l_k = 0; l_k < i_conv_desc->ifm_block*i_kw_unroll ; l_k++ ) {
    /* if we are not in LIBXSMM storage format, there are jumps */
    if ( (l_k > 0) && (l_k % i_conv_desc->ifm_block == 0) ) {
      /* input pointer advance */
      if ( (i_conv_desc->format & LIBXSMM_DNN_CONV_FORMAT_NHWC) > 0 ) {
        l_displacement_k += (i_conv_kernel_config->l_ld_ifm_act - i_conv_desc->ifm_block) * i_conv_desc->fm_lp_block;
      }
    }

    /* advance b pointer if needed */
    if ( l_displacement_k >= 128 ) {
      l_input_offset = 0;
      while ( l_displacement_k >= 128 ) {
        l_input_offset += 128;
        l_displacement_k -= 128;
        l_k_updates++;
      }

      libxsmm_generator_convolution_forward_avx512_advance_input_strides_two_rows( io_generated_code, i_gp_reg_mapping, i_conv_kernel_config,
                                                                                     i_conv_desc, l_input_offset );
    }

    if ( l_k == 0 ) {
      /* load weights */
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_conv_kernel_config->instruction_set,
                                        i_conv_kernel_config->vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_weight,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        l_filter_pos * i_conv_kernel_config->l_ld_ofm_fil * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block,
                                        i_conv_kernel_config->vector_name, 0,
                                        0, 0 );
      l_filter_pos++;

      if ( i_conv_desc->ifm_block*i_kw_unroll > 1 ) {
        for ( l_w = 1; l_w < 4; l_w++ ) {
          if ((l_k+l_w)%i_conv_desc->ifm_block == 0) {
            l_filter_pos += (i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block) * i_conv_desc->fm_lp_block;
          }
          /* second weight loaded in first iteration, in case of large blockings -> hiding L1 latencies */
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_conv_kernel_config->instruction_set,
                                            i_conv_kernel_config->vmove_instruction,
                                            i_gp_reg_mapping->gp_reg_weight,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            l_filter_pos * i_conv_kernel_config->l_ld_ofm_fil * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block,
                                            i_conv_kernel_config->vector_name, l_w,
                                            0, 0 );
          l_filter_pos++;
        }
      }
    } else if ( l_k < ((i_conv_desc->ifm_block*i_kw_unroll) - 3) ) {
      if ((l_k+3)%i_conv_desc->ifm_block == 0) {
        l_filter_pos += (i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block) * i_conv_desc->fm_lp_block;
      }
      /* pipelined load of weight, one k iteration ahead */
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_conv_kernel_config->instruction_set,
                                        i_conv_kernel_config->vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_weight,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        l_filter_pos * i_conv_kernel_config->l_ld_ofm_fil * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block,
                                        i_conv_kernel_config->vector_name, (l_k+3)%4,
                                        0, 0 );
      l_filter_pos++;
    }

    /* compute vectorwidth (A) * column broadcast (B) */
    for ( l_m = 0; l_m < i_conv_desc->ofh_rb; l_m++) {
      for ( l_n = 0; l_n < i_conv_desc->ofw_rb; l_n++) {
        /* determining base, idx and scale values */
        libxsmm_generator_convolution_forward_avx512_calc_sib_input_strides_two_rows( i_gp_reg_mapping, l_m, l_n, &l_input_reg, &l_input_idx, &l_scale );
        /* set displacement */
        l_disp = l_displacement_k * i_conv_kernel_config->datatype_size_in * i_conv_desc->fm_lp_block;
#if 0
        /* This replacement code only works for LIBXSMM format */
        l_disp = (l_k*i_conv_kernel_config->datatype_size)+(l_n*i_conv_kernel_config->datatype_size*i_conv_desc->stride_w*i_conv_desc->ifm_block)
                   + (l_m*i_conv_desc->ifw_padded*i_conv_desc->ifm_block*i_conv_kernel_config->datatype_size);
#endif
        /* depending on datatype emit the needed FMA(-sequence) */
        if ( i_conv_desc->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && i_conv_desc->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
          libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                                   i_conv_kernel_config->instruction_set,
                                                   i_conv_kernel_config->vfma_instruction,
                                                   1,
                                                   l_input_reg,
                                                   l_input_idx,
                                                   l_scale,
                                                   l_disp,
                                                   i_conv_kernel_config->vector_name,
                                                   l_k%4,
                                                   i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*i_conv_desc->ofh_rb) + l_n + (l_m*i_conv_desc->ofw_rb) );
        } else if ( (i_conv_desc->datatype_in == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_out == LIBXSMM_DNN_DATATYPE_I32) ||
                    (i_conv_desc->datatype_in == LIBXSMM_DNN_DATATYPE_I8  && i_conv_desc->datatype_out == LIBXSMM_DNN_DATATYPE_I16
                       && (i_conv_desc->option & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0)  ) {
#if 0
          l_input_reg = i_gp_reg_mapping->gp_reg_input;
          l_input_idx = LIBXSMM_X86_GP_REG_UNDEF;
          l_scale = 0;
          l_disp = (l_k*i_conv_kernel_config->datatype_size_in*i_conv_desc->fm_lp_block)
                     + (l_n * i_conv_kernel_config->datatype_size_in * i_conv_desc->stride_w * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block)
                     + (l_m * i_conv_desc->ifw_padded * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block * i_conv_kernel_config->datatype_size_in);
#endif
          /* broadcast in pairs of 8/16 bit values */
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_conv_kernel_config->instruction_set,
                                            i_conv_kernel_config->vbcst_instruction,
                                            l_input_reg,
                                            l_input_idx, l_scale,
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
                                                   i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*i_conv_desc->ofh_rb) + l_n + (l_m*i_conv_desc->ofw_rb),
                                                   i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*i_conv_desc->ofh_rb) + l_n + (l_m*i_conv_desc->ofw_rb)  );
        } else if ( (i_conv_desc->datatype_in == LIBXSMM_DNN_DATATYPE_I8  && i_conv_desc->datatype_out == LIBXSMM_DNN_DATATYPE_I32
                       && (i_conv_desc->option & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0)  ) {
          /* broadcast in quadruples of 8 bit values */
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_conv_kernel_config->instruction_set,
                                            i_conv_kernel_config->vbcst_instruction,
                                            l_input_reg,
                                            l_input_idx, l_scale,
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

          /* 32bit integer accumulation without saturation into running result buffer */
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                   i_conv_kernel_config->instruction_set,
                                                   i_conv_kernel_config->vadd_instruction,
                                                   i_conv_kernel_config->vector_name,
                                                   5,
                                                   i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*i_conv_desc->ofh_rb) + l_n + (l_m*i_conv_desc->ofw_rb),
                                                   i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*i_conv_desc->ofh_rb) + l_n + (l_m*i_conv_desc->ofw_rb)  );
        } else {
          /* shouldn't happen as error was thrown above */
        }

        /* handle prefetches for input and weights */
        if ( (l_n == 3) && (l_k < i_conv_desc->ifw_padded) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            LIBXSMM_X86_INSTR_PREFETCHT0 /*i_conv_kernel_config->prefetch_instruction*/,
                                            i_gp_reg_mapping->gp_reg_input_pf,
                                            LIBXSMM_X86_GP_REG_UNDEF,
                                            0,
                                            (l_m * i_conv_kernel_config->datatype_size_in * i_conv_kernel_config->l_ld_ifm_act
                                               * i_conv_desc->fm_lp_block * i_conv_desc->ifw_padded)
                                              + (l_k * i_conv_kernel_config->datatype_size_in * i_conv_desc->stride_w
                                                  * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block) );
        }
        if ( (l_m == 0) && (l_n == 8) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) ) {
          int l_filter_weight_pos = l_k + ((l_k/(i_conv_desc->ifm_block* i_conv_desc->fm_lp_block))
                                      *(i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block)* i_conv_desc->fm_lp_block);
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            i_conv_kernel_config->prefetch_instruction,
                                            i_gp_reg_mapping->gp_reg_weight_pf,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            l_filter_weight_pos * i_conv_kernel_config->l_ld_ofm_fil * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block );
        }
      }
    }
    l_displacement_k++;
  }

  /* we have to make sure that we are reseting the pointer to its original value in case a full unroll */
  if ( l_k_updates > 0 ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_conv_kernel_config->alu_sub_instruction,
                                     i_gp_reg_mapping->gp_reg_input, 128 * l_k_updates * i_conv_kernel_config->datatype_size_in * i_conv_desc->fm_lp_block );
  }
}

