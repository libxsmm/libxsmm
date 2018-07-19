/******************************************************************************
** Copyright (c) 2015-2018, Intel Corporation                                **
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
/* Rajkishore Barik, Ankush Mandal (Intel Corp.)
******************************************************************************/

#include "generator_convolution_weight_update_avx512.h"
#include "generator_convolution_common.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include <libxsmm_intrinsics_x86.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

LIBXSMM_API_INTERN
void libxsmm_generator_convolution_weight_update_avx512_kernel( libxsmm_generated_code*     io_generated_code,
                                                          const libxsmm_convolution_weight_update_descriptor* i_conv_desc,
                                                          const char*                       i_arch ) {
  libxsmm_convolution_kernel_config l_conv_kernel_config;
  libxsmm_convolution_weight_update_gp_reg_mapping l_gp_reg_mapping;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  unsigned int l_ofh_trips=0, l_ofw_trips=0;
  unsigned int oj=0, oi=0;
  unsigned int is_last_call;
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
  l_gp_reg_mapping.gp_reg_ifmInner_loop = LIBXSMM_X86_GP_REG_UNDEF;
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
  } else if ( strcmp( i_arch, "skx" ) == 0 ) {
    l_conv_kernel_config.instruction_set = LIBXSMM_X86_AVX512_CORE;
  } else if ( strcmp( i_arch, "icl" ) == 0 ) {
    l_conv_kernel_config.instruction_set = LIBXSMM_X86_AVX512_ICL;
  } else if ( strcmp( i_arch, "knm" ) == 0 ) {
    l_conv_kernel_config.instruction_set = LIBXSMM_X86_AVX512_KNM;
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
    return;
  }
  l_conv_kernel_config.vector_reg_count = 32;
  l_conv_kernel_config.vector_length_in = 16;
  if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_F32) {
    l_conv_kernel_config.datatype_size_in = 4;
  } else if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16) {
    l_conv_kernel_config.datatype_size_in = 2;
  }  else if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_BF16) {
    l_conv_kernel_config.datatype_size_in = 2;
  } else if (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I8) {
    l_conv_kernel_config.datatype_size_in = 1;
  }

  l_conv_kernel_config.vector_length_out = 16;
  if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_F32) {
    l_conv_kernel_config.datatype_size_out = 4;
  } else if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16) {
    l_conv_kernel_config.datatype_size_out = 2;
  } else if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_BF16) {
    l_conv_kernel_config.datatype_size_out = 2;
  } else if (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I8) {
    l_conv_kernel_config.datatype_size_out = 1;
  }
  l_conv_kernel_config.vector_length_wt = 16;
  l_conv_kernel_config.datatype_size_wt = 4;
  l_conv_kernel_config.vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
  l_conv_kernel_config.vfma_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
  l_conv_kernel_config.vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
  l_conv_kernel_config.vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
  l_conv_kernel_config.prefetch_instruction = LIBXSMM_X86_INSTR_PREFETCHT0;
  l_conv_kernel_config.alu_add_instruction = LIBXSMM_X86_INSTR_ADDQ;
  l_conv_kernel_config.alu_sub_instruction = LIBXSMM_X86_INSTR_SUBQ;
  l_conv_kernel_config.alu_cmp_instruction = LIBXSMM_X86_INSTR_CMPQ;
  l_conv_kernel_config.alu_jmp_instruction = LIBXSMM_X86_INSTR_JL;
  l_conv_kernel_config.alu_mov_instruction = LIBXSMM_X86_INSTR_MOVQ;
  l_conv_kernel_config.vector_name = 'z';

  if (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_BF16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_BF16) {
    l_conv_kernel_config.vfma_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
    l_conv_kernel_config.vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
    l_conv_kernel_config.vbcst_instruction = LIBXSMM_X86_INSTR_VPBROADCASTD;
  }

  if (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I8 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32) {
    l_conv_kernel_config.vfma_instruction = LIBXSMM_X86_INSTR_VPMADDUBSW;
    l_conv_kernel_config.vadd_instruction = LIBXSMM_X86_INSTR_VPADDD;
    l_conv_kernel_config.vbcst_instruction = LIBXSMM_X86_INSTR_VPBROADCASTD;
  }

  /* calculate leading dimension depending on format */
  if ( (i_conv_desc->format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0 ) {
    l_conv_kernel_config.l_ld_ifm_act = i_conv_desc->ifm_block_hp;
    l_conv_kernel_config.l_ld_ofm_act = i_conv_desc->ofm_block;
  }
  if ( (i_conv_desc->format & LIBXSMM_DNN_TENSOR_FORMAT_NHWC) > 0 ) {
    l_conv_kernel_config.l_ld_ifm_act = i_conv_desc->ifm_block * i_conv_desc->blocks_ifm;
    l_conv_kernel_config.l_ld_ofm_act = i_conv_desc->ofm_block * i_conv_desc->blocks_ofm;
  }

  /* check if we have full vectors */
  /*if (i_conv_desc->ofm_block % l_conv_kernel_config.vector_length_out != 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CONV_OFM_VEC );
    return;
  }*/

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* open asm */
  libxsmm_x86_instruction_open_stream_convolution( io_generated_code, l_gp_reg_mapping.gp_reg_input,
      l_gp_reg_mapping.gp_reg_weight, l_gp_reg_mapping.gp_reg_output,
      l_gp_reg_mapping.gp_reg_input_pf, l_gp_reg_mapping.gp_reg_weight_pf,
      l_gp_reg_mapping.gp_reg_output_pf, i_arch );

  /***** Code to generate; JIT this code below
    for (int ifm2 = 0; ifm2 < VLEN; ifm2+=WU_UNROLL_FACTOR_1) {
    __m512 acc00 = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&del_wt[ofm1][ifm1][kj][ki][ifm2]);
    __m512 acc01 = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&del_wt[ofm1][ifm1][kj][ki][ifm2 + 1]);
    LIBXSMM_PRAGMA_UNROLL
    for (int ij=0, oj=0; oj < ofh; ij+=stride_h, oj++) {
    LIBXSMM_PRAGMA_UNROLL
    for (int ii=0, oi=0; oi < ofw; ii+=WU_UNROLL_FACTOR_2*stride_w, oi+=WU_UNROLL_FACTOR_2) {
    __m512 out00 = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&del_output[img][ofm1][oj][oi + 0]);
    acc00 = _mm512_fmadd_ps(_mm512_set1_ps(input[img][ifm1][ij+kj][ii+ki][ifm2]), out00, acc00);
    acc01 = _mm512_fmadd_ps(_mm512_set1_ps(input[img][ifm1][ij+kj][ii+ki][ifm2+1]), out00, acc01);
    }
    }
    _mm512_store_ps( &del_wt[ofm1][ifm1][kj][ki][ifm2], acc00 );
    _mm512_store_ps( &del_wt[ofm1][ifm1][kj][ki][ifm2 + 1], acc01 );
    }
   ******/

#define UNROLL_REGISTER_BLOCK

#ifdef UNROLL_REGISTER_BLOCK
  /* initialize OFW and OFH unrolling */
  if (i_conv_desc->kw == 1 && i_conv_desc->kh == 1) {
    if (i_conv_desc->ofh_unroll != 0) {
      l_ofh_trips = i_conv_desc->ofh / i_conv_desc->ofh_rb;
    } else l_ofh_trips = 1;

    if (i_conv_desc->ofw_unroll != 0) {
      l_ofw_trips = i_conv_desc->ofw / i_conv_desc->ofw_rb;
    } else l_ofw_trips = 1;
  } else {
    l_ofh_trips = 1;
    l_ofw_trips = 1;
  }

  /* Load scratch lines for transpose */
  if (((i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32) || i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_BF16) && (l_conv_kernel_config.instruction_set == LIBXSMM_X86_AVX512_ICL || l_conv_kernel_config.instruction_set == LIBXSMM_X86_AVX512_CORE) ) {
    unsigned int rsp_offset = 56;

    libxsmm_x86_instruction_alu_reg( io_generated_code, l_conv_kernel_config.alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, l_gp_reg_mapping.gp_reg_help_5);
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        l_conv_kernel_config.alu_mov_instruction,
        l_gp_reg_mapping.gp_reg_help_5,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        rsp_offset,
        l_gp_reg_mapping.gp_reg_help_4,
        0 );

      libxsmm_x86_instruction_alu_reg( io_generated_code, l_conv_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_help_4, l_gp_reg_mapping.gp_reg_help_6);
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_help_6, 64 );

    if (i_conv_desc->avoid_output_trans) {
      /* Initialize "permute mask" in zmm3 */
      unsigned short  mask_array[32] = {0,16,  1,17,  2,18,  3,19,  4,20,  5,21,  6,22,  7,23,  8,24,  9,25,  10,26,  11,27,  12,28,  13,29,  14,30,  15,31 };

     libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
        (const unsigned char*) mask_array,
        "abs_mask",
        l_conv_kernel_config.vector_name,
        4);
    }
  }

  if (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I8 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_help_0, 65537 );
    libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_reg_mapping.gp_reg_help_0 );
    libxsmm_x86_instruction_vec_move( io_generated_code,
                                      l_conv_kernel_config.instruction_set,
                                      l_conv_kernel_config.vbcst_instruction,
                                      LIBXSMM_X86_GP_REG_RSP ,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                      l_conv_kernel_config.vector_name, 3, 0, 1, 0 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_reg_mapping.gp_reg_help_0 );
  }


  if (i_conv_desc->use_fastpath) {
    /* New clean kernel for convolution...  */
    libxsmm_generator_convolution_weight_update_load_weight( io_generated_code, &l_gp_reg_mapping, &l_conv_kernel_config, i_conv_desc );

    /* We bring image loop inside to exploit NTS  */
    if (i_conv_desc->blocks_img > 1) {
      libxsmm_generator_convolution_header_image_block_loop( io_generated_code, &l_loop_label_tracker,
          &l_conv_kernel_config, l_gp_reg_mapping.gp_reg_help_3 );
      libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_reg_mapping.gp_reg_input );
      libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_reg_mapping.gp_reg_output );
    }


    /* Assembly loop start for ofh_blocks  */
    if (i_conv_desc->blocks_h > 1) {
      libxsmm_generator_convolution_header_h_block_loop( io_generated_code, &l_loop_label_tracker,
          &l_conv_kernel_config, l_gp_reg_mapping.gp_reg_help_0 );
      libxsmm_x86_instruction_alu_reg( io_generated_code, l_conv_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_input, l_gp_reg_mapping.gp_reg_help_1);
      libxsmm_x86_instruction_alu_reg( io_generated_code, l_conv_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_output, l_gp_reg_mapping.gp_reg_help_2);
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_add_instruction,
          l_gp_reg_mapping.gp_reg_help_1, i_conv_desc->ofh_rb * i_conv_desc->stride_h * i_conv_desc->ifw_padded * l_conv_kernel_config.l_ld_ifm_act * l_conv_kernel_config.datatype_size_in  );
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_add_instruction,
          l_gp_reg_mapping.gp_reg_help_2, i_conv_desc->ofw_padded * i_conv_desc->ofh_rb * l_conv_kernel_config.l_ld_ofm_act * l_conv_kernel_config.datatype_size_out  );
      is_last_call = 0;
    } else {
      libxsmm_x86_instruction_alu_reg( io_generated_code, l_conv_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_input_pf, l_gp_reg_mapping.gp_reg_help_1);
      libxsmm_x86_instruction_alu_reg( io_generated_code, l_conv_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_output_pf, l_gp_reg_mapping.gp_reg_help_2);
      is_last_call = 1;
    }

    if (i_conv_desc->transpose_ofw_ifm == 1) {
      libxsmm_generator_convolution_weight_update_transpose_avx512_ofwloop_all_pixels_inside(  io_generated_code,
          &l_gp_reg_mapping,
         &l_conv_kernel_config,
         i_conv_desc,
         is_last_call);
    } else {
      libxsmm_generator_convolution_weight_update_avx512_ofwloop_all_pixels_inside(  io_generated_code,
         &l_gp_reg_mapping,
         &l_conv_kernel_config,
         i_conv_desc,
         is_last_call);
    }

    /* Assembly loop end for ofh_blocks */
    if ( i_conv_desc->blocks_h >  1 ) {
      /* Advance input register */
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_add_instruction,
          l_gp_reg_mapping.gp_reg_input, i_conv_desc->ofh_rb * i_conv_desc->stride_h * i_conv_desc->ifw_padded * l_conv_kernel_config.l_ld_ifm_act * l_conv_kernel_config.datatype_size_in  );
      /* Advance output register */
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_add_instruction,
          l_gp_reg_mapping.gp_reg_output, i_conv_desc->ofw_padded * i_conv_desc->ofh_rb * l_conv_kernel_config.l_ld_ofm_act * l_conv_kernel_config.datatype_size_out  );
      libxsmm_generator_convolution_footer_h_block_loop( io_generated_code, &l_loop_label_tracker,
          &l_conv_kernel_config, l_gp_reg_mapping.gp_reg_help_0, i_conv_desc->blocks_h - 1 );
    }

    /* Last peeled iteration for H block...   */
    if ( i_conv_desc->blocks_h >  1 ) {
      libxsmm_x86_instruction_alu_reg( io_generated_code, l_conv_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_input_pf, l_gp_reg_mapping.gp_reg_help_1);
      libxsmm_x86_instruction_alu_reg( io_generated_code, l_conv_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_output_pf, l_gp_reg_mapping.gp_reg_help_2);
      is_last_call = 1;
      if (i_conv_desc->transpose_ofw_ifm == 1) {
        libxsmm_generator_convolution_weight_update_transpose_avx512_ofwloop_all_pixels_inside(  io_generated_code,
            &l_gp_reg_mapping,
            &l_conv_kernel_config,
            i_conv_desc,
            is_last_call);
      } else {
        libxsmm_generator_convolution_weight_update_avx512_ofwloop_all_pixels_inside(  io_generated_code,
            &l_gp_reg_mapping,
            &l_conv_kernel_config,
            i_conv_desc,
            is_last_call);
      }
    }

    if (i_conv_desc->blocks_img > 1) {
      libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_reg_mapping.gp_reg_output );
      libxsmm_x86_instruction_pop_reg( io_generated_code, l_gp_reg_mapping.gp_reg_input );
      /* Advance input register */
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_add_instruction,
          l_gp_reg_mapping.gp_reg_input, i_conv_desc->blocks_ifm  *  i_conv_desc->ifh_padded *  i_conv_desc->ifw_padded * l_conv_kernel_config.l_ld_ifm_act * l_conv_kernel_config.datatype_size_in  );
      /* Advance output register */
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_add_instruction,
          l_gp_reg_mapping.gp_reg_output, i_conv_desc->blocks_ofm  *  i_conv_desc->ofh_padded  * i_conv_desc->ofw_padded * l_conv_kernel_config.l_ld_ofm_act * l_conv_kernel_config.datatype_size_out  );
      libxsmm_generator_convolution_footer_image_block_loop( io_generated_code, &l_loop_label_tracker,
          &l_conv_kernel_config, l_gp_reg_mapping.gp_reg_help_3, i_conv_desc->blocks_img);
    }

    libxsmm_generator_convolution_weight_update_store_weight( io_generated_code, &l_gp_reg_mapping, &l_conv_kernel_config, i_conv_desc );
  } else if (i_conv_desc->transpose_ofw_ifm == 1) {

    l_ofh_trips = 1;
    l_ofw_trips = 1;
    libxsmm_generator_convolution_weight_update_load_weight( io_generated_code, &l_gp_reg_mapping, &l_conv_kernel_config, i_conv_desc );
    libxsmm_generator_convolution_weight_update_transpose_avx512_ofwloop(  io_generated_code,
        &l_gp_reg_mapping,
        &l_conv_kernel_config,
        i_conv_desc,
        i_conv_desc->ofh_rb, 0, 0/*false*/);
    libxsmm_generator_convolution_weight_update_store_weight( io_generated_code, &l_gp_reg_mapping, &l_conv_kernel_config, i_conv_desc );
  } else { /* no transpose of ofw and ifm */

    libxsmm_generator_convolution_weight_update_load_weight( io_generated_code, &l_gp_reg_mapping, &l_conv_kernel_config, i_conv_desc );
#if 0
    if (((i_conv_desc->kh == 1) && (i_conv_desc->kw == 1)) && (((i_conv_desc->ofw_unroll == 0) && (i_conv_desc->ofw / i_conv_desc->ofw_rb) > 1) || (l_ofw_trips > 1) )) {
      /* header of oi loop */
      libxsmm_generator_convolution_header_ofw_loop(  io_generated_code, &l_loop_label_tracker,
          &l_conv_kernel_config, l_gp_reg_mapping.gp_reg_oi_loop );
    }
#endif

    /* unroll ofw */
    for ( oi = 0; oi < l_ofw_trips; oi++) {

#if 0
      if (((i_conv_desc->kh == 1) && (i_conv_desc->kw == 1)) && (((i_conv_desc->ofh_unroll == 0) && (i_conv_desc->ofh / i_conv_desc->ofh_rb) > 1)  || (l_ofh_trips > 1)))  {
        /* open KW loop, ki */
        libxsmm_generator_convolution_header_ofh_loop(  io_generated_code, &l_loop_label_tracker,
            &l_conv_kernel_config, l_gp_reg_mapping.gp_reg_oj_loop );

      }
#endif
      for ( oj = 0; oj < l_ofh_trips; oj++) {

        libxsmm_generator_convolution_weight_update_avx512_ofwloop(  io_generated_code,
            &l_gp_reg_mapping,
            &l_conv_kernel_config,
            i_conv_desc,
            i_conv_desc->ofh_rb, oj, 0/*false*/);
      }

#if 0
      if (((i_conv_desc->kh == 1) && (i_conv_desc->kw == 1)) && (((i_conv_desc->ofh_unroll == 0) && (i_conv_desc->ofh / i_conv_desc->ofh_rb) > 1)  || (l_ofh_trips > 1)))  {
        libxsmm_x86_instruction_alu_imm( io_generated_code,
            l_conv_kernel_config.alu_add_instruction,
            l_gp_reg_mapping.gp_reg_output, i_conv_desc->ofh_rb * i_conv_desc->ofw_padded*l_conv_kernel_config.vector_length * l_conv_kernel_config.datatype_size  );
        if ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2) { /* bring data to L1 for the next iteration */
          libxsmm_x86_instruction_alu_imm( io_generated_code,
              l_conv_kernel_config.alu_add_instruction,
              l_gp_reg_mapping.gp_reg_output_pf, i_conv_desc->ofh_rb * i_conv_desc->ofw_padded*l_conv_kernel_config.vector_length * l_conv_kernel_config.datatype_size  );
        }
#if 0 /* We do not need this as this is contiguous on oj space */
        libxsmm_x86_instruction_alu_imm( io_generated_code,
            l_conv_kernel_config.alu_add_instruction,
            l_gp_reg_mapping.gp_reg_input, i_conv_desc->ofh_rb * i_conv_desc->ifw_padded*l_conv_kernel_config.vector_length * l_conv_kernel_config.datatype_size  );
#endif
        /* close KW loop, ki */
        libxsmm_generator_convolution_footer_ofh_loop(  io_generated_code, &l_loop_label_tracker,
            &l_conv_kernel_config, l_gp_reg_mapping.gp_reg_oj_loop, i_conv_desc->ofh/ i_conv_desc->ofh_rb /*l_ofh_trips*/ );
      }

      /* FIXME, if you add prefetch, then remove the if below */
      if (((i_conv_desc->kh == 1) && (i_conv_desc->kw == 1)) && ((((i_conv_desc->ofw_unroll == 1) && (oi < l_ofw_trips-1))) || ((i_conv_desc->ofw_unroll == 0)))) {
#if 1
        libxsmm_x86_instruction_alu_imm( io_generated_code,
            l_conv_kernel_config.alu_sub_instruction,
            l_gp_reg_mapping.gp_reg_output, (i_conv_desc->ofh/*/i_conv_desc->ofh_rb*/) * i_conv_desc->ofw_padded *l_conv_kernel_config.vector_length * l_conv_kernel_config.datatype_size  );
        libxsmm_x86_instruction_alu_imm( io_generated_code,
            l_conv_kernel_config.alu_sub_instruction,
            l_gp_reg_mapping.gp_reg_input, (i_conv_desc->ofh/*/i_conv_desc->ofh_rb*/) * i_conv_desc->ifw_padded* /*l_conv_kernel_config.vector_length*/ i_conv_desc->stride_h * i_conv_desc->ifm_block * l_conv_kernel_config.datatype_size  );


        if ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2) { /* bring data to L1 for the next iteration */
          libxsmm_x86_instruction_alu_imm( io_generated_code,
              l_conv_kernel_config.alu_sub_instruction,
              l_gp_reg_mapping.gp_reg_output_pf, (i_conv_desc->ofh/*/i_conv_desc->ofh_rb*/) * i_conv_desc->ofw_padded *l_conv_kernel_config.vector_length * l_conv_kernel_config.datatype_size  );
        }
        if ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L2) {
          libxsmm_x86_instruction_alu_imm( io_generated_code,
              l_conv_kernel_config.alu_sub_instruction,
              l_gp_reg_mapping.gp_reg_input_pf, (i_conv_desc->ofh/*/i_conv_desc->ofh_rb*/) * i_conv_desc->ifw_padded* /*l_conv_kernel_config.vector_length*/ i_conv_desc->stride_h * i_conv_desc->ifm_block * l_conv_kernel_config.datatype_size  );
        }

        /* TODO */
        /*if (l_ofw_trips > 1) {*/
        /* Add 40 to input, output */
        libxsmm_x86_instruction_alu_imm( io_generated_code,
            l_conv_kernel_config.alu_add_instruction,
            l_gp_reg_mapping.gp_reg_output, i_conv_desc->ofw_rb * l_conv_kernel_config.vector_length *l_conv_kernel_config.datatype_size  );

        libxsmm_x86_instruction_alu_imm( io_generated_code,
            l_conv_kernel_config.alu_add_instruction,
            /*l_gp_reg_mapping.gp_reg_input, i_conv_desc->ofw_rb * i_conv_desc->ifm_block * i_conv_desc->stride_w * l_conv_kernel_config.datatype_size  );*/
          l_gp_reg_mapping.gp_reg_input, i_conv_desc->ofw_rb * i_conv_desc->ifm_block * i_conv_desc->stride_w * /*l_conv_kernel_config.vector_length * */ l_conv_kernel_config.datatype_size  );

        if ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2) { /* bring data to L1 for the next iteration */
          libxsmm_x86_instruction_alu_imm( io_generated_code,
              l_conv_kernel_config.alu_add_instruction,
              l_gp_reg_mapping.gp_reg_output_pf, i_conv_desc->ofw_rb * l_conv_kernel_config.vector_length *l_conv_kernel_config.datatype_size  );
        }
        if ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L2) {
          libxsmm_x86_instruction_alu_imm( io_generated_code,
              l_conv_kernel_config.alu_add_instruction,
              /*l_gp_reg_mapping.gp_reg_input, i_conv_desc->ofw_rb * i_conv_desc->ifm_block * i_conv_desc->stride_w * l_conv_kernel_config.datatype_size  );*/
            l_gp_reg_mapping.gp_reg_input_pf, i_conv_desc->ofw_rb * i_conv_desc->ifm_block * i_conv_desc->stride_w * /*l_conv_kernel_config.vector_length * */ l_conv_kernel_config.datatype_size  );
        }
        /*}*/
#endif
      }  /* remove this if, if you wanna add prefetch across oi iterations, does not happen in overfeat */
#endif
      /* FIXME: add prefetch for next iteration of oi (does not happen in overfeat) -- leaving for future */
    }

#if 0
    if (((i_conv_desc->kh == 1) && (i_conv_desc->kw == 1)) && (((i_conv_desc->ofw_unroll == 0) && (i_conv_desc->ofw / i_conv_desc->ofw_rb) > 1) || (l_ofw_trips > 1)) ) {
      /* close oi loop with blocking */
      libxsmm_generator_convolution_footer_ofw_loop(  io_generated_code, &l_loop_label_tracker,
          &l_conv_kernel_config, l_gp_reg_mapping.gp_reg_oi_loop, i_conv_desc->ofw / i_conv_desc->ofw_rb /*l_ofw_trips*/ );
    }
#endif

    libxsmm_generator_convolution_weight_update_store_weight( io_generated_code, &l_gp_reg_mapping, &l_conv_kernel_config, i_conv_desc );
  }

#endif
  /* close asm */
  libxsmm_x86_instruction_close_stream_convolution( io_generated_code, i_arch );
}

  LIBXSMM_API_INTERN
void libxsmm_generator_convolution_weight_update_avx512_ofwloop( libxsmm_generated_code* io_generated_code,
    const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
    const libxsmm_convolution_kernel_config*                i_conv_kernel_config,
    const libxsmm_convolution_weight_update_descriptor*     i_conv_desc,
    const unsigned int                                      i_ofh_unroll,
    const unsigned int                                      ofh_trip_counter,
    const int                                               no_unroll_no_block)
{
  /* setup input strides */
  libxsmm_generator_convolution_weight_update_avx512_init_weight_strides( io_generated_code, i_gp_reg_mapping, i_conv_kernel_config, i_conv_desc );

  libxsmm_generator_convolution_weight_update_avx512_ofwloop_sfma( io_generated_code, i_gp_reg_mapping,
      i_conv_kernel_config, i_conv_desc, i_ofh_unroll, ofh_trip_counter, no_unroll_no_block);
}

LIBXSMM_API_INTERN
void libxsmm_generator_convolution_weight_update_avx512_init_weight_strides( libxsmm_generated_code* io_generated_code,
    const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
    const libxsmm_convolution_kernel_config*                i_conv_kernel_config,
    const libxsmm_convolution_weight_update_descriptor*     i_conv_desc ) {

  int unroll_factor = i_conv_desc->ifm_block;

  /* Initialize helper registers for SIB addressing */
  /* helper 0: Index register holding ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_help_0, i_conv_kernel_config->datatype_size_in );
  /* helper 1: Index register holding 3*ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_help_1, i_conv_kernel_config->datatype_size_in * 3 );
  /* helper 2: Index register holding 5*ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_help_2, i_conv_kernel_config->datatype_size_in  * 5 );
  /* helper 3: Index register holding 7*ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_help_3, i_conv_kernel_config->datatype_size_in * 7 );

  /* helper 4: B + 9*ldb, additional base address
     helper 5: B + 18*ldb, additional base address */
  if ( unroll_factor > 9 ) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_input, i_gp_reg_mapping->gp_reg_help_4);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_help_4,  9 * i_conv_kernel_config->datatype_size_in );
  }
  if ( unroll_factor > 18 ) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_input, i_gp_reg_mapping->gp_reg_help_5);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_help_5, 18 * i_conv_kernel_config->datatype_size_in );
  }
  if ( unroll_factor > 27 ) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_input, i_gp_reg_mapping->gp_reg_help_6);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_help_6, 27 *  i_conv_kernel_config->datatype_size_in );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_convolution_weight_update_avx512_ofwloop_sfma( libxsmm_generated_code* io_generated_code,
    const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
    const libxsmm_convolution_kernel_config*                i_conv_kernel_config,
    const libxsmm_convolution_weight_update_descriptor*     i_conv_desc,
    const unsigned int                                      i_ofh_unroll,
    const unsigned int                                      ofh_trip_counter,
    const int                                               no_unroll_no_block ) {
  unsigned int l_n;
  unsigned int l_k = 0, l_k_1 = 0, l_k_2 = 0;
  unsigned int l_input_reg;
  unsigned int l_input_idx;
  unsigned int l_scale;
  unsigned int l_disp;
  unsigned int l_displacement_k = 0;
  unsigned int l_k_updates = 0;
  unsigned int l_w;
  unsigned int num_look_ahead;
  unsigned int output_counter = 0;
  unsigned int unroll_factor = i_conv_desc->ifm_block;
  unsigned int eq_index = 0;

  LIBXSMM_UNUSED(ofh_trip_counter);

  /* apply k blocking */
  for ( l_k_1 = 0; l_k_1 < i_conv_desc->ofh_rb; l_k_1++ ) {

    l_k_updates = 0;
    for ( l_k_2 = 0; l_k_2 < i_conv_desc->ofw_rb; l_k_2++, l_k++ ) {
      /* advance b pointer if needed */
      if ( (l_k_2 > 0) &&(l_k_2%8 == 0) ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_input, 8*(i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->stride_w)* i_conv_kernel_config->datatype_size_in );
        if (  unroll_factor > 9 ) {
          libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_input, i_gp_reg_mapping->gp_reg_help_4);
          libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
              i_gp_reg_mapping->gp_reg_help_4,  9 * i_conv_kernel_config->datatype_size_in );
        }
        if ( unroll_factor > 18 ) {
          libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_input, i_gp_reg_mapping->gp_reg_help_5);
          libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
              i_gp_reg_mapping->gp_reg_help_5, 18 *  i_conv_kernel_config->datatype_size_in );
        }
        if ( unroll_factor > 27 ) {
          libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_input, i_gp_reg_mapping->gp_reg_help_6);
          libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
              i_gp_reg_mapping->gp_reg_help_6, 27 *  i_conv_kernel_config->datatype_size_in );
        }
        l_displacement_k = 0;
        l_k_updates++;
      }
      num_look_ahead = i_conv_desc->ofw_rb < 4 ? i_conv_desc->ofw_rb : 4;
      /*num_look_ahead = i_conv_desc->ofw_rb < 6 ? i_conv_desc->ofw_rb : 6;*/

      if (  l_k == 0 ) {
        /* load output */
        for (eq_index = 0; eq_index < 4; eq_index++) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_conv_kernel_config->instruction_set,
              i_conv_kernel_config->vmove_instruction,
              i_gp_reg_mapping->gp_reg_output,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (l_k)*(i_conv_kernel_config->l_ld_ofm_act)*(i_conv_kernel_config->datatype_size_out) +  eq_index * (i_conv_desc->ofw_padded * i_conv_desc->ofh_padded) * (i_conv_kernel_config->l_ld_ofm_act)*(i_conv_kernel_config->datatype_size_out),
              i_conv_kernel_config->vector_name, 0 + eq_index * 4,
              0, 1, 0 );
        }

        if ( (i_conv_desc->ofw_rb * i_ofh_unroll > 1) && (num_look_ahead > 1) ) {
          for ( l_w = 1; l_w < num_look_ahead; l_w++) {
            for (eq_index = 0; eq_index < 4; eq_index++) {
              libxsmm_x86_instruction_vec_move( io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  i_conv_kernel_config->vmove_instruction,
                  i_gp_reg_mapping->gp_reg_output,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  (l_k+l_w)*(i_conv_kernel_config->l_ld_ofm_act)*(i_conv_kernel_config->datatype_size_out) + eq_index * (i_conv_desc->ofw_padded * i_conv_desc->ofh_padded) * (i_conv_kernel_config->l_ld_ofm_act)*(i_conv_kernel_config->datatype_size_out),
                  i_conv_kernel_config->vector_name, l_w + eq_index * 4,
                  0, 1, 0 );
            }
          }
          output_counter += num_look_ahead -1;
        }
      } else if ((l_k < ((i_conv_desc->ofw_rb * i_ofh_unroll) - (num_look_ahead-1))) && (l_k_2 >= (i_conv_desc->ofw_rb - (num_look_ahead-1))) && (l_k_2 < i_conv_desc->ofw_rb) ) {
        for (eq_index = 0; eq_index < 4; eq_index++  ) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_conv_kernel_config->instruction_set,
              i_conv_kernel_config->vmove_instruction,
              i_gp_reg_mapping->gp_reg_output,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              ((output_counter)*(i_conv_kernel_config->l_ld_ofm_act)*(i_conv_kernel_config->datatype_size_out) )
              + ((l_k_1 + 1) * i_conv_desc->ofw_padded*i_conv_kernel_config->l_ld_ofm_act * i_conv_kernel_config->datatype_size_out  )
              +eq_index * (i_conv_desc->ofw_padded * i_conv_desc->ofh_padded) * (i_conv_kernel_config->l_ld_ofm_act)*(i_conv_kernel_config->datatype_size_out),
              i_conv_kernel_config->vector_name, (l_k+(num_look_ahead-1))%num_look_ahead + eq_index * 4,
              0, 1, 0 );
        }
      } else if ( l_k < ((i_conv_desc->ofw_rb*i_ofh_unroll) - (num_look_ahead - 1)) ) {
        for (eq_index = 0; eq_index < 4; eq_index++  ) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_conv_kernel_config->instruction_set,
              i_conv_kernel_config->vmove_instruction,
              i_gp_reg_mapping->gp_reg_output,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              ((output_counter)*(i_conv_kernel_config->l_ld_ofm_act)*(i_conv_kernel_config->datatype_size_out) )
              + ((l_k_1) * i_conv_desc->ofw_padded*i_conv_kernel_config->l_ld_ofm_act * i_conv_kernel_config->datatype_size_out  )
              + eq_index * (i_conv_desc->ofw_padded * i_conv_desc->ofh_padded) * (i_conv_kernel_config->l_ld_ofm_act)*(i_conv_kernel_config->datatype_size_out),
              i_conv_kernel_config->vector_name, (l_k+(num_look_ahead-1))%num_look_ahead + eq_index * 4,
              0, 1, 0 );
        }
      }

      if (output_counter >= (i_conv_desc->ofw_rb-1)) {
        output_counter = 0;
      } else output_counter++;

      /* compute vectorwidth (A) * column broadcast (B) */
      for ( l_n = 0; l_n < unroll_factor; l_n++) {
        /* determining base, idx and scale values */
        /* default values */
        l_input_reg = i_gp_reg_mapping->gp_reg_input;
        l_input_idx = LIBXSMM_X86_GP_REG_UNDEF;
        l_scale = 0;
        l_disp = l_displacement_k*(i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->stride_w)*(i_conv_kernel_config->datatype_size_in);

        /* select the base register */
        if ( l_n > 26 ) {
          l_input_reg = i_gp_reg_mapping->gp_reg_help_6;
        } else if ( l_n > 17 ) {
          l_input_reg = i_gp_reg_mapping->gp_reg_help_5;
        } else if ( l_n > 8 ) {
          l_input_reg = i_gp_reg_mapping->gp_reg_help_4;
        } else {
          l_input_reg = i_gp_reg_mapping->gp_reg_input;
        }
        if ( l_n % 9 == 0 ) {
          l_input_idx = LIBXSMM_X86_GP_REG_UNDEF;
          l_scale = 0;
        } else if ( l_n % 9 == 1 ) {
          l_input_idx = i_gp_reg_mapping->gp_reg_help_0;
          l_scale = 1;
        } else if ( l_n % 9 == 2 ) {
          l_input_idx = i_gp_reg_mapping->gp_reg_help_0;
          l_scale = 2;
        } else if ( l_n % 9 == 3 ) {
          l_input_idx = i_gp_reg_mapping->gp_reg_help_1;
          l_scale = 1;
        } else if ( l_n % 9 == 4 ) {
          l_input_idx = i_gp_reg_mapping->gp_reg_help_0;
          l_scale = 4;
        } else if ( l_n % 9 == 5 ) {
          l_input_idx = i_gp_reg_mapping->gp_reg_help_2;
          l_scale = 1;
        } else if ( l_n % 9 == 6 ) {
          l_input_idx = i_gp_reg_mapping->gp_reg_help_1;
          l_scale = 2;
        } else if ( l_n % 9 == 7 ) {
          l_input_idx = i_gp_reg_mapping->gp_reg_help_3;
          l_scale = 1;
        } else if ( l_n % 9 == 8 ) {
          l_input_idx = i_gp_reg_mapping->gp_reg_help_0;
          l_scale = 8;
        } else {
          assert(0/*should not happen*/);
        }
        for (eq_index = 0; eq_index < 4; eq_index++) {
          libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
              i_conv_kernel_config->instruction_set,
              i_conv_kernel_config->vfma_instruction,
              1,
              l_input_reg,
              l_input_idx,
              l_scale,
              l_disp,
              i_conv_kernel_config->vector_name,
              l_k%num_look_ahead + eq_index * 4,
              i_conv_kernel_config->vector_reg_count - unroll_factor + l_n - eq_index * 3);
        }

#if 0
        if ( (l_n == 0) &&  ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) && (input_pf_index < input_pf_bound) ) {
          input_pf_offset = (l_k_1 * i_conv_desc->stride_h) * (i_conv_desc->ofw_rb * i_conv_desc->stride_w) + l_k_2 *  (i_conv_desc->ofw_rb * i_conv_desc->stride_w) * (i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in);
          libxsmm_x86_instruction_prefetch( io_generated_code,
              prefetch_input_type,
              i_gp_reg_mapping->gp_reg_input_pf,
              LIBXSMM_X86_GP_REG_UNDEF,
              0,
              input_pf_offset);
          input_pf_index++;
        }
        if ( (l_n == 2) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2) && (output_pf_offset < output_pf_bound) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
              prefetch_output_type,
              i_gp_reg_mapping->gp_reg_output_pf,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              output_pf_offset );
          output_pf_offset += 64;
        }
#endif

      }
      l_displacement_k++;
    }

    if (l_k_updates > 0) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_sub_instruction,
          i_gp_reg_mapping->gp_reg_input, 8*l_k_updates*(i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->stride_w)* i_conv_kernel_config->datatype_size_in );
    }

    if (!no_unroll_no_block) {
      /* Add ofw_block*40 to increment output */
      libxsmm_x86_instruction_alu_imm( io_generated_code,
          i_conv_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_input, i_conv_desc->ifw_padded * i_conv_desc->stride_h * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in  );
      if ( unroll_factor > 9 ) {
        libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_input, i_gp_reg_mapping->gp_reg_help_4);
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_help_4,  9 * i_conv_kernel_config->datatype_size_in );
      }
      if ( unroll_factor > 18 ) {
        libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_input, i_gp_reg_mapping->gp_reg_help_5);
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_help_5, 18 *  i_conv_kernel_config->datatype_size_in );
      }
      if ( unroll_factor > 27 ) {
        libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_input, i_gp_reg_mapping->gp_reg_help_6);
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_help_6, 27 *  i_conv_kernel_config->datatype_size_in );
      }
      l_displacement_k = 0;
    }

  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_convolution_weight_update_transpose_avx512_init_weight_strides( libxsmm_generated_code* io_generated_code,
    const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
    const libxsmm_convolution_kernel_config*                i_conv_kernel_config,
    const libxsmm_convolution_weight_update_descriptor*     i_conv_desc ) {

  int unroll_factor = i_conv_desc->ifm_block;

  /* Initialize helper registers for SIB addressing */
  /* helper 0: Index register holding ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_help_0, i_conv_desc->ifw_padded * i_conv_kernel_config->datatype_size_in );
  /* helper 1: Index register holding 3*ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_help_1, i_conv_desc->ifw_padded * i_conv_kernel_config->datatype_size_in * 3 );
  /* helper 2: Index register holding 5*ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_help_2, i_conv_desc->ifw_padded * i_conv_kernel_config->datatype_size_in  * 5 );
  /* helper 3: Index register holding 7*ldb*datatype_size */
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_help_3, i_conv_desc->ifw_padded * i_conv_kernel_config->datatype_size_in * 7 );

  /* helper 4: B + 9*ldb, additional base address
     helper 5: B + 18*ldb, additional base address */
  if ( unroll_factor > 9 ) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_input, i_gp_reg_mapping->gp_reg_help_4);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_help_4,  9 * i_conv_desc->ifw_padded * i_conv_kernel_config->datatype_size_in );
  }
  if ( unroll_factor > 18 ) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_input, i_gp_reg_mapping->gp_reg_help_5);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_help_5, 18 * i_conv_desc->ifw_padded * i_conv_kernel_config->datatype_size_in );
  }
  if ( unroll_factor > 27 ) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_input, i_gp_reg_mapping->gp_reg_help_6);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_help_6, 27 * i_conv_desc->ifw_padded * i_conv_kernel_config->datatype_size_in );
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_convolution_weight_update_transpose_avx512_ofwloop( libxsmm_generated_code* io_generated_code,
    const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
    const libxsmm_convolution_kernel_config*                i_conv_kernel_config,
    const libxsmm_convolution_weight_update_descriptor*     i_conv_desc,
    const unsigned int                                      i_ofh_unroll,
    const unsigned int                                      ofh_trip_counter,
    const int                                               no_unroll_no_block ) {

  unsigned int l_n;
  unsigned int l_k = 0, l_k_1 = 0, l_k_2 = 0;
  unsigned int l_input_reg;
  unsigned int l_input_idx;
  unsigned int l_scale;
  unsigned int l_disp;
  unsigned int l_displacement_k = 0;
#if 0
  unsigned int l_k_updates = 0;
#endif
  unsigned int l_w;
  unsigned int output_counter = 0;
  unsigned int unroll_factor = i_conv_desc->ifm_block;
  unsigned int step_size = 0;
  LIBXSMM_UNUSED(ofh_trip_counter);
  LIBXSMM_UNUSED(i_ofh_unroll);

  /* Currently for formats other than custom format, ifmblock<VECTOR_LENGTH scenario is not optimized*/
  if ( (i_conv_desc->format & LIBXSMM_DNN_TENSOR_FORMAT_NHWC) > 0 ) {
    unroll_factor = i_conv_desc->ifm_block;
  }

  /* apply k blocking */
  if ( i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_KNM ) {
    step_size = 4;
  } else {
    step_size = 1;
  }

  /* setup input strides */
  libxsmm_generator_convolution_weight_update_transpose_avx512_init_weight_strides( io_generated_code, i_gp_reg_mapping, i_conv_kernel_config, i_conv_desc );

  /* apply k blocking */
  for ( l_k_1 = 0; l_k_1 < i_conv_desc->ofh_rb; l_k_1++ ) {
    for ( l_k_2 = 0; l_k_2 < i_conv_desc->ofw_rb; l_k_2+=step_size, l_k+=step_size ) {
      /* for quad, we need to load weights in groups of 4 as this is the
         source block for qmadd */
      for ( l_w = 0; l_w < step_size; l_w++ ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_conv_kernel_config->instruction_set,
            i_conv_kernel_config->vmove_instruction,
            i_gp_reg_mapping->gp_reg_output,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            ((output_counter+l_w)*(i_conv_kernel_config->l_ld_ofm_act)*(i_conv_kernel_config->datatype_size_out) )
            + ((l_k_1) * i_conv_desc->ofw_padded*i_conv_kernel_config->l_ld_ofm_act * i_conv_kernel_config->datatype_size_out  ),
            i_conv_kernel_config->vector_name, l_w,
            0, 1, 0 );
      }
      output_counter += step_size;

      if (output_counter == i_conv_desc->ofw_rb ) {
        output_counter = 0;
      }

      /* compute vectorwidth (A) * column broadcast (B) */
      for ( l_n = 0; l_n < unroll_factor; l_n++) {
        /* determining base, idx and scale values */
        /* default values */
        l_input_reg = i_gp_reg_mapping->gp_reg_input;
        l_input_idx = LIBXSMM_X86_GP_REG_UNDEF;
        l_scale = 0;
        l_disp = l_displacement_k*(i_conv_desc->stride_w * i_conv_kernel_config->datatype_size_in);

        /* select the base register */
        if ( l_n > 26 ) {
          l_input_reg = i_gp_reg_mapping->gp_reg_help_6;
        } else if ( l_n > 17 ) {
          l_input_reg = i_gp_reg_mapping->gp_reg_help_5;
        } else if ( l_n > 8 ) {
          l_input_reg = i_gp_reg_mapping->gp_reg_help_4;
        } else {
          l_input_reg = i_gp_reg_mapping->gp_reg_input;
        }
        if ( l_n % 9 == 0 ) {
          l_input_idx = LIBXSMM_X86_GP_REG_UNDEF;
          l_scale = 0;
        } else if ( l_n % 9 == 1 ) {
          l_input_idx = i_gp_reg_mapping->gp_reg_help_0;
          l_scale = 1;
        } else if ( l_n % 9 == 2 ) {
          l_input_idx = i_gp_reg_mapping->gp_reg_help_0;
          l_scale = 2;
        } else if ( l_n % 9 == 3 ) {
          l_input_idx = i_gp_reg_mapping->gp_reg_help_1;
          l_scale = 1;
        } else if ( l_n % 9 == 4 ) {
          l_input_idx = i_gp_reg_mapping->gp_reg_help_0;
          l_scale = 4;
        } else if ( l_n % 9 == 5 ) {
          l_input_idx = i_gp_reg_mapping->gp_reg_help_2;
          l_scale = 1;
        } else if ( l_n % 9 == 6 ) {
          l_input_idx = i_gp_reg_mapping->gp_reg_help_1;
          l_scale = 2;
        } else if ( l_n % 9 == 7 ) {
          l_input_idx = i_gp_reg_mapping->gp_reg_help_3;
          l_scale = 1;
        } else if ( l_n % 9 == 8 ) {
          l_input_idx = i_gp_reg_mapping->gp_reg_help_0;
          l_scale = 8;
        } else {
          /* shouldn't happen.... */
        }

        if (step_size == 1) {
          libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
              i_conv_kernel_config->instruction_set,
              i_conv_kernel_config->vfma_instruction,
              1,
              l_input_reg,
              l_input_idx,
              l_scale,
              l_disp,
              i_conv_kernel_config->vector_name,
              0,
              i_conv_kernel_config->vector_reg_count - unroll_factor + l_n );
        } else {
          libxsmm_x86_instruction_vec_compute_qfma( io_generated_code,
              i_conv_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_V4FMADDPS,
              l_input_reg,
              l_input_idx,
              l_scale,
              l_disp,
              i_conv_kernel_config->vector_name,
              0,
              i_conv_kernel_config->vector_reg_count - unroll_factor + l_n );
        }

        if ( (l_n == 9) &&  ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
              LIBXSMM_X86_INSTR_PREFETCHT0,
              i_gp_reg_mapping->gp_reg_input_pf,
              LIBXSMM_X86_GP_REG_UNDEF,
              0,
              l_k_1 * i_conv_desc->stride_h * i_conv_desc->ifw_padded * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in +  l_k_2 * i_conv_desc->stride_w *  i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in );
        }

        if (step_size == 4) {
          if ( (l_n == 11) &&  ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) ) {
            libxsmm_x86_instruction_prefetch( io_generated_code,
                LIBXSMM_X86_INSTR_PREFETCHT0,
                i_gp_reg_mapping->gp_reg_input_pf,
                LIBXSMM_X86_GP_REG_UNDEF,
                0,
                l_k_1 * i_conv_desc->stride_h * i_conv_desc->ifw_padded * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in +  (l_k_2+1) * i_conv_desc->stride_w *  i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in );
          }

          if ( (l_n == 13) &&  ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) ) {
            libxsmm_x86_instruction_prefetch( io_generated_code,
                LIBXSMM_X86_INSTR_PREFETCHT0,
                i_gp_reg_mapping->gp_reg_input_pf,
                LIBXSMM_X86_GP_REG_UNDEF,
                0,
                l_k_1 * i_conv_desc->stride_h * i_conv_desc->ifw_padded * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in +  (l_k_2+2) * i_conv_desc->stride_w *  i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in );
          }

          if ( (l_n == 15) &&  ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) ) {
            libxsmm_x86_instruction_prefetch( io_generated_code,
                LIBXSMM_X86_INSTR_PREFETCHT0,
                i_gp_reg_mapping->gp_reg_input_pf,
                LIBXSMM_X86_GP_REG_UNDEF,
                0,
                l_k_1 * i_conv_desc->stride_h * i_conv_desc->ifw_padded * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in +  (l_k_2+3) * i_conv_desc->stride_w  *  i_conv_kernel_config->l_ld_ifm_act  *i_conv_kernel_config->datatype_size_in );
          }
        }

        if ( (l_n == 1) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
              i_conv_kernel_config->prefetch_instruction,
              i_gp_reg_mapping->gp_reg_output_pf,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (l_k_1 * i_conv_desc->ofw_padded + l_k_2)*(i_conv_kernel_config->l_ld_ofm_act)*(i_conv_kernel_config->datatype_size_out) );
        }
        if (step_size == 4) {
          if ( (l_n == 3) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2) ) {
            libxsmm_x86_instruction_prefetch( io_generated_code,
                i_conv_kernel_config->prefetch_instruction,
                i_gp_reg_mapping->gp_reg_output_pf,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                (l_k_1 * i_conv_desc->ofw_padded + (l_k_2+1))*(i_conv_kernel_config->l_ld_ofm_act)*(i_conv_kernel_config->datatype_size_out) );
          }
          if ( (l_n == 5) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2) ) {
            libxsmm_x86_instruction_prefetch( io_generated_code,
                i_conv_kernel_config->prefetch_instruction,
                i_gp_reg_mapping->gp_reg_output_pf,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                (l_k_1 * i_conv_desc->ofw_padded + (l_k_2+2))*(i_conv_kernel_config->l_ld_ofm_act)*(i_conv_kernel_config->datatype_size_out) );
          }
          if ( (l_n == 7) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2) ) {
            libxsmm_x86_instruction_prefetch( io_generated_code,
                i_conv_kernel_config->prefetch_instruction,
                i_gp_reg_mapping->gp_reg_output_pf,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                (l_k_1 * i_conv_desc->ofw_padded + (l_k_2+3))*(i_conv_kernel_config->l_ld_ofm_act)*(i_conv_kernel_config->datatype_size_out) );
          }
        }
      }

      l_displacement_k+=step_size;
    } /* l_k_2 */
#if 0
    if (l_k_updates > 0) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_sub_instruction,
          i_gp_reg_mapping->gp_reg_input, 8*l_k_updates*(i_conv_desc->ifm_block * i_conv_desc->stride_w)* i_conv_kernel_config->datatype_size_in );
    }
#endif

    if (!no_unroll_no_block) {
      /* Add ofw_block*40 to increment output */
      libxsmm_x86_instruction_alu_imm( io_generated_code,
          i_conv_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_input, i_conv_desc->ifw_padded * i_conv_desc->stride_h *
          i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in  );
      if ( unroll_factor > 9 ) {
        libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_input, i_gp_reg_mapping->gp_reg_help_4);
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_help_4,  9 * i_conv_desc->ifw_padded * i_conv_kernel_config->datatype_size_in );
      }
      if ( unroll_factor > 18 ) {
        libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_input, i_gp_reg_mapping->gp_reg_help_5);
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_help_5, 18 * i_conv_desc->ifw_padded * i_conv_kernel_config->datatype_size_in );
      }
      if ( unroll_factor > 27 ) {
        libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_input, i_gp_reg_mapping->gp_reg_help_6);
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_help_6, 27 * i_conv_desc->ifw_padded * i_conv_kernel_config->datatype_size_in );
      }
      l_displacement_k = 0;
    } /* if no unroll */

  } /* l_k_1 */
}


LIBXSMM_API_INTERN
void libxsmm_generator_convolution_weight_update_transpose_avx512_ofwloop_all_pixels_inside( libxsmm_generated_code* io_generated_code,
    const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
    const libxsmm_convolution_kernel_config*                i_conv_kernel_config,
    const libxsmm_convolution_weight_update_descriptor*     i_conv_desc,
    unsigned int                                            is_last_call) {
  unsigned int l_n = 0;
  unsigned int l_k_1 = 0, l_k_2 = 0;
  unsigned int l_disp;
  unsigned int l_w;
  unsigned int output_counter = 0;
  unsigned int unroll_factor = i_conv_desc->ifm_block_hp;
  unsigned int input_pf_register;
  unsigned int output_pf_register;
  unsigned int input_pf_init_offset;
  unsigned int output_pf_init_offset;
  unsigned int cache_line_offset = 0;
  unsigned int prefetch_type_input = LIBXSMM_X86_INSTR_PREFETCHT0;
  unsigned int prefetch_type_output = LIBXSMM_X86_INSTR_PREFETCHT0;
  unsigned int step_size = 0;
  unsigned int l_compute_instr = 0;
  unsigned int lp_dim_out = 1;
  unsigned int vperm_instr = LIBXSMM_X86_INSTR_VPERMW;
  unsigned int use_lp_kernel = 0;
  LIBXSMM_UNUSED(use_lp_kernel);
  LIBXSMM_UNUSED(is_last_call);

  /* depending on datatype emit the needed FMA(-sequence) */
  if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_F32 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
    l_compute_instr = LIBXSMM_X86_INSTR_V4FMADDPS;
  } else if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
    l_compute_instr = LIBXSMM_X86_INSTR_VP4DPWSSDS;
  } else {
    /* shouldn't happen */
  }

  /* calculate k blocking */
  if ( i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_KNM ) {
    if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_F32 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
      step_size = 4;
      lp_dim_out = 1;
      use_lp_kernel = 0;
    } else if ( (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32) || i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_BF16 ) {
      step_size = 8;
      lp_dim_out = 2;
      use_lp_kernel = 1;
    } else {
      /* shouldn't happen */
    }
  } else {
    if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_F32 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
      step_size = 1;
      lp_dim_out = 1;
      use_lp_kernel = 0;
    } else if ( (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32) || i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_BF16 ) {
      step_size = 2;
      lp_dim_out = 2;
      use_lp_kernel = 1;
    } else if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I8 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32 ) {
      step_size = 4;
      lp_dim_out = 4;
      use_lp_kernel = 1;
    } else {
      /* shouldn't happen */
    }
  }

  for ( l_k_1 = 0; l_k_1 < i_conv_desc->ofh_rb; l_k_1++) {
    cache_line_offset = 0;

    /* apply k blocking */
    for ( l_k_2 = 0; l_k_2 < i_conv_desc->ofw_rb; l_k_2+=step_size ) {
      /* for quad, we need to load outputs in groups of 4 as this is the source block for qmadd */
      int n_fake_pixels;
      int n_compute_pixels;
      int bound;

      /*
      if (use_lp_kernel == 0) {
        n_fake_pixels = i_conv_desc->ofw_fake_pixels;
      } else {
        n_fake_pixels = 0;
      }*/

      n_fake_pixels = i_conv_desc->ofw_fake_pixels;

      n_compute_pixels = i_conv_desc->ofw_rb - n_fake_pixels;
      bound = LIBXSMM_MIN(step_size/lp_dim_out, (n_compute_pixels-l_k_2)/lp_dim_out);

      if ( i_conv_desc->avoid_output_trans == 0 )  {
        for ( l_w = 0; l_w < (unsigned int)bound; l_w++ ) {
          if  (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_BF16) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_conv_kernel_config->instruction_set,
                i_conv_kernel_config->vmove_instruction,
                i_gp_reg_mapping->gp_reg_output,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((output_counter+l_w)*(i_conv_kernel_config->l_ld_ofm_act * lp_dim_out)*(i_conv_kernel_config->datatype_size_out) )
                + ((l_k_1) * i_conv_desc->ofw_padded*i_conv_kernel_config->l_ld_ofm_act * i_conv_kernel_config->datatype_size_out  ) ,
                i_conv_kernel_config->vector_name, l_w,
                0, 1, 0 );

            /* vpslld  */
            libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
                i_conv_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPSLLD,
                i_conv_kernel_config->vector_name,
                0,
                1,
                LIBXSMM_X86_VEC_REG_UNDEF,
                16);

            /* vpsrad */
            libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
                i_conv_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPSRAD,
                i_conv_kernel_config->vector_name,
                0,
                0,
                LIBXSMM_X86_VEC_REG_UNDEF,
                16);

            /* vpslld */
            libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
                i_conv_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPSLLD,
                i_conv_kernel_config->vector_name,
                0,
                0,
                LIBXSMM_X86_VEC_REG_UNDEF,
                16);
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_conv_kernel_config->instruction_set,
                i_conv_kernel_config->vmove_instruction,
                i_gp_reg_mapping->gp_reg_output,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((output_counter+l_w)*(i_conv_kernel_config->l_ld_ofm_act * lp_dim_out)*(i_conv_kernel_config->datatype_size_out) )
                + ((l_k_1) * i_conv_desc->ofw_padded*i_conv_kernel_config->l_ld_ofm_act * i_conv_kernel_config->datatype_size_out  ),
                i_conv_kernel_config->vector_name, l_w,
                0, 1, 0 );
          }
        }
      } else {
        l_w = 0;
        if  (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_BF16) {
          libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
              i_conv_kernel_config->instruction_set,
              vperm_instr,
              0,
              i_gp_reg_mapping->gp_reg_output,
              LIBXSMM_X86_GP_REG_UNDEF,
              LIBXSMM_X86_GP_REG_UNDEF,
              ((output_counter+l_w)*(i_conv_kernel_config->l_ld_ofm_act * lp_dim_out)*(i_conv_kernel_config->datatype_size_out) )
              + ((l_k_1) * i_conv_desc->ofw_padded*i_conv_kernel_config->l_ld_ofm_act * i_conv_kernel_config->datatype_size_out  ),
              i_conv_kernel_config->vector_name,
              4,
              0 );

          /* vpslld  */
          libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
              i_conv_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPSLLD,
              i_conv_kernel_config->vector_name,
              0,
              1,
              LIBXSMM_X86_VEC_REG_UNDEF,
              16);

          /* vpsrad */
          libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
              i_conv_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPSRAD,
              i_conv_kernel_config->vector_name,
              0,
              0,
              LIBXSMM_X86_VEC_REG_UNDEF,
              16);

          /* vpslld */
          libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
              i_conv_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPSLLD,
              i_conv_kernel_config->vector_name,
              0,
              0,
              LIBXSMM_X86_VEC_REG_UNDEF,
              16);

        } else {
          libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
              i_conv_kernel_config->instruction_set,
              vperm_instr,
              0,
              i_gp_reg_mapping->gp_reg_output,
              LIBXSMM_X86_GP_REG_UNDEF,
              LIBXSMM_X86_GP_REG_UNDEF,
              ((output_counter+l_w)*(i_conv_kernel_config->l_ld_ofm_act * lp_dim_out)*(i_conv_kernel_config->datatype_size_out) )
              + ((l_k_1) * i_conv_desc->ofw_padded*i_conv_kernel_config->l_ld_ofm_act * i_conv_kernel_config->datatype_size_out  ),
              i_conv_kernel_config->vector_name,
              4,
              0 );
        }

        l_w = bound;
      }

      for (; l_w < step_size/lp_dim_out; l_w++ ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
            i_conv_kernel_config->instruction_set,
            i_conv_kernel_config->vxor_instruction,
            i_conv_kernel_config->vector_name,
            l_w,
            l_w,
            l_w);
      }

      output_counter += step_size/lp_dim_out;

      if (output_counter == i_conv_desc->ofw_rb/lp_dim_out ) {
        output_counter = 0;
      }

      /* compute vectorwidth (A) * column broadcast (B) */
      for ( l_n = 0; l_n < unroll_factor; l_n++) {
        int pixel_id;

        /* set displacement */
        l_disp =   l_k_1 * (i_conv_desc->ifw_padded * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in)
          + l_k_2 *  i_conv_kernel_config->datatype_size_in
          + l_n * ( i_conv_desc->ifw_padded *  i_conv_kernel_config->datatype_size_in);

        if (step_size/lp_dim_out == 1) {
          if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_F32 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
            libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                i_conv_kernel_config->instruction_set,
                i_conv_kernel_config->vfma_instruction,
                1,
                i_gp_reg_mapping->gp_reg_input,
                LIBXSMM_X86_GP_REG_UNDEF,
                LIBXSMM_X86_GP_REG_UNDEF,
                l_disp,
                i_conv_kernel_config->vector_name,
                0,
                i_conv_kernel_config->vector_reg_count - unroll_factor + l_n );
          } else if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
            if ( i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_CORE ) {
              libxsmm_x86_instruction_vec_move( io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  LIBXSMM_X86_INSTR_VPBROADCASTD,
                  i_gp_reg_mapping->gp_reg_input,
                  LIBXSMM_X86_GP_REG_UNDEF,
                  0,
                  l_disp,
                  i_conv_kernel_config->vector_name,
                  1, 0, 1, 0 );
              libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  LIBXSMM_X86_INSTR_VPMADDWD,
                  i_conv_kernel_config->vector_name,
                  0,
                  1,
                  1 );
              libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  LIBXSMM_X86_INSTR_VPADDD,
                  i_conv_kernel_config->vector_name,
                  1,
                  i_conv_kernel_config->vector_reg_count - unroll_factor + l_n,
                  i_conv_kernel_config->vector_reg_count - unroll_factor + l_n );

            } else if (i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_ICL) {
              libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  LIBXSMM_X86_INSTR_VPDPWSSDS,
                  1,
                  i_gp_reg_mapping->gp_reg_input,
                  LIBXSMM_X86_GP_REG_UNDEF,
                  0,
                  l_disp,
                  i_conv_kernel_config->vector_name,
                  0,
                  i_conv_kernel_config->vector_reg_count - unroll_factor + l_n );
            } else {
              /* shouldn't happen */
            }
          } else if (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_BF16)  {
            /* bcast  */
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_conv_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPBROADCASTD,
                i_gp_reg_mapping->gp_reg_input,
                LIBXSMM_X86_GP_REG_UNDEF,
                0,
                l_disp,
                i_conv_kernel_config->vector_name,
                3, 0, 1, 0 );

            /* vpslld  */
            libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
                i_conv_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPSLLD,
                i_conv_kernel_config->vector_name,
                3,
                2,
                LIBXSMM_X86_VEC_REG_UNDEF,
                16);

            /* vfma */
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                i_conv_kernel_config->instruction_set,
                i_conv_kernel_config->vfma_instruction,
                i_conv_kernel_config->vector_name,
                1,
                2,
                i_conv_kernel_config->vector_reg_count - unroll_factor + l_n);

            /* vpsrad */
            libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
                i_conv_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPSRAD,
                i_conv_kernel_config->vector_name,
                3,
                3,
                LIBXSMM_X86_VEC_REG_UNDEF,
                16);

            /* vpslld  */
            libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
                i_conv_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPSLLD,
                i_conv_kernel_config->vector_name,
                3,
                3,
                LIBXSMM_X86_VEC_REG_UNDEF,
                16);

            /* vfma */
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                i_conv_kernel_config->instruction_set,
                i_conv_kernel_config->vfma_instruction,
                i_conv_kernel_config->vector_name,
                0,
                3,
                i_conv_kernel_config->vector_reg_count - unroll_factor + l_n);
          } else if ((i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I8  && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32 && (i_conv_desc->option & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0)) {
            /* broadcast in quadruples of 8 bit values */
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_conv_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPBROADCASTD,
                i_gp_reg_mapping->gp_reg_input,
                LIBXSMM_X86_GP_REG_UNDEF,
                0,
                l_disp,
                i_conv_kernel_config->vector_name,
                1, 0, 1, 0 );

            /* 8/16bit integer MADD with horizontal add */
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                i_conv_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPMADDUBSW,
                i_conv_kernel_config->vector_name,
                0,
                1,
                2 );

            /* 16/32bit integer MADD with horizontal add */
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                i_conv_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPMADDWD,
                i_conv_kernel_config->vector_name,
                2,
                3,
                2 );

            /* 32bit integer accumulation without saturation into running result buffer */
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                i_conv_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPADDD,
                i_conv_kernel_config->vector_name,
                2,
                i_conv_kernel_config->vector_reg_count - unroll_factor + l_n,
                i_conv_kernel_config->vector_reg_count - unroll_factor + l_n);
          }
        } else {
          libxsmm_x86_instruction_vec_compute_qfma( io_generated_code,
              i_conv_kernel_config->instruction_set,
              l_compute_instr,
              i_gp_reg_mapping->gp_reg_input,
              LIBXSMM_X86_GP_REG_UNDEF,
              0,
              l_disp,
              i_conv_kernel_config->vector_name,
              0,
              i_conv_kernel_config->vector_reg_count - unroll_factor + l_n );
        }

        if (l_k_1 < i_conv_desc->ofh_rb-1 ) {
          /* prefetch next W row from input/output registers  */
          input_pf_init_offset = (l_k_1 + 1) *  i_conv_desc->ifw_padded * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in;
          input_pf_register = i_gp_reg_mapping->gp_reg_input;
          output_pf_init_offset = (l_k_1 + 1) *  i_conv_desc->ofw_padded * i_conv_kernel_config->l_ld_ofm_act * i_conv_kernel_config->datatype_size_out;
          output_pf_register = i_gp_reg_mapping->gp_reg_output;
        } else {
          /* prefetch from reg1 (either first row of next H block or first row of next kernel call) */
          input_pf_init_offset = 0;
          input_pf_register = i_gp_reg_mapping->gp_reg_help_1;
          output_pf_init_offset = 0;
          output_pf_register = i_gp_reg_mapping->gp_reg_help_2;
        }

        /* Input prefetches */
        if (l_n == 0) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
              prefetch_type_input,
              input_pf_register,
              LIBXSMM_X86_GP_REG_UNDEF,
              0,
              input_pf_init_offset + cache_line_offset );
        }

        if (step_size == 4 || step_size == 8) {
          if (l_n == 2) {
            libxsmm_x86_instruction_prefetch( io_generated_code,
                prefetch_type_input,
                input_pf_register,
                LIBXSMM_X86_GP_REG_UNDEF,
                0,
                input_pf_init_offset + cache_line_offset + 64);
          }

          if (l_n == 4) {
            libxsmm_x86_instruction_prefetch( io_generated_code,
                prefetch_type_input,
                input_pf_register,
                LIBXSMM_X86_GP_REG_UNDEF,
                0,
                input_pf_init_offset + cache_line_offset + 128);
          }

          if (l_n == 6) {
            libxsmm_x86_instruction_prefetch( io_generated_code,
                prefetch_type_input,
                input_pf_register,
                LIBXSMM_X86_GP_REG_UNDEF,
                0,
                input_pf_init_offset + cache_line_offset + 192);
          }
        }

        /* Output prefetches */
        pixel_id = cache_line_offset / 64;
        if (l_n == 8 && pixel_id < n_compute_pixels/((int)lp_dim_out)) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
              prefetch_type_output,
              output_pf_register,
              LIBXSMM_X86_GP_REG_UNDEF,
              0,
              output_pf_init_offset + cache_line_offset );
        }

        if ( (step_size == 4) | (step_size == 8) ) {
          pixel_id = (cache_line_offset+64)/64;
          if (l_n == 10 && pixel_id < n_compute_pixels/((int)lp_dim_out)) {
            libxsmm_x86_instruction_prefetch( io_generated_code,
                prefetch_type_output,
                output_pf_register,
                LIBXSMM_X86_GP_REG_UNDEF,
                0,
                output_pf_init_offset + cache_line_offset + 64);
          }

          pixel_id = (cache_line_offset+128)/64;
          if (l_n == 12 && pixel_id < n_compute_pixels/((int)lp_dim_out)) {
            libxsmm_x86_instruction_prefetch( io_generated_code,
                prefetch_type_output,
                output_pf_register,
                LIBXSMM_X86_GP_REG_UNDEF,
                0,
                output_pf_init_offset + cache_line_offset + 128);
          }

          pixel_id = (cache_line_offset+192)/64;
          if (l_n == 14 && pixel_id < n_compute_pixels/((int)lp_dim_out)) {
            libxsmm_x86_instruction_prefetch( io_generated_code,
                prefetch_type_output,
                output_pf_register,
                LIBXSMM_X86_GP_REG_UNDEF,
                0,
                output_pf_init_offset + cache_line_offset + 192);
          }
        }

#if 1
        LIBXSMM_UNUSED(is_last_call);
#else
        int before = 0; /*atoi(getenv("BEFORE"));*/
        if ( (is_last_call == 1) && (i_conv_desc->use_nts == 0) && (l_k_2 ==  i_conv_desc->ofw_rb - before) && (l_k_1 == i_conv_desc->ofh_rb - 1)   ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
              prefetch_type_weight,
              i_gp_reg_mapping->gp_reg_weight_pf,
              LIBXSMM_X86_GP_REG_UNDEF,
              0,
              weight_pf_offset);
          weight_pf_offset += 64;
        }
#endif

      }

      cache_line_offset += (step_size/lp_dim_out) * 64;
    } /* l_k_2 */
  } /* l_k_1 */
}


LIBXSMM_API_INTERN
void libxsmm_generator_convolution_weight_update_avx512_ofwloop_all_pixels_inside( libxsmm_generated_code* io_generated_code,
    const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
    const libxsmm_convolution_kernel_config*                i_conv_kernel_config,
    const libxsmm_convolution_weight_update_descriptor*     i_conv_desc,
    unsigned int                                            is_last_call) {
  unsigned int l_n = 0;
  unsigned int l_k_1 = 0, l_k_2 = 0;
  unsigned int l_disp;
  unsigned int unroll_factor = i_conv_desc->ifm_block_hp;
  unsigned int input_pf_register;
  unsigned int output_pf_register;
  unsigned int input_pf_init_offset;
  unsigned int output_pf_init_offset;
  unsigned int prefetch_type_input = LIBXSMM_X86_INSTR_PREFETCHT0;
  unsigned int prefetch_type_output = LIBXSMM_X86_INSTR_PREFETCHT0;
  unsigned int step_size = 0;
  unsigned int l_compute_instr = 0;
  unsigned int lookahead = 1;
  unsigned int vperm_instr = LIBXSMM_X86_INSTR_VPERMW;
  unsigned int use_lp_kernel = 0;
  LIBXSMM_UNUSED(use_lp_kernel);
  LIBXSMM_UNUSED(is_last_call);


  /* depending on datatype emit the needed FMA(-sequence) */
  if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_F32 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
    l_compute_instr = LIBXSMM_X86_INSTR_VFMADD231PS;
    step_size = 1;
    use_lp_kernel = 0;
  } else if ( (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32) || i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_BF16)  {
    /* @TODO this needs to be fixed */
    l_compute_instr = LIBXSMM_X86_INSTR_VP4DPWSSDS;
    step_size = 2;
    use_lp_kernel = 1;
  } else {
    /* shouldn't happen */
    return;
  }

  for ( l_k_1 = 0; l_k_1 < i_conv_desc->ofh_rb; l_k_1++) {
    unsigned int pipeline_vperms = (use_lp_kernel == 1 && (i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_ICL || i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_CORE)) ? 1 : 0;
    unsigned int input_reg_to_use;

    /* Load+per to buf0  */
    if (pipeline_vperms) {
      l_k_2 = 0;
      libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
          i_conv_kernel_config->instruction_set,
          vperm_instr,
          0,
          i_gp_reg_mapping->gp_reg_input,
          LIBXSMM_X86_GP_REG_UNDEF,
          LIBXSMM_X86_GP_REG_UNDEF,
          l_k_1 * i_conv_desc->stride_h * i_conv_desc->ifw_padded * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in + l_k_2 * i_conv_desc->stride_w * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in,
          i_conv_kernel_config->vector_name,
          4,
          2 );

      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_conv_kernel_config->instruction_set,
          i_conv_kernel_config->vmove_instruction,
          i_gp_reg_mapping->gp_reg_help_4,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          0,
          i_conv_kernel_config->vector_name,
          2, 0, 0, 1 );
    }

    for ( l_k_2 = 0; l_k_2 < i_conv_desc->ofw_rb; l_k_2 += step_size) {
      unsigned int dst_scratch_reg;

      if ( use_lp_kernel == 0 || i_conv_desc->avoid_output_trans == 0 )  {
        if (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_BF16) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_conv_kernel_config->instruction_set,
              i_conv_kernel_config->vmove_instruction,
              i_gp_reg_mapping->gp_reg_output,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              l_k_2 * i_conv_kernel_config->l_ld_ofm_act * i_conv_kernel_config->datatype_size_out
              + l_k_1 * i_conv_desc->ofw_padded*i_conv_kernel_config->l_ld_ofm_act * i_conv_kernel_config->datatype_size_out,
              i_conv_kernel_config->vector_name, 0,
              0, 1, 0 );
          /* vpslld  */
          libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
              i_conv_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPSLLD,
              i_conv_kernel_config->vector_name,
              0,
              1,
              LIBXSMM_X86_VEC_REG_UNDEF,
              16);

          /* vpsrad */
          libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
              i_conv_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPSRAD,
              i_conv_kernel_config->vector_name,
              0,
              0,
              LIBXSMM_X86_VEC_REG_UNDEF,
              16);

          /* vpslld */
          libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
              i_conv_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPSLLD,
              i_conv_kernel_config->vector_name,
              0,
              0,
              LIBXSMM_X86_VEC_REG_UNDEF,
              16);
        } else {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_conv_kernel_config->instruction_set,
              i_conv_kernel_config->vmove_instruction,
              i_gp_reg_mapping->gp_reg_output,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              l_k_2 * i_conv_kernel_config->l_ld_ofm_act * i_conv_kernel_config->datatype_size_out
              + l_k_1 * i_conv_desc->ofw_padded*i_conv_kernel_config->l_ld_ofm_act * i_conv_kernel_config->datatype_size_out,
              i_conv_kernel_config->vector_name, 0,
              0, 1, 0 );
        }
      } else {
        if (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_BF16) {
          libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
              i_conv_kernel_config->instruction_set,
              vperm_instr,
              0,
              i_gp_reg_mapping->gp_reg_output,
              LIBXSMM_X86_GP_REG_UNDEF,
              LIBXSMM_X86_GP_REG_UNDEF,
              l_k_2 * i_conv_kernel_config->l_ld_ofm_act * i_conv_kernel_config->datatype_size_out
              + l_k_1 * i_conv_desc->ofw_padded*i_conv_kernel_config->l_ld_ofm_act * i_conv_kernel_config->datatype_size_out,
              i_conv_kernel_config->vector_name,
              4,
              0 );

          /* vpslld  */
          libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
              i_conv_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPSLLD,
              i_conv_kernel_config->vector_name,
              0,
              1,
              LIBXSMM_X86_VEC_REG_UNDEF,
              16);

          /* vpsrad */
          libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
              i_conv_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPSRAD,
              i_conv_kernel_config->vector_name,
              0,
              0,
              LIBXSMM_X86_VEC_REG_UNDEF,
              16);

          /* vpslld */
          libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
              i_conv_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPSLLD,
              i_conv_kernel_config->vector_name,
              0,
              0,
              LIBXSMM_X86_VEC_REG_UNDEF,
              16);
        } else {
          libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
              i_conv_kernel_config->instruction_set,
              vperm_instr,
              0,
              i_gp_reg_mapping->gp_reg_output,
              LIBXSMM_X86_GP_REG_UNDEF,
              LIBXSMM_X86_GP_REG_UNDEF,
              l_k_2 * i_conv_kernel_config->l_ld_ofm_act * i_conv_kernel_config->datatype_size_out
              + l_k_1 * i_conv_desc->ofw_padded*i_conv_kernel_config->l_ld_ofm_act * i_conv_kernel_config->datatype_size_out,
              i_conv_kernel_config->vector_name,
              4,
              0 );
        }
      }

      dst_scratch_reg = (l_k_2%4 == 0) ? i_gp_reg_mapping->gp_reg_help_6 : i_gp_reg_mapping->gp_reg_help_4;

      if ( (pipeline_vperms == 1) && ( (l_k_2+step_size) < i_conv_desc->ofw_rb)) {
        /* Do the "input transpose" and store the result in the vnni scratch cache line*/
        libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
            i_conv_kernel_config->instruction_set,
            vperm_instr,
            0,
            i_gp_reg_mapping->gp_reg_input,
            LIBXSMM_X86_GP_REG_UNDEF,
            LIBXSMM_X86_GP_REG_UNDEF,
            l_k_1 * i_conv_desc->stride_h * i_conv_desc->ifw_padded * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in + (l_k_2+step_size) * i_conv_desc->stride_w * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in,
            i_conv_kernel_config->vector_name,
            4,
            2 );

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_conv_kernel_config->instruction_set,
            i_conv_kernel_config->vmove_instruction,
            dst_scratch_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            i_conv_kernel_config->vector_name,
            2, 0, 0, 1 );
      }


      /* compute vectorwidth (A) * column broadcast (B) */
      for ( l_n = 0; l_n < unroll_factor; l_n++) {
        /* set displacement */
        if (pipeline_vperms == 1) {
          l_disp = l_n * step_size * i_conv_kernel_config->datatype_size_in;
          input_reg_to_use =  (l_k_2%4 == 0) ? i_gp_reg_mapping->gp_reg_help_4 : i_gp_reg_mapping->gp_reg_help_6;
        } else {
          l_disp =   l_k_1 * i_conv_desc->stride_h * i_conv_desc->ifw_padded * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in
            + l_k_2 * i_conv_desc->stride_w * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in
            + l_n * step_size * i_conv_kernel_config->datatype_size_in;
          input_reg_to_use = i_gp_reg_mapping->gp_reg_input;
        }

        if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_F32 ) {
          libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
              i_conv_kernel_config->instruction_set,
              l_compute_instr,
              1,
              i_gp_reg_mapping->gp_reg_input,
              LIBXSMM_X86_GP_REG_UNDEF,
              LIBXSMM_X86_GP_REG_UNDEF,
              l_disp,
              i_conv_kernel_config->vector_name,
              0,
              i_conv_kernel_config->vector_reg_count - unroll_factor + l_n );
        } else if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_BF16 ) {
          /* bcast  */
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_conv_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPBROADCASTD,
              input_reg_to_use,
              LIBXSMM_X86_GP_REG_UNDEF,
              0,
              l_disp,
              i_conv_kernel_config->vector_name,
              3, 0, 1, 0 );

          /* vpslld  */
          libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
              i_conv_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPSLLD,
              i_conv_kernel_config->vector_name,
              3,
              2,
              LIBXSMM_X86_VEC_REG_UNDEF,
              16);

          /* vfma */
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
              i_conv_kernel_config->instruction_set,
              i_conv_kernel_config->vfma_instruction,
              i_conv_kernel_config->vector_name,
              1,
              2,
              i_conv_kernel_config->vector_reg_count - unroll_factor + l_n);

           /* vpsrad */
           libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
              i_conv_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPSRAD,
              i_conv_kernel_config->vector_name,
              3,
              3,
              LIBXSMM_X86_VEC_REG_UNDEF,
              16);

           /* vpslld  */
           libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
              i_conv_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPSLLD,
              i_conv_kernel_config->vector_name,
              3,
              3,
              LIBXSMM_X86_VEC_REG_UNDEF,
              16);

           /* vfma */
           libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
              i_conv_kernel_config->instruction_set,
              i_conv_kernel_config->vfma_instruction,
              i_conv_kernel_config->vector_name,
              0,
              3,
              i_conv_kernel_config->vector_reg_count - unroll_factor + l_n);
        } else if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 ) {
          if ( (i_conv_kernel_config->instruction_set != LIBXSMM_X86_AVX512_ICL) ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_conv_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPBROADCASTD,
                input_reg_to_use,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                l_disp,
                i_conv_kernel_config->vector_name,
                1, 0, 1, 0 );
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                i_conv_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPMADDWD,
                i_conv_kernel_config->vector_name,
                0,
                1,
                1 );
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                i_conv_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPADDD,
                i_conv_kernel_config->vector_name,
                1,
                i_conv_kernel_config->vector_reg_count - unroll_factor + l_n,
                i_conv_kernel_config->vector_reg_count - unroll_factor + l_n);
          } else {
            libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                i_conv_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPDPWSSDS,
                1,
                input_reg_to_use,
                LIBXSMM_X86_GP_REG_UNDEF,
                0,
                l_disp,
                i_conv_kernel_config->vector_name,
                0,
                i_conv_kernel_config->vector_reg_count - unroll_factor + l_n );
          }
        } else {
          /* shouldn't happen */
        }

        if (l_k_1+lookahead < i_conv_desc->ofh_rb ) {
          /* prefetch next W row from input/output registers  */
          input_pf_init_offset = (l_k_1+lookahead) * i_conv_desc->stride_h * i_conv_desc->ifw_padded * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in
            + l_k_2 * i_conv_desc->stride_w * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in;
          input_pf_register = i_gp_reg_mapping->gp_reg_input;
          output_pf_init_offset = (l_k_1+lookahead) *  i_conv_desc->ofw_padded * i_conv_kernel_config->l_ld_ofm_act * i_conv_kernel_config->datatype_size_out
            + l_k_2 * (i_conv_kernel_config->l_ld_ofm_act * i_conv_kernel_config->datatype_size_out);
          output_pf_register = i_gp_reg_mapping->gp_reg_output;
        } else {
          /* prefetch from reg1 (either first row of next H block or first row of next kernel call) */
          input_pf_init_offset = l_k_2 * i_conv_desc->stride_w * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in;
          input_pf_register = i_gp_reg_mapping->gp_reg_help_1;
          output_pf_init_offset = l_k_2 * (i_conv_kernel_config->l_ld_ofm_act * i_conv_kernel_config->datatype_size_out);
          output_pf_register = i_gp_reg_mapping->gp_reg_help_2;
        }

        /* Input prefetches */
        if (l_n == 0) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
              prefetch_type_input,
              input_pf_register,
              LIBXSMM_X86_GP_REG_UNDEF,
              0,
              input_pf_init_offset);
        }

        /* Output prefetches */
        if (l_n == 8) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
              prefetch_type_output,
              output_pf_register,
              LIBXSMM_X86_GP_REG_UNDEF,
              0,
              output_pf_init_offset);
        }
      }
    } /* l_k_2 */
  } /* l_k_1 */
}

