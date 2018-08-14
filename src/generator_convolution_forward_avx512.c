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
/* Alexander Heinecke, Evangelos Georganas, Naveen Mellempudi (Intel Corp.)
******************************************************************************/

#include "generator_convolution_forward_avx512.h"
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
void libxsmm_generator_convolution_forward_avx512_kernel(
  libxsmm_generated_code*                       io_generated_code,
  const libxsmm_convolution_forward_descriptor* i_conv_desc,
  const char*                                   i_arch )
{
  libxsmm_convolution_kernel_config l_conv_kernel_config;
  libxsmm_convolution_forward_gp_reg_mapping l_gp_reg_mapping;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  unsigned int l_kw_trips = 1;
  unsigned int l_kh_trips = 1;
  unsigned int l_kh = 0;
  unsigned int l_found_act_format = 0;
  unsigned int l_found_fil_format = 0;
  unsigned int i_out_pf, j_out_pf;
  unsigned int rsp_maxval_offset = 56;
  int prefetch_scale_factor;
#if 0
  int prefetch_current_output;
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
  l_gp_reg_mapping.gp_reg_ifmOuter_loop = LIBXSMM_X86_GP_REG_R15;
  l_gp_reg_mapping.gp_reg_input_pf_L2 = LIBXSMM_X86_GP_REG_R13;
  l_gp_reg_mapping.gp_reg_weight_pf_L2 = LIBXSMM_X86_GP_REG_R14;
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
  if ( strcmp( i_arch, "knm" ) == 0 ) {
    l_conv_kernel_config.instruction_set = LIBXSMM_X86_AVX512_KNM;
  } else if ( strcmp( i_arch, "knl" ) == 0 ) {
    l_conv_kernel_config.instruction_set = LIBXSMM_X86_AVX512_MIC;
  } else if ( strcmp( i_arch, "skx" ) == 0 ) {
    l_conv_kernel_config.instruction_set = LIBXSMM_X86_AVX512_CORE;
  } else if ( strcmp( i_arch, "icl" ) == 0 ) {
    l_conv_kernel_config.instruction_set = LIBXSMM_X86_AVX512_ICL;
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
    l_conv_kernel_config.vector_length_in = 32;
    l_conv_kernel_config.datatype_size_in = 2;
    l_conv_kernel_config.vector_length_out = 16;
    l_conv_kernel_config.datatype_size_out = 4;
    l_conv_kernel_config.vector_length_wt = 32;
    l_conv_kernel_config.datatype_size_wt = 2;
    l_conv_kernel_config.vfma_instruction = LIBXSMM_X86_INSTR_VPMADDWD;
    l_conv_kernel_config.vadd_instruction = LIBXSMM_X86_INSTR_VPADDD;
    l_conv_kernel_config.vbcst_instruction = LIBXSMM_X86_INSTR_VPBROADCASTD;
  } else if (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_BF16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_BF16) {
    l_conv_kernel_config.vector_length_in = 32;
    l_conv_kernel_config.datatype_size_in = 2;
    l_conv_kernel_config.vector_length_out = 16;
    l_conv_kernel_config.datatype_size_out = 2;
    l_conv_kernel_config.vector_length_wt = 32;
    l_conv_kernel_config.datatype_size_wt = 2;
    l_conv_kernel_config.vfma_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
    l_conv_kernel_config.vadd_instruction = LIBXSMM_X86_INSTR_VPADDD;
    l_conv_kernel_config.vbcst_instruction = LIBXSMM_X86_INSTR_VPBROADCASTD;
  } else if (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32) {
    l_conv_kernel_config.vector_length_in = 32;
    l_conv_kernel_config.datatype_size_in = 2;
    l_conv_kernel_config.vector_length_out = 16;
    l_conv_kernel_config.datatype_size_out = 4;
    l_conv_kernel_config.vector_length_wt = 32;
    l_conv_kernel_config.datatype_size_wt = 2;
    l_conv_kernel_config.vfma_instruction = LIBXSMM_X86_INSTR_VPMADDWD;
    l_conv_kernel_config.vadd_instruction = LIBXSMM_X86_INSTR_VPADDD;
    l_conv_kernel_config.vbcst_instruction = LIBXSMM_X86_INSTR_VPBROADCASTD;
  } else if (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I8 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I16) {
    l_conv_kernel_config.vector_length_in = 64;
    l_conv_kernel_config.datatype_size_in = 1;
    l_conv_kernel_config.vector_length_out = 32;
    l_conv_kernel_config.datatype_size_out = 2;
    l_conv_kernel_config.vector_length_wt = 64;
    l_conv_kernel_config.datatype_size_wt = 1;
    l_conv_kernel_config.vfma_instruction = LIBXSMM_X86_INSTR_VPMADDUBSW;
    l_conv_kernel_config.vadd_instruction = LIBXSMM_X86_INSTR_VPADDW;
    l_conv_kernel_config.vbcst_instruction = LIBXSMM_X86_INSTR_VPBROADCASTW;
  } else if (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I8 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32) {
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
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }
  l_conv_kernel_config.vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
  l_conv_kernel_config.vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
  l_conv_kernel_config.prefetch_instruction = LIBXSMM_X86_INSTR_PREFETCHT0;
  /*l_conv_kernel_config.alu_mul_instruction = LIBXSMM_X86_INSTR_IMULQ;*/
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
  }
  if ( (i_conv_desc->format & LIBXSMM_DNN_TENSOR_FORMAT_RSCK) > 0 ) {
    l_conv_kernel_config.l_ld_ifm_fil = i_conv_desc->ifm_block * i_conv_desc->blocks_ifm;
    l_conv_kernel_config.l_ld_ofm_fil = i_conv_desc->ofm_block * i_conv_desc->blocks_ofm;
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
  if ( i_conv_desc->ofm_block % l_conv_kernel_config.vector_length_out != 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CONV_OFM_VEC );
    return;
  }

  /* initialize KW and KH unrolling */
  if (i_conv_desc->unroll_kw != 0) {
    l_kw_trips = i_conv_desc->kw;
  }
  if (i_conv_desc->unroll_kh != 0) {
    l_kh_trips = i_conv_desc->kh;
  }
#if 0
  if (i_conv_desc->datatype != i_conv_desc->datatype_itm) {
    prefetch_current_output = ( i_conv_desc->use_nts == 1 ) ? 0 : 1;
  } else {
    prefetch_current_output = 0;
  }
#endif
  if (i_conv_desc->datatype != i_conv_desc->datatype_itm) {
    prefetch_scale_factor = ( i_conv_desc->use_nts == 1 ) ? 1 : 0;
  } else {
    prefetch_scale_factor = 0;
  }


  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* open asm */
  libxsmm_x86_instruction_open_stream_convolution( io_generated_code, l_gp_reg_mapping.gp_reg_input,
      l_gp_reg_mapping.gp_reg_weight, l_gp_reg_mapping.gp_reg_output,
      l_gp_reg_mapping.gp_reg_input_pf, l_gp_reg_mapping.gp_reg_weight_pf,
      l_gp_reg_mapping.gp_reg_output_pf, i_arch );

  if ( i_conv_desc->compute_batch_stats > 0 && i_conv_desc->ifm_block != 3 ) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        l_conv_kernel_config.alu_mov_instruction,
        LIBXSMM_X86_GP_REG_RSP,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        48,
        l_gp_reg_mapping.gp_reg_help_2,
        0 );
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        l_conv_kernel_config.alu_mov_instruction,
        LIBXSMM_X86_GP_REG_RSP,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        56,
        l_gp_reg_mapping.gp_reg_help_3,
        0 );

    rsp_maxval_offset = 72;
  }

  if ( i_conv_desc->perform_relu_in_kernel == 1 ) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        l_conv_kernel_config.alu_mov_instruction,
        LIBXSMM_X86_GP_REG_RSP,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        48,
        l_gp_reg_mapping.gp_reg_help_2,
        0 );

    rsp_maxval_offset = 64;
  }

  /* load an additional temp register with 32 16bit 1s */
  if (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I8 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_help_0, 65537 );
    libxsmm_x86_instruction_push_reg( io_generated_code, l_gp_reg_mapping.gp_reg_help_0 );
    libxsmm_x86_instruction_vec_move( io_generated_code,
        l_conv_kernel_config.instruction_set,
        l_conv_kernel_config.vbcst_instruction,
        LIBXSMM_X86_GP_REG_RSP ,
        LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
        l_conv_kernel_config.vector_name, 3, 0, 1, 0 );
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

  if (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_BF16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_BF16) {
    libxsmm_generator_convolution_forward_load_output_bf16( io_generated_code, &l_gp_reg_mapping, &l_conv_kernel_config, i_conv_desc );
  } else {
    libxsmm_generator_convolution_forward_load_output( io_generated_code, &l_gp_reg_mapping, &l_conv_kernel_config, i_conv_desc );
  }

  /* loop over ifm1 blocks, begin */
  if ( i_conv_desc->blocks_ifm_blocking > 1 ) {
    unsigned int n_peeling = (i_conv_desc->extra_L2_prefetching == 0) ? 1 :LIBXSMM_MIN(i_conv_desc->blocks_ifm_blocking, i_conv_desc->lookahead);
    /* BLOCK 2: Run (n_peeling-1) iters to prefetch for extra L2 from pf* pointers if it is requested */
    unsigned int peel_index = 0;
    unsigned int out_reg_L2 = 0; /*atoi(getenv("REG_L2"));*/
    unsigned int out_reg_L1 = (unsigned int)-1; /*atoi(getenv("REG_L1"));*/
    unsigned int out_pf_L2 = 1; /*atoi(getenv("PF_L2"));*/
    unsigned int out_pf_L1 = (unsigned int)-1; /*atoi(getenv("PF_L1"));*/

    libxsmm_x86_instruction_alu_reg( io_generated_code, l_conv_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_input_pf, l_gp_reg_mapping.gp_reg_help_0);
    libxsmm_x86_instruction_alu_reg( io_generated_code, l_conv_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_weight_pf, l_gp_reg_mapping.gp_reg_help_1);

    /* BLOCK:1 Prefetching using the input/wt base pointers */
    if ( i_conv_desc->blocks_ifm_blocking-n_peeling > 0 ) {
      libxsmm_x86_instruction_alu_reg( io_generated_code, l_conv_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_input, l_gp_reg_mapping.gp_reg_input_pf);
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_add_instruction,
          l_gp_reg_mapping.gp_reg_input_pf, i_conv_desc->ifw_padded * i_conv_desc->ifh_padded  * l_conv_kernel_config.datatype_size_in * l_conv_kernel_config.l_ld_ifm_act * i_conv_desc->fm_lp_block);

      libxsmm_x86_instruction_alu_reg( io_generated_code, l_conv_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_weight, l_gp_reg_mapping.gp_reg_weight_pf);
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_add_instruction,
          l_gp_reg_mapping.gp_reg_weight_pf, i_conv_desc->weight_stride * l_conv_kernel_config.l_ld_ofm_act * l_conv_kernel_config.datatype_size_wt * l_conv_kernel_config.l_ld_ifm_act * i_conv_desc->fm_lp_block);

      if ( i_conv_desc->extra_L2_prefetching == 1 ) {
        if ( i_conv_desc->input_L2_prefetching == 1 ) {
          libxsmm_x86_instruction_alu_reg( io_generated_code, l_conv_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_input, l_gp_reg_mapping.gp_reg_input_pf_L2);
          libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_add_instruction,
              l_gp_reg_mapping.gp_reg_input_pf_L2, n_peeling * i_conv_desc->ifw_padded * i_conv_desc->ifh_padded  * l_conv_kernel_config.datatype_size_in * l_conv_kernel_config.l_ld_ifm_act * i_conv_desc->fm_lp_block);
        }

        libxsmm_x86_instruction_alu_reg( io_generated_code, l_conv_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_weight, l_gp_reg_mapping.gp_reg_weight_pf_L2);
        libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_add_instruction,
            l_gp_reg_mapping.gp_reg_weight_pf_L2, n_peeling * i_conv_desc->weight_stride * l_conv_kernel_config.l_ld_ofm_act * l_conv_kernel_config.datatype_size_wt * l_conv_kernel_config.l_ld_ifm_act * i_conv_desc->fm_lp_block);
      }

      /* open ifm1 block loop */
      libxsmm_generator_convolution_header_ifmOuter_loop(  io_generated_code, &l_loop_label_tracker,
          &l_conv_kernel_config, l_gp_reg_mapping.gp_reg_ifmOuter_loop );

#     include "template/kernel_repeat.tpl.c"

      /* adjust addresses, by moving to next ifm1 block */
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_add_instruction,
          l_gp_reg_mapping.gp_reg_input, i_conv_desc->ifw_padded*i_conv_desc->ifh_padded*
          l_conv_kernel_config.l_ld_ifm_act*i_conv_desc->fm_lp_block* l_conv_kernel_config.datatype_size_in );
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_add_instruction,
          l_gp_reg_mapping.gp_reg_input_pf, i_conv_desc->ifw_padded * i_conv_desc->ifh_padded*
          l_conv_kernel_config.l_ld_ifm_act*i_conv_desc->fm_lp_block* l_conv_kernel_config.datatype_size_in );
      libxsmm_x86_instruction_alu_imm( io_generated_code,  l_conv_kernel_config.alu_add_instruction,
          l_gp_reg_mapping.gp_reg_weight, i_conv_desc->weight_stride * i_conv_desc->kw*i_conv_desc->kh*l_conv_kernel_config.l_ld_ofm_act*
          l_conv_kernel_config.l_ld_ifm_act*i_conv_desc->fm_lp_block* l_conv_kernel_config.datatype_size_wt );
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_add_instruction,
          l_gp_reg_mapping.gp_reg_weight_pf, i_conv_desc->weight_stride * i_conv_desc->kw*i_conv_desc->kh*l_conv_kernel_config.l_ld_ofm_act*
          l_conv_kernel_config.l_ld_ifm_act*i_conv_desc->fm_lp_block* l_conv_kernel_config.datatype_size_wt );

      if ( i_conv_desc->extra_L2_prefetching == 1 ) {
        if ( i_conv_desc->input_L2_prefetching == 1 ) {
          libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_add_instruction,
              l_gp_reg_mapping.gp_reg_input_pf_L2, i_conv_desc->ifw_padded * i_conv_desc->ifh_padded*
              l_conv_kernel_config.l_ld_ifm_act*i_conv_desc->fm_lp_block* l_conv_kernel_config.datatype_size_in );
        }
        libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_add_instruction,
            l_gp_reg_mapping.gp_reg_weight_pf_L2, i_conv_desc->weight_stride * i_conv_desc->kw*i_conv_desc->kh*l_conv_kernel_config.l_ld_ofm_act*
            l_conv_kernel_config.l_ld_ifm_act*i_conv_desc->fm_lp_block* l_conv_kernel_config.datatype_size_wt );
      }

      /* close ifm1 block loop */
      libxsmm_generator_convolution_footer_ifmOuter_loop(  io_generated_code, &l_loop_label_tracker,
          &l_conv_kernel_config, l_gp_reg_mapping.gp_reg_ifmOuter_loop, i_conv_desc->blocks_ifm_blocking-n_peeling );
    }

    if (n_peeling-1 > 0) {
      if ( i_conv_desc->input_L2_prefetching == 1 ) {
        libxsmm_x86_instruction_alu_reg( io_generated_code, l_conv_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_input_pf_L2);
      }
      libxsmm_x86_instruction_alu_reg( io_generated_code, l_conv_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_weight_pf_L2);

      for (peel_index = 0; peel_index < n_peeling-1; peel_index++) {

        /* Prefetch current output block to be loaded soon...  */
        if ( (i_conv_desc->use_nts == 0) && (peel_index == out_reg_L2) ) {
          for ( i_out_pf = 0; i_out_pf < i_conv_desc->ofh_rb; i_out_pf++ ) {
            for ( j_out_pf = 0; j_out_pf < i_conv_desc->ofw_rb; j_out_pf++ ) {
              libxsmm_x86_instruction_prefetch( io_generated_code,
                  LIBXSMM_X86_INSTR_PREFETCHT1,
                  l_gp_reg_mapping.gp_reg_output,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  (i_out_pf * i_conv_desc->ofw_padded *  i_conv_desc->stride_h_store * l_conv_kernel_config.l_ld_ofm_act * l_conv_kernel_config.datatype_size_out) +
                  (j_out_pf * i_conv_desc->stride_w_store * l_conv_kernel_config.l_ld_ofm_act * l_conv_kernel_config.datatype_size_out ) );

            }
          }
        }

        if ( (i_conv_desc->use_nts == 0) && (peel_index == out_reg_L1) ) {
          for ( i_out_pf = 0; i_out_pf < i_conv_desc->ofh_rb; i_out_pf++ ) {
            for ( j_out_pf = 0; j_out_pf < i_conv_desc->ofw_rb; j_out_pf++ ) {
              libxsmm_x86_instruction_prefetch( io_generated_code,
                  LIBXSMM_X86_INSTR_PREFETCHT0,
                  l_gp_reg_mapping.gp_reg_output,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  (i_out_pf * i_conv_desc->ofw_padded *  i_conv_desc->stride_h_store * l_conv_kernel_config.l_ld_ofm_act * l_conv_kernel_config.datatype_size_out) +
                  (j_out_pf * i_conv_desc->stride_w_store * l_conv_kernel_config.l_ld_ofm_act * l_conv_kernel_config.datatype_size_out ) );

            }
          }
        }

        if ( (i_conv_desc->use_nts == 0) && (peel_index == out_pf_L2) ) {
          for ( i_out_pf = 0; i_out_pf < i_conv_desc->ofh_rb; i_out_pf++ ) {
            for ( j_out_pf = 0; j_out_pf < i_conv_desc->ofw_rb; j_out_pf++ ) {
              libxsmm_x86_instruction_prefetch( io_generated_code,
                  LIBXSMM_X86_INSTR_PREFETCHT1,
                  l_gp_reg_mapping.gp_reg_output_pf,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  (i_out_pf * i_conv_desc->ofw_padded *  i_conv_desc->stride_h_store * l_conv_kernel_config.l_ld_ofm_act * l_conv_kernel_config.datatype_size_out) +
                  (j_out_pf * i_conv_desc->stride_w_store * l_conv_kernel_config.l_ld_ofm_act * l_conv_kernel_config.datatype_size_out ) );

            }
          }
        }

        if ( (i_conv_desc->use_nts == 0) && (peel_index == out_pf_L1) ) {
          for ( i_out_pf = 0; i_out_pf < i_conv_desc->ofh_rb; i_out_pf++ ) {
            for ( j_out_pf = 0; j_out_pf < i_conv_desc->ofw_rb; j_out_pf++ ) {
              libxsmm_x86_instruction_prefetch( io_generated_code,
                  LIBXSMM_X86_INSTR_PREFETCHT0,
                  l_gp_reg_mapping.gp_reg_output_pf,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  (i_out_pf * i_conv_desc->ofw_padded *  i_conv_desc->stride_h_store * l_conv_kernel_config.l_ld_ofm_act * l_conv_kernel_config.datatype_size_out) +
                  (j_out_pf * i_conv_desc->stride_w_store * l_conv_kernel_config.l_ld_ofm_act * l_conv_kernel_config.datatype_size_out ) );

            }
          }
        }

        if ( (i_conv_desc->compute_batch_stats > 0) && (peel_index == 2) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
              LIBXSMM_X86_INSTR_PREFETCHT1,
              l_gp_reg_mapping.gp_reg_help_2,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              0);

          libxsmm_x86_instruction_prefetch( io_generated_code,
              LIBXSMM_X86_INSTR_PREFETCHT1,
              l_gp_reg_mapping.gp_reg_help_3,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              0);
#ifdef FP64_BN_STATS
          libxsmm_x86_instruction_prefetch( io_generated_code,
              LIBXSMM_X86_INSTR_PREFETCHT1,
              l_gp_reg_mapping.gp_reg_help_2,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              64);

          libxsmm_x86_instruction_prefetch( io_generated_code,
              LIBXSMM_X86_INSTR_PREFETCHT1,
              l_gp_reg_mapping.gp_reg_help_3,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              64);
#endif
        }


        if ( (i_conv_desc->perform_relu_in_kernel == 1) && (peel_index == 2) ) {
          unsigned int i, j, store_offset;
          /* Prefetch to L2 all "regular inputs" based on passed pointer (now in help2 register) */
          for (i = 0; i < i_conv_desc->ofh_rb; i++) {
            for ( j = 0; j < i_conv_desc->ofw_rb; j++ ) {
              store_offset = ((i * i_conv_desc->stride_h_store) * i_conv_desc->ofw_padded + j * i_conv_desc->stride_w_store) * l_conv_kernel_config.vector_length_out * l_conv_kernel_config.datatype_size_out;

              libxsmm_x86_instruction_prefetch( io_generated_code,
                  LIBXSMM_X86_INSTR_PREFETCHT1,
                  l_gp_reg_mapping.gp_reg_help_2,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  store_offset);

            }
          }
        }


        if ( (prefetch_scale_factor == 1)  && (peel_index == 2)) {
          libxsmm_x86_instruction_alu_reg( io_generated_code, l_conv_kernel_config.alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, l_gp_reg_mapping.gp_reg_help_2);
          libxsmm_x86_instruction_alu_mem( io_generated_code,
              l_conv_kernel_config.alu_mov_instruction,
              l_gp_reg_mapping.gp_reg_help_2,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              rsp_maxval_offset-8,
              l_gp_reg_mapping.gp_reg_help_2,
              0 );

          libxsmm_x86_instruction_prefetch( io_generated_code,
              LIBXSMM_X86_INSTR_PREFETCHT1,
              l_gp_reg_mapping.gp_reg_help_2,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              0);
        }

        if ((i_conv_desc->compute_max == 1) && (peel_index == 2)) {
          libxsmm_x86_instruction_alu_reg( io_generated_code, l_conv_kernel_config.alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, l_gp_reg_mapping.gp_reg_help_2);
          libxsmm_x86_instruction_alu_mem( io_generated_code,
              l_conv_kernel_config.alu_mov_instruction,
              l_gp_reg_mapping.gp_reg_help_2,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              rsp_maxval_offset,
              l_gp_reg_mapping.gp_reg_help_2,
              0 );

          libxsmm_x86_instruction_prefetch( io_generated_code,
              LIBXSMM_X86_INSTR_PREFETCHT1,
              l_gp_reg_mapping.gp_reg_help_2,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              0);
        }

#if 0
        if ( (prefetch_current_output == 1) && (peel_index == 2) ) {
          unsigned int i, j, store_offset;
          /* Prefetch to L2 current output in case of LP convs */
          for (i = 0; i < i_conv_desc->ofh_rb; i++) {
            for ( j = 0; j < i_conv_desc->ofw_rb; j++ ) {
              store_offset = ((i * i_conv_desc->stride_h_store) * i_conv_desc->ofw_padded + j * i_conv_desc->stride_w_store) * l_conv_kernel_config.vector_length_out * l_conv_kernel_config.datatype_size_out;
              libxsmm_x86_instruction_prefetch( io_generated_code,
                  LIBXSMM_X86_INSTR_PREFETCHT1,
                  l_gp_reg_mapping.gp_reg_output,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  store_offset);

            }
          }
        }
#endif

#        include "template/kernel_repeat.tpl.c"

        /* adjust addresses, by moving to next ifm1 block */
        libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_add_instruction,
            l_gp_reg_mapping.gp_reg_input, i_conv_desc->ifw_padded*i_conv_desc->ifh_padded*
            l_conv_kernel_config.l_ld_ifm_act*i_conv_desc->fm_lp_block* l_conv_kernel_config.datatype_size_in );
        libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_add_instruction,
            l_gp_reg_mapping.gp_reg_input_pf, i_conv_desc->ifw_padded * i_conv_desc->ifh_padded*
            l_conv_kernel_config.l_ld_ifm_act*i_conv_desc->fm_lp_block* l_conv_kernel_config.datatype_size_in );
        libxsmm_x86_instruction_alu_imm( io_generated_code,  l_conv_kernel_config.alu_add_instruction,
            l_gp_reg_mapping.gp_reg_weight, i_conv_desc->weight_stride * i_conv_desc->kw*i_conv_desc->kh*l_conv_kernel_config.l_ld_ofm_act*
            l_conv_kernel_config.l_ld_ifm_act*i_conv_desc->fm_lp_block* l_conv_kernel_config.datatype_size_wt );
        libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_add_instruction,
            l_gp_reg_mapping.gp_reg_weight_pf, i_conv_desc->weight_stride * i_conv_desc->kw*i_conv_desc->kh*l_conv_kernel_config.l_ld_ofm_act*
            l_conv_kernel_config.l_ld_ifm_act*i_conv_desc->fm_lp_block* l_conv_kernel_config.datatype_size_wt );

        if ( i_conv_desc->input_L2_prefetching == 1 ) {
          libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_add_instruction,
              l_gp_reg_mapping.gp_reg_input_pf_L2, i_conv_desc->ifw_padded * i_conv_desc->ifh_padded*
              l_conv_kernel_config.l_ld_ifm_act*i_conv_desc->fm_lp_block* l_conv_kernel_config.datatype_size_in );
        }
        libxsmm_x86_instruction_alu_imm( io_generated_code, l_conv_kernel_config.alu_add_instruction,
            l_gp_reg_mapping.gp_reg_weight_pf_L2, i_conv_desc->weight_stride * i_conv_desc->kw*i_conv_desc->kh*l_conv_kernel_config.l_ld_ofm_act*
            l_conv_kernel_config.l_ld_ifm_act*i_conv_desc->fm_lp_block* l_conv_kernel_config.datatype_size_wt );
      }
    }



    /* BLOCK 3: Last peeled iteration  */
    libxsmm_x86_instruction_alu_reg( io_generated_code, l_conv_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_input_pf);
    libxsmm_x86_instruction_alu_reg( io_generated_code, l_conv_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_help_1, l_gp_reg_mapping.gp_reg_weight_pf);

    /* Prefetch output block to be loaded after the last microkernel call...  */
    if ( (i_conv_desc->use_nts == 0) && (peel_index == out_pf_L1) ) {
      for ( i_out_pf = 0; i_out_pf < i_conv_desc->ofh_rb; i_out_pf++ ) {
        for ( j_out_pf = 0; j_out_pf < i_conv_desc->ofw_rb; j_out_pf++ ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
              LIBXSMM_X86_INSTR_PREFETCHT0,
              l_gp_reg_mapping.gp_reg_output_pf,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (i_out_pf * i_conv_desc->ofw_padded *  i_conv_desc->stride_h_store * l_conv_kernel_config.l_ld_ofm_act * l_conv_kernel_config.datatype_size_out) +
              (j_out_pf * i_conv_desc->stride_w_store * l_conv_kernel_config.l_ld_ofm_act * l_conv_kernel_config.datatype_size_out ) );

        }
      }
    }

#if 0
    if ( i_conv_desc->perform_relu_in_kernel == 1 ) {
      unsigned int i, j, store_offset;
      /* Prefetch to L2 all "regular inputs" based on passed pointer (now in help2 register) */
      for (i = 0; i < i_conv_desc->ofh_rb; i++) {
        for ( j = 0; j < i_conv_desc->ofw_rb; j++ ) {
          store_offset = ((i * i_conv_desc->stride_h_store) * i_conv_desc->ofw_padded + j * i_conv_desc->stride_w_store) * l_conv_kernel_config.vector_length_out * l_conv_kernel_config.datatype_size_out;
          libxsmm_x86_instruction_prefetch( io_generated_code,
              LIBXSMM_X86_INSTR_PREFETCHT0,
              l_gp_reg_mapping.gp_reg_help_2,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              store_offset);

        }
      }
    }
#endif

#   include "template/kernel_repeat.tpl.c"
  } else {
#   include "template/kernel_repeat.tpl.c"
  }

  /* store outputs */
  if (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_BF16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_BF16) {
    libxsmm_generator_convolution_forward_store_output_bf16( io_generated_code, &l_gp_reg_mapping, &l_conv_kernel_config, i_conv_desc );
  } else {
    libxsmm_generator_convolution_forward_store_output( io_generated_code, &l_gp_reg_mapping, &l_conv_kernel_config, i_conv_desc );
  }

  /* close asm */
  libxsmm_x86_instruction_close_stream_convolution( io_generated_code, i_arch );
}


  LIBXSMM_API_INTERN
void libxsmm_generator_convolution_forward_avx512_ifmloop( libxsmm_generated_code*                           io_generated_code,
    const libxsmm_convolution_forward_gp_reg_mapping* i_gp_reg_mapping,
    const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
    const libxsmm_convolution_forward_descriptor*     i_conv_desc,
    const unsigned int                                i_kw_unroll )
{
  if (i_conv_desc->ofh_rb == 2) {
    libxsmm_generator_convolution_forward_avx512_ifmloop_two_rows( io_generated_code, i_gp_reg_mapping,
        i_conv_kernel_config, i_conv_desc, i_kw_unroll );
  } else {
    libxsmm_generator_convolution_forward_avx512_ifmloop_one_row( io_generated_code, i_gp_reg_mapping,
        i_conv_kernel_config, i_conv_desc, i_kw_unroll );
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_convolution_forward_avx512_ifmloop_one_row( libxsmm_generated_code*                           io_generated_code,
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
  unsigned int l_w;
  unsigned int l_reg_block;
  unsigned int l_accs;
  unsigned int l_filter_pos = 0;
  unsigned int l_compute_instr = 0;
  unsigned int l_prefetch_input_index = 0;
  unsigned int prefetch_type_input = LIBXSMM_X86_INSTR_PREFETCHT0;
  unsigned int prefetch_type_weight = LIBXSMM_X86_INSTR_PREFETCHT0;
  unsigned int input_pf_L2_bound =  i_conv_desc->ofw_rb * 64;
  unsigned int input_pf_L2_offset = 0;
  unsigned int step_size = 0;

  if ( i_conv_desc->kw != 1   ) {
    prefetch_type_weight = LIBXSMM_X86_INSTR_PREFETCHT1;
  }

  /* apply k blocking */
  if ( i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_KNM ) {
    if (i_conv_desc->ifm_block > 3) {
      step_size = 4;
    } else {
      step_size =  i_conv_desc->ifm_block;
    }
  } else {
    step_size = 1;
  }

  for ( l_k = 0; l_k < i_conv_desc->ifm_block*i_kw_unroll; l_k+=step_size ) {
    /* if we are not in LIBXSMM storage format, there are jumps */
    if ( (l_k > 0) && (l_k % i_conv_desc->ifm_block == 0) ) {
      /* input pointer advance */
      if ( (i_conv_desc->format & LIBXSMM_DNN_TENSOR_FORMAT_NHWC) > 0 ) {
        l_displacement_k += (i_conv_kernel_config->l_ld_ifm_act - i_conv_desc->ifm_block) * i_conv_desc->fm_lp_block;
      }
    }

    /* load the four source registers, we cannot perform a pipeline as in case of sfma */
    for ( l_w = 0; l_w < step_size; l_w++ ) {
      if (((l_k+l_w)%i_conv_desc->ifm_block == 0) && (l_k > 0) ) {
        l_filter_pos += (i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block) * i_conv_desc->fm_lp_block;
      }
      if (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_BF16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_BF16) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_conv_kernel_config->instruction_set,
            i_conv_kernel_config->vmove_instruction,
            i_gp_reg_mapping->gp_reg_weight,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            l_filter_pos * i_conv_kernel_config->l_ld_ofm_fil * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block,
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
            i_gp_reg_mapping->gp_reg_weight,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            l_filter_pos * i_conv_kernel_config->l_ld_ofm_fil * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block,
            i_conv_kernel_config->vector_name, l_w,
            0, 1, 0 );
      }
      l_filter_pos++;
    }

    /* we QMADD for 3 input channels, 4th channel needs to be set to zero */
    if (step_size == 3) {
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
          i_conv_kernel_config->instruction_set,
          i_conv_kernel_config->vxor_instruction,
          i_conv_kernel_config->vector_name, 3, 3, 3);
    }

    /* apply additional register block to hide FMA latencies */
    l_reg_block = i_conv_desc->ofw_rb;
    if ( i_conv_desc->ofw_rb <= 14 && step_size > 1) {
      l_accs = (i_conv_desc->ofw_rb < 9) ? 3 : 2;
      l_reg_block = (((l_k%(4*l_accs))/4)+1)*i_conv_desc->ofw_rb;
    }

    /* compute vectorwidth (A) * column broadcast (B) */
    for ( l_n = 0; l_n < i_conv_desc->ofw_rb; l_n++) {
      /* @TODO, we need to fix for non-LIBXSMM format */
      l_input_reg = i_gp_reg_mapping->gp_reg_input;
      l_input_idx = LIBXSMM_X86_GP_REG_UNDEF;
      l_scale = 0;
      l_disp = (l_k * i_conv_kernel_config->datatype_size_in * i_conv_desc->fm_lp_block)
        + (l_n * i_conv_kernel_config->datatype_size_in * i_conv_desc->stride_w * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block);

      if (step_size == 1) {
        if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_F32 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
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
              i_conv_kernel_config->vector_reg_count - l_reg_block + l_n );
        } else if ( (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32) ||
            (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32)) {
          if ( i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_CORE ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_conv_kernel_config->instruction_set,
                i_conv_kernel_config->vbcst_instruction,
                l_input_reg,
                l_input_idx, l_scale,
                l_disp,
                i_conv_kernel_config->vector_name,
                1, 0, 1, 0 );
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                i_conv_kernel_config->instruction_set,
                i_conv_kernel_config->vfma_instruction,
                i_conv_kernel_config->vector_name,
                0,
                1,
                1 );
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                i_conv_kernel_config->instruction_set,
                i_conv_kernel_config->vadd_instruction,
                i_conv_kernel_config->vector_name,
                1,
                i_conv_kernel_config->vector_reg_count - l_reg_block + l_n,
                i_conv_kernel_config->vector_reg_count - l_reg_block + l_n );
          } else {
            libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                i_conv_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPDPWSSDS,
                1,
                l_input_reg,
                l_input_idx,
                l_scale,
                l_disp,
                i_conv_kernel_config->vector_name,
                0,
                i_conv_kernel_config->vector_reg_count - l_reg_block + l_n );
          }
        } else if (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_BF16) {
          if ( i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_CORE ) {
            /* bcast  */
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_conv_kernel_config->instruction_set,
                i_conv_kernel_config->vbcst_instruction,
                l_input_reg,
                l_input_idx, l_scale,
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
                i_conv_kernel_config->vector_reg_count - l_reg_block + l_n );

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
                i_conv_kernel_config->vector_reg_count - l_reg_block + l_n );
          }
        } else if ( (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I8  && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32
              && (i_conv_desc->option & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0) ) {
          /* broadcast in quadruples of 8 bit values */
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_conv_kernel_config->instruction_set,
              i_conv_kernel_config->vbcst_instruction,
              l_input_reg,
              l_input_idx, l_scale,
              l_disp,
              i_conv_kernel_config->vector_name,
              1, 0, 1, 0 );

          /* 8/16bit integer MADD with horizontal add */
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
              i_conv_kernel_config->instruction_set,
              i_conv_kernel_config->vfma_instruction,
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
              i_conv_kernel_config->vadd_instruction,
              i_conv_kernel_config->vector_name,
              2,
              i_conv_kernel_config->vector_reg_count - l_reg_block + l_n,
              i_conv_kernel_config->vector_reg_count - l_reg_block + l_n  );

        } else {
          /* shouldn't happen */
        }
      } else {
        if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_F32 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
          l_compute_instr = LIBXSMM_X86_INSTR_V4FMADDPS;
        } else if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32 ) {
          l_compute_instr = LIBXSMM_X86_INSTR_VP4DPWSSDS;
        } else if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
          l_compute_instr = LIBXSMM_X86_INSTR_VP4DPWSSDS;
        }

        libxsmm_x86_instruction_vec_compute_qfma( io_generated_code,
            i_conv_kernel_config->instruction_set,
            l_compute_instr,
            l_input_reg,
            l_input_idx,
            l_scale,
            l_disp,
            i_conv_kernel_config->vector_name,
            0,
            i_conv_kernel_config->vector_reg_count - l_reg_block + l_n );
      }

      if (l_k % 4 == 0) {
        /* handle prefetches for input and weights */
        if ( (l_n % 2 == 1) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) && (l_prefetch_input_index < i_conv_desc->ofw_rb) ) {
          unsigned int pf_offset = l_prefetch_input_index * i_conv_desc->stride_w * i_conv_kernel_config->datatype_size_in * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block;
          if (pf_offset % 64 == 0) {
            libxsmm_x86_instruction_prefetch( io_generated_code,
                prefetch_type_input,
                i_gp_reg_mapping->gp_reg_input_pf,
                LIBXSMM_X86_GP_REG_UNDEF,
                0,
                (l_prefetch_input_index * i_conv_desc->stride_w) * i_conv_kernel_config->datatype_size_in
                * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block );
          }
          l_prefetch_input_index++;
        }

        if ( (l_n == 4 || l_n == 8 )  && (i_conv_desc->extra_L2_prefetching == 1) && (i_conv_desc->input_L2_prefetching == 1) ) {
          if ( input_pf_L2_offset < input_pf_L2_bound ) {
            libxsmm_x86_instruction_prefetch( io_generated_code,
                LIBXSMM_X86_INSTR_PREFETCHT1,
                i_gp_reg_mapping->gp_reg_input_pf_L2,
                LIBXSMM_X86_GP_REG_UNDEF,
                0,
                input_pf_L2_offset);
            input_pf_L2_offset += 64;
          }
        }
      }

      assert(0 != i_conv_desc->ifm_block);
      if ( (l_n == 0) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) ) {
        int l_filter_weight_pos = l_k + ((l_k/i_conv_desc->ifm_block)*(i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block)*i_conv_desc->fm_lp_block);
        libxsmm_x86_instruction_prefetch( io_generated_code,
            prefetch_type_weight /*i_conv_kernel_config->prefetch_instruction*/,
            i_gp_reg_mapping->gp_reg_weight_pf,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            l_filter_weight_pos * i_conv_kernel_config->l_ld_ofm_fil * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block );
      }
      if ((step_size > 1) && (step_size <= 4)) {
        if ( (l_n == 2) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) ) {
          int l_filter_weight_pos = (l_k+1) + (((l_k+1)/i_conv_desc->ifm_block)*(i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block)*i_conv_desc->fm_lp_block );
          libxsmm_x86_instruction_prefetch( io_generated_code,
              prefetch_type_weight /*i_conv_kernel_config->prefetch_instruction*/,
              i_gp_reg_mapping->gp_reg_weight_pf,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              l_filter_weight_pos * i_conv_kernel_config->l_ld_ofm_fil * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block );
        }
        if ( (l_n == 4) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) ) {
          int l_filter_weight_pos = (l_k+2) + (((l_k+2)/i_conv_desc->ifm_block)*(i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block)*i_conv_desc->fm_lp_block);
          libxsmm_x86_instruction_prefetch( io_generated_code,
              prefetch_type_weight /*i_conv_kernel_config->prefetch_instruction*/,
              i_gp_reg_mapping->gp_reg_weight_pf,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              l_filter_weight_pos * i_conv_kernel_config->l_ld_ofm_fil * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block );
        }
      }
      if (step_size == 4) {
        if ( (l_n == 6) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) ) {
          int l_filter_weight_pos = (l_k+3) + (((l_k+3)/i_conv_desc->ifm_block)*(i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block)*i_conv_desc->fm_lp_block);
          libxsmm_x86_instruction_prefetch( io_generated_code,
              prefetch_type_weight /*i_conv_kernel_config->prefetch_instruction*/,
              i_gp_reg_mapping->gp_reg_weight_pf,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              l_filter_weight_pos * i_conv_kernel_config->l_ld_ofm_fil * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block );
        }
      }

      if ( i_conv_desc->extra_L2_prefetching == 1 ) {
        if ( (l_n == 8) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) ) {
          int l_filter_weight_pos = l_k + ((l_k/i_conv_desc->ifm_block)*(i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block)*i_conv_desc->fm_lp_block);
          libxsmm_x86_instruction_prefetch( io_generated_code,
              LIBXSMM_X86_INSTR_PREFETCHT1 /*i_conv_kernel_config->prefetch_instruction*/,
              i_gp_reg_mapping->gp_reg_weight_pf_L2,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              l_filter_weight_pos * i_conv_kernel_config->l_ld_ofm_fil * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block );
        }
        if ((step_size > 1) && (step_size <= 4)) {
          if ( (l_n == 10) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) ) {
            int l_filter_weight_pos = (l_k+1) + (((l_k+1)/i_conv_desc->ifm_block)*(i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block)*i_conv_desc->fm_lp_block );
            libxsmm_x86_instruction_prefetch( io_generated_code,
                LIBXSMM_X86_INSTR_PREFETCHT1 /*i_conv_kernel_config->prefetch_instruction*/,
                i_gp_reg_mapping->gp_reg_weight_pf_L2,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                l_filter_weight_pos * i_conv_kernel_config->l_ld_ofm_fil * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block );
          }
          if ( (l_n == 12) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) ) {
            int l_filter_weight_pos = (l_k+2) + (((l_k+2)/i_conv_desc->ifm_block)*(i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block)*i_conv_desc->fm_lp_block);
            libxsmm_x86_instruction_prefetch( io_generated_code,
                LIBXSMM_X86_INSTR_PREFETCHT1 /*i_conv_kernel_config->prefetch_instruction*/,
                i_gp_reg_mapping->gp_reg_weight_pf_L2,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                l_filter_weight_pos * i_conv_kernel_config->l_ld_ofm_fil * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block );
          }
        }
        if (step_size == 4) {
          if ( (l_n == 14) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) ) {
            int l_filter_weight_pos = (l_k+3) + (((l_k+3)/i_conv_desc->ifm_block)*(i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block)*i_conv_desc->fm_lp_block);
            libxsmm_x86_instruction_prefetch( io_generated_code,
                LIBXSMM_X86_INSTR_PREFETCHT1 /*i_conv_kernel_config->prefetch_instruction*/,
                i_gp_reg_mapping->gp_reg_weight_pf_L2,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                l_filter_weight_pos * i_conv_kernel_config->l_ld_ofm_fil * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block );
          }
        }
      }
    }
    if (step_size == 1) {
      l_displacement_k++;
    } else {
      l_displacement_k+=4;
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_convolution_forward_avx512_ifmloop_two_rows( libxsmm_generated_code*                           io_generated_code,
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
  unsigned int l_w;
  unsigned int l_m;
  unsigned int l_filter_pos = 0;
  unsigned int l_compute_instr = 0;
  unsigned int l_prefetch_input_index_w = 0;
  unsigned int l_prefetch_input_index_h = 0;
  unsigned int prefetch_type_input = LIBXSMM_X86_INSTR_PREFETCHT0;
  unsigned int prefetch_type_weight = LIBXSMM_X86_INSTR_PREFETCHT0;
  unsigned int input_pf_L2_bound =  i_conv_desc->ofh_rb * i_conv_desc->ofw_rb * 64;
  unsigned int input_pf_L2_offset = 0;
  unsigned int step_size = 0;

  if ( i_conv_desc->kw != 1) {
    prefetch_type_weight = LIBXSMM_X86_INSTR_PREFETCHT1;
  }

  /* apply k blocking */
  if ( i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_KNM ) {
    if (i_conv_desc->ifm_block > 3) {
      step_size = 4;
    } else {
      step_size =  i_conv_desc->ifm_block;
    }
  } else {
    step_size = 1;
  }

  /* apply k blocking */
  for ( l_k = 0; l_k < i_conv_desc->ifm_block*i_kw_unroll; l_k+=step_size ) {
    /* if we are not in LIBXSMM storage format, there are jumps */
    if ( (l_k > 0) && (l_k % i_conv_desc->ifm_block == 0) ) {
      /* input pointer advance */
      if ( (i_conv_desc->format & LIBXSMM_DNN_TENSOR_FORMAT_NHWC) > 0 ) {
        l_displacement_k += (i_conv_kernel_config->l_ld_ifm_act - i_conv_desc->ifm_block) * i_conv_desc->fm_lp_block;
      }
    }

    /* load the four source registers, we cannot perform a pipeline as in case of sfma */
    for ( l_w = 0; l_w < step_size; l_w++ ) {
      if (((l_k+l_w)%i_conv_desc->ifm_block == 0) && (l_k > 0) ) {
        l_filter_pos += (i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block) * i_conv_desc->fm_lp_block;
      }

      if (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_BF16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_BF16) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_conv_kernel_config->instruction_set,
            i_conv_kernel_config->vmove_instruction,
            i_gp_reg_mapping->gp_reg_weight,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            l_filter_pos* i_conv_kernel_config->l_ld_ofm_fil * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block,
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
            i_gp_reg_mapping->gp_reg_weight,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            l_filter_pos* i_conv_kernel_config->l_ld_ofm_fil * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block,
            i_conv_kernel_config->vector_name, l_w,
            0, 1, 0 );
      }
      l_filter_pos++;
    }

    /* we QMADD for 3 input channels, 4th channel needs to be set to zero */
    if (step_size == 3) {
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
          i_conv_kernel_config->instruction_set,
          i_conv_kernel_config->vxor_instruction,
          i_conv_kernel_config->vector_name, 3, 3, 3);
    }

    /* compute vectorwidth (A) * column broadcast (B) */
    for ( l_m = 0; l_m < i_conv_desc->ofh_rb; l_m++) {
      l_prefetch_input_index_w = 0;
      for ( l_n = 0; l_n < i_conv_desc->ofw_rb; l_n++) {
        /* @TODO, we need to fix for non-LIBXSMM format */
        l_input_reg = i_gp_reg_mapping->gp_reg_input;
        l_input_idx = LIBXSMM_X86_GP_REG_UNDEF;
        l_scale = 0;
        l_disp = (l_k*i_conv_kernel_config->datatype_size_in*i_conv_desc->fm_lp_block)
          + (l_n * i_conv_kernel_config->datatype_size_in * i_conv_desc->stride_w * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block)
          + (l_m * i_conv_desc->stride_h * i_conv_desc->ifw_padded * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block * i_conv_kernel_config->datatype_size_in);

        if (step_size == 1) {
          if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_F32 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
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
                i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*i_conv_desc->ofh_rb) + l_n + (l_m*i_conv_desc->ofw_rb) );
          } else if ( (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32) ||
              (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32) ) {
            if ( i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_CORE ) {
              libxsmm_x86_instruction_vec_move( io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  i_conv_kernel_config->vbcst_instruction,
                  l_input_reg,
                  l_input_idx, l_scale,
                  l_disp,
                  i_conv_kernel_config->vector_name,
                  1, 0, 1, 0 );
              libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  i_conv_kernel_config->vfma_instruction,
                  i_conv_kernel_config->vector_name,
                  0,
                  1,
                  1 );
              libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  i_conv_kernel_config->vadd_instruction,
                  i_conv_kernel_config->vector_name,
                  1,
                  i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*i_conv_desc->ofh_rb) + l_n + (l_m*i_conv_desc->ofw_rb),
                  i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*i_conv_desc->ofh_rb) + l_n + (l_m*i_conv_desc->ofw_rb) );
            } else {
              libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  LIBXSMM_X86_INSTR_VPDPWSSDS,
                  1,
                  l_input_reg,
                  l_input_idx,
                  l_scale,
                  l_disp,
                  i_conv_kernel_config->vector_name,
                  0,
                  i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*i_conv_desc->ofh_rb) + l_n + (l_m*i_conv_desc->ofw_rb) );
            }
          } else if (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_BF16) {
            if ( i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_CORE ) {
              /* bcast  */
              libxsmm_x86_instruction_vec_move( io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  i_conv_kernel_config->vbcst_instruction,
                  l_input_reg,
                  l_input_idx, l_scale,
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
                  i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*i_conv_desc->ofh_rb) + l_n + (l_m*i_conv_desc->ofw_rb) );

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
                  i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*i_conv_desc->ofh_rb) + l_n + (l_m*i_conv_desc->ofw_rb) );
            }
          } else if ( (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I8  && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32
                && (i_conv_desc->option & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0) ) {
            /* broadcast in quadruples of 8 bit values */
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_conv_kernel_config->instruction_set,
                i_conv_kernel_config->vbcst_instruction,
                l_input_reg,
                l_input_idx, l_scale,
                l_disp,
                i_conv_kernel_config->vector_name,
                1, 0, 1, 0 );

            /* 8/16bit integer MADD with horizontal add */
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                i_conv_kernel_config->instruction_set,
                i_conv_kernel_config->vfma_instruction,
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
                i_conv_kernel_config->vadd_instruction,
                i_conv_kernel_config->vector_name,
                2,
                i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*i_conv_desc->ofh_rb) + l_n + (l_m*i_conv_desc->ofw_rb),
                i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*i_conv_desc->ofh_rb) + l_n + (l_m*i_conv_desc->ofw_rb) );
          } else {
            /* shouldn't happen */
          }
        } else {
          /* depending on datatype emit the needed FMA(-sequence) */
          if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_F32 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
            l_compute_instr = LIBXSMM_X86_INSTR_V4FMADDPS;
          } else if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32 ) {
            l_compute_instr = LIBXSMM_X86_INSTR_VP4DPWSSDS;
          } else if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
            l_compute_instr = LIBXSMM_X86_INSTR_VP4DPWSSDS;
          }

          libxsmm_x86_instruction_vec_compute_qfma( io_generated_code,
              i_conv_kernel_config->instruction_set,
              l_compute_instr,
              l_input_reg,
              l_input_idx,
              l_scale,
              l_disp,
              i_conv_kernel_config->vector_name,
              0,
              i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*i_conv_desc->ofh_rb) + l_n + (l_m*i_conv_desc->ofw_rb) );
        }

        if ( l_k % 4 == 0 ) {
          /* handle prefetches for input and weights */
          if ( ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) && (l_prefetch_input_index_w < i_conv_desc->ofw_rb)  && (l_prefetch_input_index_h < i_conv_desc->ofh_rb)  ) {
            unsigned int pf_offset = (l_prefetch_input_index_w * i_conv_desc->stride_w + l_prefetch_input_index_h * i_conv_desc->stride_h * i_conv_desc->ifw_padded)  * i_conv_kernel_config->datatype_size_in * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block;
            if (pf_offset % 64 == 0) {
              libxsmm_x86_instruction_prefetch( io_generated_code,
                  prefetch_type_input,
                  i_gp_reg_mapping->gp_reg_input_pf,
                  LIBXSMM_X86_GP_REG_UNDEF,
                  0,
                  (l_prefetch_input_index_w * i_conv_desc->stride_w + l_prefetch_input_index_h * i_conv_desc->stride_h * i_conv_desc->ifw_padded ) * i_conv_kernel_config->datatype_size_in *
                  i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block );
            }
            l_prefetch_input_index_w++;
          }

          if ( (l_n == 4 || l_n == 8 )  && (i_conv_desc->extra_L2_prefetching == 1)  && (i_conv_desc->input_L2_prefetching == 1) ) {
            if ( input_pf_L2_offset < input_pf_L2_bound ) {
              libxsmm_x86_instruction_prefetch( io_generated_code,
                  LIBXSMM_X86_INSTR_PREFETCHT1,
                  i_gp_reg_mapping->gp_reg_input_pf_L2,
                  LIBXSMM_X86_GP_REG_UNDEF,
                  0,
                  input_pf_L2_offset);
              input_pf_L2_offset += 64;
            }
          }
        }

        assert(0 != i_conv_desc->ifm_block);
        if ( (l_n == 0) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) && (l_m == 0) ) {
          int l_filter_weight_pos = l_k + ((l_k/i_conv_desc->ifm_block)*(i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block)* i_conv_desc->fm_lp_block);
          libxsmm_x86_instruction_prefetch( io_generated_code,
              prefetch_type_weight /*i_conv_kernel_config->prefetch_instruction*/,
              i_gp_reg_mapping->gp_reg_weight_pf,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              l_filter_weight_pos * i_conv_desc->ofm_block * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block );
        }
        if (step_size == 4) {
          if ( (l_n == 2) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) && (l_m == 0) ) {
            int l_filter_weight_pos = (l_k+1) + (((l_k+1)/i_conv_desc->ifm_block)*(i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block)* i_conv_desc->fm_lp_block);
            libxsmm_x86_instruction_prefetch( io_generated_code,
                prefetch_type_weight /*i_conv_kernel_config->prefetch_instruction*/,
                i_gp_reg_mapping->gp_reg_weight_pf,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                l_filter_weight_pos * i_conv_desc->ofm_block * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block );
          }
          if ( (l_n == 4) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) && (l_m == 0) ) {
            int l_filter_weight_pos = (l_k+2) + (((l_k+2)/i_conv_desc->ifm_block)*(i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block));
            libxsmm_x86_instruction_prefetch( io_generated_code,
                prefetch_type_weight /*i_conv_kernel_config->prefetch_instruction*/,
                i_gp_reg_mapping->gp_reg_weight_pf,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                l_filter_weight_pos *i_conv_desc->ofm_block * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block );
          }
          if ( (l_n == 6) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) && (l_m == 0) ) {
            int l_filter_weight_pos = (l_k+3) + (((l_k+3)/i_conv_desc->ifm_block)*(i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block)* i_conv_desc->fm_lp_block);
            libxsmm_x86_instruction_prefetch( io_generated_code,
                prefetch_type_weight /*i_conv_kernel_config->prefetch_instruction*/,
                i_gp_reg_mapping->gp_reg_weight_pf,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                l_filter_weight_pos * i_conv_desc->ofm_block * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block );
          }
        }

        if ( i_conv_desc->extra_L2_prefetching == 1 ) {
          if ( (l_n == 8) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) && (l_m == 1) ) {
            int l_filter_weight_pos = l_k + ((l_k/i_conv_desc->ifm_block)*(i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block)* i_conv_desc->fm_lp_block);
            libxsmm_x86_instruction_prefetch( io_generated_code,
                LIBXSMM_X86_INSTR_PREFETCHT1 /*i_conv_kernel_config->prefetch_instruction*/,
                i_gp_reg_mapping->gp_reg_weight_pf_L2,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                l_filter_weight_pos * i_conv_desc->ofm_block * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block );
          }
          if (step_size == 4) {
            if ( (l_n == 10) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) && (l_m == 1) ) {
              int l_filter_weight_pos = (l_k+1) + (((l_k+1)/i_conv_desc->ifm_block)*(i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block)* i_conv_desc->fm_lp_block);
              libxsmm_x86_instruction_prefetch( io_generated_code,
                  LIBXSMM_X86_INSTR_PREFETCHT1 /*i_conv_kernel_config->prefetch_instruction*/,
                  i_gp_reg_mapping->gp_reg_weight_pf_L2,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  l_filter_weight_pos * i_conv_desc->ofm_block * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block );
            }
            if ( (l_n == 12) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) && (l_m == 1) ) {
              int l_filter_weight_pos = (l_k+2) + (((l_k+2)/i_conv_desc->ifm_block)*(i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block));
              libxsmm_x86_instruction_prefetch( io_generated_code,
                  LIBXSMM_X86_INSTR_PREFETCHT1 /*i_conv_kernel_config->prefetch_instruction*/,
                  i_gp_reg_mapping->gp_reg_weight_pf_L2,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  l_filter_weight_pos *i_conv_desc->ofm_block * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block );
            }
            if ( (l_n == 13) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) && (l_m == 1) ) {
              int l_filter_weight_pos = (l_k+3) + (((l_k+3)/i_conv_desc->ifm_block)*(i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block)* i_conv_desc->fm_lp_block);
              libxsmm_x86_instruction_prefetch( io_generated_code,
                  LIBXSMM_X86_INSTR_PREFETCHT1 /*i_conv_kernel_config->prefetch_instruction*/,
                  i_gp_reg_mapping->gp_reg_weight_pf_L2,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  l_filter_weight_pos * i_conv_desc->ofm_block * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block );
            }
          }
        }
      }
      l_prefetch_input_index_h++;
    }
    l_displacement_k+=step_size;
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_convolution_forward_avx512_ifmloop_qfma_x_rows( libxsmm_generated_code*                           io_generated_code,
    const libxsmm_convolution_forward_gp_reg_mapping* i_gp_reg_mapping,
    const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
    const libxsmm_convolution_forward_descriptor*     i_conv_desc,
    const unsigned int                                i_kw_unroll,
    const unsigned int                                i_x_rows ) {
  unsigned int l_n;
  unsigned int l_k;
  unsigned int l_input_reg;
  unsigned int l_input_idx = LIBXSMM_X86_GP_REG_UNDEF;
  unsigned int l_scale = 0;
  unsigned int l_disp;
  unsigned int l_w;
  unsigned int l_m;
  unsigned int l_filter_pos = 0;
  unsigned int l_compute_instr = 0;
  unsigned int l_prefetch_input_index_w = 0;
  unsigned int l_prefetch_input_index_h = 0;
  unsigned int prefetch_type_input = LIBXSMM_X86_INSTR_PREFETCHT0;
  unsigned int prefetch_type_weight = LIBXSMM_X86_INSTR_PREFETCHT0;
  unsigned int input_pf_L2_bound =  i_x_rows * i_conv_desc->ofw_rb * 64;
  unsigned int input_pf_L2_offset = 0;
  int moffset = 0;
  int step_size = 0;

  if ( i_conv_desc->kw != 1) {
    prefetch_type_weight = LIBXSMM_X86_INSTR_PREFETCHT1;
  }

  /* apply k blocking */
  if ( i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_KNM ) {
    if (i_conv_desc->ifm_block > 3) {
      step_size = 4;
    } else {
      step_size =  i_conv_desc->ifm_block;
    }
  } else {
    step_size = 1;
  }

  l_input_reg = i_gp_reg_mapping->gp_reg_input;
  /* apply k blocking */
  for ( l_k = 0; l_k < i_conv_desc->ifm_block*i_kw_unroll; l_k+=step_size ) {
    /* load the four source registers, we cannot perform a pipeline as in case of sfma */
    for ( l_w = 0; l_w < (unsigned int)step_size; l_w++ ) {
      if (((l_k+l_w)%i_conv_desc->ifm_block == 0) && (l_k > 0) ) {
        l_filter_pos += (i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block) * i_conv_desc->fm_lp_block;
      }
      if (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_BF16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_BF16) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_conv_kernel_config->instruction_set,
            i_conv_kernel_config->vmove_instruction,
            i_gp_reg_mapping->gp_reg_weight,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            l_filter_pos* i_conv_kernel_config->l_ld_ofm_fil * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block,
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
            i_gp_reg_mapping->gp_reg_weight,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            l_filter_pos* i_conv_kernel_config->l_ld_ofm_fil * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block,
            i_conv_kernel_config->vector_name, l_w,
            0, 1, 0 );
      }
      l_filter_pos++;
    }

    /* we QMADD for 3 input channels, 4th channel needs to be set to zero */
    if (step_size == 3) {
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
          i_conv_kernel_config->instruction_set,
          i_conv_kernel_config->vxor_instruction,
          i_conv_kernel_config->vector_name, 3, 3, 3);
    }

    /* compute vectorwidth (A) * column broadcast (B) */
    for ( l_m = 0; l_m < i_x_rows; l_m++) {
      l_prefetch_input_index_w = 0;
      for ( l_n = 0; l_n < i_conv_desc->ofw_rb; l_n++) {
        /* set displacement */
        l_disp = ((l_m * i_conv_desc->ifw_padded *  i_conv_desc->stride_h) + l_n * i_conv_desc->stride_w) * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in * i_conv_desc->fm_lp_block + moffset;

        if (step_size == 1) {
          if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_F32 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
            libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                i_conv_kernel_config->instruction_set,
                i_conv_kernel_config->vfma_instruction,
                1,
                l_input_reg,
                LIBXSMM_X86_GP_REG_UNDEF,
                0,
                l_disp,
                i_conv_kernel_config->vector_name,
                0,
                i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*i_x_rows) + l_n + (l_m*i_conv_desc->ofw_rb) );
          } else if ( (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32) ||
              (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32)    ) {
            if ( i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_CORE ) {
              libxsmm_x86_instruction_vec_move( io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  i_conv_kernel_config->vbcst_instruction,
                  l_input_reg,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  l_disp,
                  i_conv_kernel_config->vector_name,
                  1, 0, 1, 0 );
              libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  i_conv_kernel_config->vfma_instruction,
                  i_conv_kernel_config->vector_name,
                  0,
                  1,
                  1 );
              libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  i_conv_kernel_config->vadd_instruction,
                  i_conv_kernel_config->vector_name,
                  1,
                  i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*i_x_rows) + l_n + (l_m*i_conv_desc->ofw_rb),
                  i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*i_x_rows) + l_n + (l_m*i_conv_desc->ofw_rb) );
            } else {
              libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  LIBXSMM_X86_INSTR_VPDPWSSDS,
                  1,
                  l_input_reg,
                  LIBXSMM_X86_GP_REG_UNDEF,
                  0,
                  l_disp,
                  i_conv_kernel_config->vector_name,
                  0,
                  i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*i_x_rows) + l_n + (l_m*i_conv_desc->ofw_rb) );
            }
          } else if (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_BF16) {
            if ( i_conv_kernel_config->instruction_set == LIBXSMM_X86_AVX512_CORE ) {
              /* bcast  */
              libxsmm_x86_instruction_vec_move( io_generated_code,
                  i_conv_kernel_config->instruction_set,
                  i_conv_kernel_config->vbcst_instruction,
                  l_input_reg,
                  l_input_idx, l_scale,
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
                  i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*i_x_rows) + l_n + (l_m*i_conv_desc->ofw_rb) );

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
                  i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*i_x_rows) + l_n + (l_m*i_conv_desc->ofw_rb) );
            }
          } else if ( (i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I8  && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32
                && (i_conv_desc->option & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0) ) {
            /* broadcast in quadruples of 8 bit values */
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_conv_kernel_config->instruction_set,
                i_conv_kernel_config->vbcst_instruction,
                l_input_reg,
                l_input_idx, l_scale,
                l_disp,
                i_conv_kernel_config->vector_name,
                1, 0, 1, 0 );

            /* 8/16bit integer MADD with horizontal add */
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                i_conv_kernel_config->instruction_set,
                i_conv_kernel_config->vfma_instruction,
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
                i_conv_kernel_config->vadd_instruction,
                i_conv_kernel_config->vector_name,
                2,
                i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*i_x_rows) + l_n + (l_m*i_conv_desc->ofw_rb),
                i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*i_x_rows) + l_n + (l_m*i_conv_desc->ofw_rb) );
          } else {
            /* shouldn't happen */
          }
        } else {
          /* depending on datatype emit the needed FMA(-sequence) */
          if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_F32 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
            l_compute_instr = LIBXSMM_X86_INSTR_V4FMADDPS;
          } else if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_I32 ) {
            l_compute_instr = LIBXSMM_X86_INSTR_VP4DPWSSDS;
          } else if ( i_conv_desc->datatype == LIBXSMM_DNN_DATATYPE_I16 && i_conv_desc->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
            l_compute_instr = LIBXSMM_X86_INSTR_VP4DPWSSDS;
          } else {
            /* shouldn't happen */
          }

          libxsmm_x86_instruction_vec_compute_qfma( io_generated_code,
              i_conv_kernel_config->instruction_set,
              l_compute_instr,
              l_input_reg,
              LIBXSMM_X86_GP_REG_UNDEF,
              0,
              l_disp,
              i_conv_kernel_config->vector_name,
              0,
              i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*i_x_rows) + l_n + (l_m*i_conv_desc->ofw_rb) );
        }

        if ( l_k % 4 == 0 ) {
          if ( ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) && (l_prefetch_input_index_w < i_conv_desc->ofw_rb)  && (l_prefetch_input_index_h < i_conv_desc->ofh_rb)  ) {
            unsigned int pf_offset = (l_prefetch_input_index_w * i_conv_desc->stride_w + l_prefetch_input_index_h * i_conv_desc->stride_h * i_conv_desc->ifw_padded) * i_conv_kernel_config->datatype_size_in * i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block;
            if (pf_offset % 64 == 0) {
              libxsmm_x86_instruction_prefetch( io_generated_code,
                  prefetch_type_input,
                  i_gp_reg_mapping->gp_reg_input_pf,
                  LIBXSMM_X86_GP_REG_UNDEF,
                  0,
                  (l_prefetch_input_index_w * i_conv_desc->stride_w + l_prefetch_input_index_h * i_conv_desc->stride_h * i_conv_desc->ifw_padded ) * i_conv_kernel_config->datatype_size_in *
                  i_conv_kernel_config->l_ld_ifm_act * i_conv_desc->fm_lp_block );
            }
            l_prefetch_input_index_w++;
          }

          if ( (l_n == 2 || l_n == 4 )  && (i_conv_desc->extra_L2_prefetching == 1)  && (i_conv_desc->input_L2_prefetching == 1) ) {
            if ( input_pf_L2_offset < input_pf_L2_bound ) {
              libxsmm_x86_instruction_prefetch( io_generated_code,
                  LIBXSMM_X86_INSTR_PREFETCHT1,
                  i_gp_reg_mapping->gp_reg_input_pf_L2,
                  LIBXSMM_X86_GP_REG_UNDEF,
                  0,
                  input_pf_L2_offset);
              input_pf_L2_offset += 64;
            }
          }
        }

        assert(0 != i_conv_desc->ifm_block);
        if ( (l_n == 0) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) && (l_m == 0) ) {
          int l_filter_weight_pos = l_k + ((l_k/i_conv_desc->ifm_block)*(i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block)* i_conv_desc->fm_lp_block);
          libxsmm_x86_instruction_prefetch( io_generated_code,
              prefetch_type_weight /*i_conv_kernel_config->prefetch_instruction*/,
              i_gp_reg_mapping->gp_reg_weight_pf,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              l_filter_weight_pos * i_conv_desc->ofm_block * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block );
        }
        if (step_size == 4) {
          if ( (l_n == 2) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) && (l_m == 0) ) {
            int l_filter_weight_pos = (l_k+1) + (((l_k+1)/i_conv_desc->ifm_block)*(i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block)* i_conv_desc->fm_lp_block);
            libxsmm_x86_instruction_prefetch( io_generated_code,
                prefetch_type_weight /*i_conv_kernel_config->prefetch_instruction*/,
                i_gp_reg_mapping->gp_reg_weight_pf,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                l_filter_weight_pos * i_conv_desc->ofm_block * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block );
          }
          if ( (l_n == 4) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) && (l_m == 0) ) {
            int l_filter_weight_pos = (l_k+2) + (((l_k+2)/i_conv_desc->ifm_block)*(i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block));
            libxsmm_x86_instruction_prefetch( io_generated_code,
                prefetch_type_weight /*i_conv_kernel_config->prefetch_instruction*/,
                i_gp_reg_mapping->gp_reg_weight_pf,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                l_filter_weight_pos *i_conv_desc->ofm_block * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block );
          }
          if ( (l_n == 6) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) && (l_m == 0) ) {
            int l_filter_weight_pos = (l_k+3) + (((l_k+3)/i_conv_desc->ifm_block)*(i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block)* i_conv_desc->fm_lp_block);
            libxsmm_x86_instruction_prefetch( io_generated_code,
                prefetch_type_weight /*i_conv_kernel_config->prefetch_instruction*/,
                i_gp_reg_mapping->gp_reg_weight_pf,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                l_filter_weight_pos * i_conv_desc->ofm_block * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block );
          }
        }

        if ( i_conv_desc->extra_L2_prefetching == 1 ) {
          if ( (l_n == 0) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) && (l_m == 1) ) {
            int l_filter_weight_pos = l_k + ((l_k/i_conv_desc->ifm_block)*(i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block)* i_conv_desc->fm_lp_block);
            libxsmm_x86_instruction_prefetch( io_generated_code,
                LIBXSMM_X86_INSTR_PREFETCHT1 /*i_conv_kernel_config->prefetch_instruction*/,
                i_gp_reg_mapping->gp_reg_weight_pf_L2,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                l_filter_weight_pos * i_conv_desc->ofm_block * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block );
          }
          if (step_size == 4) {
            if ( (l_n == 2) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) && (l_m == 1) ) {
              int l_filter_weight_pos = (l_k+1) + (((l_k+1)/i_conv_desc->ifm_block)*(i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block)* i_conv_desc->fm_lp_block);
              libxsmm_x86_instruction_prefetch( io_generated_code,
                  LIBXSMM_X86_INSTR_PREFETCHT1 /*i_conv_kernel_config->prefetch_instruction*/,
                  i_gp_reg_mapping->gp_reg_weight_pf_L2,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  l_filter_weight_pos * i_conv_desc->ofm_block * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block );
            }
            if ( (l_n == 4) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) && (l_m == 1) ) {
              int l_filter_weight_pos = (l_k+2) + (((l_k+2)/i_conv_desc->ifm_block)*(i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block));
              libxsmm_x86_instruction_prefetch( io_generated_code,
                  LIBXSMM_X86_INSTR_PREFETCHT1 /*i_conv_kernel_config->prefetch_instruction*/,
                  i_gp_reg_mapping->gp_reg_weight_pf_L2,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  l_filter_weight_pos *i_conv_desc->ofm_block * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block );
            }
            if ( (l_n == 6) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) && (l_m == 1) ) {
              int l_filter_weight_pos = (l_k+3) + (((l_k+3)/i_conv_desc->ifm_block)*(i_conv_kernel_config->l_ld_ifm_fil-i_conv_desc->ifm_block)* i_conv_desc->fm_lp_block);
              libxsmm_x86_instruction_prefetch( io_generated_code,
                  LIBXSMM_X86_INSTR_PREFETCHT1 /*i_conv_kernel_config->prefetch_instruction*/,
                  i_gp_reg_mapping->gp_reg_weight_pf_L2,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  l_filter_weight_pos * i_conv_desc->ofm_block * i_conv_kernel_config->datatype_size_wt * i_conv_desc->fm_lp_block );
            }
          }
        }
      }
      l_prefetch_input_index_h++;
    }
    if (step_size == 1) {
      moffset += 4;
    } else {
      moffset += step_size * 4;
    }
  }
}

