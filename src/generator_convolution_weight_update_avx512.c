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
/* Rajkishore Barik (Intel Corp.)
******************************************************************************/

#include "generator_convolution_weight_update_avx512.h"
#include "generator_convolution_common.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"

#include <libxsmm_cpuid.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_weight_update_avx512_kernel( libxsmm_generated_code*     io_generated_code,
                                                          const libxsmm_convolution_weight_update_descriptor* i_conv_desc,
                                                          const char*                       i_arch ) {
  libxsmm_convolution_kernel_config l_conv_kernel_config;
  libxsmm_convolution_weight_update_gp_reg_mapping l_gp_reg_mapping;
  libxsmm_loop_label_tracker l_loop_label_tracker;

  unsigned int l_ofh_trips=0, l_ofw_trips=0;
  unsigned int oj=0, oi=0;
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
  } else {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
    return;
  }
  l_conv_kernel_config.vector_reg_count = 32;
  l_conv_kernel_config.vector_length_in = 16;
  l_conv_kernel_config.datatype_size_in = 4;
  l_conv_kernel_config.vector_length_out = 16;
  l_conv_kernel_config.datatype_size_out = 4;
  l_conv_kernel_config.vector_length_wt = 16;
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
  l_conv_kernel_config.vector_name = 'z';

  /* check if we have full vectors */
  if ( i_conv_desc->ofm_block % l_conv_kernel_config.vector_length_out != 0 ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_CONV_OFM_VEC );
    return;
  }

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* open asm */
  libxsmm_x86_instruction_open_stream_convolution( io_generated_code, l_gp_reg_mapping.gp_reg_input,
                                                   l_gp_reg_mapping.gp_reg_weight, l_gp_reg_mapping.gp_reg_output,
                                                   l_gp_reg_mapping.gp_reg_input_pf, l_gp_reg_mapping.gp_reg_weight_pf,
                                                   l_gp_reg_mapping.gp_reg_output_pf, i_arch );

  /***** Code to generate

            // JIT this code below
              for(int ifm2 = 0; ifm2 < VLEN; ifm2+=WU_UNROLL_FACTOR_1) { // We should completely unroll this!
                __m512 acc00 = _mm512_load_ps(&del_wt[ofm1][ifm1][kj][ki][ifm2]);
                __m512 acc01 = _mm512_load_ps(&del_wt[ofm1][ifm1][kj][ki][ifm2 + 1]);
#pragma unroll
                for(int ij=0, oj=0; oj < ofh; ij+=stride_h, oj++) {
#pragma unroll
                  for(int ii=0, oi=0; oi < ofw; ii+=WU_UNROLL_FACTOR_2*stride_w, oi+=WU_UNROLL_FACTOR_2) {
                    __m512 out00 = _mm512_load_ps(&del_output[img][ofm1][oj][oi + 0]);
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
  /* initilize OFW and OFH unrolling */

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


  if (i_conv_desc->transpose_ofw_ifm == 1) {
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
  if((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L2) {
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     l_conv_kernel_config.alu_sub_instruction,
                                     l_gp_reg_mapping.gp_reg_input_pf, (i_conv_desc->ofh/*/i_conv_desc->ofh_rb*/) * i_conv_desc->ifw_padded* /*l_conv_kernel_config.vector_length*/ i_conv_desc->stride_h * i_conv_desc->ifm_block * l_conv_kernel_config.datatype_size  );
  }

    /* TODO */
    /*if(l_ofw_trips > 1) {*/
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
  if((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L2) {
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

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_weight_update_avx512_ofwloop( libxsmm_generated_code*                           io_generated_code,
                                                           const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
                                                           const libxsmm_convolution_kernel_config*                i_conv_kernel_config,
                                                           const libxsmm_convolution_weight_update_descriptor*                       i_conv_desc,
                                                           const unsigned int                                      i_ofh_unroll,
                                                           const unsigned int                                      ofh_trip_counter,
                                                           const int                                               no_unroll_no_block)
{
  /* setup input strides */
  libxsmm_generator_convolution_weight_update_avx512_init_weight_strides( io_generated_code, i_gp_reg_mapping, i_conv_kernel_config, i_conv_desc );

    libxsmm_generator_convolution_weight_update_avx512_ofwloop_sfma( io_generated_code, i_gp_reg_mapping,
                                                               i_conv_kernel_config, i_conv_desc, i_ofh_unroll, ofh_trip_counter, no_unroll_no_block);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_weight_update_avx512_init_weight_strides( libxsmm_generated_code*                           io_generated_code,
                                                                      const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
                                                                      const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                                      const libxsmm_convolution_weight_update_descriptor*     i_conv_desc ) {

  int unroll_factor = (i_conv_desc->ifm_block == 1) ? i_conv_desc->kw : i_conv_desc->ifm_block;

  /* Intialize helper registers for SIB addressing */
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
     helper 5: B + 18*ldb, additional base adrress */
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

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_weight_update_avx512_ofwloop_sfma( libxsmm_generated_code*                           io_generated_code,
                                                                const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
                                                                const libxsmm_convolution_kernel_config*                i_conv_kernel_config,
                                                                const libxsmm_convolution_weight_update_descriptor*                       i_conv_desc,
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
  unsigned int unroll_factor = (i_conv_desc->ifm_block == 1) ? i_conv_desc->kw : i_conv_desc->ifm_block;

  LIBXSMM_UNUSED(ofh_trip_counter);

  /* apply k blocking */
  for ( l_k_1 = 0; l_k_1 < i_conv_desc->ofh_rb ; l_k_1++ ) {

  l_k_updates = 0;
  for ( l_k_2 = 0; l_k_2 < i_conv_desc->ofw_rb ; l_k_2++, l_k++ ) {
    /* advance b pointer if needed */
    if ( (l_k_2 > 0) &&(l_k_2%8 == 0) ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                       i_gp_reg_mapping->gp_reg_input, 8*(i_conv_desc->ifm_block * i_conv_desc->stride_w)* i_conv_kernel_config->datatype_size_in );
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
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_conv_kernel_config->instruction_set,
                                        i_conv_kernel_config->vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_output,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        (l_k)*(i_conv_kernel_config->vector_length_out)*(i_conv_kernel_config->datatype_size_out) ,
                                        i_conv_kernel_config->vector_name, 0,
                                        0, 0 );

      if ( (i_conv_desc->ofw_rb * i_ofh_unroll > 1) && (num_look_ahead > 1) ) {
        for ( l_w = 1; l_w < num_look_ahead; l_w++ ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_conv_kernel_config->instruction_set,
                                          i_conv_kernel_config->vmove_instruction,
                                          i_gp_reg_mapping->gp_reg_output,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (l_k+l_w)*(i_conv_kernel_config->vector_length_out)*(i_conv_kernel_config->datatype_size_out) ,
                                          i_conv_kernel_config->vector_name, l_w,
                                          0, 0 );
        }
        output_counter += num_look_ahead -1;
      }
    } else if ((l_k < ((i_conv_desc->ofw_rb * i_ofh_unroll) - (num_look_ahead-1))) && (l_k_2 >= (i_conv_desc->ofw_rb - (num_look_ahead-1))) && (l_k_2 < i_conv_desc->ofw_rb) ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_conv_kernel_config->instruction_set,
                                        i_conv_kernel_config->vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_output,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        ((output_counter)*(i_conv_kernel_config->vector_length_out)*(i_conv_kernel_config->datatype_size_out) )
                                        + ((l_k_1 + 1) * i_conv_desc->ofw_padded*i_conv_kernel_config->vector_length_out * i_conv_kernel_config->datatype_size_out  ),
                                        i_conv_kernel_config->vector_name, (l_k+(num_look_ahead-1))%num_look_ahead,
                                        0, 0 );
    } else if ( l_k < ((i_conv_desc->ofw_rb*i_ofh_unroll) - (num_look_ahead - 1)) ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_conv_kernel_config->instruction_set,
                                        i_conv_kernel_config->vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_output,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        ((output_counter)*(i_conv_kernel_config->vector_length_out)*(i_conv_kernel_config->datatype_size_out) )
                                        + ((l_k_1) * i_conv_desc->ofw_padded*i_conv_kernel_config->vector_length_out * i_conv_kernel_config->datatype_size_out  ),
                                        i_conv_kernel_config->vector_name, (l_k+(num_look_ahead-1))%num_look_ahead,
                                        0, 0 );
    }

    if(output_counter >= (i_conv_desc->ofw_rb-1)){
      output_counter = 0;
    } else output_counter++;

    /* compute vectorwidth (A) * column broadcast (B) */
    for ( l_n = 0; l_n < unroll_factor; l_n++) {
      /* determining base, idx and scale values */
      /* default values */
      l_input_reg = i_gp_reg_mapping->gp_reg_input;
      l_input_idx = LIBXSMM_X86_GP_REG_UNDEF;
      l_scale = 0;
      l_disp = l_displacement_k*(i_conv_desc->ifm_block * i_conv_desc->stride_w)*(i_conv_kernel_config->datatype_size_in);

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

      libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                               i_conv_kernel_config->instruction_set,
                                               i_conv_kernel_config->vfma_instruction,
                                               1,
                                               l_input_reg,
                                               l_input_idx,
                                               l_scale,
                                               l_disp,
                                               i_conv_kernel_config->vector_name,
                                               l_k%num_look_ahead,
                                               i_conv_kernel_config->vector_reg_count - unroll_factor + l_n );
      if ( (l_n == 4) &&  ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                          LIBXSMM_X86_INSTR_PREFETCHT0,
                                          i_gp_reg_mapping->gp_reg_input_pf,
                                          LIBXSMM_X86_GP_REG_UNDEF,
                                          0,
                                          l_k_1 * i_conv_desc->ifw_padded * i_conv_desc->stride_h * i_conv_desc->ifm_block * i_conv_kernel_config->datatype_size_in +  l_k_2 * i_conv_kernel_config->datatype_size_in * i_conv_desc->stride_w * i_conv_desc->ifm_block );
     }
     if ( (l_n == 8) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2) ) {
        libxsmm_x86_instruction_prefetch( io_generated_code,
                                          i_conv_kernel_config->prefetch_instruction,
                                          i_gp_reg_mapping->gp_reg_output_pf,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (l_k_1 * i_conv_desc->ofw_padded + l_k_2)*(i_conv_desc->ofm_block)*(i_conv_kernel_config->datatype_size_out) );
      }
    }
    l_displacement_k++;
  }

  if(l_k_updates > 0) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_sub_instruction,
                                       i_gp_reg_mapping->gp_reg_input, 8*l_k_updates*(i_conv_desc->ifm_block * i_conv_desc->stride_w)* i_conv_kernel_config->datatype_size_in );
  }

  if(!no_unroll_no_block) {
      /* Add ofw_block*40 to increment output */
      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_conv_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_input, i_conv_desc->ifw_padded * i_conv_desc->stride_h * i_conv_desc->ifm_block * i_conv_kernel_config->datatype_size_in  );
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

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_weight_update_transpose_avx512_ofwloop( libxsmm_generated_code*                           io_generated_code,
                                                           const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
                                                           const libxsmm_convolution_kernel_config*                i_conv_kernel_config,
                                                           const libxsmm_convolution_weight_update_descriptor*                       i_conv_desc,
                                                           const unsigned int                                      i_ofh_unroll,
                                                           const unsigned int                                      ofh_trip_counter,
                                                           const int                                               no_unroll_no_block)
{
  /* setup input strides */
  libxsmm_generator_convolution_weight_update_transpose_avx512_init_weight_strides( io_generated_code, i_gp_reg_mapping, i_conv_kernel_config, i_conv_desc );

    libxsmm_generator_convolution_weight_update_transpose_avx512_ofwloop_sfma( io_generated_code, i_gp_reg_mapping,
                                                               i_conv_kernel_config, i_conv_desc, i_ofh_unroll, ofh_trip_counter, no_unroll_no_block);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_weight_update_transpose_avx512_init_weight_strides( libxsmm_generated_code*                           io_generated_code,
                                                                      const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
                                                                      const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                                      const libxsmm_convolution_weight_update_descriptor*     i_conv_desc ) {

  int unroll_factor = (i_conv_desc->ifm_block == 1) ? i_conv_desc->kw : i_conv_desc->ifm_block;

  /* Intialize helper registers for SIB addressing */
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
     helper 5: B + 18*ldb, additional base adrress */
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

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_weight_update_transpose_avx512_ofwloop_sfma( libxsmm_generated_code*                           io_generated_code,
                                                                const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
                                                                const libxsmm_convolution_kernel_config*                i_conv_kernel_config,
                                                                const libxsmm_convolution_weight_update_descriptor*                       i_conv_desc,
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
  unsigned int unroll_factor = (i_conv_desc->ifm_block == 1) ? i_conv_desc->kw : i_conv_desc->ifm_block;

  LIBXSMM_UNUSED(ofh_trip_counter);

  /* apply k blocking */
  for ( l_k_1 = 0; l_k_1 < i_conv_desc->ofh_rb ; l_k_1++ ) {

  l_k_updates = 0;
  for ( l_k_2 = 0; l_k_2 < i_conv_desc->ofw_rb ; l_k_2++, l_k++ ) {

#if 0
    /* advance b pointer if needed */
    if ( (l_k_2 > 0) &&(l_k_2%8 == 0) ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                       i_gp_reg_mapping->gp_reg_input, 8*(i_conv_desc->ifm_block * i_conv_desc->stride_w)* i_conv_kernel_config->datatype_size );
      if (  unroll_factor > 9 ) {
        libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_input, i_gp_reg_mapping->gp_reg_help_4);
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                         i_gp_reg_mapping->gp_reg_help_4,  9 * i_conv_kernel_config->datatype_size );
      }
      if ( unroll_factor > 18 ) {
        libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_input, i_gp_reg_mapping->gp_reg_help_5);
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                         i_gp_reg_mapping->gp_reg_help_5, 18 *  i_conv_kernel_config->datatype_size );
      }
      if ( unroll_factor > 27 ) {
        libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_input, i_gp_reg_mapping->gp_reg_help_6);
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction,
                                         i_gp_reg_mapping->gp_reg_help_6, 27 *  i_conv_kernel_config->datatype_size );
      }
      l_displacement_k = 0;
      l_k_updates++;
    }
#endif
    num_look_ahead = i_conv_desc->ofw_rb < 4 ? i_conv_desc->ofw_rb : 4;
    /*num_look_ahead = i_conv_desc->ofw_rb < 6 ? i_conv_desc->ofw_rb : 6; */

    if (  l_k == 0 ) {
      /* load output */
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_conv_kernel_config->instruction_set,
                                        i_conv_kernel_config->vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_output,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        (l_k)*(i_conv_kernel_config->vector_length_out)*(i_conv_kernel_config->datatype_size_out) ,
                                        i_conv_kernel_config->vector_name, 0,
                                        0, 0 );

      if ( (i_conv_desc->ofw_rb * i_ofh_unroll > 1) && (num_look_ahead > 1) ) {
        for ( l_w = 1; l_w < num_look_ahead; l_w++ ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_conv_kernel_config->instruction_set,
                                          i_conv_kernel_config->vmove_instruction,
                                          i_gp_reg_mapping->gp_reg_output,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (l_k+l_w)*(i_conv_kernel_config->vector_length_out)*(i_conv_kernel_config->datatype_size_out) ,
                                          i_conv_kernel_config->vector_name, l_w,
                                          0, 0 );
        }
        output_counter += num_look_ahead -1;
      }
    } else if ((l_k < ((i_conv_desc->ofw_rb * i_ofh_unroll) - (num_look_ahead-1))) && (l_k_2 >= (i_conv_desc->ofw_rb - (num_look_ahead-1))) && (l_k_2 < i_conv_desc->ofw_rb) ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_conv_kernel_config->instruction_set,
                                        i_conv_kernel_config->vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_output,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        ((output_counter)*(i_conv_kernel_config->vector_length_out)*(i_conv_kernel_config->datatype_size_out) )
                                        + ((l_k_1 + 1) * i_conv_desc->ofw_padded*i_conv_kernel_config->vector_length_out * i_conv_kernel_config->datatype_size_out  ),
                                        i_conv_kernel_config->vector_name, (l_k+(num_look_ahead-1))%num_look_ahead,
                                        0, 0 );
    } else if ( l_k < ((i_conv_desc->ofw_rb*i_ofh_unroll) - (num_look_ahead - 1)) ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_conv_kernel_config->instruction_set,
                                        i_conv_kernel_config->vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_output,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        ((output_counter)*(i_conv_kernel_config->vector_length_out)*(i_conv_kernel_config->datatype_size_out) )
                                        + ((l_k_1) * i_conv_desc->ofw_padded*i_conv_kernel_config->vector_length_out * i_conv_kernel_config->datatype_size_out  ),
                                        i_conv_kernel_config->vector_name, (l_k+(num_look_ahead-1))%num_look_ahead,
                                        0, 0 );
    }

    if(output_counter >= (i_conv_desc->ofw_rb-1)){
      output_counter = 0;
    } else output_counter++;

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

      libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                               i_conv_kernel_config->instruction_set,
                                               i_conv_kernel_config->vfma_instruction,
                                               1,
                                               l_input_reg,
                                               l_input_idx,
                                               l_scale,
                                               l_disp,
                                               i_conv_kernel_config->vector_name,
                                               l_k%num_look_ahead,
                                               i_conv_kernel_config->vector_reg_count - unroll_factor + l_n );
      if ( (l_n == 4) &&  ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                          LIBXSMM_X86_INSTR_PREFETCHT0,
                                          i_gp_reg_mapping->gp_reg_input_pf,
                                          LIBXSMM_X86_GP_REG_UNDEF,
                                          0,
                                          l_k_1 * i_conv_desc->ifw_padded * i_conv_desc->stride_h * i_conv_desc->ifm_block * i_conv_kernel_config->datatype_size_in +  l_k_2 * i_conv_kernel_config->datatype_size_in * i_conv_desc->stride_w );
     }
     if ( (l_n == 8) && ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L2) ) {
        libxsmm_x86_instruction_prefetch( io_generated_code,
                                          i_conv_kernel_config->prefetch_instruction,
                                          i_gp_reg_mapping->gp_reg_output_pf,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (l_k_1 * i_conv_desc->ofw_padded + l_k_2)*(i_conv_desc->ofm_block)*(i_conv_kernel_config->datatype_size_out) );
      }
    }

    l_displacement_k++;
  } /* l_k_2 */

  if(l_k_updates > 0) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_sub_instruction,
                                       i_gp_reg_mapping->gp_reg_input, 8*l_k_updates*(i_conv_desc->ifm_block * i_conv_desc->stride_w)* i_conv_kernel_config->datatype_size_in );
  }

  if(!no_unroll_no_block) {
      /* Add ofw_block*40 to increment output */
      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_conv_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_input, i_conv_desc->ifw_padded * i_conv_desc->stride_h * i_conv_desc->ifm_block * i_conv_kernel_config->datatype_size_in  );
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
