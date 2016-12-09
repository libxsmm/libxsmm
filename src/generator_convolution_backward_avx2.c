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

#include "generator_convolution_backward_avx2.h"
#include "generator_convolution_common.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"

#include <libxsmm_intrinsics_x86.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_backward_avx2_kernel( libxsmm_generated_code*                        io_generated_code,
                                                         const libxsmm_convolution_backward_descriptor* i_conv_desc,
                                                         const char*                                    i_arch ) {
  /* code gen datastructures */
  libxsmm_convolution_kernel_config l_conv_kernel_config = { 0/*avoid warning "maybe used uninitialized" */ };
  libxsmm_convolution_backward_gp_reg_mapping l_gp_reg_mapping;
  libxsmm_loop_label_tracker l_loop_label_tracker;

  /* checks for data format settings */
  unsigned int l_found_act_format = 0;
  unsigned int l_found_fil_format = 0;

  /* local ofw_rb blocking as we JIT the entire OFW loop */
  unsigned int l_ofw_rb = 0;
  unsigned int l_ofw_rb_2 = 0;
  unsigned int l_ofw_rb_trips = 0;

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
  l_gp_reg_mapping.gp_reg_oi_loop = LIBXSMM_X86_GP_REG_R12;
  l_gp_reg_mapping.gp_reg_kh_loop = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_ofmInner_loop = LIBXSMM_X86_GP_REG_RAX;
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
       strcmp( i_arch, "hsw" ) == 0 ) {
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
  if ( i_conv_desc->ifm_block % l_conv_kernel_config.vector_length_in != 0 ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_CONV_IFM_VEC );
    return;
  }

  /* check if we have  stride of 1 */
  if ( i_conv_desc->stride_h != 1 || i_conv_desc->stride_w != 1 ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_CONV_CONT_STRIDE );
    return;
  }

  /* initilize KW and OFW unrolling */
  if (i_conv_desc->unroll_kw != 0) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_INVALID_KW_UNROLL );
    return;
  }

  /* calculate ofw blocking */
  l_ofw_rb = LIBXSMM_MIN(3, i_conv_desc->ofw);
  l_ofw_rb_2 = (i_conv_desc->ofw > 3) ? i_conv_desc->ofw % 3 : 0;
  l_ofw_rb_trips = i_conv_desc->ofw/l_ofw_rb;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* open asm */
  libxsmm_x86_instruction_open_stream_convolution( io_generated_code, l_gp_reg_mapping.gp_reg_input,
                                                   l_gp_reg_mapping.gp_reg_weight, l_gp_reg_mapping.gp_reg_output,
                                                   l_gp_reg_mapping.gp_reg_input_pf, l_gp_reg_mapping.gp_reg_weight_pf,
                                                   l_gp_reg_mapping.gp_reg_output_pf, i_arch );

  /* ofw loop, blocked peeled */
  libxsmm_generator_convolution_backward_avx2_ofwloop(   io_generated_code,
                                                        &l_gp_reg_mapping,
                                                        &l_conv_kernel_config,
                                                         i_conv_desc,
                                                        &l_loop_label_tracker,
                                                         l_ofw_rb,
                                                         l_ofw_rb_trips );

  /* ofw loop, remainder handling */
  if (l_ofw_rb_2 != 0) {
    libxsmm_generator_convolution_backward_avx2_ofwloop(   io_generated_code,
                                                          &l_gp_reg_mapping,
                                                          &l_conv_kernel_config,
                                                           i_conv_desc,
                                                          &l_loop_label_tracker,
                                                           l_ofw_rb_2,
                                                           1 );
  }

  /* close asm */
  libxsmm_x86_instruction_close_stream_convolution( io_generated_code, i_arch );
}


LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_backward_avx2_ofwloop( libxsmm_generated_code*                             io_generated_code,
                                                          const libxsmm_convolution_backward_gp_reg_mapping*  i_gp_reg_mapping,
                                                          const libxsmm_convolution_kernel_config*            i_conv_kernel_config,
                                                          const libxsmm_convolution_backward_descriptor*      i_conv_desc,
                                                          libxsmm_loop_label_tracker*                         i_loop_label_tracker,
                                                          const unsigned int                                  i_ofw_rb,
                                                          const unsigned int                                  i_ofw_rb_trips )
{
  /* blocking trip counters */
  unsigned int l_m, l_n;
  /* deriving register blocking from kernel config */
  unsigned int l_ifm_blocking = 0;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = 16;

  /* calculate blocking */
  l_ifm_blocking = i_conv_desc->ifm_block / i_conv_kernel_config->vector_length_out;
  l_vec_reg_acc_start = 16 - (l_ifm_blocking * i_ofw_rb);

  /* generate ofw loop header */
  if ( i_ofw_rb_trips > 1 ) {
    /* header of oi loop */
    libxsmm_generator_convolution_header_oi_loop(  io_generated_code, i_loop_label_tracker,
                                                     i_conv_kernel_config, i_gp_reg_mapping->gp_reg_oi_loop );
  }

  if ( i_conv_desc->kw > 1 ) {
    /* open KW loop, ki */
    libxsmm_generator_convolution_header_kw_loop(  io_generated_code, i_loop_label_tracker,
                                                     i_conv_kernel_config, i_gp_reg_mapping->gp_reg_kw_loop );
  }

  /* load input */
  for ( l_n = 0; l_n < i_ofw_rb; l_n++ ) {
    for ( l_m = 0; l_m < l_ifm_blocking; l_m++ ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_conv_kernel_config->instruction_set,
                                        i_conv_kernel_config->vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_input,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        (l_m * i_conv_kernel_config->vector_length_in * i_conv_kernel_config->datatype_size_in) +
                                          (l_n * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in),
                                        i_conv_kernel_config->vector_name,
                                        l_vec_reg_acc_start + l_m + (l_n * l_ifm_blocking), 0, 0 );
    }
  }

  libxsmm_generator_convolution_backward_avx2_ofmloop(  io_generated_code,
                                                        i_gp_reg_mapping,
                                                        i_conv_kernel_config,
                                                        i_conv_desc,
                                                        i_loop_label_tracker,
                                                        1,
                                                        i_ofw_rb );

  /* store input */
  for ( l_n = 0; l_n < i_ofw_rb; l_n++ ) {
    for ( l_m = 0; l_m < l_ifm_blocking; l_m++ ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_conv_kernel_config->instruction_set,
                                        i_conv_kernel_config->vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_input,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        (l_m * i_conv_kernel_config->vector_length_in * i_conv_kernel_config->datatype_size_in) +
                                          (l_n * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in),
                                        i_conv_kernel_config->vector_name,
                                        l_vec_reg_acc_start + l_m + (l_n * l_ifm_blocking), 0, 1 );
    }
  }

  if ( i_conv_desc->kw > 1 ) {
    /* advance input */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_conv_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_input,
                                     i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in  );

    /* advance weight */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_conv_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_weight,
                                     i_conv_kernel_config->l_ld_ifm_fil * i_conv_kernel_config->l_ld_ofm_fil * i_conv_kernel_config->datatype_size_wt  );
    /* close KW loop, ki */
    libxsmm_generator_convolution_footer_kw_loop( io_generated_code, i_loop_label_tracker,
                                                  i_conv_kernel_config, i_gp_reg_mapping->gp_reg_kw_loop, i_conv_desc->kw /*l_kw_trips*/ );
    /* reset input */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_conv_kernel_config->alu_sub_instruction,
                                     i_gp_reg_mapping->gp_reg_input,
                                     i_conv_desc->kw * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in  );

    /* reset weight */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_conv_kernel_config->alu_sub_instruction,
                                     i_gp_reg_mapping->gp_reg_weight,
                                     i_conv_desc->kw * i_conv_kernel_config->l_ld_ifm_fil * i_conv_kernel_config->l_ld_ofm_fil * i_conv_kernel_config->datatype_size_wt  );
  }

  if ( (i_ofw_rb_trips > 1) || (i_conv_desc->ofw > 3 && i_conv_desc->ofw < 6) ) {
    /* advance input */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_conv_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_input,
                                     i_ofw_rb * i_conv_kernel_config->l_ld_ifm_act * i_conv_kernel_config->datatype_size_in  );
    /* advance output */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_conv_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_output,
                                     i_ofw_rb *  i_conv_kernel_config->l_ld_ofm_act * i_conv_kernel_config->datatype_size_out  );
  }

  if ( i_ofw_rb_trips > 1 ) {
    /* close oi loop with blocking */
    libxsmm_generator_convolution_footer_oi_loop( io_generated_code, i_loop_label_tracker,
                                                  i_conv_kernel_config, i_gp_reg_mapping->gp_reg_oi_loop, i_ofw_rb_trips );
  }
}


LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_backward_avx2_ofmloop( libxsmm_generated_code*                             io_generated_code,
                                                          const libxsmm_convolution_backward_gp_reg_mapping*  i_gp_reg_mapping,
                                                          const libxsmm_convolution_kernel_config*            i_conv_kernel_config,
                                                          const libxsmm_convolution_backward_descriptor*      i_conv_desc,
                                                          libxsmm_loop_label_tracker*                         i_loop_label_tracker,
                                                          const unsigned int                                  i_kw_unroll,
                                                          const unsigned int                                  i_ofw_rb )
{
  /* deriving register blocking from kernel config */
  unsigned int l_ifm_blocking = i_conv_desc->ifm_block / i_conv_kernel_config->vector_length_out;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = 16 - (l_ifm_blocking * i_ofw_rb);
  /* blocking trip counters */
  unsigned int l_k, l_n, l_m;
  /* unrolling ofm loop */
  unsigned int l_ofm_trip_count = 0;
  /* total iterations */
  unsigned int l_total_trips = i_kw_unroll*i_conv_desc->ofm_block;
  l_k = l_n = l_m = 0;

  if ( i_conv_desc->ofm_block % i_conv_kernel_config->vector_length_in != 0 ) {
    l_ofm_trip_count = i_conv_desc->ofm_block*i_kw_unroll;
  } else {
    l_ofm_trip_count = i_conv_kernel_config->vector_length_in/2;
  }

  /* Some checks */
  if ( i_conv_desc->ofh_rb != 1 ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_INVALID_OFH_UNROLL );
    return;
  }
  if ( i_ofw_rb*l_ifm_blocking > i_conv_kernel_config->vector_reg_count-4 ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_INVALID_CONV_ACC );
    return;
  }

  libxsmm_generator_convolution_header_ofm_loop( io_generated_code, i_loop_label_tracker,
                                                 i_conv_kernel_config, i_gp_reg_mapping->gp_reg_ofmInner_loop, l_ofm_trip_count );

  /* apply k blocking */
  for ( l_k = 0; l_k < l_ofm_trip_count; l_k++ ) {
    for ( l_m = 0; l_m < l_ifm_blocking; l_m++ ) {
      /* load weights */
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_conv_kernel_config->instruction_set,
                                        i_conv_kernel_config->vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_weight,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (l_k*i_conv_kernel_config->l_ld_ifm_fil*i_conv_kernel_config->datatype_size_wt)
                                        + (l_m*i_conv_kernel_config->vector_length_wt*i_conv_kernel_config->datatype_size_wt),
                                        i_conv_kernel_config->vector_name, i_ofw_rb,
                                        0, 0  );
      if ( l_m == 0 ) {
        /* broadcast input values into registers 0 -> ofw_rb */
        for ( l_n = 0; l_n < i_ofw_rb; l_n++ ) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_conv_kernel_config->instruction_set,
                                            LIBXSMM_X86_INSTR_VBROADCASTSS,
                                            i_gp_reg_mapping->gp_reg_output,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              (i_conv_kernel_config->datatype_size_out * i_conv_kernel_config->l_ld_ofm_act * l_n)
                                            + (l_k * i_conv_kernel_config->datatype_size_out),
                                            i_conv_kernel_config->vector_name,
                                            l_n, 0, 0 );
        }


        if (l_k == l_ofm_trip_count - 1) {
          libxsmm_x86_instruction_alu_imm( io_generated_code,
                                           i_conv_kernel_config->alu_add_instruction,
                                           i_gp_reg_mapping->gp_reg_output,
                                           l_ofm_trip_count * i_conv_kernel_config->datatype_size_out );
        }
      }

      if ( (l_k == l_ofm_trip_count - 1) && (l_m == l_ifm_blocking-1) ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code,
                                         i_conv_kernel_config->alu_add_instruction,
                                         i_gp_reg_mapping->gp_reg_weight,
                                         l_ofm_trip_count*i_conv_kernel_config->l_ld_ifm_fil*i_conv_kernel_config->datatype_size_wt );
      }

      /* convolute! */
      for ( l_n = 0; l_n < i_ofw_rb; l_n++ ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_conv_kernel_config->instruction_set,
                                                 i_conv_kernel_config->vfma_instruction,
                                                 i_conv_kernel_config->vector_name,
                                                 i_ofw_rb,
                                                 l_n,
                                                 l_vec_reg_acc_start + l_m + (l_ifm_blocking * l_n) );
      }
    }
  }

  libxsmm_generator_convolution_footer_ofm_loop( io_generated_code,  i_loop_label_tracker,
                                                 i_conv_kernel_config, i_gp_reg_mapping->gp_reg_ofmInner_loop, l_total_trips );

  libxsmm_x86_instruction_alu_imm( io_generated_code,
                                   i_conv_kernel_config->alu_sub_instruction,
                                   i_gp_reg_mapping->gp_reg_output,
                                   l_total_trips*i_conv_kernel_config->datatype_size_out );

  libxsmm_x86_instruction_alu_imm( io_generated_code,
                                   i_conv_kernel_config->alu_sub_instruction,
                                   i_gp_reg_mapping->gp_reg_weight,
                                   l_total_trips*i_conv_kernel_config->l_ld_ifm_fil*i_conv_kernel_config->datatype_size_wt );
}

