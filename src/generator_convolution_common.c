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
/* Alexander Heinecke (Intel Corp.), Rajkishore Barik (Intel Corp.)
******************************************************************************/

#include "generator_x86_instructions.h"
#include "generator_convolution_common.h"
#include "generator_common.h"

#include <libxsmm_cpuid.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_header_oi_loop( libxsmm_generated_code*                   io_generated_code,
                                                   libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                   const unsigned int                        i_gp_reg_oi_loop ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_oi_loop, 0);
  libxsmm_x86_instruction_register_jump_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction, i_gp_reg_oi_loop, 1);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_footer_oi_loop( libxsmm_generated_code*                       io_generated_code,
                                                   libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*      i_conv_kernel_config,
                                                   const unsigned int                            i_gp_reg_oi_loop,
                                                   const unsigned int                            i_oi ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_cmp_instruction, i_gp_reg_oi_loop, i_oi );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_conv_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_header_oj_loop( libxsmm_generated_code*                   io_generated_code,
                                                   libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                   const unsigned int                        i_gp_reg_oj_loop ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_oj_loop, 0);
  libxsmm_x86_instruction_register_jump_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction, i_gp_reg_oj_loop, 1);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_footer_oj_loop( libxsmm_generated_code*                       io_generated_code,
                                                   libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*      i_conv_kernel_config,
                                                   const unsigned int                            i_gp_reg_oj_loop,
                                                   const unsigned int                            i_oj ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_cmp_instruction, i_gp_reg_oj_loop, i_oj );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_conv_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_header_ofw_loop( libxsmm_generated_code*                   io_generated_code,
                                                   libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                   const unsigned int                        i_gp_reg_oi_loop ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_oi_loop, 0);
  libxsmm_x86_instruction_register_jump_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction, i_gp_reg_oi_loop, 1);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_footer_ofw_loop( libxsmm_generated_code*                       io_generated_code,
                                                   libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*      i_conv_kernel_config,
                                                   const unsigned int                            i_gp_reg_oi_loop,
                                                   const unsigned int                            i_ofw ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_cmp_instruction, i_gp_reg_oi_loop, i_ofw );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_conv_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_header_ofh_loop( libxsmm_generated_code*                   io_generated_code,
                                                   libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                   const unsigned int                        i_gp_reg_ofh_loop ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_ofh_loop, 0);
  libxsmm_x86_instruction_register_jump_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction, i_gp_reg_ofh_loop, 1);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_footer_ofh_loop( libxsmm_generated_code*                       io_generated_code,
                                                   libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*      i_conv_kernel_config,
                                                   const unsigned int                            i_gp_reg_ofh_loop,
                                                   const unsigned int                            i_ofh ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_cmp_instruction, i_gp_reg_ofh_loop, i_ofh );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_conv_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_header_kh_loop( libxsmm_generated_code*                   io_generated_code,
                                                   libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                   const unsigned int                        i_gp_reg_kh_loop ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_kh_loop, 0);
  libxsmm_x86_instruction_register_jump_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction, i_gp_reg_kh_loop, 1);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_footer_kh_loop( libxsmm_generated_code*                       io_generated_code,
                                                   libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*      i_conv_kernel_config,
                                                   const unsigned int                            i_gp_reg_kh_loop,
                                                   const unsigned int                            i_kh ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_cmp_instruction, i_gp_reg_kh_loop, i_kh );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_conv_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_header_kw_loop( libxsmm_generated_code*                   io_generated_code,
                                                   libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                   const unsigned int                        i_gp_reg_kw_loop ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_kw_loop, 0);
  libxsmm_x86_instruction_register_jump_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction, i_gp_reg_kw_loop, 1);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_footer_kw_loop( libxsmm_generated_code*                       io_generated_code,
                                                   libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                                   const libxsmm_convolution_kernel_config*      i_conv_kernel_config,
                                                   const unsigned int                            i_gp_reg_kw_loop,
                                                   const unsigned int                            i_kw ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_cmp_instruction, i_gp_reg_kw_loop, i_kw );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_conv_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_header_ifm_loop( libxsmm_generated_code*                   io_generated_code,
                                                    libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                    const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                    const unsigned int                        i_gp_reg_ifmInner_loop,
                                                    const unsigned int                        i_unrolled_trips ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_ifmInner_loop, 0);
  libxsmm_x86_instruction_register_jump_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction, i_gp_reg_ifmInner_loop, i_unrolled_trips);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_footer_ifm_loop( libxsmm_generated_code*                   io_generated_code,
                                                    libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                    const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                    const unsigned int                        i_gp_reg_ifmInner_loop,
                                                    const unsigned int                        i_trip_count ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_cmp_instruction, i_gp_reg_ifmInner_loop, i_trip_count );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_conv_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_header_ofm_loop( libxsmm_generated_code*                   io_generated_code,
                                                    libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                    const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                    const unsigned int                        i_gp_reg_ofmInner_loop,
                                                    const unsigned int                        i_unrolled_trips ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_ofmInner_loop, 0);
  libxsmm_x86_instruction_register_jump_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_add_instruction, i_gp_reg_ofmInner_loop, i_unrolled_trips);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_footer_ofm_loop( libxsmm_generated_code*                   io_generated_code,
                                                    libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                                    const libxsmm_convolution_kernel_config*  i_conv_kernel_config,
                                                    const unsigned int                        i_gp_reg_ofmInner_loop,
                                                    const unsigned int                        i_trip_count ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_conv_kernel_config->alu_cmp_instruction, i_gp_reg_ofmInner_loop, i_trip_count );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_conv_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_forward_load_output( libxsmm_generated_code*                           io_generated_code,
                                                        const libxsmm_convolution_forward_gp_reg_mapping* i_gp_reg_mapping,
                                                        const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                        const libxsmm_convolution_forward_descriptor*     i_conv_desc ) {
  /* determine the number of registers needed for an ofm block */
  const unsigned int l_reg_per_block = i_conv_desc->ofm_block / i_conv_kernel_config->vector_length_out;
  /* start register of accumulator */
  const unsigned int l_vec_reg_acc_start = i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofh_rb * i_conv_desc->ofw_rb * l_reg_per_block);
  /* register blocking counter  */
  unsigned int l_i, l_j, l_k, l_accs;
  /* block-feature map offset, leading dimension */
  unsigned int l_lead_dim = 0;

#if !defined(NDEBUG)
  /* Do some test if it's possible to generated the requested code.
     This is not done in release mode and therefore bad
     things might happen.... HUAAH */
#endif /*NDEBUG*/
  if ( i_conv_desc->ofh_rb*i_conv_desc->ofw_rb*l_reg_per_block > i_conv_kernel_config->vector_reg_count-4 ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_INVALID_CONV_ACC );
    return;
  }
  if ( i_conv_desc->ofm_block % i_conv_kernel_config->vector_length_out != 0) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_CONV_OFM_VEC );
    return;
  }

  /* calculate leading dimension depending on format */
  if ( (i_conv_desc->format & LIBXSMM_DNN_CONV_FORMAT_NHWC) > 0 ) {
    l_lead_dim = i_conv_desc->ofm_block * i_conv_desc->blocks_ofm;
  } else if ( (i_conv_desc->format & LIBXSMM_DNN_CONV_FORMAT_LIBXSMM) > 0 ) {
    l_lead_dim = i_conv_desc->ofm_block;
  } else {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_UNSUP_CONV_FORMAT );
    return;
  }

  if ( (i_conv_desc->ofw_rb < 12 && i_conv_desc->ofh_rb == 1 && l_reg_per_block == 1) && (i_conv_kernel_config->instruction_set != LIBXSMM_X86_AVX2) ) {
    /* determining the number of accumulators */
    l_accs = (i_conv_desc->ofw_rb < 9) ? 3 : 2;

    /* adding to C, so let's load C and init additional accumulators */
    for ( l_j = 0; l_j < i_conv_desc->ofw_rb; l_j++ ) {
      for ( l_i = l_accs; l_i > 1; l_i-- ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_conv_kernel_config->instruction_set,
                                                 i_conv_kernel_config->vxor_instruction,
                                                 i_conv_kernel_config->vector_name,
                                                 i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*l_i) + l_j,
                                                 i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*l_i) + l_j,
                                                 i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*l_i) + l_j );
      }
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_conv_kernel_config->instruction_set,
                                        i_conv_kernel_config->vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_output,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        l_j * l_lead_dim * i_conv_kernel_config->datatype_size_out,
                                        i_conv_kernel_config->vector_name,
                                        i_conv_kernel_config->vector_reg_count - i_conv_desc->ofw_rb + l_j , 0, 0 );
#if 1
      if ( (i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1 ) {
         libxsmm_x86_instruction_prefetch( io_generated_code,
                                           LIBXSMM_X86_INSTR_PREFETCHT0 /*i_conv_kernel_config->prefetch_instruction*/,
                                           i_gp_reg_mapping->gp_reg_output_pf,
                                           LIBXSMM_X86_GP_REG_UNDEF, 0,
                                           l_j * l_lead_dim * i_conv_kernel_config->datatype_size_out );
      }
#endif
    }
  } else {
    /* adding to C, so let's load C */
    for ( l_i = 0; l_i < i_conv_desc->ofh_rb; l_i++ ) {
      for ( l_j = 0; l_j < i_conv_desc->ofw_rb; l_j++ ) {
        for ( l_k = 0; l_k < l_reg_per_block ; l_k++ ) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_conv_kernel_config->instruction_set,
                                            i_conv_kernel_config->vmove_instruction,
                                            i_gp_reg_mapping->gp_reg_output,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            ( l_i * i_conv_desc->ofw_padded * l_lead_dim * i_conv_kernel_config->datatype_size_out) +
                                            ( l_j * l_lead_dim * i_conv_kernel_config->datatype_size_out ) +
                                            ( l_k * i_conv_kernel_config->vector_length_out * i_conv_kernel_config->datatype_size_out ),
                                            i_conv_kernel_config->vector_name,
                                            l_vec_reg_acc_start + l_k + (l_j * l_reg_per_block) + (i_conv_desc->ofw_rb * l_reg_per_block * l_i), 0, 0 );
#if 1
          if ( (i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_OUTPUT_L1 ) {
            libxsmm_x86_instruction_prefetch( io_generated_code,
                                              LIBXSMM_X86_INSTR_PREFETCHT0 /*i_conv_kernel_config->prefetch_instruction*/,
                                              i_gp_reg_mapping->gp_reg_output_pf,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              ( l_i * i_conv_desc->ofw_padded * l_lead_dim * i_conv_kernel_config->datatype_size_out) +
                                              ( l_j * l_lead_dim * i_conv_kernel_config->datatype_size_out ) +
                                              ( l_k * i_conv_kernel_config->vector_length_out * i_conv_kernel_config->datatype_size_out ) );
          }
#endif
        }
      }
    }
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_forward_store_output( libxsmm_generated_code*                           io_generated_code,
                                                         const libxsmm_convolution_forward_gp_reg_mapping* i_gp_reg_mapping,
                                                         const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                         const libxsmm_convolution_forward_descriptor*     i_conv_desc ) {
  /* determine the number of registers needed for an ofm block */
  const unsigned int l_reg_per_block = i_conv_desc->ofm_block / i_conv_kernel_config->vector_length_out;
  /* start register of accumulator */
  const unsigned int l_vec_reg_acc_start = i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofh_rb * i_conv_desc->ofw_rb * l_reg_per_block);
  /* register blocking counter  */
  unsigned int l_i, l_j, l_k, l_accs;
  /* block-feature map offset, leading dimension */
  unsigned int l_lead_dim = 0;

#if !defined(NDEBUG)
  /* Do some test if it's possible to generated the requested code.
     This is not done in release mode and therefore bad
     things might happen.... HUAAH */
#endif /*NDEBUG*/
  if ( i_conv_desc->ofh_rb*i_conv_desc->ofw_rb*l_reg_per_block > i_conv_kernel_config->vector_reg_count-4 ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_INVALID_CONV_ACC );
    return;
  }
  if ( i_conv_desc->ofm_block % i_conv_kernel_config->vector_length_out != 0) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_CONV_OFM_VEC );
    return;
  }

  /* calculate leading dimension depending on format */
  if ( (i_conv_desc->format & LIBXSMM_DNN_CONV_FORMAT_NHWC) > 0 ) {
    l_lead_dim = i_conv_desc->ofm_block * i_conv_desc->blocks_ofm;
  } else if ( (i_conv_desc->format & LIBXSMM_DNN_CONV_FORMAT_LIBXSMM) > 0 ) {
    l_lead_dim = i_conv_desc->ofm_block;
  } else {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_UNSUP_CONV_FORMAT );
    return;
  }

  if ( (i_conv_desc->ofw_rb < 12 && i_conv_desc->ofh_rb == 1 && l_reg_per_block == 1) && (i_conv_kernel_config->instruction_set != LIBXSMM_X86_AVX2) ) {
    /* determining the number of accumulators */
    l_accs = (i_conv_desc->ofw_rb < 9) ? 3 : 2;

    /* adding up accumulators, adding different order to avoid stalls to some extent.... */
    for ( l_i = l_accs; l_i > 1; l_i-- ) {
      for ( l_j = 0; l_j < i_conv_desc->ofw_rb; l_j++ ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_conv_kernel_config->instruction_set,
                                                 i_conv_kernel_config->vadd_instruction,
                                                 i_conv_kernel_config->vector_name,
                                                 i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb*l_i) + l_j,
                                                 i_conv_kernel_config->vector_reg_count - i_conv_desc->ofw_rb + l_j,
                                                 i_conv_kernel_config->vector_reg_count - i_conv_desc->ofw_rb + l_j );
      }
    }

    for ( l_j = 0; l_j < i_conv_desc->ofw_rb; l_j++ ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_conv_kernel_config->instruction_set,
                                        i_conv_kernel_config->vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_output,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        l_j * l_lead_dim * i_conv_kernel_config->datatype_size_out,
                                        i_conv_kernel_config->vector_name,
                                        i_conv_kernel_config->vector_reg_count - i_conv_desc->ofw_rb + l_j, 0, 1 );
    }
  } else {
    /* adding to C, so let's store C */
    for ( l_i = 0; l_i < i_conv_desc->ofh_rb; l_i++ ) {
      for ( l_j = 0; l_j < i_conv_desc->ofw_rb; l_j++ ) {
        for ( l_k = 0; l_k < l_reg_per_block ; l_k++ ) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_conv_kernel_config->instruction_set,
                                            i_conv_kernel_config->vmove_instruction,
                                            i_gp_reg_mapping->gp_reg_output,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            ( l_i * i_conv_desc->ofw_padded * l_lead_dim * i_conv_kernel_config->datatype_size_out) +
                                            ( l_j * l_lead_dim * i_conv_kernel_config->datatype_size_out ) +
                                            ( l_k * i_conv_kernel_config->vector_length_out * i_conv_kernel_config->datatype_size_out ),
                                            i_conv_kernel_config->vector_name,
                                            l_vec_reg_acc_start + l_k + (l_j * l_reg_per_block) + (i_conv_desc->ofw_rb * l_reg_per_block * l_i), 0, 1 );
        }
      }
    }
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_reset_x86_convolution_forward_gp_reg_mapping( libxsmm_convolution_forward_gp_reg_mapping* io_gp_reg_mapping ) {
  io_gp_reg_mapping->gp_reg_input = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_weight = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_output = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_input_pf = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_weight_pf = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_output_pf = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_kw_loop = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_kh_loop = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_ifmInner_loop = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_0 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_1 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_2 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_3 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_4 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_5 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_6 = LIBXSMM_X86_GP_REG_UNDEF;
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_reset_x86_convolution_backward_gp_reg_mapping( libxsmm_convolution_backward_gp_reg_mapping* io_gp_reg_mapping ) {
  io_gp_reg_mapping->gp_reg_input = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_weight = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_output = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_input_pf = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_weight_pf = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_output_pf = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_kw_loop = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_oi_loop = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_ofmInner_loop = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_0 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_1 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_2 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_3 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_4 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_5 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_6 = LIBXSMM_X86_GP_REG_UNDEF;
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_reset_x86_convolution_weight_update_gp_reg_mapping( libxsmm_convolution_weight_update_gp_reg_mapping* io_gp_reg_mapping ) {
  io_gp_reg_mapping->gp_reg_input = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_weight = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_output = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_input_pf = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_weight_pf = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_output_pf = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_oj_loop = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_oi_loop = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_ifmInner_loop = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_0 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_1 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_2 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_3 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_4 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_5 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_6 = LIBXSMM_X86_GP_REG_UNDEF;
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_init_convolution_kernel_config( libxsmm_convolution_kernel_config* io_conv_kernel_config ) {
  io_conv_kernel_config->instruction_set = LIBXSMM_X86_INSTR_UNDEF;
  io_conv_kernel_config->vector_reg_count = 0;
  io_conv_kernel_config->vector_length_in = 0;
  io_conv_kernel_config->datatype_size_in = 0;
  io_conv_kernel_config->vector_length_out = 0;
  io_conv_kernel_config->datatype_size_out = 0;
  io_conv_kernel_config->vector_length_wt = 0;
  io_conv_kernel_config->datatype_size_wt = 0;
  io_conv_kernel_config->vmove_instruction = LIBXSMM_X86_INSTR_UNDEF;
  io_conv_kernel_config->vfma_instruction = LIBXSMM_X86_INSTR_UNDEF;
  io_conv_kernel_config->prefetch_instruction = LIBXSMM_X86_INSTR_UNDEF;
  io_conv_kernel_config->alu_add_instruction = LIBXSMM_X86_INSTR_UNDEF;
  io_conv_kernel_config->alu_sub_instruction = LIBXSMM_X86_INSTR_UNDEF;
  io_conv_kernel_config->alu_cmp_instruction = LIBXSMM_X86_INSTR_UNDEF;
  io_conv_kernel_config->alu_jmp_instruction = LIBXSMM_X86_INSTR_UNDEF;
  io_conv_kernel_config->alu_mov_instruction = LIBXSMM_X86_INSTR_UNDEF;
  io_conv_kernel_config->vector_name = '\0';
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_backward_fma( libxsmm_generated_code*                           io_generated_code,
                                                        const libxsmm_convolution_backward_gp_reg_mapping* i_gp_reg_mapping,
                                                        const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                        const libxsmm_convolution_backward_descriptor*     i_conv_desc ) {
  libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                               i_conv_kernel_config->instruction_set,
                                               i_conv_kernel_config->vfma_instruction,
                                               1, /* use broadcast*/
                                               i_gp_reg_mapping->gp_reg_output,
                                               LIBXSMM_X86_GP_REG_UNDEF /* for not using SIB addressing */,
                                               0 /* no scale for no SIB addressing */,
                                               /* disp */(i_gp_reg_mapping->gp_reg_oi_loop * i_conv_desc->ofm_block + i_gp_reg_mapping->gp_reg_ofmInner_loop) * i_conv_kernel_config->datatype_size_out,
                                               i_conv_kernel_config->vector_name,
                                               1 /* weight */,
                                               4 /* input loaded */);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_backward_load_weight( libxsmm_generated_code*                           io_generated_code,
                                                        const libxsmm_convolution_backward_gp_reg_mapping* i_gp_reg_mapping,
                                                        const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                        const libxsmm_convolution_backward_descriptor*     i_conv_desc ) {
  /* add 5 and 6 */
  /* get the gp_reg_input to help_0 */
  libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_weight, i_gp_reg_mapping->gp_reg_help_1 );

  /* add kw's offset */
  libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_5, i_gp_reg_mapping->gp_reg_help_1);
  /* add oi's offset */
  libxsmm_x86_instruction_alu_reg( io_generated_code, i_conv_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_6, i_gp_reg_mapping->gp_reg_help_1);
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_conv_kernel_config->instruction_set,
                                          i_conv_kernel_config->vmove_instruction,
                                          i_gp_reg_mapping->gp_reg_help_1,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          0,
                                          i_conv_kernel_config->vector_name,
                                          1 , 0, 0 );
  LIBXSMM_UNUSED(i_conv_desc);
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_backward_load_input( libxsmm_generated_code*                           io_generated_code,
                                                        const libxsmm_convolution_backward_gp_reg_mapping* i_gp_reg_mapping,
                                                        const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                        const libxsmm_convolution_backward_descriptor*     i_conv_desc) {
  /* determine the number of registers needed for an ifm block */
  const unsigned int l_reg_per_block = i_conv_desc->ofm_block / i_conv_kernel_config->vector_length_in;
  /* start register of accumulator */
  const unsigned int l_vec_reg_acc_start = i_conv_kernel_config->vector_reg_count - ( i_conv_desc->ofw_rb * l_reg_per_block);
  /* register blocking counter */
  unsigned int reg_count = 0;
  unsigned int l_j, l_k;
  /* adding to C, so let's load C */
    for ( l_j = 0; l_j < i_conv_desc->ofw_rb; l_j++ ) {
      for ( l_k = 0; l_k < l_reg_per_block ; l_k++, reg_count++ ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_conv_kernel_config->instruction_set,
                                    i_conv_kernel_config->vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_input,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    (reg_count)* i_conv_kernel_config->vector_length_in * i_conv_kernel_config->datatype_size_in,
                                    i_conv_kernel_config->vector_name,
                                    l_vec_reg_acc_start + reg_count , 0, 0 );
      }
    }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_backward_store_input( libxsmm_generated_code*                           io_generated_code,
                                                        const libxsmm_convolution_backward_gp_reg_mapping* i_gp_reg_mapping,
                                                        const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                        const libxsmm_convolution_backward_descriptor*     i_conv_desc ) {

  /* determine the number of registers needed for an ofm block */
  const unsigned int l_reg_per_block = i_conv_desc->ofm_block / i_conv_kernel_config->vector_length_in;
  /* start register of accumulator */
  const unsigned int l_vec_reg_acc_start = i_conv_kernel_config->vector_reg_count - (i_conv_desc->ofw_rb * l_reg_per_block);
  /* register blocking counter */
  unsigned int reg_count = 0;
  unsigned int l_j, l_k;
  /* adding to C, so let's load C */
    for ( l_j = 0; l_j < i_conv_desc->ofw_rb; l_j++ ) {
      for ( l_k = 0; l_k < l_reg_per_block ; l_k++, reg_count++ ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_conv_kernel_config->instruction_set,
                                    i_conv_kernel_config->vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_input,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    (reg_count)  *i_conv_kernel_config->vector_length_in * i_conv_kernel_config->datatype_size_in,
                                    i_conv_kernel_config->vector_name,
                                    l_vec_reg_acc_start + reg_count , 0, 1 );
      }
    }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_weight_update_load_weight( libxsmm_generated_code*                           io_generated_code,
                                                        const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
                                                        const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                        const libxsmm_convolution_weight_update_descriptor*     i_conv_desc) {
  /* determine the number of registers needed for an ofm block */
  const unsigned int l_reg_per_block = (i_conv_desc->ifm_block == 1) ? (i_conv_desc->kw) : (i_conv_desc->ifm_block / i_conv_kernel_config->vector_length_wt);
  /* start register of accumulator */
  const unsigned int l_vec_reg_acc_start = i_conv_kernel_config->vector_reg_count - ( i_conv_desc->ifm_block * l_reg_per_block);
  /* register blocking counter */
  unsigned int reg_count = 0;
  unsigned int l_j, l_k;
  /* adding to C, so let's load C */
    for ( l_j = 0; l_j < i_conv_desc->ifm_block; l_j++ ) {
      for ( l_k = 0; l_k < l_reg_per_block ; l_k++, reg_count++ ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_conv_kernel_config->instruction_set,
                                    i_conv_kernel_config->vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_weight,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    (reg_count)*i_conv_kernel_config->vector_length_wt * i_conv_kernel_config->datatype_size_wt,
                                    i_conv_kernel_config->vector_name,
                                    l_vec_reg_acc_start + reg_count , 0, 0) ;
        if ( (i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L1 ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            LIBXSMM_X86_INSTR_PREFETCHT0 ,
                                            i_gp_reg_mapping->gp_reg_weight_pf,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    (reg_count)*i_conv_kernel_config->vector_length_wt * i_conv_kernel_config->datatype_size_wt);
        }
      }
    }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_weight_update_transpose_load_weight( libxsmm_generated_code*                           io_generated_code,
                                                        const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
                                                        const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                        const libxsmm_convolution_weight_update_descriptor*     i_conv_desc) {
  /* determine the number of registers needed for an ofm block */
  const unsigned int l_reg_per_block = (i_conv_desc->ofm_block / i_conv_kernel_config->vector_length_wt);
  /* start register of accumulator */
  const unsigned int l_vec_reg_acc_start = i_conv_kernel_config->vector_reg_count - ( i_conv_desc->ofm_block * l_reg_per_block);
  /* register blocking counter */
  unsigned int reg_count = 0;
  unsigned int l_j, l_k;
  /* adding to C, so let's load C */
    for ( l_j = 0; l_j < i_conv_desc->ofm_block; l_j++ ) {
      for ( l_k = 0; l_k < l_reg_per_block ; l_k++, reg_count++ ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_conv_kernel_config->instruction_set,
                                    i_conv_kernel_config->vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_weight,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    (reg_count)*i_conv_kernel_config->vector_length_wt * i_conv_kernel_config->datatype_size_wt,
                                    i_conv_kernel_config->vector_name,
                                    l_vec_reg_acc_start + reg_count , 0, 0 );
        if ( (i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L1 ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            LIBXSMM_X86_INSTR_PREFETCHT0 ,
                                            i_gp_reg_mapping->gp_reg_weight_pf,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    (reg_count)*i_conv_kernel_config->vector_length_wt *i_conv_kernel_config->datatype_size_wt);
        }
      }
    }
}
LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_weight_update_store_weight( libxsmm_generated_code*                           io_generated_code,
                                                        const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
                                                        const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                        const libxsmm_convolution_weight_update_descriptor*     i_conv_desc) {
  /* determine the number of registers needed for an ofm block */
  const unsigned int l_reg_per_block = (i_conv_desc->ifm_block == 1) ? (i_conv_desc->kw) : (i_conv_desc->ifm_block / i_conv_kernel_config->vector_length_wt);
  /* start register of accumulator */
  const unsigned int l_vec_reg_acc_start = i_conv_kernel_config->vector_reg_count - ( i_conv_desc->ifm_block * l_reg_per_block);
  /* register blocking counter  */
  unsigned int reg_count = 0;
  unsigned int l_j, l_k;
  /* adding to C, so let's load C */
    for ( l_j = 0; l_j < i_conv_desc->ifm_block; l_j++ ) {
      for ( l_k = 0; l_k < l_reg_per_block ; l_k++, reg_count++ ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_conv_kernel_config->instruction_set,
                                    i_conv_kernel_config->vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_weight,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    (reg_count)*i_conv_kernel_config->vector_length_wt * i_conv_kernel_config->datatype_size_wt,
                                    i_conv_kernel_config->vector_name,
                                    l_vec_reg_acc_start + reg_count , 0, 1 );
      }
    }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_weight_update_transpose_store_weight( libxsmm_generated_code*                           io_generated_code,
                                                        const libxsmm_convolution_weight_update_gp_reg_mapping* i_gp_reg_mapping,
                                                        const libxsmm_convolution_kernel_config*          i_conv_kernel_config,
                                                        const libxsmm_convolution_weight_update_descriptor*     i_conv_desc) {
  /* determine the number of registers needed for an ofm block */
  const unsigned int l_reg_per_block = (i_conv_desc->ofm_block / i_conv_kernel_config->vector_length_wt);
  /* start register of accumulator */
  const unsigned int l_vec_reg_acc_start = i_conv_kernel_config->vector_reg_count - ( i_conv_desc->ofm_block * l_reg_per_block);
  /* register blocking counter  */
  unsigned int reg_count = 0;
  unsigned int l_j, l_k;
  /* adding to C, so let's load C */
    for ( l_j = 0; l_j < i_conv_desc->ofm_block; l_j++ ) {
      for ( l_k = 0; l_k < l_reg_per_block ; l_k++, reg_count++ ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_conv_kernel_config->instruction_set,
                                    i_conv_kernel_config->vmove_instruction,
                                    i_gp_reg_mapping->gp_reg_weight,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    (reg_count)*i_conv_kernel_config->vector_length_wt *i_conv_kernel_config->datatype_size_wt,
                                    i_conv_kernel_config->vector_name,
                                    l_vec_reg_acc_start + reg_count , 0, 1 );
      }
    }
}
