/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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
/* Kunal Banerjee (Intel Corp.), Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include "generator_convolution_winograd_weight_update_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"

#include <libxsmm_cpuid.h>
#include <libxsmm_macros.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_convolution_winograd_weight_update_avx512( libxsmm_generated_code*                        io_generated_code,
                                                                  const libxsmm_convolution_winograd_descriptor* i_conv_desc,
                                                                  const char*                                    i_arch ) {
  /* @TODO: switch to libxsmm_convolution_kernel_config? */
  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  /* @TODO: create a libxsmm_convolution_winograd_gp_reg_mapping which can be shared for forward, weight update and backward? */
  libxsmm_gp_reg_mapping l_gp_reg_mapping;

  unsigned int l_b_reg;
  unsigned int l_b_idx = 0;
  unsigned int l_scale = 0;
  unsigned int index;
  unsigned int max_index;
  unsigned int next_index;
  unsigned int urf;
  unsigned int offset;
  unsigned int offset2;
  unsigned int m_dist;
  unsigned int l_qinstr = 0;
  unsigned int qfac   = 0;
  unsigned int qindex = 0;
  int toggle;
  int lookahead = 32;

  /* define gp register mapping */
  libxsmm_reset_x86_gp_reg_mapping( &l_gp_reg_mapping );
  /* matching calling convention on Linux */
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R8;
  l_gp_reg_mapping.gp_reg_mloop = LIBXSMM_X86_GP_REG_R12;
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_X86_GP_REG_R13;
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_X86_GP_REG_R14;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_X86_GP_REG_R15; /* masking */
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_X86_GP_REG_RAX; /* B stride helper */
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_X86_GP_REG_RBX; /* B stride helper */
  l_gp_reg_mapping.gp_reg_help_3 = LIBXSMM_X86_GP_REG_R9;  /* B stride helper */
  l_gp_reg_mapping.gp_reg_help_4 = LIBXSMM_X86_GP_REG_R10; /* B stride helper */
  l_gp_reg_mapping.gp_reg_help_5 = LIBXSMM_X86_GP_REG_R11; /* B stride helper */

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define the micro kernel code gen properties */
  if ( strcmp( i_arch, "knm" ) == 0 ) {
    l_micro_kernel_config.instruction_set = LIBXSMM_X86_AVX512_KNM;
  } else if ((strcmp( i_arch, "knl" ) == 0)) {
    l_micro_kernel_config.instruction_set = LIBXSMM_X86_AVX512_MIC;
  } else {
    l_micro_kernel_config.instruction_set = LIBXSMM_X86_AVX512_CORE;
  }
  l_micro_kernel_config.vector_reg_count = 32;
  l_micro_kernel_config.use_masking_a_c = 0;
  l_micro_kernel_config.vector_name = 'z';
  l_micro_kernel_config.vector_length = 16;
  l_micro_kernel_config.datatype_size = 4;
  l_micro_kernel_config.a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
  l_micro_kernel_config.b_vmove_instruction = LIBXSMM_X86_INSTR_VBROADCASTSS;
  l_micro_kernel_config.b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
  l_micro_kernel_config.c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
  l_micro_kernel_config.vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
  l_micro_kernel_config.vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
  l_micro_kernel_config.vadd_instruction = LIBXSMM_X86_INSTR_VADDPS;
  l_micro_kernel_config.prefetch_instruction = LIBXSMM_X86_INSTR_PREFETCHT1;
  l_micro_kernel_config.alu_add_instruction = LIBXSMM_X86_INSTR_ADDQ;
  l_micro_kernel_config.alu_sub_instruction = LIBXSMM_X86_INSTR_SUBQ;
  l_micro_kernel_config.alu_cmp_instruction = LIBXSMM_X86_INSTR_CMPQ;
  l_micro_kernel_config.alu_jmp_instruction = LIBXSMM_X86_INSTR_JL;
  l_micro_kernel_config.alu_mov_instruction = LIBXSMM_X86_INSTR_MOVQ;
  l_qinstr = LIBXSMM_X86_INSTR_V4FMADDPS;

  /*printf("\nGenerating assembly code with values:: itiles:%d jtiles:%d bimg:%d ur:%d\n", i_conv_desc->itiles, i_conv_desc->jtiles, i_conv_desc->bimg, i_conv_desc->ur);*/

  /* open asm */
  libxsmm_x86_instruction_open_stream_convolution( io_generated_code, l_gp_reg_mapping.gp_reg_a,
                                                   l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_c,
                                                   l_gp_reg_mapping.gp_reg_a_prefetch, l_gp_reg_mapping.gp_reg_b_prefetch,
                                                   LIBXSMM_X86_GP_REG_R9, i_arch );

  m_dist = l_micro_kernel_config.vector_length*l_micro_kernel_config.datatype_size;

  if ( l_micro_kernel_config.instruction_set == LIBXSMM_X86_AVX512_KNM && i_conv_desc->bimg*i_conv_desc->jtiles*i_conv_desc->itiles % (i_conv_desc->ur*4) == 0 ) {
    qfac = 4;
  } else {
    qfac = 1;
  }

  max_index = i_conv_desc->ur - 1;

  /* Initialize helper registers for SIB addressing */
  if ( max_index >= 1 ) {
    /* helper 0: Index register holding ldb*datatype_size */
    libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                     l_gp_reg_mapping.gp_reg_help_0, l_micro_kernel_config.datatype_size * l_micro_kernel_config.vector_length * qfac );
  }
  if ( max_index >= 3 )
  {
    /* helper 1: Index register holding 3*ldb*datatype_size */
    libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                     l_gp_reg_mapping.gp_reg_help_1, l_micro_kernel_config.datatype_size * l_micro_kernel_config.vector_length * 3 * qfac );
  }
  if ( max_index >= 5 )
  {
    /* helper 2: Index register holding 5*ldb*datatype_size */
    libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                     l_gp_reg_mapping.gp_reg_help_2, l_micro_kernel_config.datatype_size * l_micro_kernel_config.vector_length * 5 * qfac );
  }
  if ( max_index >= 7 )
  {
    /* helper 3: Index register holding 7*ldb*datatype_size */
    libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                     l_gp_reg_mapping.gp_reg_help_3, l_micro_kernel_config.datatype_size * l_micro_kernel_config.vector_length * 7 * qfac );
  }

  for ( urf = 0; urf < l_micro_kernel_config.vector_length; urf++ ) {
    offset = urf*l_micro_kernel_config.vector_length*l_micro_kernel_config.datatype_size;
    libxsmm_x86_instruction_vec_move( io_generated_code,
                                      l_micro_kernel_config.instruction_set,
                                      l_micro_kernel_config.c_vmove_instruction,
                                      l_gp_reg_mapping.gp_reg_c,
                                      LIBXSMM_X86_GP_REG_UNDEF,
                                      0,
                                      offset,
                                      l_micro_kernel_config.vector_name,
                                      urf,
                                      0,
                                      0 );
    if ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_ALL) == LIBXSMM_CONVOLUTION_PREFETCH_ALL) {
      libxsmm_x86_instruction_prefetch( io_generated_code,
                                        l_micro_kernel_config.prefetch_instruction,
                                        l_gp_reg_mapping.gp_reg_c,
                                        LIBXSMM_X86_GP_REG_UNDEF,
                                        0,
                                        offset + i_conv_desc->bimg*i_conv_desc->itiles*i_conv_desc->jtiles*lookahead/2 );
    }
  }

  if ( i_conv_desc->itiles*i_conv_desc->jtiles*i_conv_desc->bimg/qfac > i_conv_desc->ur ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       l_gp_reg_mapping.gp_reg_mloop, 0 );
      libxsmm_x86_instruction_register_jump_label( io_generated_code, &l_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction,
                                       l_gp_reg_mapping.gp_reg_mloop, i_conv_desc->ur);
  }

  if ( max_index >= 9 ) {
    /* helper 4: B + 9*ldb, additional base address */
    libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                     l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_4);
    libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction,
                                     l_gp_reg_mapping.gp_reg_help_4,  9 * l_micro_kernel_config.datatype_size * l_micro_kernel_config.vector_length * qfac );
  }
  if ( max_index >= 18 ) {
    /* helper 5: B + 18*ldb, additional base address */
    libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                     l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_5);
    libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction,
                                     l_gp_reg_mapping.gp_reg_help_5, 18 * l_micro_kernel_config.datatype_size * l_micro_kernel_config.vector_length * qfac );
  }

  for ( qindex = 0; qindex < qfac; qindex++ ) {
    offset = qindex*l_micro_kernel_config.vector_length*l_micro_kernel_config.datatype_size;
    libxsmm_x86_instruction_vec_move( io_generated_code,
                                      l_micro_kernel_config.instruction_set,
                                      l_micro_kernel_config.a_vmove_instruction,
                                      l_gp_reg_mapping.gp_reg_a,
                                      LIBXSMM_X86_GP_REG_UNDEF,
                                      0,
                                      offset,
                                      l_micro_kernel_config.vector_name,
                                      l_micro_kernel_config.vector_length + qindex,
                                      0,
                                      0 );
    if ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_ALL) == LIBXSMM_CONVOLUTION_PREFETCH_ALL) {
      libxsmm_x86_instruction_prefetch( io_generated_code,
                                        l_micro_kernel_config.prefetch_instruction,
                                        l_gp_reg_mapping.gp_reg_a,
                                        LIBXSMM_X86_GP_REG_UNDEF,
                                        0,
                                        offset + i_conv_desc->bimg*i_conv_desc->itiles*i_conv_desc->jtiles*lookahead );
      if (qindex == 0) {
        libxsmm_x86_instruction_prefetch( io_generated_code,
                                          l_micro_kernel_config.prefetch_instruction,
                                          l_gp_reg_mapping.gp_reg_b,
                                          LIBXSMM_X86_GP_REG_UNDEF,
                                          0,
                                          i_conv_desc->bimg*i_conv_desc->itiles*i_conv_desc->jtiles*lookahead );
      }
    }
  }

  toggle = 0;

  for ( index = 0; index < i_conv_desc->ur; index++ ) {
    /* select the base register */
    if ( index > 17 ) {
      l_b_reg = l_gp_reg_mapping.gp_reg_help_5;
    } else if ( index > 8 ) {
      l_b_reg = l_gp_reg_mapping.gp_reg_help_4;
    } else {
      l_b_reg = l_gp_reg_mapping.gp_reg_b;
    }
    /* Select SIB */
    if ( index % 9 == 0 ) {
      l_b_idx = LIBXSMM_X86_GP_REG_UNDEF;
      l_scale = 0;
    } else if ( index % 9 == 1 ) {
      l_b_idx = l_gp_reg_mapping.gp_reg_help_0;
      l_scale = 1;
    } else if ( index % 9 == 2 ) {
      l_b_idx = l_gp_reg_mapping.gp_reg_help_0;
      l_scale = 2;
    } else if ( index % 9 == 3 ) {
      l_b_idx = l_gp_reg_mapping.gp_reg_help_1;
      l_scale = 1;
    } else if ( index % 9 == 4 ) {
      l_b_idx = l_gp_reg_mapping.gp_reg_help_0;
      l_scale = 4;
    } else if ( index % 9 == 5 ) {
      l_b_idx = l_gp_reg_mapping.gp_reg_help_2;
      l_scale = 1;
    } else if ( index % 9 == 6 ) {
      l_b_idx = l_gp_reg_mapping.gp_reg_help_1;
      l_scale = 2;
    } else if ( index % 9 == 7 ) {
      l_b_idx = l_gp_reg_mapping.gp_reg_help_3;
      l_scale = 1;
    } else if ( index % 9 == 8 ) {
      l_b_idx = l_gp_reg_mapping.gp_reg_help_0;
      l_scale = 8;
    }
#if !defined(NDEBUG)
    else {
       assert(0/*should not happen*/);
       l_b_idx = 0;
       l_scale = 0;
    }
#endif
    for ( urf = 0; urf < l_micro_kernel_config.vector_length; urf++ ) {
      if ( index < 27 ) {
        offset2 = urf*l_micro_kernel_config.datatype_size*qfac;
        if ( l_micro_kernel_config.instruction_set == LIBXSMM_X86_AVX512_KNM && 4 == qfac ) {
          libxsmm_x86_instruction_vec_compute_qfma( io_generated_code,
                                                    l_micro_kernel_config.instruction_set,
                                                    l_qinstr,
                                                    l_b_reg,
                                                    l_b_idx,
                                                    l_scale,
                                                    offset2,
                                                    l_micro_kernel_config.vector_name,
                                                    l_micro_kernel_config.vector_length + toggle*qfac,
                                                    urf );
        } else {
          libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                                   l_micro_kernel_config.instruction_set,
                                                   l_micro_kernel_config.vmul_instruction, /* vmul == vfma in KNL */
                                                   1, /* use broadcast*/
                                                   l_b_reg,
                                                   l_b_idx,
                                                   l_scale,
                                                   offset2,
                                                   l_micro_kernel_config.vector_name,
                                                   l_micro_kernel_config.vector_length + toggle*qfac,
                                                   urf );
        }
      } else {
        /* The following statements are for direct offset computation */
        offset2 = (m_dist*index + urf*l_micro_kernel_config.datatype_size)*qfac;
        if ( l_micro_kernel_config.instruction_set == LIBXSMM_X86_AVX512_KNM && 4 == qfac ) {
          libxsmm_x86_instruction_vec_compute_qfma( io_generated_code,
                                                    l_micro_kernel_config.instruction_set,
                                                    l_qinstr,
                                                    l_gp_reg_mapping.gp_reg_b,
                                                    LIBXSMM_X86_GP_REG_UNDEF, /* for not using SIB addressing */
                                                    0, /* no scale for no SIB addressing */
                                                    offset2,
                                                    l_micro_kernel_config.vector_name,
                                                    l_micro_kernel_config.vector_length + toggle*qfac,
                                                    urf );
        } else {
          libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                                   l_micro_kernel_config.instruction_set,
                                                   l_micro_kernel_config.vmul_instruction,
                                                   1, /* use broadcast*/
                                                   l_gp_reg_mapping.gp_reg_b,
                                                   LIBXSMM_X86_GP_REG_UNDEF, /* for not using SIB addressing */
                                                   0, /* no scale for no SIB addressing */
                                                   offset2,
                                                   l_micro_kernel_config.vector_name,
                                                   l_micro_kernel_config.vector_length + toggle*qfac,
                                                   urf );
        }
      }
    }

    toggle = !toggle;
    if ( index < (i_conv_desc->ur - 1) ) {
      next_index = index + 1;
    } else {
      next_index = 0;
    }

    if ( next_index != 0 ) {
      for ( qindex = 0; qindex < qfac; qindex++ ) {
        offset = m_dist*(next_index*qfac + qindex);
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          l_micro_kernel_config.instruction_set,
                                          l_micro_kernel_config.a_vmove_instruction,
                                          l_gp_reg_mapping.gp_reg_a,
                                          LIBXSMM_X86_GP_REG_UNDEF,
                                          0,
                                          offset,
                                          l_micro_kernel_config.vector_name,
                                          l_micro_kernel_config.vector_length + toggle*qfac + qindex,
                                          0,
                                          0 );
        if ((i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_ALL) == LIBXSMM_CONVOLUTION_PREFETCH_ALL) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            l_micro_kernel_config.prefetch_instruction,
                                            l_gp_reg_mapping.gp_reg_a,
                                            LIBXSMM_X86_GP_REG_UNDEF,
                                            0,
                                            offset + i_conv_desc->bimg*i_conv_desc->itiles*i_conv_desc->jtiles*lookahead );
          if (qindex == 0) {
            libxsmm_x86_instruction_prefetch( io_generated_code,
                                              l_micro_kernel_config.prefetch_instruction,
                                              l_gp_reg_mapping.gp_reg_b,
                                              LIBXSMM_X86_GP_REG_UNDEF,
                                              0,
                                              m_dist*next_index + i_conv_desc->bimg*i_conv_desc->itiles*i_conv_desc->jtiles*lookahead );
          }
        }
      }
    }
  }/* end for tm */

  if ( i_conv_desc->itiles*i_conv_desc->jtiles*i_conv_desc->bimg/qfac > i_conv_desc->ur ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction,
                                     l_gp_reg_mapping.gp_reg_a, i_conv_desc->ur*l_micro_kernel_config.vector_length*l_micro_kernel_config.datatype_size*qfac );
    libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction,
                                     l_gp_reg_mapping.gp_reg_b, i_conv_desc->ur*l_micro_kernel_config.vector_length*l_micro_kernel_config.datatype_size*qfac );
    libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_cmp_instruction,
                                     l_gp_reg_mapping.gp_reg_mloop, i_conv_desc->itiles*i_conv_desc->jtiles*i_conv_desc->bimg/qfac );
    libxsmm_x86_instruction_jump_back_to_label( io_generated_code, l_micro_kernel_config.alu_jmp_instruction, &l_loop_label_tracker );
  }

  for ( urf = 0; urf < l_micro_kernel_config.vector_length; urf++ ) {
    offset = urf*l_micro_kernel_config.vector_length*l_micro_kernel_config.datatype_size;
    libxsmm_x86_instruction_vec_move( io_generated_code,
                                      l_micro_kernel_config.instruction_set,
                                      l_micro_kernel_config.c_vmove_instruction,
                                      l_gp_reg_mapping.gp_reg_c,
                                      LIBXSMM_X86_GP_REG_UNDEF,
                                      0,
                                      offset,
                                      l_micro_kernel_config.vector_name,
                                      urf,
                                      0,
                                      1 ); /* store */
  }

  /* close asm */
  libxsmm_x86_instruction_close_stream_convolution( io_generated_code, i_arch );
}
