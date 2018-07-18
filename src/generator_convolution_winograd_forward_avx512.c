/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
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
/* Kunal Banerjee, Alexander Heinecke, Jongsoo Park (Intel Corp.)
******************************************************************************/

#include "generator_convolution_winograd_forward_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include <libxsmm_cpuid.h>

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
void libxsmm_generator_convolution_winograd_forward_avx512( libxsmm_generated_code*                        io_generated_code,
                                                            const libxsmm_convolution_winograd_descriptor* i_conv_desc,
                                                            const char*                                    i_arch ) {
  /* @TODO: switch to libxsmm_convolution_kernel_config? */
  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  /* @TODO: create a libxsmm_convolution_winograd_gp_reg_mapping which can be shared for forward, weight update and backward? */
  libxsmm_gp_reg_mapping l_gp_reg_mapping;

  unsigned int l_b_reg = 0;
  unsigned int l_b_idx = 0;
  unsigned int l_scale = 0;
  unsigned int idx     = 0;
  unsigned int max_idx = 0;
  unsigned int ifm     = 0;
  unsigned int offset  = 0;
  unsigned int boffset = 0;
  unsigned int constoffset = 0;
  unsigned int m_dist = 0;
  unsigned int num_regs = 0;
  unsigned int l_qinstr = 0;
  unsigned int qfac   = 0;
  unsigned int qindex = 0;
  int reg  = 0;
  int wreg = 0;
  int nprefetches_per_group, is_epilogue;
  unsigned int ur;

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
  } else if ( strcmp( i_arch, "knl" ) == 0 ) {
    l_micro_kernel_config.instruction_set = LIBXSMM_X86_AVX512_MIC;
  } else if ( strcmp( i_arch, "skx" ) == 0 ) {
    l_micro_kernel_config.instruction_set = LIBXSMM_X86_AVX512_CORE;
  } else if ( strcmp( i_arch, "icl" ) == 0 ) {
    l_micro_kernel_config.instruction_set = LIBXSMM_X86_AVX512_ICL;
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
    return;
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

  /*printf("\nGenerating assembly code with values:: itiles:%d jtiles:%d bimg:%d ur:%d ur_ifm:%d\n", i_conv_desc->itiles, i_conv_desc->jtiles, i_conv_desc->bimg, i_conv_desc->ur, i_conv_desc->ur_ifm);*/

  /* open asm */
  libxsmm_x86_instruction_open_stream_convolution( io_generated_code, l_gp_reg_mapping.gp_reg_a,
                                                   l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_c,
                                                   l_gp_reg_mapping.gp_reg_a_prefetch, l_gp_reg_mapping.gp_reg_b_prefetch,
                                                   LIBXSMM_X86_GP_REG_R9, i_arch );

  m_dist = l_micro_kernel_config.vector_length*l_micro_kernel_config.datatype_size;
  num_regs = l_micro_kernel_config.vector_reg_count - 2;

  if ( l_micro_kernel_config.instruction_set == LIBXSMM_X86_AVX512_KNM ) {
    qfac = 4;
  } else {
    qfac = 1;
  }

  max_idx = i_conv_desc->ur - 1;

  if ( l_micro_kernel_config.instruction_set != LIBXSMM_X86_AVX512_KNM ) {
    /* Initialize helper registers for SIB addressing */
    if ( max_idx >= 1 ) {
      /* helper 0: Index register holding ldb*datatype_size */
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       l_gp_reg_mapping.gp_reg_help_0,
                                       i_conv_desc->ur_ifm * m_dist );
    }
    if ( max_idx >= 3 ) {
      /* helper 1: Index register holding 3*ldb*datatype_size */
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       l_gp_reg_mapping.gp_reg_help_1,
                                       i_conv_desc->ur_ifm * m_dist * 3 );
    }
    if ( max_idx >= 5 ) {
      /* helper 2: Index register holding 5*ldb*datatype_size */
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       l_gp_reg_mapping.gp_reg_help_2,
                                       i_conv_desc->ur_ifm * m_dist * 5 );
    }
    if ( max_idx >= 7 ) {
      /* helper 3: Index register holding 7*ldb*datatype_size */
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       l_gp_reg_mapping.gp_reg_help_3,
                                       i_conv_desc->ur_ifm * m_dist * 7 );
    }
  } /*end if helper registers*/

  for ( is_epilogue = 0; is_epilogue < 2; ++is_epilogue ) {
    int remainder = (i_conv_desc->itiles*i_conv_desc->jtiles*i_conv_desc->bimg) % i_conv_desc->ur;
    if ( is_epilogue && 0 == remainder ) break;

    if ( !is_epilogue ) {
      if ( i_conv_desc->itiles*i_conv_desc->jtiles*i_conv_desc->bimg > i_conv_desc->ur ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                         l_gp_reg_mapping.gp_reg_mloop, 0 );
        libxsmm_x86_instruction_register_jump_back_label( io_generated_code, &l_loop_label_tracker );
        libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction,
                                         l_gp_reg_mapping.gp_reg_mloop, i_conv_desc->ur);
      }
    }

    ur = (0 != is_epilogue ? ((unsigned int)remainder) : i_conv_desc->ur);

    for ( idx = 0; idx < ur; idx++ ) { /* load output */
      offset = m_dist*idx;
      reg = num_regs - ur + idx;
      if ( i_conv_desc->ur_ifm == i_conv_desc->blocks_ifm ) {
        /* when we process an ofm block in one shot, don't need to read
         * the current value of the ofm block to save BW */
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               l_micro_kernel_config.instruction_set,
                                               l_micro_kernel_config.vxor_instruction,
                                               l_micro_kernel_config.vector_name,
                                               reg,
                                               reg,
                                               reg );
      }
      else {
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          l_micro_kernel_config.instruction_set,
                                          l_micro_kernel_config.c_vmove_instruction,
                                          l_gp_reg_mapping.gp_reg_c,
                                          LIBXSMM_X86_GP_REG_UNDEF,
                                          0,
                                          offset,
                                          l_micro_kernel_config.vector_name,
                                          reg,
                                          0, 1,
                                          0 );
      }
    }

    if ( i_conv_desc->ur_ifm > 1 ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                       l_gp_reg_mapping.gp_reg_nloop, 0 );
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, &l_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction,
                                       l_gp_reg_mapping.gp_reg_nloop, 1);
    }

    if ( l_micro_kernel_config.instruction_set != LIBXSMM_X86_AVX512_KNM ) {
      if ( max_idx >= 9 ) {
        /* helper 4: B + 9*ldb, additional base address */
        libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                         l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_4 );
        libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction,
                                         l_gp_reg_mapping.gp_reg_help_4,
                                         i_conv_desc->ur_ifm * 9 * l_micro_kernel_config.datatype_size * l_micro_kernel_config.vector_length );
      }
      if ( max_idx >= 18 ) {
        /* helper 5: B + 18*ldb, additional base address */
        libxsmm_x86_instruction_alu_reg( io_generated_code, l_micro_kernel_config.alu_mov_instruction,
                                         l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_5 );
        libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction,
                                         l_gp_reg_mapping.gp_reg_help_5,
                                         i_conv_desc->ur_ifm * 18 * l_micro_kernel_config.datatype_size * l_micro_kernel_config.vector_length );
      }
    } /*end if helper registers*/

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
                                        qindex,
                                        0, 1,
                                        0 );
    }

    wreg = 0;
    for ( ifm =0; ifm < l_micro_kernel_config.vector_length; ifm+=qfac ) {
      boffset = ifm*l_micro_kernel_config.datatype_size;
      wreg = 1 - wreg;
      if ( (ifm + qfac) < l_micro_kernel_config.vector_length )
      {
        for ( qindex = 0; qindex < qfac; qindex++ ) {
          offset = (ifm + qfac)*l_micro_kernel_config.vector_length*l_micro_kernel_config.datatype_size + qindex*l_micro_kernel_config.vector_length*l_micro_kernel_config.datatype_size;
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            l_micro_kernel_config.instruction_set,
                                            l_micro_kernel_config.a_vmove_instruction,
                                            l_gp_reg_mapping.gp_reg_a,
                                            LIBXSMM_X86_GP_REG_UNDEF,
                                            0,
                                            offset,
                                            l_micro_kernel_config.vector_name,
                                            wreg*qfac + qindex,
                                            0, 1,
                                            0 );
        }
      }

      /* sprinkle prefetches to avoid having too many outstanding prefetches at the same time */
      nprefetches_per_group = (ur + l_micro_kernel_config.vector_length/qfac)/(l_micro_kernel_config.vector_length/qfac);
      for ( idx = ifm/qfac*nprefetches_per_group; idx < LIBXSMM_MIN((ifm/qfac + 1)*nprefetches_per_group, ur); idx++ ) {
        if ( i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L2 ) {
          /* prefetch next input img when we're working on the last ofm block of the current image */
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            LIBXSMM_X86_INSTR_PREFETCHT1,
                                            l_gp_reg_mapping.gp_reg_b,
                                            LIBXSMM_X86_GP_REG_UNDEF,
                                            0,
                                            (i_conv_desc->ur_ifm*idx + i_conv_desc->alpha*i_conv_desc->alpha*i_conv_desc->itiles*i_conv_desc->jtiles*i_conv_desc->bimg*i_conv_desc->ur_ifm)*m_dist );
        }
        if ( i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1 ) { /* prefetch next ur block */
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            LIBXSMM_X86_INSTR_PREFETCHT0,
                                            l_gp_reg_mapping.gp_reg_b,
                                            LIBXSMM_X86_GP_REG_UNDEF,
                                            0,
                                            (i_conv_desc->ur_ifm*idx + i_conv_desc->ur*i_conv_desc->ur_ifm)*m_dist );
        }
      }

      for ( idx = 0; idx < ur; idx++ ) {
        reg = num_regs - ur + idx;

        if ( idx > 27 || l_micro_kernel_config.instruction_set == LIBXSMM_X86_AVX512_KNM ) {
#if !defined(NDEBUG)
          if ( ifm == 0 ) {
            fprintf(stderr, "LIBXSMM warning: Not using optimal blocking.. >8 byte fma generated...idx = %u\n", idx);
          }
#endif
          constoffset = boffset + m_dist*i_conv_desc->ur_ifm*idx;
          if ( l_micro_kernel_config.instruction_set == LIBXSMM_X86_AVX512_KNM ) {
            libxsmm_x86_instruction_vec_compute_qfma( io_generated_code,
                                                      l_micro_kernel_config.instruction_set,
                                                      l_qinstr,
                                                      l_gp_reg_mapping.gp_reg_b,
                                                      LIBXSMM_X86_GP_REG_UNDEF,
                                                      0,
                                                      constoffset,
                                                      l_micro_kernel_config.vector_name,
                                                      (1 - wreg)*4,
                                                      reg );
          } else {
            libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                                     l_micro_kernel_config.instruction_set,
                                                     l_micro_kernel_config.vmul_instruction,
                                                     1, /* use broadcast*/
                                                     l_gp_reg_mapping.gp_reg_b,
                                                     LIBXSMM_X86_GP_REG_UNDEF,
                                                     0,
                                                     constoffset,
                                                     l_micro_kernel_config.vector_name,
                                                     1 - wreg,
                                                     reg );
          }
        } else {
          /* select the base register */
          if ( idx > 17 ) {
            l_b_reg = l_gp_reg_mapping.gp_reg_help_5;
          } else if ( idx > 8 ) {
            l_b_reg = l_gp_reg_mapping.gp_reg_help_4;
          } else {
            l_b_reg = l_gp_reg_mapping.gp_reg_b;
          }
          /* Select SIB */
          if ( idx % 9 == 0 ) {
            l_b_idx = LIBXSMM_X86_GP_REG_UNDEF;
            l_scale = 0;
          } else if ( idx % 9 == 1 ) {
            l_b_idx = l_gp_reg_mapping.gp_reg_help_0;
            l_scale = 1;
          } else if ( idx % 9 == 2 ) {
            l_b_idx = l_gp_reg_mapping.gp_reg_help_0;
            l_scale = 2;
          } else if ( idx % 9 == 3 ) {
            l_b_idx = l_gp_reg_mapping.gp_reg_help_1;
            l_scale = 1;
          } else if ( idx % 9 == 4 ) {
            l_b_idx = l_gp_reg_mapping.gp_reg_help_0;
            l_scale = 4;
          } else if ( idx % 9 == 5 ) {
            l_b_idx = l_gp_reg_mapping.gp_reg_help_2;
            l_scale = 1;
          } else if ( idx % 9 == 6 ) {
            l_b_idx = l_gp_reg_mapping.gp_reg_help_1;
            l_scale = 2;
          } else if ( idx % 9 == 7 ) {
            l_b_idx = l_gp_reg_mapping.gp_reg_help_3;
            l_scale = 1;
          } else if ( idx % 9 == 8 ) {
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
          if ( l_micro_kernel_config.instruction_set == LIBXSMM_X86_AVX512_KNM ) {
            libxsmm_x86_instruction_vec_compute_qfma( io_generated_code,
                                                      l_micro_kernel_config.instruction_set,
                                                      l_qinstr,
                                                      l_b_reg,
                                                      l_b_idx,
                                                      l_scale,
                                                      boffset,
                                                      l_micro_kernel_config.vector_name,
                                                      (1 - wreg)*4,
                                                      reg );
           } else {
             libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                                      l_micro_kernel_config.instruction_set,
                                                      l_micro_kernel_config.vmul_instruction,
                                                      1,
                                                      l_b_reg,
                                                      l_b_idx,
                                                      l_scale,
                                                      boffset,
                                                      l_micro_kernel_config.vector_name,
                                                      1 - wreg,
                                                      reg );
          }
        }
      }
    }

    if ( i_conv_desc->ur_ifm > 1 ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_b,
                                       m_dist);
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction,
                                       l_gp_reg_mapping.gp_reg_a, l_micro_kernel_config.vector_length*m_dist);
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_cmp_instruction,
                                       l_gp_reg_mapping.gp_reg_nloop, i_conv_desc->ur_ifm );
      libxsmm_x86_instruction_jump_back_to_label( io_generated_code, l_micro_kernel_config.alu_jmp_instruction, &l_loop_label_tracker );
    }

    for ( idx = 0; idx < ur; idx++ ) {
      offset = m_dist*idx;
      reg = num_regs - ur + idx;
      if ( (unsigned int)reg < l_micro_kernel_config.vector_reg_count ) {
        if ( i_conv_desc->ur_ifm == i_conv_desc->blocks_ifm ) {
          /* use stream store when ur_ifm == blocks_ifm */
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            l_micro_kernel_config.instruction_set,
                                            LIBXSMM_X86_INSTR_VMOVNTPS,
                                            l_gp_reg_mapping.gp_reg_c,
                                            LIBXSMM_X86_GP_REG_UNDEF,
                                            0,
                                            offset,
                                            l_micro_kernel_config.vector_name,
                                            reg,
                                            0, 0,
                                            1 ); /* store */
        }
        else {
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            l_micro_kernel_config.instruction_set,
                                            l_micro_kernel_config.c_vmove_instruction,
                                            l_gp_reg_mapping.gp_reg_c,
                                            LIBXSMM_X86_GP_REG_UNDEF,
                                            0,
                                            offset,
                                            l_micro_kernel_config.vector_name,
                                            reg,
                                            0, 0,
                                            1 ); /* store */
        }
      }
    }

    if ( !is_epilogue ) {
      if ( i_conv_desc->itiles*i_conv_desc->jtiles*i_conv_desc->bimg > i_conv_desc->ur ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction,
                                         l_gp_reg_mapping.gp_reg_c, i_conv_desc->ur*l_micro_kernel_config.vector_length*l_micro_kernel_config.datatype_size );
        if ( i_conv_desc->ur_ifm == 1 ) {
          libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction,
                                           l_gp_reg_mapping.gp_reg_b,
                                           i_conv_desc->ur*i_conv_desc->ur_ifm*m_dist );
        } else {
          libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction,
                                           l_gp_reg_mapping.gp_reg_b, (i_conv_desc->ur - 1)*i_conv_desc->ur_ifm*m_dist );
          libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_sub_instruction,
                                           l_gp_reg_mapping.gp_reg_a, i_conv_desc->ur_ifm*l_micro_kernel_config.vector_length*m_dist);
        }
        libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_cmp_instruction,
                                         l_gp_reg_mapping.gp_reg_mloop, (i_conv_desc->itiles*i_conv_desc->jtiles*i_conv_desc->bimg)/i_conv_desc->ur*i_conv_desc->ur );
        libxsmm_x86_instruction_jump_back_to_label( io_generated_code, l_micro_kernel_config.alu_jmp_instruction, &l_loop_label_tracker );
      }
    }
  } /* is_epilogue */

  /* close asm */
  libxsmm_x86_instruction_close_stream_convolution( io_generated_code, i_arch );
}
