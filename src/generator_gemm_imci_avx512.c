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
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include "generator_gemm_imci_avx512.h"
#include "generator_gemm_imci_microkernel.h"
#include "generator_gemm_avx512_microkernel.h"
#include "generator_gemm_common.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"

#include <libxsmm_cpuid.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

LIBXSMM_INLINE
void libxsmm_generator_gemm_imci_avx512_kernel_initialize_mask( libxsmm_generated_code*            io_generated_code,
                                                                const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                unsigned int                       i_m_done ) {
  unsigned int l_mask;

  /* init full mask */
  if ( (LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) == 0 ) {
    l_mask = 0xff;
  } else {
    l_mask = 0xffff;
  }
  /* shift right by "inverse" remainder */
  l_mask = l_mask >> ( i_micro_kernel_config->vector_length - (i_xgemm_desc->m - i_m_done) );

  /* move mask to GP register */
  libxsmm_x86_instruction_alu_imm( io_generated_code,
                               i_micro_kernel_config->alu_mov_instruction,
                               i_gp_reg_mapping->gp_reg_help_5,
                               l_mask );

  if ( i_micro_kernel_config->instruction_set == LIBXSMM_X86_IMCI ) {
    libxsmm_x86_instruction_mask_move( io_generated_code,
                                   LIBXSMM_X86_INSTR_KMOV,
                                   i_gp_reg_mapping->gp_reg_help_5,
                                   LIBXSMM_X86_IMCI_AVX512_MASK );
  } else if ( i_micro_kernel_config->instruction_set == LIBXSMM_X86_AVX512_MIC ||
              i_micro_kernel_config->instruction_set == LIBXSMM_X86_AVX512_CORE   ) {
    libxsmm_x86_instruction_mask_move( io_generated_code,
                                   LIBXSMM_X86_INSTR_KMOVW,
                                   i_gp_reg_mapping->gp_reg_help_5,
                                   LIBXSMM_X86_IMCI_AVX512_MASK );
  } else {}
}

LIBXSMM_INLINE
void libxsmm_generator_gemm_imci_avx512_kernel_mloop( libxsmm_generated_code*            io_generated_code,
                                                      libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                      const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                      const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                      const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                      const char*                        i_arch,
                                                      unsigned int                       i_n_blocking ) {
  /* set function pointers for AVX512 and IMCI */
  unsigned int (*l_generator_microkernel_kloop)( libxsmm_generated_code*, libxsmm_loop_label_tracker*, const libxsmm_gp_reg_mapping*, const libxsmm_micro_kernel_config*,
                                                 const libxsmm_gemm_descriptor*, const char*, unsigned int ) = 0;
  void (*l_generator_load)( libxsmm_generated_code*, const libxsmm_gp_reg_mapping*, const libxsmm_micro_kernel_config*,
                            const libxsmm_gemm_descriptor*, const unsigned int, const unsigned int ) = 0;
  void (*l_generator_store)( libxsmm_generated_code*, const libxsmm_gp_reg_mapping*, const libxsmm_micro_kernel_config*,
                            const libxsmm_gemm_descriptor*, const unsigned int, const unsigned int ) = 0;
  unsigned int l_k_unrolled;
  unsigned int l_m_done;

  if ( (strcmp(i_arch, "knl") == 0) ) {
    l_generator_microkernel_kloop = libxsmm_generator_gemm_avx512_kernel_kloop;
    l_generator_load = libxsmm_generator_gemm_load_C;
    l_generator_store = libxsmm_generator_gemm_store_C;
  } else if ( (strcmp(i_arch, "skx") == 0) ) {
    l_generator_microkernel_kloop = libxsmm_generator_gemm_avx512_kernel_kloop;
    l_generator_load = libxsmm_generator_gemm_load_C;
    l_generator_store = libxsmm_generator_gemm_store_C;
  } else if ( (strcmp(i_arch, "knc") == 0) ) {
    l_generator_microkernel_kloop = libxsmm_generator_gemm_imci_kernel_kloop;
    l_generator_load = libxsmm_generator_gemm_load_C_imci;
    l_generator_store = libxsmm_generator_gemm_store_C_imci;
  } else {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }

  /* we proceed as much as we can in vector length steps, remainder is handled uisng masking */
  l_m_done = (i_xgemm_desc->m / i_micro_kernel_config->vector_length) * i_micro_kernel_config->vector_length;

  /* multiples of vector_length in M */
  if (l_m_done > 0) {
    libxsmm_generator_gemm_header_mloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_micro_kernel_config->vector_length );
    l_generator_load( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
                      i_xgemm_desc, i_micro_kernel_config->vector_length, i_n_blocking );

    /* if we are generating for KNL && i_n_blocking is greater 26 && we prefetch via C -> push prefetch gpr */
    if ( (i_n_blocking > 26)          &&
         (strcmp(i_arch, "knc") != 0) &&
         (i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_BL2_VIA_C ||
          i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C ||
          i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD ||
          i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST ) ) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_b_prefetch );
    }

    l_k_unrolled = l_generator_microkernel_kloop( io_generated_code,
                                                  io_loop_label_tracker,
                                                  i_gp_reg_mapping,
                                                  i_micro_kernel_config,
                                                  i_xgemm_desc,
                                                  i_arch,
                                                  i_n_blocking );

    /* if we are generating for KNL && i_n_blocking is greater 26 && we prefetch via C -> push prefetch gpr */
    if ( (i_n_blocking > 26)          &&
         (strcmp(i_arch, "knc") != 0) &&
         (i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_BL2_VIA_C ||
          i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C ||
          i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD ||
          i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST ) ) {
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_b_prefetch );
    }

    l_generator_store( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
                       i_xgemm_desc, i_micro_kernel_config->vector_length, i_n_blocking  );
    libxsmm_generator_gemm_footer_mloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc,
                                          i_micro_kernel_config->vector_length, l_m_done, l_k_unrolled );
  }

  /* Remainder Handling using Masking, we are using M loop counter register as GP register for the mask */
  if ( l_m_done != (unsigned int)i_xgemm_desc->m ) {
    /* request masking support, @TODO performance penality here, as a new object is created */
    libxsmm_micro_kernel_config l_micro_kernel_config_mask;
    libxsmm_generator_gemm_init_micro_kernel_config_fullvector( &l_micro_kernel_config_mask, i_xgemm_desc, i_arch, 1 );

    /* initialize k1 register */
    libxsmm_generator_gemm_imci_avx512_kernel_initialize_mask( io_generated_code,
                                                                i_gp_reg_mapping,
                                                                &l_micro_kernel_config_mask,
                                                                i_xgemm_desc,
                                                                l_m_done );

    /* run masked micro kernel */
    l_generator_load( io_generated_code, i_gp_reg_mapping, &l_micro_kernel_config_mask,
                      i_xgemm_desc, l_micro_kernel_config_mask.vector_length, i_n_blocking );

    /* if we are generating for KNL && i_n_blocking is greater 26 && we prefetch via C -> push prefetch gpr */
    if ( (i_n_blocking > 26)          &&
         (strcmp(i_arch, "knc") != 0) &&
         (i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_BL2_VIA_C ||
          i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C ||
          i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD ||
          i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST ) ) {
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_b_prefetch );
    }

    l_k_unrolled = l_generator_microkernel_kloop( io_generated_code,
                                                  io_loop_label_tracker,
                                                  i_gp_reg_mapping,
                                                  &l_micro_kernel_config_mask,
                                                  i_xgemm_desc,
                                                  i_arch,
                                                  i_n_blocking );

    /* if we are generating for KNL && i_n_blocking is greater 26 && we prefetch via C -> push prefetch gpr */
    if ( (i_n_blocking > 26)          &&
         (strcmp(i_arch, "knc") != 0) &&
         (i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_BL2_VIA_C ||
          i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C ||
          i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD ||
          i_xgemm_desc->prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST ) ) {
      libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_b_prefetch );
    }

    l_generator_store( io_generated_code, i_gp_reg_mapping, &l_micro_kernel_config_mask,
                       i_xgemm_desc, l_micro_kernel_config_mask.vector_length, i_n_blocking  );

    /* adjust pointers as we don't have a m-loop body */
    /* C */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                 l_micro_kernel_config_mask.alu_add_instruction,
                                 i_gp_reg_mapping->gp_reg_c,
                                 (i_xgemm_desc->m - l_m_done) * l_micro_kernel_config_mask.datatype_size );
    /* A */
    if (l_k_unrolled == 0) {
      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                   l_micro_kernel_config_mask.alu_sub_instruction,
                                   i_gp_reg_mapping->gp_reg_a,
                                   (i_xgemm_desc->k * l_micro_kernel_config_mask.datatype_size * i_xgemm_desc->lda) - ((i_xgemm_desc->m - l_m_done) * l_micro_kernel_config_mask.datatype_size) );
    } else {
      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                   l_micro_kernel_config_mask.alu_add_instruction,
                                   i_gp_reg_mapping->gp_reg_a,
                                   ((i_xgemm_desc->m - l_m_done) * l_micro_kernel_config_mask.datatype_size) );
    }
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_gemm_imci_avx512_kernel( libxsmm_generated_code*         io_generated_code,
                                                 const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                 const char*                    i_arch ) {
  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;

  unsigned int l_number_of_chunks = 1+((i_xgemm_desc->n-1)/30);
  unsigned int l_modulo = i_xgemm_desc->n%l_number_of_chunks;
  unsigned int l_n2 = i_xgemm_desc->n/l_number_of_chunks;
  unsigned int l_n1 = l_n2 + 1;
  unsigned int l_N2 = 0;
  unsigned int l_N1 = 0;
  unsigned int l_chunk = 0;

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
  libxsmm_generator_gemm_init_micro_kernel_config_fullvector( &l_micro_kernel_config, i_xgemm_desc, i_arch, 0 );

  if (l_n1 > 30) l_n1 = 30; /* this just the case if i_xgemm_desc->n/l_number_of_chunks has no remainder */
  for (l_chunk = 0; l_chunk < l_number_of_chunks; l_chunk++) {
    if (l_chunk < l_modulo) {
      l_N1 += l_n1;
    } else {
      l_N2 += l_n2;
    }
  }

  /* printf("N splitting of DP AVX512 Kernel: %i %i %i %i\n", l_N1, l_N2, l_n1, l_n2); */

  /* open asm */
  libxsmm_x86_instruction_open_stream( io_generated_code, &l_gp_reg_mapping, i_arch, i_xgemm_desc->prefetch );

  if (l_number_of_chunks == 1) {
    libxsmm_generator_gemm_imci_avx512_kernel_mloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config,
                                                      i_xgemm_desc, i_arch, i_xgemm_desc->n);
  } else {
    if ((l_N2 > 0) && (l_N1 > 0)) {
      libxsmm_generator_gemm_header_nloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, l_n1 );
      libxsmm_generator_gemm_imci_avx512_kernel_mloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config,
                                                        i_xgemm_desc, i_arch, l_n1);
      libxsmm_generator_gemm_footer_nloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc, l_n1, l_N1 );

      libxsmm_generator_gemm_header_nloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, l_n2 );
      libxsmm_generator_gemm_imci_avx512_kernel_mloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config,
                                                        i_xgemm_desc, i_arch, l_n2);
      libxsmm_generator_gemm_footer_nloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc, l_n2, i_xgemm_desc->n );
    } else if ((l_N2 > 0) && (l_N1 == 0)) {
      libxsmm_generator_gemm_header_nloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, l_n2 );
      libxsmm_generator_gemm_imci_avx512_kernel_mloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config,
                                                        i_xgemm_desc, i_arch, l_n2);
      libxsmm_generator_gemm_footer_nloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc, l_n2, i_xgemm_desc->n );
    } else {}
  }

  /* close asm */
  libxsmm_x86_instruction_close_stream( io_generated_code, &l_gp_reg_mapping, i_arch, i_xgemm_desc->prefetch );
}

