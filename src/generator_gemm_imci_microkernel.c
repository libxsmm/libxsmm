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

#include "generator_gemm_imci_microkernel.h"
#include "generator_x86_instructions.h"
#include <libxsmm_cpuid.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_gemm_imci_microkernel( libxsmm_generated_code*             io_generated_code,
                                               const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                               const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                               const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                               const unsigned int                 i_n_blocking,
                                               const unsigned int                 i_k_blocking,
                                               const int                          i_offset )
{
  unsigned int l_n;
  unsigned int l_k;

#if !defined(NDEBUG)
  if ( i_n_blocking > 30 ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }
  if ( (i_offset >= 0) && (i_k_blocking != 1) ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_K_BLOCK );
    return;
  }
#endif

  /* if we have an offset greater-equal -> external k-unrolling */
  if (i_offset != (-1)) {
    /* load A */
    libxsmm_x86_instruction_vec_move_imci( io_generated_code,
                                       i_micro_kernel_config->instruction_set,
                                       i_micro_kernel_config->a_vmove_instruction,
                                       i_gp_reg_mapping->gp_reg_a,
                                       LIBXSMM_X86_GP_REG_UNDEF, 0,
                                       i_xgemm_desc->lda * i_offset * i_micro_kernel_config->datatype_size,
                                       i_micro_kernel_config->vector_name,
                                       0,
                                       i_micro_kernel_config->use_masking_a_c, 0 );

    /* compute vectorwidth (A) * column broadcast (B) */
    for ( l_n = 0; l_n < i_n_blocking; l_n++) {
      libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           i_micro_kernel_config->vmul_instruction,
                                           1,
                                           i_gp_reg_mapping->gp_reg_b,
                                           LIBXSMM_X86_GP_REG_UNDEF,
                                           0,
                                           (i_offset * i_micro_kernel_config->datatype_size) + (i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size * l_n),
                                           i_micro_kernel_config->vector_name,
                                           0,
                                           i_micro_kernel_config->vector_reg_count - i_n_blocking + l_n );
    }
  } else {
    unsigned int l_b_prefetches = 0;
    /* apply k blocking */
    for ( l_k = 0; l_k < i_k_blocking; l_k++ ) {
      /* load A */
      libxsmm_x86_instruction_vec_move_imci( io_generated_code,
                                         i_micro_kernel_config->instruction_set,
                                         i_micro_kernel_config->a_vmove_instruction,
                                         i_gp_reg_mapping->gp_reg_a,
                                         LIBXSMM_X86_GP_REG_UNDEF, 0,
                                         i_xgemm_desc->lda * l_k * i_micro_kernel_config->datatype_size,
                                         i_micro_kernel_config->vector_name,
                                         0,
                                         i_micro_kernel_config->use_masking_a_c, 0 );

      /* compute vectorwidth (A) * column broadcast (B) */
      for ( l_n = 0; l_n < i_n_blocking; l_n++) {
        if ( l_n == 0 ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                        LIBXSMM_X86_INSTR_VPREFETCH0,
                                        i_gp_reg_mapping->gp_reg_a,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        (i_xgemm_desc->lda * (l_k+1) * i_micro_kernel_config->datatype_size) );
        }
        if ( l_n == 1 ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                        LIBXSMM_X86_INSTR_VPREFETCH1,
                                        i_gp_reg_mapping->gp_reg_a,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        (i_xgemm_desc->lda * l_k * i_micro_kernel_config->datatype_size)+64 );
        }
        if ( (l_n == 2) && (l_b_prefetches < i_n_blocking) && (i_k_blocking == 8) ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                        LIBXSMM_X86_INSTR_VPREFETCH0,
                                        i_gp_reg_mapping->gp_reg_b,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        (i_xgemm_desc->ldb * (l_b_prefetches) * i_micro_kernel_config->datatype_size)+64 );
          l_b_prefetches++;
        }
        libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             i_micro_kernel_config->vmul_instruction,
                                             1,
                                             i_gp_reg_mapping->gp_reg_b,
                                             LIBXSMM_X86_GP_REG_UNDEF,
                                             0,
                                             (l_k * i_micro_kernel_config->datatype_size)+(i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size * l_n),
                                             i_micro_kernel_config->vector_name,
                                             0,
                                             i_micro_kernel_config->vector_reg_count - i_n_blocking + l_n );
      }
    }

    /* in last k-iteration: advance pointers */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                 i_micro_kernel_config->alu_add_instruction,
                                 i_gp_reg_mapping->gp_reg_a,
                                 i_k_blocking * i_micro_kernel_config->datatype_size * i_xgemm_desc->lda );

    /* advance pointers of B only when we are not fully unrolling K*/
    if ( i_k_blocking < (unsigned int)i_xgemm_desc->k ) {
      /* advance pointers of B */
      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                   i_micro_kernel_config->alu_add_instruction,
                                   i_gp_reg_mapping->gp_reg_b,
                                   i_k_blocking * i_micro_kernel_config->datatype_size );
    }
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
unsigned int libxsmm_generator_gemm_imci_kernel_kloop( libxsmm_generated_code*             io_generated_code,
                                                        libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                        const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                        const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                        const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                        const char*                        i_arch,
                                                        unsigned int                       i_n_blocking ) {
  const unsigned int l_k_blocking = 8;
  const unsigned int l_k_threshold = 8;
  unsigned int l_k_unrolled = 0;

  LIBXSMM_UNUSED(i_arch);

  /* Let's do something special for SeisSol with k=9, fully unroll */
  if ((i_xgemm_desc->k == 9)) {
    libxsmm_generator_gemm_imci_microkernel( io_generated_code,
                                                i_gp_reg_mapping,
                                                i_micro_kernel_config,
                                                i_xgemm_desc,
                                                i_n_blocking,
                                                i_xgemm_desc->k,
                                                -1 );
  } else if ( (i_xgemm_desc->k % l_k_blocking == 0) && (l_k_threshold <= (unsigned int)i_xgemm_desc->k) ) {
    if (l_k_blocking != (unsigned int)i_xgemm_desc->k) {
      libxsmm_generator_gemm_header_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config,
                                            i_micro_kernel_config->vector_length, l_k_blocking);
    }

    libxsmm_generator_gemm_imci_microkernel( io_generated_code,
                                                i_gp_reg_mapping,
                                                i_micro_kernel_config,
                                                i_xgemm_desc,
                                                i_n_blocking,
                                                l_k_blocking,
                                                -1 );

    if (l_k_blocking != (unsigned int)i_xgemm_desc->k) {
      libxsmm_generator_gemm_footer_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config,
                                           i_xgemm_desc, i_micro_kernel_config->vector_length, i_xgemm_desc->k, 1 );
    }
  } else {
    unsigned int l_max_blocked_k = (i_xgemm_desc->k/l_k_blocking)*l_k_blocking;
    unsigned int l_k;
    if (l_max_blocked_k > 0 ) {
      libxsmm_generator_gemm_header_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config,
                                            i_micro_kernel_config->vector_length, l_k_blocking);

      libxsmm_generator_gemm_imci_microkernel( io_generated_code,
                                                i_gp_reg_mapping,
                                                i_micro_kernel_config,
                                                i_xgemm_desc,
                                                i_n_blocking,
                                                l_k_blocking,
                                                -1 );

      libxsmm_generator_gemm_footer_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config,
                                             i_xgemm_desc, i_micro_kernel_config->vector_length, l_max_blocked_k, 0 );
    }
    for ( l_k = l_max_blocked_k; l_k < (unsigned int)i_xgemm_desc->k; l_k++) {
      libxsmm_generator_gemm_imci_microkernel( io_generated_code,
                                                i_gp_reg_mapping,
                                                i_micro_kernel_config,
                                                i_xgemm_desc,
                                                i_n_blocking,
                                                1,
                                                l_k-l_max_blocked_k );
    }
    /* update A, B and a_prefetch pointers */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                 i_micro_kernel_config->alu_add_instruction,
                                 i_gp_reg_mapping->gp_reg_a,
                                 (i_xgemm_desc->k - l_max_blocked_k) * i_micro_kernel_config->datatype_size * i_xgemm_desc->lda );

    /* reset on B is just needed when we had more than iterations left */
    if (l_max_blocked_k > 0 ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                   i_micro_kernel_config->alu_sub_instruction,
                                   i_gp_reg_mapping->gp_reg_b,
                                   l_max_blocked_k * i_micro_kernel_config->datatype_size );
    }
  }

  return l_k_unrolled;
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_x86_instruction_vec_move_imci( libxsmm_generated_code* io_generated_code,
                                        const unsigned int      i_instruction_set,
                                        const unsigned int      i_vmove_instr,
                                        const unsigned int      i_gp_reg_base,
                                        const unsigned int      i_gp_reg_idx,
                                        const unsigned int      i_scale,
                                        const int               i_displacement,
                                        const char              i_vector_name,
                                        const unsigned int      i_vec_reg_number_0,
                                        const unsigned int      i_use_masking,
                                        const unsigned int      i_is_store ) {
  if ( (i_vmove_instr == LIBXSMM_X86_INSTR_VMOVAPD) ||
       (i_vmove_instr == LIBXSMM_X86_INSTR_VMOVAPS)    ) {
    libxsmm_x86_instruction_vec_move( io_generated_code, i_instruction_set, i_vmove_instr,
                                  i_gp_reg_base, i_gp_reg_idx, i_scale, i_displacement, i_vector_name, i_vec_reg_number_0, i_use_masking, i_is_store );
  } else if ( i_vmove_instr == LIBXSMM_X86_INSTR_VMOVUPD ) {
    unsigned int l_instr_1 = 0;
    unsigned int l_instr_2 = 0;
    if (i_is_store == 0) {
      l_instr_1 = LIBXSMM_X86_INSTR_VLOADUNPACKLPD;
      l_instr_2 = LIBXSMM_X86_INSTR_VLOADUNPACKHPD;
    } else {
      l_instr_1 = LIBXSMM_X86_INSTR_VPACKSTORELPD;
      l_instr_2 = LIBXSMM_X86_INSTR_VPACKSTOREHPD;
    }
    libxsmm_x86_instruction_vec_move( io_generated_code, i_instruction_set, l_instr_1,
                                  i_gp_reg_base, i_gp_reg_idx, i_scale, i_displacement, i_vector_name, i_vec_reg_number_0, i_use_masking, i_is_store );
    libxsmm_x86_instruction_vec_move( io_generated_code, i_instruction_set, l_instr_2,
                                  i_gp_reg_base, i_gp_reg_idx, i_scale, i_displacement+64, i_vector_name, i_vec_reg_number_0, i_use_masking, i_is_store );
  } else if ( i_vmove_instr == LIBXSMM_X86_INSTR_VMOVUPS ) {
    unsigned int l_instr_1 = 0;
    unsigned int l_instr_2 = 0;
    if (i_is_store == 0) {
      l_instr_1 = LIBXSMM_X86_INSTR_VLOADUNPACKLPS;
      l_instr_2 = LIBXSMM_X86_INSTR_VLOADUNPACKHPS;
    } else {
      l_instr_1 = LIBXSMM_X86_INSTR_VPACKSTORELPS;
      l_instr_2 = LIBXSMM_X86_INSTR_VPACKSTOREHPS;
    }
    libxsmm_x86_instruction_vec_move( io_generated_code, i_instruction_set, l_instr_1,
                                  i_gp_reg_base, i_gp_reg_idx, i_scale, i_displacement, i_vector_name, i_vec_reg_number_0, i_use_masking, i_is_store );
    libxsmm_x86_instruction_vec_move( io_generated_code, i_instruction_set, l_instr_2,
                                  i_gp_reg_base, i_gp_reg_idx, i_scale, i_displacement+64, i_vector_name, i_vec_reg_number_0, i_use_masking, i_is_store );
  } else {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_VEC_MOVE_IMCI );
    return;
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_gemm_load_C_imci( libxsmm_generated_code*             io_generated_code,
                                          const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                          const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                          const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                          const unsigned int                 i_m_blocking,
                                          const unsigned int                 i_n_blocking ) {
  /* deriving register blocking from kernel config */
  unsigned int l_m_blocking = i_m_blocking/i_micro_kernel_config->vector_length;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = i_micro_kernel_config->vector_reg_count - (i_n_blocking * l_m_blocking);
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;

#if !defined(NDEBUG)
  /* Do some test if it's possible to generated the requested code.
     This is not done in release mode and therefore bad
     things might happen.... HUAAH */
  if (i_micro_kernel_config->instruction_set != LIBXSMM_X86_IMCI ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_NO_IMCI );
    return;
  }
  if ( (i_n_blocking > 30) || (i_n_blocking < 1) || (i_m_blocking != i_micro_kernel_config->vector_length) ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
    return;
  }
  if ( i_m_blocking % i_micro_kernel_config->vector_length != 0 ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_M_BLOCK );
    return;
  }
#endif /*NDEBUG*/

  /* load C accumulator */
  if (i_xgemm_desc->beta == 1) {
    /* adding to C, so let's load C */
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      libxsmm_x86_instruction_vec_move_imci( io_generated_code,
                                         i_micro_kernel_config->instruction_set,
                                         i_micro_kernel_config->c_vmove_instruction,
                                         i_gp_reg_mapping->gp_reg_c,
                                         LIBXSMM_X86_GP_REG_UNDEF, 0,
                                         (l_n * i_xgemm_desc->ldc * i_micro_kernel_config->datatype_size),
                                         i_micro_kernel_config->vector_name,
                                         l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), i_micro_kernel_config->use_masking_a_c, 0 );
      if ( i_micro_kernel_config->c_vmove_instruction == LIBXSMM_X86_INSTR_VMOVAPD ||
           i_micro_kernel_config->c_vmove_instruction == LIBXSMM_X86_INSTR_VMOVAPS    ) {
        libxsmm_x86_instruction_prefetch( io_generated_code,
                                      LIBXSMM_X86_INSTR_VPREFETCH1,
                                      i_gp_reg_mapping->gp_reg_c,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      (l_n * i_xgemm_desc->ldc * i_micro_kernel_config->datatype_size)+64 );
      }
    }
  } else {
    /* overwriting C, so let's xout accumulator */
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           i_micro_kernel_config->vxor_instruction,
                                           i_micro_kernel_config->vector_name,
                                           l_vec_reg_acc_start + l_m + (l_m_blocking * l_n),
                                           l_vec_reg_acc_start + l_m + (l_m_blocking * l_n),
                                           l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
      libxsmm_x86_instruction_prefetch( io_generated_code,
                                    LIBXSMM_X86_INSTR_VPREFETCH1,
                                    i_gp_reg_mapping->gp_reg_c,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    (l_n * i_xgemm_desc->ldc * i_micro_kernel_config->datatype_size) );
    }
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_gemm_store_C_imci( libxsmm_generated_code*             io_generated_code,
                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                           const unsigned int                 i_m_blocking,
                                           const unsigned int                 i_n_blocking )
{
  /* deriving register blocking from kernel config */
  unsigned int l_m_blocking = i_m_blocking/i_micro_kernel_config->vector_length;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = i_micro_kernel_config->vector_reg_count - (i_n_blocking * l_m_blocking);

#if !defined(NDEBUG)
  /* Do some test if it's possible to generated the requested code.
     This is not done in release mode and therefore bad
     things might happen.... HUAAH */
  if (i_micro_kernel_config->instruction_set != LIBXSMM_X86_IMCI ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_NO_IMCI );
    return;
  }
  if ( (i_n_blocking > 30) || (i_n_blocking < 1) || (i_m_blocking != i_micro_kernel_config->vector_length) ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
    return;
  }
  if ( i_m_blocking % i_micro_kernel_config->vector_length != 0 ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_M_BLOCK );
    return;
  }
#endif

  /* storing C accumulator */
  /* adding to C, so let's load C */
  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    libxsmm_x86_instruction_vec_move_imci( io_generated_code,
                                       i_micro_kernel_config->instruction_set,
                                       i_micro_kernel_config->c_vmove_instruction,
                                       i_gp_reg_mapping->gp_reg_c,
                                       LIBXSMM_X86_GP_REG_UNDEF, 0,
                                       ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size),
                                       i_micro_kernel_config->vector_name,
                                       l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), i_micro_kernel_config->use_masking_a_c, 1 );
  }
}


