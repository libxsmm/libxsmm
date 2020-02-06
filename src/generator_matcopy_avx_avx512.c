/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_matcopy_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
void libxsmm_generator_matcopy_header_m_loop( libxsmm_generated_code*                   io_generated_code,
                                              libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                              const libxsmm_matcopy_kernel_config*      i_kernel_config,
                                              const unsigned int                        i_gp_reg_m_loop ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_add_instruction, i_gp_reg_m_loop, 1);
}

LIBXSMM_API_INTERN
void libxsmm_generator_matcopy_footer_m_loop( libxsmm_generated_code*                       io_generated_code,
                                              libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                              const libxsmm_matcopy_kernel_config*          i_kernel_config,
                                              const unsigned int                            i_gp_reg_m_loop,
                                              const unsigned int                            i_m ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_cmp_instruction, i_gp_reg_m_loop, i_m );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_matcopy_header_n_loop( libxsmm_generated_code*                  io_generated_code,
                                              libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                              const libxsmm_matcopy_kernel_config*      i_kernel_config,
                                              const unsigned int                        i_gp_reg_n_loop ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_add_instruction, i_gp_reg_n_loop, 1);
}

LIBXSMM_API_INTERN
void libxsmm_generator_matcopy_footer_n_loop( libxsmm_generated_code*                      io_generated_code,
                                              libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                              const libxsmm_matcopy_kernel_config*          i_kernel_config,
                                              const unsigned int                            i_gp_reg_n_loop,
                                              const unsigned int                            i_n ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_cmp_instruction, i_gp_reg_n_loop, i_n );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_API_INLINE
void libxsmm_generator_matcopy_avx_avx512_kernel_initialize_mask( libxsmm_generated_code*               io_generated_code,
                                                                 const libxsmm_matcopy_gp_reg_mapping*  i_gp_reg_mapping,
                                                                 const libxsmm_matcopy_kernel_config*   i_micro_kernel_config,
                                                                 unsigned int                           remainder ) {
  unsigned long long l_mask = (1ULL << remainder) - 1;

  /* If we have int16 input and KNM arch, we should make the remainder mask "half", since we have only VMOVUPS instruction (i.e. treat the int16 entries in pairs, thus the mask length should be half) */
  if ( (i_micro_kernel_config->vector_length == 32) && (i_micro_kernel_config->instruction_set == LIBXSMM_X86_AVX512_KNM) ) {
    l_mask = (1ULL << (remainder/2)) - 1;
  }

  /* Move mask to GP register */
  if (i_micro_kernel_config->vector_length == 64) {
    libxsmm_x86_instruction_alu_imm_i64( io_generated_code,
                                  i_micro_kernel_config->alu_mov_instruction,
                                  i_gp_reg_mapping->gp_reg_help_0,
                                  (size_t)l_mask );
  } else {
     libxsmm_x86_instruction_alu_imm( io_generated_code,
                                  i_micro_kernel_config->alu_mov_instruction,
                                  i_gp_reg_mapping->gp_reg_help_0,
                                  /* immediate is passed as an integer */
                                  (int)l_mask );
  }

  /* Set mask register */
  if ( i_micro_kernel_config->instruction_set >= LIBXSMM_X86_AVX512_CORE ) {
    libxsmm_x86_instruction_mask_move( io_generated_code,
                                      LIBXSMM_X86_INSTR_KMOVQ,
                                      i_gp_reg_mapping->gp_reg_help_0,
                                      1 );
  } else if ( i_micro_kernel_config->instruction_set == LIBXSMM_X86_AVX512_MIC || i_micro_kernel_config->instruction_set == LIBXSMM_X86_AVX512_KNM ) {
    libxsmm_x86_instruction_mask_move( io_generated_code,
                                      LIBXSMM_X86_INSTR_KMOVW,
                                      i_gp_reg_mapping->gp_reg_help_0,
                                      1 );
  } else {
    /* Should not happen! */
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_matcopy_avx_avx512_kernel( libxsmm_generated_code*         io_generated_code,
                                                  const libxsmm_mcopy_descriptor* i_matcopy_desc,
                                                  const char*                     i_arch ) {
  libxsmm_matcopy_kernel_config l_kernel_config;
  libxsmm_matcopy_gp_reg_mapping l_gp_reg_mapping;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  unsigned int m_trips, remaining_unrolled, remaining, i;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define gp register mapping */
  memset(&l_gp_reg_mapping, 0, sizeof(l_gp_reg_mapping)); /* avoid warning "maybe used uninitialized" */
#if defined(_WIN32) || defined(__CYGWIN__)
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_lda = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_R8;
  l_gp_reg_mapping.gp_reg_ldb = LIBXSMM_X86_GP_REG_R9;
  /* TODO: full support for Windows calling convention */
  l_gp_reg_mapping.gp_reg_a_pf = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_b_pf = LIBXSMM_X86_GP_REG_RSI;
#else /* match calling convention on Linux */
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_lda = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_ldb = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_a_pf = LIBXSMM_X86_GP_REG_R8;
  l_gp_reg_mapping.gp_reg_b_pf = LIBXSMM_X86_GP_REG_R9;
#endif
  l_gp_reg_mapping.gp_reg_m_loop = LIBXSMM_X86_GP_REG_R15;
  l_gp_reg_mapping.gp_reg_n_loop = LIBXSMM_X86_GP_REG_R12;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_X86_GP_REG_RAX;

  /* define matcopy kernel config */
  memset(&l_kernel_config, 0, sizeof(l_kernel_config)); /* avoid warning "maybe used uninitialized" */
  if ( strcmp( i_arch, "snb" ) == 0 ) {
    l_kernel_config.instruction_set = LIBXSMM_X86_AVX;
    l_kernel_config.vector_reg_count = 16;
    l_kernel_config.vector_name = 'y';
    l_kernel_config.vxor_instruction = LIBXSMM_X86_INSTR_VXORPS;
  } else if ( strcmp( i_arch, "hsw" ) == 0 ) {
    l_kernel_config.instruction_set = LIBXSMM_X86_AVX2;
    l_kernel_config.vector_reg_count = 16;
    l_kernel_config.vector_name = 'y';
    l_kernel_config.vxor_instruction = LIBXSMM_X86_INSTR_VXORPS;
  } else if ( strcmp( i_arch, "skx" ) == 0 ) {
    l_kernel_config.instruction_set = LIBXSMM_X86_AVX512_CORE;
    l_kernel_config.vector_reg_count = 32;
    l_kernel_config.vector_name = 'z';
    l_kernel_config.vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
  } else if ( strcmp( i_arch, "clx" ) == 0 ) {
    l_kernel_config.instruction_set = LIBXSMM_X86_AVX512_CLX;
    l_kernel_config.vector_reg_count = 32;
    l_kernel_config.vector_name = 'z';
    l_kernel_config.vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
  } else if ( strcmp( i_arch, "cpx" ) == 0 ) {
    l_kernel_config.instruction_set = LIBXSMM_X86_AVX512_CPX;
    l_kernel_config.vector_reg_count = 32;
    l_kernel_config.vector_name = 'z';
    l_kernel_config.vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
  } else if ( strcmp( i_arch, "knl" ) == 0 ) {
    /* For now make the code work for KNL */
    l_kernel_config.instruction_set = LIBXSMM_X86_AVX512_MIC;
    l_kernel_config.vector_reg_count = 32;
    l_kernel_config.vector_name = 'z';
    l_kernel_config.vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
  } else if ( strcmp( i_arch, "knm" ) == 0 ) {
    l_kernel_config.instruction_set = LIBXSMM_X86_AVX512_KNM;
    l_kernel_config.vector_reg_count = 32;
    l_kernel_config.vector_name = 'z';
    l_kernel_config.vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
    return;
  }

  /* More setup in the kernel config based on architecture and data type */
  if ( l_kernel_config.vector_name == 'y' ) {
    assert(0 < i_matcopy_desc->typesize);
    l_kernel_config.datatype_size = i_matcopy_desc->typesize;
    if ( i_matcopy_desc->typesize == 4  ) {
      l_kernel_config.vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      l_kernel_config.vector_length = 8;
    } else if ( i_matcopy_desc->typesize == 2  ) {
      l_kernel_config.vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      l_kernel_config.vector_length = 16;
    } else if ( i_matcopy_desc->typesize == 1  ) {
      l_kernel_config.vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      l_kernel_config.vector_length = 32;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
    assert(l_kernel_config.vector_length == 32 / l_kernel_config.datatype_size);
  } else if ( l_kernel_config.vector_name == 'z' ) {
    assert(0 < i_matcopy_desc->typesize);
    l_kernel_config.datatype_size = i_matcopy_desc->typesize;
    if ( i_matcopy_desc->typesize == 4  ) {
      /* TODO: use streaming stores if we want to zero the destination */
      /*if (0 != (LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE & i_matcopy_desc->flags)) {
        l_kernel_config.vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      } else*/ {
        l_kernel_config.vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      }
      l_kernel_config.vector_length = 16;
    } else if ( i_matcopy_desc->typesize == 2  ) {
      if (l_kernel_config.instruction_set == LIBXSMM_X86_AVX512_KNM) {
        l_kernel_config.vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      } else if ( l_kernel_config.instruction_set >= LIBXSMM_X86_AVX512_CORE ) {
        l_kernel_config.vmove_instruction = LIBXSMM_X86_INSTR_VMOVDQU16;
      } else {
        /* Should not happen!!! */
      }
      l_kernel_config.vector_length = 32;
    } else if ( i_matcopy_desc->typesize == 1  ) {
      l_kernel_config.vmove_instruction = LIBXSMM_X86_INSTR_VMOVDQU8;
      l_kernel_config.vector_length = 64;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
    assert(l_kernel_config.vector_length == 64 / l_kernel_config.datatype_size);
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
    return;
  }

  l_kernel_config.alu_add_instruction = LIBXSMM_X86_INSTR_ADDQ;
  l_kernel_config.alu_cmp_instruction = LIBXSMM_X86_INSTR_CMPQ;
  l_kernel_config.alu_mov_instruction = LIBXSMM_X86_INSTR_MOVQ;
  l_kernel_config.alu_jmp_instruction = LIBXSMM_X86_INSTR_JL;
  l_kernel_config.prefetch_instruction = LIBXSMM_X86_INSTR_PREFETCHT2;

  /* Calculate the trips in the m dimension (perform unrolling if requested) */
  assert(0 != i_matcopy_desc->unroll_level);
  m_trips = i_matcopy_desc->m / (l_kernel_config.vector_length * i_matcopy_desc->unroll_level);
  remaining_unrolled = (i_matcopy_desc->m % (l_kernel_config.vector_length * i_matcopy_desc->unroll_level)) / l_kernel_config.vector_length;
  remaining = (i_matcopy_desc->m % (l_kernel_config.vector_length * i_matcopy_desc->unroll_level)) % l_kernel_config.vector_length;

  /* open asm */
  libxsmm_x86_instruction_open_stream_matcopy( io_generated_code, l_gp_reg_mapping.gp_reg_a,
      l_gp_reg_mapping.gp_reg_lda, l_gp_reg_mapping.gp_reg_b,
      l_gp_reg_mapping.gp_reg_ldb, l_gp_reg_mapping.gp_reg_a_pf,
      l_gp_reg_mapping.gp_reg_b_pf, i_arch );

  /* In case we should do masked load/store and we have AVX512 arch, precompute the mask */
  if (remaining && l_kernel_config.instruction_set >= LIBXSMM_X86_AVX512) {
    libxsmm_generator_matcopy_avx_avx512_kernel_initialize_mask(io_generated_code,
        &l_gp_reg_mapping,
        &l_kernel_config,
        remaining);
  }

  /* Initialize register 0 with zeros if we want to zero the destination */
  if (0 != (LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE & i_matcopy_desc->flags)) {
    libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
        l_kernel_config.instruction_set,
        l_kernel_config.vxor_instruction,
        l_kernel_config.vector_name,
        0,
        0,
        0);
    /* In case of AVX/AVX2 and if we have remaining, set also scalar register to zero */
    if (remaining && (l_kernel_config.instruction_set == LIBXSMM_X86_AVX || l_kernel_config.instruction_set == LIBXSMM_X86_AVX2)) {
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
          l_kernel_config.instruction_set,
          LIBXSMM_X86_INSTR_VXORPS,
          'x',
          0,
          0,
          0);
    }
  }

  if (i_matcopy_desc->n > 1) {
    /* open n loop */
    libxsmm_generator_matcopy_header_n_loop(  io_generated_code, &l_loop_label_tracker,
        &l_kernel_config, l_gp_reg_mapping.gp_reg_n_loop );
  }

  if (m_trips > 1) {
    /* open m loop */
    libxsmm_generator_matcopy_header_m_loop(  io_generated_code, &l_loop_label_tracker,
        &l_kernel_config, l_gp_reg_mapping.gp_reg_m_loop );
  }

  if (m_trips >= 1) {
    /* Unroll the innermost loop as requested */
    for (i = 0; i < i_matcopy_desc->unroll_level; i++) {

      if (0 == (LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE & i_matcopy_desc->flags)) {
        /* load input line to register 0 */
        libxsmm_x86_instruction_vec_move( io_generated_code,
            l_kernel_config.instruction_set,
            l_kernel_config.vmove_instruction,
            l_gp_reg_mapping.gp_reg_a,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            i*l_kernel_config.vector_length*l_kernel_config.datatype_size,
            l_kernel_config.vector_name, 0,
            0, 1, 0 );
      }

      /* Prefetch if requested */
      if (i_matcopy_desc->prefetch) {
        libxsmm_x86_instruction_prefetch( io_generated_code,
            l_kernel_config.prefetch_instruction,
            l_gp_reg_mapping.gp_reg_a_pf,
            LIBXSMM_X86_GP_REG_UNDEF,
            0,
            i*l_kernel_config.vector_length*l_kernel_config.datatype_size );
      }

      /* store register 0 to destination line */
      libxsmm_x86_instruction_vec_move( io_generated_code,
          l_kernel_config.instruction_set,
          l_kernel_config.vmove_instruction,
          l_gp_reg_mapping.gp_reg_b,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          i*l_kernel_config.vector_length*l_kernel_config.datatype_size,
          l_kernel_config.vector_name, 0,
          0, 0, 1 );
    }

    if (0 == (LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE & i_matcopy_desc->flags)) {
      /* adjust input pointer by VLEN * unroll-level elements */
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          l_kernel_config.alu_add_instruction,
          l_gp_reg_mapping.gp_reg_a,
          i_matcopy_desc->unroll_level * l_kernel_config.vector_length * l_kernel_config.datatype_size);
    }

    /* adjust destination pointer by VLEN * unroll-level elements */
    libxsmm_x86_instruction_alu_imm(  io_generated_code,
        l_kernel_config.alu_add_instruction,
        l_gp_reg_mapping.gp_reg_b,
        i_matcopy_desc->unroll_level * l_kernel_config.vector_length * l_kernel_config.datatype_size);

    /* Adjust prefetch pointer by VLEN * unroll-level elements */
    if (i_matcopy_desc->prefetch) {
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          l_kernel_config.alu_add_instruction,
          l_gp_reg_mapping.gp_reg_a_pf,
          i_matcopy_desc->unroll_level * l_kernel_config.vector_length * l_kernel_config.datatype_size);
    }
  }

  if (m_trips > 1) {
    /* close m loop */
    libxsmm_generator_matcopy_footer_m_loop(  io_generated_code, &l_loop_label_tracker,
        &l_kernel_config, l_gp_reg_mapping.gp_reg_m_loop, m_trips );
  }

  /* Add unrolled load/stores for remaining without mask */
  for (i = 0; i < remaining_unrolled; i++) {
    if (0 == (LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE & i_matcopy_desc->flags)) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
          l_kernel_config.instruction_set,
          l_kernel_config.vmove_instruction,
          l_gp_reg_mapping.gp_reg_a,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          i*l_kernel_config.vector_length*l_kernel_config.datatype_size,
          l_kernel_config.vector_name, 0,
          0, 1, 0 );
    }
    if (i_matcopy_desc->prefetch) {
      libxsmm_x86_instruction_prefetch( io_generated_code,
          l_kernel_config.prefetch_instruction,
          l_gp_reg_mapping.gp_reg_a_pf,
          LIBXSMM_X86_GP_REG_UNDEF,
          0,
          i*l_kernel_config.vector_length*l_kernel_config.datatype_size );
    }
    libxsmm_x86_instruction_vec_move( io_generated_code,
        l_kernel_config.instruction_set,
        l_kernel_config.vmove_instruction,
        l_gp_reg_mapping.gp_reg_b,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        i*l_kernel_config.vector_length*l_kernel_config.datatype_size,
        l_kernel_config.vector_name, 0,
        0, 0, 1 );
  }

  /* Add load/store with mask if there is remaining and we have AVX512 arch */
  if (remaining && l_kernel_config.instruction_set >= LIBXSMM_X86_AVX512) {
    if (0 == (LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE & i_matcopy_desc->flags)) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
          l_kernel_config.instruction_set,
          l_kernel_config.vmove_instruction,
          l_gp_reg_mapping.gp_reg_a,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          remaining_unrolled * l_kernel_config.vector_length * l_kernel_config.datatype_size,
          l_kernel_config.vector_name, 0,
          1, 1, 0 );
    }
    if (i_matcopy_desc->prefetch) {
      libxsmm_x86_instruction_prefetch( io_generated_code,
          l_kernel_config.prefetch_instruction,
          l_gp_reg_mapping.gp_reg_a_pf,
          LIBXSMM_X86_GP_REG_UNDEF,
          0,
          remaining_unrolled * l_kernel_config.vector_length * l_kernel_config.datatype_size );
    }
    libxsmm_x86_instruction_vec_move( io_generated_code,
        l_kernel_config.instruction_set,
        l_kernel_config.vmove_instruction,
        l_gp_reg_mapping.gp_reg_b,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        remaining_unrolled * l_kernel_config.vector_length * l_kernel_config.datatype_size,
        l_kernel_config.vector_name, 0,
        1, 0, 1 );
  } else if (remaining && (l_kernel_config.instruction_set == LIBXSMM_X86_AVX || l_kernel_config.instruction_set == LIBXSMM_X86_AVX2)) {
    /* Use scalar moves in case of remaining and AVX/AVX2 arch */
    for (i=0; i<remaining; i++) {
      if (0 == (LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE & i_matcopy_desc->flags)) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            l_kernel_config.instruction_set,
            LIBXSMM_X86_INSTR_VMOVSS,
            l_gp_reg_mapping.gp_reg_a,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (remaining_unrolled * l_kernel_config.vector_length + i) * l_kernel_config.datatype_size,
            'x', 0,
            0, 1, 0 );
      }
      if (i_matcopy_desc->prefetch) {
        /* Issue just one prefetch */
        if (i == 0) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
              l_kernel_config.prefetch_instruction,
              l_gp_reg_mapping.gp_reg_a_pf,
              LIBXSMM_X86_GP_REG_UNDEF,
              0,
              remaining_unrolled * l_kernel_config.vector_length * l_kernel_config.datatype_size );
        }
      }
      libxsmm_x86_instruction_vec_move( io_generated_code,
          l_kernel_config.instruction_set,
          LIBXSMM_X86_INSTR_VMOVSS,
          l_gp_reg_mapping.gp_reg_b,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (remaining_unrolled * l_kernel_config.vector_length + i) * l_kernel_config.datatype_size,
          'x', 0,
          0, 0, 1 );
    }
  }

  if (i_matcopy_desc->n > 1) {
    if (0 == (LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE & i_matcopy_desc->flags)) {
      /* adjust input pointer by (lda - m_trips * VLEN * unroll-level) elements (already has been increased by m_trips * VLEN * unroll-level in the above m_trips loop ) */
      if ( (i_matcopy_desc->ldi - m_trips * l_kernel_config.vector_length * i_matcopy_desc->unroll_level) != 0 ) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
            l_kernel_config.alu_add_instruction,
            l_gp_reg_mapping.gp_reg_a,
            (i_matcopy_desc->ldi - m_trips * l_kernel_config.vector_length * i_matcopy_desc->unroll_level) * l_kernel_config.datatype_size);
      }
    }
    /* adjust destination pointer by (ldb - m_trips * VLEN * unroll-level) elements (already has been increased by m_trips * VLEN * unroll-level in the above m_trips loop ) */
    if ( (i_matcopy_desc->ldo - m_trips * l_kernel_config.vector_length * i_matcopy_desc->unroll_level) != 0 ) {
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          l_kernel_config.alu_add_instruction,
          l_gp_reg_mapping.gp_reg_b,
          (i_matcopy_desc->ldo - m_trips * l_kernel_config.vector_length * i_matcopy_desc->unroll_level) * l_kernel_config.datatype_size);
    }
    /* Adjust prefetch pointer if requested */
    if (i_matcopy_desc->prefetch) {
      if ( (i_matcopy_desc->ldi - m_trips * l_kernel_config.vector_length * i_matcopy_desc->unroll_level) != 0 ) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
            l_kernel_config.alu_add_instruction,
            l_gp_reg_mapping.gp_reg_a_pf,
            (i_matcopy_desc->ldi - m_trips * l_kernel_config.vector_length * i_matcopy_desc->unroll_level) * l_kernel_config.datatype_size);
      }
    }
    /* close n loop */
    libxsmm_generator_matcopy_footer_n_loop(  io_generated_code, &l_loop_label_tracker,
        &l_kernel_config, l_gp_reg_mapping.gp_reg_n_loop, i_matcopy_desc->n );
  }

  /* close asm */
  libxsmm_x86_instruction_close_stream_matcopy( io_generated_code, i_arch );
}

