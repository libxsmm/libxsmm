/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
*               Intel Corporation - All rights reserved                       *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Breuer (Univ. Jena), Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include "generator_aarch64_instructions.h"
#include "generator_common_aarch64.h"
#include "generator_common.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
void libxsmm_generator_set_p_register_aarch64_sve( libxsmm_generated_code* io_generated_code,
                                                   unsigned char           i_p_reg,
                                                   int                     i_n_bits,
                                                   unsigned char           i_gp_reg_scratch ) {
  if( i_n_bits < 0 ) {
    libxsmm_aarch64_instruction_sve_pcompute( io_generated_code,
                                              LIBXSMM_AARCH64_INSTR_SVE_PTRUE,
                                              i_p_reg,
                                              LIBXSMM_AARCH64_ASIMD_REG_UNDEF,
                                              LIBXSMM_AARCH64_GP_WIDTH_W,
                                              LIBXSMM_AARCH64_ASIMD_REG_UNDEF,
                                              LIBXSMM_AARCH64_SVE_PATTERN_ALL,
                                              LIBXSMM_AARCH64_SVE_TYPE_B );
  }
  else {
    /* store number of bits in gp register */
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code,
                                               i_gp_reg_scratch,
                                               i_n_bits );

    libxsmm_aarch64_instruction_sve_pcompute( io_generated_code,
                                              LIBXSMM_AARCH64_INSTR_SVE_WHILELT,
                                              i_p_reg,
                                              LIBXSMM_AARCH64_GP_REG_XZR,
                                              LIBXSMM_AARCH64_GP_WIDTH_X,
                                              i_gp_reg_scratch,
                                              LIBXSMM_AARCH64_SVE_PATTERN_ALL,
                                              LIBXSMM_AARCH64_SVE_TYPE_B );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_loop_header_aarch64( libxsmm_generated_code*     io_generated_code,
                                            libxsmm_loop_label_tracker* io_loop_label_tracker,
                                            const unsigned int          i_gp_reg_loop_cnt,
                                            const unsigned int          i_trips ) {
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_loop_cnt, (unsigned long long)i_trips );
  libxsmm_aarch64_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_loop_footer_aarch64( libxsmm_generated_code*     io_generated_code,
                                            libxsmm_loop_label_tracker* io_loop_label_tracker,
                                            const unsigned int          i_gp_reg_loop_cnt,
                                            const unsigned int          i_loop_blocking ) {
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I,
                                                 i_gp_reg_loop_cnt, i_gp_reg_loop_cnt, i_loop_blocking, 0 );
  libxsmm_aarch64_instruction_cond_jump_back_to_label( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_CBNZ,
                                                       i_gp_reg_loop_cnt, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_load_2dregblock_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                      const unsigned int      i_gp_reg_addr,
                                                      const unsigned int      i_gp_reg_scratch,
                                                      const unsigned int      i_vec_length,
                                                      const unsigned int      i_vec_reg_count,
                                                      const unsigned int      i_m_blocking,
                                                      const unsigned int      i_n_blocking,
                                                      const unsigned int      i_ld,
                                                      const unsigned int      i_zero ) {
  unsigned int l_vec_reg_acc_start;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* VLEN per l_m interations */
  unsigned int l_m_blocks[3];  /* 0: 128bit, 1: 64bit, 2: 32bit */
  unsigned int l_m_total_blocks;
  unsigned int l_m_bytes = 0;

  /* deriving register blocking from kernel config */
  l_m_blocks[0] =  i_m_blocking/i_vec_length;                    /* number of 128 bit stores */
  l_m_blocks[1] = (i_m_blocking%i_vec_length)/(i_vec_length/2);  /* number of  64 bit stores */
  l_m_blocks[2] = (i_m_blocking%i_vec_length)%(i_vec_length/2);  /* number of  32 but stores */
  l_m_total_blocks = l_m_blocks[0] + l_m_blocks[1] + l_m_blocks[2];
  l_m_bytes = l_m_blocks[0]*16 +  l_m_blocks[1]*8 + l_m_blocks[2]*4;

  /* start register of accumulator */
  l_vec_reg_acc_start = i_vec_reg_count - (i_n_blocking * l_m_total_blocks);

  /* load C accumulator */
  if (i_zero == 0) {
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, 16,
                                                l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n),
                                                LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
      }
      for ( l_m = 0; l_m < l_m_blocks[1]; l_m++ ) {
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, 8,
                                                l_vec_reg_acc_start + l_m + l_m_blocks[0] + (l_m_total_blocks * l_n),
                                                LIBXSMM_AARCH64_ASIMD_WIDTH_D );
      }
      for ( l_m = 0; l_m < l_m_blocks[2]; l_m++ ) {
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, 4,
                                                l_vec_reg_acc_start + l_m + l_m_blocks[0] + l_m_blocks[1] + (l_m_total_blocks * l_n),
                                                LIBXSMM_AARCH64_ASIMD_WIDTH_S );
      }
      if ( i_ld-l_m_bytes > 0 ) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_addr, i_gp_reg_scratch, i_gp_reg_addr,
                                                       (unsigned long long)((unsigned long long)(i_ld-l_m_bytes)) );
      }
    }
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_addr, i_gp_reg_scratch, i_gp_reg_addr,
                                                   (unsigned long long)((unsigned long long)(i_ld*i_n_blocking)) );
  } else {
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                   l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n),
                                                   l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n), 0,
                                                   l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n),
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
      }
      for ( l_m = 0; l_m < l_m_blocks[1]; l_m++ ) {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                   l_vec_reg_acc_start + l_m + l_m_blocks[0] + (l_m_total_blocks * l_n),
                                                   l_vec_reg_acc_start + l_m + l_m_blocks[0] + (l_m_total_blocks * l_n), 0,
                                                   l_vec_reg_acc_start + l_m + l_m_blocks[0] + (l_m_total_blocks * l_n),
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_8B );
      }
      for ( l_m = 0; l_m < l_m_blocks[2]; l_m++ ) {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                   l_vec_reg_acc_start + l_m + l_m_blocks[0] + l_m_blocks[1] + (l_m_total_blocks * l_n),
                                                   l_vec_reg_acc_start + l_m + l_m_blocks[0] + l_m_blocks[1] + (l_m_total_blocks * l_n), 0,
                                                   l_vec_reg_acc_start + l_m + l_m_blocks[0] + l_m_blocks[1] + (l_m_total_blocks * l_n),
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_8B );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_load_2dregblock_aarch64_sve( libxsmm_generated_code* io_generated_code,
                                                    const unsigned int      i_gp_reg_addr,
                                                    const unsigned int      i_gp_reg_scratch,
                                                    const unsigned int      i_vec_length,
                                                    const unsigned int      i_vec_reg_count,
                                                    const unsigned int      i_m_blocking,
                                                    const unsigned int      i_n_blocking,
                                                    const unsigned int      i_ld,
                                                    const unsigned int      i_data_size,
                                                    const unsigned int      i_zero ) {
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;

  unsigned int l_m_blocks[2] = { 0 }; /* 0: #full vector loads, 1: #predicate loads (0 or 1) */
  unsigned int l_m_total_blocks = 0;
  unsigned int l_m_bytes_full = 0;
  unsigned int l_vec_reg_acc_start = 0;
  unsigned int l_remainder_size = 0;

  l_m_blocks[0] = i_m_blocking / i_vec_length;
  l_remainder_size = i_m_blocking % i_vec_length;
  l_m_blocks[1] = (l_remainder_size > 0);
  l_m_total_blocks = l_m_blocks[0] + l_m_blocks[1];
  l_m_bytes_full = l_m_blocks[0] * i_vec_length * i_data_size;

  /* start register of accumulator */
  l_vec_reg_acc_start = i_vec_reg_count - (i_n_blocking * l_m_total_blocks);

  /* loads C accumulator from memory */
  if( i_zero == 0 ) {
    /* this is the jump size to be performed after a n-block is complete */
    unsigned long long l_jump_block_n_last = 0;

    /* full vector loads */
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      /* this is the jump size to be performed after a m-block is complete */
      unsigned long long l_jump_block_m_last = 0;

      for( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
        libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                              LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF,
                                              i_gp_reg_addr,
                                              LIBXSMM_AARCH64_GP_REG_UNDEF,
                                              0,
                                              l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m,
                                              LIBXSMM_AARCH64_SVE_REG_UNDEF );
        /* increase pointer in m-dimension.
           but only if
             1) remainder follows
             or
             2) we are not at the end of the m-loop
        */
        if( l_m_blocks[1] != 0 || l_m != l_m_blocks[0] - 1 ) {
          libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code,
                                                         LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                         i_gp_reg_addr,
                                                         i_gp_reg_addr,
                                                         i_vec_length * i_data_size,
                                                         0 );
        }
        /* combine the m-jump with the n one*/
        else {
          l_jump_block_m_last = i_vec_length * i_data_size;
        }
      }

      if( l_m_blocks[1] != 0 ) {
        libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                              LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF,
                                              i_gp_reg_addr,
                                              LIBXSMM_AARCH64_GP_REG_UNDEF,
                                              0,
                                              l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m_blocks[0],
                                              LIBXSMM_AARCH64_SVE_REG_P1 );
      }

      l_jump_block_m_last += i_ld - l_m_bytes_full;

      if( l_n != i_n_blocking - 1 ) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                       LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_addr,
                                                       i_gp_reg_scratch,
                                                       i_gp_reg_addr,
                                                       l_jump_block_m_last );
      }
      else {
        l_jump_block_n_last = l_jump_block_m_last;
      }
    }

    /* reset C-ptr to original address */
    l_jump_block_n_last = i_ld * i_n_blocking - l_jump_block_n_last;
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_addr,
                                                   i_gp_reg_scratch,
                                                   i_gp_reg_addr,
                                                   l_jump_block_n_last );
  }
  /* init C accumulator to zero */
  else {
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      for( l_m = 0; l_m < l_m_total_blocks; l_m++ ) {
        libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                                 LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                                 l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m,
                                                 l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m,
                                                 -1,
                                                 l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m,
                                                 LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                                 LIBXSMM_AARCH64_SVE_TYPE_D );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_store_2dregblock_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                       const unsigned int      i_gp_reg_addr,
                                                       const unsigned int      i_gp_reg_scratch,
                                                       const unsigned int      i_vec_length,
                                                       const unsigned int      i_vec_reg_count,
                                                       const unsigned int      i_m_blocking,
                                                       const unsigned int      i_n_blocking,
                                                       const unsigned int      i_ld ) {
  unsigned int l_vec_reg_acc_start;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* VLEN per l_m interations */
  unsigned int l_m_blocks[3];  /* 0: 128bit, 1: 64bit, 2: 32bit */
  unsigned int l_m_total_blocks;
  unsigned int l_m_bytes = 0;

  /* deriving register blocking from kernel config */
  l_m_blocks[0] =  i_m_blocking/i_vec_length;                    /* number of 128 bit stores */
  l_m_blocks[1] = (i_m_blocking%i_vec_length)/(i_vec_length/2);  /* number of  64 bit stores */
  l_m_blocks[2] = (i_m_blocking%i_vec_length)%(i_vec_length/2);  /* number of  32 but stores */
  l_m_total_blocks = l_m_blocks[0] + l_m_blocks[1] + l_m_blocks[2];
  l_m_bytes = l_m_blocks[0]*16 +  l_m_blocks[1]*8 + l_m_blocks[2]*4;

  /* start register of accumulator */
  l_vec_reg_acc_start = i_vec_reg_count - (i_n_blocking * l_m_total_blocks);

  /* store C accumulator */
  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                              i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, 16,
                                              l_vec_reg_acc_start + l_m + (l_m_total_blocks * l_n),
                                              LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
    }
    for ( l_m = 0; l_m < l_m_blocks[1]; l_m++ ) {
      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                              i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, 8,
                                              l_vec_reg_acc_start + l_m + l_m_blocks[0] + (l_m_total_blocks * l_n),
                                              LIBXSMM_AARCH64_ASIMD_WIDTH_D );
    }
    for ( l_m = 0; l_m < l_m_blocks[2]; l_m++ ) {
      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                              i_gp_reg_addr, LIBXSMM_AARCH64_GP_REG_UNDEF, 4,
                                              l_vec_reg_acc_start + l_m + l_m_blocks[0] + l_m_blocks[1] + (l_m_total_blocks * l_n),
                                              LIBXSMM_AARCH64_ASIMD_WIDTH_S );
    }
    if ( i_ld-l_m_bytes > 0 ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_addr, i_gp_reg_scratch, i_gp_reg_addr,
                                                     (unsigned long long)((unsigned long long)(i_ld-l_m_bytes)) );
    }
  }
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 i_gp_reg_addr, i_gp_reg_scratch, i_gp_reg_addr,
                                                 (unsigned long long)((unsigned long long)(i_ld*i_n_blocking)) );
}

LIBXSMM_API_INTERN
void libxsmm_generator_store_2dregblock_aarch64_sve( libxsmm_generated_code* io_generated_code,
                                                     const unsigned int      i_gp_reg_addr,
                                                     const unsigned int      i_gp_reg_scratch,
                                                     const unsigned int      i_vec_length,
                                                     const unsigned int      i_vec_reg_count,
                                                     const unsigned int      i_m_blocking,
                                                     const unsigned int      i_n_blocking,
                                                     const unsigned int      i_ld,
                                                     const unsigned int      i_data_size ) {
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;

  unsigned int l_m_blocks[2] = { 0 }; /* 0: #full vector stores, 1: #predicate stores (0 or 1) */
  unsigned int l_m_total_blocks = 0;
  unsigned int l_m_bytes_full = 0;
  unsigned int l_vec_reg_acc_start = 0;
  unsigned int l_remainder_size = 0;
  unsigned long long l_jump_block_n_last = 0; /* this is the jump size to be performed after a n-block is complete */

  l_m_blocks[0] = i_m_blocking / i_vec_length;
  l_remainder_size = i_m_blocking % i_vec_length;
  l_m_blocks[1] = (l_remainder_size > 0);
  l_m_total_blocks = l_m_blocks[0] + l_m_blocks[1];
  l_m_bytes_full = l_m_blocks[0] * i_vec_length * i_data_size;

  /* start register of accumulator */
  l_vec_reg_acc_start = i_vec_reg_count - (i_n_blocking * l_m_total_blocks);

  /* stores C accumulator to memory */
  /* full vector stores */
  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    /* this is the jump size to be performed after a m-block is complete */
    unsigned long long l_jump_block_m_last = 0;

    for( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
      libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                            LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                            i_gp_reg_addr,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF,
                                            0,
                                            l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m,
                                            LIBXSMM_AARCH64_SVE_REG_UNDEF );
      /* increase pointer in m-dimension.
          but only if
            1) remainder follows
            or
            2) we are not at the end of the m-loop
      */
      if( l_m_blocks[1] != 0 || l_m != l_m_blocks[0] - 1 ) {
        libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code,
                                                        LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                        i_gp_reg_addr,
                                                        i_gp_reg_addr,
                                                        i_vec_length * i_data_size,
                                                        0 );
      }
      /* combine the m-jump with the n one */
      else {
        l_jump_block_m_last = i_vec_length * i_data_size;
      }
    }

    if( l_m_blocks[1] != 0 ) {
      libxsmm_aarch64_instruction_sve_move( io_generated_code,
                                            LIBXSMM_AARCH64_INSTR_SVE_ST1W_I_OFF,
                                            i_gp_reg_addr,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF,
                                            0,
                                            l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m_blocks[0],
                                            LIBXSMM_AARCH64_SVE_REG_P1 );
    }

    l_jump_block_m_last += i_ld - l_m_bytes_full;

    if( l_n != i_n_blocking - 1 ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_addr,
                                                     i_gp_reg_scratch,
                                                     i_gp_reg_addr,
                                                     l_jump_block_m_last );
    }
    else {
      l_jump_block_n_last = l_jump_block_m_last;
    }
  }

  /* reset C-ptr to original address */
  l_jump_block_n_last = i_ld * i_n_blocking - l_jump_block_n_last;
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code,
                                                  LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                  i_gp_reg_addr,
                                                  i_gp_reg_scratch,
                                                  i_gp_reg_addr,
                                                  l_jump_block_n_last );
}

#if 0
LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_setup_stack_frame( libxsmm_generated_code*            io_generated_code,
                                                       const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                       const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                       libxsmm_micro_kernel_config*       i_micro_kernel_config );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_destroy_stack_frame( libxsmm_generated_code*            io_generated_code,
                                                         const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                         const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                         const libxsmm_micro_kernel_config* i_micro_kernel_config );
#endif

