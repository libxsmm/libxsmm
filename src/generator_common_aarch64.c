/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
*               Intel Corporation - All rights reserved                       *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Breuer (Univ. Jena), Alexander Heinecke, Evangelos Georganas (Intel Corp.)
******************************************************************************/

#include "generator_aarch64_instructions.h"
#include "generator_common_aarch64.h"
#include "generator_common.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
void libxsmm_generator_hinstrps_aarch64( libxsmm_generated_code* io_generated_code,
    unsigned int                                   instr,
    const unsigned int                             i_vec_inout ) {
    libxsmm_aarch64_instruction_asimd_compute( io_generated_code, instr, i_vec_inout, i_vec_inout, 0, i_vec_inout, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
    libxsmm_aarch64_instruction_asimd_compute( io_generated_code, instr, i_vec_inout, i_vec_inout, 0, i_vec_inout, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
}

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
void libxsmm_generator_loop_header_gp_reg_bound_aarch64( libxsmm_generated_code*     io_generated_code,
                                            libxsmm_loop_label_tracker* io_loop_label_tracker,
                                            const unsigned int          i_gp_reg_loop_cnt,
                                            const unsigned int          i_gp_reg_bound ) {
  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ORR_SR, i_gp_reg_bound, i_gp_reg_bound, i_gp_reg_loop_cnt, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
  libxsmm_aarch64_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
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
                                                 (unsigned char)-1,
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

LIBXSMM_API_INTERN
void libxsmm_generator_load_prng_state_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                      const unsigned int      i_gp_reg_prng_state_ptr,
                                                      const unsigned int      prng_state0_vreg,
                                                      const unsigned int      prng_state1_vreg,
                                                      const unsigned int      prng_state2_vreg,
                                                      const unsigned int      prng_state3_vreg ) {
  /* load RNG state */
  libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF, i_gp_reg_prng_state_ptr,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF,   0, prng_state0_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
  libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF, i_gp_reg_prng_state_ptr,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF,  64, prng_state1_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
  libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF, i_gp_reg_prng_state_ptr,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 128, prng_state2_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
  libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF, i_gp_reg_prng_state_ptr,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 192, prng_state3_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
}

LIBXSMM_API_INTERN
void libxsmm_generator_store_prng_state_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                       const unsigned int      i_gp_reg_prng_state_ptr,
                                                       const unsigned int      prng_state0_vreg,
                                                       const unsigned int      prng_state1_vreg,
                                                       const unsigned int      prng_state2_vreg,
                                                       const unsigned int      prng_state3_vreg ) {
  /* store RNG state */
  libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF, i_gp_reg_prng_state_ptr,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF,   0, prng_state0_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
  libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF, i_gp_reg_prng_state_ptr,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF,  64, prng_state1_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
  libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF, i_gp_reg_prng_state_ptr,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 128, prng_state2_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
  libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF, i_gp_reg_prng_state_ptr,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 192, prng_state3_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
}

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_dropout_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                      const unsigned int      i_gp_reg_tmp,
                                                      const unsigned int      i_gp_reg_prob_ptr,
                                                      const unsigned int      dropout_vreg_one,
                                                      const unsigned int      dropout_prob_vreg,
                                                      const unsigned int      dropout_invprob_vreg ) {
  /* load constant register */
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp, 0x3f800000 );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I,
                                                 LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 16, 0 );
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XZR,
                                        0, i_gp_reg_tmp);
  libxsmm_aarch64_instruction_asimd_struct_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                 dropout_vreg_one, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                 LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 16, 0 );

  /* load probability */
  libxsmm_aarch64_instruction_asimd_struct_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R, i_gp_reg_prob_ptr, LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                 dropout_prob_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FSUB_V,
                                             dropout_vreg_one, dropout_prob_vreg, 0, dropout_prob_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  /* load 1/prob */
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FDIV_V,
                                             dropout_vreg_one, dropout_prob_vreg, 0, dropout_invprob_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
}

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_dropout_inv_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                          const unsigned int      i_gp_reg_tmp,
                                                          const unsigned int      i_gp_reg_prob_ptr,
                                                          const unsigned int      dropout_vreg_one,
                                                          const unsigned int      dropout_vreg_zero,
                                                          const unsigned int      dropout_prob_vreg ) {
  /* load constant register */
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp, 0x3f800000 );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I,
                                                 LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 16, 0 );
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XZR,
                                        0, i_gp_reg_tmp);
  libxsmm_aarch64_instruction_asimd_struct_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                 dropout_vreg_one, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                 LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 16, 0 );

  /* load probability */
  libxsmm_aarch64_instruction_asimd_struct_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R, i_gp_reg_prob_ptr, LIBXSMM_AARCH64_GP_REG_UNDEF,
                                                 dropout_prob_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FSUB_V,
                                             dropout_vreg_one, dropout_prob_vreg, 0, dropout_prob_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  /* load 1/prob */
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FDIV_V,
                                             dropout_vreg_one, dropout_prob_vreg, 0, dropout_prob_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  /* load zero, for masking */
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                             dropout_vreg_zero, dropout_vreg_zero, 0, dropout_vreg_zero, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
}

LIBXSMM_API_INTERN
void libxsmm_generator_xoshiro128pp_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                   const unsigned int      i_vec_reg_rng_state_0,
                                                   const unsigned int      i_vec_reg_rng_state_1,
                                                   const unsigned int      i_vec_reg_rng_state_2,
                                                   const unsigned int      i_vec_reg_rng_state_3,
                                                   const unsigned int      i_vec_reg_rng_tmp_0,
                                                   const unsigned int      i_vec_reg_rng_tmp_1,
                                                   const unsigned int      o_vec_reg_rng ) {
  LIBXSMM_UNUSED(io_generated_code);
  LIBXSMM_UNUSED(i_vec_reg_rng_state_0);
  LIBXSMM_UNUSED(i_vec_reg_rng_state_1);
  LIBXSMM_UNUSED(i_vec_reg_rng_state_2);
  LIBXSMM_UNUSED(i_vec_reg_rng_state_3);
  LIBXSMM_UNUSED(i_vec_reg_rng_tmp_0);
  LIBXSMM_UNUSED(i_vec_reg_rng_tmp_1);
  LIBXSMM_UNUSED(o_vec_reg_rng);
  /* @TODO: needs validation */
#if 0
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ADD_V,
                                             i_vec_reg_rng_state_0, i_vec_reg_rng_state_3, 0, i_vec_reg_rng_tmp_0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_SHL_I_V,
                                             i_vec_reg_rng_tmp_0, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 7, i_vec_reg_rng_tmp_1, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_USHR_I_V,
                                             i_vec_reg_rng_tmp_0, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 25, i_vec_reg_rng_tmp_0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_reg_rng_tmp_0, i_vec_reg_rng_tmp_1, 0, i_vec_reg_rng_tmp_0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ADD_V,
                                             i_vec_reg_rng_tmp_0, i_vec_reg_rng_state_0, 0, o_vec_reg_rng, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_SHL_I_V,
                                             i_vec_reg_rng_state_1, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 9, i_vec_reg_rng_tmp_0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                             i_vec_reg_rng_state_2, i_vec_reg_rng_state_0, 0, i_vec_reg_rng_state_2, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                             i_vec_reg_rng_state_3, i_vec_reg_rng_state_1, 0, i_vec_reg_rng_state_3, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                             i_vec_reg_rng_state_1, i_vec_reg_rng_state_2, 0, i_vec_reg_rng_state_1, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                             i_vec_reg_rng_state_0, i_vec_reg_rng_state_3, 0, i_vec_reg_rng_state_0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                             i_vec_reg_rng_state_2, i_vec_reg_rng_tmp_0, 0, i_vec_reg_rng_state_2, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_SHL_I_V,
                                             i_vec_reg_rng_state_3, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 11, i_vec_reg_rng_tmp_0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_USHR_I_V,
                                             i_vec_reg_rng_state_3, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 21, i_vec_reg_rng_tmp_1, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_reg_rng_tmp_0, i_vec_reg_rng_tmp_1, 0, i_vec_reg_rng_state_3, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
#endif
}

LIBXSMM_API_INTERN
void libxsmm_generator_xoshiro128p_f32_aarch64_asimd( libxsmm_generated_code* io_generated_code,
                                                      const unsigned int      i_vec_reg_rng_state_0,
                                                      const unsigned int      i_vec_reg_rng_state_1,
                                                      const unsigned int      i_vec_reg_rng_state_2,
                                                      const unsigned int      i_vec_reg_rng_state_3,
                                                      const unsigned int      i_vec_reg_rng_tmp_0,
                                                      const unsigned int      i_vec_reg_rng_tmp_1,
                                                      const unsigned int      i_vec_reg_rng_one,
                                                      const unsigned int      o_vec_reg_rng ) {
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ADD_V,
                                             i_vec_reg_rng_state_0, i_vec_reg_rng_state_3, 0, o_vec_reg_rng, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_USHR_I_V,
                                             o_vec_reg_rng, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 9, o_vec_reg_rng, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             o_vec_reg_rng, i_vec_reg_rng_one, 0, o_vec_reg_rng, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FSUB_V,
                                             o_vec_reg_rng, i_vec_reg_rng_one, 0, o_vec_reg_rng, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_SHL_I_V,
                                             i_vec_reg_rng_state_1, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 9, i_vec_reg_rng_tmp_0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                             i_vec_reg_rng_state_2, i_vec_reg_rng_state_0, 0, i_vec_reg_rng_state_2, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                             i_vec_reg_rng_state_3, i_vec_reg_rng_state_1, 0, i_vec_reg_rng_state_3, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                             i_vec_reg_rng_state_1, i_vec_reg_rng_state_2, 0, i_vec_reg_rng_state_1, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                             i_vec_reg_rng_state_0, i_vec_reg_rng_state_3, 0, i_vec_reg_rng_state_0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                             i_vec_reg_rng_state_2, i_vec_reg_rng_tmp_0, 0, i_vec_reg_rng_state_2, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_SHL_I_V,
                                             i_vec_reg_rng_state_3, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 11, i_vec_reg_rng_tmp_0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_USHR_I_V,
                                             i_vec_reg_rng_state_3, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 21, i_vec_reg_rng_tmp_1, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                             i_vec_reg_rng_tmp_0, i_vec_reg_rng_tmp_1, 0, i_vec_reg_rng_state_3, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
}

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_gelu_ps_minimax3_aarch64( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_thres,
    const unsigned int                             i_vec_absmask,
    const unsigned int                             i_vec_scale,
    const unsigned int                             i_vec_shifter,
    const unsigned int                             i_vec_half,
    const unsigned int                             i_vec_c0,
    const unsigned int                             i_vec_c1,
    const unsigned int                             i_vec_c2,
    const unsigned int                             i_gp_reg_tmp,
    const libxsmm_aarch64_asimd_tupletype i_tupletype ) {

  unsigned long long thres_array = 0x40879fff ;
  unsigned long long absmask_array = 0x7fffffff ;
  unsigned long long scale_array = 0x406a0ea1 ;
  unsigned long long shifter_array = 0x4b400000 ;
  unsigned long long half_array = 0x3f000000 ;
/*   unsigned long long c0_array = 0x3ecc4231u ;
  unsigned int c1_array = 0xb7c7fb58u ;
  unsigned long long c2_array = 0xbd877b85u ; */
/*   unsigned long long c0_array[16] = { 0x3f4c4245u, 0x3f4c927bu, 0x3f5085f8u, 0x3f5d7bdau, 0x3f73ea12u, 0x3f86142fu, 0x3f8d3df4u, 0x3f8b4b0fu, 0x3f8022c8u, 0x3f5e5423u, 0x3f39ceb5u, 0x3f199bedu, 0x3f00bee0u, 0x3ede1737u, 0x3ec59b86u, 0x3eb4454cu };
  unsigned long long c1_array[16] = { 0xb930e738u, 0xbc4b28bau, 0xbda4212fu, 0xbe5feb0eu, 0xbec8b0e5u, 0xbf09e61bu, 0xbf1c403fu, 0xbf185954u, 0xbf03e1eeu, 0xbed08a61u, 0xbe9b4508u, 0xbe61788bu, 0xbe257770u, 0xbdfc542au, 0xbdca014eu, 0xbda8d7e9u };
  unsigned long long c2_array[16] = { 0xbe87047bu, 0xbe6eb875u, 0xbe2210c1u, 0xbd81727fu, 0x3cb9625cu, 0x3da2cbe8u, 0x3dd1d4d1u, 0x3dca0bd0u, 0x3da47dd0u, 0x3d6f1bd3u, 0x3d216381u, 0x3cd2618cu, 0x3c89f6e6u, 0x3c3ca672u, 0x3c08ed08u, 0x3bd26a14u };

 */
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec ( io_generated_code, i_vec_thres, i_gp_reg_tmp, i_tupletype , thres_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec ( io_generated_code, i_vec_absmask, i_gp_reg_tmp, i_tupletype , absmask_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec ( io_generated_code, i_vec_scale, i_gp_reg_tmp, i_tupletype , scale_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec ( io_generated_code, i_vec_shifter, i_gp_reg_tmp, i_tupletype , shifter_array  );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec ( io_generated_code, i_vec_half, i_gp_reg_tmp, i_tupletype , half_array );
/*   libxsmm_aarch64_instruction_full_vec_load ( io_generated_code, i_vec_c0, i_gp_reg_tmp, i_tupletype , &c0_array );
  libxsmm_aarch64_instruction_full_vec_load ( io_generated_code, i_vec_c1, i_gp_reg_tmp, i_tupletype , &c1_array );
  libxsmm_aarch64_instruction_full_vec_load ( io_generated_code, i_vec_c2, i_gp_reg_tmp, i_tupletype , &c2_array ); */
}

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_gelu_inv_ps_minimax3_aarch64( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_thres,
    const unsigned int                             i_vec_absmask,
    const unsigned int                             i_vec_scale,
    const unsigned int                             i_vec_shifter,
    const unsigned int                             i_vec_half,
    const unsigned int                             i_vec_c0,
    const unsigned int                             i_vec_c1,
    const unsigned int                             i_vec_c2,
    const unsigned int                             i_gp_reg_tmp,
    const libxsmm_aarch64_asimd_tupletype          i_tupletype ) {

  unsigned long long thres_array = 0x408f5fff ;
  unsigned long long absmask_array = 0x7fffffff ;
  unsigned long long scale_array = 0x405d67c9 ;
  unsigned long long shifter_array = 0x4b400000 ;
  unsigned long long half_array = 0x3f000000 ;
/*   unsigned long long c0_array = 0x3f4c4245u ;
  unsigned int c1_array = 0xb930e738u ;
  unsigned long long c2_array = 0xbe87047bu ; */

/*   unsigned int c0_array[16] = { 0x3f4c4245u, 0x3f4c927bu, 0x3f5085f8u, 0x3f5d7bdau, 0x3f73ea12u, 0x3f86142fu, 0x3f8d3df4u, 0x3f8b4b0fu, 0x3f8022c8u, 0x3f5e5423u, 0x3f39ceb5u, 0x3f199bedu, 0x3f00bee0u, 0x3ede1737u, 0x3ec59b86u, 0x3eb4454cu };
  unsigned int c1_array[16] = { 0xb930e738u, 0xbc4b28bau, 0xbda4212fu, 0xbe5feb0eu, 0xbec8b0e5u, 0xbf09e61bu, 0xbf1c403fu, 0xbf185954u, 0xbf03e1eeu, 0xbed08a61u, 0xbe9b4508u, 0xbe61788bu, 0xbe257770u, 0xbdfc542au, 0xbdca014eu, 0xbda8d7e9u };
  unsigned int c2_array[16] = { 0xbe87047bu, 0xbe6eb875u, 0xbe2210c1u, 0xbd81727fu, 0x3cb9625cu, 0x3da2cbe8u, 0x3dd1d4d1u, 0x3dca0bd0u, 0x3da47dd0u, 0x3d6f1bd3u, 0x3d216381u, 0x3cd2618cu, 0x3c89f6e6u, 0x3c3ca672u, 0x3c08ed08u, 0x3bd26a14u };
 */

  libxsmm_aarch64_instruction_broadcast_scalar_to_vec ( io_generated_code, i_vec_thres, i_gp_reg_tmp, i_tupletype , thres_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec ( io_generated_code, i_vec_absmask, i_gp_reg_tmp, i_tupletype , absmask_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec ( io_generated_code, i_vec_scale, i_gp_reg_tmp, i_tupletype , scale_array );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec ( io_generated_code, i_vec_shifter, i_gp_reg_tmp, i_tupletype , shifter_array  );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec ( io_generated_code, i_vec_half, i_gp_reg_tmp, i_tupletype , half_array );
 /*  libxsmm_aarch64_instruction_full_vec_load ( io_generated_code, i_vec_c0, i_gp_reg_tmp, i_tupletype , c0_array );
  libxsmm_aarch64_instruction_full_vec_load ( io_generated_code, i_vec_c1, i_gp_reg_tmp, i_tupletype , c1_array );
  libxsmm_aarch64_instruction_full_vec_load ( io_generated_code, i_vec_c2, i_gp_reg_tmp, i_tupletype , c2_array );
 */
}

LIBXSMM_API_INTERN
void libxsmm_generator_gelu_ps_minimax3_aarch64( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_x,
    const unsigned int                             i_vec_xr,
    const unsigned int                             i_vec_xa,
    const unsigned int                             i_vec_index,
    const unsigned int                             i_vec_C0,
    const unsigned int                             i_vec_C1,
    const unsigned int                             i_vec_C2,
    const unsigned int                             i_vec_thres,
    const unsigned int                             i_vec_absmask,
    const unsigned int                             i_vec_scale,
    const unsigned int                             i_vec_shifter,
    const unsigned int                             i_vec_half,
    const unsigned int                             i_vec_c0,
    const unsigned int                             i_vec_c1,
    const unsigned int                             i_vec_c2,
    const libxsmm_aarch64_asimd_tupletype          i_tupletype ) {
/*
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                              i_vec_C0, i_vec_C0, 0, i_vec_x,
                                              i_tupletype ); */

#if 0
  if (io_generated_code->arch >= LIBXSMM_X86_AVX512_CORE) {
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
        LIBXSMM_X86_INSTR_VRANGEPS, 'z',
        i_vec_thres,
        i_vec_x,
        i_vec_xr,
        0, 0, 8, 2);


    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                         LIBXSMM_X86_INSTR_VPANDD,
                                         'z',
                                         i_vec_xr, i_vec_absmask, i_vec_xa );
  } else {
    libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64, 'z', i_vec_x, i_vec_xr );
    libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSRLD_I, 'z', i_vec_xr, i_vec_xr, 31 );
    libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, 'z', i_vec_xr, i_vec_xr, 31 );
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPANDD, 'z', i_vec_x, i_vec_absmask, i_vec_xa );
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMINPS, 'z',  i_vec_xa, i_vec_thres, i_vec_xa );
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPORD, 'z', i_vec_xr, i_vec_xa, i_vec_xr );
  }

  libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VMOVDQU64,
                                       'z',
                                       i_vec_xa, i_vec_index );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'z',
                                       i_vec_shifter, i_vec_scale, i_vec_index );

  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
      LIBXSMM_X86_INSTR_VPERMD, 'z',
      i_vec_c2,
      i_vec_index,
      i_vec_C2,
      0, 0, 0, 0);

  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
      LIBXSMM_X86_INSTR_VPERMD, 'z',
      i_vec_c1,
      i_vec_index,
      i_vec_C1,
      0, 0, 0, 0);

  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
      LIBXSMM_X86_INSTR_VPERMD, 'z',
      i_vec_c0,
      i_vec_index,
      i_vec_C0,
      0, 0, 0, 0);

   libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'z',
                                       i_vec_C1, i_vec_xa, i_vec_C2 );

   libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'z',
                                       i_vec_C0, i_vec_xa, i_vec_C2 );

   libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'z',
                                       i_vec_half, i_vec_xr, i_vec_C2 );

   libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VMULPS,
                                       'z',
                                       i_vec_x, i_vec_C2, i_vec_x );
#endif
}

LIBXSMM_API_INTERN
void libxsmm_generator_gelu_inv_ps_minimax3_aarch64(  libxsmm_generated_code*                        io_generated_code,
                                                      const unsigned int                             i_vec_x,
                                                      const unsigned int                             i_vec_xr,
                                                      const unsigned int                             i_vec_xa,
                                                      const unsigned int                             i_vec_index,
                                                      const unsigned int                             i_vec_C0,
                                                      const unsigned int                             i_vec_C1,
                                                      const unsigned int                             i_vec_C2,
                                                      const unsigned int                             i_vec_thres,
                                                      const unsigned int                             i_vec_absmask,
                                                      const unsigned int                             i_vec_scale,
                                                      const unsigned int                             i_vec_shifter,
                                                      const unsigned int                             i_vec_half,
                                                      const unsigned int                             i_vec_c0,
                                                      const unsigned int                             i_vec_c1,
                                                      const unsigned int                             i_vec_c2,
                                                      const libxsmm_aarch64_asimd_tupletype          i_tupletype ) {
#if 0
  if (io_generated_code->arch >= LIBXSMM_X86_AVX512_CORE) {
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
        LIBXSMM_X86_INSTR_VRANGEPS, 'z',
        i_vec_thres,
        i_vec_x,
        i_vec_xr,
        0, 0, 8, 2);


    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                         LIBXSMM_X86_INSTR_VPANDD,
                                         'z',
                                         i_vec_xr, i_vec_absmask, i_vec_xa );
  } else {
    libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64, 'z', i_vec_x, i_vec_xr );
    libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSRLD_I, 'z', i_vec_xr, i_vec_xr, 31 );
    libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, 'z', i_vec_xr, i_vec_xr, 31 );
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPANDD, 'z', i_vec_x, i_vec_absmask, i_vec_xa );
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMINPS, 'z',  i_vec_xa, i_vec_thres, i_vec_xa );
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPORD, 'z', i_vec_xr, i_vec_xa, i_vec_xr );
  }

  libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VMOVDQU64,
                                       'z',
                                       i_vec_xa, i_vec_index );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'z',
                                       i_vec_shifter, i_vec_scale, i_vec_index );

  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
      LIBXSMM_X86_INSTR_VPERMD, 'z',
      i_vec_c2,
      i_vec_index,
      i_vec_C2,
      0, 0, 0, 0);

  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
      LIBXSMM_X86_INSTR_VPERMD, 'z',
      i_vec_c1,
      i_vec_index,
      i_vec_C1,
      0, 0, 0, 0);

  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
      LIBXSMM_X86_INSTR_VPERMD, 'z',
      i_vec_c0,
      i_vec_index,
      i_vec_C0,
      0, 0, 0, 0);

   libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'z',
                                       i_vec_C1, i_vec_xa, i_vec_C2 );

   libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'z',
                                       i_vec_C0, i_vec_xa, i_vec_C2 );

   libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'z',
                                       i_vec_half, i_vec_xr, i_vec_C2 );

   libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VMOVDQU64,
                                       'z',
                                       i_vec_C2, i_vec_x );
#endif
}

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_exp_ps_3dts_aarch64( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_c0,
    const unsigned int                             i_vec_c1,
    const unsigned int                             i_vec_c2,
    const unsigned int                             i_vec_c3,
    const unsigned int                             i_vec_halves,
    const unsigned int                             i_vec_log2e,
    const unsigned int                             i_vec_expmask,
    const unsigned int                             i_vec_hi_bound,
    const unsigned int                             i_vec_lo_bound,
    const unsigned int                             i_gp_reg_tmp,
    const libxsmm_aarch64_asimd_tupletype i_tupletype ) {

  unsigned long long vec_c0 = 0x3f34e022 ;
  unsigned long long vec_c1 = 0x3efd357f ;
  unsigned long long vec_c2 = 0x3e20bcd5 ;
  unsigned long long vec_c3 = 0x3d635847 ;
  unsigned long long vec_halves = 0x3f000000 ;
  unsigned long long vec_log2e = 0x3fb8aa3b ;
  unsigned int vec_expmask = 0x7f ;
  unsigned long long vec_hi_bound = 0x42b0c0a5 ;
  unsigned long long vec_lo_bound = 0xc2b0c0a5 ;

  libxsmm_aarch64_instruction_broadcast_scalar_to_vec ( io_generated_code, i_vec_halves, i_gp_reg_tmp, i_tupletype , vec_halves );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec ( io_generated_code, i_vec_log2e, i_gp_reg_tmp, i_tupletype , vec_log2e );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec ( io_generated_code, i_vec_c0, i_gp_reg_tmp, i_tupletype , vec_c0 );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec ( io_generated_code, i_vec_c1, i_gp_reg_tmp, i_tupletype , vec_c1  );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec ( io_generated_code, i_vec_c2, i_gp_reg_tmp, i_tupletype , vec_c2 );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec ( io_generated_code, i_vec_c3, i_gp_reg_tmp, i_tupletype , vec_c3 );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec ( io_generated_code, i_vec_expmask, i_gp_reg_tmp, i_tupletype , vec_expmask );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec ( io_generated_code, i_vec_hi_bound, i_gp_reg_tmp, i_tupletype , vec_hi_bound );
  libxsmm_aarch64_instruction_broadcast_scalar_to_vec ( io_generated_code, i_vec_lo_bound, i_gp_reg_tmp, i_tupletype , vec_lo_bound );
}

LIBXSMM_API_INTERN
void libxsmm_generator_exp_ps_3dts_aarch64( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_x,
    const unsigned int                             i_vec_y,
    const unsigned int                             i_vec_z,
    const unsigned int                             i_vec_c0,
    const unsigned int                             i_vec_c1,
    const unsigned int                             i_vec_c2,
    const unsigned int                             i_vec_c3,
    const unsigned int                             i_vec_halves,
    const unsigned int                             i_vec_log2e,
    const unsigned int                             i_vec_expmask,
    const unsigned int                             i_vec_hi_bound,
    const unsigned int                             i_vec_lo_bound,
    const libxsmm_aarch64_asimd_tupletype i_tupletype ) {

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMAX_V, i_vec_x, i_vec_lo_bound, 0, i_vec_x, i_tupletype );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMIN_V, i_vec_x, i_vec_hi_bound, 0, i_vec_x, i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                              i_vec_x, i_vec_log2e, 0, i_vec_x,
                                              i_tupletype  );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V,
                                              i_vec_halves, i_vec_x, 0, i_vec_x,
                                              i_tupletype  );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FRINTM_V,
                                              i_vec_x, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, i_vec_y,
                                              i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FSUB_V,
                                              i_vec_x, i_vec_y, 0, i_vec_y,
                                              i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                              i_vec_y, i_vec_y, 0, i_vec_z,
                                              i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                              i_vec_z, i_vec_c3, 0, i_vec_z,
                                              i_tupletype  );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V,
                                              i_vec_z, i_vec_c2, 0, i_vec_z,
                                              i_tupletype  );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                              i_vec_z, i_vec_y, 0, i_vec_z,
                                              i_tupletype  );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V,
                                              i_vec_c1, i_vec_z, 0, i_vec_z,
                                              i_tupletype  );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                              i_vec_z, i_vec_y, 0, i_vec_z,
                                              i_tupletype  );
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V,
                                              i_vec_c0, i_vec_z, 0, i_vec_z,
                                              i_tupletype  );

  libxsmm_generator_scalefps_aarch64( io_generated_code, i_vec_z, i_vec_x, i_vec_x, i_vec_expmask, i_tupletype );
}

LIBXSMM_API_INTERN
void libxsmm_generator_scalefps_aarch64( libxsmm_generated_code*                        io_generated_code,
                                        const unsigned int                             i_vec_x,
                                        const unsigned int                             i_vec_y,
                                        const unsigned int                             i_vec_z,
                                        const unsigned int                             i_vec_expmask,
                                        const libxsmm_aarch64_asimd_tupletype i_tupletype  ) {
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FRINTM_V,
                                              i_vec_y, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, i_vec_y,
                                              i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FCVTMS_V,
                                              i_vec_y, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, i_vec_y,
                                              i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ADD_V,
                                              i_vec_y, i_vec_expmask, 0, i_vec_y,
                                              i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_SHL_I_V,
                                              i_vec_y, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 23, i_vec_y,
                                              i_tupletype );

  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                              i_vec_x, i_vec_y, 0, i_vec_z,
                                              i_tupletype );
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_broadcast_scalar_to_vec ( libxsmm_generated_code *io_generated_code,
                                                              const unsigned char     i_vec_reg,
                                                              const unsigned int      i_gp_reg_tmp,
                                                              const libxsmm_aarch64_asimd_tupletype i_tupletype,
                                                              const unsigned long long imm64) {

    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp, imm64 );
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I,
                                                    LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 16, 0 );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XZR,
    0, i_gp_reg_tmp);
    libxsmm_aarch64_instruction_asimd_struct_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R,
                                                    LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_UNDEF, i_vec_reg,
                                                    i_tupletype );
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                    LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 16, 0 );
}

#if 0
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_full_vec_load( libxsmm_generated_code *io_generated_code,
                                                const unsigned char     i_vec_reg,
                                                const unsigned int      i_gp_reg_tmp,
                                                const libxsmm_aarch64_asimd_tupletype i_tupletype,
                                                const unsigned long long imm64) {

    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp, imm64 );
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I,
                                                    LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 16, 0 );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XZR,
    0, i_gp_reg_tmp);
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                            LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_UNDEF, 0,
                                            i_vec_reg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q);
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                    LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 16, 0 );
}
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

