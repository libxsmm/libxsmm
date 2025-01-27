/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
*               Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Siddharth Rai, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_mateltwise_rv64.h"
#include "generator_rv64_instructions.h"
#include "generator_common_rv64.h"
#include "generator_common.h"
#include "generator_mateltwise_unary_binary_rv64.h"

LIBXSMM_API_INTERN
void libxsmm_generator_loop_header_rv64( libxsmm_generated_code*     io_generated_code,
                                         libxsmm_loop_label_tracker* io_loop_label_tracker,
                                         const unsigned int          i_gp_reg_loop_cnt,
                                         const unsigned int          i_trips ) {
  libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_loop_cnt, i_trips );
  libxsmm_rv64_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_loop_footer_rv64( libxsmm_generated_code*     io_generated_code,
                                         libxsmm_loop_label_tracker* io_loop_label_tracker,
                                         const unsigned int          i_gp_reg_loop_cnt,
                                         const unsigned int          i_loop_blocking ) {
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI,
                                                 i_gp_reg_loop_cnt, i_gp_reg_loop_cnt, -1 * i_loop_blocking );

  libxsmm_rv64_instruction_cond_jump_back_to_label( io_generated_code, LIBXSMM_RV64_INSTR_GP_BNE,
                                                    i_gp_reg_loop_cnt, LIBXSMM_RV64_GP_REG_X0, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_bcastload_masked_vreg_rv64( libxsmm_generated_code* io_generated_code,
                                                   const unsigned int      i_gp_reg_addr,
                                                   const unsigned int      i_gp_reg_scratch,
                                                   const unsigned int      i_vec_reg,
                                                   const unsigned int      i_datatype_size,
                                                   const unsigned int      i_masked_elems,
                                                   const unsigned int      i_vlen,
                                                   const unsigned int      i_avlen,
                                                   const unsigned int      i_adv_gpr ) {
  unsigned char l_offset = (unsigned char)(( i_adv_gpr == 0 ) ? 0 : i_datatype_size);

  /* different element sizes use different instructions; load a single element, broadcast it, and set the rest to zero */
  int l_instr = i_datatype_size == 1 ? LIBXSMM_RV64_INSTR_RVV_VLE8_V :
    i_datatype_size == 2 ? LIBXSMM_RV64_INSTR_RVV_VLE16_V : i_datatype_size == 4 ? LIBXSMM_RV64_INSTR_RVV_VLE32_V :
    LIBXSMM_RV64_INSTR_RVV_VLE64_V ;
  int l_sew = i_datatype_size == 1 ? LIBXSMM_RV64_SEW_B : i_datatype_size == 2 ? LIBXSMM_RV64_SEW_W : i_datatype_size == 4 ? LIBXSMM_RV64_SEW_D : LIBXSMM_RV64_SEW_Q ;

  /* Set vector length and load required number of elements */
  if (i_masked_elems)
    libxsmm_rv64_instruction_rvv_setvli( io_generated_code, i_avlen, LIBXSMM_RV64_GP_REG_X31, l_sew, LIBXSMM_RV64_LMUL_M1);

  libxsmm_rv64_instruction_rvv_move( io_generated_code, l_instr, i_gp_reg_addr, i_gp_reg_scratch, i_vec_reg, 1 );

  /* Broadcast loaded value to */
  libxsmm_rv64_instruction_rvv_compute( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VFMV_F_S, LIBXSMM_RV64_GP_REG_V0, i_vec_reg, LIBXSMM_RV64_GP_REG_F10, 1);

  libxsmm_rv64_instruction_rvv_compute( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VFMV_V_F, LIBXSMM_RV64_GP_REG_F10, LIBXSMM_RV64_GP_REG_V0, i_vec_reg, 1);

  if ( l_offset ){/* post increment address by offset */
    libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, i_gp_reg_addr, i_gp_reg_addr, l_offset);
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_load_2dregblock_rv64_rvv( libxsmm_generated_code* io_generated_code,
                                                 const libxsmm_datatype  i_datatype,
                                                 const unsigned int      i_gp_reg_addr,
                                                 const unsigned int      i_gp_reg_scratch,
                                                 const unsigned int      i_vec_length,
                                                 const unsigned int      i_vec_reg_count,
                                                 const unsigned int      i_m_blocking,
                                                 const unsigned int      i_n_blocking,
                                                 const unsigned int      i_ld,
                                                 const unsigned int      i_zero ) {
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;

  unsigned int l_m_blocks[2] = { 0 }; /* 0: #full vector loads, 1: #predicated loads (0 or 1) */
  unsigned int l_m_total_blocks = 0;
  unsigned int l_m_bytes_full = 0;
  unsigned int l_vec_reg_acc_start = 0;
  unsigned int l_remainder_size = 0;
  unsigned int l_datatype_size = LIBXSMM_TYPESIZE(i_datatype);
  unsigned int l_load_instr = (l_datatype_size == 4) ? LIBXSMM_RV64_INSTR_RVV_VLE32_V : LIBXSMM_RV64_INSTR_RVV_VLE64_V;
  unsigned int l_masked_load_instr = (l_datatype_size == 4) ? LIBXSMM_RV64_INSTR_RVV_VLE32_V : LIBXSMM_RV64_INSTR_RVV_VLE64_V;

  l_m_blocks[0] = i_m_blocking / i_vec_length;
  l_remainder_size = i_m_blocking % i_vec_length;
  l_m_blocks[1] = (l_remainder_size > 0);
  l_m_total_blocks = l_m_blocks[0] + l_m_blocks[1];
  l_m_bytes_full = l_m_blocks[0] * i_vec_length * l_datatype_size;

  /* start register of accumulator */
  l_vec_reg_acc_start = i_vec_reg_count - (i_n_blocking * l_m_total_blocks);

  /* loads C accumulator from memory */
  if ( i_zero == 0 ) {
    /* this is the jump size to be performed after a n-block is complete */
    unsigned long long l_jump_block_n_last = 0;

    /* full vector loads */
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      /* this is the jump size to be performed after a m-block is complete */
      unsigned long long l_jump_block_m_last = 0;

      for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
        libxsmm_rv64_instruction_rvv_move( io_generated_code,
                                           l_load_instr,
                                           i_gp_reg_addr,
                                           0,
                                           l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m,
                                           1 );

        /* increase pointer in m-dimension.
           but only if
             1) remainder follows
             or
             2) we are not at the end of the m-loop
        */
        if ( l_m_blocks[1] != 0 || l_m != l_m_blocks[0] - 1 ) {
          libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code,
                                                      LIBXSMM_RV64_INSTR_GP_ADD,
                                                      i_gp_reg_addr,
                                                      i_gp_reg_scratch,
                                                      i_gp_reg_addr,
                                                      i_vec_length * l_datatype_size
                                                      );
        }
        /* combine the m-jump with the n one*/
        else {
          l_jump_block_m_last = (long long)i_vec_length * l_datatype_size;
        }
      }

      if ( l_m_blocks[1] != 0 ) {
        libxsmm_rv64_instruction_rvv_move( io_generated_code,
                                           l_masked_load_instr,
                                           i_gp_reg_addr,
                                           0,
                                           l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m_blocks[0],
                                           1 );
      }

      l_jump_block_m_last += (long long)i_ld - l_m_bytes_full;

      if ( l_n != i_n_blocking - 1 ) {
        libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code,
                                                    LIBXSMM_RV64_INSTR_GP_ADD,
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
    l_jump_block_n_last = (long long)i_ld * i_n_blocking - l_jump_block_n_last;
    libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code,
                                                   LIBXSMM_RV64_INSTR_GP_SUB,
                                                   i_gp_reg_addr,
                                                   i_gp_reg_scratch,
                                                   i_gp_reg_addr,
                                                   l_jump_block_n_last );
  }
  /* init C accumulator to zero */
  else {
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      for ( l_m = 0; l_m < l_m_total_blocks; l_m++ ) {
        libxsmm_rv64_instruction_rvv_compute( io_generated_code,
                                                 LIBXSMM_RV64_INSTR_RVV_VXOR_VV,
                                                 l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m,
                                                 l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m,
                                                 l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m,
                                                 1 );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_store_2dregblock_rv64_rvv( libxsmm_generated_code* io_generated_code,
                                                     const libxsmm_datatype  i_datatype,
                                                     const unsigned int      i_gp_reg_addr,
                                                     const unsigned int      i_gp_reg_scratch,
                                                     const unsigned int      i_vec_length,
                                                     const unsigned int      i_vec_reg_count,
                                                     const unsigned int      i_m_blocking,
                                                     const unsigned int      i_n_blocking,
                                                     const unsigned int      i_ld,
                                                     const libxsmm_datatype  i_inp_datatype,
                                                     const unsigned int      i_aux_gp_reg,
                                                     const unsigned int      i_reduce_on_output  ) {
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
  unsigned int l_datatype_size = LIBXSMM_TYPESIZE(i_datatype);
  unsigned int l_store_instr =  (l_datatype_size == 4) ? LIBXSMM_RV64_INSTR_RVV_VSE32_V : LIBXSMM_RV64_INSTR_RVV_VSE64_V;
  unsigned int l_masked_store_instr = (l_datatype_size == 4) ? LIBXSMM_RV64_INSTR_RVV_VSE32_V : LIBXSMM_RV64_INSTR_RVV_VSE64_V;
  unsigned int l_tmp_vreg = 0;
  unsigned int l_tmp_vreg2 = 0;

  LIBXSMM_UNUSED(l_tmp_vreg);
  LIBXSMM_UNUSED(l_tmp_vreg2);
  LIBXSMM_UNUSED(l_m_bytes_full);

  l_m_blocks[0] = i_m_blocking / i_vec_length;
  l_remainder_size = i_m_blocking % i_vec_length;
  l_m_blocks[1] = (l_remainder_size > 0);
  l_m_total_blocks = l_m_blocks[0] + l_m_blocks[1];
  l_m_bytes_full = l_m_blocks[0] * i_vec_length * l_datatype_size;

  /* start register of accumulator */
  l_vec_reg_acc_start = i_vec_reg_count - (i_n_blocking * l_m_total_blocks);
  l_tmp_vreg = l_vec_reg_acc_start - 1;
  l_tmp_vreg2 = l_vec_reg_acc_start - 2;

  /* stores C accumulator to memory */
  /* full vector stores */
  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    /* this is the jump size to be performed after a m-block is complete */
    unsigned long long l_jump_block_m_last = 0;

    for ( l_m = 0; l_m < l_m_blocks[0]; l_m++ ) {
      libxsmm_rv64_instruction_rvv_move( io_generated_code,
                                            l_store_instr,
                                            i_gp_reg_addr,
                                            0,
                                            l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m,
                                            1);
      /* increase pointer in m-dimension.
          but only if
            1) remainder follows
            or
            2) we are not at the end of the m-loop
      */
      if ( l_m_blocks[1] != 0 || l_m != l_m_blocks[0] - 1 ) {
        libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code,
                                                        LIBXSMM_RV64_INSTR_GP_ADDI,
                                                        i_gp_reg_addr,
                                                        i_gp_reg_addr,
                                                        i_vec_length * l_datatype_size);
      }
      /* combine the m-jump with the n one */
      else {
        l_jump_block_m_last = (long long)i_vec_length * l_datatype_size;
      }
    }

    if ( l_m_blocks[1] != 0 ) {
      libxsmm_rv64_instruction_rvv_move( io_generated_code,
                                            l_masked_store_instr,
                                            i_gp_reg_addr,
                                            0,
                                            l_vec_reg_acc_start + l_m_total_blocks * l_n + l_m_blocks[0],
                                            1 );
    }

    l_jump_block_m_last += (long long)i_ld - l_m_bytes_full;

    if ( l_n != i_n_blocking - 1 ) {
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code,
                                                     LIBXSMM_RV64_INSTR_GP_ADD,
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
  l_jump_block_n_last = (long long)i_ld * i_n_blocking - l_jump_block_n_last;
  libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code,
                                                  LIBXSMM_RV64_INSTR_GP_SUB,
                                                  i_gp_reg_addr,
                                                  i_gp_reg_scratch,
                                                  i_gp_reg_addr,
                                                  l_jump_block_n_last );
}
