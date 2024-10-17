/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include "generator_matequation_aarch64.h"
#include "generator_aarch64_instructions.h"
#include "generator_mateltwise_aarch64.h"
#include "generator_common_aarch64.h"
#include "generator_common.h"
#include "generator_mateltwise_unary_binary_aarch64.h"
#include "generator_matequation_regblocks_avx_avx512.h"
#include "generator_matequation_regblocks_aarch64.h"
#include "generator_matequation_avx_avx512.h"


LIBXSMM_API_INTERN
void libxsmm_generator_copy_opargs_aarch64(libxsmm_generated_code*        io_generated_code,
    libxsmm_matequation_gp_reg_mapping  *i_gp_reg_mapping,
    libxsmm_matequation_kernel_config   *i_micro_kernel_config,
    libxsmm_meqn_elem             *cur_node,
    unsigned int                        *oparg_id,
    libxsmm_meqn_tmp_info         *oparg_info,
    unsigned int                        input_reg) {

  unsigned int temp_reg = i_gp_reg_mapping->temp_reg;
  unsigned int temp_reg2 = i_gp_reg_mapping->temp_reg2;
  unsigned int temp_reg3 = i_gp_reg_mapping->temp_reg3;
  unsigned int n_args = i_micro_kernel_config->n_args;
  if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
    /* Do nothing */
  } else if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) {
    if (cur_node->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_DUMP) {
      unsigned int cur_pos = *oparg_id;
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, input_reg, temp_reg, temp_reg2, (long long)cur_node->info.u_op.op_arg_pos*32 );
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, temp_reg2, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, temp_reg );
      libxsmm_generator_meqn_getaddr_stack_tmpaddr_i_aarch64( io_generated_code, n_args * 8 + cur_pos * 32, temp_reg3, temp_reg2);
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF, temp_reg2, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, temp_reg);
      oparg_info[cur_pos] = cur_node->tmp;
      oparg_info[cur_pos].id = cur_node->info.u_op.op_arg_pos;
      *oparg_id = cur_pos + 1;
    }
    libxsmm_generator_copy_opargs_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->le, oparg_id, oparg_info, input_reg);
  } else if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) {
    libxsmm_generator_copy_opargs_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->le, oparg_id, oparg_info, input_reg);
    libxsmm_generator_copy_opargs_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->ri, oparg_id, oparg_info, input_reg);
  } else if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) {
    libxsmm_generator_copy_opargs_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->le, oparg_id, oparg_info, input_reg);
    libxsmm_generator_copy_opargs_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->ri, oparg_id, oparg_info, input_reg);
    libxsmm_generator_copy_opargs_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->r2, oparg_id, oparg_info, input_reg);
  } else {
    /* This should not happen */
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_copy_input_args_aarch64(libxsmm_generated_code*        io_generated_code,
    libxsmm_matequation_gp_reg_mapping  *i_gp_reg_mapping,
    libxsmm_matequation_kernel_config   *i_micro_kernel_config,
    libxsmm_meqn_elem             *cur_node,
    unsigned int                        *arg_id,
    libxsmm_meqn_arg           *arg_info,
    unsigned int                        input_reg) {

  unsigned int temp_reg = i_gp_reg_mapping->temp_reg;
  unsigned int temp_reg2 = i_gp_reg_mapping->temp_reg2;
  unsigned int temp_reg3 = i_gp_reg_mapping->temp_reg3;
  if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
    unsigned int cur_pos = *arg_id;
    if (cur_node->info.arg.in_pos >= 0){
      if (cur_pos < i_micro_kernel_config->n_avail_gpr) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, input_reg, temp_reg, temp_reg2, (long long)cur_node->info.arg.in_pos*32 );
        libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, temp_reg2, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_micro_kernel_config->gpr_pool[cur_pos] );
      } else {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, input_reg, temp_reg, temp_reg2, (long long)cur_node->info.arg.in_pos*32 );
        libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, temp_reg2, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, temp_reg );
        libxsmm_generator_meqn_getaddr_stack_tmpaddr_i_aarch64( io_generated_code, cur_pos * 8, temp_reg3, temp_reg2);
        libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF, temp_reg2, LIBXSMM_AARCH64_GP_REG_XZR, 0, temp_reg);
      }
    } else {
      libxsmm_generator_meqn_getaddr_stack_tmp_i_aarch64( io_generated_code, (-cur_node->info.arg.in_pos-1) * i_micro_kernel_config->tmp_size, temp_reg3, temp_reg);
      if (cur_pos < i_micro_kernel_config->n_avail_gpr) {
        libxsmm_generator_mov_aarch64( io_generated_code, temp_reg, i_micro_kernel_config->gpr_pool[cur_pos] );
      } else {
        libxsmm_generator_meqn_getaddr_stack_tmpaddr_i_aarch64( io_generated_code, cur_pos * 8, temp_reg3, temp_reg2);
        libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF, temp_reg2, LIBXSMM_AARCH64_GP_REG_XZR, 0, temp_reg);
      }
    }
    arg_info[cur_pos] = cur_node->info.arg;
    *arg_id = cur_pos + 1;
  } else if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) {
    libxsmm_generator_copy_input_args_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->le, arg_id, arg_info, input_reg);
  } else if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) {
    i_micro_kernel_config->contains_binary_op = 1;
    if (cur_node->le->reg_score >= cur_node->ri->reg_score) {
      libxsmm_generator_copy_input_args_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->le, arg_id, arg_info, input_reg);
      libxsmm_generator_copy_input_args_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->ri, arg_id, arg_info, input_reg);
    } else {
      libxsmm_generator_copy_input_args_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->ri, arg_id, arg_info, input_reg);
      libxsmm_generator_copy_input_args_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->le, arg_id, arg_info, input_reg);
    }
  } else if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) {
    i_micro_kernel_config->contains_ternary_op = 1;
    if ((cur_node->le->reg_score >= cur_node->ri->reg_score) && (cur_node->le->reg_score >= cur_node->r2->reg_score)) {
      libxsmm_generator_copy_input_args_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->le, arg_id, arg_info, input_reg);
      if (cur_node->ri->reg_score >= cur_node->r2->reg_score) {
        libxsmm_generator_copy_input_args_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->ri, arg_id, arg_info, input_reg);
        libxsmm_generator_copy_input_args_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->r2, arg_id, arg_info, input_reg);
      } else {
        libxsmm_generator_copy_input_args_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->r2, arg_id, arg_info, input_reg);
        libxsmm_generator_copy_input_args_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->ri, arg_id, arg_info, input_reg);
      }
    } else if ((cur_node->ri->reg_score >= cur_node->le->reg_score) && (cur_node->ri->reg_score >= cur_node->r2->reg_score)) {
      libxsmm_generator_copy_input_args_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->ri, arg_id, arg_info, input_reg);
      if (cur_node->le->reg_score >= cur_node->r2->reg_score) {
        libxsmm_generator_copy_input_args_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->le, arg_id, arg_info, input_reg);
        libxsmm_generator_copy_input_args_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->r2, arg_id, arg_info, input_reg);
      } else {
        libxsmm_generator_copy_input_args_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->r2, arg_id, arg_info, input_reg);
        libxsmm_generator_copy_input_args_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->le, arg_id, arg_info, input_reg);
      }
    } else {
      libxsmm_generator_copy_input_args_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->r2, arg_id, arg_info, input_reg);
      if (cur_node->le->reg_score >= cur_node->ri->reg_score) {
        libxsmm_generator_copy_input_args_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->le, arg_id, arg_info, input_reg);
        libxsmm_generator_copy_input_args_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->ri, arg_id, arg_info, input_reg);
      } else {
        libxsmm_generator_copy_input_args_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->ri, arg_id, arg_info, input_reg);
        libxsmm_generator_copy_input_args_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->le, arg_id, arg_info, input_reg);
      }
    }
  } else {
    /* This should not happen */
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_adjust_opargs_addr_aarch64(libxsmm_generated_code*        io_generated_code,
    const libxsmm_meqn_descriptor       *i_mateqn_desc,
    libxsmm_matequation_gp_reg_mapping  *i_gp_reg_mapping,
    libxsmm_matequation_kernel_config   *i_micro_kernel_config,
    unsigned int                        i_adjust_instr,
    unsigned int                        i_adjust_amount,
    unsigned int                        i_adjust_type,
    libxsmm_meqn_tmp_info         *oparg_info) {
  unsigned int n_args = i_micro_kernel_config->n_args;
  unsigned int n_opargs = i_micro_kernel_config->n_opargs;
  unsigned int i;
  unsigned int adjust_val = 0;
  unsigned int temp_reg = i_gp_reg_mapping->temp_reg;
  unsigned int temp_reg2 = i_gp_reg_mapping->temp_reg2;
  unsigned int temp_reg3 = i_gp_reg_mapping->temp_reg3;
  LIBXSMM_UNUSED(i_mateqn_desc);

  /* Adjust input args */
  for (i = 0; i < n_opargs; i++) {
    unsigned int tsize = LIBXSMM_TYPESIZE(oparg_info[i].dtype);
    if (i_adjust_type == M_ADJUSTMENT) {
      adjust_val = i_adjust_amount * tsize;
    } else if (i_adjust_type == N_ADJUSTMENT) {
      adjust_val = i_adjust_amount * oparg_info[i].ld * tsize;
    }
    if (adjust_val != 0) {
      libxsmm_generator_meqn_getaddr_stack_tmpaddr_i_aarch64( io_generated_code, n_args * 8 + i * 32, temp_reg3, temp_reg);
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, temp_reg, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, temp_reg2 );
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, temp_reg2, temp_reg3, temp_reg2, adjust_val );
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF, temp_reg, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, temp_reg2);
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_adjust_args_addr_aarch64(libxsmm_generated_code*        io_generated_code,
    const libxsmm_meqn_descriptor       *i_mateqn_desc,
    libxsmm_matequation_gp_reg_mapping  *i_gp_reg_mapping,
    libxsmm_matequation_kernel_config   *i_micro_kernel_config,
    unsigned int                        i_adjust_instr,
    unsigned int                        i_adjust_amount,
    unsigned int                        i_adjust_type,
    libxsmm_meqn_arg           *arg_info) {

  unsigned int n_args = i_micro_kernel_config->n_args;
  unsigned int i;
  unsigned int adjust_val = 0;
  unsigned int output_tsize = LIBXSMM_TYPESIZE(LIBXSMM_GETENUM_OUT(i_mateqn_desc->datatype));
  unsigned int temp_reg = i_gp_reg_mapping->temp_reg;
  unsigned int temp_reg2 = i_gp_reg_mapping->temp_reg2;
  unsigned int temp_reg3 = i_gp_reg_mapping->temp_reg3;

  /* Adjust input args */
  for (i = 0; i < n_args; i++) {
    unsigned int tsize = LIBXSMM_TYPESIZE(arg_info[i].dtype);
    if (i_adjust_type == M_ADJUSTMENT) {
      if (arg_info[i].bcast_type == LIBXSMM_MATRIX_EQN_BCAST_TYPE_NONE || arg_info[i].bcast_type == LIBXSMM_MATRIX_EQN_BCAST_TYPE_COL) {
        adjust_val = i_adjust_amount * tsize;
      } else {
        adjust_val = 0;
      }
    } else if (i_adjust_type == N_ADJUSTMENT) {
      if (arg_info[i].bcast_type == LIBXSMM_MATRIX_EQN_BCAST_TYPE_NONE) {
        adjust_val = i_adjust_amount * arg_info[i].ld * tsize;
      } else if (arg_info[i].bcast_type == LIBXSMM_MATRIX_EQN_BCAST_TYPE_ROW) {
        adjust_val = i_adjust_amount * arg_info[i].ld * tsize;
      } else {
        adjust_val = 0;
      }
    }
    if (adjust_val != 0) {
      if ( i < i_micro_kernel_config->n_avail_gpr ) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_micro_kernel_config->gpr_pool[i], temp_reg, i_micro_kernel_config->gpr_pool[i], adjust_val );
      } else {
        libxsmm_generator_meqn_getaddr_stack_tmpaddr_i_aarch64( io_generated_code, i * 8, temp_reg3, temp_reg);
        libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, temp_reg, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, temp_reg2 );
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, temp_reg2, temp_reg3, temp_reg2, adjust_val );
        libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF, temp_reg, LIBXSMM_AARCH64_GP_REG_XZR, 0, temp_reg2);
      }
    }
  }

  /* Adjust output */
  if (i_micro_kernel_config->is_head_reduce_to_scalar == 0) {
    if (i_adjust_type == M_ADJUSTMENT) {
      adjust_val = i_adjust_amount * output_tsize;
    } else if (i_adjust_type == N_ADJUSTMENT) {
      adjust_val = i_adjust_amount * i_mateqn_desc->ldo * output_tsize;
    }
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg_mapping->gp_reg_out, temp_reg3, i_gp_reg_mapping->gp_reg_out, adjust_val );
  }
}

LIBXSMM_API_INTERN
void libxsmm_configure_mateqn_microkernel_loops_aarch64( libxsmm_generated_code*                io_generated_code,
                                                 libxsmm_matequation_kernel_config       *i_micro_kernel_config,
                                                 libxsmm_matrix_eqn                      *i_eqn,
                                                 unsigned int                            i_m,
                                                 unsigned int                            i_n,
                                                 unsigned int                            i_use_m_input_masking,
                                                 unsigned int*                           i_m_trips,
                                                 unsigned int*                           i_n_trips,
                                                 unsigned int*                           i_m_unroll_factor,
                                                 unsigned int*                           i_n_unroll_factor,
                                                 unsigned int*                           i_m_assm_trips,
                                                 unsigned int*                           i_n_assm_trips) {
  unsigned int m_trips, n_trips, m_unroll_factor = 0, n_unroll_factor = 0, m_assm_trips = 1, n_assm_trips = 1;
  unsigned int max_nm_unrolling = 32;
  unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;
  unsigned int i_vlen_in = i_micro_kernel_config->vlen_in;
  unsigned int n_tmp_reg_blocks = i_eqn->eqn_root->reg_score;

  LIBXSMM_UNUSED( io_generated_code );

  if (i_micro_kernel_config->contains_binary_op > 0) {
    n_tmp_reg_blocks++;
    if (i_micro_kernel_config->contains_ternary_op > 0) {
      n_tmp_reg_blocks++;
    }
  } else if (i_micro_kernel_config->contains_ternary_op > 0) {
    n_tmp_reg_blocks += 2;
  }

  i_micro_kernel_config->n_tmp_reg_blocks = n_tmp_reg_blocks;

  m_trips               = (i_m + i_vlen_in - 1) / i_vlen_in;
  n_trips               = i_n;

  max_nm_unrolling = max_nm_unrolling - reserved_zmms;
  max_nm_unrolling = max_nm_unrolling / n_tmp_reg_blocks;

  if (max_nm_unrolling < 1) {
    printf("Cannot generate run this code variant, ran out of zmm registers...\n");
  }
  if ((max_nm_unrolling < m_trips) && (i_use_m_input_masking == 1)) {
    printf("Cannot generate run this code variant, ran out of zmm registers and we want to mask M...\n");
  }

  if (i_use_m_input_masking == 1) {
    m_unroll_factor = m_trips;
  } else {
    m_unroll_factor = LIBXSMM_MIN(m_trips,16);
  }

  if (m_unroll_factor > max_nm_unrolling) {
    m_unroll_factor = max_nm_unrolling;
  }

  if (m_unroll_factor > 0) {
    while (m_trips % m_unroll_factor != 0) {
      m_unroll_factor--;
    }
  }

  n_unroll_factor = n_trips;
  while (m_unroll_factor * n_unroll_factor > max_nm_unrolling) {
    n_unroll_factor--;
  }

  if (n_unroll_factor > 0) {
    while (n_trips % n_unroll_factor != 0) {
      n_unroll_factor--;
    }
  }

  if (m_unroll_factor < 1) {
    m_unroll_factor = 1;
  }

  if (n_unroll_factor < 1) {
    n_unroll_factor = 1;
  }

  m_assm_trips = m_trips/m_unroll_factor;
  n_assm_trips = n_trips/n_unroll_factor;

  *i_m_trips = m_trips;
  *i_n_trips = n_trips;
  *i_m_unroll_factor = m_unroll_factor;
  *i_n_unroll_factor = n_unroll_factor;
  *i_m_assm_trips = m_assm_trips;
  *i_n_assm_trips = n_assm_trips;
}

LIBXSMM_API_INTERN
void libxsmm_meqn_setup_input_output_masks_aarch64( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
                                                 libxsmm_matequation_kernel_config*      i_micro_kernel_config,
                                                 const libxsmm_meqn_descriptor*          i_meqn_desc,
                                                 unsigned int                            i_tmp_reg,
                                                 unsigned int                            i_m,
                                                 unsigned int*                           i_use_m_input_masking,
                                                 unsigned int*                           i_mask_reg_in,
                                                 unsigned int*                           i_use_m_output_masking,
                                                 unsigned int*                           i_mask_reg_out) {
  unsigned int mask_reg_in = 0, mask_reg_out = 0, use_m_input_masking, use_m_output_masking;
  unsigned int i_vlen_in = i_micro_kernel_config->vlen_in;
  unsigned int i_vlen_out = i_micro_kernel_config->vlen_out;
  unsigned int reserved_mask_regs = i_micro_kernel_config->reserved_mask_regs;
  unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  unsigned int mask_tsize = (LIBXSMM_TYPESIZE(LIBXSMM_GETENUM_OUT(i_meqn_desc->datatype)) == 8) ? 8 : 4;

  LIBXSMM_UNUSED(i_meqn_desc);
  LIBXSMM_UNUSED(i_tmp_reg);

  use_m_input_masking   = i_m % i_vlen_in;
  use_m_output_masking  = i_m % i_vlen_out;

  if ( l_is_sve ) {
    /* BF16 mask for full vload */
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, reserved_mask_regs, i_vlen_in * 2, i_gp_reg_mapping->gp_reg_scratch_0 );
    i_micro_kernel_config->full_vlen_bf16_mask = reserved_mask_regs;
    reserved_mask_regs++;
    if (use_m_input_masking) {
      /* F32 mask */
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, reserved_mask_regs, (i_m % i_vlen_in) * mask_tsize, i_gp_reg_mapping->gp_reg_scratch_0 );
      i_micro_kernel_config->in_f32_mask = reserved_mask_regs;
      /* BF16 mask */
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, reserved_mask_regs + 1, (i_m % i_vlen_in) * 2, i_gp_reg_mapping->gp_reg_scratch_0 );
      i_micro_kernel_config->in_bf16_mask = reserved_mask_regs + 1;
      reserved_mask_regs += 2;
    }
    if (use_m_output_masking) {
      /* F32 mask */
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, reserved_mask_regs, (i_m % i_vlen_out) * mask_tsize, i_gp_reg_mapping->gp_reg_scratch_0 );
      i_micro_kernel_config->out_f32_mask = reserved_mask_regs;
      /* BF16 mask */
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, reserved_mask_regs + 1, (i_m % i_vlen_out) * 2, i_gp_reg_mapping->gp_reg_scratch_0 );
      i_micro_kernel_config->out_bf16_mask = reserved_mask_regs + 1;
      reserved_mask_regs += 2;
    }
  } else {
    mask_reg_in = i_m % i_vlen_in;
    mask_reg_out = i_m % i_vlen_out;
  }

  i_micro_kernel_config->reserved_mask_regs = reserved_mask_regs;
  *i_mask_reg_in = mask_reg_in;
  *i_use_m_input_masking = use_m_input_masking;
  *i_mask_reg_out = mask_reg_out;
  *i_use_m_output_masking = use_m_output_masking;
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_load_arg_to_2d_reg_block_aarch64( libxsmm_generated_code*          io_generated_code,
                                                 libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
                                                 libxsmm_matequation_kernel_config*      i_micro_kernel_config,
                                                 const libxsmm_meqn_descriptor*          i_meqn_desc,
                                                 unsigned int                            i_arg_id,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_reg_block_id,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg,
                                                 unsigned int                            i_skip_dtype_cvt ) {

  unsigned int in, im;
  unsigned int temp_reg = i_gp_reg_mapping->temp_reg;
  unsigned int temp_reg2 = i_gp_reg_mapping->temp_reg2;
  unsigned int temp_reg3 = i_gp_reg_mapping->temp_reg3;
  unsigned int cur_vreg;
  libxsmm_meqn_arg  *arg_info = i_micro_kernel_config->arg_info;
  unsigned int i_start_vreg = libxsmm_generator_matequation_regblocks_get_start_of_register_block(i_micro_kernel_config, i_reg_block_id);
  unsigned int input_reg = 0;
  unsigned int l_ld_bytes = arg_info[i_arg_id].ld * LIBXSMM_TYPESIZE(arg_info[i_arg_id].dtype);
  unsigned int l_m_adjust = ( i_mask_last_m_chunk == 0 ) ? LIBXSMM_TYPESIZE(arg_info[i_arg_id].dtype) * i_vlen * i_m_blocking : LIBXSMM_TYPESIZE(arg_info[i_arg_id].dtype) * ( (i_vlen * (i_m_blocking-1)) + i_mask_last_m_chunk );
  unsigned int offset = 0;
  unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  libxsmm_aarch64_sve_type l_sve_type = libxsmm_generator_aarch64_get_sve_type(4);
  LIBXSMM_UNUSED(i_meqn_desc);
  LIBXSMM_UNUSED(i_mask_reg);

  if (i_arg_id < i_micro_kernel_config->n_avail_gpr) {
    input_reg = i_micro_kernel_config->gpr_pool[i_arg_id];
  } else {
    libxsmm_generator_meqn_getaddr_stack_tmpaddr_i_aarch64( io_generated_code, i_arg_id * 8, temp_reg3, temp_reg2);
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, temp_reg2, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, temp_reg );
    input_reg = temp_reg;
  }

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      unsigned int l_is_f32_or_f64 = (LIBXSMM_DATATYPE_F32 == arg_info[i_arg_id].dtype || LIBXSMM_DATATYPE_F64 == arg_info[i_arg_id].dtype) ? 1 : 0;
      unsigned int l_masked_elements = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0
                                                         : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_mask_last_m_chunk : i_vlen : i_vlen;
      unsigned int l_mask_load = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_micro_kernel_config->in_f32_mask : 0 : 0
                                                   : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_micro_kernel_config->in_bf16_mask : i_micro_kernel_config->full_vlen_bf16_mask : i_micro_kernel_config->full_vlen_bf16_mask;
      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      if (arg_info[i_arg_id].bcast_type == LIBXSMM_MATRIX_EQN_BCAST_TYPE_NONE) {
        offset = (l_ld_bytes*i_n_blocking);
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, input_reg, i_gp_reg_mapping->gp_reg_scratch_0, cur_vreg, LIBXSMM_TYPESIZE(arg_info[i_arg_id].dtype), l_masked_elements, 1, 0, LIBXSMM_CAST_UCHAR(l_mask_load));
        if (i_skip_dtype_cvt == 0) {
          if ( arg_info[i_arg_id].dtype == LIBXSMM_DATATYPE_BF16 ) {
            libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, cur_vreg, 0);
          }
        }
      } else if ((arg_info[i_arg_id].bcast_type == LIBXSMM_MATRIX_EQN_BCAST_TYPE_ROW) || (arg_info[i_arg_id].bcast_type == LIBXSMM_MATRIX_EQN_BCAST_TYPE_SCALAR)) {
        offset = (arg_info[i_arg_id].bcast_type == LIBXSMM_MATRIX_EQN_BCAST_TYPE_SCALAR) ?  LIBXSMM_TYPESIZE(arg_info[i_arg_id].dtype) : l_ld_bytes * i_n_blocking;
        if (im == 0) {
          if ((arg_info[i_arg_id].bcast_type == LIBXSMM_MATRIX_EQN_BCAST_TYPE_ROW) || ((arg_info[i_arg_id].bcast_type == LIBXSMM_MATRIX_EQN_BCAST_TYPE_SCALAR) && (in == 0))) {
            libxsmm_generator_bcastload_masked_vreg_aarch64_asimd( io_generated_code, input_reg, i_gp_reg_mapping->gp_reg_scratch_0, cur_vreg, LIBXSMM_TYPESIZE(arg_info[i_arg_id].dtype), (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0, 1 );
            if (i_skip_dtype_cvt == 0) {
              if ( arg_info[i_arg_id].dtype == LIBXSMM_DATATYPE_BF16 ) {
                libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, cur_vreg, 0);
              }
            }
          } else if ((arg_info[i_arg_id].bcast_type == LIBXSMM_MATRIX_EQN_BCAST_TYPE_SCALAR) && (in > 0)) {
            if (l_is_sve) {
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V, i_start_vreg, i_start_vreg, 0, cur_vreg, 0, l_sve_type);
            } else {
              libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V, i_start_vreg, i_start_vreg, 0, cur_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S);
            }
          }
        } else {
          if (l_is_sve) {
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V, i_start_vreg + in * i_m_blocking, i_start_vreg + in * i_m_blocking, 0, cur_vreg, 0, l_sve_type);
          } else {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V, i_start_vreg + in * i_m_blocking, i_start_vreg + in * i_m_blocking, 0, cur_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
          }
        }
      } else if (arg_info[i_arg_id].bcast_type == LIBXSMM_MATRIX_EQN_BCAST_TYPE_COL) {
        offset = ( i_mask_last_m_chunk == 0 ) ? LIBXSMM_TYPESIZE(arg_info[i_arg_id].dtype) * i_vlen * i_m_blocking : LIBXSMM_TYPESIZE(arg_info[i_arg_id].dtype) * ( (i_vlen * (i_m_blocking-1)) + i_mask_last_m_chunk );
        if (in == 0) {
          libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, input_reg, i_gp_reg_mapping->gp_reg_scratch_0, cur_vreg, LIBXSMM_TYPESIZE(arg_info[i_arg_id].dtype), l_masked_elements, 1, 0, LIBXSMM_CAST_UCHAR(l_mask_load));
          if (i_skip_dtype_cvt == 0) {
            if ( arg_info[i_arg_id].dtype == LIBXSMM_DATATYPE_BF16 ) {
              libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, cur_vreg, 0);
            }
          }
        } else {
          if (l_is_sve) {
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V, i_start_vreg + im, i_start_vreg + im, 0, cur_vreg, 0, l_sve_type );
          } else {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V, i_start_vreg + im, i_start_vreg + im, 0, cur_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
          }
        }
      } else {
        /* Should not happen */
      }
    }
    if (arg_info[i_arg_id].bcast_type == LIBXSMM_MATRIX_EQN_BCAST_TYPE_NONE) {
      if ( l_ld_bytes != l_m_adjust ) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, input_reg, i_gp_reg_mapping->gp_reg_scratch_0, input_reg, ((long long)l_ld_bytes - l_m_adjust) );
      }
    } else {
      if (arg_info[i_arg_id].bcast_type == LIBXSMM_MATRIX_EQN_BCAST_TYPE_ROW) {
        if ( l_ld_bytes != (unsigned int)LIBXSMM_TYPESIZE(arg_info[i_arg_id].dtype) ) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, input_reg, i_gp_reg_mapping->gp_reg_scratch_0, input_reg, ((long long)l_ld_bytes - LIBXSMM_TYPESIZE(arg_info[i_arg_id].dtype)) );
        }
      }
    }
  }
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB, input_reg, i_gp_reg_mapping->gp_reg_scratch_0, input_reg, offset );
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_dump_2d_reg_block_aarch64( libxsmm_generated_code*          io_generated_code,
                                                 libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
                                                 libxsmm_matequation_kernel_config*      i_micro_kernel_config,
                                                 const libxsmm_meqn_descriptor*          i_meqn_desc,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_m,
                                                 unsigned int                            i_ld,
                                                 unsigned int                            i_reg_block_id,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg,
                                                 libxsmm_datatype                        i_regblock_dtype,
                                                 unsigned int                            i_gp_reg_out ) {
  unsigned int in, im;
  unsigned int cur_vreg;
  unsigned int i_start_vreg = libxsmm_generator_matequation_regblocks_get_start_of_register_block(i_micro_kernel_config, i_reg_block_id);
  unsigned int l_ld_bytes = i_ld * LIBXSMM_TYPESIZE(i_regblock_dtype);
  unsigned int l_m_adjust = ( i_mask_last_m_chunk == 0 ) ? LIBXSMM_TYPESIZE(i_regblock_dtype) * i_vlen * i_m_blocking : LIBXSMM_TYPESIZE(i_regblock_dtype) * ( (i_vlen * (i_m_blocking-1)) + i_mask_last_m_chunk );
  LIBXSMM_UNUSED(i_mask_reg);
  LIBXSMM_UNUSED(i_m);
  LIBXSMM_UNUSED(i_meqn_desc);

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      unsigned int l_is_f32_or_f64 = (LIBXSMM_DATATYPE_F32 == i_regblock_dtype || LIBXSMM_DATATYPE_F64 == i_regblock_dtype) ? 1 : 0;
      unsigned int l_masked_elements = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0
                                                         : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_mask_last_m_chunk : i_vlen : i_vlen;
      unsigned int l_mask_store = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_micro_kernel_config->out_f32_mask : 0 : 0
                                                    : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_micro_kernel_config->out_bf16_mask : i_micro_kernel_config->full_vlen_bf16_mask : i_micro_kernel_config->full_vlen_bf16_mask;
      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, cur_vreg, LIBXSMM_TYPESIZE(i_regblock_dtype), l_masked_elements, 1, 1, LIBXSMM_CAST_UCHAR(l_mask_store));
    }
    if ( l_ld_bytes != l_m_adjust ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, i_gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_out, ((long long)l_ld_bytes - l_m_adjust) );
    }
  }
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB, i_gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_out, (long long)l_ld_bytes*i_n_blocking );
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_store_2d_reg_block_aarch64( libxsmm_generated_code*          io_generated_code,
                                                 libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
                                                 libxsmm_matequation_kernel_config*      i_micro_kernel_config,
                                                 const libxsmm_meqn_descriptor*          i_meqn_desc,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_reg_block_id,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg ) {
  unsigned int in, im;
  unsigned int cur_vreg;
  unsigned int i_start_vreg = libxsmm_generator_matequation_regblocks_get_start_of_register_block(i_micro_kernel_config, i_reg_block_id);
  unsigned int l_ld_bytes = i_meqn_desc->ldo * LIBXSMM_TYPESIZE(LIBXSMM_GETENUM_OUT(i_meqn_desc->datatype));
  unsigned int l_m_adjust = ( i_mask_last_m_chunk == 0 ) ? LIBXSMM_TYPESIZE(LIBXSMM_GETENUM_OUT(i_meqn_desc->datatype)) * i_vlen * i_m_blocking : LIBXSMM_TYPESIZE(LIBXSMM_GETENUM_OUT(i_meqn_desc->datatype)) * ( (i_vlen * (i_m_blocking-1)) + i_mask_last_m_chunk );
  LIBXSMM_UNUSED(i_mask_reg);

  if (i_micro_kernel_config->is_head_reduce_to_scalar > 0) return;

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      unsigned int l_is_f32_or_f64 = (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT(i_meqn_desc->datatype) || LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_OUT(i_meqn_desc->datatype)) ? 1 : 0;
      unsigned int l_masked_elements = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0
                                                         : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_mask_last_m_chunk : i_vlen : i_vlen;
      unsigned int l_mask_store = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_micro_kernel_config->out_f32_mask : 0 : 0
                                                    : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_micro_kernel_config->out_bf16_mask : i_micro_kernel_config->full_vlen_bf16_mask : i_micro_kernel_config->full_vlen_bf16_mask;
      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      if (i_micro_kernel_config->cvt_result_to_bf16 == 1) {
        libxsmm_generator_vcvt_f32bf16_aarch64( io_generated_code, cur_vreg, 0);
      }
      libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, cur_vreg, LIBXSMM_TYPESIZE(LIBXSMM_GETENUM_OUT(i_meqn_desc->datatype)), l_masked_elements, 1, 1, LIBXSMM_CAST_UCHAR(l_mask_store));
    }
    if ( l_ld_bytes != l_m_adjust ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out, ((long long)l_ld_bytes - l_m_adjust) );
    }
  }
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB, i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out, (long long)l_ld_bytes*i_n_blocking );
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_unpackstore_2d_reg_block_aarch64( libxsmm_generated_code*          io_generated_code,
                                                 libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
                                                 libxsmm_matequation_kernel_config*      i_micro_kernel_config,
                                                 const libxsmm_meqn_descriptor*          i_meqn_desc,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_reg_block_id,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg ) {
  unsigned int in, im;
  unsigned int cur_vreg;
  unsigned int i_start_vreg = libxsmm_generator_matequation_regblocks_get_start_of_register_block(i_micro_kernel_config, i_reg_block_id);
  unsigned int l_ld_bytes = i_meqn_desc->ldo * LIBXSMM_TYPESIZE(LIBXSMM_GETENUM_OUT(i_meqn_desc->datatype));
  unsigned int l_m_adjust = ( i_mask_last_m_chunk == 0 ) ? LIBXSMM_TYPESIZE(LIBXSMM_GETENUM_OUT(i_meqn_desc->datatype)) * i_vlen * i_m_blocking : LIBXSMM_TYPESIZE(LIBXSMM_GETENUM_OUT(i_meqn_desc->datatype)) * ( (i_vlen * (i_m_blocking-1)) + i_mask_last_m_chunk );
  LIBXSMM_UNUSED( i_mask_reg );

  if (i_micro_kernel_config->is_head_reduce_to_scalar > 0) return;

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      unsigned int l_masked_elements = (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT(i_meqn_desc->datatype)) ? (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0
                                                                                                                                       : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_mask_last_m_chunk : i_vlen : i_vlen;
      unsigned int l_mask_store = (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT(i_meqn_desc->datatype)) ? (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_micro_kernel_config->out_f32_mask : 0 : 0
                                                                                                                                  : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_micro_kernel_config->out_bf16_mask : i_micro_kernel_config->full_vlen_bf16_mask : i_micro_kernel_config->full_vlen_bf16_mask;
      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UZP1_V, cur_vreg, cur_vreg, 0, i_micro_kernel_config->tmp_vreg, 0, libxsmm_generator_aarch64_get_sve_type(2) );
      libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_1, i_micro_kernel_config->tmp_vreg, LIBXSMM_TYPESIZE(LIBXSMM_GETENUM_OUT(i_meqn_desc->datatype)), l_masked_elements, 1, 1, LIBXSMM_CAST_UCHAR(l_mask_store));
    }
    if ( l_ld_bytes != l_m_adjust ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out,
                                                     ((long long)l_ld_bytes - l_m_adjust) );
    }
  }
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out,
                                                 (long long)l_ld_bytes*i_n_blocking );

  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                         i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_offset, i_gp_reg_mapping->gp_reg_out,
                                                         0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      unsigned int l_masked_elements = (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT(i_meqn_desc->datatype)) ? (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0
                                                                                                                                       : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_mask_last_m_chunk : i_vlen : i_vlen;
      unsigned int l_mask_store = (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT(i_meqn_desc->datatype)) ? (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_micro_kernel_config->out_f32_mask : 0 : 0
                                                                                                                                  : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_micro_kernel_config->out_bf16_mask : i_micro_kernel_config->full_vlen_bf16_mask : i_micro_kernel_config->full_vlen_bf16_mask;
      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UZP2_V, cur_vreg, cur_vreg, 0, i_micro_kernel_config->tmp_vreg, 0, libxsmm_generator_aarch64_get_sve_type(2) );
      libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_1, i_micro_kernel_config->tmp_vreg,
                                                            LIBXSMM_TYPESIZE(LIBXSMM_GETENUM_OUT(i_meqn_desc->datatype)), l_masked_elements, 1, 1, LIBXSMM_CAST_UCHAR(l_mask_store));
    }
    if ( l_ld_bytes != l_m_adjust ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out,
                                                     ((long long)l_ld_bytes - l_m_adjust) );
    }
  }
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out,
                                                 (long long)l_ld_bytes*i_n_blocking );

  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_SR,
                                                         i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_offset, i_gp_reg_mapping->gp_reg_out,
                                                         0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_store_reduce_to_scalar_output_aarch64( libxsmm_generated_code*          io_generated_code,
                                                 libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
                                                 libxsmm_matequation_kernel_config*      i_micro_kernel_config,
                                                 const libxsmm_meqn_descriptor*          i_meqn_desc ) {
  libxsmm_aarch64_sve_type sve_type = libxsmm_generator_aarch64_get_sve_type(4);
  unsigned char is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  unsigned int hreduce_instr = is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FADDV_V_P : LIBXSMM_AARCH64_INSTR_ASIMD_FADDP_V;

  if ( is_sve ) {
    libxsmm_aarch64_instruction_sve_compute(io_generated_code, hreduce_instr, i_micro_kernel_config->reduce_vreg, i_micro_kernel_config->reduce_vreg, 0, i_micro_kernel_config->reduce_vreg, 0, sve_type );
  } else {
    libxsmm_generator_hinstrps_aarch64(io_generated_code, hreduce_instr, i_micro_kernel_config->reduce_vreg, (LIBXSMM_TYPESIZE(LIBXSMM_GETENUM_OUT(i_meqn_desc->datatype)) == 8) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S);
  }

  if (i_micro_kernel_config->cvt_result_to_bf16 == 1) {
    libxsmm_generator_vcvt_f32bf16_aarch64( io_generated_code, i_micro_kernel_config->reduce_vreg, 0);
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST, i_gp_reg_mapping->gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF,
        LIBXSMM_TYPESIZE(LIBXSMM_GETENUM_OUT(i_meqn_desc->datatype)), i_micro_kernel_config->reduce_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_H );
  } else {
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST, i_gp_reg_mapping->gp_reg_out, LIBXSMM_AARCH64_GP_REG_UNDEF,
        LIBXSMM_TYPESIZE(LIBXSMM_GETENUM_OUT(i_meqn_desc->datatype)), i_micro_kernel_config->reduce_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_S );
  }
  return ;
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_compute_unary_op_2d_reg_block_aarch64( libxsmm_generated_code*     io_generated_code,
                                                 libxsmm_matequation_kernel_config*      i_meqn_micro_kernel_config,
                                                 libxsmm_meltw_unary_type                i_op_type,
                                                 unsigned int                            i_reg_block_id,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 libxsmm_datatype                        i_dtype ) {

  unsigned int i_start_vreg = libxsmm_generator_matequation_regblocks_get_start_of_register_block(i_meqn_micro_kernel_config, i_reg_block_id);
  unsigned int im, in, cur_vreg;
  libxsmm_mateltwise_kernel_config *i_micro_kernel_config = &(i_meqn_micro_kernel_config->meltw_kernel_config);
  unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  libxsmm_aarch64_sve_type l_sve_type = libxsmm_generator_aarch64_get_sve_type((i_dtype == LIBXSMM_DATATYPE_F64) ? 8 : 4);
  libxsmm_aarch64_asimd_tupletype l_tupletype = (i_dtype == LIBXSMM_DATATYPE_F64) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S;
  unsigned int l_pred_reg = 0;

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_X2) {
        if ( l_is_sve ) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V, cur_vreg, cur_vreg, 0, cur_vreg, 0, l_sve_type );
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V, cur_vreg, cur_vreg, 0, cur_vreg, l_tupletype );
        }
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_TO_SCALAR_OP_ADD) {
        unsigned int reduce_instr = l_is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FADD_V : LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V;
        if ( l_is_sve ) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, reduce_instr, cur_vreg, i_meqn_micro_kernel_config->reduce_vreg, 0, i_meqn_micro_kernel_config->reduce_vreg, 0, l_sve_type );
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, reduce_instr, cur_vreg, i_meqn_micro_kernel_config->reduce_vreg, 0, i_meqn_micro_kernel_config->reduce_vreg, l_tupletype );
        }
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_NEGATE) {
        if ( l_is_sve ) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FNEG_V_P, cur_vreg, cur_vreg, 0, cur_vreg, l_pred_reg, l_sve_type );
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FNEG_V, cur_vreg, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, cur_vreg, l_tupletype );
        }
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_INC) {
        if ( l_is_sve ) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FADD_V, cur_vreg, i_micro_kernel_config->vec_ones, 0, cur_vreg, l_pred_reg, l_sve_type );
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V, cur_vreg, i_micro_kernel_config->vec_ones, 0, cur_vreg, l_tupletype );
        }
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL) {
        if ( l_is_sve ) {
          if (libxsmm_get_ulp_precision() != LIBXSMM_ULP_PRECISION_ESTIMATE){
            unsigned char tmp_vreg = LIBXSMM_CAST_UCHAR(i_micro_kernel_config->tmp_vreg);
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FRECPE_V, /* save the estimate in tmp */
                                                     cur_vreg, cur_vreg, 0, tmp_vreg, l_pred_reg, l_sve_type );
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FRECPS_V, /* compute the improvement by tmp,cur into cur */
                                                     cur_vreg, tmp_vreg, 0, cur_vreg, l_pred_reg, l_sve_type);
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V, /* apply the improvement on tmp, and write result into cur */
                                                     cur_vreg, tmp_vreg, 0, cur_vreg, l_pred_reg, l_sve_type);
          } else {/* if we do not really care about precision, we can skip the extra iteration */
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FRECPE_V,
                                                     cur_vreg, cur_vreg, 0, cur_vreg, l_pred_reg, l_sve_type );
          }
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FRECPE_V,
                                                     cur_vreg, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, i_micro_kernel_config->vec_tmp0,
                                                     l_tupletype );
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FRECPS_V,
                                                     cur_vreg, i_micro_kernel_config->vec_tmp0, 0, cur_vreg,
                                                     l_tupletype );
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                                     cur_vreg, i_micro_kernel_config->vec_tmp0, 0, cur_vreg,
                                                     l_tupletype );
        }
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT) {
        const int niterations = (1 << (int)l_sve_type) - 1;
        unsigned char max_num_iterations = (unsigned char)niterations;
        unsigned char num_iterations = (libxsmm_get_ulp_precision() == LIBXSMM_ULP_PRECISION_ESTIMATE ? 0 : max_num_iterations);
        if ( l_is_sve ) {
          unsigned char tmp_guess = LIBXSMM_CAST_UCHAR(i_micro_kernel_config->tmp_vreg);
          unsigned char tmp_guess_squared = LIBXSMM_CAST_UCHAR(i_micro_kernel_config->tmp_vreg2);
          unsigned char i;
          /* coverity[dead_error_line] */
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FRSQRTE_V,
                                                   cur_vreg, cur_vreg, 0, num_iterations > 0 ? tmp_guess : cur_vreg, l_pred_reg, l_sve_type);
          /* Newton iteration: guess *= (3-guess*guess*x)/2 */
          for ( i=0; i<num_iterations; i++){
            unsigned char dst_reg = LIBXSMM_CAST_UCHAR(i == (num_iterations-1) ? cur_vreg : tmp_guess); /* improve the guess; then save it */
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                                     tmp_guess, tmp_guess, 0, tmp_guess_squared, l_pred_reg, l_sve_type);
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FRSQRTS_V, /* dst = (3-s0*s1)/2 */
                                                     cur_vreg, tmp_guess_squared, 0, tmp_guess_squared, l_pred_reg, l_sve_type);
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                                     tmp_guess, tmp_guess_squared, 0, dst_reg, l_pred_reg, l_sve_type);
          }
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FRSQRTE_V,
                                                     cur_vreg, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, cur_vreg,
                                                     l_tupletype );
        }
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_SQRT) {
        if ( l_is_sve ) {
          /* the SQRT instruction is very slow on A64FX, only as fast as ASIMD, so maybe even serial performance */
          /* LIBXSMM is a machine learning oriented library and instructions like 1/x are inexact, so let's make this inexact as well */
          if (libxsmm_get_ulp_precision() != LIBXSMM_ULP_PRECISION_ESTIMATE){
            /* old & slow way */
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FSQRT_V_P,
                                                     cur_vreg, cur_vreg, 0, cur_vreg, l_pred_reg, l_sve_type);
          } else {
            /* inverse */
            unsigned char tmp_vreg = LIBXSMM_CAST_UCHAR(i_micro_kernel_config->tmp_vreg);
            /* then 1/sqrt */
            /* fp32, num_iterations=0 -> 0.07    relative error, 27.0x speedup */
            /* fp32, num_iterations=1 -> 0.0002  relative error, 16.3x speedup */
            /* fp32, num_iterations=2 -> 0.00007 relative error,  9.6x speedup */
            unsigned char num_iterations = 1;
            unsigned char tmp_guess = LIBXSMM_CAST_UCHAR(i_micro_kernel_config->tmp_vreg);
            unsigned char tmp_guess_squared = LIBXSMM_CAST_UCHAR(i_micro_kernel_config->tmp_vreg2);
            unsigned char i;

            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FRECPE_V, /* save the estimate in tmp */
                                                     cur_vreg, cur_vreg, 0, tmp_vreg, l_pred_reg, l_sve_type );
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FRECPS_V, /* compute the improvement by tmp,cur into cur */
                                                     cur_vreg, tmp_vreg, 0, cur_vreg, l_pred_reg, l_sve_type);
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V, /* apply the improvement on tmp, and write result into cur */
                                                     cur_vreg, tmp_vreg, 0, cur_vreg, l_pred_reg, l_sve_type);
            /* coverity[dead_error_line] */
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FRSQRTE_V,
                                                    cur_vreg, cur_vreg, 0, num_iterations > 0 ? tmp_guess : cur_vreg, l_pred_reg, l_sve_type);
            /* Newton iteration: guess *= (3-guess*guess*x)/2 */
            for (i=0;i<num_iterations;i++){
              unsigned char dst_reg = LIBXSMM_CAST_UCHAR(i == (num_iterations-1) ? cur_vreg : tmp_guess); /* improve the guess; then save it */
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                                       tmp_guess, tmp_guess, 0, tmp_guess_squared, l_pred_reg, l_sve_type);
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FRSQRTS_V, /* dst = (3-s0*s1)/2 */
                                                       cur_vreg, tmp_guess_squared, 0, tmp_guess_squared, l_pred_reg, l_sve_type);
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                                       tmp_guess, tmp_guess_squared, 0, dst_reg, l_pred_reg, l_sve_type);
            }
          }
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FSQRT_V,
                                                     cur_vreg, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, cur_vreg,
                                                     l_tupletype );
        }
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_EXP) {
        if (l_is_sve){
          libxsmm_generator_exp_ps_3dts_aarch64_sve(
            io_generated_code,
            cur_vreg,
            i_micro_kernel_config->vec_y,
            i_micro_kernel_config->vec_z,
            i_micro_kernel_config->vec_c0,
            i_micro_kernel_config->vec_c1,
            i_micro_kernel_config->vec_c2,
            i_micro_kernel_config->vec_c3,
            i_micro_kernel_config->vec_halves,
            i_micro_kernel_config->vec_log2e,
            i_micro_kernel_config->vec_expmask,
            i_micro_kernel_config->vec_hi_bound,
            i_micro_kernel_config->vec_lo_bound,
            l_sve_type, LIBXSMM_CAST_UCHAR(l_pred_reg) );
        } else {
          libxsmm_generator_exp_ps_3dts_aarch64_asimd(
            io_generated_code,
            cur_vreg,
            i_micro_kernel_config->vec_y,
            i_micro_kernel_config->vec_z,
            i_micro_kernel_config->vec_c0,
            i_micro_kernel_config->vec_c1,
            i_micro_kernel_config->vec_c2,
            i_micro_kernel_config->vec_c3,
            i_micro_kernel_config->vec_halves,
            i_micro_kernel_config->vec_log2e,
            i_micro_kernel_config->vec_expmask,
            i_micro_kernel_config->vec_hi_bound,
            i_micro_kernel_config->vec_lo_bound,
            l_tupletype );
        }
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_TANH || i_op_type == LIBXSMM_MELTW_TYPE_UNARY_TANH_INV ) {
        if (l_is_sve) {
          libxsmm_generator_tanh_ps_rational_78_aarch64_sve(
            io_generated_code,
            cur_vreg,
            i_micro_kernel_config->vec_x2,
            i_micro_kernel_config->vec_nom,
            i_micro_kernel_config->vec_denom,
            i_micro_kernel_config->mask_hi,
            i_micro_kernel_config->mask_lo,
            i_micro_kernel_config->vec_c0,
            i_micro_kernel_config->vec_c1,
            i_micro_kernel_config->vec_c2,
            i_micro_kernel_config->vec_c3,
            i_micro_kernel_config->vec_c1_d,
            i_micro_kernel_config->vec_c2_d,
            i_micro_kernel_config->vec_c3_d,
            i_micro_kernel_config->vec_hi_bound,
            i_micro_kernel_config->vec_lo_bound,
            i_micro_kernel_config->vec_ones,
            i_micro_kernel_config->vec_neg_ones,
            i_micro_kernel_config->vec_tmp0,
            l_sve_type, LIBXSMM_CAST_UCHAR(l_pred_reg) );
        } else {
          libxsmm_generator_tanh_ps_rational_78_aarch64_asimd(
            io_generated_code,
            cur_vreg,
            i_micro_kernel_config->vec_x2,
            i_micro_kernel_config->vec_nom,
            i_micro_kernel_config->vec_denom,
            i_micro_kernel_config->mask_hi,
            i_micro_kernel_config->mask_lo,
            i_micro_kernel_config->vec_c0,
            i_micro_kernel_config->vec_c1,
            i_micro_kernel_config->vec_c2,
            i_micro_kernel_config->vec_c3,
            i_micro_kernel_config->vec_c1_d,
            i_micro_kernel_config->vec_c2_d,
            i_micro_kernel_config->vec_c3_d,
            i_micro_kernel_config->vec_hi_bound,
            i_micro_kernel_config->vec_lo_bound,
            i_micro_kernel_config->vec_ones,
            i_micro_kernel_config->vec_neg_ones,
            i_micro_kernel_config->vec_tmp0,
            l_tupletype );
        }
        if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_TANH_INV) {/* 1st derivative of tanh(x) = 1-tanh(x)^2 */
          if (l_is_sve){
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                                     cur_vreg, cur_vreg, 0, i_micro_kernel_config->vec_tmp0,
                                                     l_pred_reg, l_sve_type );
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FNEG_V_P,
                                                     i_micro_kernel_config->vec_tmp0, LIBXSMM_AARCH64_SVE_REG_UNDEF, 0,  i_micro_kernel_config->vec_tmp0,
                                                     l_pred_reg, l_sve_type );
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FSUB_V,
                                                     i_micro_kernel_config->vec_tmp0, i_micro_kernel_config->vec_neg_ones, 0, cur_vreg,
                                                     l_pred_reg, l_sve_type );
          } else {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                                       cur_vreg, cur_vreg, 0, i_micro_kernel_config->vec_tmp0,
                                                       l_tupletype );
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FNEG_V,
                                                       i_micro_kernel_config->vec_tmp0, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0,  i_micro_kernel_config->vec_tmp0,
                                                       l_tupletype );
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FSUB_V,
                                                       i_micro_kernel_config->vec_tmp0, i_micro_kernel_config->vec_neg_ones, 0, cur_vreg,
                                                       l_tupletype );
          }
        }
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID || i_op_type == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV) {
        if (l_is_sve){
          libxsmm_generator_sigmoid_ps_rational_78_aarch64_sve(
            io_generated_code,
            cur_vreg,
            i_micro_kernel_config->vec_x2,
            i_micro_kernel_config->vec_nom,
            i_micro_kernel_config->vec_denom,
            i_micro_kernel_config->mask_hi,
            i_micro_kernel_config->mask_lo,
            i_micro_kernel_config->vec_c0,
            i_micro_kernel_config->vec_c1,
            i_micro_kernel_config->vec_c2,
            i_micro_kernel_config->vec_c3,
            i_micro_kernel_config->vec_c1_d,
            i_micro_kernel_config->vec_c2_d,
            i_micro_kernel_config->vec_c3_d,
            i_micro_kernel_config->vec_hi_bound,
            i_micro_kernel_config->vec_lo_bound,
            i_micro_kernel_config->vec_ones,
            i_micro_kernel_config->vec_neg_ones,
            i_micro_kernel_config->vec_halves,
            i_micro_kernel_config->vec_tmp0,
            l_sve_type, LIBXSMM_CAST_UCHAR(l_pred_reg) );
        } else {
          libxsmm_generator_sigmoid_ps_rational_78_aarch64_asimd(
            io_generated_code,
            cur_vreg,
            i_micro_kernel_config->vec_x2,
            i_micro_kernel_config->vec_nom,
            i_micro_kernel_config->vec_denom,
            i_micro_kernel_config->mask_hi,
            i_micro_kernel_config->mask_lo,
            i_micro_kernel_config->vec_c0,
            i_micro_kernel_config->vec_c1,
            i_micro_kernel_config->vec_c2,
            i_micro_kernel_config->vec_c3,
            i_micro_kernel_config->vec_c1_d,
            i_micro_kernel_config->vec_c2_d,
            i_micro_kernel_config->vec_c3_d,
            i_micro_kernel_config->vec_hi_bound,
            i_micro_kernel_config->vec_lo_bound,
            i_micro_kernel_config->vec_ones,
            i_micro_kernel_config->vec_neg_ones,
            i_micro_kernel_config->vec_halves,
            i_micro_kernel_config->vec_tmp0,
            l_tupletype );
        }

        if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV) {
          if (l_is_sve){
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FSUB_V,
                                                     i_micro_kernel_config->vec_ones, cur_vreg, 0, i_micro_kernel_config->vec_x2,
                                                     l_pred_reg, l_sve_type );
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                                     i_micro_kernel_config->vec_x2, cur_vreg, 0, cur_vreg,
                                                     l_pred_reg, l_sve_type );
          } else {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FSUB_V,
                                                       i_micro_kernel_config->vec_ones, cur_vreg, 0, i_micro_kernel_config->vec_x2,
                                                       l_tupletype );
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                                       i_micro_kernel_config->vec_x2, cur_vreg, 0, cur_vreg,
                                                       l_tupletype );
          }
        }
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_GELU) {
        if (l_is_sve){
          libxsmm_generator_gelu_ps_minimax3_aarch64_sve( io_generated_code,
            cur_vreg,
            i_micro_kernel_config->vec_xr,
            i_micro_kernel_config->vec_xa,
            i_micro_kernel_config->vec_index,
            i_micro_kernel_config->vec_C0,
            i_micro_kernel_config->vec_C1,
            i_micro_kernel_config->vec_C2,
            i_micro_kernel_config->vec_thres,
            i_micro_kernel_config->vec_absmask,
            i_micro_kernel_config->vec_scale,
            i_micro_kernel_config->vec_shifter,
            i_micro_kernel_config->vec_halves,
            i_micro_kernel_config->vec_c0,
            i_micro_kernel_config->vec_c01,
            i_micro_kernel_config->vec_c02,
            i_micro_kernel_config->vec_c03,
            i_micro_kernel_config->vec_c1,
            i_micro_kernel_config->vec_c11,
            i_micro_kernel_config->vec_c12,
            i_micro_kernel_config->vec_c13,
            i_micro_kernel_config->vec_c2,
            i_micro_kernel_config->vec_c21,
            i_micro_kernel_config->vec_c22,
            i_micro_kernel_config->vec_c23,
            i_micro_kernel_config->vec_tmp0, /* expmask */
            i_micro_kernel_config->vec_tmp1,
            l_sve_type, LIBXSMM_CAST_UCHAR(l_pred_reg) );
        } else {
          libxsmm_generator_gelu_ps_minimax3_aarch64_asimd( io_generated_code,
            cur_vreg,
            i_micro_kernel_config->vec_xr,
            i_micro_kernel_config->vec_xa,
            i_micro_kernel_config->vec_index,
            i_micro_kernel_config->vec_C0,
            i_micro_kernel_config->vec_C1,
            i_micro_kernel_config->vec_C2,
            i_micro_kernel_config->vec_thres,
            i_micro_kernel_config->vec_absmask,
            i_micro_kernel_config->vec_scale,
            i_micro_kernel_config->vec_shifter,
            i_micro_kernel_config->vec_halves,
            i_micro_kernel_config->vec_c0,
            i_micro_kernel_config->vec_c1,
            i_micro_kernel_config->vec_c2,
            i_micro_kernel_config->vec_tmp0,
            i_micro_kernel_config->vec_tmp1,
            l_tupletype );
        }
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV) {
        if (l_is_sve){
          libxsmm_generator_gelu_inv_ps_minimax3_aarch64_sve( io_generated_code,
            cur_vreg,
            i_micro_kernel_config->vec_xr,
            i_micro_kernel_config->vec_xa,
            i_micro_kernel_config->vec_index,
            i_micro_kernel_config->vec_C0,
            i_micro_kernel_config->vec_C1,
            i_micro_kernel_config->vec_C2,
            i_micro_kernel_config->vec_thres,
            i_micro_kernel_config->vec_absmask,
            i_micro_kernel_config->vec_scale,
            i_micro_kernel_config->vec_shifter,
            i_micro_kernel_config->vec_halves,
            i_micro_kernel_config->vec_c0,
            i_micro_kernel_config->vec_c01,
            i_micro_kernel_config->vec_c02,
            i_micro_kernel_config->vec_c03,
            i_micro_kernel_config->vec_c1,
            i_micro_kernel_config->vec_c11,
            i_micro_kernel_config->vec_c12,
            i_micro_kernel_config->vec_c13,
            i_micro_kernel_config->vec_c2,
            i_micro_kernel_config->vec_c21,
            i_micro_kernel_config->vec_c22,
            i_micro_kernel_config->vec_c23,
            i_micro_kernel_config->vec_tmp0, /* expmask */
            i_micro_kernel_config->vec_tmp1,
            l_sve_type, LIBXSMM_CAST_UCHAR(l_pred_reg) );
        } else {
          libxsmm_generator_gelu_inv_ps_minimax3_aarch64_asimd( io_generated_code,
            cur_vreg,
            i_micro_kernel_config->vec_xr,
            i_micro_kernel_config->vec_xa,
            i_micro_kernel_config->vec_index,
            i_micro_kernel_config->vec_C0,
            i_micro_kernel_config->vec_C1,
            i_micro_kernel_config->vec_C2,
            i_micro_kernel_config->vec_thres,
            i_micro_kernel_config->vec_absmask,
            i_micro_kernel_config->vec_scale,
            i_micro_kernel_config->vec_shifter,
            i_micro_kernel_config->vec_halves,
            i_micro_kernel_config->vec_c0,
            i_micro_kernel_config->vec_c1,
            i_micro_kernel_config->vec_c2,
            i_micro_kernel_config->vec_tmp0,
            i_micro_kernel_config->vec_tmp1,
            l_tupletype );
        }
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_compute_binary_op_2d_reg_block_aarch64( libxsmm_generated_code*    io_generated_code,
                                                 libxsmm_matequation_kernel_config*      i_meqn_micro_kernel_config,
                                                 libxsmm_meltw_binary_type               i_op_type,
                                                 unsigned int                            i_left_reg_block_id,
                                                 unsigned int                            i_right_reg_block_id,
                                                 unsigned int                            i_dst_reg_block_id,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 libxsmm_datatype                        i_dtype ) {
  unsigned int i_left_start_vreg = libxsmm_generator_matequation_regblocks_get_start_of_register_block(i_meqn_micro_kernel_config, i_left_reg_block_id);
  unsigned int i_right_start_vreg = libxsmm_generator_matequation_regblocks_get_start_of_register_block(i_meqn_micro_kernel_config, i_right_reg_block_id);
  unsigned int i_dst_start_vreg = libxsmm_generator_matequation_regblocks_get_start_of_register_block(i_meqn_micro_kernel_config, i_dst_reg_block_id);
  unsigned int im, in, left_vreg, right_vreg, dst_vreg;
  unsigned int binary_op_instr = 0;
  unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  unsigned char l_pred_reg = 0;
  libxsmm_aarch64_sve_type l_sve_type = libxsmm_generator_aarch64_get_sve_type((i_dtype == LIBXSMM_DATATYPE_F64) ? 8 : 4);
  libxsmm_aarch64_asimd_tupletype l_tupletype = (i_dtype == LIBXSMM_DATATYPE_F64) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S;

  switch ((int)i_op_type) {
    case LIBXSMM_MELTW_TYPE_BINARY_ADD: {
      binary_op_instr = l_is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FADD_V : LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_MUL: {
      binary_op_instr = l_is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FMUL_V : LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_SUB: {
      binary_op_instr = l_is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FSUB_V : LIBXSMM_AARCH64_INSTR_ASIMD_FSUB_V;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_DIV: {
      binary_op_instr = l_is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FDIV_V_P : LIBXSMM_AARCH64_INSTR_ASIMD_FDIV_V;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD: {
      binary_op_instr = l_is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_P : LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_MAX: {
      binary_op_instr = l_is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FMAX_V_P : LIBXSMM_AARCH64_INSTR_ASIMD_FMAX_V;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_MIN: {
      binary_op_instr = l_is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FMIN_V_P : LIBXSMM_AARCH64_INSTR_ASIMD_FMIN_V;
    } break;
    default:;
  }

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      left_vreg = i_left_start_vreg + in * i_m_blocking + im;
      right_vreg = i_right_start_vreg + in * i_m_blocking + im;
      dst_vreg = i_dst_start_vreg + in * i_m_blocking + im;
      if (i_op_type == LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD) {
        dst_vreg = i_meqn_micro_kernel_config->reduce_vreg;
      }
      if (i_op_type == LIBXSMM_MELTW_TYPE_BINARY_ZIP) {
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UUNPKLO_V, left_vreg, LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, left_vreg, l_pred_reg, l_sve_type );
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UUNPKLO_V, right_vreg, LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, right_vreg, l_pred_reg, l_sve_type);
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LSL_I_V, right_vreg, LIBXSMM_AARCH64_SVE_REG_UNDEF, 16, right_vreg, l_pred_reg, l_sve_type);
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V, left_vreg, right_vreg, 0, dst_vreg, l_pred_reg, l_sve_type );
      } else {
        if ( l_is_sve ) {
          if ((binary_op_instr == LIBXSMM_AARCH64_INSTR_SVE_FDIV_V_P) && (left_vreg != dst_vreg) && (right_vreg != dst_vreg)) {
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V, left_vreg, left_vreg, 0, dst_vreg, l_pred_reg, l_sve_type );
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, binary_op_instr, dst_vreg, right_vreg, 0, dst_vreg, l_pred_reg, l_sve_type );
          } else {
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, binary_op_instr, left_vreg, right_vreg, 0, dst_vreg, l_pred_reg, l_sve_type );
          }
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, binary_op_instr, left_vreg, right_vreg, 0, dst_vreg, l_tupletype);
        }
      }
    }
  }

  return;
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_compute_ternary_op_2d_reg_block_aarch64( libxsmm_generated_code*    io_generated_code,
                                                 libxsmm_matequation_kernel_config*      i_meqn_micro_kernel_config,
                                                 libxsmm_meltw_ternary_type              i_op_type,
                                                 unsigned int                            i_left_reg_block_id,
                                                 unsigned int                            i_right_reg_block_id,
                                                 unsigned int                            i_dst_reg_block_id,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 libxsmm_datatype                        i_dtype ) {
  unsigned int i_left_start_vreg = libxsmm_generator_matequation_regblocks_get_start_of_register_block(i_meqn_micro_kernel_config, i_left_reg_block_id);
  unsigned int i_right_start_vreg = libxsmm_generator_matequation_regblocks_get_start_of_register_block(i_meqn_micro_kernel_config, i_right_reg_block_id);
  unsigned int i_dst_start_vreg = libxsmm_generator_matequation_regblocks_get_start_of_register_block(i_meqn_micro_kernel_config, i_dst_reg_block_id);
  unsigned int im, in, left_vreg, right_vreg, dst_vreg;
  unsigned int ternary_op_instr = 0;
  unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  unsigned char l_pred_reg = 0;
  libxsmm_aarch64_sve_type l_sve_type = libxsmm_generator_aarch64_get_sve_type((i_dtype == LIBXSMM_DATATYPE_F64) ? 8 : 4);
  libxsmm_aarch64_asimd_tupletype l_tupletype = (i_dtype == LIBXSMM_DATATYPE_F64) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S;

  switch ((int)i_op_type) {
    case LIBXSMM_MELTW_TYPE_TERNARY_MULADD: {
      ternary_op_instr =  l_is_sve ? LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_P : LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V;
    } break;
    default:;
  }
  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      left_vreg = i_left_start_vreg + in * i_m_blocking + im;
      right_vreg = i_right_start_vreg + in * i_m_blocking + im;
      dst_vreg = i_dst_start_vreg + in * i_m_blocking + im;
      if (i_op_type == LIBXSMM_MELTW_TYPE_TERNARY_NMULADD) {
        if ( l_is_sve ) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V, left_vreg, dst_vreg, 0, dst_vreg, l_pred_reg, l_sve_type );
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FSUB_V, right_vreg, dst_vreg, 0, dst_vreg, l_pred_reg, l_sve_type );
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V, left_vreg, dst_vreg, 0, dst_vreg, l_tupletype);
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FSUB_V, right_vreg, dst_vreg, 0, dst_vreg, l_tupletype);
        }
      } else {
        if ( l_is_sve ) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, ternary_op_instr, left_vreg, right_vreg, 0, dst_vreg, l_pred_reg, l_sve_type );
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, ternary_op_instr, left_vreg, right_vreg, 0, dst_vreg, l_tupletype);
        }
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_2d_microkernel_aarch64( libxsmm_generated_code*                    io_generated_code,
                                                 libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                 libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
                                                 libxsmm_matequation_kernel_config*      i_micro_kernel_config,
                                                 const libxsmm_meqn_descriptor*          i_meqn_desc,
                                                 libxsmm_matrix_eqn                      *i_eqn,
                                                 unsigned int                            i_m,
                                                 unsigned int                            i_n,
                                                 unsigned int                            skip_n_loop_reg_cleanup ) {

  unsigned int use_m_input_masking, use_m_output_masking, m_trips, m_unroll_factor, m_assm_trips, n_trips, n_unroll_factor, n_assm_trips;
  unsigned int mask_reg_in, mask_reg_out;
  unsigned int i_vlen_in = i_micro_kernel_config->vlen_in;
  unsigned int i_vlen_out = i_micro_kernel_config->vlen_out;
  unsigned int temp_reg = i_gp_reg_mapping->temp_reg;
  unsigned int last_timestamp = i_eqn->eqn_root->visit_timestamp;
  unsigned int arg_id = 0;
  unsigned int aux_reg_block = 0;
  unsigned int aux_reg_block2 = 0;
  unsigned int right_reg_block = 0;
  unsigned int right2_reg_block = 0;
  unsigned int left_reg_block = 0;
  unsigned int timestamp = 0;

  /* Configure microkernel masks */
  libxsmm_meqn_setup_input_output_masks_aarch64( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
      temp_reg, i_m, &use_m_input_masking, &mask_reg_in, &use_m_output_masking, &mask_reg_out);

  /* Configure microkernel loops */
  libxsmm_configure_mateqn_microkernel_loops_aarch64( io_generated_code, i_micro_kernel_config, i_eqn, i_m, i_n, use_m_input_masking,
      &m_trips, &n_trips, &m_unroll_factor, &n_unroll_factor, &m_assm_trips, &n_assm_trips);

  i_micro_kernel_config->register_block_size = m_unroll_factor * n_unroll_factor;

  aux_reg_block = i_micro_kernel_config->n_tmp_reg_blocks - 1;
  aux_reg_block2 = i_micro_kernel_config->n_tmp_reg_blocks - 2;

  /* Headers of microkernel loops */
  if (n_assm_trips > 1) {
    libxsmm_generator_loop_header_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, n_trips);
  }

  if (m_assm_trips > 1) {
    libxsmm_generator_loop_header_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, m_trips);
  }

  /* Traverse equation tree based on optimal execution plan and emit code */
  for (timestamp = 0; timestamp <= last_timestamp; timestamp++) {
    libxsmm_meqn_elem *cur_op = libxsmm_generator_matequation_find_op_at_timestamp(i_eqn->eqn_root, timestamp);
    if (cur_op->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) {
      if (cur_op->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
        /* We have to load the input from the argument tensor using the node's assigned tmp reg block */
        left_reg_block = cur_op->tmp.id;
        libxsmm_generator_mateqn_load_arg_to_2d_reg_block_aarch64( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
            arg_id, i_vlen_in, left_reg_block, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in, (cur_op->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_DUMP) ? 1 : 0 );
        arg_id++;
      }
      if (cur_op->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_DUMP) {
        /* Prepare the register with the dump address */
        unsigned int n_opargs = i_micro_kernel_config->n_opargs, _i = 0;
        unsigned int n_args = i_micro_kernel_config->n_args;
        for (_i = 0; _i < n_opargs; _i++) {
          if (i_micro_kernel_config->oparg_info[_i].id == cur_op->info.u_op.op_arg_pos) {
            libxsmm_generator_meqn_getaddr_stack_tmpaddr_i_aarch64( io_generated_code, n_args * 8 + _i * 32, i_gp_reg_mapping->temp_reg3, i_gp_reg_mapping->temp_reg2);
            libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->temp_reg2, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, temp_reg);
            break;
          }
        }
        libxsmm_generator_mateqn_dump_2d_reg_block_aarch64( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
            i_vlen_in, cur_op->tmp.m,  cur_op->tmp.ld, cur_op->tmp.id, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in, cur_op->tmp.dtype, temp_reg );
      } else {
        libxsmm_generator_mateqn_compute_unary_op_2d_reg_block_aarch64( io_generated_code, i_micro_kernel_config,
            cur_op->info.u_op.type, cur_op->tmp.id, m_unroll_factor, n_unroll_factor, cur_op->info.u_op.dtype);
      }
    } else if (cur_op->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) {
      if ((cur_op->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_op->ri->type == LIBXSMM_MATRIX_EQN_NODE_ARG)) {
        /* We have to load the input from the argument tensor using the node's assigned tmp reg block */
        left_reg_block = cur_op->tmp.id;
        libxsmm_generator_mateqn_load_arg_to_2d_reg_block_aarch64( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
            arg_id, i_vlen_in, left_reg_block, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in, (cur_op->info.b_op.type == LIBXSMM_MELTW_TYPE_BINARY_ZIP) ? 1 : 0 );
        arg_id++;
        right_reg_block = aux_reg_block;
        libxsmm_generator_mateqn_load_arg_to_2d_reg_block_aarch64( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
            arg_id, i_vlen_in, right_reg_block, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in, (cur_op->info.b_op.type == LIBXSMM_MELTW_TYPE_BINARY_ZIP) ? 1 : 0 );
        arg_id++;
      } else if ((cur_op->le->type != LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_op->ri->type != LIBXSMM_MATRIX_EQN_NODE_ARG)) {
        left_reg_block = cur_op->le->tmp.id;
        right_reg_block = cur_op->ri->tmp.id;
      } else {
        if (cur_op->ri->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
          /* We have to load the input from the argument tensor using the auxiliary tmp reg block */
          left_reg_block = cur_op->le->tmp.id;
          right_reg_block = aux_reg_block;
          libxsmm_generator_mateqn_load_arg_to_2d_reg_block_aarch64( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
              arg_id, i_vlen_in, aux_reg_block, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in, (cur_op->info.b_op.type == LIBXSMM_MELTW_TYPE_BINARY_ZIP) ? 1 : 0 );
          arg_id++;
        } else {
          left_reg_block = aux_reg_block;
          right_reg_block = cur_op->ri->tmp.id;
          libxsmm_generator_mateqn_load_arg_to_2d_reg_block_aarch64( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
              arg_id, i_vlen_in, aux_reg_block, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in, (cur_op->info.b_op.type == LIBXSMM_MELTW_TYPE_BINARY_ZIP) ? 1 : 0 );
          arg_id++;
        }
      }
      libxsmm_generator_mateqn_compute_binary_op_2d_reg_block_aarch64( io_generated_code, i_micro_kernel_config,
          cur_op->info.b_op.type, left_reg_block, right_reg_block, cur_op->tmp.id, m_unroll_factor, n_unroll_factor, cur_op->info.b_op.dtype);
    } else if (cur_op->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) {
      if (cur_op->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
        left_reg_block = aux_reg_block;
        libxsmm_generator_mateqn_load_arg_to_2d_reg_block_aarch64( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
            arg_id, i_vlen_in, left_reg_block, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in, 0 );
        arg_id++;
      } else {
        left_reg_block = cur_op->le->tmp.id;
      }
      if (cur_op->ri->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
        right_reg_block = aux_reg_block2;
        libxsmm_generator_mateqn_load_arg_to_2d_reg_block_aarch64( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
            arg_id, i_vlen_in, right_reg_block, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in, 0 );
        arg_id++;
      } else {
        right_reg_block = cur_op->ri->tmp.id;
      }
      if (cur_op->r2->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
        right2_reg_block = cur_op->tmp.id;
        libxsmm_generator_mateqn_load_arg_to_2d_reg_block_aarch64( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
            arg_id, i_vlen_in, right2_reg_block, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in, 0 );
        arg_id++;
      } else {
        right2_reg_block = cur_op->r2->tmp.id;
      }
      libxsmm_generator_mateqn_compute_ternary_op_2d_reg_block_aarch64( io_generated_code, i_micro_kernel_config,
          cur_op->info.t_op.type, left_reg_block, right_reg_block, right2_reg_block, m_unroll_factor, n_unroll_factor, cur_op->info.t_op.dtype);
    } else {
      /* This should not happen */
    }
  }

  /* Store the computed register block to output  */
  if ((i_eqn->eqn_root->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (i_eqn->eqn_root->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_UNZIP)) {
    libxsmm_generator_mateqn_unpackstore_2d_reg_block_aarch64( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
        i_vlen_out, i_eqn->eqn_root->tmp.id, m_unroll_factor, n_unroll_factor, use_m_output_masking, mask_reg_out );
  } else {
    libxsmm_generator_mateqn_store_2d_reg_block_aarch64( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
        i_vlen_out, i_eqn->eqn_root->tmp.id, m_unroll_factor, n_unroll_factor, use_m_output_masking, mask_reg_out );
  }

  /* Footers of microkernel loops */
  if (m_assm_trips > 1) {
    libxsmm_generator_mateqn_adjust_args_addr_aarch64(io_generated_code, i_meqn_desc, i_gp_reg_mapping, i_micro_kernel_config, LIBXSMM_AARCH64_INSTR_GP_META_ADD, m_unroll_factor * i_vlen_in , M_ADJUSTMENT, i_micro_kernel_config->arg_info);
    libxsmm_generator_mateqn_adjust_opargs_addr_aarch64(io_generated_code, i_meqn_desc, i_gp_reg_mapping, i_micro_kernel_config, LIBXSMM_AARCH64_INSTR_GP_META_ADD, m_unroll_factor * i_vlen_in , M_ADJUSTMENT, i_micro_kernel_config->oparg_info);

    libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, m_unroll_factor);

    libxsmm_generator_mateqn_adjust_args_addr_aarch64(io_generated_code, i_meqn_desc, i_gp_reg_mapping, i_micro_kernel_config, LIBXSMM_AARCH64_INSTR_GP_META_SUB, m_unroll_factor * i_vlen_in * m_assm_trips, M_ADJUSTMENT, i_micro_kernel_config->arg_info);
    libxsmm_generator_mateqn_adjust_opargs_addr_aarch64(io_generated_code, i_meqn_desc, i_gp_reg_mapping, i_micro_kernel_config, LIBXSMM_AARCH64_INSTR_GP_META_SUB, m_unroll_factor * i_vlen_in * m_assm_trips, M_ADJUSTMENT, i_micro_kernel_config->oparg_info);
  }

  if (n_assm_trips > 1) {
    libxsmm_generator_mateqn_adjust_args_addr_aarch64(io_generated_code, i_meqn_desc, i_gp_reg_mapping, i_micro_kernel_config, LIBXSMM_AARCH64_INSTR_GP_META_ADD, n_unroll_factor, N_ADJUSTMENT, i_micro_kernel_config->arg_info);
    libxsmm_generator_mateqn_adjust_opargs_addr_aarch64(io_generated_code, i_meqn_desc, i_gp_reg_mapping, i_micro_kernel_config, LIBXSMM_AARCH64_INSTR_GP_META_ADD, n_unroll_factor, N_ADJUSTMENT, i_micro_kernel_config->oparg_info);
    libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, n_unroll_factor);
    if (skip_n_loop_reg_cleanup == 0) {
      libxsmm_generator_mateqn_adjust_args_addr_aarch64(io_generated_code, i_meqn_desc, i_gp_reg_mapping, i_micro_kernel_config, LIBXSMM_AARCH64_INSTR_GP_META_SUB, n_unroll_factor * n_assm_trips, N_ADJUSTMENT, i_micro_kernel_config->arg_info);
      libxsmm_generator_mateqn_adjust_opargs_addr_aarch64(io_generated_code, i_meqn_desc, i_gp_reg_mapping, i_micro_kernel_config, LIBXSMM_AARCH64_INSTR_GP_META_SUB, n_unroll_factor * n_assm_trips, N_ADJUSTMENT, i_micro_kernel_config->oparg_info);
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_configure_M_N_blocking_aarch64( libxsmm_generated_code* io_generated_code, libxsmm_matequation_kernel_config* i_micro_kernel_config, libxsmm_matrix_eqn *i_eqn, unsigned int m, unsigned int n, unsigned int vlen, unsigned int *m_blocking, unsigned int *n_blocking) {
  /* The m blocking is done in chunks of vlen */
  unsigned int m_chunks = (m+vlen-1)/vlen;
  unsigned int m_chunk_remainder = 8;
  unsigned int m_range, m_block_size, foo1, foo2;
  unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;
  unsigned int n_tmp_reg_blocks = i_eqn->eqn_root->reg_score;
  unsigned int max_nm_unrolling = 32 - reserved_zmms;

  LIBXSMM_UNUSED( io_generated_code );

  if (i_micro_kernel_config->contains_binary_op > 0) {
    n_tmp_reg_blocks++;
    if (i_micro_kernel_config->contains_ternary_op > 0) {
      n_tmp_reg_blocks++;
    }
  } else if (i_micro_kernel_config->contains_ternary_op > 0) {
    n_tmp_reg_blocks += 2;
  }

  max_nm_unrolling = max_nm_unrolling / n_tmp_reg_blocks;
  if (m % vlen == 0) {
    /* If there is not remainder in M, then we block M in order to limit block size */
    if (m_chunks > 32) {
      libxsmm_compute_equalized_blocking(m_chunks, (m_chunks+1)/2, &m_range, &m_block_size, &foo1, &foo2);
      *m_blocking = m_range * vlen;
    } else {
      *m_blocking = m;
    }
  } else {
    /* If there is remainder we make sure we can fully unroll the kernel with masks */
    if (m_chunk_remainder > max_nm_unrolling) {
      m_chunk_remainder = max_nm_unrolling;
    }
    if (m_chunk_remainder >= m_chunks) {
      *m_blocking = m;
    } else {
      *m_blocking = (m_chunks - m_chunk_remainder) * vlen;
    }
  }

  /* FIXME: When we dont have nice N values AND we have more register unrolling oportunities, apply n blocking...
   * For now not any additional blocking in N */
  *n_blocking = n;
}

LIBXSMM_API_INTERN
void libxsmm_generator_configure_equation_aarch64_vlens(libxsmm_generated_code*    io_generated_code, libxsmm_matequation_kernel_config* i_micro_kernel_config, libxsmm_matrix_eqn *eqn)  {
  /* First, determine the vlen compute based on the min. compute of the equation */
  unsigned int l_vector_length = libxsmm_cpuid_vlen(io_generated_code->arch);
  int tree_max_comp_tsize = eqn->eqn_root->tree_max_comp_tsize;
  i_micro_kernel_config->vlen_comp = l_vector_length/tree_max_comp_tsize;
  /* The vlen_in and vlen_out are aligned with the vlen compute */
  i_micro_kernel_config->vlen_in = i_micro_kernel_config->vlen_comp;
  i_micro_kernel_config->vlen_out = i_micro_kernel_config->vlen_comp;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_matequation_regblocks_unary_op_req_zmms_aarch64(libxsmm_generated_code*    io_generated_code, libxsmm_meltw_unary_type u_type) {
  unsigned int result = 0;
  unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);

  switch ((int)u_type) {
    case LIBXSMM_MELTW_TYPE_UNARY_REDUCE_TO_SCALAR_OP_ADD: {
      result = 1;
    } break;
    case LIBXSMM_MELTW_TYPE_UNARY_XOR: {
      result = 1;
    } break;
    case LIBXSMM_MELTW_TYPE_UNARY_INC: {
      result = 1;
    } break;
    case LIBXSMM_MELTW_TYPE_UNARY_GELU: {
      result = (l_is_sve > 0) ? 15 : 25;
    } break;
    case LIBXSMM_MELTW_TYPE_UNARY_GELU_INV: {
      result = (l_is_sve > 0) ? 15 : 25;
    } break;
    case LIBXSMM_MELTW_TYPE_UNARY_EXP: {
      result = 9;
    } break;
    case LIBXSMM_MELTW_TYPE_UNARY_TANH: {
      result = 17;
    } break;
    case LIBXSMM_MELTW_TYPE_UNARY_TANH_INV: {
      result = 17;
    } break;
    case LIBXSMM_MELTW_TYPE_UNARY_SIGMOID: {
      result = 18;
    } break;
    case LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV: {
      result = 18;
    } break;
    default:;
  }
  return result;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_matequation_regblocks_binary_op_req_zmms_aarch64( libxsmm_generated_code*    io_generated_code, libxsmm_meltw_binary_type b_type) {
  unsigned int result = 0;
  LIBXSMM_UNUSED(io_generated_code);
  switch ((int)b_type) {
    case LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD: {
      result = 1;
    } break;
    default:;
  }
  return result;
}

LIBXSMM_API_INTERN
void libxsmm_adjust_required_zmms_aarch64( libxsmm_generated_code*    io_generated_code,  libxsmm_matequation_kernel_config* i_micro_kernel_config, libxsmm_meltw_unary_type u_type, libxsmm_meltw_binary_type b_type, unsigned int pool_id ) {
  unsigned int n_req_zmms = 0;
  if (pool_id == UNARY_OP_POOL) {
    if (i_micro_kernel_config->unary_ops_pool[u_type] == 0) {
      n_req_zmms = libxsmm_generator_matequation_regblocks_unary_op_req_zmms_aarch64( io_generated_code,  u_type);
      i_micro_kernel_config->reserved_zmms += n_req_zmms;
      i_micro_kernel_config->unary_ops_pool[u_type] = 1;
    }
  } else if (pool_id == BINARY_OP_POOL) {
    if (i_micro_kernel_config->binary_ops_pool[b_type] == 0) {
      n_req_zmms = libxsmm_generator_matequation_regblocks_binary_op_req_zmms_aarch64( io_generated_code,  b_type);
      i_micro_kernel_config->reserved_zmms += n_req_zmms;
      i_micro_kernel_config->binary_ops_pool[b_type] = 1;
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_mark_reserved_zmms_aarch64( libxsmm_generated_code*    io_generated_code, libxsmm_matequation_kernel_config* i_micro_kernel_config, libxsmm_meqn_elem *cur_node ) {
  if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) {
    libxsmm_adjust_required_zmms_aarch64( io_generated_code,  i_micro_kernel_config, cur_node->info.u_op.type, LIBXSMM_MELTW_TYPE_BINARY_NONE, UNARY_OP_POOL);
    libxsmm_mark_reserved_zmms_aarch64( io_generated_code, i_micro_kernel_config, cur_node->le);
  } else if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) {
    libxsmm_adjust_required_zmms_aarch64(io_generated_code, i_micro_kernel_config, LIBXSMM_MELTW_TYPE_UNARY_NONE, cur_node->info.b_op.type, BINARY_OP_POOL);
    libxsmm_mark_reserved_zmms_aarch64(io_generated_code, i_micro_kernel_config, cur_node->le);
    libxsmm_mark_reserved_zmms_aarch64(io_generated_code, i_micro_kernel_config, cur_node->ri);
  } else if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) {
    libxsmm_mark_reserved_zmms_aarch64(io_generated_code, i_micro_kernel_config, cur_node->le);
    libxsmm_mark_reserved_zmms_aarch64(io_generated_code, i_micro_kernel_config, cur_node->ri);
    libxsmm_mark_reserved_zmms_aarch64(io_generated_code, i_micro_kernel_config, cur_node->r2);
  }
}

LIBXSMM_API_INTERN
void libxsmm_configure_reserved_zmms_and_masks_aarch64(libxsmm_generated_code* io_generated_code,
    const libxsmm_meqn_descriptor*          i_mateqn_desc,
    libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
    libxsmm_matequation_kernel_config*      i_micro_kernel_config,
    libxsmm_matrix_eqn                      *eqn ) {
  unsigned int i = 0;
  libxsmm_mateltwise_kernel_config *meltw_config;
  libxsmm_datatype eqn_root_dtype = LIBXSMM_DATATYPE_F32;
  unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);

  libxsmm_mark_reserved_zmms_aarch64( io_generated_code, i_micro_kernel_config, eqn->eqn_root);
  i_micro_kernel_config->meltw_kernel_config.reserved_zmms = 0;
  i_micro_kernel_config->meltw_kernel_config.reserved_mask_regs = 1;
  i_micro_kernel_config->meltw_kernel_config.datatype_size_in = (LIBXSMM_GETENUM_OUT(i_mateqn_desc->datatype) == LIBXSMM_DATATYPE_F64) ? 8 : 4; /* Default to SP computations */
  meltw_config = (libxsmm_mateltwise_kernel_config*) &(i_micro_kernel_config->meltw_kernel_config);

  if (l_is_sve) libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, 0, -1, i_gp_reg_mapping->gp_reg_scratch_0 );

  /* TODO: some diagnostic if we need excessive number of required zmms for the equation and bail out */
  for (i = 0 ; i < 64; i++) {
    if (i_micro_kernel_config->unary_ops_pool[i] > 0) {
      /* TODO: Evangelos: see the last to args... they are needed for dropout... */
      libxsmm_configure_unary_aarch64_kernel_vregs_masks( io_generated_code, meltw_config, i, 0, i_gp_reg_mapping->temp_reg, i_gp_reg_mapping->temp_reg2, i_gp_reg_mapping->temp_reg, i_gp_reg_mapping->temp_reg2 );
    }
  }

  i_micro_kernel_config->reserved_zmms = meltw_config->reserved_zmms;
  i_micro_kernel_config->reserved_mask_regs = meltw_config->reserved_mask_regs;

  /* Configure Reduce-to-scalar zmms and mask */
  i_micro_kernel_config->is_head_reduce_to_scalar = 0;
  for (i = 0 ; i < 64; i++) {
    if (((i_micro_kernel_config->unary_ops_pool[i] > 0) && (i == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_TO_SCALAR_OP_ADD)) ||
        ((i_micro_kernel_config->binary_ops_pool[i] > 0) && (i == LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD)) ) {
      i_micro_kernel_config->is_head_reduce_to_scalar = 1;
      i_micro_kernel_config->reduce_vreg = i_micro_kernel_config->reserved_zmms;
      i_micro_kernel_config->reserved_zmms += i_micro_kernel_config->reserved_zmms + 1;

      if ( l_is_sve ) {
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                                 i_micro_kernel_config->reduce_vreg, i_micro_kernel_config->reduce_vreg, 0, i_micro_kernel_config->reduce_vreg,
                                                 0, libxsmm_generator_aarch64_get_sve_type(4) );
      } else {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                   i_micro_kernel_config->reduce_vreg, i_micro_kernel_config->reduce_vreg, 0, i_micro_kernel_config->reduce_vreg,
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
      }
    }

    if ((i_micro_kernel_config->unary_ops_pool[i] > 0) && (i == LIBXSMM_MELTW_TYPE_UNARY_UNZIP)) {
      i_micro_kernel_config->tmp_vreg = i_micro_kernel_config->reserved_zmms;
      i_micro_kernel_config->reserved_zmms += i_micro_kernel_config->reserved_zmms + 1;
    }
  }

  /* Check if we need to downconvert result from f32->bf16 eventually and if need be assign auc registers */
  if (eqn->eqn_root->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) {
    eqn_root_dtype = eqn->eqn_root->info.u_op.dtype;
  } else if (eqn->eqn_root->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) {
    eqn_root_dtype = eqn->eqn_root->info.b_op.dtype;
  } else if (eqn->eqn_root->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) {
    eqn_root_dtype = eqn->eqn_root->info.t_op.dtype;
  } else {
    /* Should not happen */
  }

  if ((eqn_root_dtype == LIBXSMM_DATATYPE_F32) && (LIBXSMM_GETENUM_OUT(i_mateqn_desc->datatype) == LIBXSMM_DATATYPE_BF16)) {
    i_micro_kernel_config->cvt_result_to_bf16 = 1;
    i_micro_kernel_config->use_fp32bf16_cvt_replacement = 0;
  } else {
    i_micro_kernel_config->cvt_result_to_bf16 = 0;
    i_micro_kernel_config->use_fp32bf16_cvt_replacement = 0;
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_tmp_register_block_aarch64_kernel( libxsmm_generated_code* io_generated_code,
    const libxsmm_meqn_descriptor*          i_mateqn_desc,
    libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
    libxsmm_matequation_kernel_config*      i_micro_kernel_config_orig,
    libxsmm_loop_label_tracker*             io_loop_label_tracker,
    libxsmm_matrix_eqn*                     eqn ) {
  libxsmm_meqn_arg              *arg_info;
  libxsmm_meqn_tmp_info            *oparg_info;
  unsigned int arg_id = 0, i = 0, oparg_id = 0;
  unsigned int m_blocking = 0, n_blocking = 0, cur_n = 0, cur_m = 0, n_microkernel = 0, m_microkernel = 0, adjusted_aux_vars = 0;
  libxsmm_matequation_kernel_config l_meqn_kernel_config = *i_micro_kernel_config_orig;
  libxsmm_matequation_kernel_config *i_micro_kernel_config = &l_meqn_kernel_config;

  if ( eqn == NULL ) {
    fprintf( stderr, "The requested equation does not exist... nothing to JIT,,,\n" );
    return;
  }

  /* Adjusting n_args for the current equation */
  i_micro_kernel_config->n_args = eqn->eqn_root->n_args;

  for (i = 0 ; i < 64; i++) {
    i_micro_kernel_config->unary_ops_pool[i] = 0;
    i_micro_kernel_config->binary_ops_pool[i] = 0;
  }

  /* Propagate bcast info in the tree */
  libxsmm_generator_matequation_regblocks_propagate_bcast_info(eqn);

  /* Iterate over the equation tree and copy the args ptrs in the auxiliary scratch */
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 8, i_gp_reg_mapping->gp_reg_scratch_0 );
  arg_info = (libxsmm_meqn_arg*) malloc(i_micro_kernel_config->n_args * sizeof(libxsmm_meqn_arg));
  oparg_info = (libxsmm_meqn_tmp_info*) malloc((eqn->eqn_root->visit_timestamp + 1) * sizeof(libxsmm_meqn_tmp_info));
  i_micro_kernel_config->contains_binary_op = 0;
  i_micro_kernel_config->contains_ternary_op = 0;
  libxsmm_generator_copy_input_args_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, eqn->eqn_root, &arg_id, arg_info, i_gp_reg_mapping->gp_reg_scratch_0);
  i_micro_kernel_config->arg_info = arg_info;

  /* Iterate over the equation tree and copy the opargs ptrs in the auxiliary scratch */
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_gp_reg_mapping->gp_reg_scratch_0);
  libxsmm_generator_copy_opargs_aarch64(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, eqn->eqn_root, &oparg_id, oparg_info, i_gp_reg_mapping->gp_reg_scratch_0);
  i_micro_kernel_config->oparg_info = oparg_info;
  i_micro_kernel_config->n_opargs = oparg_id;

  /* Setup output reg */
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 16, i_gp_reg_mapping->gp_reg_out );

  /* Configure equation vlens */
  libxsmm_generator_configure_equation_aarch64_vlens(io_generated_code, i_micro_kernel_config, eqn);

  /* Assign reserved zmms by parsing the equation */
  libxsmm_configure_reserved_zmms_and_masks_aarch64(io_generated_code, i_mateqn_desc, i_gp_reg_mapping, i_micro_kernel_config, eqn );

  /* Configure M and N blocking factors */
  libxsmm_generator_matequation_configure_M_N_blocking_aarch64(io_generated_code, i_micro_kernel_config, eqn, i_mateqn_desc->m, i_mateqn_desc->n, i_micro_kernel_config->vlen_in, &m_blocking, &n_blocking);

  cur_n = 0;
  while (cur_n != i_mateqn_desc->n) {
    cur_m = 0;
    adjusted_aux_vars = 0;
    n_microkernel = (cur_n < n_blocking) ? n_blocking : i_mateqn_desc->n - cur_n;
    while (cur_m != i_mateqn_desc->m) {
      unsigned int skip_n_loop_reg_cleanup = ((cur_n + n_microkernel == i_mateqn_desc->n) && (cur_m + m_microkernel == i_mateqn_desc->m)) ? 1 : 0 ;
      m_microkernel = (cur_m < m_blocking) ? m_blocking : i_mateqn_desc->m - cur_m;
      libxsmm_generator_mateqn_2d_microkernel_aarch64(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_mateqn_desc, eqn, m_microkernel, n_microkernel, skip_n_loop_reg_cleanup);
      cur_m += m_microkernel;
      if (cur_m != i_mateqn_desc->m) {
        adjusted_aux_vars = 1;
        libxsmm_generator_mateqn_adjust_args_addr_aarch64(io_generated_code, i_mateqn_desc, i_gp_reg_mapping, i_micro_kernel_config, LIBXSMM_AARCH64_INSTR_GP_META_ADD, m_microkernel, M_ADJUSTMENT, arg_info);
        libxsmm_generator_mateqn_adjust_opargs_addr_aarch64(io_generated_code, i_mateqn_desc, i_gp_reg_mapping, i_micro_kernel_config, LIBXSMM_AARCH64_INSTR_GP_META_ADD, m_microkernel, M_ADJUSTMENT, oparg_info);
      }
    }
    if (adjusted_aux_vars == 1) {
      libxsmm_generator_mateqn_adjust_args_addr_aarch64(io_generated_code, i_mateqn_desc, i_gp_reg_mapping, i_micro_kernel_config, LIBXSMM_AARCH64_INSTR_GP_META_SUB, m_microkernel, M_ADJUSTMENT, arg_info);
      libxsmm_generator_mateqn_adjust_opargs_addr_aarch64(io_generated_code, i_mateqn_desc, i_gp_reg_mapping, i_micro_kernel_config, LIBXSMM_AARCH64_INSTR_GP_META_SUB, m_microkernel, M_ADJUSTMENT, oparg_info);
    }
    cur_n += n_microkernel;
    if (cur_n != i_mateqn_desc->n) {
      libxsmm_generator_mateqn_adjust_args_addr_aarch64(io_generated_code, i_mateqn_desc, i_gp_reg_mapping, i_micro_kernel_config, LIBXSMM_AARCH64_INSTR_GP_META_ADD, n_microkernel, N_ADJUSTMENT,  arg_info);
      libxsmm_generator_mateqn_adjust_opargs_addr_aarch64(io_generated_code, i_mateqn_desc, i_gp_reg_mapping, i_micro_kernel_config, LIBXSMM_AARCH64_INSTR_GP_META_ADD, n_microkernel, N_ADJUSTMENT,  oparg_info);
    }
  }

  /* Store to output the scalar result */
  if (i_micro_kernel_config->is_head_reduce_to_scalar > 0) {
    libxsmm_generator_mateqn_store_reduce_to_scalar_output_aarch64( io_generated_code,  i_gp_reg_mapping, i_micro_kernel_config, i_mateqn_desc );
  }

  /* Free aux data structure */
  free(arg_info);
  free(oparg_info);
}

