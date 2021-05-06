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
#include "generator_matequation_avx_avx512.h"
#include "generator_mateltwise_sse_avx_avx512.h"
#include "generator_mateltwise_unary_binary_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "libxsmm_main.h"
#include "generator_common_x86.h"
#include "generator_matequation_regblocks_avx_avx512.h"

LIBXSMM_API_INTERN
unsigned int get_start_of_register_block(libxsmm_matequation_kernel_config *i_micro_kernel_config, unsigned int i_reg_block_id) {
  unsigned int result;
  result = i_micro_kernel_config->reserved_zmms + i_reg_block_id * i_micro_kernel_config->register_block_size;
  return result;
}

LIBXSMM_API_INTERN
void libxsmm_generator_copy_input_args(libxsmm_generated_code*        io_generated_code,
    libxsmm_matequation_gp_reg_mapping  *i_gp_reg_mapping,
    libxsmm_matequation_kernel_config   *i_micro_kernel_config,
    libxsmm_matrix_eqn_elem             *cur_node,
    unsigned int                        *arg_id,
    libxsmm_matrix_eqn_arg              *arg_info,
    unsigned int                        input_reg) {

  unsigned int temp_reg = i_gp_reg_mapping->temp_reg;
  unsigned int temp_reg2 = i_gp_reg_mapping->temp_reg2;
  if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
    unsigned int cur_pos = *arg_id;
    if (cur_node->info.arg.in_pos >= 0){
      if (cur_pos < i_micro_kernel_config->n_avail_gpr) {
        libxsmm_x86_instruction_alu_mem( io_generated_code,
            i_micro_kernel_config->alu_mov_instruction,
            input_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            cur_node->info.arg.in_pos*24,
            i_micro_kernel_config->gpr_pool[cur_pos],
            0 );
      } else {
        libxsmm_x86_instruction_alu_mem( io_generated_code,
            i_micro_kernel_config->alu_mov_instruction,
            input_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            cur_node->info.arg.in_pos*24,
            temp_reg,
            0 );
        libxsmm_generator_meqn_getaddr_stack_tmpaddr_i( io_generated_code, cur_pos * 3 * 8, temp_reg2);
        libxsmm_x86_instruction_alu_mem( io_generated_code,
            i_micro_kernel_config->alu_mov_instruction,
            temp_reg2,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            temp_reg,
            1 );
      }
    } else {
      libxsmm_generator_meqn_getaddr_stack_tmp_i( io_generated_code, (-cur_node->info.arg.in_pos-1) * 3 * i_micro_kernel_config->tmp_size, temp_reg);
      if (cur_pos < i_micro_kernel_config->n_avail_gpr) {
        libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, temp_reg, i_micro_kernel_config->gpr_pool[cur_pos]);
      } else {
        libxsmm_generator_meqn_getaddr_stack_tmpaddr_i( io_generated_code, cur_pos * 3 * 8, temp_reg2);
        libxsmm_x86_instruction_alu_mem( io_generated_code,
            i_micro_kernel_config->alu_mov_instruction,
            temp_reg2,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            temp_reg,
            1 );
      }
    }
    arg_info[cur_pos] = cur_node->info.arg;
    *arg_id = cur_pos + 1;
  } else if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) {
    libxsmm_generator_copy_input_args(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->le, arg_id, arg_info, input_reg);
  } else if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) {
    i_micro_kernel_config->contains_binary_op = 1;
    if (cur_node->le->reg_score >= cur_node->ri->reg_score) {
      libxsmm_generator_copy_input_args(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->le, arg_id, arg_info, input_reg);
      libxsmm_generator_copy_input_args(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->ri, arg_id, arg_info, input_reg);
    } else {
      libxsmm_generator_copy_input_args(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->ri, arg_id, arg_info, input_reg);
      libxsmm_generator_copy_input_args(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->le, arg_id, arg_info, input_reg);
    }
  } else if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) {
    i_micro_kernel_config->contains_ternary_op = 1;
    if ((cur_node->le->reg_score >= cur_node->ri->reg_score) && (cur_node->le->reg_score >= cur_node->r2->reg_score)) {
      libxsmm_generator_copy_input_args(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->le, arg_id, arg_info, input_reg);
      if (cur_node->ri->reg_score >= cur_node->r2->reg_score) {
        libxsmm_generator_copy_input_args(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->ri, arg_id, arg_info, input_reg);
        libxsmm_generator_copy_input_args(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->r2, arg_id, arg_info, input_reg);
      } else {
        libxsmm_generator_copy_input_args(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->r2, arg_id, arg_info, input_reg);
        libxsmm_generator_copy_input_args(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->ri, arg_id, arg_info, input_reg);
      }
    } else if ((cur_node->ri->reg_score >= cur_node->le->reg_score) && (cur_node->ri->reg_score >= cur_node->r2->reg_score)) {
      libxsmm_generator_copy_input_args(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->ri, arg_id, arg_info, input_reg);
      if (cur_node->le->reg_score >= cur_node->r2->reg_score) {
        libxsmm_generator_copy_input_args(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->le, arg_id, arg_info, input_reg);
        libxsmm_generator_copy_input_args(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->r2, arg_id, arg_info, input_reg);
      } else {
        libxsmm_generator_copy_input_args(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->r2, arg_id, arg_info, input_reg);
        libxsmm_generator_copy_input_args(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->le, arg_id, arg_info, input_reg);
      }
    } else {
      libxsmm_generator_copy_input_args(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->r2, arg_id, arg_info, input_reg);
      if (cur_node->le->reg_score >= cur_node->ri->reg_score) {
        libxsmm_generator_copy_input_args(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->le, arg_id, arg_info, input_reg);
        libxsmm_generator_copy_input_args(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->ri, arg_id, arg_info, input_reg);
      } else {
        libxsmm_generator_copy_input_args(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->ri, arg_id, arg_info, input_reg);
        libxsmm_generator_copy_input_args(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, cur_node->le, arg_id, arg_info, input_reg);
      }
    }
  } else {
    /* This should not happen */
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_adjust_args_addr(libxsmm_generated_code*        io_generated_code,
    const libxsmm_meqn_descriptor       *i_mateqn_desc,
    libxsmm_matequation_gp_reg_mapping  *i_gp_reg_mapping,
    libxsmm_matequation_kernel_config   *i_micro_kernel_config,
    unsigned int                        i_adjust_instr,
    unsigned int                        i_adjust_amount,
    unsigned int                        i_adjust_type,
    libxsmm_matrix_eqn_arg              *arg_info) {

  unsigned int n_args = i_micro_kernel_config->n_args;
  unsigned int i;
  unsigned int adjust_val = 0;
  unsigned int output_tsize = LIBXSMM_TYPESIZE(LIBXSMM_GETENUM_OUT(i_mateqn_desc->datatype));
  unsigned int temp_reg = i_gp_reg_mapping->temp_reg;
  unsigned int temp_reg2 = i_gp_reg_mapping->temp_reg2;

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
        adjust_val = i_adjust_amount * tsize;
      } else {
        adjust_val = 0;
      }
    }
    if (adjust_val != 0) {
      if ( i < i_micro_kernel_config->n_avail_gpr ) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_micro_kernel_config->gpr_pool[i], adjust_val);
      } else {
        libxsmm_generator_meqn_getaddr_stack_tmpaddr_i( io_generated_code, i * 3 * 8, temp_reg);
        libxsmm_x86_instruction_alu_mem( io_generated_code,
            i_micro_kernel_config->alu_mov_instruction,
            temp_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            temp_reg2,
            0 );
        libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, temp_reg2, adjust_val);
        libxsmm_x86_instruction_alu_mem( io_generated_code,
            i_micro_kernel_config->alu_mov_instruction,
            temp_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            temp_reg2,
            1 );
      }
    }
  }

  /* Adjust output  */
  if (i_micro_kernel_config->is_head_reduce_to_scalar == 0) {
    if (i_adjust_type == M_ADJUSTMENT) {
      adjust_val = i_adjust_amount * output_tsize;
    } else if (i_adjust_type == N_ADJUSTMENT) {
      adjust_val = i_adjust_amount * i_mateqn_desc->ldo * output_tsize;
    }
    libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg_mapping->gp_reg_out, adjust_val);
  }
}

LIBXSMM_API_INTERN
void libxsmm_configure_mateqn_microkernel_loops( libxsmm_generated_code*                io_generated_code,
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

  unsigned int m_trips, n_trips, m_unroll_factor = 32, n_unroll_factor = 32, m_assm_trips = 1, n_assm_trips = 1;
  unsigned int max_nm_unrolling = 32;
  unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;
  unsigned int i_vlen_in = i_micro_kernel_config->vlen_in;
  unsigned int n_tmp_reg_blocks = i_eqn->eqn_root->reg_score;

  if (io_generated_code->arch < LIBXSMM_X86_AVX512) {
    m_unroll_factor = 16;
    n_unroll_factor = 16;
    max_nm_unrolling = 16;
  }

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
    printf("Can't nenerate run this code variant, ran out of zmm registers...\n");
  }
  if ((max_nm_unrolling < m_trips) && (i_use_m_input_masking == 1)) {
    printf("Can't generate run this code variant, ran out of zmm registers and we want to mask M...\n");
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
void libxsmm_meqn_setup_input_output_masks( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_matequation_kernel_config*      i_micro_kernel_config,
                                                 const libxsmm_meqn_descriptor*          i_meqn_desc,
                                                 unsigned int                            i_tmp_reg,
                                                 unsigned int                            i_m,
                                                 unsigned int*                           i_use_m_input_masking,
                                                 unsigned int*                           i_mask_reg_in,
                                                 unsigned int*                           i_use_m_output_masking,
                                                 unsigned int*                           i_mask_reg_out) {
  unsigned int mask_in_count, mask_out_count, mask_reg_in = 0, mask_reg_out = 0, use_m_input_masking, use_m_output_masking;
  unsigned int i_vlen_in = i_micro_kernel_config->vlen_in;
  unsigned int i_vlen_out = i_micro_kernel_config->vlen_out;
  unsigned int reserved_mask_regs = i_micro_kernel_config->reserved_mask_regs;
  LIBXSMM_UNUSED(i_meqn_desc);

  use_m_input_masking   = (i_m % i_vlen_in == 0 ) ? 0 : 1;
  use_m_output_masking  = (i_m % i_vlen_out == 0 ) ? 0 : 1;

  if (use_m_input_masking == 1) {
    if (io_generated_code->arch >= LIBXSMM_X86_AVX512) {
      libxsmm_datatype fake_dt;
      if (i_vlen_in == 64) {
        fake_dt = LIBXSMM_DATATYPE_I8;
      } else if(i_vlen_in == 32) {
        fake_dt = LIBXSMM_DATATYPE_BF16;
      } else {
        fake_dt = LIBXSMM_DATATYPE_F32;
      }
      mask_in_count = i_vlen_in - i_m % i_vlen_in;
      mask_reg_in   = reserved_mask_regs;
      libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, i_tmp_reg, mask_reg_in, mask_in_count, fake_dt);
      reserved_mask_regs++;
    } else {
      mask_reg_in = i_micro_kernel_config->inout_vreg_mask;
      libxsmm_generator_mateltwise_initialize_avx_mask(io_generated_code, mask_reg_in, i_m % i_vlen_in);
    }
  }

  if (use_m_output_masking == 1) {
    if (i_vlen_in == i_vlen_out) {
      mask_reg_out = mask_reg_in;
    } else {
      if (io_generated_code->arch >= LIBXSMM_X86_AVX512) {
        libxsmm_datatype fake_dt;
        if (i_vlen_out == 64) {
          fake_dt = LIBXSMM_DATATYPE_I8;
        } else if(i_vlen_out == 32) {
          fake_dt = LIBXSMM_DATATYPE_BF16;
        } else {
          fake_dt = LIBXSMM_DATATYPE_F32;
        }
        mask_out_count = i_vlen_out - i_m % i_vlen_out;
        mask_reg_out   = reserved_mask_regs;
        libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, i_tmp_reg, mask_reg_out, mask_out_count, fake_dt);
        reserved_mask_regs++;
      } else {
        mask_reg_out = i_micro_kernel_config->reserved_zmms;
        i_micro_kernel_config->reserved_zmms =  i_micro_kernel_config->reserved_zmms + 1;
        libxsmm_generator_mateltwise_initialize_avx_mask(io_generated_code, mask_reg_out, i_m % i_vlen_out);
      }
    }
  }

  i_micro_kernel_config->reserved_mask_regs = reserved_mask_regs;
  *i_mask_reg_in = mask_reg_in;
  *i_use_m_input_masking = use_m_input_masking;
  *i_mask_reg_out = mask_reg_out;
  *i_use_m_output_masking = use_m_output_masking;
}

LIBXSMM_API_INTERN
unsigned int vmove_instruction(libxsmm_datatype  dtype) {
  if ( LIBXSMM_DATATYPE_F64 == dtype ) {
    return  LIBXSMM_X86_INSTR_VMOVUPD;
  } else if ( (LIBXSMM_DATATYPE_F32 == dtype) || (LIBXSMM_DATATYPE_I32 == dtype) ) {
    return LIBXSMM_X86_INSTR_VMOVUPS;
  } else if ( (LIBXSMM_DATATYPE_BF16 == dtype) || (LIBXSMM_DATATYPE_I16 == dtype) ) {
    return LIBXSMM_X86_INSTR_VMOVDQU16;
  } else if ( LIBXSMM_DATATYPE_F16 == dtype ) {
    return LIBXSMM_X86_INSTR_VMOVDQU16;
  } else if ( LIBXSMM_DATATYPE_I8 == dtype ) {
    return LIBXSMM_X86_INSTR_VMOVDQU8;
  } else {
    return 0;
  }
}

LIBXSMM_API_INTERN
unsigned int vbcast_instruction(libxsmm_datatype  dtype) {
  if ( LIBXSMM_DATATYPE_F64 == dtype ) {
    return  LIBXSMM_X86_INSTR_VPBROADCASTQ;
  } else if ( (LIBXSMM_DATATYPE_F32 == dtype) || (LIBXSMM_DATATYPE_I32 == dtype)) {
    return LIBXSMM_X86_INSTR_VBROADCASTSS;
  } else if ( (LIBXSMM_DATATYPE_BF16 == dtype) || (LIBXSMM_DATATYPE_I16 == dtype)) {
    return LIBXSMM_X86_INSTR_VPBROADCASTW;
  } else if ( LIBXSMM_DATATYPE_F16 == dtype ) {
    return LIBXSMM_X86_INSTR_VPBROADCASTW;
  } else if ( LIBXSMM_DATATYPE_I8 == dtype ) {
    return LIBXSMM_X86_INSTR_VPBROADCASTB;
  } else {
    return 0;
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_load_arg_to_2d_reg_block( libxsmm_generated_code*          io_generated_code,
                                                 libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
                                                 libxsmm_matequation_kernel_config*      i_micro_kernel_config,
                                                 const libxsmm_meqn_descriptor*          i_meqn_desc,
                                                 unsigned int                            i_arg_id,
                                                 unsigned int                            i_vlen,
                                                 unsigned int                            i_reg_block_id,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking,
                                                 unsigned int                            i_mask_last_m_chunk,
                                                 unsigned int                            i_mask_reg ) {

  unsigned int in, im;
  unsigned int temp_reg = i_gp_reg_mapping->temp_reg;
  unsigned int temp_reg2 = i_gp_reg_mapping->temp_reg2;
  unsigned int cur_vreg;
  libxsmm_matrix_eqn_arg  *arg_info = i_micro_kernel_config->arg_info;
  unsigned int i_start_vreg = get_start_of_register_block(i_micro_kernel_config, i_reg_block_id);
  unsigned int input_reg = 0;
  char vname = (LIBXSMM_TYPESIZE(arg_info[i_arg_id].dtype) * i_vlen == 64) ? 'z' : 'y';
  LIBXSMM_UNUSED(i_meqn_desc);

  if (i_arg_id < i_micro_kernel_config->n_avail_gpr) {
    input_reg = i_micro_kernel_config->gpr_pool[i_arg_id];
  } else {
    libxsmm_generator_meqn_getaddr_stack_tmpaddr_i( io_generated_code, i_arg_id * 3 * 8, temp_reg2);
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        temp_reg2,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        0,
        temp_reg,
        0 );
    input_reg = temp_reg;
  }

  if (arg_info[i_arg_id].bcast_type == LIBXSMM_MATRIX_EQN_BCAST_TYPE_NONE) {
    for (in = 0; in < i_n_blocking; in++) {
      for (im = 0; im < i_m_blocking; im++) {
        cur_vreg = i_start_vreg + in * i_m_blocking + im;
        if ( arg_info[i_arg_id].dtype == LIBXSMM_DATATYPE_I16 ) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              io_generated_code->arch,
              LIBXSMM_X86_INSTR_VPMOVZXWD,
              input_reg,
              LIBXSMM_X86_GP_REG_UNDEF,
              0,
              (im * i_vlen + in * arg_info[i_arg_id].ld) * LIBXSMM_TYPESIZE(arg_info[i_arg_id].dtype),
              'z',
              cur_vreg,
              ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0,
              0, 0);

        } else {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
              vmove_instruction(arg_info[i_arg_id].dtype),
              input_reg,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (im * i_vlen + in * arg_info[i_arg_id].ld) * LIBXSMM_TYPESIZE(arg_info[i_arg_id].dtype),
              vname,
              cur_vreg, ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? 1 : 0, ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0, 0 );

          if ( arg_info[i_arg_id].dtype == LIBXSMM_DATATYPE_BF16 ) {
            libxsmm_generator_cvtbf16ps_avx512( io_generated_code, 'z', cur_vreg, cur_vreg );
          }
        }
      }
    }
  } else if (arg_info[i_arg_id].bcast_type == LIBXSMM_MATRIX_EQN_BCAST_TYPE_ROW) {
    for (in = 0; in < i_n_blocking; in++) {
      im = 0;
      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      libxsmm_x86_instruction_unified_vec_move( io_generated_code,
          vbcast_instruction(arg_info[i_arg_id].dtype),
          input_reg,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          in * LIBXSMM_TYPESIZE(arg_info[i_arg_id].dtype),
          vname,
          cur_vreg, 0, 0, 0 );
      if ( arg_info[i_arg_id].dtype == LIBXSMM_DATATYPE_BF16 ) {
        libxsmm_generator_cvtbf16ps_avx512( io_generated_code, 'z', cur_vreg, cur_vreg );
      }
      for (im = 1; im < i_m_blocking; im++) {
        char copy_vname = (io_generated_code->arch >= LIBXSMM_X86_AVX512) ? 'z' : 'y';
        cur_vreg = i_start_vreg + in * i_m_blocking + im;
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS, copy_vname, i_start_vreg + in * i_m_blocking, cur_vreg );
      }
    }
  } else if (arg_info[i_arg_id].bcast_type == LIBXSMM_MATRIX_EQN_BCAST_TYPE_COL) {
    in = 0;
    for (im = 0; im < i_m_blocking; im++) {
      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      libxsmm_x86_instruction_unified_vec_move( io_generated_code,
          vmove_instruction(arg_info[i_arg_id].dtype),
          input_reg,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * i_vlen + in * arg_info[i_arg_id].ld) * LIBXSMM_TYPESIZE(arg_info[i_arg_id].dtype),
          vname,
          cur_vreg, ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? 1 : 0, ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0, 0 );

      if ( arg_info[i_arg_id].dtype == LIBXSMM_DATATYPE_BF16 ) {
        libxsmm_generator_cvtbf16ps_avx512( io_generated_code, 'z', cur_vreg, cur_vreg );
      }
    }
    for (in = 1; in < i_n_blocking; in++) {
      for (im = 0; im < i_m_blocking; im++) {
        char copy_vname = (io_generated_code->arch >= LIBXSMM_X86_AVX512) ? 'z' : 'y';
        cur_vreg = i_start_vreg + in * i_m_blocking + im;
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS, copy_vname, i_start_vreg + im, cur_vreg );
      }
    }
  } else if (arg_info[i_arg_id].bcast_type == LIBXSMM_MATRIX_EQN_BCAST_TYPE_SCALAR) {
    im = 0;
    in = 0;
    cur_vreg = i_start_vreg + in * i_m_blocking + im;
    libxsmm_x86_instruction_unified_vec_move( io_generated_code,
        vbcast_instruction(arg_info[i_arg_id].dtype),
        input_reg,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        0,
        vname,
        cur_vreg, 0, 0, 0 );
    if ( arg_info[i_arg_id].dtype == LIBXSMM_DATATYPE_BF16 ) {
      libxsmm_generator_cvtbf16ps_avx512( io_generated_code, 'z', cur_vreg, cur_vreg );
    }
    for (in = 0; in < i_n_blocking; in++) {
      for (im = 0; im < i_m_blocking; im++) {
        char copy_vname = (io_generated_code->arch >= LIBXSMM_X86_AVX512) ? 'z' : 'y';
        cur_vreg = i_start_vreg + in * i_m_blocking + im;
        if (cur_vreg != i_start_vreg) {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS, copy_vname, i_start_vreg, cur_vreg );
        }
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_store_2d_reg_block( libxsmm_generated_code*          io_generated_code,
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
  unsigned int i_start_vreg = get_start_of_register_block(i_micro_kernel_config, i_reg_block_id);

  if (i_micro_kernel_config->is_head_reduce_to_scalar > 0) return;

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      cur_vreg = i_start_vreg + in * i_m_blocking + im;

      if (i_micro_kernel_config->cvt_result_to_bf16 == 1) {
        if (i_micro_kernel_config->use_fp32bf16_cvt_replacement == 1) {
          libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, 'z', cur_vreg, cur_vreg,
              i_micro_kernel_config->dcvt_zmm_aux0, i_micro_kernel_config->dcvt_zmm_aux1, i_micro_kernel_config->dcvt_mask_aux0, i_micro_kernel_config->dcvt_mask_aux1);
        } else {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, 'z', cur_vreg, cur_vreg );
        }
      }

      libxsmm_x86_instruction_unified_vec_move( io_generated_code,
          vmove_instruction((libxsmm_datatype)LIBXSMM_GETENUM_OUT(i_meqn_desc->datatype)),
          i_gp_reg_mapping->gp_reg_out,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * i_vlen + in * i_meqn_desc->ldo) * LIBXSMM_TYPESIZE(LIBXSMM_GETENUM_OUT(i_meqn_desc->datatype)),
          ((i_micro_kernel_config->cvt_result_to_bf16 == 1) || (io_generated_code->arch < LIBXSMM_X86_AVX512)) ? 'y' : 'z',
          cur_vreg, ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? 1 : 0, ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0, 1 );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_unpackstore_2d_reg_block( libxsmm_generated_code*          io_generated_code,
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
  unsigned int i_start_vreg = get_start_of_register_block(i_micro_kernel_config, i_reg_block_id);

  if (i_micro_kernel_config->is_head_reduce_to_scalar > 0) return;

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      libxsmm_x86_instruction_vec_move( io_generated_code,
          io_generated_code->arch,
          LIBXSMM_X86_INSTR_VPMOVDW,
          i_gp_reg_mapping->gp_reg_out,
          LIBXSMM_X86_GP_REG_UNDEF,
          0,
          (im * i_vlen + in * i_meqn_desc->ldo) * LIBXSMM_TYPESIZE(LIBXSMM_GETENUM_OUT(i_meqn_desc->datatype)),
          'z',
          cur_vreg,
          ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0,
          0, 1);
      libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSRAD_I, 'z', cur_vreg, cur_vreg, 16 );
      libxsmm_x86_instruction_vec_move( io_generated_code,
          io_generated_code->arch,
          LIBXSMM_X86_INSTR_VPMOVDW,
          i_gp_reg_mapping->gp_reg_out,
          i_gp_reg_mapping->gp_reg_offset,
          1,
          (im * i_vlen + in * i_meqn_desc->ldo) * LIBXSMM_TYPESIZE(LIBXSMM_GETENUM_OUT(i_meqn_desc->datatype)),
          'z',
          cur_vreg,
          ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0,
          0, 1);
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_store_reduce_to_scalar_output( libxsmm_generated_code*          io_generated_code,
                                                 libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
                                                 libxsmm_matequation_kernel_config*      i_micro_kernel_config,
                                                 const libxsmm_meqn_descriptor*          i_meqn_desc ) {
  if (io_generated_code->arch < LIBXSMM_X86_AVX512) {
    libxsmm_generator_hinstrps_avx( io_generated_code, LIBXSMM_X86_INSTR_VADDPS, i_micro_kernel_config->reduce_vreg, 14, 15);
  } else {
    libxsmm_generator_hinstrps_avx512( io_generated_code, LIBXSMM_X86_INSTR_VADDPS, i_micro_kernel_config->reduce_vreg, i_micro_kernel_config->reduce_vreg+1, i_micro_kernel_config->reduce_vreg+2);
  }

  if (i_micro_kernel_config->cvt_result_to_bf16 == 1) {
    if (i_micro_kernel_config->use_fp32bf16_cvt_replacement == 1) {
      libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, 'z', i_micro_kernel_config->reduce_vreg, i_micro_kernel_config->reduce_vreg,
          i_micro_kernel_config->dcvt_zmm_aux0, i_micro_kernel_config->dcvt_zmm_aux1, i_micro_kernel_config->dcvt_mask_aux0, i_micro_kernel_config->dcvt_mask_aux1);
    } else {
      libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, 'z', i_micro_kernel_config->reduce_vreg, i_micro_kernel_config->reduce_vreg );
    }
  }

  libxsmm_x86_instruction_unified_vec_move( io_generated_code,
      vmove_instruction((libxsmm_datatype)LIBXSMM_GETENUM_OUT(i_meqn_desc->datatype)),
      i_gp_reg_mapping->gp_reg_out,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      0,
      ((i_micro_kernel_config->cvt_result_to_bf16 == 1) || (io_generated_code->arch < LIBXSMM_X86_AVX512)) ? 'y' : 'z',
      i_micro_kernel_config->reduce_vreg, 1, i_micro_kernel_config->out_mask, 1 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_compute_unary_op_2d_reg_block( libxsmm_generated_code*     io_generated_code,
                                                 libxsmm_matequation_kernel_config*      i_meqn_micro_kernel_config,
                                                 libxsmm_meltw_unary_type                i_op_type,
                                                 unsigned int                            i_reg_block_id,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking ) {

  unsigned int i_start_vreg = get_start_of_register_block(i_meqn_micro_kernel_config, i_reg_block_id);
  unsigned int im, in, cur_vreg;
  libxsmm_mateltwise_kernel_config *i_micro_kernel_config = &(i_meqn_micro_kernel_config->meltw_kernel_config);

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_X2) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMULPS, i_micro_kernel_config->vector_name, cur_vreg, cur_vreg, cur_vreg );
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_TO_SCALAR_OP_ADD) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VADDPS, i_micro_kernel_config->vector_name, cur_vreg, i_meqn_micro_kernel_config->reduce_vreg, i_meqn_micro_kernel_config->reduce_vreg );
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_NEGATE) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VXORPS, i_micro_kernel_config->vector_name, cur_vreg, i_micro_kernel_config->neg_signs_vreg, cur_vreg );
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_INC) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VADDPS, i_micro_kernel_config->vector_name, cur_vreg, i_micro_kernel_config->vec_ones, cur_vreg );
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL) {
        if (io_generated_code->arch < LIBXSMM_X86_AVX512) {
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VRCPPS, 'y', cur_vreg, LIBXSMM_X86_VEC_REG_UNDEF, cur_vreg, 0, 0, 0, 0);
        } else {
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VRCP14PS, 'z', cur_vreg, LIBXSMM_X86_VEC_REG_UNDEF, cur_vreg, 0, 0, 0, 0);
        }
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT) {
        if (io_generated_code->arch < LIBXSMM_X86_AVX512) {
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VRSQRTPS, 'y', cur_vreg, LIBXSMM_X86_VEC_REG_UNDEF, cur_vreg, 0, 0, 0, 0);
        } else {
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VRSQRT14PS, 'z', cur_vreg, LIBXSMM_X86_VEC_REG_UNDEF, cur_vreg, 0, 0, 0, 0);
        }
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_SQRT) {
        if (io_generated_code->arch < LIBXSMM_X86_AVX512) {
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VRSQRTPS, 'y', cur_vreg, LIBXSMM_X86_VEC_REG_UNDEF, cur_vreg, 0, 0, 0, 0);
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VRCPPS, 'y', cur_vreg, LIBXSMM_X86_VEC_REG_UNDEF, cur_vreg, 0, 0, 0, 0);
        } else {
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VRSQRT14PS, 'z', cur_vreg, LIBXSMM_X86_VEC_REG_UNDEF, cur_vreg, 0, 0, 0, 0);
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VRCP14PS, 'z', cur_vreg, LIBXSMM_X86_VEC_REG_UNDEF, cur_vreg, 0, 0, 0, 0);
        }
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_EXP) {
        if (io_generated_code->arch < LIBXSMM_X86_AVX512) {
          libxsmm_generator_exp_ps_3dts_avx( io_generated_code,
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
              i_micro_kernel_config->vec_lo_bound );
        } else {
          libxsmm_generator_exp_ps_3dts_avx512( io_generated_code,
              cur_vreg,
              i_micro_kernel_config->vec_y,
              i_micro_kernel_config->vec_z,
              i_micro_kernel_config->vec_c0,
              i_micro_kernel_config->vec_c1,
              i_micro_kernel_config->vec_c2,
              i_micro_kernel_config->vec_c3,
              i_micro_kernel_config->vec_halves,
              i_micro_kernel_config->vec_log2e);
        }
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_TANH || i_op_type == LIBXSMM_MELTW_TYPE_UNARY_TANH_INV ) {
        libxsmm_generator_tanh_ps_rational_78_avx512( io_generated_code,
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
            i_micro_kernel_config->vec_neg_ones);

        if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_TANH_INV) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
              LIBXSMM_X86_INSTR_VFNMSUB213PS, 'z', i_micro_kernel_config->vec_neg_ones, cur_vreg, cur_vreg );
        }
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID || i_op_type == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV) {
        libxsmm_generator_sigmoid_ps_rational_78_avx512( io_generated_code,
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
            i_micro_kernel_config->vec_halves );

        if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
              LIBXSMM_X86_INSTR_VSUBPS, 'z', cur_vreg, i_micro_kernel_config->vec_ones, i_micro_kernel_config->vec_x2 );
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
              LIBXSMM_X86_INSTR_VMULPS, 'z', i_micro_kernel_config->vec_x2, cur_vreg, cur_vreg );
        }
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_GELU) {
        libxsmm_generator_gelu_ps_minimax3_avx512( io_generated_code,
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
            i_micro_kernel_config->vec_c2 );
      } else if (i_op_type == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV) {
        libxsmm_generator_gelu_inv_ps_minimax3_avx512( io_generated_code,
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
            i_micro_kernel_config->vec_c2 );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_compute_binary_op_2d_reg_block( libxsmm_generated_code*    io_generated_code,
                                                 libxsmm_matequation_kernel_config*      i_meqn_micro_kernel_config,
                                                 libxsmm_meltw_binary_type               i_op_type,
                                                 unsigned int                            i_left_reg_block_id,
                                                 unsigned int                            i_right_reg_block_id,
                                                 unsigned int                            i_dst_reg_block_id,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking ) {
  unsigned int i_left_start_vreg = get_start_of_register_block(i_meqn_micro_kernel_config, i_left_reg_block_id);
  unsigned int i_right_start_vreg = get_start_of_register_block(i_meqn_micro_kernel_config, i_right_reg_block_id);
  unsigned int i_dst_start_vreg = get_start_of_register_block(i_meqn_micro_kernel_config, i_dst_reg_block_id);
  unsigned int im, in, left_vreg, right_vreg, dst_vreg;
  unsigned int binary_op_instr = 0;

  switch (i_op_type) {
    case LIBXSMM_MELTW_TYPE_BINARY_ADD: {
      binary_op_instr = LIBXSMM_X86_INSTR_VADDPS;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_MUL: {
      binary_op_instr = LIBXSMM_X86_INSTR_VMULPS;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_SUB: {
      binary_op_instr = LIBXSMM_X86_INSTR_VSUBPS;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_DIV: {
      binary_op_instr = LIBXSMM_X86_INSTR_VDIVPS;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD: {
      binary_op_instr = LIBXSMM_X86_INSTR_VFMADD231PS;
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
      if (i_op_type == LIBXSMM_MELTW_TYPE_BINARY_PACK) {
        libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, 'z', right_vreg, right_vreg, 16 );
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPORD, 'z', right_vreg, left_vreg, dst_vreg );
      } else {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, binary_op_instr, i_meqn_micro_kernel_config->vector_name, right_vreg, left_vreg, dst_vreg );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_compute_ternary_op_2d_reg_block( libxsmm_generated_code*    io_generated_code,
                                                 libxsmm_matequation_kernel_config*      i_meqn_micro_kernel_config,
                                                 libxsmm_meltw_ternary_type              i_op_type,
                                                 unsigned int                            i_left_reg_block_id,
                                                 unsigned int                            i_right_reg_block_id,
                                                 unsigned int                            i_dst_reg_block_id,
                                                 unsigned int                            i_m_blocking,
                                                 unsigned int                            i_n_blocking ) {
  unsigned int i_left_start_vreg = get_start_of_register_block(i_meqn_micro_kernel_config, i_left_reg_block_id);
  unsigned int i_right_start_vreg = get_start_of_register_block(i_meqn_micro_kernel_config, i_right_reg_block_id);
  unsigned int i_dst_start_vreg = get_start_of_register_block(i_meqn_micro_kernel_config, i_dst_reg_block_id);
  unsigned int im, in, left_vreg, right_vreg, dst_vreg;
  unsigned int ternary_op_instr = 0;

  switch (i_op_type) {
    case LIBXSMM_MELTW_TYPE_TERNARY_MULADD: {
      ternary_op_instr = LIBXSMM_X86_INSTR_VFMADD231PS;
    } break;
    case LIBXSMM_MELTW_TYPE_TERNARY_NMULADD: {
      ternary_op_instr = LIBXSMM_X86_INSTR_VFNMADD213PS;
    } break;
    default:;
  }

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      left_vreg = i_left_start_vreg + in * i_m_blocking + im;
      right_vreg = i_right_start_vreg + in * i_m_blocking + im;
      dst_vreg = i_dst_start_vreg + in * i_m_blocking + im;
      libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, ternary_op_instr, i_meqn_micro_kernel_config->vector_name, right_vreg, left_vreg, dst_vreg );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateqn_2d_microkernel( libxsmm_generated_code*                    io_generated_code,
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
  libxsmm_meqn_setup_input_output_masks( io_generated_code, i_micro_kernel_config, i_meqn_desc,
      temp_reg, i_m, &use_m_input_masking, &mask_reg_in, &use_m_output_masking, &mask_reg_out);

  /* Configure microkernel loops */
  libxsmm_configure_mateqn_microkernel_loops( io_generated_code, i_micro_kernel_config, i_eqn, i_m, i_n, use_m_input_masking,
      &m_trips, &n_trips, &m_unroll_factor, &n_unroll_factor, &m_assm_trips, &n_assm_trips);

  i_micro_kernel_config->register_block_size = m_unroll_factor * n_unroll_factor;

  aux_reg_block = i_micro_kernel_config->n_tmp_reg_blocks - 1;
  aux_reg_block2 = i_micro_kernel_config->n_tmp_reg_blocks - 2;

  /* Headers of microkernel loops */
  if (n_assm_trips > 1) {
    libxsmm_generator_generic_loop_header(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, 0, n_unroll_factor);
  }

  if (m_assm_trips > 1) {
    libxsmm_generator_generic_loop_header(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, 0, m_unroll_factor);
  }

  /* Traverse equation tree based on optimal execution plan and emit code  */
  for (timestamp = 0; timestamp <= last_timestamp; timestamp++) {
    libxsmm_matrix_eqn_elem *cur_op = find_op_at_timestamp(i_eqn->eqn_root, timestamp);
    if (cur_op->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) {
      if (cur_op->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
        /* We have to load the input from the argument tensor using the node's assigned tmp reg block */
        left_reg_block = cur_op->tmp.id;
        libxsmm_generator_mateqn_load_arg_to_2d_reg_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
            arg_id, i_vlen_in, left_reg_block, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in );
        arg_id++;
      }
      libxsmm_generator_mateqn_compute_unary_op_2d_reg_block( io_generated_code, i_micro_kernel_config,
          cur_op->info.u_op.type, cur_op->tmp.id, m_unroll_factor, n_unroll_factor);
    } else if (cur_op->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) {
      if ((cur_op->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_op->ri->type == LIBXSMM_MATRIX_EQN_NODE_ARG)) {
        /* We have to load the input from the argument tensor using the node's assigned tmp reg block */
        left_reg_block = cur_op->tmp.id;
        libxsmm_generator_mateqn_load_arg_to_2d_reg_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
            arg_id, i_vlen_in, left_reg_block, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in );
        arg_id++;
        right_reg_block = aux_reg_block;
        libxsmm_generator_mateqn_load_arg_to_2d_reg_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
            arg_id, i_vlen_in, right_reg_block, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in );
        arg_id++;
      } else if ((cur_op->le->type != LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_op->ri->type != LIBXSMM_MATRIX_EQN_NODE_ARG)) {
        left_reg_block = cur_op->le->tmp.id;
        right_reg_block = cur_op->ri->tmp.id;
      } else {
        if (cur_op->ri->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
          /* We have to load the input from the argument tensor using the auxiliary tmp reg block */
          left_reg_block = cur_op->le->tmp.id;
          right_reg_block = aux_reg_block;
          libxsmm_generator_mateqn_load_arg_to_2d_reg_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
              arg_id, i_vlen_in, aux_reg_block, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in );
          arg_id++;
        } else {
          left_reg_block = aux_reg_block;
          right_reg_block = cur_op->ri->tmp.id;
          libxsmm_generator_mateqn_load_arg_to_2d_reg_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
              arg_id, i_vlen_in, aux_reg_block, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in );
          arg_id++;
        }
      }
      libxsmm_generator_mateqn_compute_binary_op_2d_reg_block( io_generated_code, i_micro_kernel_config,
          cur_op->info.b_op.type, left_reg_block, right_reg_block, cur_op->tmp.id, m_unroll_factor, n_unroll_factor);
    } else if (cur_op->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) {
      if (cur_op->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
        left_reg_block = aux_reg_block;
        libxsmm_generator_mateqn_load_arg_to_2d_reg_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
            arg_id, i_vlen_in, left_reg_block, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in );
        arg_id++;
      } else {
        left_reg_block = cur_op->le->tmp.id;
      }
      if (cur_op->ri->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
        right_reg_block = aux_reg_block2;
        libxsmm_generator_mateqn_load_arg_to_2d_reg_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
            arg_id, i_vlen_in, right_reg_block, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in );
        arg_id++;
      } else {
        right_reg_block = cur_op->ri->tmp.id;
      }
      if (cur_op->r2->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
        right2_reg_block = cur_op->tmp.id;
        libxsmm_generator_mateqn_load_arg_to_2d_reg_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
            arg_id, i_vlen_in, right2_reg_block, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in );
        arg_id++;
      } else {
        right2_reg_block = cur_op->r2->tmp.id;
      }
      libxsmm_generator_mateqn_compute_ternary_op_2d_reg_block( io_generated_code, i_micro_kernel_config,
          cur_op->info.t_op.type, left_reg_block, right_reg_block, right2_reg_block, m_unroll_factor, n_unroll_factor);
    } else {
      /* This should not happen */
    }
  }

  /* Store the computed register block to output  */
  if ((i_eqn->eqn_root->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (i_eqn->eqn_root->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_UNPACK_TO_BLOCKS)) {
    libxsmm_generator_mateqn_unpackstore_2d_reg_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
        i_vlen_out, i_eqn->eqn_root->tmp.id, m_unroll_factor, n_unroll_factor, use_m_output_masking, mask_reg_out );
  } else {
    libxsmm_generator_mateqn_store_2d_reg_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_meqn_desc,
        i_vlen_out, i_eqn->eqn_root->tmp.id, m_unroll_factor, n_unroll_factor, use_m_output_masking, mask_reg_out );
  }

  /* Footers of microkernel loops */
  if (m_assm_trips > 1) {
    libxsmm_generator_mateqn_adjust_args_addr(io_generated_code, i_meqn_desc, i_gp_reg_mapping, i_micro_kernel_config, i_micro_kernel_config->alu_add_instruction, m_unroll_factor * i_vlen_in , M_ADJUSTMENT, i_micro_kernel_config->arg_info);
    libxsmm_generator_generic_loop_footer(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_m_loop, m_trips);
    libxsmm_generator_mateqn_adjust_args_addr(io_generated_code, i_meqn_desc, i_gp_reg_mapping, i_micro_kernel_config, i_micro_kernel_config->alu_sub_instruction, m_unroll_factor * i_vlen_in * m_assm_trips, M_ADJUSTMENT, i_micro_kernel_config->arg_info);
  }

  if (n_assm_trips > 1) {
    libxsmm_generator_mateqn_adjust_args_addr(io_generated_code, i_meqn_desc, i_gp_reg_mapping, i_micro_kernel_config, i_micro_kernel_config->alu_add_instruction, n_unroll_factor, N_ADJUSTMENT, i_micro_kernel_config->arg_info);
    libxsmm_generator_generic_loop_footer(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_n_loop, n_trips);
    if (skip_n_loop_reg_cleanup == 0) {
      libxsmm_generator_mateqn_adjust_args_addr(io_generated_code, i_meqn_desc, i_gp_reg_mapping, i_micro_kernel_config, i_micro_kernel_config->alu_sub_instruction, n_unroll_factor * n_assm_trips, N_ADJUSTMENT, i_micro_kernel_config->arg_info);
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_configure_M_N_blocking( libxsmm_generated_code* io_generated_code, libxsmm_matequation_kernel_config* i_micro_kernel_config, libxsmm_matrix_eqn *i_eqn, unsigned int m, unsigned int n, unsigned int vlen, unsigned int *m_blocking, unsigned int *n_blocking) {
  /* The m blocking is done in chunks of vlen */
  unsigned int m_chunks = (m+vlen-1)/vlen;
  unsigned int m_chunk_remainder = 8;
  unsigned int m_range, m_block_size, foo1, foo2;
  unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;
  unsigned int n_tmp_reg_blocks = i_eqn->eqn_root->reg_score;
  unsigned int max_nm_unrolling = 32 - reserved_zmms;

  if (io_generated_code->arch < LIBXSMM_X86_AVX512) {
    max_nm_unrolling = 16 - reserved_zmms;
    m_chunk_remainder = 1;
  }

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
void libxsmm_generator_configure_equation_avx512_vlens(libxsmm_generated_code*    io_generated_code, libxsmm_matequation_kernel_config* i_micro_kernel_config, libxsmm_matrix_eqn *eqn)  {
  /* First, determine the vlen compute based on the min. compute of the equation */
  int tree_max_comp_tsize = eqn->eqn_root->tree_max_comp_tsize;
  if (tree_max_comp_tsize == 1) {
    i_micro_kernel_config->vlen_comp = 64;
  } else if (tree_max_comp_tsize == 2) {
    i_micro_kernel_config->vlen_comp = 32;
  } else if (tree_max_comp_tsize == 4) {
    i_micro_kernel_config->vlen_comp = 16;
  } else if (tree_max_comp_tsize == 8) {
    i_micro_kernel_config->vlen_comp = 8;
  }

  if (io_generated_code->arch < LIBXSMM_X86_AVX512) {
    i_micro_kernel_config->vlen_comp = i_micro_kernel_config->vlen_comp/2;
  }

  /* The vlen_in and vlen_out are aligned with the vlen compute */
  i_micro_kernel_config->vlen_in = i_micro_kernel_config->vlen_comp;
  i_micro_kernel_config->vlen_out = i_micro_kernel_config->vlen_comp;
}

LIBXSMM_API_INTERN
unsigned int unary_op_req_zmms(libxsmm_generated_code*    io_generated_code, libxsmm_meltw_unary_type u_type) {
  unsigned int result = 0;

  switch (u_type) {
    case LIBXSMM_MELTW_TYPE_UNARY_REDUCE_TO_SCALAR_OP_ADD: {
      result = 1;
    } break;
    case LIBXSMM_MELTW_TYPE_UNARY_XOR: {
      result = 1;
    } break;
    case LIBXSMM_MELTW_TYPE_UNARY_NEGATE: {
      result = 1;
    } break;
    case LIBXSMM_MELTW_TYPE_UNARY_INC: {
      result = 1;
    } break;
    case LIBXSMM_MELTW_TYPE_UNARY_GELU: {
      result = 14;
    } break;
    case LIBXSMM_MELTW_TYPE_UNARY_GELU_INV: {
      result = 14;
    } break;
    case LIBXSMM_MELTW_TYPE_UNARY_EXP: {
      if (io_generated_code->arch < LIBXSMM_X86_AVX512) {
        result = 9;
      } else {
        result = 8;
      }
    } break;
    case LIBXSMM_MELTW_TYPE_UNARY_TANH: {
      result = 14;
    } break;
    case LIBXSMM_MELTW_TYPE_UNARY_TANH_INV: {
      result = 14;
    } break;
    case LIBXSMM_MELTW_TYPE_UNARY_SIGMOID: {
      result = 15;
    } break;
    case LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV: {
      result = 15;
    } break;
    default:;
  }
  return result;
}

LIBXSMM_API_INTERN
unsigned int binary_op_req_zmms( libxsmm_generated_code*    io_generated_code, libxsmm_meltw_binary_type b_type) {
  unsigned int result = 0;
  LIBXSMM_UNUSED(io_generated_code);
  switch (b_type) {
    case LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD: {
      result = 1;
    } break;
    default:;
  }
  return result;
}

LIBXSMM_API_INTERN
void libxsmm_adjust_required_zmms( libxsmm_generated_code*    io_generated_code,  libxsmm_matequation_kernel_config* i_micro_kernel_config, libxsmm_meltw_unary_type u_type, libxsmm_meltw_binary_type b_type, unsigned int pool_id ) {
  unsigned int n_req_zmms = 0;
  if (pool_id == UNARY_OP_POOL) {
    if (i_micro_kernel_config->unary_ops_pool[u_type] == 0) {
      n_req_zmms = unary_op_req_zmms( io_generated_code,  u_type);
      i_micro_kernel_config->reserved_zmms += n_req_zmms;
      i_micro_kernel_config->unary_ops_pool[u_type] = 1;
    }
  } else if (pool_id == BINARY_OP_POOL) {
    if (i_micro_kernel_config->binary_ops_pool[b_type] == 0) {
      n_req_zmms = binary_op_req_zmms( io_generated_code,  b_type);
      i_micro_kernel_config->reserved_zmms += n_req_zmms;
      i_micro_kernel_config->binary_ops_pool[b_type] = 1;
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_mark_reserved_zmms( libxsmm_generated_code*    io_generated_code, libxsmm_matequation_kernel_config* i_micro_kernel_config, libxsmm_matrix_eqn_elem *cur_node ) {
  if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) {
    libxsmm_adjust_required_zmms( io_generated_code,  i_micro_kernel_config, cur_node->info.u_op.type, LIBXSMM_MELTW_TYPE_BINARY_NONE, UNARY_OP_POOL);
    libxsmm_mark_reserved_zmms( io_generated_code, i_micro_kernel_config, cur_node->le);
  } else if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) {
    libxsmm_adjust_required_zmms(io_generated_code, i_micro_kernel_config, LIBXSMM_MELTW_TYPE_UNARY_NONE, cur_node->info.b_op.type, BINARY_OP_POOL);
    libxsmm_mark_reserved_zmms(io_generated_code, i_micro_kernel_config, cur_node->le);
    libxsmm_mark_reserved_zmms(io_generated_code, i_micro_kernel_config, cur_node->ri);
  } else if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) {
    libxsmm_mark_reserved_zmms(io_generated_code, i_micro_kernel_config, cur_node->le);
    libxsmm_mark_reserved_zmms(io_generated_code, i_micro_kernel_config, cur_node->ri);
    libxsmm_mark_reserved_zmms(io_generated_code, i_micro_kernel_config, cur_node->r2);
  }
}

LIBXSMM_API_INTERN
void libxsmm_configure_reserved_zmms_and_masks(libxsmm_generated_code* io_generated_code,
    const libxsmm_meqn_descriptor*          i_mateqn_desc,
    libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
    libxsmm_matequation_kernel_config*      i_micro_kernel_config,
    libxsmm_matrix_eqn                      *eqn ) {
  unsigned int i = 0;
  libxsmm_mateltwise_kernel_config *meltw_config;
  libxsmm_datatype eqn_root_dtype = LIBXSMM_DATATYPE_F32;

  libxsmm_mark_reserved_zmms( io_generated_code, i_micro_kernel_config, eqn->eqn_root);
  i_micro_kernel_config->meltw_kernel_config.reserved_zmms = 0;
  i_micro_kernel_config->meltw_kernel_config.reserved_mask_regs = 1;
  meltw_config = (libxsmm_mateltwise_kernel_config*) &(i_micro_kernel_config->meltw_kernel_config);

  /* TODO: some diagnostic if we need excessive number of required zmms for the equation and bail out */
  for (i = 0 ; i < 64; i++) {
    if (i_micro_kernel_config->unary_ops_pool[i] > 0) {
      /* @TODO Evangelos: see the last to args... they are needed for dropout... */
      libxsmm_configure_unary_kernel_vregs_masks( io_generated_code, meltw_config, i, 0, i_gp_reg_mapping->temp_reg, LIBXSMM_X86_GP_REG_UNDEF, LIBXSMM_X86_GP_REG_UNDEF);
    }
  }

  i_micro_kernel_config->reserved_zmms = meltw_config->reserved_zmms;
  i_micro_kernel_config->reserved_mask_regs = meltw_config->reserved_mask_regs;

  /* Reserve in this case vreg mask*/
  if ((io_generated_code->arch < LIBXSMM_X86_AVX512) && (i_mateqn_desc->m % i_micro_kernel_config->vlen_in != 0)) {
    i_micro_kernel_config->inout_vreg_mask = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
  }

  /* Configure Reduce-to-scalar zmms and mask */
  i_micro_kernel_config->is_head_reduce_to_scalar = 0;
  for (i = 0 ; i < 64; i++) {
    if (((i_micro_kernel_config->unary_ops_pool[i] > 0) && (i == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_TO_SCALAR_OP_ADD)) ||
        ((i_micro_kernel_config->binary_ops_pool[i] > 0) && (i == LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD)) ) {
      unsigned int i_vlen_out = i_micro_kernel_config->vlen_out, mask_out_count = 0;
      i_micro_kernel_config->is_head_reduce_to_scalar = 1;
      i_micro_kernel_config->reduce_vreg = i_micro_kernel_config->reserved_zmms;
      i_micro_kernel_config->reserved_zmms += i_micro_kernel_config->reserved_zmms + 1;
      libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, i_micro_kernel_config->vector_name, i_micro_kernel_config->reduce_vreg, i_micro_kernel_config->reduce_vreg, i_micro_kernel_config->reduce_vreg );
      if (io_generated_code->arch >= LIBXSMM_X86_AVX512) {
        libxsmm_datatype fake_dt;
        /* Configure Reduce-to-scalar output_mask */
        if (i_vlen_out == 64) {
          fake_dt = LIBXSMM_DATATYPE_I8;
        } else if(i_vlen_out == 32) {
          fake_dt = LIBXSMM_DATATYPE_BF16;
        } else {
          fake_dt = LIBXSMM_DATATYPE_F32;
        }
        mask_out_count = i_vlen_out - 1;
        i_micro_kernel_config->out_mask = i_micro_kernel_config->reserved_mask_regs;
        libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, i_gp_reg_mapping->temp_reg, i_micro_kernel_config->out_mask, mask_out_count, fake_dt);
        i_micro_kernel_config->reserved_mask_regs += i_micro_kernel_config->reserved_mask_regs + 1;
      } else {
        i_micro_kernel_config->out_mask = i_micro_kernel_config->reserved_zmms;
        i_micro_kernel_config->reserved_zmms =  i_micro_kernel_config->reserved_zmms + 1;
        libxsmm_generator_mateltwise_initialize_avx_mask(io_generated_code, i_micro_kernel_config->out_mask, 1);
      }
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
    /* Should not happen  */
  }

  if ((eqn_root_dtype == LIBXSMM_DATATYPE_F32) && (LIBXSMM_GETENUM_OUT(i_mateqn_desc->datatype) == LIBXSMM_DATATYPE_BF16)) {
    i_micro_kernel_config->cvt_result_to_bf16 = 1;
    if (io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) {
      i_micro_kernel_config->use_fp32bf16_cvt_replacement = 1;
      libxsmm_generator_vcvtneps2bf16_avx512_prep_stack( io_generated_code, i_gp_reg_mapping->temp_reg );
      i_micro_kernel_config->dcvt_mask_aux0 = i_micro_kernel_config->reserved_mask_regs;
      i_micro_kernel_config->dcvt_mask_aux1 = i_micro_kernel_config->reserved_mask_regs + 1;
      i_micro_kernel_config->reserved_mask_regs = i_micro_kernel_config->reserved_mask_regs + 2;
      i_micro_kernel_config->dcvt_zmm_aux0 = i_micro_kernel_config->reserved_zmms;
      i_micro_kernel_config->dcvt_zmm_aux1 = i_micro_kernel_config->reserved_zmms + 1;
      i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 2;
    }
  } else {
    i_micro_kernel_config->cvt_result_to_bf16 = 0;
    i_micro_kernel_config->use_fp32bf16_cvt_replacement = 0;
  }
}

LIBXSMM_API_INTERN
void get_parent_bcast_info(libxsmm_matrix_eqn_elem* cur_node) {
  if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
    if ( cur_node->up->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
      cur_node->info.arg.bcast_type = get_bcast_type_unary( cur_node->up->info.u_op.flags );
    } else if ( cur_node->up->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
      if ( cur_node->up->le == cur_node ) {
        cur_node->info.arg.bcast_type = get_bcast_type_binary( cur_node->up->info.b_op.flags, LEFT );
      } else {
        cur_node->info.arg.bcast_type = get_bcast_type_binary( cur_node->up->info.b_op.flags, RIGHT );
      }
    } else if ( cur_node->up->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY ) {
      if ( cur_node->up->le == cur_node ) {
        cur_node->info.arg.bcast_type = get_bcast_type_ternary( cur_node->up->info.t_op.flags, LEFT );
      } else if ( cur_node->up->ri == cur_node ) {
        cur_node->info.arg.bcast_type = get_bcast_type_ternary( cur_node->up->info.t_op.flags, RIGHT );
      } else {
        cur_node->info.arg.bcast_type = get_bcast_type_ternary( cur_node->up->info.t_op.flags, RIGHT2 );
      }
    }
  } else {
    if ( cur_node->up->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
      cur_node->tmp.bcast_type = get_bcast_type_unary( cur_node->up->info.u_op.flags );
    } else if ( cur_node->up->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
      if ( cur_node->up->le == cur_node ) {
        cur_node->tmp.bcast_type = get_bcast_type_binary( cur_node->up->info.b_op.flags, LEFT );
      } else {
        cur_node->tmp.bcast_type = get_bcast_type_binary( cur_node->up->info.b_op.flags, RIGHT );
      }
    } else if ( cur_node->up->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY ) {
      if ( cur_node->up->le == cur_node ) {
        cur_node->tmp.bcast_type = get_bcast_type_ternary( cur_node->up->info.t_op.flags, LEFT );
      } else if ( cur_node->up->ri == cur_node ) {
        cur_node->tmp.bcast_type = get_bcast_type_ternary( cur_node->up->info.t_op.flags, RIGHT );
      } else {
        cur_node->tmp.bcast_type = get_bcast_type_ternary( cur_node->up->info.t_op.flags, RIGHT2 );
      }
    }
  }
}

LIBXSMM_API_INTERN
void assign_bcast_info(libxsmm_matrix_eqn_elem* cur_node) {
  get_parent_bcast_info(cur_node);
  if (cur_node->le != NULL) {
    assign_bcast_info(cur_node->le);
  }
  if (cur_node->ri != NULL) {
    assign_bcast_info(cur_node->ri);
  }
  if (cur_node->r2 != NULL) {
    assign_bcast_info(cur_node->r2);
  }
}

LIBXSMM_API_INTERN
void propagate_bcast_info( libxsmm_matrix_eqn *eqn ) {
  libxsmm_matrix_eqn_elem* root = eqn->eqn_root;
  if (root->le != NULL) {
    assign_bcast_info(root->le);
  }
  if (root->ri != NULL) {
    assign_bcast_info(root->ri);
  }
  if (root->r2 != NULL) {
    assign_bcast_info(root->r2);
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_tmp_register_block_avx_avx512_kernel( libxsmm_generated_code* io_generated_code,
    const libxsmm_meqn_descriptor*          i_mateqn_desc,
    libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
    libxsmm_matequation_kernel_config*      i_micro_kernel_config,
    libxsmm_loop_label_tracker*             io_loop_label_tracker,
    libxsmm_matrix_eqn*                     eqn ) {
  libxsmm_matrix_eqn_arg              *arg_info;
  unsigned int arg_id = 0, i = 0;
  unsigned int m_blocking = 0, n_blocking = 0, cur_n = 0, cur_m = 0, n_microkernel = 0, m_microkernel = 0, adjusted_aux_vars = 0;
  if ( eqn == NULL ) {
    fprintf( stderr, "The requested equation doesn't exist... nothing to JIT,,,\n" );
    return;
  }

  for (i = 0 ; i < 64; i++) {
    i_micro_kernel_config->unary_ops_pool[i] = 0;
    i_micro_kernel_config->binary_ops_pool[i] = 0;
  }

  /* Propagate bcast info in the tree */
  propagate_bcast_info(eqn);

  /* Iterate over the equation tree and copy the args ptrs in the auxiliary scratch */
  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      0,
      LIBXSMM_X86_GP_REG_R15,
      0 );

  arg_info = (libxsmm_matrix_eqn_arg*) malloc(i_micro_kernel_config->n_args * sizeof(libxsmm_matrix_eqn_arg));
  i_micro_kernel_config->contains_binary_op = 0;
  i_micro_kernel_config->contains_ternary_op = 0;
  libxsmm_generator_copy_input_args(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, eqn->eqn_root, &arg_id, arg_info, LIBXSMM_X86_GP_REG_R15);
  i_micro_kernel_config->arg_info = arg_info;

  /* Setup output reg */
  libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_X86_GP_REG_UNDEF, 0, 8, i_gp_reg_mapping->gp_reg_out, 0 );

  /* Configure equation vlens  */
  libxsmm_generator_configure_equation_avx512_vlens(io_generated_code, i_micro_kernel_config, eqn);

  /* Assign reserved zmms by parsing the equation */
  libxsmm_configure_reserved_zmms_and_masks(io_generated_code, i_mateqn_desc, i_gp_reg_mapping, i_micro_kernel_config, eqn );

  /* Configure M and N blocking factors */
  libxsmm_generator_matequation_configure_M_N_blocking(io_generated_code, i_micro_kernel_config, eqn, i_mateqn_desc->m, i_mateqn_desc->n, i_micro_kernel_config->vlen_in, &m_blocking, &n_blocking);

  cur_n = 0;
  while (cur_n != i_mateqn_desc->n) {
    cur_m = 0;
    adjusted_aux_vars = 0;
    n_microkernel = (cur_n < n_blocking) ? n_blocking : i_mateqn_desc->n - cur_n;
    while (cur_m != i_mateqn_desc->m) {
      unsigned int skip_n_loop_reg_cleanup = ((cur_n + n_microkernel == i_mateqn_desc->n) && (cur_m + m_microkernel == i_mateqn_desc->m)) ? 1 : 0 ;
      m_microkernel = (cur_m < m_blocking) ? m_blocking : i_mateqn_desc->m - cur_m;
      libxsmm_generator_mateqn_2d_microkernel(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_mateqn_desc, eqn, m_microkernel, n_microkernel, skip_n_loop_reg_cleanup);
      cur_m += m_microkernel;
      if (cur_m != i_mateqn_desc->m) {
        adjusted_aux_vars = 1;
        libxsmm_generator_mateqn_adjust_args_addr(io_generated_code, i_mateqn_desc, i_gp_reg_mapping, i_micro_kernel_config, i_micro_kernel_config->alu_add_instruction, m_microkernel, M_ADJUSTMENT, arg_info);
      }
    }
    if (adjusted_aux_vars == 1) {
      libxsmm_generator_mateqn_adjust_args_addr(io_generated_code, i_mateqn_desc, i_gp_reg_mapping, i_micro_kernel_config, i_micro_kernel_config->alu_sub_instruction, m_microkernel, M_ADJUSTMENT, arg_info);
    }
    cur_n += n_microkernel;
    if (cur_n != i_mateqn_desc->n) {
      libxsmm_generator_mateqn_adjust_args_addr(io_generated_code, i_mateqn_desc, i_gp_reg_mapping, i_micro_kernel_config, i_micro_kernel_config->alu_add_instruction, n_microkernel, N_ADJUSTMENT,  arg_info);
    }
  }

  /* Store to output the scalar result */
  if (i_micro_kernel_config->is_head_reduce_to_scalar > 0) {
    libxsmm_generator_mateqn_store_reduce_to_scalar_output( io_generated_code,  i_gp_reg_mapping, i_micro_kernel_config, i_mateqn_desc );
  }

  /* @TODO Evangelos: can you please the finalize kernel from mateltwise and how we can apply this here ?! */
  if (i_micro_kernel_config->use_fp32bf16_cvt_replacement == 1) {
    libxsmm_generator_vcvtneps2bf16_avx512_clean_stack( io_generated_code, LIBXSMM_X86_GP_REG_R15 );
  }
  /* Free aux data structure */
  free(arg_info);
}

