/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
*               Friedrich Schiller University Jena - All rights reserved.     *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Deepti Aggarwal, Alexander Heinecke (Intel Corp.), Antonio Noack (FSU Jena)
******************************************************************************/
#include "generator_aarch64_instructions.h"
#include "generator_mateltwise_aarch64.h"
#include "generator_common_aarch64.h"
#include "generator_common.h"
#include "generator_mateltwise_unary_binary_aarch64.h"
#include "generator_mateltwise_common.h"

#define MN_LOOP_ORDER 0
#define NM_LOOP_ORDER 1
#define LOOP_TYPE_M 0
#define LOOP_TYPE_N 1


LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_aarch64_gp_reg( libxsmm_generated_code*                 io_generated_code,
                                                   libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                   libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                   const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                   unsigned int                            i_gp_reg,
                                                   unsigned int                            i_adjust_instr,
                                                   unsigned int                            m_microkernel,
                                                   unsigned int                            n_microkernel,
                                                   unsigned int                            i_loop_type ) {
  unsigned int is_inp_gp_reg = ((i_gp_reg == i_gp_reg_mapping->gp_reg_in) || ((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY )) || ((i_gp_reg == i_gp_reg_mapping->gp_reg_in3) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY))) ? 1 : 0;
  unsigned int is_out_gp_reg = (i_gp_reg == i_gp_reg_mapping->gp_reg_out || ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP) && (i_gp_reg == i_gp_reg_mapping->gp_reg_out2))) ? 1 : 0;
  unsigned int bcast_row = (((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_0) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_1) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in3) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_2) > 0))) ? 1 : 0;
  unsigned int bcast_col = (((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in3) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2) > 0))) ? 1 : 0;
  unsigned int bcast_scalar = (((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR) > 0)) ||
                               ((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0) > 0)) ||
                               ((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_0) > 0)) ||
                               ((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1) > 0)) ||
                               ((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1) > 0)) ||
                               ((i_gp_reg == i_gp_reg_mapping->gp_reg_in3) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2) > 0))) ? 1 : 0;
  unsigned int bcast_input = ( bcast_row == 1 || bcast_col == 1 || bcast_scalar == 1 ) ? 1 : 0;

  if ((is_inp_gp_reg > 0) || (is_out_gp_reg > 0)) {
    unsigned int tsize  = (is_inp_gp_reg > 0) ? (((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY)) ? i_micro_kernel_config->datatype_size_in1 : i_micro_kernel_config->datatype_size_in) : i_micro_kernel_config->datatype_size_out;
    unsigned int ld     = (is_inp_gp_reg > 0) ? (((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY)) ? i_mateltwise_desc->ldi2 : i_mateltwise_desc->ldi) : i_mateltwise_desc->ldo;

    if ((is_inp_gp_reg > 0) && (i_gp_reg == i_gp_reg_mapping->gp_reg_in3) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY)) {
      tsize = i_micro_kernel_config->datatype_size_in2;
      ld = i_mateltwise_desc->ldi3;
      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_TERNARY_SELECT) {
        tsize = 1;
        m_microkernel = m_microkernel/8;
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BITMASK_2BYTEMULT) > 0) {
          ld = (LIBXSMM_UPDIV(ld, 16)*16)/8;
        } else {
          ld = ld/8;
        }
      }
    }

    if ((libxsmm_generator_mateltwise_is_binary_cmp_op(i_mateltwise_desc) > 0) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BITMASK_2BYTEMULT) > 0) && (is_out_gp_reg > 0)) {
      tsize = 1;
      m_microkernel = m_microkernel/8;
      ld = i_micro_kernel_config->ldo_mask;
    }

    if (bcast_input == 0) {
      if (i_loop_type == LOOP_TYPE_M) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)m_microkernel * tsize );
      } else {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)ld * n_microkernel * tsize);
      }
    } else {
      if (bcast_row > 0) {
        if (i_loop_type == LOOP_TYPE_N) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)ld * n_microkernel * tsize);
        }
      }
      if (bcast_col > 0) {
        if (i_loop_type == LOOP_TYPE_M) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)m_microkernel * tsize);
        }
      }
    }
  } else {
    /* Advance relumasks if need be */
    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY ) {
      if ( ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) )
           && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0)) {
        if (i_loop_type == LOOP_TYPE_M) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)m_microkernel/8);
        } else {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, ((long long)i_micro_kernel_config->ldo_mask * n_microkernel)/8);
        }
      }

      if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV)       ||
           (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
           (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV)           ) {
        if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && (i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)m_microkernel/8);
          } else {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, ((long long)i_micro_kernel_config->ldi_mask * n_microkernel)/8);
          }
        } else {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)m_microkernel * i_micro_kernel_config->datatype_size_in);
          } else {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)i_mateltwise_desc->ldi * n_microkernel * i_micro_kernel_config->datatype_size_in );
          }
        }
      }

      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0)) {
        if (i_loop_type == LOOP_TYPE_M) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)m_microkernel/8);
        } else {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, ((long long)i_micro_kernel_config->ldo_mask * n_microkernel)/8);
        }
      }

      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)m_microkernel/8);
          } else {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, ((long long)i_micro_kernel_config->ldi_mask * n_microkernel)/8);
          }
        } else {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BITMASK_REQUIRED );
          return;
        }
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( libxsmm_generated_code*                 io_generated_code,
                                                libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                unsigned int                            i_gp_reg,
                                                unsigned int                            i_adjust_instr,
                                                unsigned int                            i_adjust_param,
                                                unsigned int                            i_loop_type ) {
  unsigned int is_inp_gp_reg = ((i_gp_reg == i_gp_reg_mapping->gp_reg_in) || ((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY )) || ((i_gp_reg == i_gp_reg_mapping->gp_reg_in3) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY))) ? 1 : 0;
  unsigned int is_out_gp_reg = (i_gp_reg == i_gp_reg_mapping->gp_reg_out || ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP) && (i_gp_reg == i_gp_reg_mapping->gp_reg_out2))) ? 1 : 0;
  unsigned int bcast_row = (((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_0) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_1) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in3) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_2) > 0))) ? 1 : 0;
  unsigned int bcast_col = (((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in3) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2) > 0))) ? 1 : 0;
  unsigned int bcast_scalar = (((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR) > 0)) ||
                               ((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0) > 0)) ||
                               ((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_0) > 0)) ||
                               ((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1) > 0)) ||
                               ((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1) > 0)) ||
                               ((i_gp_reg == i_gp_reg_mapping->gp_reg_in3) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2) > 0))) ? 1 : 0;
  unsigned int bcast_input = ( bcast_row == 1 || bcast_col == 1 || bcast_scalar == 1 ) ? 1 : 0;

  if ((is_inp_gp_reg > 0) || (is_out_gp_reg > 0)) {
    unsigned int vlen   = (is_inp_gp_reg > 0) ? i_micro_kernel_config->vlen_in : i_micro_kernel_config->vlen_out;
    unsigned int tsize  = (is_inp_gp_reg > 0) ? (((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY)) ? i_micro_kernel_config->datatype_size_in1 : i_micro_kernel_config->datatype_size_in) : i_micro_kernel_config->datatype_size_out;
    unsigned int ld     = (is_inp_gp_reg > 0) ? (((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY)) ? i_mateltwise_desc->ldi2 : i_mateltwise_desc->ldi) : i_mateltwise_desc->ldo;

    long long m_adjust = (long long)vlen * i_adjust_param * tsize;
    long long n_adjust = (long long)ld * i_adjust_param * tsize;

    if ((is_inp_gp_reg > 0) && (i_gp_reg == i_gp_reg_mapping->gp_reg_in3) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY)) {
      tsize = i_micro_kernel_config->datatype_size_in2;
      ld = i_mateltwise_desc->ldi3;
      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_TERNARY_SELECT) {
        tsize = 1;
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BITMASK_2BYTEMULT) > 0) {
          ld = LIBXSMM_UPDIV(ld, 16)*16;
        }
        m_adjust = (long long)(vlen * i_adjust_param * tsize)/8;
        n_adjust = (long long)(ld * i_adjust_param * tsize)/8;
      }
    }

    if ((libxsmm_generator_mateltwise_is_binary_cmp_op(i_mateltwise_desc) > 0) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BITMASK_2BYTEMULT) > 0) && (is_out_gp_reg > 0)) {
      tsize = 1;
      ld = i_micro_kernel_config->ldo_mask;
      m_adjust = (long long)(vlen * i_adjust_param * tsize)/8;
      n_adjust = (long long)(ld * i_adjust_param * tsize)/8;
    }

    if (bcast_input == 0) {
      if (i_loop_type == LOOP_TYPE_M) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)m_adjust );
      } else {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)n_adjust );
      }
    } else {
      if (bcast_row > 0) {
        if (i_loop_type == LOOP_TYPE_N) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)n_adjust );
        }
      }
      if (bcast_col > 0) {
        if (i_loop_type == LOOP_TYPE_M) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)m_adjust );
        }
      }
    }
  } else {
    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY ) {
      if (((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU)) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0)) {
        /* TODO: Evangelos: why is here i_gp_reg_mapping->gp_reg_relumask used? */
        if (i_loop_type == LOOP_TYPE_M) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg_mapping->gp_reg_relumask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_relumask, ((long long)i_micro_kernel_config->vlen_out * i_adjust_param)/8 );
        } else {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg_mapping->gp_reg_relumask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_relumask, ((long long)i_micro_kernel_config->ldo_mask * i_adjust_param)/8 );
        }
      }

      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV)) {
        if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && (i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, ((long long)i_micro_kernel_config->vlen_in * i_adjust_param)/8);
          } else {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, ((long long)i_micro_kernel_config->ldi_mask * i_adjust_param)/8);
          }
        } else {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)i_micro_kernel_config->vlen_in * i_adjust_param * i_micro_kernel_config->datatype_size_in);
          } else {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)i_mateltwise_desc->ldi * i_adjust_param * i_micro_kernel_config->datatype_size_in );
          }
        }
      }

      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0)) {
        /* TODO: Evangelos: copied from ReLU.... */
        if (i_loop_type == LOOP_TYPE_M) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg_mapping->gp_reg_dropoutmask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_dropoutmask, ((long long)i_micro_kernel_config->vlen_out * i_adjust_param)/8 );
        } else {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg_mapping->gp_reg_dropoutmask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_dropoutmask, ((long long)i_micro_kernel_config->ldo_mask * i_adjust_param)/8 );
        }
      }

      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, ((long long)i_micro_kernel_config->vlen_in * i_adjust_param)/8);
          } else {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, ((long long)i_micro_kernel_config->ldi_mask * i_adjust_param)/8);
          }
        } else {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BITMASK_REQUIRED );
          return;
        }
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_configure_aarch64_vlens(const libxsmm_meltw_descriptor* i_mateltwise_desc, libxsmm_mateltwise_kernel_config* i_micro_kernel_config) {
  /* First, determine the vlen compute based on the architecture; there may be architectures with different widths for different types */
  /* At the moment, all types are assumed to be of the same length */
  unsigned int l_asimd_bytes_per_register = libxsmm_cpuid_vlen(i_micro_kernel_config->instruction_set);

  unsigned char l_inp_type = (LIBXSMM_DATATYPE_IMPLICIT == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)) ? LIBXSMM_CAST_UCHAR(libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) : LIBXSMM_CAST_UCHAR(libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP));
  unsigned int  l_inp_type_size = LIBXSMM_TYPESIZE(l_inp_type); /* like libxsmm_typesize; returns 0 if type is unknown */
  /* The vlen_out depends on the output datatype */
  unsigned char l_out_type = LIBXSMM_CAST_UCHAR(libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT));
  unsigned int  l_out_type_size = (libxsmm_generator_mateltwise_is_binary_cmp_op(i_mateltwise_desc) > 0) ? 1 : LIBXSMM_TYPESIZE(l_out_type);
  unsigned int  l_is_comp_zip = ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_ZIP)) ? 1 : 0;

  if (l_inp_type_size > 0) i_micro_kernel_config->vlen_comp = l_asimd_bytes_per_register / l_inp_type_size;
  if (l_out_type_size > 0) i_micro_kernel_config->vlen_out = l_asimd_bytes_per_register / l_out_type_size;

  if ( l_is_comp_zip ) {
    i_micro_kernel_config->vlen_comp = l_asimd_bytes_per_register / l_out_type_size;
  }

  /* The vlen_in is the same as vlen compute */
  i_micro_kernel_config->vlen_in = i_micro_kernel_config->vlen_comp;

  if ((libxsmm_generator_mateltwise_is_binary_cmp_op(i_mateltwise_desc) > 0)) {
    i_micro_kernel_config->vlen_out= i_micro_kernel_config->vlen_comp;
  }

  /* if the computation is done in F32 or the input is in F32, then set vlen_out to 16 */
  if ( LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ||
       LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ||
       LIBXSMM_DATATYPE_I8  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ||
       LIBXSMM_DATATYPE_U16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ||
       LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) {
    if ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) || LIBXSMM_DATATYPE_IMPLICIT == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ||
         LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) )
    {
      i_micro_kernel_config->vlen_out= i_micro_kernel_config->vlen_comp;
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_configure_aarch64_M_N_blocking( libxsmm_generated_code*         io_generated_code,
                                                       const libxsmm_meltw_descriptor* i_mateltwise_desc,
                                                       unsigned int m,
                                                       unsigned int n,
                                                       unsigned int vlen,
                                                       unsigned int *m_blocking,
                                                       unsigned int *n_blocking,
                                                       unsigned int available_vregs ) {
  /* The m blocking is done in chunks of vlen */
  unsigned int m_chunks = 0;
  /* TODO: Make m chunk remainder depend on number of available vector registers */
  unsigned int m_chunk_remainder = 4; /* how many registers are required for other "functions" */
  unsigned int m_chunk_boundary = 32;
  unsigned int m_range, m_block_size, foo1, foo2;
  unsigned int l_bitmask_2byte_mult = (((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) || ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BITMASK_2BYTEMULT) > 0) || ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BITMASK_2BYTEMULT) > 0)) ? 1 : 0;

  /* in order to work with bitmasks we need at least 8 entries, on ASIMD, that means 2 registers */
  /* TODO: for SVE, we maybe could use the predicate registers, so we do not need to half the count; however, we only have 15 predicate registers, so this still may be correct*/
  if ( (l_bitmask_2byte_mult > 0) && (vlen == 4) ) {
    vlen *= 2;
    if ( available_vregs < 3 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BITMASK_REQUIRED );
      return;
    }
    available_vregs /= 2;
    m_chunk_boundary /= 2;
  }

  m_chunks = LIBXSMM_UPDIV(m, vlen); /* (m+vlen-1)/vlen; */
  m_chunks &= ~1; /* round down to even numbers, (m_chunks/2)*2 */

  if (m % vlen == 0) {
    /* If there is not remainder in M, then we block M in order to limit block size */
    if (m_chunks > m_chunk_boundary) {
      libxsmm_compute_equalized_blocking(m_chunks, (m_chunks+1)/2, &m_range, &m_block_size, &foo1, &foo2);
      *m_blocking = m_range * vlen;
    } else {
      *m_blocking = m;
    }
  } else {
    /* If there is remainder we make sure we can fully unroll the kernel with masks */
    if (m_chunks > (m_chunk_boundary/2)) {
      if ( (l_bitmask_2byte_mult > 0) && (vlen == 4) ) {
        *m_blocking = (m_chunks - m_chunk_remainder) * vlen;
      } else {
        /* Find m_blocking that allows maximum unrolling in the "full vlen" kernel" */
        unsigned int res_m_blocking = 0;
        unsigned int m_achieved_unroll_factor = 0;
        unsigned int max_m_achieved_unroll_factor = 0;
        unsigned int im = 0;
        for (im = m_chunk_remainder; im < LIBXSMM_MIN(m_chunks, available_vregs); im++) {
          m_achieved_unroll_factor = available_vregs;
          while ((m_chunks - im) % m_achieved_unroll_factor != 0) {
            m_achieved_unroll_factor--;
          }
          if (m_achieved_unroll_factor > max_m_achieved_unroll_factor) {
            max_m_achieved_unroll_factor = m_achieved_unroll_factor;
            res_m_blocking = (m_chunks - im) * vlen;
          }
        }
        *m_blocking = res_m_blocking;
      }
    } else {
      if (available_vregs * vlen >= m) {
        *m_blocking = m;
      } else {
        if ( (l_bitmask_2byte_mult > 0) && (vlen == 4) ) {
          *m_blocking = (m_chunks - 1) * vlen;
        } else {
          /* Find m_blocking that allows maximum unrolling in the "full vlen" kernel" */
          unsigned int res_m_blocking = 0;
          unsigned int m_achieved_unroll_factor = 0;
          unsigned int max_m_achieved_unroll_factor = 0;
          unsigned int im = 0;
          for (im = 1; im < LIBXSMM_MIN(m_chunks, available_vregs); im++) {
            m_achieved_unroll_factor = available_vregs;
            while ((m_chunks - im) % m_achieved_unroll_factor != 0) {
              m_achieved_unroll_factor--;
            }
            if (m_achieved_unroll_factor > max_m_achieved_unroll_factor) {
              max_m_achieved_unroll_factor = m_achieved_unroll_factor;
              res_m_blocking = (m_chunks - im) * vlen;
            }
          }
          *m_blocking = res_m_blocking;
        }
      }
    }
  }
  /* For now not any additional blocking in N */
  *n_blocking = n;
}

LIBXSMM_API_INTERN
void libxsmm_generator_configure_aarch64_loop_order(const libxsmm_meltw_descriptor* i_mateltwise_desc, unsigned int *loop_order, unsigned int *m_blocking, unsigned int *n_blocking, unsigned int *out_blocking, unsigned int *inner_blocking, unsigned int *out_bound, unsigned int *inner_bound) {
  unsigned int _loop_order = NM_LOOP_ORDER;

  /* TODO: Potentially reorder loops given the kernel type */
  *loop_order = _loop_order;

#if 0
  if (_loop_order == NM_LOOP_ORDER) {
#endif
    *out_blocking = *n_blocking;
    *out_bound = i_mateltwise_desc->n;
    *inner_blocking = *m_blocking;
    *inner_bound = i_mateltwise_desc->m;
#if 0
  } else {
    *out_blocking = *m_blocking;
    *out_bound = i_mateltwise_desc->m;
    *inner_blocking = *n_blocking;
    *inner_bound = i_mateltwise_desc->n;
  }
#endif
}

LIBXSMM_API_INTERN
void libxsmm_load_aarch64_2d_reg_block( libxsmm_generated_code*                 io_generated_code,
                                        libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                        const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                        const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                        unsigned int                            i_vlen,
                                        unsigned int                            i_start_vreg,
                                        unsigned int                            i_m_blocking,
                                        unsigned int                            i_n_blocking,
                                        unsigned int                            i_mask_last_m_chunk,
                                        unsigned int                            i_mask_reg) {
  unsigned int im, in, cur_vreg;
  unsigned int bcast_row = (((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW) > 0)) ||
                            ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0) > 0)) ||
                            ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_0) > 0))) ? 1 : 0;
  unsigned int bcast_col = (((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL) > 0)) ||
                            ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0) > 0)) ||
                            ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0) > 0))) ? 1 : 0;
  unsigned int bcast_scalar = (((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR) > 0)) ||
                               ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0) > 0)) ||
                               ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_0) > 0))) ? 1 : 0;
  unsigned int bcast_input = ( bcast_row == 1 || bcast_col == 1 || bcast_scalar == 1 ) ? 1 : 0;
  unsigned int skip_scf_cvt = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_NO_SCF_QUANT) > 0) ? 1 : 0;
  unsigned int sign_sat = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_SIGN_SAT_QUANT) > 0) ? 1 : 0;
  unsigned int l_ld_bytes = i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in;
  unsigned int l_m_adjust = ( i_mask_last_m_chunk == 0 ) ? i_micro_kernel_config->datatype_size_in * i_vlen * i_m_blocking : i_micro_kernel_config->datatype_size_in * ( (i_vlen * (i_m_blocking-1)) + i_mask_last_m_chunk );
  unsigned int offset = 0;
  unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  unsigned int l_is_comp_unzip = ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_UNZIP)) ? 1 : 0;
  unsigned int l_is_comp_zip = ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_ZIP)) ? 1 : 0;
  libxsmm_aarch64_sve_type l_sve_type = (l_is_comp_unzip > 0 ) ? libxsmm_generator_aarch64_get_sve_type(LIBXSMM_TYPESIZE(libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)))
                                                               : (l_is_comp_zip) ? libxsmm_generator_aarch64_get_sve_type(LIBXSMM_TYPESIZE(libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)))
                                                               : libxsmm_generator_aarch64_get_sve_type(LIBXSMM_TYPESIZE(libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)));
  LIBXSMM_UNUSED(i_mask_reg);

  /* In this case we do not have to load any data */
  if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_XOR)) return;

  if ( l_is_sve ) {
    /* define predicate registers 0 and 1 for loading */
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, 0, -1, i_gp_reg_mapping->gp_reg_scratch_0 );
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, 1, i_mask_last_m_chunk * i_micro_kernel_config->datatype_size_in, i_gp_reg_mapping->gp_reg_scratch_0 );
    if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ||
        LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ||
        LIBXSMM_DATATYPE_U16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ||
        LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ||
        ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) &&
         (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1) ||
          LIBXSMM_DATATYPE_U16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1) ||
          LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1) )) ||
        ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) &&
         (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN2) ||
          LIBXSMM_DATATYPE_U16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN2) ||
          LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN2) ))) {
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, 2, i_vlen * i_micro_kernel_config->datatype_size_in, i_gp_reg_mapping->gp_reg_scratch_0 );
    }
  }

  if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_ZIP)) {
    if (l_is_sve > 0) {
      offset = (l_ld_bytes*i_n_blocking);
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, 2, i_vlen * i_micro_kernel_config->datatype_size_in, i_gp_reg_mapping->gp_reg_scratch_0 );
      for (in = 0; in < i_n_blocking; in++) {
        for (im = 0; im < i_m_blocking; im++) {
          unsigned int l_is_f32_or_f64 = (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) || LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ? 1 : 0;
          unsigned int l_masked_elements = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0
                                                             : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_mask_last_m_chunk : i_vlen : i_vlen;
          unsigned int l_mask_load = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? 1 : 0 : 0
                                                       : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? 1 : 2 : 2;
          cur_vreg = i_start_vreg + in * i_m_blocking + im;
          if ( bcast_input == 0) {
            libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_1, cur_vreg,
                                                                  i_micro_kernel_config->datatype_size_in, l_masked_elements, 1, 0, LIBXSMM_CAST_UCHAR(l_mask_load) );
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UUNPKLO_V, cur_vreg, LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, cur_vreg, 0, libxsmm_generator_aarch64_get_sve_type(4) );
          } else if ( (bcast_row == 1) || (bcast_scalar == 1) ) {
            offset = (bcast_scalar == 1) ?  i_micro_kernel_config->datatype_size_in:l_ld_bytes*i_n_blocking;
            if (im == 0) {
              if ((bcast_row == 1) || ((bcast_scalar == 1) && (in == 0))) {
                libxsmm_generator_bcastload_masked_vreg_aarch64_asimd( io_generated_code, i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_1, cur_vreg,
                                                                       i_micro_kernel_config->datatype_size_in, (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0, 1 );
                libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UUNPKLO_V,
                    cur_vreg, LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, cur_vreg, 0, libxsmm_generator_aarch64_get_sve_type(4) );
              } else if ((bcast_scalar == 1) && (in > 0) ) {
                libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                                         i_start_vreg, i_start_vreg, 0, cur_vreg, 0, l_sve_type);
              }
            } else if (im > 0) {
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                                       i_start_vreg + in * i_m_blocking, i_start_vreg + in * i_m_blocking, 0, cur_vreg, 0, l_sve_type);
            }
          } else if ( bcast_col == 1 ) {
            offset = ( i_mask_last_m_chunk == 0 ) ? i_micro_kernel_config->datatype_size_in * i_vlen * i_m_blocking : i_micro_kernel_config->datatype_size_in * ( (i_vlen * (i_m_blocking-1)) + i_mask_last_m_chunk );
            if (in == 0) {
              libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_1, cur_vreg,
                                                                  i_micro_kernel_config->datatype_size_in, l_masked_elements, 1, 0, LIBXSMM_CAST_UCHAR(l_mask_load) );
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UUNPKLO_V,
                  cur_vreg, LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, cur_vreg, 0, libxsmm_generator_aarch64_get_sve_type(4) );
            } else if (in > 0) {
              libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                                       i_start_vreg + im, i_start_vreg + im, 0, cur_vreg, 0, l_sve_type );
            }
          }
        }
        if ( bcast_input == 0 ) {
          if ( l_ld_bytes != l_m_adjust ) {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                          i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in,
                                                          ((long long)l_ld_bytes - l_m_adjust) );
          }
        } else {
          if (bcast_row == 1) {
            if ( l_ld_bytes != i_micro_kernel_config->datatype_size_in ) {
              libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                            i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in,
                                                            ((long long)l_ld_bytes - i_micro_kernel_config->datatype_size_in) );
            }
          }
        }
      }
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                  i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in,
                                                  offset );
    }
    return;
  }

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      unsigned int l_is_f32_or_f64 = (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) || LIBXSMM_DATATYPE_U32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) || LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) || LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ? 1 : 0;
      unsigned int l_masked_elements = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0
                                                         : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_mask_last_m_chunk : i_vlen : i_vlen;
      unsigned int l_mask_load = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? 1 : 0 : 0
                                                   : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? 1 : 2 : 2;
      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      if ( bcast_input == 0) {
        offset = (l_ld_bytes*i_n_blocking);
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_1, cur_vreg,
                                                                i_micro_kernel_config->datatype_size_in, l_masked_elements, 1, 0, LIBXSMM_CAST_UCHAR(l_mask_load) );
        /* If compute is in F32 and input is BF16 (or input is BF16 and output is F32), then upconvert BF16 -> FP32 */
        if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) && LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ||
             (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) && LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ) {
          libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, cur_vreg, 0);
        }
        /* dequantize tensor in case of DEQUANT TPP */
        if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DEQUANT)) {
          if ( ( LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) &&
               ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ) {
            libxsmm_generator_vcvt_i8f32_aarch64( io_generated_code, cur_vreg, i_micro_kernel_config->quant_vreg_scf,  0, skip_scf_cvt);
          }
        }
        if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_QUANT)) {
          if ( ( LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) &&
               ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) ) {
            libxsmm_generator_vcvt_f32i8_aarch64( io_generated_code, cur_vreg, i_micro_kernel_config->quant_vreg_scf,  0, skip_scf_cvt, sign_sat);
          }
        }

      } else {
        if ( (bcast_row == 1) || (bcast_scalar == 1) ) {
          offset = (bcast_scalar == 1) ?  i_micro_kernel_config->datatype_size_in:l_ld_bytes*i_n_blocking;
          if (im == 0) {
            if ((bcast_row == 1) || ((bcast_scalar == 1) && (in == 0))) {
              libxsmm_generator_bcastload_masked_vreg_aarch64_asimd( io_generated_code, i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_1, cur_vreg,
                                                                     i_micro_kernel_config->datatype_size_in, (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0, 1 );
              if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) && LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ||
                   (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) && LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ) {
                libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, cur_vreg, 0);
              }
              if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DEQUANT)) {
                if ( ( LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) &&
                     ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ) {
                  libxsmm_generator_vcvt_i8f32_aarch64( io_generated_code, cur_vreg, i_micro_kernel_config->quant_vreg_scf,  0, skip_scf_cvt);
                }
              }
            } else if ((bcast_scalar == 1) && (in > 0) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) ) {
              if (l_is_sve) {
                libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                                         i_start_vreg, i_start_vreg, 0, cur_vreg, 0, l_sve_type);
              } else {
                libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                                           i_start_vreg, i_start_vreg, 0, cur_vreg,
                                                           (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
              }
            }
          }

           if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY)) {
             /* Copy the register to the rest of the "M-registers" in this case.... */
             if (im > 0) {
               if (l_is_sve) {
                 libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                                          i_start_vreg + in * i_m_blocking, i_start_vreg + in * i_m_blocking, 0, cur_vreg, 0, l_sve_type);
               } else {
                 libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                                            i_start_vreg + in * i_m_blocking, i_start_vreg + in * i_m_blocking, 0, cur_vreg,
                                                            (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
               }
            }
          }

        }

        if ( bcast_col == 1 ) {
          offset = ( i_mask_last_m_chunk == 0 ) ? i_micro_kernel_config->datatype_size_in * i_vlen * i_m_blocking : i_micro_kernel_config->datatype_size_in * ( (i_vlen * (i_m_blocking-1)) + i_mask_last_m_chunk );
          if (in == 0) {
            libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_1, cur_vreg,
                                                                i_micro_kernel_config->datatype_size_in, l_masked_elements, 1, 0, LIBXSMM_CAST_UCHAR(l_mask_load) );
            if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) && LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ||
                 (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) && LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ) {
              libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, cur_vreg, 0);
            }
            /* dequantize tensor in case of DEQUANT TPP */
            if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DEQUANT)) {
              if ( ( LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) &&
                   ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ) {
                libxsmm_generator_vcvt_i8f32_aarch64( io_generated_code, cur_vreg, i_micro_kernel_config->quant_vreg_scf,  0, skip_scf_cvt);
              }
            }
          }
          if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY)) {
            /* Copy the register to the rest of the "N-REGISTERS" in this case.... */
            if (in > 0) {
              if (l_is_sve) {
                libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                                         i_start_vreg + im, i_start_vreg + im, 0, cur_vreg, 0, l_sve_type );
              } else {
                libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                                           i_start_vreg + im, i_start_vreg + im, 0, cur_vreg,
                                                           (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
              }
            }
          }
        }
      }
    }
    if ( bcast_input == 0 ) {
      if ( l_ld_bytes != l_m_adjust ) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                      i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in,
                                                      ((long long)l_ld_bytes - l_m_adjust) );
      }
    } else {
      if (bcast_row == 1) {
        if ( l_ld_bytes != i_micro_kernel_config->datatype_size_in ) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                        i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in,
                                                        ((long long)l_ld_bytes - i_micro_kernel_config->datatype_size_in) );
        }
      }
    }

  }
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                  i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in,
                                                  offset );
}

LIBXSMM_API_INTERN
void libxsmm_store_aarch64_2d_reg_block( libxsmm_generated_code*                 io_generated_code,
                                         libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                         unsigned int                            i_vlen,
                                         unsigned int                            i_start_vreg,
                                         unsigned int                            i_m_blocking,
                                         unsigned int                            i_n_blocking,
                                         unsigned int                            i_mask_last_m_chunk,
                                         unsigned int                            i_mask_reg) {
  unsigned int im, in, cur_vreg, cur_vreg_real;
  unsigned int bcast_row = (((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW) > 0))) ? 1 : 0;
  unsigned int bcast_col = (((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL) > 0))) ? 1 : 0;
  unsigned int bcast_scalar = (((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR) > 0))) ? 1 : 0;
  unsigned int l_ld_bytes = i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out;
  unsigned int l_m_adjust = ( i_mask_last_m_chunk == 0 ) ? i_micro_kernel_config->datatype_size_out * i_vlen * i_m_blocking : i_micro_kernel_config->datatype_size_out * ( (i_vlen * (i_m_blocking-1)) + i_mask_last_m_chunk );
  unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  LIBXSMM_UNUSED(i_mask_reg);

  if (libxsmm_generator_mateltwise_is_binary_cmp_op(i_mateltwise_desc) > 0) {
    /* In this case the store part is fused with the compute */
    return;
  }

  if ( l_is_sve ) {
    /* define predicate registers 0 and 1 for storing */
    /* TODO: implement predication without re-initializing the predicate registers all the time */
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, 0, -1, i_gp_reg_mapping->gp_reg_scratch_0 );
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, 1, i_mask_last_m_chunk * i_micro_kernel_config->datatype_size_out, i_gp_reg_mapping->gp_reg_scratch_0 );
    if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) || LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, 2, i_vlen * i_micro_kernel_config->datatype_size_out, i_gp_reg_mapping->gp_reg_scratch_0 );
    }
  }

  if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_UNZIP)) {
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, 2, i_vlen * i_micro_kernel_config->datatype_size_out, i_gp_reg_mapping->gp_reg_scratch_0 );
    for (in = 0; in < i_n_blocking; in++) {
      for (im = 0; im < i_m_blocking; im++) {
        unsigned int l_is_f32_or_f64 = (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) || LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ? 1 : 0;
        unsigned int l_masked_elements = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0
                                                           : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_mask_last_m_chunk : i_vlen : i_vlen;
        unsigned int l_mask_store = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? 1 : 0 : 0
                                                      : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? 1 : 2 : 2;
        cur_vreg_real = i_start_vreg + in * i_m_blocking + im;
        if (bcast_row == 1) {
          cur_vreg = i_start_vreg + in * i_m_blocking;
        } else if (bcast_scalar == 1) {
          cur_vreg = i_start_vreg;
        } else if (bcast_col == 1) {
          cur_vreg = i_start_vreg + im;
        } else {
          cur_vreg = cur_vreg_real;
        }
        if (((bcast_row == 0) && (bcast_col == 0) && (bcast_scalar == 0)) ||
            ((bcast_scalar == 1) && (in == 0) && (im == 0)) ||
            ((bcast_row == 1) && (im == 0)) ||
            (bcast_col == 1)) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UZP1_V, cur_vreg, cur_vreg, 0, i_micro_kernel_config->tmp_vreg, 0, libxsmm_generator_aarch64_get_sve_type(2) );
        }
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_1, i_micro_kernel_config->tmp_vreg,
                                                          i_micro_kernel_config->datatype_size_out, l_masked_elements, 1, 1, LIBXSMM_CAST_UCHAR(l_mask_store) );
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
        unsigned int l_is_f32_or_f64 = (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) || LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ? 1 : 0;
        unsigned int l_masked_elements = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0
                                                           : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_mask_last_m_chunk : i_vlen : i_vlen;
        unsigned int l_mask_store = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? 1 : 0 : 0
                                                      : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? 1 : 2 : 2;
        cur_vreg_real = i_start_vreg + in * i_m_blocking + im;
        if (bcast_row == 1) {
          cur_vreg = i_start_vreg + in * i_m_blocking;
        } else if (bcast_scalar == 1) {
          cur_vreg = i_start_vreg;
        } else if (bcast_col == 1) {
          cur_vreg = i_start_vreg + im;
        } else {
          cur_vreg = cur_vreg_real;
        }
        if (((bcast_row == 0) && (bcast_col == 0) && (bcast_scalar == 0)) ||
            ((bcast_scalar == 1) && (in == 0) && (im == 0)) ||
            ((bcast_row == 1) && (im == 0)) ||
            (bcast_col == 1)) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UZP2_V, cur_vreg, cur_vreg, 0, i_micro_kernel_config->tmp_vreg2, 0, libxsmm_generator_aarch64_get_sve_type(2) );
        }
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_1, i_micro_kernel_config->tmp_vreg2,
                                                          i_micro_kernel_config->datatype_size_out, l_masked_elements, 1, 1, LIBXSMM_CAST_UCHAR(l_mask_store) );
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
  } else if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X2) ||
                                                                                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3)    ) ) {
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                         i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_offset, i_gp_reg_mapping->gp_reg_shift_vals,
                                                         0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3 ) {
      libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_SR,
                                                           i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_offset_2, i_gp_reg_mapping->gp_reg_shift_vals2,
                                                           0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    }

    for (in = 0; in < i_n_blocking; in++) {
      for (im = 0; im < i_m_blocking; im++) {
        unsigned int l_is_f32_or_f64 = (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) || LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ? 1 : 0;
        unsigned int l_masked_elements = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0
                                                           : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_mask_last_m_chunk : i_vlen : i_vlen;
        unsigned int l_mask_store = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? 1 : 0 : 0
                                                      : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? 1 : 2 : 2;

        cur_vreg = i_start_vreg + in * i_m_blocking + im;
        if (bcast_row == 1) {
          cur_vreg = i_start_vreg + in * i_m_blocking;
        }
        if (bcast_scalar == 1) {
          cur_vreg = i_start_vreg;
        }
        if (bcast_col == 1) {
          cur_vreg = i_start_vreg + im;
        }
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_AND_V, cur_vreg, i_micro_kernel_config->mask_helper0_vreg, 0, i_micro_kernel_config->vec_tmp0, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FSUB_V, cur_vreg, i_micro_kernel_config->vec_tmp0, 0, i_micro_kernel_config->vec_tmp2, 0, LIBXSMM_AARCH64_SVE_TYPE_S );

  if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3 ) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_AND_V, i_micro_kernel_config->vec_tmp2, i_micro_kernel_config->mask_helper0_vreg, 0, i_micro_kernel_config->vec_tmp1, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FSUB_V, i_micro_kernel_config->vec_tmp2, i_micro_kernel_config->vec_tmp1, 0, i_micro_kernel_config->vec_tmp2, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
          libxsmm_generator_vcvt_f32bf16_aarch64_sve( io_generated_code, i_micro_kernel_config->vec_tmp2, 0);
        } else {
          libxsmm_generator_vcvt_f32bf16_aarch64_sve( io_generated_code, i_micro_kernel_config->vec_tmp2, 0);
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V, i_micro_kernel_config->vec_tmp2, i_micro_kernel_config->vec_tmp2, 0, i_micro_kernel_config->vec_tmp1, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
        }

  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UZP2_V, i_micro_kernel_config->vec_tmp0, i_micro_kernel_config->vec_tmp0, 0, i_micro_kernel_config->vec_tmp0, 0, LIBXSMM_AARCH64_SVE_TYPE_H );
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_1, i_micro_kernel_config->vec_tmp0,
                                                          i_micro_kernel_config->datatype_size_out, l_masked_elements, 1, 1, LIBXSMM_CAST_UCHAR(l_mask_store) );

  if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3 ) {
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UZP2_V, i_micro_kernel_config->vec_tmp1, i_micro_kernel_config->vec_tmp1, 0, i_micro_kernel_config->vec_tmp1, 0, LIBXSMM_AARCH64_SVE_TYPE_H );
          libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_shift_vals, i_gp_reg_mapping->gp_reg_scratch_1, i_micro_kernel_config->vec_tmp1,
                                                            i_micro_kernel_config->datatype_size_out, l_masked_elements, 1, 1, LIBXSMM_CAST_UCHAR(l_mask_store) );
          libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_shift_vals2, i_gp_reg_mapping->gp_reg_scratch_1, i_micro_kernel_config->vec_tmp2,
                                                            i_micro_kernel_config->datatype_size_out, l_masked_elements, 1, 1, LIBXSMM_CAST_UCHAR(l_mask_store) );
  } else {
          libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_shift_vals, i_gp_reg_mapping->gp_reg_scratch_1, i_micro_kernel_config->vec_tmp1,
                                                            i_micro_kernel_config->datatype_size_out, l_masked_elements, 1, 1, LIBXSMM_CAST_UCHAR(l_mask_store) );
   }
      }
      if ( l_ld_bytes != l_m_adjust ) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out,
                                                       ((long long)l_ld_bytes - l_m_adjust) );
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_mapping->gp_reg_shift_vals, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_shift_vals,
                                                       ((long long)l_ld_bytes - l_m_adjust) );
        if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3 ) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                         i_gp_reg_mapping->gp_reg_shift_vals2, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_shift_vals2,
                                                         ((long long)l_ld_bytes - l_m_adjust) );
  }
      }
    }
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out,
                                                   (long long)l_ld_bytes*i_n_blocking );
  } else {
    for (in = 0; in < i_n_blocking; in++) {
      for (im = 0; im < i_m_blocking; im++) {
        unsigned int l_is_f32_or_f64 = (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) || LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) || LIBXSMM_DATATYPE_U32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) || LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ? 1 : 0;
        unsigned int l_masked_elements = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0
                                                           : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_mask_last_m_chunk : i_vlen : i_vlen;
        unsigned int l_mask_store = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? 1 : 0 : 0
                                                      : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? 1 : 2 : 2;

        cur_vreg_real = i_start_vreg + in * i_m_blocking + im;
        cur_vreg = i_start_vreg + in * i_m_blocking + im;
        /* In the XOR case we have a constant vreg  */
        if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_XOR)) {
          cur_vreg = i_micro_kernel_config->zero_vreg;
        } else {
          if (bcast_row == 1) {
            cur_vreg = i_start_vreg + in * i_m_blocking;
          }
          if (bcast_scalar == 1) {
            cur_vreg = i_start_vreg;
          }
          if (bcast_col == 1) {
            cur_vreg = i_start_vreg + im;
          }
        }

         if ((cur_vreg == cur_vreg_real) &&
             ((LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) || LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) &&
               LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) )  {
          libxsmm_generator_vcvt_f32bf16_aarch64( io_generated_code, cur_vreg, 0);
        }
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_1, cur_vreg,
                                                                i_micro_kernel_config->datatype_size_out, l_masked_elements, 1, 1, LIBXSMM_CAST_UCHAR(l_mask_store) );

        if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP)) {
          libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_out2, i_gp_reg_mapping->gp_reg_scratch_1, cur_vreg,
                                                                  i_micro_kernel_config->datatype_size_out, l_masked_elements, 1, 1, LIBXSMM_CAST_UCHAR(l_mask_store) );
        }
      }
      if ( l_ld_bytes != l_m_adjust ) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out,
                                                       ((long long)l_ld_bytes - l_m_adjust) );
        if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP)) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                         i_gp_reg_mapping->gp_reg_out2, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out2,
                                                         ((long long)l_ld_bytes - l_m_adjust) );
        }
      }
    }
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out,
                                                   (long long)l_ld_bytes*i_n_blocking );
    if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP)) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                     i_gp_reg_mapping->gp_reg_out2, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out2,
                                                     (long long)l_ld_bytes*i_n_blocking );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_compute_unary_aarch64_2d_reg_block_op( libxsmm_generated_code*                 io_generated_code,
                                                    libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                    const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                    const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                    unsigned int                            i_vlen,
                                                    unsigned int                            i_start_vreg,
                                                    unsigned int                            i_m_blocking,
                                                    unsigned int                            i_n_blocking,
                                                    unsigned int                            i_mask_last_m_chunk,
                                                    unsigned int                            i_mask_reg) {
  unsigned int im, in, cur_vreg;
  unsigned int bcast_row = (((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW) > 0))) ? 1 : 0;
  unsigned int bcast_col = (((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL) > 0))) ? 1 : 0;
  unsigned int bcast_scalar = (((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR) > 0))) ? 1 : 0;
  unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  unsigned char l_pred_reg = LIBXSMM_CAST_UCHAR(i_mask_reg);
  unsigned char l_op_needs_predicates =
    i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_X2 &&
    i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL &&
    i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT;
  libxsmm_aarch64_sve_type l_sve_type = libxsmm_generator_aarch64_get_sve_type(LIBXSMM_TYPESIZE(libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)));
  libxsmm_aarch64_asimd_tupletype l_tupletype = (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D;

  LIBXSMM_UNUSED(i_gp_reg_mapping);
  LIBXSMM_UNUSED(i_vlen);
  LIBXSMM_UNUSED(i_mask_last_m_chunk);

  if (l_op_needs_predicates && l_is_sve) {
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, l_pred_reg, -1, 0 );
  }

  for (in = 0; in < i_n_blocking; in++) {
    if ((bcast_col == 1) && (in > 0)) {
      continue;
    }
    for (im = 0; im < i_m_blocking; im++) {
      if ((bcast_row == 1) && (im > 0)) {
        continue;
      }
      if ((bcast_scalar == 1) && ((im > 0) || (in > 0))) {
        continue;
      }

      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_X2) {
        if ( l_is_sve ) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                                   cur_vreg, cur_vreg, 0, cur_vreg, l_pred_reg, l_sve_type );
        } else {/* ASIMD */
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                                     cur_vreg, cur_vreg, 0, cur_vreg,
                                                     l_tupletype );
        }
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_NEGATE) {
        if ( l_is_sve ) {
          /* fneg only exists predicated */
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FNEG_V_P,
                                                   cur_vreg, cur_vreg, 0, cur_vreg, l_pred_reg, l_sve_type );
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FNEG_V,
                                                     cur_vreg, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, cur_vreg,
                                                     l_tupletype );
        }
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_INC) {
        if ( l_is_sve ) {
          /* using the immediate-add instruction is 7/6x slower on 64x64 elements on A64FX than using a simple add function with a constant */
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FADD_V,
                                                   cur_vreg, i_micro_kernel_config->vec_ones, 0, cur_vreg, l_pred_reg, l_sve_type );
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V,
                                                     cur_vreg, i_micro_kernel_config->vec_ones, 0, cur_vreg,
                                                     l_tupletype );
        }
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL) {
        if ( l_is_sve ) {
          /* can we improve the performance by using multiple temporary registers? no,still 2.60x faster than ASIMD on 64x64 */
          /* one iteration step is close to perfect with 1/[1..50] */
          if (libxsmm_get_ulp_precision() != LIBXSMM_ULP_PRECISION_ESTIMATE) {
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
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT) {
        /* typical relative error in tests (iterations = 0, fp32): 5-8% */
        /* typical relative error in tests (iterations = 1, fp32): 0.06% */
        /* typical relative error in tests (iterations = 2, fp32): 0.001% */
        /* typical relative error in tests (iterations = 3, fp32): 0.00002% */
        /* typical relative error in tests (iterations = 4, fp32): 0.0002% */

        /* number needs to be adjusted, if the type is fp64 or bf16 */
        /* fp32 is type 0x02; number of iterations for bytes: 0, bf16: 1, fp32: 3, fp64: 7 */
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
          for ( i=0; i<num_iterations; i++) {
            unsigned char dst_reg = LIBXSMM_CAST_UCHAR(i == (num_iterations-1) ? cur_vreg : tmp_guess); /* improve the guess; then save it */
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                                     tmp_guess, tmp_guess, 0, tmp_guess_squared, l_pred_reg, l_sve_type);
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FRSQRTS_V, /* dst = (3-s0*s1)/2 */
                                                     cur_vreg, tmp_guess_squared, 0, tmp_guess_squared, l_pred_reg, l_sve_type);
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                                     tmp_guess, tmp_guess_squared, 0, dst_reg, l_pred_reg, l_sve_type);
          }
        } else {
          /* todo: this only is an estimate as well, apply Newton iterations to improve the results */
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FRSQRTE_V,
                                                     cur_vreg, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, cur_vreg,
                                                     l_tupletype );
        }
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_SQRT) {
        if ( l_is_sve ) {
          /* the SQRT instruction is very slow on A64FX, only as fast as ASIMD, so maybe even serial performance */
          /* LIBXSMM is a machine learning oriented library and instructions like 1/x are inexact, so let's make this inexact as well */
          if (libxsmm_get_ulp_precision() != LIBXSMM_ULP_PRECISION_ESTIMATE) {
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
            for (i=0;i<num_iterations;i++) {
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
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_EXP) {
        if (l_is_sve) {
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
            l_sve_type, l_pred_reg );
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
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TANH || i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TANH_INV) {
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
            l_sve_type, l_pred_reg );
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

        if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TANH_INV) {/* 1st derivative of tanh(x) = 1-tanh(x)^2 */
          if (l_is_sve) {
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
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID || i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV) {
        if (l_is_sve) {
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
            l_sve_type, l_pred_reg );
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

        if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV) {
          if (l_is_sve) {
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
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GELU) {
        if (l_is_sve) {
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
                                                          i_micro_kernel_config->vec_tmp0,
                                                          i_micro_kernel_config->vec_tmp1,
                                                          l_sve_type,
                                                          l_pred_reg );
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
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV) {
        if (l_is_sve) {
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
                                                              i_micro_kernel_config->vec_tmp0,
                                                              i_micro_kernel_config->vec_tmp1,
                                                              l_sve_type,
                                                              l_pred_reg );
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
void libxsmm_compute_unary_aarch64_2d_reg_block_relu( libxsmm_generated_code*                 io_generated_code,
                                                      libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                      unsigned int                            i_vlen,
                                                      unsigned int                            i_start_vreg,
                                                      unsigned int                            i_m_blocking,
                                                      unsigned int                            i_n_blocking,
                                                      unsigned int                            i_mask_last_m_chunk,
                                                      unsigned int                            i_mask_reg) {
  unsigned int im, in;
  unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  LIBXSMM_UNUSED(i_mask_last_m_chunk);
  LIBXSMM_UNUSED(i_vlen);

  if ( l_is_sve ) {
    unsigned char l_pred_reg = LIBXSMM_CAST_UCHAR(i_mask_reg);
    unsigned char l_blend_reg = 7; /* sve predicate register for blending; todo should be a function input / part of the config */
    unsigned char l_tmp_pred_reg0 = 6; /* tmp sve predicate register for blending; todo should be a function input / part of the config */
    unsigned char l_tmp_pred_reg1 = 5;
    libxsmm_aarch64_sve_type l_sve_type = libxsmm_generator_aarch64_get_sve_type(LIBXSMM_TYPESIZE(libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)));
    unsigned int l_bf16_compute = ( LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) ? 1 : 0;

    if (l_bf16_compute > 0) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }

    for (in = 0; in < i_n_blocking; in++) {
      unsigned int l_mask_adv = 0;
      for (im = 0; im < i_m_blocking; im++) {
        /*unsigned int l_vlen = ( l_bf16_compute > 0 ) ? 32 : 16;*/
        unsigned int cur_vreg = i_start_vreg + in * i_m_blocking + im;

        if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) ) {
          if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) ) {
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FCMGT_Z_V,
                                                     cur_vreg, LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, l_blend_reg,
                                                     l_pred_reg, l_sve_type );
          }

          if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) {
            libxsmm_generator_unary_binary_aarch64_store_bitmask_2bytemult_sve( io_generated_code, i_mateltwise_desc->m, im, i_m_blocking,
                                                                                LIBXSMM_CAST_UCHAR(i_micro_kernel_config->tmp_vreg),
                                                                                LIBXSMM_CAST_UCHAR(i_gp_reg_mapping->gp_reg_relumask),
                                                                                l_blend_reg, l_tmp_pred_reg0, l_tmp_pred_reg1,
                                                                                LIBXSMM_CAST_UCHAR(i_gp_reg_mapping->gp_reg_scratch_0),
                                                                                &l_mask_adv );
          }

          if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU ) {
            /* ReLU */
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMAX_V_P, /* only exists predicated */
                                                     cur_vreg, i_micro_kernel_config->zero_vreg, 0, cur_vreg,
                                                     l_pred_reg, l_sve_type );
          } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU  ) {
            /* we need to multiply with alpha */
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                                     cur_vreg,  i_micro_kernel_config->fam_lu_vreg_alpha, 0, i_micro_kernel_config->tmp_vreg2,
                                                     l_pred_reg, l_sve_type);

            /* now we blend both together */
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SEL_V_P,
                                                     cur_vreg, i_micro_kernel_config->tmp_vreg2, 0, cur_vreg,
                                                     l_blend_reg, l_sve_type );
          } else {
            /* should not happen */
          }
        } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU )  {
          /* Compute exp */
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                                   cur_vreg, cur_vreg, 0, i_micro_kernel_config->tmp_vreg2,
                                                   l_pred_reg, l_sve_type );
          libxsmm_generator_exp_ps_3dts_aarch64_sve( io_generated_code,
              i_micro_kernel_config->tmp_vreg2, /* x */
              i_micro_kernel_config->tmp_vreg,  /* y */
              i_micro_kernel_config->tmp_vreg3, /* z */
              i_micro_kernel_config->vec_c0,
              i_micro_kernel_config->vec_c1,
              i_micro_kernel_config->vec_c2,
              i_micro_kernel_config->vec_c3,
              i_micro_kernel_config->vec_halves,
              i_micro_kernel_config->vec_log2e,
              i_micro_kernel_config->vec_expmask,
              i_micro_kernel_config->vec_hi_bound,
              i_micro_kernel_config->vec_lo_bound,
              l_sve_type, l_pred_reg );

          /* compute mask */
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FCMGT_Z_V,
                                                   cur_vreg, LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, l_blend_reg,
                                                   l_pred_reg, l_sve_type );

          /* compute ELU for < 0 */
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                                   i_micro_kernel_config->tmp_vreg2, i_micro_kernel_config->fam_lu_vreg_alpha, 0, i_micro_kernel_config->tmp_vreg2,
                                                   l_pred_reg, l_sve_type );
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FSUB_V,
                                                   i_micro_kernel_config->tmp_vreg2, i_micro_kernel_config->fam_lu_vreg_alpha, 0, i_micro_kernel_config->tmp_vreg2,
                                                   l_pred_reg, l_sve_type );

          /* blend exp-fma result with input reg based on elu mask */
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SEL_V_P,
                                                   cur_vreg, i_micro_kernel_config->tmp_vreg2, 0, cur_vreg,
                                                   l_blend_reg, l_sve_type );
        } else {
          /* should not happen */
        }
      }
      if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && (i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_ELU) ) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_mapping->gp_reg_relumask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_relumask,
                                                       ((long long)i_micro_kernel_config->ldo_mask - ((long long)l_mask_adv*8))/8 );
      }
    }
    if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && (i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_ELU) ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                     i_gp_reg_mapping->gp_reg_relumask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_relumask,
                                                     (long long)i_n_blocking * ((long long)i_micro_kernel_config->ldo_mask/8) );
    }

  } else {

    libxsmm_aarch64_asimd_tupletype l_tupletype = (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D;

    for (in = 0; in < i_n_blocking; in++) {
      unsigned int l_mask_adv = 0;
      for (im = 0; im < i_m_blocking; im++) {
        /* unsigned int l_vlen = ( l_bf16_compute > 0 ) ? 32 : 16; removed because bf16 is not supported for ASIMD */
        unsigned int cur_vreg = i_start_vreg + in * i_m_blocking + im;

        if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) ) {
          if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) ) {
            /* tmp_vreg = (cur_vreg > 0) */
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FCMGT_Z_V,
                                                       cur_vreg, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, i_micro_kernel_config->tmp_vreg,
                                                       LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
          }

          if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) {
            libxsmm_generator_unary_binary_aarch64_store_bitmask_2bytemult_asimd( io_generated_code, im, i_m_blocking,
                                                                                  LIBXSMM_CAST_UCHAR(i_micro_kernel_config->mask_helper0_vreg),
                                                                                  LIBXSMM_CAST_UCHAR(i_micro_kernel_config->mask_helper1_vreg),
                                                                                  LIBXSMM_CAST_UCHAR(i_micro_kernel_config->tmp_vreg),
                                                                                  LIBXSMM_CAST_UCHAR(i_micro_kernel_config->tmp_vreg2),
                                                                                  LIBXSMM_CAST_UCHAR(i_micro_kernel_config->tmp_vreg3),
                                                                                  LIBXSMM_CAST_UCHAR(i_gp_reg_mapping->gp_reg_relumask),
                                                                                  &l_mask_adv );
          }

          if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU ) {
            /* ReLU */
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMAX_V, cur_vreg, i_micro_kernel_config->zero_vreg, 0, cur_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
          } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU  ) {
            /* we need to multiply with alpha */
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V, cur_vreg,  i_micro_kernel_config->fam_lu_vreg_alpha, 0, i_micro_kernel_config->tmp_vreg2, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

            /* now we blend both together */
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_BIF_V, i_micro_kernel_config->tmp_vreg2, i_micro_kernel_config->tmp_vreg, 0, cur_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
          } else {
            /* should not happen */
          }
        } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU )  {
          /* compute exp */
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V, cur_vreg, cur_vreg, 0, i_micro_kernel_config->tmp_vreg2, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
          libxsmm_generator_exp_ps_3dts_aarch64_asimd( io_generated_code,
              i_micro_kernel_config->tmp_vreg2,
              i_micro_kernel_config->tmp_vreg,
              i_micro_kernel_config->tmp_vreg3,
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

          /* compute mask */
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FCMGT_Z_V, cur_vreg, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, i_micro_kernel_config->tmp_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

          /* compute ELU for < 0 */
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                                     i_micro_kernel_config->tmp_vreg2, i_micro_kernel_config->fam_lu_vreg_alpha, 0, i_micro_kernel_config->tmp_vreg2,
                                                     LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FSUB_V,
                                                     i_micro_kernel_config->tmp_vreg2, i_micro_kernel_config->fam_lu_vreg_alpha, 0, i_micro_kernel_config->tmp_vreg2,
                                                     LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );

          /* blend exp-fma result with input reg based on elu mask */
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_BIF_V,
                                                     i_micro_kernel_config->tmp_vreg2, i_micro_kernel_config->tmp_vreg, 0, cur_vreg,
                                                     LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
        } else {
          /* should not happen */
        }
      }
      if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && (i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_ELU) ) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_mapping->gp_reg_relumask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_relumask,
                                                       ((long long)i_micro_kernel_config->ldo_mask - ((long long)l_mask_adv*8))/8 );
      }
    }
    if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && (i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_ELU) ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                     i_gp_reg_mapping->gp_reg_relumask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_relumask,
                                                     (long long)i_n_blocking * ((long long)i_micro_kernel_config->ldo_mask/8) );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_compute_unary_aarch64_2d_reg_block_relu_inv( libxsmm_generated_code*                 io_generated_code,
                                                          libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                          const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                          const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                          unsigned int                            i_vlen,
                                                          unsigned int                            i_start_vreg,
                                                          unsigned int                            i_m_blocking,
                                                          unsigned int                            i_n_blocking,
                                                          unsigned int                            i_mask_last_m_chunk,
                                                          unsigned int                            i_mask_reg) {
  unsigned int im, in;
  unsigned int l_ld_bytes = i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in;
  unsigned int l_m_adjust = ( i_mask_last_m_chunk == 0 ) ? i_micro_kernel_config->datatype_size_in * i_vlen * i_m_blocking : i_micro_kernel_config->datatype_size_in * ( (i_vlen * (i_m_blocking-1)) + i_mask_last_m_chunk );
  unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);

  if ( l_is_sve ) {
    unsigned char l_tmp_pred_reg = 6;
    unsigned char l_blend_reg = 7; /* vector register for blending; todo should be a function input / part of the reg mapping */
    libxsmm_aarch64_sve_type l_sve_type = libxsmm_generator_aarch64_get_sve_type(LIBXSMM_TYPESIZE(libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)));

    for (in = 0; in < i_n_blocking; in++) {
      unsigned int l_mask_adv = 0;
      for (im = 0; im < i_m_blocking; im++) {
        unsigned int l_masked_elements = (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ? (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0
                                                                                                                                             : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_mask_last_m_chunk : i_vlen : i_vlen;
        unsigned int l_mask_load = (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ? (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? 1 : 0 : 0
                                                                                                                                       : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? 1 : 2 : 2;

        unsigned int cur_vreg = i_start_vreg + in * i_m_blocking + im;

        if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && (i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
          libxsmm_generator_unary_binary_aarch64_load_bitmask_2bytemult_sve( io_generated_code, i_mateltwise_desc->m, im, i_m_blocking,
                                                                             LIBXSMM_CAST_UCHAR(i_micro_kernel_config->tmp_vreg),
                                                                             LIBXSMM_CAST_UCHAR(i_gp_reg_mapping->gp_reg_relumask),
                                                                             l_blend_reg,
                                                                             LIBXSMM_CAST_UCHAR(i_gp_reg_mapping->gp_reg_scratch_0),
                                                                             l_tmp_pred_reg,
                                                                             &l_mask_adv );
        } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV ) {
          libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_relumask, i_gp_reg_mapping->gp_reg_scratch_1, i_micro_kernel_config->tmp_vreg,
                                                                  i_micro_kernel_config->datatype_size_in, l_masked_elements, 1, 0, LIBXSMM_CAST_UCHAR(l_mask_load) );
          /* If compute is in F32 and input is BF16 (or input is BF16 and output is F32), then upconvert BF16 -> FP32 */
          if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) && LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ||
               (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) && LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ) {
            libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, i_micro_kernel_config->tmp_vreg, 0);
          }


          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FCMGT_Z_V,
                                                   i_micro_kernel_config->tmp_vreg, LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, l_blend_reg, /* dst was i_micro_kernel_config->tmp_vreg2 */
                                                   i_mask_reg, l_sve_type );
        } else {
          /* should not happen */
        }

        if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV ) {
          /* we need to multiply with alpha */
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                                   cur_vreg, i_micro_kernel_config->fam_lu_vreg_alpha, 0, i_micro_kernel_config->tmp_vreg2,
                                                   i_mask_reg, l_sve_type );

          /* now we blend both together */
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SEL_V_P,
                                                   cur_vreg, i_micro_kernel_config->tmp_vreg2, 0, cur_vreg, /* mask was i_micro_kernel_config->tmp_vreg */
                                                   l_blend_reg, l_sve_type );
        } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV ) {
          /* now we blend both together */
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SEL_V_P,
                                                   cur_vreg, i_micro_kernel_config->zero_vreg, 0, cur_vreg, /* mask was i_micro_kernel_config->tmp_vreg */
                                                   l_blend_reg, l_sve_type );
        } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV ) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FADD_V,
                                                   i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->fam_lu_vreg_alpha, 0, i_micro_kernel_config->tmp_vreg,
                                                   i_mask_reg, l_sve_type );

          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                                   i_micro_kernel_config->tmp_vreg, cur_vreg, 0, i_micro_kernel_config->tmp_vreg,
                                                   i_mask_reg, l_sve_type );

          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SEL_V_P,
                                                   cur_vreg, i_micro_kernel_config->tmp_vreg, 0, cur_vreg,
                                                   l_blend_reg, l_sve_type );
        } else {
          /* should not happen */
        }
      }
      if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && ( i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_ELU_INV ) ) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_mapping->gp_reg_relumask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_relumask,
                                                       ((long long)i_micro_kernel_config->ldi_mask - ((long long)l_mask_adv*8))/8 );
      } else {
        if ( l_ld_bytes != l_m_adjust ) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                        i_gp_reg_mapping->gp_reg_relumask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_relumask,
                                                        ((long long)l_ld_bytes - l_m_adjust) );
        }
      }
    }

  } else {

    for (in = 0; in < i_n_blocking; in++) {
      unsigned int l_mask_adv = 0;
      for (im = 0; im < i_m_blocking; im++) {
        unsigned int l_masked_elements = (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ? (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0
                                                                                                                                             : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_mask_last_m_chunk : i_vlen : i_vlen;
        unsigned int l_mask_load = (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ? (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? 1 : 0 : 0
                                                                                                                                       : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? 1 : 2 : 2;
        unsigned int cur_vreg = i_start_vreg + in * i_m_blocking + im;

        if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && (i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
          libxsmm_generator_unary_binary_aarch64_load_bitmask_2bytemult_asimd( io_generated_code, im, i_m_blocking,
                                                                               LIBXSMM_CAST_UCHAR(i_micro_kernel_config->mask_helper0_vreg),
                                                                               LIBXSMM_CAST_UCHAR(i_micro_kernel_config->mask_helper1_vreg),
                                                                               LIBXSMM_CAST_UCHAR(i_micro_kernel_config->tmp_vreg),
                                                                               LIBXSMM_CAST_UCHAR(i_gp_reg_mapping->gp_reg_relumask),
                                                                               &l_mask_adv );
        } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV ) {
          libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_relumask, i_gp_reg_mapping->gp_reg_scratch_1, i_micro_kernel_config->tmp_vreg,
                                                                  i_micro_kernel_config->datatype_size_in, l_masked_elements, 1, 0, LIBXSMM_CAST_UCHAR(l_mask_load) );
          /* If compute is in F32 and input is BF16 (or input is BF16 and output is F32), then upconvert BF16 -> FP32 */
          if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) && LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ||
               (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) && LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ) {
            libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, i_micro_kernel_config->tmp_vreg, 0);
          }
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FCMGT_Z_V,
                                                     i_micro_kernel_config->tmp_vreg, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, i_micro_kernel_config->tmp_vreg2,
                                                     LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
        } else {
          /* should not happen */
        }

        if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV ) {
          /* we need to multiply with alpha */
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                                     cur_vreg, i_micro_kernel_config->fam_lu_vreg_alpha, 0, i_micro_kernel_config->tmp_vreg2,
                                                     LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

          /* now we blend both together */
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_BIF_V,
                                                     i_micro_kernel_config->tmp_vreg2, i_micro_kernel_config->tmp_vreg, 0, cur_vreg,
                                                     LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
        } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV ) {
          /* now we blend both together */
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_BIF_V,
                                                     i_micro_kernel_config->zero_vreg, i_micro_kernel_config->tmp_vreg, 0, cur_vreg,
                                                     LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
        } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV ) {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V,
                                                     i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->fam_lu_vreg_alpha, 0, i_micro_kernel_config->tmp_vreg,
                                                     LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                                     i_micro_kernel_config->tmp_vreg, cur_vreg, 0, i_micro_kernel_config->tmp_vreg,
                                                     LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_BIF_V,
                                                     i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->tmp_vreg2, 0, cur_vreg,
                                                     LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
        } else {
          /* should not happen */
        }
      }
      if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && ( i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_ELU_INV ) ) {
        /* adjust reg_relumask for stride */
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_mapping->gp_reg_relumask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_relumask,
                                                       ((long long)i_micro_kernel_config->ldi_mask - ((long long)l_mask_adv*8))/8 );
      } else {
        if ( l_ld_bytes != l_m_adjust ) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                        i_gp_reg_mapping->gp_reg_relumask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_relumask,
                                                        ((long long)l_ld_bytes - l_m_adjust) );
        }
      }
    }

  }

  if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && ( i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_ELU_INV ) ) {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_relumask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_relumask,
                                                   (long long)i_n_blocking * (i_micro_kernel_config->ldi_mask/8) );
  } else {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_relumask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_relumask,
                                                   (long long)l_ld_bytes*i_n_blocking );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_unary_binary_aarch64_load_bitmask_2bytemult_asimd( libxsmm_generated_code* io_generated_code,
                                                                          const unsigned int      im,
                                                                          const unsigned int      i_m_blocking,
                                                                          const unsigned char     i_mask_helper0_vreg, /* i_micro_kernel_config->mask_helper0_vreg */
                                                                          const unsigned char     i_mask_helper1_vreg, /* i_micro_kernel_config->mask_helper1_vreg */
                                                                          const unsigned char     i_tmp0_vreg, /* i_micro_kernel_config->dropout_vreg_tmp0 */
                                                                          const unsigned char     i_gp_reg_mask, /* i_gp_reg_mapping->gp_reg_dropoutmask */
                                                                          unsigned int* const     io_mask_adv ) {
  if ( im % 2 == 0 ) {
    if ( im == i_m_blocking - 1 ) {
      libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R_R_POST,
                                                       i_gp_reg_mask, LIBXSMM_AARCH64_GP_REG_XZR, i_tmp0_vreg,
                                                       LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
      io_mask_adv[0]++;
    } else {
      libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R,
                                                       i_gp_reg_mask, LIBXSMM_AARCH64_GP_REG_UNDEF, i_tmp0_vreg,
                                                       LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
    }
    libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_AND_V,
                                               i_tmp0_vreg, i_mask_helper0_vreg, 0, i_tmp0_vreg,
                                               LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );

    libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_CMEQ_R_V,
                                               i_tmp0_vreg, i_mask_helper0_vreg, 0, i_tmp0_vreg,
                                               LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
  } else {
    libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R_R_POST,
                                                     i_gp_reg_mask, LIBXSMM_AARCH64_GP_REG_XZR, i_tmp0_vreg,
                                                     LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
    io_mask_adv[0]++;

    libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_AND_V,
                                               i_tmp0_vreg, i_mask_helper1_vreg, 0, i_tmp0_vreg,
                                               LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );

    libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_CMEQ_R_V,
                                               i_tmp0_vreg, i_mask_helper1_vreg, 0, i_tmp0_vreg,
                                               LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_unary_binary_aarch64_store_bitmask_2bytemult_asimd( libxsmm_generated_code* io_generated_code,
                                                                           const unsigned int      im,
                                                                           const unsigned int      i_m_blocking,
                                                                           const unsigned char     i_mask_helper0_vreg, /* i_micro_kernel_config->mask_helper1_vreg */
                                                                           const unsigned char     i_mask_helper1_vreg, /* i_micro_kernel_config->mask_helper1_vreg */
                                                                           const unsigned char     i_tmp_vreg0, /* i_micro_kernel_config->dropout_vreg_tmp0 */
                                                                           const unsigned char     i_tmp_vreg1, /* i_micro_kernel_config->dropout_vreg_tmp1 */
                                                                           const unsigned char     i_tmp_vreg2, /* i_micro_kernel_config->dropout_vreg_tmp2 */
                                                                           const unsigned char     i_gp_reg_mask,
                                                                           unsigned int* const     io_mask_adv ) {
  if ( im % 2 == 0 ) {
    libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_AND_V,
                                               i_tmp_vreg0, i_mask_helper0_vreg, 0, i_tmp_vreg1,
                                               LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
  } else {
    libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_AND_V,
                                               i_tmp_vreg0, i_mask_helper1_vreg, 0, i_tmp_vreg1,
                                               LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
  }
  libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ADDV_V,
                                             i_tmp_vreg1, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, i_tmp_vreg1,
                                             LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
  if ( im % 2 == 0 ) {
    if ( im == i_m_blocking - 1 ) {
      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                              i_gp_reg_mask, LIBXSMM_AARCH64_GP_REG_UNDEF, 1, i_tmp_vreg1,
                                              LIBXSMM_AARCH64_ASIMD_WIDTH_B );
      io_mask_adv[0]++;
    } else {
      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF,
                                              i_gp_reg_mask, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_tmp_vreg1,
                                              LIBXSMM_AARCH64_ASIMD_WIDTH_B );
    }
  } else {
    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                            i_gp_reg_mask, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_tmp_vreg2,
                                            LIBXSMM_AARCH64_ASIMD_WIDTH_B );

    libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                               i_tmp_vreg1, i_tmp_vreg2, 0, i_tmp_vreg1,
                                               LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

    libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                            i_gp_reg_mask, LIBXSMM_AARCH64_GP_REG_UNDEF, 1, i_tmp_vreg1,
                                            LIBXSMM_AARCH64_ASIMD_WIDTH_B );
    io_mask_adv[0]++;
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_unary_binary_aarch64_load_bitmask_2bytemult_sve( libxsmm_generated_code* io_generated_code,
                                                                        const unsigned int      m,
                                                                        const unsigned int      im,
                                                                        const unsigned int      i_m_blocking,
                                                                        const unsigned char     i_tmp0_vreg,
                                                                        const unsigned char     i_gp_reg_mask,
                                                                        const unsigned char     i_blend_reg,
                                                                        const unsigned char     i_scratch_gp_reg,
                                                                        const unsigned char     i_tmp_pred_reg,
                                                                        unsigned int* const     io_mask_adv ) {

  unsigned int l_vector_length = libxsmm_cpuid_vlen(io_generated_code->arch); /* in bytes, 512 bit -> 64 bytes */
  unsigned int l_predicate_length = l_vector_length / 8; /* 4 bytes/float, 8 bits in 1 byte, 512 bit -> 64 bytes -> 8 bytes*/
  unsigned int l_data_length = l_predicate_length / 4; /* only every 4th bit needs to be read, 512 bit -> .. -> 2 bytes */
  unsigned char im_mod = im & 3; /* 4 = number of bits per float within predicate register */

  if ( !( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE256) && (io_generated_code->arch < LIBXSMM_AARCH64_ALLFEAT) ) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
  }

#if defined(SVE_SLOW_COPY)
  /* for 64x8 on A64FX, this is 3x slower */
  /* warning! this only works for the vector length 256 and 512; 128 needs byte-mixing
 *    * 1024 needs to respect the size of the mask data array */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 LIBXSMM_AARCH64_GP_REG_XSP, i_scratch_gp_reg, LIBXSMM_AARCH64_GP_REG_XSP, l_vector_length );

  /* load <l_data_length> bytes from i_gp_reg_mask, store them into xsp, then load from xsp */
  libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, i_tmp_pred_reg, l_data_length, i_scratch_gp_reg );

  libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1B_I_OFF,
                                        i_gp_reg_mask, 0, 0, i_tmp0_vreg, i_tmp_pred_reg );
  libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ST1B_I_OFF,
                                        LIBXSMM_AARCH64_GP_REG_XSP, 0, 0, i_tmp0_vreg, i_tmp_pred_reg );
  libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LDR_P_I_OFF,
                                        LIBXSMM_AARCH64_GP_REG_XSP, 0, 0, i_blend_reg, 0 );

  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 LIBXSMM_AARCH64_GP_REG_XSP, i_scratch_gp_reg, LIBXSMM_AARCH64_GP_REG_XSP, l_vector_length );
#else
  /* warning! this only works for the vector length 256 and 512; 128 needs byte-mixing
 *    * 1024 needs to respect the size of the mask data array */
  if (im_mod == 0) {
    /* count of sections, that will be loaded: max 4, max i_m_blocking & 3 at the end */
    unsigned int copied_sections = i_m_blocking - im < 3 ? i_m_blocking & 3 : 4;
    unsigned int l_mask_length_bytes = LIBXSMM_UPDIV(m,16)*2;
    unsigned int l_remaining_data = l_mask_length_bytes - io_mask_adv[0];
    unsigned int copied_length = LIBXSMM_MIN(copied_sections * l_data_length, l_remaining_data);
    unsigned int stack_offset = LIBXSMM_UPDIV(copied_length,16)*16; /* stack addresses need to be aligned to 16 bytes */
    /* starting a new "chunk", sp -= pl */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   LIBXSMM_AARCH64_GP_REG_XSP, i_scratch_gp_reg, LIBXSMM_AARCH64_GP_REG_XSP, stack_offset );
    /* copy gp_mask_reg to sp */
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, i_tmp_pred_reg, copied_length, i_scratch_gp_reg );
    libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1B_I_OFF,
                                          i_gp_reg_mask, 0, 0, i_tmp0_vreg, i_tmp_pred_reg );
    libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ST1B_I_OFF,
                                          LIBXSMM_AARCH64_GP_REG_XSP, 0, 0, i_tmp0_vreg, i_tmp_pred_reg );
    /* load data, and adjust pointer; sp will be reset automatically */
    libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LDR_P_I_OFF,
                                          LIBXSMM_AARCH64_GP_REG_XSP, 0, 0, i_tmp_pred_reg, 0 );
    /* adjust gp_reg_mask pointers */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mask, i_scratch_gp_reg, i_gp_reg_mask, copied_length );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   LIBXSMM_AARCH64_GP_REG_XSP, i_scratch_gp_reg, LIBXSMM_AARCH64_GP_REG_XSP, stack_offset );
    io_mask_adv[0] += copied_length;
  }

  /* zip, 8 bits, low */
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, im & 2 ? LIBXSMM_AARCH64_INSTR_SVE_ZIP_P_H : LIBXSMM_AARCH64_INSTR_SVE_ZIP_P_L,
                                           i_tmp_pred_reg, i_tmp_pred_reg, 0, i_blend_reg,
                                           0, LIBXSMM_AARCH64_SVE_TYPE_B );

  /* zip, 16 bits, low */
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, im & 1 ? LIBXSMM_AARCH64_INSTR_SVE_ZIP_P_H : LIBXSMM_AARCH64_INSTR_SVE_ZIP_P_L,
                                           i_blend_reg, i_blend_reg, 0, i_blend_reg,
                                           0, LIBXSMM_AARCH64_SVE_TYPE_H );

#endif

#if defined(SVE_MASKS_HAVE_PADDING) || defined(SVE_SLOW_COPY)

  /* zip, 8 bits, low */
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP_P_L,
                                           i_blend_reg, i_blend_reg, 0, i_blend_reg,
                                           0, LIBXSMM_AARCH64_SVE_TYPE_B );

  /* zip, 16 bits, low */
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ZIP_P_L,
                                           i_blend_reg, i_blend_reg, 0, i_blend_reg,
                                           0, LIBXSMM_AARCH64_SVE_TYPE_H );

  /* increment gp_reg_relumask by <l_data_length> */
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                 i_gp_reg_mask, i_gp_reg_mask, l_data_length, 0 );
  /* we read <l_data_length> bytes */
  io_mask_adv[0] += l_data_length;
#endif

}


LIBXSMM_API_INTERN
void libxsmm_generator_unary_binary_aarch64_store_bitmask_2bytemult_sve( libxsmm_generated_code* io_generated_code,
                                                                         const unsigned int      m,
                                                                         const unsigned int      im,
                                                                         const unsigned int      i_m_blocking,
                                                                         const unsigned char     i_tmp_vreg0,
                                                                         const unsigned char     i_gp_reg_mask,
                                                                         const unsigned char     i_blend_reg,
                                                                         const unsigned char     i_tmp_pred_reg0,
                                                                         const unsigned char     i_tmp_pred_reg1,
                                                                         const unsigned char     i_gp_reg_scratch,
                                                                         unsigned int* const     io_mask_adv ) {
  unsigned int l_vector_length = libxsmm_cpuid_vlen(io_generated_code->arch);
  unsigned int l_predicate_length = l_vector_length / 8; /* 4 bytes/float, 8 bits in 1 byte */
  unsigned int l_data_length = l_predicate_length / 4; /* only every 4th bit needs to be stored */
  unsigned char im_mod = im & 3;

  if ( !( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE256) && (io_generated_code->arch < LIBXSMM_AARCH64_ALLFEAT) ) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }

  /* store bitflags (l_blend_reg) to bitflag array in memory */
#if defined(SVE_MASKS_HAVE_PADDING) || defined(SVE_SLOW_COPY)
  /* mv l_blend_reg into l_tmp_pred_reg */
  /* UZP_P_E, 16 bits */
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UZP_P_E,
                                           i_blend_reg, i_blend_reg, 0, i_tmp_pred_reg0,
                                           0, LIBXSMM_AARCH64_SVE_TYPE_H );
  /* UZP_P_E, 8 bits */
  libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UZP_P_E,
                                           i_tmp_pred_reg0, i_tmp_pred_reg0, 0, i_tmp_pred_reg0,
                                           0, LIBXSMM_AARCH64_SVE_TYPE_B );
#endif
#ifdef SVE_MASKS_HAVE_PADDING
  /* ideal: store predicate into register, store register into memory (only 2 bytes) */
  /* Antonio cannot find an instruction for that -> ensure the buffer has padding, and write over the end :/ */
  /* a hacky but less illegal way to do that would be to write to the stack, load from it, and store to the correct location */
  libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_STR_P_I_OFF,
                                        i_gp_reg_mask, 0, 0, i_tmp_pred_reg0, 0 );
#elif defined(SVE_SLOW_COPY)
  /* load <l_data_length> bytes from i_gp_reg_mask, store them into xsp, then load from xsp; only works for 256/512 bit vector lengths */
  unsigned int l_stack_offset = LIBXSMM_UPDIV(l_predicate_length, 16) * 16; /* for sp alignment */
  libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, i_tmp_pred_reg1, l_data_length, i_gp_reg_scratch );
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 LIBXSMM_AARCH64_GP_REG_XSP, i_gp_reg_scratch, LIBXSMM_AARCH64_GP_REG_XSP, l_stack_offset );
  libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_STR_P_I_OFF,
                                        LIBXSMM_AARCH64_GP_REG_XSP, 0, 0, i_tmp_pred_reg0, 0 );
  libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1B_I_OFF,
                                        LIBXSMM_AARCH64_GP_REG_XSP, 0, 0, i_tmp_vreg0, i_tmp_pred_reg1 );
  libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ST1B_I_OFF,
                                        i_gp_reg_mask, 0, 0, i_tmp_vreg0, i_tmp_pred_reg1 );
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 LIBXSMM_AARCH64_GP_REG_XSP, i_gp_reg_scratch, LIBXSMM_AARCH64_GP_REG_XSP, l_stack_offset );
#else
  if (im_mod == 3 || im == i_m_blocking-1) {
    unsigned int copied_sections = im_mod + 1;
    unsigned int l_mask_length_bytes = LIBXSMM_UPDIV(m, 16)*2;
    unsigned int l_remaining_data = l_mask_length_bytes - io_mask_adv[0];
    unsigned int copied_length = LIBXSMM_MIN(copied_sections * l_data_length, l_remaining_data);
    /* save the result to sp, into vreg, save to mask_reg, inc mask reg ptr */
    unsigned int l_stack_offset = LIBXSMM_UPDIV(l_predicate_length,16)*16; /* for sp alignment */
     /* finish 4-part-block, save it; depending on how many elements we collected, we need different methods to join all values */
    if (im_mod == 0) {
      libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UZP_P_E,
                                              i_blend_reg, i_blend_reg, 0, i_tmp_pred_reg0,
                                              0, LIBXSMM_AARCH64_SVE_TYPE_H );
      libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UZP_P_E,
                                              i_tmp_pred_reg0, i_tmp_pred_reg0, 0, i_tmp_pred_reg0,
                                              0, LIBXSMM_AARCH64_SVE_TYPE_B );
    } else if (im_mod == 1) {
      libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UZP_P_E,
                                              i_tmp_pred_reg0, i_blend_reg, 0, i_tmp_pred_reg0,
                                              0, LIBXSMM_AARCH64_SVE_TYPE_H );
      libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UZP_P_E,
                                              i_tmp_pred_reg0, i_tmp_pred_reg0, 0, i_tmp_pred_reg0,
                                              0, LIBXSMM_AARCH64_SVE_TYPE_B );
    } else if (im_mod == 2) {
      libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UZP_P_E,
                                              i_blend_reg, i_blend_reg, 0, i_tmp_pred_reg1,
                                              0, LIBXSMM_AARCH64_SVE_TYPE_H );
      libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UZP_P_E,
                                              i_tmp_pred_reg0, i_tmp_pred_reg1, 0, i_tmp_pred_reg0,
                                              0, LIBXSMM_AARCH64_SVE_TYPE_B );
    } else {/* im_mod == 3 */
      libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UZP_P_E,
                                              i_tmp_pred_reg1, i_blend_reg, 0, i_tmp_pred_reg1,
                                              0, LIBXSMM_AARCH64_SVE_TYPE_H );
      libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UZP_P_E,
                                              i_tmp_pred_reg0, i_tmp_pred_reg1, 0, i_tmp_pred_reg0,
                                              0, LIBXSMM_AARCH64_SVE_TYPE_B );
    }
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, i_tmp_pred_reg1, copied_length, i_gp_reg_scratch );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                  LIBXSMM_AARCH64_GP_REG_XSP, i_gp_reg_scratch, LIBXSMM_AARCH64_GP_REG_XSP, l_stack_offset );
    libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_STR_P_I_OFF,
                                          LIBXSMM_AARCH64_GP_REG_XSP, 0, 0, i_tmp_pred_reg0, 0 );
    libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1B_I_OFF,
                                          LIBXSMM_AARCH64_GP_REG_XSP, 0, 0, i_tmp_vreg0, i_tmp_pred_reg1 );
    libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ST1B_I_OFF,
                                          i_gp_reg_mask, 0, 0, i_tmp_vreg0, i_tmp_pred_reg1 );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                  LIBXSMM_AARCH64_GP_REG_XSP, i_gp_reg_scratch, LIBXSMM_AARCH64_GP_REG_XSP, l_stack_offset );
    /* increment gp_reg_relumask by <l_data_length> */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                  i_gp_reg_mask, i_gp_reg_scratch, i_gp_reg_mask, copied_length );
    io_mask_adv[0] += copied_length; /* we wrote <l_data_length> bytes */
   } else if (im_mod == 0) {/* save blend_reg into tmp_pred_reg for later storage */
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_P,
                                             i_blend_reg, i_blend_reg, 0, i_tmp_pred_reg0, 0, LIBXSMM_AARCH64_SVE_TYPE_B );
  } else if (im_mod == 1) {/* combine blend_reg with the value from im_mod == 0, save it into tmp_pred_reg0 */
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UZP_P_E,
                                             i_tmp_pred_reg0, i_blend_reg, 0, i_tmp_pred_reg0,
                                             0, LIBXSMM_AARCH64_SVE_TYPE_H );
  } else {/* im_mod == 2, store blend_reg into i_tmp_pred_reg1, so all four quarters can be combined in the next iteration */
    libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_P,
                                             i_blend_reg, i_blend_reg, 0, i_tmp_pred_reg1, 0, LIBXSMM_AARCH64_SVE_TYPE_B );
  }

#endif

#if defined(SVE_MASKS_HAVE_PADDING) || defined(SVE_SLOW_COPY)
  /* increment gp_reg_relumask by <l_data_length> */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 i_gp_reg_mask, i_gp_reg_scratch, i_gp_reg_mask, l_data_length );
  io_mask_adv[0] += l_data_length; /* we wrote <l_data_length> bytes */
#endif

}


LIBXSMM_API_INTERN
void libxsmm_compute_unary_aarch64_2d_reg_block_dropout( libxsmm_generated_code*                 io_generated_code,
                                                         libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                         unsigned int                            i_vlen,
                                                         unsigned int                            i_start_vreg,
                                                         unsigned int                            i_m_blocking,
                                                         unsigned int                            i_n_blocking,
                                                         unsigned int                            i_mask_last_m_chunk,
                                                         unsigned int                            i_mask_reg) {
  unsigned int im, in;
  unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  LIBXSMM_UNUSED(i_mask_last_m_chunk);
  LIBXSMM_UNUSED(i_vlen);

  for (in = 0; in < i_n_blocking; in++) {
    unsigned int l_mask_adv = 0;
    for (im = 0; im < i_m_blocking; im++) {
      /*unsigned int l_vlen = ( l_bf16_compute > 0 ) ? 32 : 16;*/
      unsigned int cur_vreg = i_start_vreg + in * i_m_blocking + im;

      if ( l_is_sve ) {
        /* could/should be function parameters */
        unsigned char l_blend_reg = 7;
        unsigned char l_tmp_pred_reg0 = 6;
        unsigned char l_tmp_pred_reg1 = 5;

        libxsmm_aarch64_sve_type l_sve_type = LIBXSMM_AARCH64_SVE_TYPE_S;

        /* draw a random number */
        libxsmm_generator_xoshiro128p_f32_aarch64_sve( io_generated_code,
                                                       i_micro_kernel_config->prng_state0_vreg,
                                                       i_micro_kernel_config->prng_state1_vreg,
                                                       i_micro_kernel_config->prng_state2_vreg,
                                                       i_micro_kernel_config->prng_state3_vreg,
                                                       i_micro_kernel_config->dropout_vreg_tmp0,
                                                       i_micro_kernel_config->dropout_vreg_tmp1,
                                                       i_micro_kernel_config->dropout_vreg_one,
                                                       i_micro_kernel_config->dropout_vreg_tmp2,
                                                       LIBXSMM_AARCH64_SVE_TYPE_S, LIBXSMM_CAST_UCHAR(i_mask_reg) );

        /* compare with p */
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FCMGT_P_V,
                                                 i_micro_kernel_config->dropout_prob_vreg,
                                                 i_micro_kernel_config->dropout_vreg_tmp2,
                                                 0, l_blend_reg, /* i_micro_kernel_config->dropout_vreg_tmp0 */
                                                 i_mask_reg, l_sve_type );

        if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) {
          libxsmm_generator_unary_binary_aarch64_store_bitmask_2bytemult_sve( io_generated_code, i_mateltwise_desc->m, im, i_m_blocking,
                                                                              LIBXSMM_CAST_UCHAR(i_micro_kernel_config->dropout_vreg_tmp0),
                                                                              LIBXSMM_CAST_UCHAR(i_gp_reg_mapping->gp_reg_dropoutmask),
                                                                              l_blend_reg, l_tmp_pred_reg0, l_tmp_pred_reg1,
                                                                              LIBXSMM_CAST_UCHAR(i_gp_reg_mapping->gp_reg_scratch_0),
                                                                              &l_mask_adv
          );
        }

        /* weight */
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                                 cur_vreg, i_micro_kernel_config->dropout_invprob_vreg, 0, cur_vreg,
                                                 i_mask_reg, l_sve_type );

        /* todo why is this constant not saved in a register? */
        /* blend zero and multiplication result together */
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                                 i_micro_kernel_config->dropout_vreg_tmp1, i_micro_kernel_config->dropout_vreg_tmp1,
                                                 0, i_micro_kernel_config->dropout_vreg_tmp1,
                                                 i_mask_reg, l_sve_type );

        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SEL_V_P,
                                                 cur_vreg, i_micro_kernel_config->dropout_vreg_tmp1, 0, cur_vreg,
                                                 l_blend_reg, l_sve_type );

      } else {

        /* draw a random number */
        libxsmm_generator_xoshiro128p_f32_aarch64_asimd( io_generated_code,
                                                         i_micro_kernel_config->prng_state0_vreg,
                                                         i_micro_kernel_config->prng_state1_vreg,
                                                         i_micro_kernel_config->prng_state2_vreg,
                                                         i_micro_kernel_config->prng_state3_vreg,
                                                         i_micro_kernel_config->dropout_vreg_tmp0,
                                                         i_micro_kernel_config->dropout_vreg_tmp1,
                                                         i_micro_kernel_config->dropout_vreg_one,
                                                         i_micro_kernel_config->dropout_vreg_tmp2 );

        /* compare with p */
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FCMGT_R_V,
                                                   i_micro_kernel_config->dropout_prob_vreg,
                                                   i_micro_kernel_config->dropout_vreg_tmp2,
                                                   0, i_micro_kernel_config->dropout_vreg_tmp0,
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

        if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) {
          libxsmm_generator_unary_binary_aarch64_store_bitmask_2bytemult_asimd( io_generated_code, im, i_m_blocking,
                                                                                LIBXSMM_CAST_UCHAR(i_micro_kernel_config->mask_helper0_vreg),
                                                                                LIBXSMM_CAST_UCHAR(i_micro_kernel_config->mask_helper1_vreg),
                                                                                LIBXSMM_CAST_UCHAR(i_micro_kernel_config->dropout_vreg_tmp0),
                                                                                LIBXSMM_CAST_UCHAR(i_micro_kernel_config->dropout_vreg_tmp1),
                                                                                LIBXSMM_CAST_UCHAR(i_micro_kernel_config->dropout_vreg_tmp2),
                                                                                LIBXSMM_CAST_UCHAR(i_gp_reg_mapping->gp_reg_dropoutmask),
                                                                                &l_mask_adv );
        }

        /* weight */
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                                   cur_vreg, i_micro_kernel_config->dropout_invprob_vreg, 0, cur_vreg,
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

        /* blend zero and multiplication result together */
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                   i_micro_kernel_config->dropout_vreg_tmp1, i_micro_kernel_config->dropout_vreg_tmp1, 0,
                                                   i_micro_kernel_config->dropout_vreg_tmp1, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_BIF_V,
                                                   i_micro_kernel_config->dropout_vreg_tmp1, i_micro_kernel_config->dropout_vreg_tmp0, 0,
                                                   cur_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );

      }
    }
    if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_mapping->gp_reg_dropoutmask, i_gp_reg_mapping->gp_reg_scratch_0,
                                                     i_gp_reg_mapping->gp_reg_dropoutmask, ((long long)i_micro_kernel_config->ldo_mask - ((long long)l_mask_adv*8))/8 );
    }
  }
  if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_dropoutmask, i_gp_reg_mapping->gp_reg_scratch_0,
                                                   i_gp_reg_mapping->gp_reg_dropoutmask, (long long)i_n_blocking * ((long long)i_micro_kernel_config->ldo_mask/8) );
  }
}

LIBXSMM_API_INTERN
void libxsmm_compute_unary_aarch64_2d_reg_block_dropout_inv( libxsmm_generated_code*                 io_generated_code,
                                                             libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                             const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                             const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                             unsigned int                            i_vlen,
                                                             unsigned int                            i_start_vreg,
                                                             unsigned int                            i_m_blocking,
                                                             unsigned int                            i_n_blocking,
                                                             unsigned int                            i_mask_last_m_chunk,
                                                             unsigned int                            i_mask_reg ) {
  unsigned int im, in;
  unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  LIBXSMM_UNUSED(i_mask_last_m_chunk);
  LIBXSMM_UNUSED(i_mask_reg);
  LIBXSMM_UNUSED(i_vlen);

  for (in = 0; in < i_n_blocking; in++) {
    unsigned int l_mask_adv = 0;
    for (im = 0; im < i_m_blocking; im++) {
      unsigned int cur_vreg = i_start_vreg + in * i_m_blocking + im;

      if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) <= 0) ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BITMASK_REQUIRED );
        return;
      }

      if ( l_is_sve ) {
        /* these predicate registers maybe should be function parameters */
        unsigned char l_tmp_pred_reg = 6;
        unsigned char l_blend_reg = 7;

        libxsmm_generator_unary_binary_aarch64_load_bitmask_2bytemult_sve( io_generated_code, i_mateltwise_desc->m, im, i_m_blocking,
                                                                           LIBXSMM_CAST_UCHAR(i_micro_kernel_config->dropout_vreg_tmp0),
                                                                           LIBXSMM_CAST_UCHAR(i_gp_reg_mapping->gp_reg_dropoutmask),
                                                                           l_blend_reg,
                                                                           LIBXSMM_CAST_UCHAR(i_gp_reg_mapping->gp_reg_scratch_0),
                                                                           l_tmp_pred_reg,
                                                                           &l_mask_adv );

        /* weight */
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_FMUL_V,
                                                 cur_vreg, i_micro_kernel_config->dropout_prob_vreg, 0, cur_vreg,
                                                 l_blend_reg, LIBXSMM_AARCH64_SVE_TYPE_S );

        /* select which value is set to 0 */
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SEL_V_P,
                                                 cur_vreg, i_micro_kernel_config->dropout_vreg_zero, 0, cur_vreg,
                                                 l_blend_reg, LIBXSMM_AARCH64_SVE_TYPE_S );
      } else {

        libxsmm_generator_unary_binary_aarch64_load_bitmask_2bytemult_asimd( io_generated_code, im, i_m_blocking,
                                                                             LIBXSMM_CAST_UCHAR(i_micro_kernel_config->mask_helper0_vreg),
                                                                             LIBXSMM_CAST_UCHAR(i_micro_kernel_config->mask_helper1_vreg),
                                                                             LIBXSMM_CAST_UCHAR(i_micro_kernel_config->dropout_vreg_tmp0),
                                                                             LIBXSMM_CAST_UCHAR(i_gp_reg_mapping->gp_reg_dropoutmask),
                                                                             &l_mask_adv );

        /* weight */
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                                   cur_vreg, i_micro_kernel_config->dropout_prob_vreg, 0, cur_vreg,
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

        /* select which value is set to 0 */
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_BIF_V,
                                                   i_micro_kernel_config->dropout_vreg_zero, i_micro_kernel_config->dropout_vreg_tmp0, 0, cur_vreg,
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
      }
    }
    if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_mapping->gp_reg_dropoutmask, i_gp_reg_mapping->gp_reg_scratch_0,
                                                     i_gp_reg_mapping->gp_reg_dropoutmask, ((long long)i_micro_kernel_config->ldi_mask - ((long long)l_mask_adv*8))/8 );
    }
  }
  if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_dropoutmask, i_gp_reg_mapping->gp_reg_scratch_0,
                                                   i_gp_reg_mapping->gp_reg_dropoutmask, (long long)i_n_blocking * ((long long)i_micro_kernel_config->ldi_mask/8) );
  }
}

LIBXSMM_API_INTERN
void libxsmm_compute_ternary_aarch64_2d_reg_block( libxsmm_generated_code*                 io_generated_code,
                                                  libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                  const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                  const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                  unsigned int                            i_vlen,
                                                  unsigned int                            i_start_vreg,
                                                  unsigned int                            i_m_blocking,
                                                  unsigned int                            i_n_blocking,
                                                  unsigned int                            i_mask_last_m_chunk,
                                                  unsigned int                            i_mask_reg) {
  unsigned int im, in, cur_vreg;
  unsigned int bcast_row = (((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_1) > 0)) ? 1 : 0;
  unsigned int bcast_col = (((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1) > 0)) ? 1 : 0;
  unsigned int bcast_scalar = (((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1) > 0)) ? 1 : 0;
  unsigned int bcast_input = ( bcast_row == 1 || bcast_col == 1 || bcast_scalar == 1 ) ? 1 : 0;
  unsigned int l_ld_bytes_in2 = i_mateltwise_desc->ldi2 * i_micro_kernel_config->datatype_size_in1;
  unsigned int l_m_adjust_in2 = ( i_mask_last_m_chunk == 0 ) ? i_micro_kernel_config->datatype_size_in1 * i_vlen * i_m_blocking : i_micro_kernel_config->datatype_size_in1 * ( (i_vlen * (i_m_blocking-1)) + i_mask_last_m_chunk );
  unsigned int offset2 = 0;
  unsigned int _in_blocking = (bcast_col == 1) ? 1 : i_n_blocking;
  unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  libxsmm_aarch64_sve_type l_sve_type = libxsmm_generator_aarch64_get_sve_type(LIBXSMM_TYPESIZE(libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)));

  if ( l_is_sve ) {
    /* define predicate registers 0 and 1 for loading */
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, 1, i_mask_last_m_chunk * i_micro_kernel_config->datatype_size_in1, i_gp_reg_mapping->gp_reg_scratch_1 );
    if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1) || LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN2)) {
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, 2, i_vlen * 2, i_gp_reg_mapping->gp_reg_scratch_1 );
    }
  }

  for (in = 0; in < i_n_blocking; in++) {
    unsigned int l_mask_adv = 0;
    for (im = 0; im < i_m_blocking; im++) {
      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      if ( (bcast_row == 1) || (bcast_scalar == 1) ) {
        offset2 = (bcast_scalar == 1) ?  0:i_micro_kernel_config->datatype_size_in1*i_mateltwise_desc->ldi2*i_n_blocking;
        libxsmm_generator_bcastload_masked_vreg_aarch64_asimd( io_generated_code, i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_1, i_micro_kernel_config->tmp_vreg,
                                                               i_micro_kernel_config->datatype_size_in1, (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0, 0 );
        /* If compute is in F32 and input is BF16 (or input is BF16 and output is F32), then upconvert BF16 -> FP32 */
        if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) && LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) ||
             (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) && LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) ) {
          libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, i_micro_kernel_config->tmp_vreg, 0);
        }
      }
      if ( bcast_input == 0 || bcast_col == 1) {
        unsigned int l_is_f32_or_f64 = (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1) || LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) ? 1 : 0;
        unsigned int l_masked_elements = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0
                                                           : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_mask_last_m_chunk : i_vlen : i_vlen;
        unsigned int l_mask_load = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? 1 : 0 : 0
                                                     : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? 1 : 2 : 2;

        offset2 = (bcast_col == 1) ? l_m_adjust_in2:(l_ld_bytes_in2*_in_blocking);
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_1, i_micro_kernel_config->tmp_vreg,
                                                                i_micro_kernel_config->datatype_size_in1, l_masked_elements, 1, 0, LIBXSMM_CAST_UCHAR(l_mask_load) );
        /* If compute is in F32 and input is BF16 (or input is BF16 and output is F32), then upconvert BF16 -> FP32 */
        if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) && LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) ||
             (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) && LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) ) {
          libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, i_micro_kernel_config->tmp_vreg, 0);
        }
      }
      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_TERNARY_SELECT) {
        unsigned char l_tmp_pred_reg = 6;
        unsigned char l_blend_reg = 7;
        /* Load mask and perform select operations */
        if (l_is_sve) {
          libxsmm_generator_unary_binary_aarch64_load_bitmask_2bytemult_sve( io_generated_code, i_mateltwise_desc->m, im, i_m_blocking,
                                                                             LIBXSMM_CAST_UCHAR(i_micro_kernel_config->tmp_vreg2),
                                                                             LIBXSMM_CAST_UCHAR(i_gp_reg_mapping->gp_reg_in3),
                                                                             l_blend_reg,
                                                                             LIBXSMM_CAST_UCHAR(i_gp_reg_mapping->gp_reg_scratch_0),
                                                                             l_tmp_pred_reg,
                                                                             &l_mask_adv );
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_SEL_V_P,
                                                   i_micro_kernel_config->tmp_vreg, cur_vreg, 0, cur_vreg,
                                                   l_blend_reg, l_sve_type );
        } else {
          libxsmm_generator_unary_binary_aarch64_load_bitmask_2bytemult_asimd( io_generated_code, im, i_m_blocking,
                                                                               LIBXSMM_CAST_UCHAR(i_micro_kernel_config->mask_helper0_vreg),
                                                                               LIBXSMM_CAST_UCHAR(i_micro_kernel_config->mask_helper1_vreg),
                                                                               LIBXSMM_CAST_UCHAR(i_micro_kernel_config->tmp_vreg2),
                                                                               LIBXSMM_CAST_UCHAR(i_gp_reg_mapping->gp_reg_in3),
                                                                               &l_mask_adv );
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_BIT_V,
                                                     i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->tmp_vreg2, 0, cur_vreg,
                                                     LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
        }
      }
    }

    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_TERNARY_SELECT) {
      unsigned int l_ld_mask = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BITMASK_2BYTEMULT) > 0) ? LIBXSMM_UPDIV(i_mateltwise_desc->ldi3, 16) * 16 : i_mateltwise_desc->ldi3;
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_mapping->gp_reg_in3, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in3,
                                                     ((long long)l_ld_mask - ((long long)l_mask_adv*8))/8 );
    }
    if ( bcast_input == 0 ) {
      if ( l_ld_bytes_in2 != l_m_adjust_in2 ) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in2,
                                                       (long long)l_ld_bytes_in2 - l_m_adjust_in2 );
      }
    }
    if (bcast_col == 1 && in < (i_n_blocking-1)) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                     i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in2,
                                                     offset2 );
    }

    if (bcast_row == 1) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in2,
                                                     (long long)i_mateltwise_desc->ldi2 * i_micro_kernel_config->datatype_size_in1 );
    }
  }
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in2,
                                                 offset2 );
  if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_TERNARY_SELECT) {
    unsigned int l_ld_mask = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BITMASK_2BYTEMULT) > 0) ? LIBXSMM_UPDIV(i_mateltwise_desc->ldi3, 16) * 16 : i_mateltwise_desc->ldi3;
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_in3, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in3,
                                                   (long long)i_n_blocking * (l_ld_mask/8) );
  }
}

LIBXSMM_API_INTERN
void libxsmm_compute_binary_aarch64_2d_reg_block( libxsmm_generated_code*                 io_generated_code,
                                                  libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                  const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                  const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                  unsigned int                            i_vlen,
                                                  unsigned int                            i_start_vreg,
                                                  unsigned int                            i_m_blocking,
                                                  unsigned int                            i_n_blocking,
                                                  unsigned int                            i_mask_last_m_chunk,
                                                  unsigned int                            i_mask_reg) {
  unsigned int im, in, cur_vreg;
  unsigned int binary_op_instr = 0;
  unsigned int bcast_row = (((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1) > 0)) ? 1 : 0;
  unsigned int bcast_col = (((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1) > 0)) ? 1 : 0;
  unsigned int bcast_scalar = (((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1) > 0)) ? 1 : 0;
  unsigned int bcast_input = ( bcast_row == 1 || bcast_col == 1 || bcast_scalar == 1 ) ? 1 : 0;
  unsigned int l_ld_bytes = i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out;
  unsigned int l_ld_bytes_in2 = i_mateltwise_desc->ldi2 * i_micro_kernel_config->datatype_size_in1;
  unsigned int l_m_adjust = ( i_mask_last_m_chunk == 0 ) ? i_micro_kernel_config->datatype_size_out * i_vlen * i_m_blocking : i_micro_kernel_config->datatype_size_out * ( (i_vlen * (i_m_blocking-1)) + i_mask_last_m_chunk );
  unsigned int l_m_adjust_in2 = ( i_mask_last_m_chunk == 0 ) ? i_micro_kernel_config->datatype_size_in1 * i_vlen * i_m_blocking : i_micro_kernel_config->datatype_size_in1 * ( (i_vlen * (i_m_blocking-1)) + i_mask_last_m_chunk );
  unsigned int offset = 0, offset2 = 0;
  unsigned int _in_blocking = (bcast_col == 1) ? 1 : i_n_blocking;
  unsigned int l_flip_args = 0;

  unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  unsigned char l_is_predicated = (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_DIV) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_MULADD) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_MAX) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_MIN) || (libxsmm_generator_mateltwise_is_binary_cmp_op(i_mateltwise_desc) > 0);
  unsigned char l_pred_reg = LIBXSMM_CAST_UCHAR(i_mask_reg);
  unsigned int l_is_comp_zip = ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_ZIP)) ? 1 : 0;
  libxsmm_aarch64_sve_type l_sve_type = (l_is_comp_zip > 0) ? libxsmm_generator_aarch64_get_sve_type(LIBXSMM_TYPESIZE(libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)))
                                                            : libxsmm_generator_aarch64_get_sve_type(LIBXSMM_TYPESIZE(libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)));

  switch (i_mateltwise_desc->param) {
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
    case LIBXSMM_MELTW_TYPE_BINARY_MULADD: {
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

  if (libxsmm_generator_mateltwise_is_binary_cmp_op(i_mateltwise_desc) > 0) {
    switch (i_mateltwise_desc->param) {
      case LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_GT: {
        binary_op_instr = (l_is_sve > 0) ? LIBXSMM_AARCH64_INSTR_SVE_FCMGT_P_V : LIBXSMM_AARCH64_INSTR_ASIMD_FCMGT_R_V;
      } break;
      case LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_GE: {
        binary_op_instr = (l_is_sve > 0) ? LIBXSMM_AARCH64_INSTR_SVE_FCMGE_P_V : LIBXSMM_AARCH64_INSTR_ASIMD_FCMGE_R_V;
      } break;
      case LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_LT: {
        binary_op_instr = (l_is_sve > 0) ? LIBXSMM_AARCH64_INSTR_SVE_FCMGT_P_V : LIBXSMM_AARCH64_INSTR_ASIMD_FCMGT_R_V;
      } break;
      case LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_LE: {
        binary_op_instr = (l_is_sve > 0) ? LIBXSMM_AARCH64_INSTR_SVE_FCMGE_P_V : LIBXSMM_AARCH64_INSTR_ASIMD_FCMGE_R_V;
      } break;
      case LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_EQ: {
        binary_op_instr = (l_is_sve > 0) ? LIBXSMM_AARCH64_INSTR_SVE_FCMEQ_P_V : LIBXSMM_AARCH64_INSTR_ASIMD_FCMEQ_R_V;
      } break;
      case LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_NE: {
        binary_op_instr = (l_is_sve > 0) ? LIBXSMM_AARCH64_INSTR_SVE_FCMNE_P_V : LIBXSMM_AARCH64_INSTR_ASIMD_FCMEQ_R_V;
      } break;
      default:;
    }
    if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_LT || i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_LE ) {
      l_flip_args = 1;
    }
  }

  if ( l_is_sve && l_is_predicated ) {/* set the whole predicate register to true */
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, l_pred_reg, -1, 0 );
  }
  if ( l_is_sve ) {
    /* define predicate registers 0 and 1 for loading */
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, 1, i_mask_last_m_chunk * i_micro_kernel_config->datatype_size_in1, i_gp_reg_mapping->gp_reg_scratch_1 );
    if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1) || LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN2) || LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, 2, i_vlen * 2, i_gp_reg_mapping->gp_reg_scratch_1 );
    }
  }

  if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_ZIP) {
    if (l_is_sve > 0) {
      offset = (l_ld_bytes_in2*i_n_blocking);
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, l_pred_reg, -1, 0 );
      libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, 2, i_vlen * i_micro_kernel_config->datatype_size_in, i_gp_reg_mapping->gp_reg_scratch_0 );
      for (in = 0; in < i_n_blocking; in++) {
        for (im = 0; im < i_m_blocking; im++) {
          unsigned int l_is_f32_or_f64 = (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) || LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ? 1 : 0;
          unsigned int l_masked_elements = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0
                                                             : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_mask_last_m_chunk : i_vlen : i_vlen;
          unsigned int l_mask_load = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? 1 : 0 : 0
                                                       : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? 1 : 2 : 2;
          cur_vreg = i_start_vreg + in * i_m_blocking + im;
          if ( bcast_input == 0 || bcast_col == 1) {
            offset = (bcast_col == 1) ? l_m_adjust_in2 : (l_ld_bytes_in2*_in_blocking);
            libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_1, i_micro_kernel_config->tmp_vreg,
                                                                  i_micro_kernel_config->datatype_size_in, l_masked_elements, 1, 0, LIBXSMM_CAST_UCHAR(l_mask_load) );
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UUNPKLO_V, i_micro_kernel_config->tmp_vreg, LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, i_micro_kernel_config->tmp_vreg, 0, libxsmm_generator_aarch64_get_sve_type(4));
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LSL_I_V, i_micro_kernel_config->tmp_vreg, LIBXSMM_AARCH64_SVE_REG_UNDEF, 16, i_micro_kernel_config->tmp_vreg, 0, libxsmm_generator_aarch64_get_sve_type(4) );
          } else  if ( (bcast_row == 1) || (bcast_scalar == 1) ) {
            offset = (bcast_scalar == 1) ? 0 : i_micro_kernel_config->datatype_size_in1*i_mateltwise_desc->ldi2*i_n_blocking;
            libxsmm_generator_bcastload_masked_vreg_aarch64_asimd( io_generated_code, i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_1, i_micro_kernel_config->tmp_vreg,
                                                                   i_micro_kernel_config->datatype_size_in1, (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0, 0 );
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_UUNPKLO_V, i_micro_kernel_config->tmp_vreg, LIBXSMM_AARCH64_SVE_REG_UNDEF, 0, i_micro_kernel_config->tmp_vreg, 0, libxsmm_generator_aarch64_get_sve_type(4));
            libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LSL_I_V, i_micro_kernel_config->tmp_vreg, LIBXSMM_AARCH64_SVE_REG_UNDEF, 16, i_micro_kernel_config->tmp_vreg, 0, libxsmm_generator_aarch64_get_sve_type(4) );
          }
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V, cur_vreg, i_micro_kernel_config->tmp_vreg, 0, cur_vreg, l_pred_reg, l_sve_type );
        }
        if ( bcast_input == 0 ) {
          if ( l_ld_bytes_in2 != l_m_adjust_in2 ) {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                          i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in2,
                                                          ((long long)l_ld_bytes_in2 - l_m_adjust_in2) );
          }
        }
        if (bcast_col == 1 && in < (i_n_blocking-1)) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                         i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in2,
                                                         offset );
        }
        if (bcast_row == 1) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                         i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in2,
                                                         (long long)i_mateltwise_desc->ldi2 * i_micro_kernel_config->datatype_size_in1 );
        }
      }
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                  i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in2,
                                                  offset );
    }
    return;
  }

  for (in = 0; in < i_n_blocking; in++) {
    unsigned int l_mask_adv = 0;
    for (im = 0; im < i_m_blocking; im++) {
      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      if ( (bcast_row == 1) || (bcast_scalar == 1) ) {
        offset2 = (bcast_scalar == 1) ?  0:i_micro_kernel_config->datatype_size_in1*i_mateltwise_desc->ldi2*i_n_blocking;
        offset = (l_ld_bytes*i_n_blocking);
        libxsmm_generator_bcastload_masked_vreg_aarch64_asimd( io_generated_code, i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_1, i_micro_kernel_config->tmp_vreg,
                                                               i_micro_kernel_config->datatype_size_in1, (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0, 0 );
        /* If compute is in F32 and input is BF16 (or input is BF16 and output is F32), then upconvert BF16 -> FP32 */
        if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) && LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) ||
             (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) && LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) ) {
          libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, i_micro_kernel_config->tmp_vreg, 0);
        }
      }
      if ( bcast_input == 0 || bcast_col == 1) {
        unsigned int l_is_f32_or_f64 = (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1) || LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) ? 1 : 0;
        unsigned int l_masked_elements = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0
                                                           : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_mask_last_m_chunk : i_vlen : i_vlen;
        unsigned int l_mask_load = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? 1 : 0 : 0
                                                     : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? 1 : 2 : 2;

        offset = (l_ld_bytes*i_n_blocking);
        offset2 = (bcast_col == 1) ? l_m_adjust_in2:(l_ld_bytes_in2*_in_blocking);
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_1, i_micro_kernel_config->tmp_vreg,
                                                                i_micro_kernel_config->datatype_size_in1, l_masked_elements, 1, 0, LIBXSMM_CAST_UCHAR(l_mask_load) );
        /* If compute is in F32 and input is BF16 (or input is BF16 and output is F32), then upconvert BF16 -> FP32 */
        if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) && LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) ||
             (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) && LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) ) {
          libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, i_micro_kernel_config->tmp_vreg, 0);
        }
      }

      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_MULADD) {
        unsigned int l_is_f32_or_f64 = (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) || LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ? 1 : 0;
        unsigned int l_masked_elements = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0
                                                           : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_mask_last_m_chunk : i_vlen : i_vlen;
        unsigned int l_mask_load = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? 1 : 0 : 0
                                                     : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? 1 : 2 : 2;

        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_1, i_micro_kernel_config->tmp_vreg2,
                                                                i_micro_kernel_config->datatype_size_out, l_masked_elements, 1, 0, LIBXSMM_CAST_UCHAR(l_mask_load) );
        /* If compute is in F32 and input is BF16 (or input is BF16 and output is F32), then upconvert BF16 -> FP32 */
        if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) && LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ||
             (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) && LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ) {
          libxsmm_generator_vcvt_bf16f32_aarch64( io_generated_code, i_micro_kernel_config->tmp_vreg2, 0);
        }
        /* the result is temporarily stored in tmp_vreg2, because the instruction computes dst += src0*src1 */
        if ( l_is_sve ) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, binary_op_instr,
                                                   cur_vreg, i_micro_kernel_config->tmp_vreg, 0, i_micro_kernel_config->tmp_vreg2,
                                                   l_pred_reg, l_sve_type );
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_ORR_V,
                                                   i_micro_kernel_config->tmp_vreg2, i_micro_kernel_config->tmp_vreg2, 0, cur_vreg,
                                                   l_pred_reg, l_sve_type );
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, binary_op_instr,
                                                    cur_vreg, i_micro_kernel_config->tmp_vreg, 0, i_micro_kernel_config->tmp_vreg2,
                                                    (i_micro_kernel_config->datatype_size_in1 == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                                    i_micro_kernel_config->tmp_vreg2, i_micro_kernel_config->tmp_vreg2, 0, cur_vreg,
                                                    (i_micro_kernel_config->datatype_size_in1 == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
        }
      } else if (libxsmm_generator_mateltwise_is_binary_cmp_op(i_mateltwise_desc) > 0) {
        unsigned char l_blend_reg = 7; /* sve predicate register for blending; todo should be a function input / part of the config */
        unsigned char l_tmp_pred_reg0 = 6; /* tmp sve predicate register for blending; todo should be a function input / part of the config */
        unsigned char l_tmp_pred_reg1 = 5;
        if ( l_is_sve ) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, binary_op_instr, (l_flip_args > 0) ? i_micro_kernel_config->tmp_vreg : cur_vreg, (l_flip_args > 0) ? cur_vreg : i_micro_kernel_config->tmp_vreg, 0, l_blend_reg, l_pred_reg, l_sve_type );
          libxsmm_generator_unary_binary_aarch64_store_bitmask_2bytemult_sve( io_generated_code, i_mateltwise_desc->m, im, i_m_blocking,
                                                                                LIBXSMM_CAST_UCHAR(i_micro_kernel_config->tmp_vreg),
                                                                                LIBXSMM_CAST_UCHAR(i_gp_reg_mapping->gp_reg_out),
                                                                                l_blend_reg, l_tmp_pred_reg0, l_tmp_pred_reg1,
                                                                                LIBXSMM_CAST_UCHAR(i_gp_reg_mapping->gp_reg_scratch_0),
                                                                                &l_mask_adv );
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, binary_op_instr, (l_flip_args > 0) ? i_micro_kernel_config->tmp_vreg : cur_vreg, (l_flip_args > 0) ? cur_vreg : i_micro_kernel_config->tmp_vreg, 0, i_micro_kernel_config->tmp_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
          if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_NE) {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_NOT_V, i_micro_kernel_config->tmp_vreg, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, i_micro_kernel_config->tmp_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
          }
          libxsmm_generator_unary_binary_aarch64_store_bitmask_2bytemult_asimd( io_generated_code, im, i_m_blocking,
              LIBXSMM_CAST_UCHAR(i_micro_kernel_config->mask_helper0_vreg),
              LIBXSMM_CAST_UCHAR(i_micro_kernel_config->mask_helper1_vreg),
              LIBXSMM_CAST_UCHAR(i_micro_kernel_config->tmp_vreg),
              LIBXSMM_CAST_UCHAR(i_micro_kernel_config->tmp_vreg2),
              LIBXSMM_CAST_UCHAR(i_micro_kernel_config->tmp_vreg3),
              LIBXSMM_CAST_UCHAR(i_gp_reg_mapping->gp_reg_out),
              &l_mask_adv );
        }
      } else {
        if ( l_is_sve ) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code, binary_op_instr,
                                                   cur_vreg, i_micro_kernel_config->tmp_vreg, 0, cur_vreg,
                                                   l_pred_reg, l_sve_type );
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, binary_op_instr,
                                                     cur_vreg, i_micro_kernel_config->tmp_vreg, 0, cur_vreg,
                                                     (i_micro_kernel_config->datatype_size_in1 == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
        }
      }
    }
    if ( (l_ld_bytes != l_m_adjust) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_MULADD) ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_mapping->gp_reg_out,i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out,
                                                     ((long long)l_ld_bytes - l_m_adjust) );
    }

    if (libxsmm_generator_mateltwise_is_binary_cmp_op(i_mateltwise_desc) > 0) {
      unsigned int l_ld_mask = i_micro_kernel_config->ldo_mask;
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out,
                                                     ((long long)l_ld_mask - ((long long)l_mask_adv*8))/8 );
    }

    if ( bcast_input == 0 ) {
      if ( l_ld_bytes_in2 != l_m_adjust_in2 ) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in2,
                                                       (long long)l_ld_bytes_in2 - l_m_adjust_in2 );
      }
    }
    if (bcast_col == 1 && in < (i_n_blocking-1)) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                     i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in2,
                                                     offset2 );
    }

    if (bcast_row == 1) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in2,
                                                     (long long)i_mateltwise_desc->ldi2 * i_micro_kernel_config->datatype_size_in1 );
    }
  }
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in2,
                                                 offset2 );
  if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_MULADD) {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out,
                                                   offset );
  }

  if (libxsmm_generator_mateltwise_is_binary_cmp_op(i_mateltwise_desc) > 0) {
    unsigned int l_ld_mask = i_micro_kernel_config->ldo_mask;
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out,
                                                   (long long)i_n_blocking * (l_ld_mask/8) );
  }
}

LIBXSMM_API_INTERN
void libxsmm_compute_unary_binary_aarch64_2d_reg_block( libxsmm_generated_code*                 io_generated_code,
                                                        libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                        const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                        const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                        unsigned int                            i_vlen,
                                                        unsigned int                            i_start_vreg,
                                                        unsigned int                            i_m_blocking,
                                                        unsigned int                            i_n_blocking,
                                                        unsigned int                            i_mask_last_m_chunk,
                                                        unsigned int                            i_mask_reg) {
  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
    switch (i_mateltwise_desc->param) {
      case LIBXSMM_MELTW_TYPE_UNARY_TANH:
      case LIBXSMM_MELTW_TYPE_UNARY_EXP:
      case LIBXSMM_MELTW_TYPE_UNARY_SIGMOID:
      case LIBXSMM_MELTW_TYPE_UNARY_TANH_INV:
      case LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV:
      case LIBXSMM_MELTW_TYPE_UNARY_GELU:
      case LIBXSMM_MELTW_TYPE_UNARY_GELU_INV:
      case LIBXSMM_MELTW_TYPE_UNARY_SQRT:
      case LIBXSMM_MELTW_TYPE_UNARY_NEGATE:
      case LIBXSMM_MELTW_TYPE_UNARY_INC:
      case LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL:
      case LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT:
      case LIBXSMM_MELTW_TYPE_UNARY_X2: {
        libxsmm_compute_unary_aarch64_2d_reg_block_op( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_vlen, i_start_vreg, i_m_blocking, i_n_blocking, i_mask_last_m_chunk, i_mask_reg);
      } break;
      case LIBXSMM_MELTW_TYPE_UNARY_RELU:
      case LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU:
      case LIBXSMM_MELTW_TYPE_UNARY_ELU: {
        libxsmm_compute_unary_aarch64_2d_reg_block_relu( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_vlen, i_start_vreg, i_m_blocking, i_n_blocking, i_mask_last_m_chunk, i_mask_reg);
      } break;
      case LIBXSMM_MELTW_TYPE_UNARY_RELU_INV:
      case LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV:
      case LIBXSMM_MELTW_TYPE_UNARY_ELU_INV: {
        libxsmm_compute_unary_aarch64_2d_reg_block_relu_inv( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_vlen, i_start_vreg, i_m_blocking, i_n_blocking, i_mask_last_m_chunk, i_mask_reg);
      } break;
      case LIBXSMM_MELTW_TYPE_UNARY_DROPOUT: {
        libxsmm_compute_unary_aarch64_2d_reg_block_dropout( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_vlen, i_start_vreg, i_m_blocking, i_n_blocking, i_mask_last_m_chunk, i_mask_reg);
      } break;
      case LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV: {
        libxsmm_compute_unary_aarch64_2d_reg_block_dropout_inv( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_vlen, i_start_vreg, i_m_blocking, i_n_blocking, i_mask_last_m_chunk, i_mask_reg);
      } break;
    }
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
    libxsmm_compute_binary_aarch64_2d_reg_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_vlen, i_start_vreg, i_m_blocking, i_n_blocking, i_mask_last_m_chunk, i_mask_reg);
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
    libxsmm_compute_ternary_aarch64_2d_reg_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_vlen, i_start_vreg, i_m_blocking, i_n_blocking, i_mask_last_m_chunk, i_mask_reg);
  }
}

LIBXSMM_API_INTERN
void libxsmm_setup_input_output_aarch64_masks( libxsmm_generated_code*                 io_generated_code,
                                               libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                               const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                               unsigned int                            i_tmp_reg,
                                               unsigned int                            i_m,
                                               unsigned int*                           i_use_m_input_masking,
                                               unsigned int*                           i_mask_reg_in,
                                               unsigned int*                           i_use_m_output_masking,
                                               unsigned int*                           i_mask_reg_out) {
  unsigned int i_vlen_in = i_micro_kernel_config->vlen_in;
  unsigned int i_vlen_out = i_micro_kernel_config->vlen_out;
  unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  LIBXSMM_UNUSED(io_generated_code);
  LIBXSMM_UNUSED(i_micro_kernel_config);
  LIBXSMM_UNUSED(i_mateltwise_desc);
  LIBXSMM_UNUSED(i_tmp_reg);

  *i_mask_reg_in = 0;
  *i_use_m_input_masking = i_m % i_vlen_in;
  *i_mask_reg_out = 0;
  *i_use_m_output_masking = i_m % i_vlen_out;

  if ( l_is_sve ) {
    /* reserving predicate 0 and 1 for ptrue and remainder (when loading/storing) */
    i_micro_kernel_config->reserved_mask_regs += 2;
  }

}

LIBXSMM_API_INTERN
void libxsmm_configure_microkernel_aarch64_loops( libxsmm_generated_code*                 io_generated_code,
                                                  libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                  libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                  const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                  unsigned int                            i_m,
                                                  unsigned int                            i_n,
                                                  unsigned int                            i_use_m_input_masking,
                                                  unsigned int*                           i_m_trips,
                                                  unsigned int*                           i_n_trips,
                                                  unsigned int*                           i_m_unroll_factor,
                                                  unsigned int*                           i_n_unroll_factor,
                                                  unsigned int*                           i_m_assm_trips,
                                                  unsigned int*                           i_n_assm_trips,
                                                  unsigned int*                           i_out_loop_trips,
                                                  unsigned int*                           i_inner_loop_trips,
                                                  unsigned int*                           i_out_loop_bound,
                                                  unsigned int*                           i_inner_loop_bound,
                                                  unsigned int*                           i_out_loop_reg,
                                                  unsigned int*                           i_inner_loop_reg,
                                                  unsigned int*                           i_out_unroll_factor,
                                                  unsigned int*                           i_inner_unroll_factor) {
  unsigned int m_trips, n_trips, m_unroll_factor, n_unroll_factor, m_assm_trips, n_assm_trips, out_loop_trips, inner_loop_trips, out_loop_bound, inner_loop_bound, out_loop_reg, inner_loop_reg, out_unroll_factor, inner_unroll_factor;
  unsigned int max_nm_unrolling = 32;
  unsigned int i_loop_order = i_micro_kernel_config->loop_order;
  unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;
  unsigned int i_vlen_in = i_micro_kernel_config->vlen_in;
  unsigned int l_bitmask_2byte_mult = (((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) || ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BITMASK_2BYTEMULT) > 0) || ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BITMASK_2BYTEMULT) > 0)) ? 1 : 0;
  LIBXSMM_UNUSED(i_mateltwise_desc);
  LIBXSMM_UNUSED(io_generated_code);

  m_trips               = LIBXSMM_UPDIV(i_m, i_vlen_in);
  n_trips               = i_n;

  max_nm_unrolling  = max_nm_unrolling - reserved_zmms;

  if ( (l_bitmask_2byte_mult > 0) && (i_vlen_in == 4) ) {
    m_unroll_factor = m_trips;
  } else {
    if (i_use_m_input_masking != 0) {
      m_unroll_factor = m_trips;
    } else {
      m_unroll_factor = LIBXSMM_MIN(m_trips,16);
    }
  }

  if (m_unroll_factor > max_nm_unrolling) {
    m_unroll_factor = max_nm_unrolling;
  }

  while (m_trips % m_unroll_factor != 0) {
    m_unroll_factor--;
  }

  n_unroll_factor = n_trips;
  while (m_unroll_factor * n_unroll_factor > max_nm_unrolling) {
    n_unroll_factor--;
  }

  while (n_trips % n_unroll_factor != 0) {
    n_unroll_factor--;
  }

  m_assm_trips = m_trips/m_unroll_factor;
  n_assm_trips = n_trips/n_unroll_factor;

  out_loop_trips      = (i_loop_order == NM_LOOP_ORDER) ? n_assm_trips : m_assm_trips;
  out_loop_bound      = (i_loop_order == NM_LOOP_ORDER) ? n_trips : m_trips;
  out_loop_reg        = (i_loop_order == NM_LOOP_ORDER) ? i_gp_reg_mapping->gp_reg_n_loop : i_gp_reg_mapping->gp_reg_m_loop;
  out_unroll_factor   = (i_loop_order == NM_LOOP_ORDER) ? n_unroll_factor : m_unroll_factor;

  inner_loop_trips    = (i_loop_order == MN_LOOP_ORDER) ? n_assm_trips : m_assm_trips;
  inner_loop_bound    = (i_loop_order == MN_LOOP_ORDER) ? n_trips : m_trips;
  inner_loop_reg      = (i_loop_order == MN_LOOP_ORDER) ? i_gp_reg_mapping->gp_reg_n_loop : i_gp_reg_mapping->gp_reg_m_loop;
  inner_unroll_factor = (i_loop_order == MN_LOOP_ORDER) ? n_unroll_factor : m_unroll_factor;

  *i_m_trips = m_trips;
  *i_n_trips = n_trips;
  *i_m_unroll_factor = m_unroll_factor;
  *i_n_unroll_factor = n_unroll_factor;
  *i_m_assm_trips = m_assm_trips;
  *i_n_assm_trips = n_assm_trips;
  *i_out_loop_trips = out_loop_trips;
  *i_inner_loop_trips = inner_loop_trips;
  *i_out_loop_bound = out_loop_bound;
  *i_inner_loop_bound = inner_loop_bound;
  *i_out_loop_reg = out_loop_reg;
  *i_inner_loop_reg = inner_loop_reg;
  *i_out_unroll_factor = out_unroll_factor;
  *i_inner_unroll_factor = inner_unroll_factor;
}

LIBXSMM_API_INTERN
void libxsmm_configure_unary_aarch64_kernel_vregs_masks(  libxsmm_generated_code*                 io_generated_code,
                                                          libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                          unsigned int                            op,
                                                          unsigned int                            flags,
                                                          unsigned int                            i_gp_reg_tmp0,
                                                          unsigned int                            i_gp_reg_tmp1,
                                                          const unsigned int                      i_gp_reg_aux0,
                                                          const unsigned int                      i_gp_reg_aux1 ) {
  unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  unsigned char l_is_sve_256 = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE256) && (io_generated_code->arch < LIBXSMM_AARCH64_SVE512);
  unsigned char l_is_sve_512 = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE512) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
  unsigned char l_pred_reg = 0; /* todo decide which predicate register to use */
  libxsmm_aarch64_sve_type l_sve_type = libxsmm_generator_aarch64_get_sve_type((i_micro_kernel_config->datatype_size_in == 8) ? 8 : 4);
  libxsmm_aarch64_asimd_tupletype l_tupletype = (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D;

  LIBXSMM_UNUSED(flags);

  if (l_is_sve) libxsmm_generator_set_p_register_aarch64_sve(io_generated_code, l_pred_reg, -1, 0);

  if (op == LIBXSMM_MELTW_TYPE_UNARY_UNZIP) {
    i_micro_kernel_config->tmp_vreg  = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->tmp_vreg2 = i_micro_kernel_config->reserved_zmms + 1;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 2;
  }

  if ( (op == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X2) ||
       (op == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3) ) {
    unsigned long long l_fp32_lsb_mak        = 0xffff0000;
    i_micro_kernel_config->mask_helper0_vreg = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->vec_tmp0          = i_micro_kernel_config->reserved_zmms + 1;
    i_micro_kernel_config->vec_tmp1          = i_micro_kernel_config->reserved_zmms + 2;
    i_micro_kernel_config->vec_tmp2          = i_micro_kernel_config->reserved_zmms + 3;
    i_micro_kernel_config->reserved_zmms     = i_micro_kernel_config->reserved_zmms + 4;

    /* setting up FP32 LSB mask */
    if ( l_is_sve ) {
      libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_micro_kernel_config->mask_helper0_vreg), i_gp_reg_tmp0,
                                                                LIBXSMM_AARCH64_SVE_TYPE_S, l_pred_reg, l_fp32_lsb_mak );
    } else {
      /* nothing to do */
    }

    /* load offsets */
    if ( op == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3 ) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_aux0,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 8, i_gp_reg_aux1 );
    }
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_aux0,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_gp_reg_aux0 );
  }

  if ((op == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (op == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV)   ||
      (op == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (op == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV)) {
    i_micro_kernel_config->zero_vreg = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->tmp_vreg  = i_micro_kernel_config->reserved_zmms + 1;
    i_micro_kernel_config->tmp_vreg2 = i_micro_kernel_config->reserved_zmms + 2;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 3;

    if ( (flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) {
      i_micro_kernel_config->tmp_vreg3 = i_micro_kernel_config->reserved_zmms;
      i_micro_kernel_config->mask_helper0_vreg =  i_micro_kernel_config->reserved_zmms + 1;
      i_micro_kernel_config->mask_helper1_vreg =  i_micro_kernel_config->reserved_zmms + 2;
      i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 3;

      if ( l_is_sve ) {

        /* no masks are needed for sve */

      } else {

        /* load 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80 into mask_helper0/1_vreg */

        /* stack pointer -= 32, so prepare to use 32 bytes = 8 floats of stack memory */
        /* while LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF supports a signed offset, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF does not */
        libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I,
                                                       LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 32, 0 );

        libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp0, 0x200000001 ); /* int32 0x02, 0x01, little endian -> 0x01, 0x02 */
        libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp1, 0x800000004 ); /* int32 0x08, 0x04, little endian -> 0x04, 0x08 */

        /* store a pair of fp registers to memory: 1,2,4,8 is being loaded into the stack memory */
        libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF,
                                                   LIBXSMM_AARCH64_GP_REG_XSP, 16,
                                                   i_gp_reg_tmp0, i_gp_reg_tmp1 );

        /* now those 32 bytes are stored into mask_helper0_vreg */
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                                LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_UNDEF, 16,
                                                i_micro_kernel_config->mask_helper0_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );

        libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp0, 0x2000000010 ); /* int32 0x20, 0x10 */
        libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp1, 0x8000000040 ); /* int32 0x80, 0x40 */

        libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF,
                                                   LIBXSMM_AARCH64_GP_REG_XSP, 0,
                                                   i_gp_reg_tmp0, i_gp_reg_tmp1 );

        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                                LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_UNDEF, 0,
                                                i_micro_kernel_config->mask_helper1_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );

        /* reset stack pointer to its original position */
        libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                       LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 32, 0 );

      }
    }

    if ( (op == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (op == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ) {
      i_micro_kernel_config->fam_lu_vreg_alpha = i_micro_kernel_config->reserved_zmms;
      i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;

      /* load alpha */
      if ( l_is_sve ) {
        libxsmm_aarch64_instruction_sve_move( io_generated_code, /* load 1 value and broadcast it to all elements */
          l_sve_type == LIBXSMM_AARCH64_SVE_TYPE_B ? LIBXSMM_AARCH64_INSTR_SVE_LD1RB_I_OFF :
          l_sve_type == LIBXSMM_AARCH64_SVE_TYPE_H ? LIBXSMM_AARCH64_INSTR_SVE_LD1RH_I_OFF :
          l_sve_type == LIBXSMM_AARCH64_SVE_TYPE_S ? LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF :
                                                     LIBXSMM_AARCH64_INSTR_SVE_LD1RD_I_OFF,
                                              i_gp_reg_aux1, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_micro_kernel_config->fam_lu_vreg_alpha,
                                              l_pred_reg );
      } else {
        libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R,
                                                         i_gp_reg_aux1, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->fam_lu_vreg_alpha,
                                                         LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
      }
    }

    /* Set zero register needed for relu */
    if ( l_is_sve ) {
      libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                               i_micro_kernel_config->zero_vreg, i_micro_kernel_config->zero_vreg, 0, i_micro_kernel_config->zero_vreg,
                                               l_pred_reg, l_sve_type );
    } else {
      libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                 i_micro_kernel_config->zero_vreg, i_micro_kernel_config->zero_vreg, 0, i_micro_kernel_config->zero_vreg,
                                                 LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
    }
  }

  if ( op == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL ) {
    i_micro_kernel_config->tmp_vreg = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
  }

  if ( op == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT ) {
    /* two temporary registers are required, 3 in total: original, guess, guess squared */
    i_micro_kernel_config->tmp_vreg = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->tmp_vreg2 = i_micro_kernel_config->reserved_zmms + 1;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 2;
  }

  if (op == LIBXSMM_MELTW_TYPE_UNARY_SQRT) {
    if (l_is_sve) {
      i_micro_kernel_config->tmp_vreg = i_micro_kernel_config->reserved_zmms;
      i_micro_kernel_config->tmp_vreg2 = i_micro_kernel_config->reserved_zmms + 1;
      i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 2;
    }
  }

  if ((op == LIBXSMM_MELTW_TYPE_UNARY_ELU) || (op == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV)) {
    i_micro_kernel_config->tmp_vreg = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->tmp_vreg2 = i_micro_kernel_config->reserved_zmms + 1;
    i_micro_kernel_config->tmp_vreg3 = i_micro_kernel_config->reserved_zmms + 2;
    i_micro_kernel_config->fam_lu_vreg_alpha = i_micro_kernel_config->reserved_zmms + 3;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 4;

    /* load alpha */
    if ( l_is_sve ) {
      libxsmm_aarch64_instruction_sve_move( io_generated_code, /* load 1 value and broadcast it to all elements */
        l_sve_type == LIBXSMM_AARCH64_SVE_TYPE_B ? LIBXSMM_AARCH64_INSTR_SVE_LD1RB_I_OFF :
        l_sve_type == LIBXSMM_AARCH64_SVE_TYPE_H ? LIBXSMM_AARCH64_INSTR_SVE_LD1RH_I_OFF :
        l_sve_type == LIBXSMM_AARCH64_SVE_TYPE_S ? LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF :
                                                   LIBXSMM_AARCH64_INSTR_SVE_LD1RD_I_OFF,
                                            i_gp_reg_aux1, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_micro_kernel_config->fam_lu_vreg_alpha,
                                            l_pred_reg );
    } else {
      libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R,
                                                       i_gp_reg_aux1, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->fam_lu_vreg_alpha,
                                                       LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
    }
  }

  if ((op == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (op == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
    i_micro_kernel_config->mask_helper0_vreg =  i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->mask_helper1_vreg =  i_micro_kernel_config->reserved_zmms + 1;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 2;

    if (l_is_sve) {
      /* no masks are needed in SVE */
    } else {
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I,
                                                     LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 32, 0 );

      libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp0, 0x200000001 );
      libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp1, 0x800000004 );

      libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 16,
                                                 i_gp_reg_tmp0, i_gp_reg_tmp1 );

      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                              LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_UNDEF, 16,
                                              i_micro_kernel_config->mask_helper0_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );

      libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp0, 0x2000000010 );
      libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp1, 0x8000000040 );

      libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 0,
                                                 i_gp_reg_tmp0, i_gp_reg_tmp1 );

      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                              LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_UNDEF, 0,
                                              i_micro_kernel_config->mask_helper1_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );

      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                     LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 32, 0 );
    }

    if (op == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) {
      i_micro_kernel_config->reserved_zmms += 10;

      i_micro_kernel_config->prng_state0_vreg     = i_micro_kernel_config->reserved_zmms - 1;
      i_micro_kernel_config->prng_state1_vreg     = i_micro_kernel_config->reserved_zmms - 2;
      i_micro_kernel_config->prng_state2_vreg     = i_micro_kernel_config->reserved_zmms - 3;
      i_micro_kernel_config->prng_state3_vreg     = i_micro_kernel_config->reserved_zmms - 4;
      i_micro_kernel_config->dropout_vreg_tmp0    = i_micro_kernel_config->reserved_zmms - 5;
      i_micro_kernel_config->dropout_vreg_tmp1    = i_micro_kernel_config->reserved_zmms - 6;
      i_micro_kernel_config->dropout_vreg_tmp2    = i_micro_kernel_config->reserved_zmms - 7;
      i_micro_kernel_config->dropout_vreg_one     = i_micro_kernel_config->reserved_zmms - 8;
      i_micro_kernel_config->dropout_prob_vreg    = i_micro_kernel_config->reserved_zmms - 9;
      i_micro_kernel_config->dropout_invprob_vreg = i_micro_kernel_config->reserved_zmms - 10;

      libxsmm_generator_load_prng_state_aarch64_asimd( io_generated_code, i_gp_reg_aux0,
                                                       i_micro_kernel_config->prng_state0_vreg, i_micro_kernel_config->prng_state1_vreg,
                                                       i_micro_kernel_config->prng_state2_vreg, i_micro_kernel_config->prng_state3_vreg );
      libxsmm_generator_prepare_dropout_aarch64_asimd( io_generated_code, i_gp_reg_tmp0, i_gp_reg_aux1,
                                                       i_micro_kernel_config->dropout_vreg_one,
                                                       i_micro_kernel_config->dropout_prob_vreg,
                                                       i_micro_kernel_config->dropout_invprob_vreg );
    }

    if (op == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV) {
      i_micro_kernel_config->reserved_zmms += 4;

      i_micro_kernel_config->dropout_vreg_tmp0 = i_micro_kernel_config->reserved_zmms - 1;
      i_micro_kernel_config->dropout_vreg_one  = i_micro_kernel_config->reserved_zmms - 2;
      i_micro_kernel_config->dropout_vreg_zero = i_micro_kernel_config->reserved_zmms - 3;
      i_micro_kernel_config->dropout_prob_vreg = i_micro_kernel_config->reserved_zmms - 4;

      libxsmm_generator_prepare_dropout_inv_aarch64_asimd( io_generated_code, i_gp_reg_tmp0, i_gp_reg_aux1,
                                                           i_micro_kernel_config->dropout_vreg_one,
                                                           i_micro_kernel_config->dropout_vreg_zero,
                                                           i_micro_kernel_config->dropout_prob_vreg );
    }
  }

  if (op == LIBXSMM_MELTW_TYPE_UNARY_XOR) {
    unsigned char l_zero_reg = LIBXSMM_CAST_UCHAR(i_micro_kernel_config->reserved_zmms);
    i_micro_kernel_config->reserved_zmms = l_zero_reg + 1;
    i_micro_kernel_config->zero_vreg = l_zero_reg;
    if ( l_is_sve ) {
      /* the sve data type does not matter, maybe we should add a new enum value for that */
      libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V, l_zero_reg, l_zero_reg, 0, l_zero_reg, 0, LIBXSMM_AARCH64_SVE_TYPE_S );
    } else {
      libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V, l_zero_reg, l_zero_reg, 0, l_zero_reg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
    }
  }

  if ((op == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL) || (op == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT)) {
    i_micro_kernel_config->vec_tmp0 = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
  }

  if (op == LIBXSMM_MELTW_TYPE_UNARY_INC) {
    i_micro_kernel_config->vec_ones = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
    if (l_is_sve) {/* while sve has an inc instruction, that one is slower for 64x64 (2.95x -> 3.49x speedup vs ASIMD on A64FX) */
      libxsmm_aarch64_instruction_broadcast_scalar_to_vec_sve ( io_generated_code, LIBXSMM_CAST_UCHAR(i_micro_kernel_config->vec_ones), i_gp_reg_tmp0,
                                                            l_sve_type, l_pred_reg, (i_micro_kernel_config->datatype_size_in == 8) ? 0x3ff0000000000000 : 0x3f800000  );
    } else {
      libxsmm_aarch64_instruction_broadcast_scalar_to_vec_asimd ( io_generated_code, LIBXSMM_CAST_UCHAR(i_micro_kernel_config->vec_ones), i_gp_reg_tmp0,
                                                            l_tupletype, (i_micro_kernel_config->datatype_size_in == 8) ? 0x3ff0000000000000 : 0x3f800000 );
    }
  }

  if (op == LIBXSMM_MELTW_TYPE_UNARY_EXP) {
    i_micro_kernel_config->vec_y = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->vec_z = i_micro_kernel_config->reserved_zmms + 1;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 2;
  }

  if (op == LIBXSMM_MELTW_TYPE_UNARY_GELU || op == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV) {
    unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;

    if (l_is_sve_256) {
      reserved_zmms += 19;
      i_micro_kernel_config->vec_xr         = reserved_zmms - 1;
      i_micro_kernel_config->vec_xa         = reserved_zmms - 2;
      i_micro_kernel_config->vec_index      = reserved_zmms - 3;
      i_micro_kernel_config->vec_C0         = reserved_zmms - 4;
      i_micro_kernel_config->vec_C1         = reserved_zmms - 5;
      i_micro_kernel_config->vec_C2         = reserved_zmms - 6;
      i_micro_kernel_config->vec_thres      = reserved_zmms - 7;
      i_micro_kernel_config->vec_absmask    = reserved_zmms - 8;
      i_micro_kernel_config->vec_scale      = reserved_zmms - 9;
      i_micro_kernel_config->vec_shifter    = reserved_zmms - 10;
      i_micro_kernel_config->vec_halves     = reserved_zmms - 11;
      i_micro_kernel_config->vec_c01        = reserved_zmms - 12;
      i_micro_kernel_config->vec_c0         = reserved_zmms - 13;
      i_micro_kernel_config->vec_c11        = reserved_zmms - 14;
      i_micro_kernel_config->vec_c1         = reserved_zmms - 15;
      i_micro_kernel_config->vec_c21        = reserved_zmms - 16;
      i_micro_kernel_config->vec_c2         = reserved_zmms - 17;
      i_micro_kernel_config->vec_tmp0       = reserved_zmms - 18;
      i_micro_kernel_config->vec_tmp1       = reserved_zmms - 19;
    } else if (l_is_sve_512) {
      reserved_zmms += 15;
      i_micro_kernel_config->vec_xr         = reserved_zmms - 1;
      i_micro_kernel_config->vec_xa         = reserved_zmms - 2;
      i_micro_kernel_config->vec_index      = reserved_zmms - 3;
      i_micro_kernel_config->vec_C0         = reserved_zmms - 4;
      i_micro_kernel_config->vec_C1         = reserved_zmms - 5;
      i_micro_kernel_config->vec_C2         = reserved_zmms - 6;
      i_micro_kernel_config->vec_thres      = reserved_zmms - 7;
      i_micro_kernel_config->vec_absmask    = reserved_zmms - 8;
      i_micro_kernel_config->vec_scale      = reserved_zmms - 9;
      i_micro_kernel_config->vec_shifter    = reserved_zmms - 10;
      i_micro_kernel_config->vec_halves     = reserved_zmms - 11;
      i_micro_kernel_config->vec_c0         = reserved_zmms - 12;
      i_micro_kernel_config->vec_c1         = reserved_zmms - 13;
      i_micro_kernel_config->vec_c2         = reserved_zmms - 14;
      i_micro_kernel_config->vec_tmp0       = reserved_zmms - 15;
    } else {
      reserved_zmms += 25;
      i_micro_kernel_config->vec_xr         = reserved_zmms - 1;
      i_micro_kernel_config->vec_xa         = reserved_zmms - 2;
      i_micro_kernel_config->vec_index      = reserved_zmms - 3;
      i_micro_kernel_config->vec_C0         = reserved_zmms - 4;
      i_micro_kernel_config->vec_C1         = reserved_zmms - 5;
      i_micro_kernel_config->vec_C2         = reserved_zmms - 6;
      i_micro_kernel_config->vec_thres      = reserved_zmms - 7;
      i_micro_kernel_config->vec_absmask    = reserved_zmms - 8;
      i_micro_kernel_config->vec_scale      = reserved_zmms - 9;
      i_micro_kernel_config->vec_shifter    = reserved_zmms - 10;
      i_micro_kernel_config->vec_halves     = reserved_zmms - 11;
      i_micro_kernel_config->vec_c03        = reserved_zmms - 12;
      i_micro_kernel_config->vec_c02        = reserved_zmms - 13;
      i_micro_kernel_config->vec_c01        = reserved_zmms - 14;
      i_micro_kernel_config->vec_c0         = reserved_zmms - 15;
      i_micro_kernel_config->vec_c13        = reserved_zmms - 16;
      i_micro_kernel_config->vec_c12        = reserved_zmms - 17;
      i_micro_kernel_config->vec_c11        = reserved_zmms - 18;
      i_micro_kernel_config->vec_c1         = reserved_zmms - 19;
      i_micro_kernel_config->vec_c23        = reserved_zmms - 20;
      i_micro_kernel_config->vec_c22        = reserved_zmms - 21;
      i_micro_kernel_config->vec_c21        = reserved_zmms - 22;
      i_micro_kernel_config->vec_c2         = reserved_zmms - 23;
      i_micro_kernel_config->vec_tmp0       = reserved_zmms - 24;
      i_micro_kernel_config->vec_tmp1       = reserved_zmms - 25;
    }

    if (op == LIBXSMM_MELTW_TYPE_UNARY_GELU ) {
      if (l_is_sve) {
        libxsmm_generator_prepare_coeffs_gelu_ps_minimax3_aarch64_sve( io_generated_code,
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
                                                                       i_gp_reg_tmp0,
                                                                       i_gp_reg_tmp1,
                                                                       l_sve_type,
                                                                       l_pred_reg );
      } else {
        libxsmm_generator_prepare_coeffs_gelu_ps_minimax3_aarch64_asimd( io_generated_code,
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
          i_micro_kernel_config->vec_tmp0,
          i_micro_kernel_config->vec_tmp1,
          i_gp_reg_tmp0,
          i_gp_reg_tmp1,
          l_tupletype );
      }
    }

    if (op == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV ) {
      if (l_is_sve) {
        libxsmm_generator_prepare_coeffs_gelu_inv_ps_minimax3_aarch64_sve( io_generated_code,
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
                                                                           i_micro_kernel_config->vec_tmp0,
                                                                           i_gp_reg_tmp0,
                                                                           i_gp_reg_tmp1,
                                                                           l_sve_type,
                                                                           l_pred_reg );
      } else {
        libxsmm_generator_prepare_coeffs_gelu_inv_ps_minimax3_aarch64_asimd( io_generated_code,
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
          i_micro_kernel_config->vec_tmp0,
          i_micro_kernel_config->vec_tmp1,
          i_gp_reg_tmp0,
          i_gp_reg_tmp1,
          l_tupletype );
      }
    }

    i_micro_kernel_config->reserved_zmms = reserved_zmms;
  }

  if ((op == LIBXSMM_MELTW_TYPE_UNARY_EXP) || (op == LIBXSMM_MELTW_TYPE_UNARY_ELU)) {
    unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;

    reserved_zmms += 9;
    i_micro_kernel_config->vec_halves     = reserved_zmms - 1;
    i_micro_kernel_config->vec_c0         = reserved_zmms - 2;
    i_micro_kernel_config->vec_c1         = reserved_zmms - 3;
    i_micro_kernel_config->vec_c2         = reserved_zmms - 4;
    i_micro_kernel_config->vec_c3         = reserved_zmms - 5;
    i_micro_kernel_config->vec_log2e      = reserved_zmms - 6;
    i_micro_kernel_config->vec_expmask    = reserved_zmms - 7;
    i_micro_kernel_config->vec_hi_bound   = reserved_zmms - 8;
    i_micro_kernel_config->vec_lo_bound   = reserved_zmms - 9;

    if (l_is_sve) {
      libxsmm_generator_prepare_coeffs_exp_ps_3dts_aarch64_sve( io_generated_code,
        i_micro_kernel_config->vec_c0,
        i_micro_kernel_config->vec_c1,
        i_micro_kernel_config->vec_c2,
        i_micro_kernel_config->vec_c3,
        i_micro_kernel_config->vec_halves,
        i_micro_kernel_config->vec_log2e,
        i_micro_kernel_config->vec_expmask,
        i_micro_kernel_config->vec_hi_bound,
        i_micro_kernel_config->vec_lo_bound,
        i_gp_reg_tmp0,
        l_sve_type, l_pred_reg );
    } else {
      libxsmm_generator_prepare_coeffs_exp_ps_3dts_aarch64_asimd( io_generated_code,
        i_micro_kernel_config->vec_c0,
        i_micro_kernel_config->vec_c1,
        i_micro_kernel_config->vec_c2,
        i_micro_kernel_config->vec_c3,
        i_micro_kernel_config->vec_halves,
        i_micro_kernel_config->vec_log2e,
        i_micro_kernel_config->vec_expmask,
        i_micro_kernel_config->vec_hi_bound,
        i_micro_kernel_config->vec_lo_bound,
        i_gp_reg_tmp0,
        l_tupletype );
    }
    i_micro_kernel_config->reserved_zmms = reserved_zmms;
  }
  if (op == LIBXSMM_MELTW_TYPE_UNARY_TANH || op == LIBXSMM_MELTW_TYPE_UNARY_TANH_INV) {
    unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;
    unsigned int reserved_mask_regs = i_micro_kernel_config->reserved_mask_regs;
    reserved_zmms += 17;

    i_micro_kernel_config->vec_x2        = reserved_zmms - 1;
    if ( l_is_sve ) {
      i_micro_kernel_config->mask_hi     = reserved_mask_regs++;
      i_micro_kernel_config->mask_lo     = reserved_mask_regs++;
    } else {
      i_micro_kernel_config->mask_hi     = reserved_zmms - 2;
      i_micro_kernel_config->mask_lo     = reserved_zmms - 3;
    }
    i_micro_kernel_config->vec_nom       = reserved_zmms - 4;
    i_micro_kernel_config->vec_denom     = reserved_zmms - 5;
    i_micro_kernel_config->vec_c0        = reserved_zmms - 6;
    i_micro_kernel_config->vec_c1        = reserved_zmms - 7;
    i_micro_kernel_config->vec_c2        = reserved_zmms - 8;
    i_micro_kernel_config->vec_c3        = reserved_zmms - 9;
    i_micro_kernel_config->vec_c1_d      = reserved_zmms - 10;
    i_micro_kernel_config->vec_c2_d      = reserved_zmms - 11;
    i_micro_kernel_config->vec_c3_d      = reserved_zmms - 12;
    i_micro_kernel_config->vec_hi_bound  = reserved_zmms - 13;
    i_micro_kernel_config->vec_lo_bound  = reserved_zmms - 14;
    i_micro_kernel_config->vec_ones      = reserved_zmms - 15;
    i_micro_kernel_config->vec_neg_ones  = reserved_zmms - 16;
    i_micro_kernel_config->vec_tmp0      = reserved_zmms - 17;

    if (l_is_sve) {
      libxsmm_generator_prepare_coeffs_tanh_ps_rational_78_aarch64_sve( io_generated_code,
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
        i_gp_reg_tmp1,
        l_sve_type, l_pred_reg );
    } else {
      libxsmm_generator_prepare_coeffs_tanh_ps_rational_78_aarch64_asimd( io_generated_code,
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
        i_gp_reg_tmp1,
        l_tupletype );
    }

    i_micro_kernel_config->reserved_mask_regs = reserved_mask_regs;
    i_micro_kernel_config->reserved_zmms = reserved_zmms;
  }

  if (op == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID || op == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV) {
    unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;
    unsigned int reserved_mask_regs = i_micro_kernel_config->reserved_mask_regs;
    reserved_zmms += 18;

    i_micro_kernel_config->vec_x2        = reserved_zmms - 1;
    if ( l_is_sve ) {
      i_micro_kernel_config->mask_hi     = reserved_mask_regs++;
      i_micro_kernel_config->mask_lo     = reserved_mask_regs++;
    } else {
      i_micro_kernel_config->mask_hi     = reserved_zmms - 2;
      i_micro_kernel_config->mask_lo     = reserved_zmms - 3;
    }
    i_micro_kernel_config->vec_nom       = reserved_zmms - 4;
    i_micro_kernel_config->vec_denom     = reserved_zmms - 5;
    i_micro_kernel_config->vec_c0        = reserved_zmms - 6;
    i_micro_kernel_config->vec_c1        = reserved_zmms - 7;
    i_micro_kernel_config->vec_c2        = reserved_zmms - 8;
    i_micro_kernel_config->vec_c3        = reserved_zmms - 9;
    i_micro_kernel_config->vec_c1_d      = reserved_zmms - 10;
    i_micro_kernel_config->vec_c2_d      = reserved_zmms - 11;
    i_micro_kernel_config->vec_c3_d      = reserved_zmms - 12;
    i_micro_kernel_config->vec_hi_bound  = reserved_zmms - 13;
    i_micro_kernel_config->vec_lo_bound  = reserved_zmms - 14;
    i_micro_kernel_config->vec_ones      = reserved_zmms - 15;
    i_micro_kernel_config->vec_neg_ones  = reserved_zmms - 16;
    i_micro_kernel_config->vec_tmp0      = reserved_zmms - 17;
    i_micro_kernel_config->vec_halves    = reserved_zmms - 18;

    if (l_is_sve) {
      libxsmm_generator_prepare_coeffs_sigmoid_ps_rational_78_aarch64_sve( io_generated_code,
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
        i_gp_reg_tmp1,
        l_sve_type, l_pred_reg );
    } else {
      libxsmm_generator_prepare_coeffs_sigmoid_ps_rational_78_aarch64_asimd( io_generated_code,
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
        i_gp_reg_tmp1,
        l_tupletype );
    }

    i_micro_kernel_config->reserved_mask_regs = reserved_mask_regs;
    i_micro_kernel_config->reserved_zmms = reserved_zmms;
  }

  if (op == LIBXSMM_MELTW_TYPE_UNARY_QUANT || op == LIBXSMM_MELTW_TYPE_UNARY_DEQUANT) {
    if ((flags & LIBXSMM_MELTW_FLAG_UNARY_NO_SCF_QUANT) == 0) {
      i_micro_kernel_config->quant_vreg_scf = i_micro_kernel_config->reserved_zmms;
      i_micro_kernel_config->reserved_zmms++;
      /* TODO: need fixing in case of different scaling (per channel etc) */
      if (l_is_sve > 0){
          libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF,
                                                i_gp_reg_aux0, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_micro_kernel_config->quant_vreg_scf, 0 );
      } else {
        /* Should not happen */
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_finalize_unary_aarch64_kernel_vregs_masks( libxsmm_generated_code*                 io_generated_code,
                                                        libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                        unsigned int                            op,
                                                        unsigned int                            flags,
                                                        unsigned int                            i_gp_reg_tmp,
                                                        const unsigned int                      i_gp_reg_aux0,
                                                        const unsigned int                      i_gp_reg_aux1 ) {

  LIBXSMM_UNUSED(flags);
  LIBXSMM_UNUSED(i_gp_reg_tmp);
  LIBXSMM_UNUSED(i_gp_reg_aux1);

  if ( op == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT ) {
    libxsmm_generator_store_prng_state_aarch64_asimd( io_generated_code, i_gp_reg_aux0,
                                                      i_micro_kernel_config->prng_state0_vreg, i_micro_kernel_config->prng_state1_vreg,
                                                      i_micro_kernel_config->prng_state2_vreg, i_micro_kernel_config->prng_state3_vreg );
  }
}

LIBXSMM_API_INTERN
void libxsmm_configure_aarch64_kernel_vregs_masks( libxsmm_generated_code*                       io_generated_code,
                                                 libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_gp_reg_tmp0,
                                                 unsigned int                            i_gp_reg_tmp1,
                                                 const unsigned int                      i_gp_reg_aux0,
                                                 const unsigned int                      i_gp_reg_aux1) {
  /* initialize some values */
  i_micro_kernel_config->reserved_zmms = 0;
  i_micro_kernel_config->reserved_mask_regs = 2;
  i_micro_kernel_config->use_fp32bf16_cvt_replacement = 0;

  /* if we need FP32->BF16 downconverts and we do not have native instruction, then prepare stack */
#if 0
  if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) || LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) &&
       LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) && (io_generated_code->arch < LIBXSMM_X86_AVX512_CPX)) {
    i_micro_kernel_config->use_fp32bf16_cvt_replacement = 1;
    libxsmm_generator_vcvtneps2bf16_avx512_prep_stack( io_generated_code, i_gp_reg_tmp );
    i_micro_kernel_config->dcvt_mask_aux0 = i_micro_kernel_config->reserved_mask_regs;
    i_micro_kernel_config->dcvt_mask_aux1 = i_micro_kernel_config->reserved_mask_regs + 1;
    i_micro_kernel_config->reserved_mask_regs = i_micro_kernel_config->reserved_mask_regs + 2;
    i_micro_kernel_config->dcvt_zmm_aux0 = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->dcvt_zmm_aux1 = i_micro_kernel_config->reserved_zmms + 1;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 2;
  }
#endif
  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
    libxsmm_configure_unary_aarch64_kernel_vregs_masks( io_generated_code, i_micro_kernel_config, i_mateltwise_desc->param, i_mateltwise_desc->flags, i_gp_reg_tmp0, i_gp_reg_tmp1, i_gp_reg_aux0, i_gp_reg_aux1);
  }

  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
    /* This is the temp register used to load the second input */
    i_micro_kernel_config->tmp_vreg = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;

    if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY && i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_TERNARY_SELECT) ||
        (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY && (libxsmm_generator_mateltwise_is_binary_cmp_op(i_mateltwise_desc) > 0))) {
      unsigned char l_is_sve = (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT);
      i_micro_kernel_config->tmp_vreg2 = i_micro_kernel_config->reserved_zmms;
      i_micro_kernel_config->mask_helper0_vreg =  i_micro_kernel_config->reserved_zmms + 1;
      i_micro_kernel_config->mask_helper1_vreg =  i_micro_kernel_config->reserved_zmms + 2;
      i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 3;
      if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY && (libxsmm_generator_mateltwise_is_binary_cmp_op(i_mateltwise_desc) > 0)) {
        i_micro_kernel_config->tmp_vreg3 = i_micro_kernel_config->reserved_zmms;
        i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
      }
      if ( l_is_sve == 0 ) {
        /* load 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80 into mask_helper0/1_vreg */
        /* stack pointer -= 32, so prepare to use 32 bytes = 8 floats of stack memory */
        /* while LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF supports a signed offset, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF does not */
        libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I,
                                                       LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 32, 0 );
        libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp0, 0x200000001 ); /* int32 0x02, 0x01, little endian -> 0x01, 0x02 */
        libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp1, 0x800000004 ); /* int32 0x08, 0x04, little endian -> 0x04, 0x08 */
        /* store a pair of fp registers to memory: 1,2,4,8 is being loaded into the stack memory */
        libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF,
                                                   LIBXSMM_AARCH64_GP_REG_XSP, 16,
                                                   i_gp_reg_tmp0, i_gp_reg_tmp1 );
        /* now those 32 bytes are stored into mask_helper0_vreg */
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                                LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_UNDEF, 16,
                                                i_micro_kernel_config->mask_helper0_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
        libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp0, 0x2000000010 ); /* int32 0x20, 0x10 */
        libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp1, 0x8000000040 ); /* int32 0x80, 0x40 */
        libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF,
                                                   LIBXSMM_AARCH64_GP_REG_XSP, 0,
                                                   i_gp_reg_tmp0, i_gp_reg_tmp1 );
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                                LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_UNDEF, 0,
                                                i_micro_kernel_config->mask_helper1_vreg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
        /* reset stack pointer to its original position */
        libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                       LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 32, 0 );
      }
    }
    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY && i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_MULADD) {
      i_micro_kernel_config->tmp_vreg2 = i_micro_kernel_config->reserved_zmms;
      i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_finalize_kernel_vregs_aarch64_masks( libxsmm_generated_code*                 io_generated_code,
                                                  libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                  const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                  unsigned int                            i_gp_reg_tmp,
                                                  const unsigned int                      i_gp_reg_aux0,
                                                  const unsigned int                      i_gp_reg_aux1) {
  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
    libxsmm_finalize_unary_aarch64_kernel_vregs_masks( io_generated_code, i_micro_kernel_config, i_mateltwise_desc->param, i_mateltwise_desc->flags, i_gp_reg_tmp, i_gp_reg_aux0, i_gp_reg_aux1);
  }
#if 0
  if (i_micro_kernel_config->use_fp32bf16_cvt_replacement == 1) {
    libxsmm_generator_vcvtneps2bf16_avx512_clean_stack( io_generated_code, i_gp_reg_tmp );
  }
#endif
}

LIBXSMM_API_INTERN
void libxsmm_generator_unary_aarch64_binary_2d_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                            libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                            libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                            libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                            const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                            unsigned int                            i_m,
                                                            unsigned int                            i_n) {

  unsigned int use_m_input_masking, use_m_output_masking, m_trips, m_unroll_factor, m_assm_trips, n_trips, n_unroll_factor, n_assm_trips;
  unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;
  unsigned int out_loop_trips, inner_loop_trips, out_loop_reg, inner_loop_reg, out_loop_bound, inner_loop_bound, out_unroll_factor, inner_unroll_factor;
  unsigned int mask_reg_in, mask_reg_out;
  unsigned int i_vlen_in = i_micro_kernel_config->vlen_in;
  unsigned int i_vlen_out = i_micro_kernel_config->vlen_out;
  unsigned int loop_type;

  use_m_input_masking = 0;
  use_m_output_masking = 0;
  mask_reg_in = LIBXSMM_AARCH64_GP_REG_UNDEF;
  mask_reg_out = LIBXSMM_AARCH64_GP_REG_UNDEF;

  /* Configure microkernel masks */
  libxsmm_setup_input_output_aarch64_masks( io_generated_code, i_micro_kernel_config, i_mateltwise_desc,
    i_gp_reg_mapping->gp_reg_scratch_0, i_m, &use_m_input_masking, &mask_reg_in, &use_m_output_masking, &mask_reg_out);
  reserved_zmms = i_micro_kernel_config->reserved_zmms;

  /* Configure microkernel loops */
  libxsmm_configure_microkernel_aarch64_loops( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc, i_m, i_n, use_m_input_masking,
    &m_trips, &n_trips, &m_unroll_factor, &n_unroll_factor, &m_assm_trips, &n_assm_trips,
    &out_loop_trips, &inner_loop_trips, &out_loop_bound, &inner_loop_bound, &out_loop_reg, &inner_loop_reg, &out_unroll_factor, &inner_unroll_factor );

  /* Headers of microkernel loops */
  if (out_loop_trips > 1) {
    libxsmm_generator_loop_header_aarch64(io_generated_code, io_loop_label_tracker, out_loop_reg, out_loop_bound);
  }

  if (inner_loop_trips > 1) {
    libxsmm_generator_loop_header_aarch64(io_generated_code, io_loop_label_tracker, inner_loop_reg, inner_loop_bound);
  }

  /* Load block of registers */
  libxsmm_load_aarch64_2d_reg_block(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
      i_vlen_in, reserved_zmms, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in);

  /* Compute on registers */
  libxsmm_compute_unary_binary_aarch64_2d_reg_block(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
      i_vlen_in, reserved_zmms, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in);

  /* Store block of registers */
  libxsmm_store_aarch64_2d_reg_block(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
      i_vlen_out, reserved_zmms, m_unroll_factor, n_unroll_factor, use_m_output_masking, mask_reg_out);


  /* Footers of microkernel loops */
  if (inner_loop_trips > 1) {
    /* Advance input/output pointers */
    loop_type = (inner_loop_reg == i_gp_reg_mapping->gp_reg_m_loop) ? LOOP_TYPE_M : LOOP_TYPE_N;

    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_in, LIBXSMM_AARCH64_INSTR_GP_META_ADD, inner_unroll_factor, loop_type);

    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_out, LIBXSMM_AARCH64_INSTR_GP_META_ADD, inner_unroll_factor, loop_type);

    if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP)) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_out2, LIBXSMM_AARCH64_INSTR_GP_META_ADD, inner_unroll_factor, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in2, LIBXSMM_AARCH64_INSTR_GP_META_ADD, inner_unroll_factor, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in3, LIBXSMM_AARCH64_INSTR_GP_META_ADD, inner_unroll_factor, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {/* adjust relu/dropout mask pointers */
      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) ||
          (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
          (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU)        || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_AARCH64_INSTR_GP_META_ADD, inner_unroll_factor, loop_type);
      } else if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_dropoutmask, LIBXSMM_AARCH64_INSTR_GP_META_ADD, inner_unroll_factor, loop_type);
      }
    }

    libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, inner_loop_reg, inner_unroll_factor);

    /* Reset input/output pointers */
    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_in, LIBXSMM_AARCH64_INSTR_GP_META_SUB, inner_unroll_factor * inner_loop_trips, loop_type);

    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_out, LIBXSMM_AARCH64_INSTR_GP_META_SUB, inner_unroll_factor * inner_loop_trips, loop_type);

    if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP)) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_out2, LIBXSMM_AARCH64_INSTR_GP_META_SUB, inner_unroll_factor * inner_loop_trips, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in2, LIBXSMM_AARCH64_INSTR_GP_META_SUB, inner_unroll_factor * inner_loop_trips, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in3, LIBXSMM_AARCH64_INSTR_GP_META_SUB, inner_unroll_factor * inner_loop_trips, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) ||
          (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
          (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU)        || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_AARCH64_INSTR_GP_META_SUB, inner_unroll_factor * inner_loop_trips, loop_type);
      } else if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_dropoutmask, LIBXSMM_AARCH64_INSTR_GP_META_SUB, inner_unroll_factor * inner_loop_trips, loop_type);
      }
    }
  }

  if (out_loop_trips > 1) {
    /* Advance input/output pointers */
    loop_type = (out_loop_reg == i_gp_reg_mapping->gp_reg_m_loop) ? LOOP_TYPE_M : LOOP_TYPE_N;

    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_in, LIBXSMM_AARCH64_INSTR_GP_META_ADD, out_unroll_factor, loop_type);

    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_out, LIBXSMM_AARCH64_INSTR_GP_META_ADD, out_unroll_factor, loop_type);

    if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP)) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_out2, LIBXSMM_AARCH64_INSTR_GP_META_ADD, out_unroll_factor, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in2, LIBXSMM_AARCH64_INSTR_GP_META_ADD, out_unroll_factor, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in3, LIBXSMM_AARCH64_INSTR_GP_META_ADD, out_unroll_factor, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) ||
          (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
          (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU)        || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_AARCH64_INSTR_GP_META_ADD, out_unroll_factor, loop_type);
      } else if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_dropoutmask, LIBXSMM_AARCH64_INSTR_GP_META_ADD, out_unroll_factor, loop_type);
      }
    }

    libxsmm_generator_loop_footer_aarch64(io_generated_code, io_loop_label_tracker, out_loop_reg, out_unroll_factor);

    /* Reset input/output pointers */
    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_in, LIBXSMM_AARCH64_INSTR_GP_META_SUB, out_unroll_factor * out_loop_trips, loop_type);

    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_out, LIBXSMM_AARCH64_INSTR_GP_META_SUB, out_unroll_factor * out_loop_trips, loop_type);

    if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP)) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_out2, LIBXSMM_AARCH64_INSTR_GP_META_SUB, out_unroll_factor * out_loop_trips, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in2, LIBXSMM_AARCH64_INSTR_GP_META_SUB, out_unroll_factor * out_loop_trips, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in3, LIBXSMM_AARCH64_INSTR_GP_META_SUB, out_unroll_factor * out_loop_trips, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) ||
          (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
          (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU)        || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_AARCH64_INSTR_GP_META_SUB, out_unroll_factor * out_loop_trips, loop_type);
      } else if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_dropoutmask, LIBXSMM_AARCH64_INSTR_GP_META_SUB, out_unroll_factor * out_loop_trips, loop_type);
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_unary_binary_aarch64_microkernel( libxsmm_generated_code*          io_generated_code,
                                                 libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int loop_order, m_blocking = 0, out_blocking = 0, out_bound = 0, out_block = 0, n_blocking = 0, inner_blocking = 0, inner_block = 0, inner_bound = 0, n_microkernel = 0, m_microkernel = 0;
  unsigned int out_ind, inner_ind, reset_regs, loop_type;
  unsigned int available_vregs = 32;
  unsigned int l_gp_reg_tmp = LIBXSMM_AARCH64_GP_REG_X16;
  unsigned int l_gp_reg_aux0 = LIBXSMM_AARCH64_GP_REG_UNDEF;
  unsigned int l_gp_reg_aux1 = LIBXSMM_AARCH64_GP_REG_UNDEF;

  /* Some rudimentary checking of M, N and LDs*/
#if 0
  if ( (i_mateltwise_desc->m > i_mateltwise_desc->ldi) ||
       (i_mateltwise_desc->m > i_mateltwise_desc->ldo)    ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
    return;
  }
#endif

  /* check datatype */
  if ( ((( LIBXSMM_DATATYPE_F32  == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || ( LIBXSMM_DATATYPE_U32  == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || ( LIBXSMM_DATATYPE_I32  == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || ( LIBXSMM_DATATYPE_BF16  == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || ( LIBXSMM_DATATYPE_U16  == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || ( LIBXSMM_DATATYPE_I16  == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)))
       &&
       ((LIBXSMM_DATATYPE_F32  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_BF16  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_U16  == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)))) ||
       ( LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) && LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ||
       ( LIBXSMM_DATATYPE_U16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) && LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ||
       ( LIBXSMM_DATATYPE_U16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) && LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ||
       ( LIBXSMM_DATATYPE_U16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) && LIBXSMM_DATATYPE_U32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ||
       ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) && LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ||
       ( LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) && LIBXSMM_DATATYPE_F32  == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ||
       ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) && LIBXSMM_DATATYPE_I8  == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ||
       ( (libxsmm_generator_mateltwise_is_binary_cmp_op(i_mateltwise_desc) > 0) && (LIBXSMM_DATATYPE_IMPLICIT == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ) ||
       ( LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) && LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ) {
    /* fine */
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }

  /* Configure vlens */
  libxsmm_generator_configure_aarch64_vlens(i_mateltwise_desc, i_micro_kernel_config);

  /* set mask lds */
  if ( (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) ) {
    i_micro_kernel_config->ldo_mask = (i_mateltwise_desc->ldo+15) - ((i_mateltwise_desc->ldo+15)%16);
    i_micro_kernel_config->ldi_mask = (i_mateltwise_desc->ldi+15) - ((i_mateltwise_desc->ldi+15)%16);
  }

  if ((libxsmm_generator_mateltwise_is_binary_cmp_op(i_mateltwise_desc) > 0) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BITMASK_2BYTEMULT) > 0)) {
    i_micro_kernel_config->ldo_mask = (i_mateltwise_desc->ldo+15) - ((i_mateltwise_desc->ldo+15)%16);
  }

  /* let's check that we have bitmask set for dropout and relu backward */
  if ( ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
         (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV) ) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) == 0) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BITMASK_REQUIRED );
    return;
  }

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in        = LIBXSMM_AARCH64_GP_REG_X8;
  i_gp_reg_mapping->gp_reg_out       = LIBXSMM_AARCH64_GP_REG_X9;
  i_gp_reg_mapping->gp_reg_m_loop    = LIBXSMM_AARCH64_GP_REG_X10;
  i_gp_reg_mapping->gp_reg_n_loop    = LIBXSMM_AARCH64_GP_REG_X11;
  i_gp_reg_mapping->gp_reg_scratch_0 = LIBXSMM_AARCH64_GP_REG_X16;
  i_gp_reg_mapping->gp_reg_scratch_1 = LIBXSMM_AARCH64_GP_REG_X17;
  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY ) {
    i_gp_reg_mapping->gp_reg_in2  = LIBXSMM_AARCH64_GP_REG_X12;
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY ) {
    i_gp_reg_mapping->gp_reg_in2  = LIBXSMM_AARCH64_GP_REG_X12;
    i_gp_reg_mapping->gp_reg_in3  = LIBXSMM_AARCH64_GP_REG_X13;
  } else {
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP) {
      i_gp_reg_mapping->gp_reg_out2 = LIBXSMM_AARCH64_GP_REG_X12;
    }
    if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV)       ||
         (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV)    ) {
      i_gp_reg_mapping->gp_reg_relumask = LIBXSMM_AARCH64_GP_REG_X12;
    }
    if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
      i_gp_reg_mapping->gp_reg_relumask = LIBXSMM_AARCH64_GP_REG_X12;
      i_gp_reg_mapping->gp_reg_fam_lualpha = LIBXSMM_AARCH64_GP_REG_X13;
    }
    if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ) {
      i_gp_reg_mapping->gp_reg_fam_lualpha = LIBXSMM_AARCH64_GP_REG_X13;
    }
    if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_UNZIP ) {
      i_gp_reg_mapping->gp_reg_offset = LIBXSMM_AARCH64_GP_REG_X12;
    }
    if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X2 ) {
      i_gp_reg_mapping->gp_reg_offset = LIBXSMM_AARCH64_GP_REG_X12;
      i_gp_reg_mapping->gp_reg_shift_vals = LIBXSMM_AARCH64_GP_REG_X14;
    }
    if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3 ) {
      i_gp_reg_mapping->gp_reg_offset = LIBXSMM_AARCH64_GP_REG_X12;
      i_gp_reg_mapping->gp_reg_offset_2 = LIBXSMM_AARCH64_GP_REG_X13;
      i_gp_reg_mapping->gp_reg_shift_vals = LIBXSMM_AARCH64_GP_REG_X14;
      i_gp_reg_mapping->gp_reg_shift_vals2 = LIBXSMM_AARCH64_GP_REG_X15;
    }
    if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV) ) {
      i_gp_reg_mapping->gp_reg_dropoutmask = LIBXSMM_AARCH64_GP_REG_X12;
      i_gp_reg_mapping->gp_reg_dropoutprob = LIBXSMM_AARCH64_GP_REG_X13;
      if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT ) {
        i_gp_reg_mapping->gp_reg_prngstate = LIBXSMM_AARCH64_GP_REG_X14;
      } else {
        i_gp_reg_mapping->gp_reg_prngstate = LIBXSMM_AARCH64_GP_REG_UNDEF;
      }
    } else {
      i_gp_reg_mapping->gp_reg_dropoutmask = LIBXSMM_AARCH64_GP_REG_UNDEF;
      i_gp_reg_mapping->gp_reg_dropoutprob = LIBXSMM_AARCH64_GP_REG_UNDEF;
      i_gp_reg_mapping->gp_reg_prngstate = LIBXSMM_AARCH64_GP_REG_UNDEF;
    }
    if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_QUANT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DEQUANT) ) {
      i_gp_reg_mapping->gp_reg_quant_sf = LIBXSMM_AARCH64_GP_REG_X12;
    } else {
      i_gp_reg_mapping->gp_reg_quant_sf = LIBXSMM_X86_GP_REG_UNDEF;
    }
  }

  /* load the input pointer and output pointer */
  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY ) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 32, i_gp_reg_mapping->gp_reg_in );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 64, i_gp_reg_mapping->gp_reg_out );
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 72, i_gp_reg_mapping->gp_reg_out2 );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 72, i_gp_reg_mapping->gp_reg_relumask );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 40, i_gp_reg_mapping->gp_reg_relumask );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU)  {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_gp_reg_mapping->gp_reg_fam_lualpha );
      l_gp_reg_aux1 = i_gp_reg_mapping->gp_reg_fam_lualpha;
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 40, i_gp_reg_mapping->gp_reg_relumask );
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_gp_reg_mapping->gp_reg_fam_lualpha );
      l_gp_reg_aux1 = i_gp_reg_mapping->gp_reg_fam_lualpha;
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_gp_reg_mapping->gp_reg_fam_lualpha );
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 72, i_gp_reg_mapping->gp_reg_relumask );
      l_gp_reg_aux1 = i_gp_reg_mapping->gp_reg_fam_lualpha;
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 40, i_gp_reg_mapping->gp_reg_relumask );
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_gp_reg_mapping->gp_reg_fam_lualpha );
      l_gp_reg_aux1 = i_gp_reg_mapping->gp_reg_fam_lualpha;
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_UNZIP) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 72, i_gp_reg_mapping->gp_reg_offset );
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_offset,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_gp_reg_mapping->gp_reg_offset );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X2) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 72, i_gp_reg_mapping->gp_reg_offset );
      l_gp_reg_aux0 = i_gp_reg_mapping->gp_reg_offset;
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 72, i_gp_reg_mapping->gp_reg_offset );
      l_gp_reg_aux0 = i_gp_reg_mapping->gp_reg_offset;
      l_gp_reg_aux1 = i_gp_reg_mapping->gp_reg_offset_2;
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 8, i_gp_reg_mapping->gp_reg_prngstate );
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_gp_reg_mapping->gp_reg_dropoutprob );
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 72, i_gp_reg_mapping->gp_reg_dropoutmask );
      l_gp_reg_aux0 = i_gp_reg_mapping->gp_reg_prngstate;
      l_gp_reg_aux1 = i_gp_reg_mapping->gp_reg_dropoutprob;
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 40, i_gp_reg_mapping->gp_reg_dropoutmask );
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_gp_reg_mapping->gp_reg_dropoutprob );
      l_gp_reg_aux1 = i_gp_reg_mapping->gp_reg_dropoutprob;
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_QUANT) {
      if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_NO_SCF_QUANT) == 0) {
        libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                              LIBXSMM_AARCH64_GP_REG_UNDEF, 40, i_gp_reg_mapping->gp_reg_quant_sf );
      }
      l_gp_reg_aux0 = i_gp_reg_mapping->gp_reg_quant_sf;
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DEQUANT) {
      if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_NO_SCF_QUANT) == 0) {
        libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                              LIBXSMM_AARCH64_GP_REG_UNDEF, 40, i_gp_reg_mapping->gp_reg_quant_sf );
      }
      l_gp_reg_aux0 = i_gp_reg_mapping->gp_reg_quant_sf;
    }
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY ) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 32, i_gp_reg_mapping->gp_reg_in );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 64, i_gp_reg_mapping->gp_reg_in2 );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 96, i_gp_reg_mapping->gp_reg_out );
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY ) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 32, i_gp_reg_mapping->gp_reg_in );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 64, i_gp_reg_mapping->gp_reg_in2 );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 96, i_gp_reg_mapping->gp_reg_in3 );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 128, i_gp_reg_mapping->gp_reg_out );
  } else {
    /* This should not happen */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }

  /* Based on kernel type reserve zmms and mask registers */
  libxsmm_configure_aarch64_kernel_vregs_masks( io_generated_code, i_micro_kernel_config, i_mateltwise_desc, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_scratch_1, l_gp_reg_aux0, l_gp_reg_aux1 );

  available_vregs = available_vregs - i_micro_kernel_config->reserved_zmms;

  /* Configure M and N blocking factors */
  /* todo sve: here we might intercept the m, n blocking size */
  libxsmm_generator_configure_aarch64_M_N_blocking( io_generated_code, i_mateltwise_desc, i_mateltwise_desc->m, i_mateltwise_desc->n, i_micro_kernel_config->vlen_in, &m_blocking, &n_blocking, available_vregs);
  libxsmm_generator_configure_aarch64_loop_order(i_mateltwise_desc, &loop_order, &m_blocking, &n_blocking, &out_blocking, &inner_blocking, &out_bound, &inner_bound);
  i_micro_kernel_config->loop_order = loop_order;

  out_ind = 0;
  while ( out_ind != out_bound ) {
    inner_ind = 0;
    reset_regs = 0;
    while( inner_ind != inner_bound ) {

      out_block = (out_ind < out_blocking) ? out_blocking : out_bound - out_ind;
      inner_block  = (inner_ind < inner_blocking ) ? inner_blocking : inner_bound - inner_ind;
      n_microkernel = (loop_order == NM_LOOP_ORDER) ? out_block : inner_block;
      m_microkernel = (loop_order == MN_LOOP_ORDER) ? out_block : inner_block;


      libxsmm_generator_unary_aarch64_binary_2d_microkernel(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc, m_microkernel, n_microkernel);

      inner_ind += inner_block;

      if (inner_ind != inner_bound) {
        reset_regs = 1;
        /* Advance input/output pointers */
        loop_type = (loop_order == NM_LOOP_ORDER) ? LOOP_TYPE_M : LOOP_TYPE_N;

        libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_in, LIBXSMM_AARCH64_INSTR_GP_META_ADD, m_microkernel, n_microkernel, loop_type );

        libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_out, LIBXSMM_AARCH64_INSTR_GP_META_ADD, m_microkernel, n_microkernel, loop_type );

        if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP)) {
          libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
              i_gp_reg_mapping->gp_reg_out2, LIBXSMM_AARCH64_INSTR_GP_META_ADD, m_microkernel, n_microkernel, loop_type );
        }

        if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
          libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
              i_gp_reg_mapping->gp_reg_in2, LIBXSMM_AARCH64_INSTR_GP_META_ADD, m_microkernel, n_microkernel, loop_type );
        }

        if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
          libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
              i_gp_reg_mapping->gp_reg_in3, LIBXSMM_AARCH64_INSTR_GP_META_ADD, m_microkernel, n_microkernel, loop_type );
        }

        if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
          if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) ||
              (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
              (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU)        || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
            libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
                i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_AARCH64_INSTR_GP_META_ADD, m_microkernel, n_microkernel, loop_type );
          } else if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
            libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
                i_gp_reg_mapping->gp_reg_dropoutmask, LIBXSMM_AARCH64_INSTR_GP_META_ADD, m_microkernel, n_microkernel, loop_type );
          }
        }
      }
    }

    /* If needed, readjust the registers */
    if (reset_regs == 1) {
      loop_type = (loop_order == NM_LOOP_ORDER) ? LOOP_TYPE_M : LOOP_TYPE_N;

      libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in, LIBXSMM_AARCH64_INSTR_GP_META_SUB, m_microkernel, n_microkernel, loop_type );

      libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_out, LIBXSMM_AARCH64_INSTR_GP_META_SUB, m_microkernel, n_microkernel, loop_type );

      if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP)) {
        libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_out2, LIBXSMM_AARCH64_INSTR_GP_META_SUB, m_microkernel, n_microkernel, loop_type );
      }

      if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
        libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_in2, LIBXSMM_AARCH64_INSTR_GP_META_SUB, m_microkernel, n_microkernel, loop_type );
      }

      if ( i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
        libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_in3, LIBXSMM_AARCH64_INSTR_GP_META_SUB, m_microkernel, n_microkernel, loop_type );
      }

      if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
        if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) ||
            (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
            (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU)        || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
          libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
              i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_AARCH64_INSTR_GP_META_SUB, m_microkernel, n_microkernel, loop_type );
        } else if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
          libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
              i_gp_reg_mapping->gp_reg_dropoutmask, LIBXSMM_AARCH64_INSTR_GP_META_SUB, m_microkernel, n_microkernel, loop_type );
        }
      }
    }

    out_ind += out_block;
    if (out_ind != out_bound) {
      /* Advance input/output pointers */
      loop_type = (loop_order == MN_LOOP_ORDER) ? LOOP_TYPE_M : LOOP_TYPE_N;

      libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in, LIBXSMM_AARCH64_INSTR_GP_META_ADD, m_microkernel, n_microkernel, loop_type );

      libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_out, LIBXSMM_AARCH64_INSTR_GP_META_ADD, m_microkernel, n_microkernel, loop_type );

      if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP)) {
        libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_out2, LIBXSMM_AARCH64_INSTR_GP_META_ADD, m_microkernel, n_microkernel, loop_type );
      }

      if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
        libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_in2, LIBXSMM_AARCH64_INSTR_GP_META_ADD, m_microkernel, n_microkernel, loop_type );
      }

      if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
        libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_in3, LIBXSMM_AARCH64_INSTR_GP_META_ADD, m_microkernel, n_microkernel, loop_type );
      }

      if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
        if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) ||
            (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
            (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU)        || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
          libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
              i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_AARCH64_INSTR_GP_META_ADD, m_microkernel, n_microkernel, loop_type );
        } else if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
          libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
              i_gp_reg_mapping->gp_reg_dropoutmask, LIBXSMM_AARCH64_INSTR_GP_META_ADD, m_microkernel, n_microkernel, loop_type );
        }
      }
    }
  }

  /* save some globale state if needed */
  libxsmm_finalize_kernel_vregs_aarch64_masks( io_generated_code, i_micro_kernel_config, i_mateltwise_desc, l_gp_reg_tmp, l_gp_reg_aux0, l_gp_reg_aux1 );
}
