/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_common_x86.h"
#include "generator_mateltwise_unary_binary_avx_avx512.h"
#include "generator_mateltwise_sse_avx_avx512.h"
#include "generator_mateltwise_common.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"

#define MN_LOOP_ORDER 0
#define NM_LOOP_ORDER 1
#define LOOP_TYPE_M 0
#define LOOP_TYPE_N 1

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_gp_reg( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_gp_reg,
                                                 unsigned int                            i_adjust_instr,
                                                 unsigned int                            i_m_microkernel,
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
  unsigned int m_microkernel = i_m_microkernel;

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
        libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, (long long)m_microkernel * tsize);
      } else {
        libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, (long long)ld * n_microkernel * tsize);
      }
    } else {
      if (bcast_row > 0) {
        if (i_loop_type == LOOP_TYPE_N) {
          libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, (long long)ld * n_microkernel * tsize);
        }
      }
      if (bcast_col > 0) {
        if (i_loop_type == LOOP_TYPE_M) {
          libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, (long long)m_microkernel * tsize);
        }
      }
    }
  } else {
    /* Advance relumasks if need be */
    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY ) {
      if ( ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) )
           && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0)) {
        if (i_loop_type == LOOP_TYPE_M) {
          libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, (long long)m_microkernel/8);
        } else {
          libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, ((long long)i_micro_kernel_config->ldo_mask * n_microkernel)/8);
        }
      }

      if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV)       ||
           (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
           (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV)           ) {
        if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && (i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, (long long)m_microkernel/8);
          } else {
            libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, ((long long)i_micro_kernel_config->ldi_mask * n_microkernel)/8);
          }
        } else {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, (long long)m_microkernel * i_micro_kernel_config->datatype_size_in);
          } else {
            libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, (long long)i_mateltwise_desc->ldi * n_microkernel * i_micro_kernel_config->datatype_size_in );
          }
        }
      }

      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0)) {
        if (i_loop_type == LOOP_TYPE_M) {
          libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, (long long)m_microkernel/8);
        } else {
          libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, ((long long)i_micro_kernel_config->ldo_mask * n_microkernel)/8);
        }
      }

      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, (long long)m_microkernel/8);
          } else {
            libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, ((long long)i_micro_kernel_config->ldi_mask * n_microkernel)/8);
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
void libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( libxsmm_generated_code*                 io_generated_code,
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
        libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, m_adjust);
      } else {
        libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, n_adjust);
      }
    } else {
      if (bcast_row > 0) {
        if (i_loop_type == LOOP_TYPE_N) {
          libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, n_adjust);
        }
      }
      if (bcast_col > 0) {
        if (i_loop_type == LOOP_TYPE_M) {
          libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, m_adjust);
        }
      }
    }
  } else {
    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY ) {
      if (((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU)) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0)) {
        /* TODO: Evangelos: why us here  i_gp_reg_mapping->gp_reg_relumask used? */
        if (i_loop_type == LOOP_TYPE_M) {
          libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg_mapping->gp_reg_relumask, ((long long)i_micro_kernel_config->vlen_out * i_adjust_param)/8 );
        } else {
          libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg_mapping->gp_reg_relumask, ((long long)i_micro_kernel_config->ldo_mask * i_adjust_param)/8 );
        }
      }

      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV)) {
        if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && (i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, ((long long)i_micro_kernel_config->vlen_in * i_adjust_param)/8);
          } else {
            libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, ((long long)i_micro_kernel_config->ldi_mask * i_adjust_param)/8);
          }
        } else {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, (long long)i_micro_kernel_config->vlen_in * i_adjust_param * i_micro_kernel_config->datatype_size_in);
          } else {
            libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, (long long)i_mateltwise_desc->ldi * i_adjust_param * i_micro_kernel_config->datatype_size_in );
          }
        }
      }

      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0)) {
        /* TODO: Evangelos: copied from ReLU.... */
        if (i_loop_type == LOOP_TYPE_M) {
          libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg_mapping->gp_reg_dropoutmask, ((long long)i_micro_kernel_config->vlen_out * i_adjust_param)/8 );
        } else {
          libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg_mapping->gp_reg_dropoutmask, ((long long)i_micro_kernel_config->ldo_mask * i_adjust_param)/8 );
        }
      }

      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, ((long long)i_micro_kernel_config->vlen_in * i_adjust_param)/8);
          } else {
            libxsmm_x86_instruction_alu_imm(  io_generated_code, i_adjust_instr, i_gp_reg, ((long long)i_micro_kernel_config->ldi_mask * i_adjust_param)/8);
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
void libxsmm_generator_configure_avx512_vlens(const libxsmm_meltw_descriptor* i_mateltwise_desc, libxsmm_mateltwise_kernel_config* i_micro_kernel_config) {
  /* First, determine the vlen compute based on the operation */
  if ( LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ||
       LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ||
       LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) {
    i_micro_kernel_config->vlen_comp = 64;
  } else if ( LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ||
       LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ||
       LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)) {
    i_micro_kernel_config->vlen_comp = 32;
  } else if ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ||
       LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP))   {
    i_micro_kernel_config->vlen_comp = 16;
  } else if ( LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ||
       LIBXSMM_DATATYPE_I64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP))   {
    i_micro_kernel_config->vlen_comp = 8;
  }

  /* The vlen_in is aligned with the vlen compute */
  i_micro_kernel_config->vlen_in = i_micro_kernel_config->vlen_comp;

  /* The vlen_out depends on the output datatype */
  if ( LIBXSMM_DATATYPE_IMPLICIT == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
    i_micro_kernel_config->vlen_out = i_micro_kernel_config->vlen_in;
  } else  if ( LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
    /* if the computation is done in F32 or the input is in F32, then set vlen_out to 16 */
    if ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ||
        LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) )
    {
      i_micro_kernel_config->vlen_out = 16;
    } else {
      i_micro_kernel_config->vlen_out = 64;
    }
  } else if ( LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ||
       LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ||
       LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
    /* if the computation is done in F32 or the input is in F32, then set vlen_out to 16 */
    if ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ||
        LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) )
    {
      i_micro_kernel_config->vlen_out= 16;
    } else {
      i_micro_kernel_config->vlen_out = 32;
    }
  } else if ( (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ||
              (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ) {
    /* if the computation is done in F32 or the input is in F32, then set vlen_out to 16 */
    if ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ||
        LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) )
    {
      i_micro_kernel_config->vlen_out= 16;
    } else {
      i_micro_kernel_config->vlen_out = 64;
    }
  } else if ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ||
       LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT))   {
    i_micro_kernel_config->vlen_out = 16;
  } else if ( LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ||
       LIBXSMM_DATATYPE_I64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT))   {
    i_micro_kernel_config->vlen_out = 8;
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_configure_M_N_blocking(unsigned int m, unsigned int n, unsigned int vlen, unsigned int *m_blocking, unsigned int *n_blocking, unsigned int available_vregs) {
  /* The m blocking is done in chunks of vlen */
  unsigned int m_chunks = (m+vlen-1)/vlen;
  /* TODO: Make m chunk remainder depend on number of available zmm registers */
  unsigned int m_chunk_remainder = 8;
  unsigned int m_range, m_block_size, foo1, foo2;

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
    if (m_chunks > 16) {
      *m_blocking = (m_chunks - m_chunk_remainder) * vlen;
    } else {
      if (available_vregs * vlen >= m) {
        *m_blocking = m;
      } else {
        *m_blocking = (m_chunks - 1) * vlen;
      }
    }
  }

  /* For now not any additional blocking in N */
  *n_blocking = n;
}

LIBXSMM_API_INTERN
void libxsmm_generator_configure_loop_order(const libxsmm_meltw_descriptor* i_mateltwise_desc, unsigned int *loop_order, unsigned int *m_blocking, unsigned int *n_blocking, unsigned int *out_blocking, unsigned int *inner_blocking, unsigned int *out_bound, unsigned int *inner_bound) {
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
void libxsmm_load_2d_reg_block( libxsmm_generated_code*                 io_generated_code,
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
  char vname = (i_vlen * i_micro_kernel_config->datatype_size_in == 64) ? 'z' : ( (i_vlen * i_micro_kernel_config->datatype_size_in == 32) ? 'y' : 'x' );
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
  unsigned int vbcast_instr = ( i_micro_kernel_config->datatype_size_in == 4 ) ? ((io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? LIBXSMM_X86_INSTR_VBROADCASTSS : LIBXSMM_X86_INSTR_VPBROADCASTD) : (( i_micro_kernel_config->datatype_size_in == 2 ) ? LIBXSMM_X86_INSTR_VPBROADCASTW : LIBXSMM_X86_INSTR_VPBROADCASTB);
  unsigned int skip_scf_cvt = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_NO_SCF_QUANT) > 0) ? 1 : 0;

  if ( i_micro_kernel_config->datatype_size_in == 8 ) {
    if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
      vbcast_instr = LIBXSMM_X86_INSTR_VPBROADCASTQ_VEX;
    } else {
      vbcast_instr = LIBXSMM_X86_INSTR_VPBROADCASTQ;
    }
  }

  if ( ( LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ||
       ( LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ||
       ( LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0))) {
    if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
      if (i_mask_last_m_chunk > 0) {
        i_mask_reg = i_mateltwise_desc->m % i_vlen;
      }
    }
  }

  /* In this case we do not have to load any data */
  if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_XOR)) return;

  if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_ZIP)) {
    for (in = 0; in < i_n_blocking; in++) {
      for (im = 0; im < i_m_blocking; im++) {
        char pack_vname = (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) ? 'z' : 'y';
        cur_vreg = i_start_vreg + in * i_m_blocking + im;
        if ( bcast_input == 0) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              io_generated_code->arch,
              LIBXSMM_X86_INSTR_VPMOVZXWD,
              i_gp_reg_mapping->gp_reg_in,
              LIBXSMM_X86_GP_REG_UNDEF,
              0,
              (im * i_vlen + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
              pack_vname,
              cur_vreg,
              ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0,
              0, 0);
        } else {
          if ( (bcast_row == 1) || (bcast_scalar == 1) ) {
            if (im == 0) {
              if ((bcast_row == 1) || ((bcast_scalar == 1) && (in == 0))) {
                char copy_vname = (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) ? 'z' : 'y';
                libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                    vbcast_instr,
                    i_gp_reg_mapping->gp_reg_in,
                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                    in * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                    vname,
                    cur_vreg, 0, 0, 0 );
                libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VPMOVZXWD, copy_vname, cur_vreg, cur_vreg );
              } else if ((bcast_scalar == 1) && (in > 0) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) ) {
                char copy_vname = (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) ? 'z' : 'y';
                libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS, copy_vname, i_start_vreg, cur_vreg );
              }
            } else {
              char copy_vname = (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) ? 'z' : 'y';
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS, copy_vname, i_start_vreg + in * i_m_blocking, cur_vreg );
            }
          } else if ( bcast_col == 1 ) {
            if (in == 0) {
              libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                  LIBXSMM_X86_INSTR_VPMOVZXWD,
                  i_gp_reg_mapping->gp_reg_in,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  im * i_vlen * i_micro_kernel_config->datatype_size_in,
                  pack_vname,
                  cur_vreg, ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? 1 : 0, ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0, 0 );
            } else {
              char copy_vname = (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) ? 'z' : 'y';
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS, copy_vname, i_start_vreg + im, cur_vreg );
            }
          }
        }
      }
    }
    return;
  }

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      if ( bcast_input == 0) {
        if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DEQUANT)) {
          if ( (  LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) &&
               ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ) {
            vname = 'x';
            i_vlen = 16;
            if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
              i_vlen = 8;
              if (i_mask_last_m_chunk > 0 && io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX) {
                i_mask_reg = i_mateltwise_desc->m % i_vlen;
              }
            }
          }
          if ( ( LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) &&
               ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ) {
            vname = 'y';
            i_vlen = 16;
            if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
              vname = 'x';
              i_vlen = 8;
            }
          }
          if ( ( LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) &&
               ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ) {
            vname = 'z';
            i_vlen = 16;
            if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
              vname = 'y';
              i_vlen = 8;
            }
          }
        }

        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            ( ((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0))) && (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && !((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ) ? LIBXSMM_X86_INSTR_VMOVSD : i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * i_vlen + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
            vname,
            cur_vreg, ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? 1 : 0, ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0, 0 );

        /* If compute is in F32 and input is BF16 (or input is BF16 and output is F32), then upconvert BF16 -> F32 */
        if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) && LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ||
             (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) && LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ) {
          libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, cur_vreg, cur_vreg );
        } else if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) && LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ||
                    (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) && LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ) {
          /* If compute is in F32 and input is F16 (or input is F16 and output is F32), then upconvert F16 -> F32 */
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, cur_vreg, cur_vreg );
        } else if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) && LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ||
                    (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) && LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ) {
          /* If compute is in F32 and input is BF8 (or input is BF8 and output is F32), then upconvert BF8 -> F32 */
          libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, cur_vreg, cur_vreg );
        } else if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) && LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ||
                    (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) && LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ) {
          /* If compute is in F32 and input is HF8 (or input is HF8 and output is F32), then upconvert HF8 -> FP32 */
          libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name, cur_vreg, cur_vreg,
              i_micro_kernel_config->dcvt_zmm_aux0, i_micro_kernel_config->dcvt_zmm_aux1, i_micro_kernel_config->dcvt_mask_aux0, i_micro_kernel_config->dcvt_mask_aux1 );
        }

        /* quantize tensor */
        if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DEQUANT)) {
          if ( ( LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) &&
               ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ) {
            libxsmm_generator_vcvtint2ps_avx512( io_generated_code, LIBXSMM_DATATYPE_I8, cur_vreg, i_micro_kernel_config->quant_vreg_scf, skip_scf_cvt );
          }
          if ( ( LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) &&
               ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ) {
            libxsmm_generator_vcvtint2ps_avx512( io_generated_code, LIBXSMM_DATATYPE_I16, cur_vreg, i_micro_kernel_config->quant_vreg_scf, skip_scf_cvt );
          }
          if ( ( LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) &&
               ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ) {
            libxsmm_generator_vcvtint2ps_avx512( io_generated_code, LIBXSMM_DATATYPE_I32, cur_vreg, i_micro_kernel_config->quant_vreg_scf, skip_scf_cvt );
          }
        }
      } else {
        if ( (bcast_row == 1) || (bcast_scalar == 1) ) {
          if (im == 0) {
            if ((bcast_row == 1) || ((bcast_scalar == 1) && (in == 0))) {
              if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DEQUANT)) {
                if ( (  LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) &&
                     ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ) {
                  vname = 'x';
                  i_vlen = 16;
                  if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
                    vname = 'x';
                    i_vlen = 8;
                  }
                }
                if ( ( LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) &&
                     ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ) {
                  vname = 'y';
                  i_vlen = 16;
                  if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
                    vname = 'x';
                    i_vlen = 8;
                  }
                }
                if ( ( LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) &&
                     ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ) {
                  vname = 'z';
                  i_vlen = 16;
                  if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
                    vname = 'y';
                    i_vlen = 8;
                  }
                }
              }

              libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                  vbcast_instr,
                  i_gp_reg_mapping->gp_reg_in,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  in * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
                  vname,
                  cur_vreg, 0, 0, 0 );

              /* If compute is in F32 and input is BF16 (or input is BF16 and output is F32), then upconvert BF16 -> FP32 */
              if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) && LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ||
                   (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) && LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ) {
                libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, cur_vreg, cur_vreg );
              } else if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) && LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ||
                          (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) && LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ) {
                /* If compute is in F32 and input is F16 (or input is F16 and output is F32), then upconvert F16 -> F32 */
                libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, cur_vreg, cur_vreg );
              } else if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) && LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ||
                          (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) && LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ) {
                /* If compute is in F32 and input is BF8 (or input is BF8 and output is F32), then upconvert BF8 -> FP32 */
                libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, cur_vreg, cur_vreg );
              } else if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) && LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ||
                          (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) && LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ) {
                /* If compute is in F32 and input is HF8 (or input is HF8 and output is F32), then upconvert HF8 -> FP32 */
                libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name, cur_vreg, cur_vreg,
                    i_micro_kernel_config->dcvt_zmm_aux0, i_micro_kernel_config->dcvt_zmm_aux1, i_micro_kernel_config->dcvt_mask_aux0, i_micro_kernel_config->dcvt_mask_aux1 );
              }

              /* quantize tensor */
              if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DEQUANT)) {
                if ( (  LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) &&
                     ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ) {
                  libxsmm_generator_vcvtint2ps_avx512( io_generated_code, LIBXSMM_DATATYPE_I8, cur_vreg, i_micro_kernel_config->quant_vreg_scf, skip_scf_cvt );
                }
                if ( ( LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) &&
                     ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ) {
                  libxsmm_generator_vcvtint2ps_avx512( io_generated_code, LIBXSMM_DATATYPE_I16, cur_vreg, i_micro_kernel_config->quant_vreg_scf, skip_scf_cvt );
                }
                if ( ( LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) &&
                     ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ) {
                  libxsmm_generator_vcvtint2ps_avx512( io_generated_code, LIBXSMM_DATATYPE_I32, cur_vreg, i_micro_kernel_config->quant_vreg_scf, skip_scf_cvt );
                }
              }
            } else if ((bcast_scalar == 1) && (in > 0) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) ) {
              char copy_vname = (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) ? 'z' : 'y';
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS, copy_vname, i_start_vreg, cur_vreg );
            }
          }

          if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY)) {
            /* Copy the register to the rest of the "M-registers" in this case.... */
            if (im > 0) {
              char copy_vname = (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) ? 'z' : 'y';
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS, copy_vname, i_start_vreg + in * i_m_blocking, cur_vreg );
            }
          }
        }

        if ( bcast_col == 1 ) {
          if (in == 0) {
            if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DEQUANT)) {
              if ( (  LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) &&
                   ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ) {
                vname = 'x';
                i_vlen = 16;
                if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
                  vname = 'x';
                  i_vlen = 8;
                }
              }
              if ( ( LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) &&
                   ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ) {
                vname = 'y';
                i_vlen = 16;
                if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
                  vname = 'x';
                  i_vlen = 8;
                }
              }
              if ( ( LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) &&
                   ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ) {
                vname = 'z';
                i_vlen = 16;
                if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
                  vname = 'y';
                  i_vlen = 8;
                }
              }
            }

            libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                ( ((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0))) && (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && !((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ) ? LIBXSMM_X86_INSTR_VMOVSD : i_micro_kernel_config->vmove_instruction_in,
                i_gp_reg_mapping->gp_reg_in,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                im * i_vlen * i_micro_kernel_config->datatype_size_in,
                vname,
                cur_vreg, ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? 1 : 0, ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0, 0 );

            /* If compute is in F32 and input is BF16 (or input is BF16 and output is F32), then upconvert BF16 -> FP32 */
            if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) && LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ||
                 (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) && LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ) {
              libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, cur_vreg, cur_vreg );
            } else if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) && LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ||
                        (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) && LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ) {
              /* If compute is in F32 and input is F16 (or input is F16 and output is F32), then upconvert F16 -> F32 */
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, cur_vreg, cur_vreg );
            } else if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) && LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ||
                        (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) && LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ) {
              /* If compute is in F32 and input is BF8 (or input is BF8 and output is F32), then upconvert BF8 -> FP32 */
              libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, cur_vreg, cur_vreg );
            } else if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) && LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ||
                        (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) && LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ) {
              /* If compute is in F32 and input is HF8 (or input is HF8 and output is F32), then upconvert HF8 -> FP32 */
              libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name, cur_vreg, cur_vreg,
                  i_micro_kernel_config->dcvt_zmm_aux0, i_micro_kernel_config->dcvt_zmm_aux1, i_micro_kernel_config->dcvt_mask_aux0, i_micro_kernel_config->dcvt_mask_aux1 );
            }

            /* quantize tensor */
            if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DEQUANT)) {
              if ( (  LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) &&
                   ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ) {
                libxsmm_generator_vcvtint2ps_avx512( io_generated_code, LIBXSMM_DATATYPE_I8, cur_vreg, i_micro_kernel_config->quant_vreg_scf, skip_scf_cvt );
              }
              if ( ( LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) &&
                   ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ) {
                libxsmm_generator_vcvtint2ps_avx512( io_generated_code, LIBXSMM_DATATYPE_I16, cur_vreg, i_micro_kernel_config->quant_vreg_scf, skip_scf_cvt );
              }
              if ( ( LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) &&
                   ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ) {
                libxsmm_generator_vcvtint2ps_avx512( io_generated_code, LIBXSMM_DATATYPE_I32, cur_vreg, i_micro_kernel_config->quant_vreg_scf, skip_scf_cvt );
              }
            }
          }

          if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) || (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY)) {
            /* Copy the register to the rest of the "N-REGISTERS" in this case.... */
            if (in > 0) {
              char copy_vname = (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) ? 'z' : 'y';
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS, copy_vname, i_start_vreg + im, cur_vreg );
            }
          }
        }
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_store_2d_reg_block( libxsmm_generated_code*                 io_generated_code,
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
  char vname = (i_vlen * i_micro_kernel_config->datatype_size_out == 64) ? 'z' : ( (i_vlen * i_micro_kernel_config->datatype_size_out == 32) ? 'y' : 'x' );
  unsigned int bcast_row = (((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW) > 0))) ? 1 : 0;
  unsigned int bcast_col = (((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL) > 0))) ? 1 : 0;
  unsigned int bcast_scalar = (((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR) > 0))) ? 1 : 0;
  unsigned int skip_scf_cvt = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_NO_SCF_QUANT) > 0) ? 1 : 0;
  unsigned int sign_sat = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_SIGN_SAT_QUANT) > 0) ? 1 : 0;

  if (libxsmm_generator_mateltwise_is_binary_cmp_op(i_mateltwise_desc) > 0) {
    /* In this case the store part is fused with the compute */
    return;
  }

  if ( ( LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ||
       ( LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ||
       ( LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT))) {
    if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
      if (i_mask_last_m_chunk > 0) {
        i_mask_reg = i_mateltwise_desc->m % i_vlen;
      }
    }
  }

  if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_UNZIP)) {
    for (in = 0; in < i_n_blocking; in++) {
      for (im = 0; im < i_m_blocking; im++) {
        char unpack_vname = (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) ? 'z' : 'y';
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
        libxsmm_x86_instruction_vec_move( io_generated_code,
            io_generated_code->arch,
            LIBXSMM_X86_INSTR_VPMOVDW,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF,
            0,
            (im * i_vlen + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
            unpack_vname,
            cur_vreg,
            ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0,
            0, 1);
        if (((bcast_row == 0) && (bcast_col == 0) && (bcast_scalar == 0)) ||
            ((bcast_scalar == 1) && (in == 0) && (im == 0)) ||
            ((bcast_row == 1) && (im == 0)) ||
            (bcast_col == 1)) {
          libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSRAD_I, unpack_vname, cur_vreg, i_micro_kernel_config->tmp_vreg, 16 );
        }
        libxsmm_x86_instruction_vec_move( io_generated_code,
            io_generated_code->arch,
            LIBXSMM_X86_INSTR_VPMOVDW,
            i_gp_reg_mapping->gp_reg_out,
            i_gp_reg_mapping->gp_reg_offset,
            1,
            (im * i_vlen + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
            unpack_vname,
            i_micro_kernel_config->tmp_vreg,
            ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0,
            0, 1);
      }
    }
  } else if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X2) ||
                                                                                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3)    ) ) {
    char vname_in  = (i_vlen * i_micro_kernel_config->datatype_size_in  == 64) ? 'z' : ( (i_vlen * i_micro_kernel_config->datatype_size_in  == 32) ? 'y' : 'x' );
    char vname_out = (i_vlen * i_micro_kernel_config->datatype_size_out == 32) ? 'y' : 'x';
    for (in = 0; in < i_n_blocking; in++) {
      for (im = 0; im < i_m_blocking; im++) {
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
        libxsmm_x86_instruction_vec_compute_2reg_mask( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU16, vname_in,
                                                       cur_vreg, i_micro_kernel_config->vec_tmp0,
                                                       i_micro_kernel_config->mask_hi, 1);
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VSUBPS, vname_in, i_micro_kernel_config->vec_tmp0, cur_vreg, i_micro_kernel_config->vec_tmp2 );
        if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3 ) {
          libxsmm_x86_instruction_vec_compute_2reg_mask( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU16, vname_in,
                                                         i_micro_kernel_config->vec_tmp2, i_micro_kernel_config->vec_tmp1,
                                                         i_micro_kernel_config->mask_hi, 1);
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VSUBPS, vname_in, i_micro_kernel_config->vec_tmp1, i_micro_kernel_config->vec_tmp2, i_micro_kernel_config->vec_tmp2 );
          if ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_CPX) || (io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CPX) || (io_generated_code->arch == LIBXSMM_X86_AVX2_SRF) ) {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16,
                                                      vname_in,
                                                      i_micro_kernel_config->vec_tmp2, i_micro_kernel_config->vec_tmp2 );
          } else {
            libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, vname_in,
                                                                 i_micro_kernel_config->vec_tmp2, i_micro_kernel_config->vec_tmp2,
                                                                 i_micro_kernel_config->dcvt_zmm_aux0, i_micro_kernel_config->dcvt_zmm_aux1,
                                                                 i_micro_kernel_config->dcvt_mask_aux0, i_micro_kernel_config->dcvt_mask_aux1, 0 );
          }
        } else {
          if ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_CPX) || (io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CPX) || (io_generated_code->arch == LIBXSMM_X86_AVX2_SRF) ) {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16,
                                                      vname_in,
                                                      i_micro_kernel_config->vec_tmp2, i_micro_kernel_config->vec_tmp1 );
          } else {
            libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, vname_in,
                                                                 i_micro_kernel_config->vec_tmp2, i_micro_kernel_config->vec_tmp1,
                                                                 i_micro_kernel_config->dcvt_zmm_aux0, i_micro_kernel_config->dcvt_zmm_aux1,
                                                                 i_micro_kernel_config->dcvt_mask_aux0, i_micro_kernel_config->dcvt_mask_aux1, 0 );
          }
        }
        libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSRAD_I, vname_in, i_micro_kernel_config->vec_tmp0, i_micro_kernel_config->vec_tmp0, 16 );
        libxsmm_x86_instruction_vec_move( io_generated_code,
            io_generated_code->arch,
            LIBXSMM_X86_INSTR_VPMOVDW,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF,
            0,
            (im * i_vlen + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
            vname_in,
            i_micro_kernel_config->vec_tmp0,
            ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0,
            0, 1);
        if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3 ) {
          libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSRAD_I, vname_in, i_micro_kernel_config->vec_tmp1, i_micro_kernel_config->vec_tmp1, 16 );
          libxsmm_x86_instruction_vec_move( io_generated_code,
              io_generated_code->arch,
              LIBXSMM_X86_INSTR_VPMOVDW,
              i_gp_reg_mapping->gp_reg_out,
              i_gp_reg_mapping->gp_reg_offset,
              1,
              (im * i_vlen + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
              vname_in,
              i_micro_kernel_config->vec_tmp1,
              ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0,
              0, 1);
          libxsmm_x86_instruction_vec_move( io_generated_code,
              io_generated_code->arch,
              LIBXSMM_X86_INSTR_VMOVDQU16,
              i_gp_reg_mapping->gp_reg_out,
              i_gp_reg_mapping->gp_reg_offset_2,
              1,
              (im * i_vlen + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
              vname_out,
              i_micro_kernel_config->vec_tmp2,
              ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0,
              0, 1);
        } else {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              io_generated_code->arch,
              LIBXSMM_X86_INSTR_VMOVDQU16,
              i_gp_reg_mapping->gp_reg_out,
              i_gp_reg_mapping->gp_reg_offset,
              1,
              (im * i_vlen + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
              vname_out,
              i_micro_kernel_config->vec_tmp1,
              ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0,
              0, 1);
        }
      }
    }
  } else if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_ZIP)) {
    for (in = 0; in < i_n_blocking; in++) {
      for (im = 0; im < i_m_blocking; im++) {
        char pack_vname = (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) ? 'z' : 'y';
        cur_vreg = i_start_vreg + in * i_m_blocking + im;
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            i_micro_kernel_config->vmove_instruction_out,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * i_vlen + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
            pack_vname,
            cur_vreg, ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? 1 : 0, ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0, 1 );
      }
    }
  }  else {
    for (in = 0; in < i_n_blocking; in++) {
      for (im = 0; im < i_m_blocking; im++) {
        cur_vreg = i_start_vreg + in * i_m_blocking + im;
        cur_vreg_real = i_start_vreg + in * i_m_blocking + im;
        /* In the XOR case we have a constnt vreg */
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

        /* If compute is in F32 and output is BF16 (or input is F32 and output is BF16), then downconvert BF16 -> FP32 */
        if ( cur_vreg == cur_vreg_real ) {
          if ( ((LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) || LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) &&
               LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ) {
            if ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_CPX) || (io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CPX) || (io_generated_code->arch == LIBXSMM_X86_AVX2_SRF) ) {
              libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16,
                                                        i_micro_kernel_config->vector_name,
                                                        cur_vreg, cur_vreg );
            } else {
              libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                                                                   cur_vreg, cur_vreg,
                                                                   i_micro_kernel_config->dcvt_zmm_aux0, i_micro_kernel_config->dcvt_zmm_aux1, i_micro_kernel_config->dcvt_mask_aux0, i_micro_kernel_config->dcvt_mask_aux1, 0 );
            }
          } else if ( ((LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) || LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) &&
               LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ) {
            libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2PH, i_micro_kernel_config->vector_name,
                                                                    cur_vreg, cur_vreg, 0, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 0 : 1, 0x00 );
          } else if ( ((LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) || LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) &&
               LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ) {

            unsigned int stochastic_rnd = ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_STOCHASTIC_ROUND) > 0) ||
                                          ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_STOCHASTIC_ROUND) > 0) ||
                                          ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_STOCHASTIC_ROUND) > 0);
            if ( stochastic_rnd == 1 ) {
              libxsmm_generator_xoshiro128pp_axv2_avx512 ( io_generated_code, i_micro_kernel_config->vector_name,
                                                       i_micro_kernel_config->prng_state0_vreg, i_micro_kernel_config->prng_state1_vreg,
                                                       i_micro_kernel_config->prng_state2_vreg, i_micro_kernel_config->prng_state3_vreg,
                                                       i_micro_kernel_config->prng_vreg_tmp0, i_micro_kernel_config->prng_vreg_tmp1,
                                                       i_micro_kernel_config->prng_vreg_rand);
            }

            libxsmm_generator_vcvtneps2bf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                                                                cur_vreg, cur_vreg,
                                                                i_micro_kernel_config->dcvt_zmm_aux0, i_micro_kernel_config->dcvt_zmm_aux1,
                                                                i_micro_kernel_config->dcvt_mask_aux0, i_micro_kernel_config->dcvt_mask_aux1,
                                                                stochastic_rnd, i_micro_kernel_config->prng_vreg_rand);
          } else if ( ((LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) || LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) &&
               LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ) {
            libxsmm_generator_vcvtf32_to_hf8_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                cur_vreg, cur_vreg,
                i_micro_kernel_config->dcvt_zmm_aux0, i_micro_kernel_config->dcvt_zmm_aux1, i_micro_kernel_config->dcvt_zmm_aux2, i_micro_kernel_config->dcvt_zmm_aux3,
                i_micro_kernel_config->dcvt_mask_aux0, i_micro_kernel_config->dcvt_mask_aux1, i_micro_kernel_config->dcvt_mask_aux2 );
          }
          if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_QUANT)) {
            /* quantize tensor */
            if ( (( LIBXSMM_DATATYPE_F32  == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) ||
                  ( LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) ) &&
                 (  LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ) {
              if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX) {
                libxsmm_generator_vcvtneps2int_avx( io_generated_code, LIBXSMM_DATATYPE_I8, cur_vreg, i_micro_kernel_config->quant_vreg_scf, i_micro_kernel_config->vec_tmp1, i_micro_kernel_config->vec_tmp2, skip_scf_cvt, sign_sat );
                vname = 'x';
                i_vlen = 8;
                if (i_mask_last_m_chunk > 0) {
                  i_mask_reg = i_mateltwise_desc->m % i_vlen;
                }
              } else {
                libxsmm_generator_vcvtneps2int_avx512( io_generated_code, LIBXSMM_DATATYPE_I8, cur_vreg, i_micro_kernel_config->quant_vreg_scf, skip_scf_cvt, sign_sat );
                vname = 'x';
                i_vlen = 16;
                if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
                  i_vlen = 8;
                }
              }
            }
            if ( (( LIBXSMM_DATATYPE_F32  == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) ||
                  ( LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) ) &&
                  ( LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ) {
              if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
                libxsmm_generator_vcvtneps2int_avx( io_generated_code, LIBXSMM_DATATYPE_I16, cur_vreg, i_micro_kernel_config->quant_vreg_scf, i_micro_kernel_config->vec_tmp1, i_micro_kernel_config->vec_tmp2, skip_scf_cvt, sign_sat );
                vname = 'x';
                i_vlen = 8;
                if (i_mask_last_m_chunk > 0 && io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX) {
                  i_mask_reg = i_mateltwise_desc->m % i_vlen;
                }
              } else {
                libxsmm_generator_vcvtneps2int_avx512( io_generated_code, LIBXSMM_DATATYPE_I16, cur_vreg, i_micro_kernel_config->quant_vreg_scf, skip_scf_cvt, sign_sat );
                vname = 'y';
                i_vlen = 16;
              }
            }
            if ( (( LIBXSMM_DATATYPE_F32  == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) ||
                  ( LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ) ) &&
                 ( LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ) {
              libxsmm_generator_vcvtneps2int_avx512( io_generated_code, LIBXSMM_DATATYPE_I32, cur_vreg, i_micro_kernel_config->quant_vreg_scf, skip_scf_cvt, sign_sat );
              vname = 'z';
              i_vlen = 16;
              if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
                vname = 'y';
                i_vlen = 8;
              }
            }
          }
        }

        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            ( ((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT))) && (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && !((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ) ? LIBXSMM_X86_INSTR_VMOVSD : i_micro_kernel_config->vmove_instruction_out,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * i_vlen + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
            vname,
            cur_vreg, ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? 1 : 0, ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0, 1 );

        if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP)) {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
              ( ((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT))) && (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && !((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ) ? LIBXSMM_X86_INSTR_VMOVSD : i_micro_kernel_config->vmove_instruction_out,
              i_gp_reg_mapping->gp_reg_out2,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (im * i_vlen + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
              vname,
              cur_vreg, ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? 1 : 0, ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0, 1 );
        }
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_compute_unary_2d_reg_block_op( libxsmm_generated_code*                 io_generated_code,
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

  LIBXSMM_UNUSED(i_gp_reg_mapping);
  LIBXSMM_UNUSED(i_vlen);
  LIBXSMM_UNUSED(i_mask_last_m_chunk);
  LIBXSMM_UNUSED(i_mask_reg);

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
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) ? LIBXSMM_X86_INSTR_VMULPS : LIBXSMM_X86_INSTR_VMULPD, i_micro_kernel_config->vector_name, cur_vreg, cur_vreg, cur_vreg );
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_NEGATE) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) ? LIBXSMM_X86_INSTR_VXORPS : LIBXSMM_X86_INSTR_VXORPD, i_micro_kernel_config->vector_name, cur_vreg, i_micro_kernel_config->neg_signs_vreg, cur_vreg );
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_INC) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) ? LIBXSMM_X86_INSTR_VADDPS : LIBXSMM_X86_INSTR_VADDPD, i_micro_kernel_config->vector_name, cur_vreg, i_micro_kernel_config->vec_ones, cur_vreg );
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL) {
        if (libxsmm_cpuid_x86_use_high_prec_eltwise_approx() == 0) {
          if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
            if ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) {
              libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VRCPPS, i_micro_kernel_config->vector_name, cur_vreg, LIBXSMM_X86_VEC_REG_UNDEF, cur_vreg, 0, 0, 0, 0);
            } else if ( LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) {
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VDIVPD, i_micro_kernel_config->vector_name, cur_vreg, i_micro_kernel_config->vec_ones, cur_vreg );
            } else {

            }
          } else {
            libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) ? LIBXSMM_X86_INSTR_VRCP14PS : LIBXSMM_X86_INSTR_VRCP14PD, i_micro_kernel_config->vector_name, cur_vreg, LIBXSMM_X86_VEC_REG_UNDEF, cur_vreg, 0, 0, 0, 0);
          }
        } else {
          if ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VDIVPS, i_micro_kernel_config->vector_name, cur_vreg, i_micro_kernel_config->vec_ones, cur_vreg );
          } else if ( LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VDIVPD, i_micro_kernel_config->vector_name, cur_vreg, i_micro_kernel_config->vec_ones, cur_vreg );
          } else {

          }
        }
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT) {
        if (libxsmm_cpuid_x86_use_high_prec_eltwise_approx() == 0) {
          if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
            if ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) {
              libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VRSQRTPS, i_micro_kernel_config->vector_name, cur_vreg, LIBXSMM_X86_VEC_REG_UNDEF, cur_vreg, 0, 0, 0, 0);
            } else if ( LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) {
              libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VSQRTPD, i_micro_kernel_config->vector_name, cur_vreg, LIBXSMM_X86_VEC_REG_UNDEF, cur_vreg, 0, 0, 0, 0);
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VDIVPD, i_micro_kernel_config->vector_name, cur_vreg, i_micro_kernel_config->vec_ones, cur_vreg );
            } else {
            }
          } else {
            libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) ? LIBXSMM_X86_INSTR_VRSQRT14PS : LIBXSMM_X86_INSTR_VRSQRT14PD, i_micro_kernel_config->vector_name, cur_vreg, LIBXSMM_X86_VEC_REG_UNDEF, cur_vreg, 0, 0, 0, 0);
          }
        } else {
          if ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) {
            libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VSQRTPS, i_micro_kernel_config->vector_name, cur_vreg, LIBXSMM_X86_VEC_REG_UNDEF, cur_vreg, 0, 0, 0, 0);
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VDIVPS, i_micro_kernel_config->vector_name, cur_vreg, i_micro_kernel_config->vec_ones, cur_vreg );
          } else if ( LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) {
            libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VSQRTPD, i_micro_kernel_config->vector_name, cur_vreg, LIBXSMM_X86_VEC_REG_UNDEF, cur_vreg, 0, 0, 0, 0);
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VDIVPD, i_micro_kernel_config->vector_name, cur_vreg, i_micro_kernel_config->vec_ones, cur_vreg );
          } else {
          }
        }
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_SQRT) {
        if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
          if ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) {
            libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VRSQRTPS, i_micro_kernel_config->vector_name, cur_vreg, LIBXSMM_X86_VEC_REG_UNDEF, cur_vreg, 0, 0, 0, 0);
            libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VRCPPS, i_micro_kernel_config->vector_name, cur_vreg, LIBXSMM_X86_VEC_REG_UNDEF, cur_vreg, 0, 0, 0, 0);
          } else if ( LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) {
            libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VSQRTPD, i_micro_kernel_config->vector_name, cur_vreg, LIBXSMM_X86_VEC_REG_UNDEF, cur_vreg, 0, 0, 0, 0);
          } else {

          }
        } else {
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) ? LIBXSMM_X86_INSTR_VRSQRT14PS : LIBXSMM_X86_INSTR_VRSQRT14PD, i_micro_kernel_config->vector_name, cur_vreg, LIBXSMM_X86_VEC_REG_UNDEF, cur_vreg, 0, 0, 0, 0);
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) ? LIBXSMM_X86_INSTR_VRCP14PS : LIBXSMM_X86_INSTR_VRCP14PD, i_micro_kernel_config->vector_name, cur_vreg, LIBXSMM_X86_VEC_REG_UNDEF, cur_vreg, 0, 0, 0, 0);
        }
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_EXP) {
        if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
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
          if (libxsmm_cpuid_x86_use_high_prec_eltwise_approx() == 0) {
            libxsmm_generator_exp_ps_3dts_avx512( io_generated_code,
                cur_vreg,
                i_micro_kernel_config->vec_y,
                i_micro_kernel_config->vec_z,
                i_micro_kernel_config->vec_c0,
                i_micro_kernel_config->vec_c1,
                i_micro_kernel_config->vec_c2,
                i_micro_kernel_config->vec_c3,
                i_micro_kernel_config->vec_halves,
                i_micro_kernel_config->vec_log2e,
                i_micro_kernel_config->vector_name );
          } else {
            libxsmm_generator_exp_ps_5dts_avx512( io_generated_code,
                cur_vreg,
                i_micro_kernel_config->vec_y,
                i_micro_kernel_config->vec_z,
                i_micro_kernel_config->aux_mask,
                i_micro_kernel_config->vec_c0,
                i_micro_kernel_config->vec_c1,
                i_micro_kernel_config->vec_c2,
                i_micro_kernel_config->vec_c3,
                i_micro_kernel_config->vec_c4,
                i_micro_kernel_config->vec_c5,
                i_micro_kernel_config->vec_halves,
                i_micro_kernel_config->vec_log2e,
                i_micro_kernel_config->vec_ln2,
                i_micro_kernel_config->vec_expmask,
                i_micro_kernel_config->vec_logfmax,
                i_micro_kernel_config->vec_logfmin,
                i_micro_kernel_config->vector_name);

          }
        }
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TANH || i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TANH_INV ) {
        if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
          libxsmm_generator_tanh_ps_rational_78_avx( io_generated_code,
              cur_vreg,
              i_micro_kernel_config->vec_x2,
              i_micro_kernel_config->vec_nom,
              i_micro_kernel_config->vec_denom,
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
              i_micro_kernel_config->vec_neg_ones );
        } else {
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
              i_micro_kernel_config->vec_neg_ones,
              i_micro_kernel_config->vector_name );
        }

        if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TANH_INV) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
              LIBXSMM_X86_INSTR_VFNMSUB213PS, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_neg_ones, cur_vreg, cur_vreg );
        }
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID || i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV) {
          if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
            libxsmm_generator_sigmoid_ps_rational_78_avx( io_generated_code,
              cur_vreg,
              i_micro_kernel_config->vec_x2,
              i_micro_kernel_config->vec_nom,
              i_micro_kernel_config->vec_denom,
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
              i_micro_kernel_config->vec_neg_ones );
          } else {
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
              i_micro_kernel_config->vec_halves,
              i_micro_kernel_config->vector_name );
          }

        if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
              LIBXSMM_X86_INSTR_VSUBPS, i_micro_kernel_config->vector_name, cur_vreg, i_micro_kernel_config->vec_ones, i_micro_kernel_config->vec_x2 );
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
              LIBXSMM_X86_INSTR_VMULPS, i_micro_kernel_config->vector_name, i_micro_kernel_config->vec_x2, cur_vreg, cur_vreg );
        }
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GELU) {
        if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
          libxsmm_generator_gelu_ps_minimax3_avx( io_generated_code,
              cur_vreg,
              i_micro_kernel_config->vec_c0_lo,
              i_micro_kernel_config->vec_c0_hi,
              i_micro_kernel_config->vec_c1_lo,
              i_micro_kernel_config->vec_c1_hi,
              i_micro_kernel_config->vec_c2_lo,
              i_micro_kernel_config->vec_c2_hi,
              i_micro_kernel_config->vec_tmp0,
              i_micro_kernel_config->vec_tmp1,
              i_micro_kernel_config->vec_tmp2,
              i_micro_kernel_config->vec_tmp3,
              i_micro_kernel_config->vec_tmp4,
              i_micro_kernel_config->vec_tmp5,
              i_micro_kernel_config->vec_tmp6,
              i_micro_kernel_config->vec_tmp7,
              i_micro_kernel_config->rbp_offs_thres,
              i_micro_kernel_config->rbp_offs_signmask,
              i_micro_kernel_config->rbp_offs_absmask,
              i_micro_kernel_config->rbp_offs_scale,
              i_micro_kernel_config->rbp_offs_shifter,
              i_micro_kernel_config->rbp_offs_half );
        } else if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
          libxsmm_generator_gelu_ps_minimax3_avx512_vl256( io_generated_code,
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
              i_micro_kernel_config->vec_c0_lo,
              i_micro_kernel_config->vec_c0_hi,
              i_micro_kernel_config->vec_c1_lo,
              i_micro_kernel_config->vec_c1_hi,
              i_micro_kernel_config->vec_c2_lo,
              i_micro_kernel_config->vec_c2_hi,
              i_micro_kernel_config->vec_tmp0,
              i_micro_kernel_config->vec_tmp1 );
        } else {
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
        }
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV) {
        if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
          libxsmm_generator_gelu_inv_ps_minimax3_avx( io_generated_code,
              cur_vreg,
              i_micro_kernel_config->vec_c0_lo,
              i_micro_kernel_config->vec_c0_hi,
              i_micro_kernel_config->vec_c1_lo,
              i_micro_kernel_config->vec_c1_hi,
              i_micro_kernel_config->vec_c2_lo,
              i_micro_kernel_config->vec_c2_hi,
              i_micro_kernel_config->vec_tmp0,
              i_micro_kernel_config->vec_tmp1,
              i_micro_kernel_config->vec_tmp2,
              i_micro_kernel_config->vec_tmp3,
              i_micro_kernel_config->vec_tmp4,
              i_micro_kernel_config->vec_tmp5,
              i_micro_kernel_config->vec_tmp6,
              i_micro_kernel_config->vec_tmp7,
              i_micro_kernel_config->rbp_offs_thres,
              i_micro_kernel_config->rbp_offs_signmask,
              i_micro_kernel_config->rbp_offs_absmask,
              i_micro_kernel_config->rbp_offs_scale,
              i_micro_kernel_config->rbp_offs_shifter,
              i_micro_kernel_config->rbp_offs_half );
        } else if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
          libxsmm_generator_gelu_inv_ps_minimax3_avx512_vl256( io_generated_code,
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
              i_micro_kernel_config->vec_c0_lo,
              i_micro_kernel_config->vec_c0_hi,
              i_micro_kernel_config->vec_c1_lo,
              i_micro_kernel_config->vec_c1_hi,
              i_micro_kernel_config->vec_c2_lo,
              i_micro_kernel_config->vec_c2_hi,
              i_micro_kernel_config->vec_tmp0,
              i_micro_kernel_config->vec_tmp1 );
        } else {
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
}

LIBXSMM_API_INTERN
void libxsmm_compute_unary_2d_reg_block_relu( libxsmm_generated_code*                 io_generated_code,
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
  LIBXSMM_UNUSED(i_mask_last_m_chunk);
  LIBXSMM_UNUSED(i_mask_reg);
  LIBXSMM_UNUSED(i_vlen);

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      unsigned int l_bf16_compute = ( LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) ? 1 : 0;
      unsigned int n_available_mask_regs = 8 - i_micro_kernel_config->reserved_mask_regs;
      unsigned int cur_mask_reg = i_micro_kernel_config->reserved_mask_regs + (in * i_m_blocking + im) % n_available_mask_regs;
      unsigned int l_vcmp_instr = ( l_bf16_compute > 0 ) ? LIBXSMM_X86_INSTR_VPCMPW : LIBXSMM_X86_INSTR_VCMPPS;
      unsigned int l_vblend_instr = ( l_bf16_compute > 0 ) ? LIBXSMM_X86_INSTR_VPBLENDMW : LIBXSMM_X86_INSTR_VPBLENDMD;
      unsigned int l_mask_st_instr = ( l_bf16_compute > 0  ) ? LIBXSMM_X86_INSTR_KMOVD_ST : LIBXSMM_X86_INSTR_KMOVW_ST;
      unsigned int l_vlen = ( l_bf16_compute > 0 ) ? 32 : 16;
      cur_vreg = i_start_vreg + in * i_m_blocking + im;

      if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
        if ( l_bf16_compute != 0 ) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
          return;
        }
        l_vlen = l_vlen/2;
        cur_mask_reg = i_micro_kernel_config->tmp_vreg;
        if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU ) {
          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCMPPS, i_micro_kernel_config->vector_name, i_micro_kernel_config->zero_vreg, cur_vreg, cur_mask_reg, 6 );

          if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) {
            libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_scratch_0 );
            libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VMOVMSKPS, i_micro_kernel_config->vector_name, cur_mask_reg, LIBXSMM_X86_VEC_REG_UNDEF, i_gp_reg_mapping->gp_reg_scratch_0, 0 );
            libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVB, i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_X86_GP_REG_UNDEF, 0, (im * l_vlen + in * i_micro_kernel_config->ldo_mask)/8, i_gp_reg_mapping->gp_reg_scratch_0,1);
            libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_scratch_0 );
          }

          /* we need to multiply with alpha */
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMULPS, i_micro_kernel_config->vector_name, cur_vreg, i_micro_kernel_config->fam_lu_vreg_alpha, i_micro_kernel_config->tmp_vreg2 );

          /* now we blend both together */
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8(io_generated_code,
              LIBXSMM_X86_INSTR_VBLENDVPS,
              'y',
              cur_vreg,
              i_micro_kernel_config->tmp_vreg2,
              cur_vreg,
              0, 0, 0, (cur_mask_reg) << 4);
        } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU ) {
          if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) {
            libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_scratch_0 );
            libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCMPPS, i_micro_kernel_config->vector_name, i_micro_kernel_config->zero_vreg, cur_vreg, cur_mask_reg, 6 );
            libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VMOVMSKPS, i_micro_kernel_config->vector_name, cur_mask_reg, LIBXSMM_X86_VEC_REG_UNDEF, i_gp_reg_mapping->gp_reg_scratch_0, 0 );
            libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVB, i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_X86_GP_REG_UNDEF, 0, (im * l_vlen + in * i_micro_kernel_config->ldo_mask)/8, i_gp_reg_mapping->gp_reg_scratch_0, 1);
            libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_scratch_0 );
          }

          /* ReLU */
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMAXPS, i_micro_kernel_config->vector_name, cur_vreg, i_micro_kernel_config->zero_vreg, cur_vreg );
        } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU )  {
          /* Compute exp */
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS, 'y', cur_vreg, i_micro_kernel_config->tmp_vreg2 );
          libxsmm_generator_exp_ps_3dts_avx( io_generated_code,
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
                                             i_micro_kernel_config->vec_lo_bound );

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCMPPS, i_micro_kernel_config->vector_name, i_micro_kernel_config->zero_vreg, cur_vreg, cur_mask_reg, 6 );

          /* FMA */
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VFMSUB213PS, 'y',
                                                    i_micro_kernel_config->fam_lu_vreg_alpha, i_micro_kernel_config->fam_lu_vreg_alpha, i_micro_kernel_config->tmp_vreg2 );

          /* Blend exp-fma result with input reg based on elu mask */
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8(io_generated_code,
              LIBXSMM_X86_INSTR_VBLENDVPS,
              'y',
              cur_vreg,
              i_micro_kernel_config->tmp_vreg2,
              cur_vreg,
              0, 0, 0, (cur_mask_reg) << 4);
        } else {
          /* should not happen */
        }
      } else {
        /* Compare to generate mask */
        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, l_vcmp_instr, i_micro_kernel_config->vector_name, i_micro_kernel_config->zero_vreg, cur_vreg, cur_mask_reg, 6 );
        l_vlen = (io_generated_code->arch <= LIBXSMM_X86_AVX512_VL256_CPX) ? l_vlen/2:l_vlen ;
        /* Store mask relu */
        if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && (i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_ELU) ) {
          libxsmm_x86_instruction_mask_move_mem( io_generated_code, l_mask_st_instr, i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_X86_GP_REG_UNDEF, 0, (im * l_vlen + in * i_micro_kernel_config->ldo_mask)/8, cur_mask_reg );
        }

        if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU ) {
          libxsmm_x86_instruction_mask_compute_reg( io_generated_code, LIBXSMM_X86_INSTR_KNOTD, cur_mask_reg, LIBXSMM_X86_VEC_REG_UNDEF, cur_mask_reg, LIBXSMM_X86_IMM_UNDEF );
          /* we need to multiply with alpha */
          libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code, LIBXSMM_X86_INSTR_VMULPS, i_micro_kernel_config->vector_name, cur_vreg, i_micro_kernel_config->fam_lu_vreg_alpha, cur_vreg, cur_mask_reg, 0 );
        } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU ) {
          /* Blend output result with zero reg based on relu mask */
          libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code, l_vblend_instr, i_micro_kernel_config->vector_name, cur_vreg, i_micro_kernel_config->zero_vreg, cur_vreg, cur_mask_reg, 0 );
        } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU ) {
          /* Compute exp */
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS, i_micro_kernel_config->vector_name , cur_vreg, i_micro_kernel_config->tmp_vreg2 );
          libxsmm_generator_exp_ps_3dts_avx512( io_generated_code,
                                                i_micro_kernel_config->tmp_vreg2,
                                                i_micro_kernel_config->tmp_vreg,
                                                i_micro_kernel_config->tmp_vreg3,
                                                i_micro_kernel_config->vec_c0,
                                                i_micro_kernel_config->vec_c1,
                                                i_micro_kernel_config->vec_c2,
                                                i_micro_kernel_config->vec_c3,
                                                i_micro_kernel_config->vec_halves,
                                                i_micro_kernel_config->vec_log2e,
                                                i_micro_kernel_config->vector_name );

          /* FMA */
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VFMSUB213PS, i_micro_kernel_config->vector_name ,
                                                    i_micro_kernel_config->fam_lu_vreg_alpha, i_micro_kernel_config->fam_lu_vreg_alpha, i_micro_kernel_config->tmp_vreg2 );

          /* Blend exp-fma result with input reg based on elu mask */
          libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code, l_vblend_instr, i_micro_kernel_config->vector_name, cur_vreg, i_micro_kernel_config->tmp_vreg2, cur_vreg, cur_mask_reg, 0 );
        } else {
          /* should not happen */
        }
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_compute_unary_2d_reg_block_relu_inv( libxsmm_generated_code*                 io_generated_code,
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
  LIBXSMM_UNUSED(i_vlen);

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      unsigned int l_bf16_compute = ( LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) ? 1 : 0;
      unsigned int n_available_mask_regs = 8 - i_micro_kernel_config->reserved_mask_regs;
      unsigned int cur_mask_reg = i_micro_kernel_config->reserved_mask_regs + (in * i_m_blocking + im) % n_available_mask_regs;
      unsigned int l_vcmp_instr = ( l_bf16_compute > 0 ) ? LIBXSMM_X86_INSTR_VPCMPW : LIBXSMM_X86_INSTR_VCMPPS;
      unsigned int l_vblend_instr = ( l_bf16_compute > 0 ) ? LIBXSMM_X86_INSTR_VPBLENDMW : LIBXSMM_X86_INSTR_VPBLENDMD;
      unsigned int l_mask_ld_instr = ( l_bf16_compute > 0  ) ? LIBXSMM_X86_INSTR_KMOVD_LD : LIBXSMM_X86_INSTR_KMOVW_LD;
      unsigned int l_vlen = ( l_bf16_compute > 0 ) ? 32 : 16;
      cur_vreg = i_start_vreg + in * i_m_blocking + im;

      if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
        if ( l_bf16_compute != 0 ) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
          return;
        }
        l_vlen = l_vlen/2;

        if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && (i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
              LIBXSMM_X86_INSTR_VPBROADCASTB,
              i_gp_reg_mapping->gp_reg_relumask,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (im * l_vlen + in * i_micro_kernel_config->ldi_mask)/8,
              i_micro_kernel_config->vector_name,
              i_micro_kernel_config->tmp_vreg, 0, 0, 0 );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VPANDD, i_micro_kernel_config->vector_name,
                                       i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->vec_tmp0, i_micro_kernel_config->tmp_vreg);

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPCMPEQD, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->vec_tmp0, i_micro_kernel_config->tmp_vreg, 0);
        } else {
          char vname = (l_vlen * i_micro_kernel_config->datatype_size_in == 64) ? 'z' : ( (l_vlen * i_micro_kernel_config->datatype_size_in == 32) ? 'y' : 'x' );
          if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ||
              LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
            if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
              if (i_mask_last_m_chunk > 0) {
                i_mask_reg = i_mateltwise_desc->m % i_vlen;
              }
            }
          }

          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_relumask,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * l_vlen + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
            vname,
            i_micro_kernel_config->tmp_vreg, ( (i_mask_last_m_chunk == 1) && ( im == (i_m_blocking-1)) ) ? 1 : 0, ( (i_mask_last_m_chunk == 1) && ( im == (i_m_blocking-1)) ) ? i_mask_reg : 0, 0 );

          if ( (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) &&
               (l_bf16_compute == 0)     ) {
            libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->tmp_vreg );
          }
          if ( (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) &&
               (l_bf16_compute == 0)     ) {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->tmp_vreg );
          }

          if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV ) {
            libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8(io_generated_code,
                LIBXSMM_X86_INSTR_VCMPPS,
                'y',
                i_micro_kernel_config->zero_vreg,
                i_micro_kernel_config->tmp_vreg,
                i_micro_kernel_config->tmp_vreg2,
                0, 0, 0, 14);
          } else {
            libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8(io_generated_code,
                LIBXSMM_X86_INSTR_VCMPPS,
                'y',
                i_micro_kernel_config->zero_vreg,
                i_micro_kernel_config->tmp_vreg,
                i_micro_kernel_config->tmp_vreg,
                0, 0, 0, 14);
          }
        }

        if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV ) {
          /* we need to multiply with alpha */
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMULPS, i_micro_kernel_config->vector_name, cur_vreg, i_micro_kernel_config->fam_lu_vreg_alpha, i_micro_kernel_config->tmp_vreg2 );

          /* now we blend both together */
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8(io_generated_code,
              LIBXSMM_X86_INSTR_VBLENDVPS,
              'y',
              cur_vreg,
              i_micro_kernel_config->tmp_vreg2,
              cur_vreg,
              0, 0, 0, (i_micro_kernel_config->tmp_vreg) << 4);

        } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV ) {
          /* just blend with zero */
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8(io_generated_code,
              LIBXSMM_X86_INSTR_VBLENDVPS,
              'y',
              cur_vreg,
              i_micro_kernel_config->zero_vreg,
              cur_vreg,
              0, 0, 0, (i_micro_kernel_config->tmp_vreg) << 4);
        } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                    LIBXSMM_X86_INSTR_VADDPS,
                                                    'y',
                                                    i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->fam_lu_vreg_alpha, i_micro_kernel_config->tmp_vreg );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                    LIBXSMM_X86_INSTR_VMULPS,
                                                    'y',
                                                    i_micro_kernel_config->tmp_vreg, cur_vreg, i_micro_kernel_config->tmp_vreg );

          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8(io_generated_code,
                                                                 LIBXSMM_X86_INSTR_VBLENDVPS,
                                                                 'y',
                                                                 cur_vreg,
                                                                 i_micro_kernel_config->tmp_vreg,
                                                                 cur_vreg,
                                                                 0, 0, 0, (i_micro_kernel_config->tmp_vreg2) << 4);
        } else {
          /* should not happen */
        }
      } else {
        l_vlen = (io_generated_code->arch <=LIBXSMM_X86_AVX512_VL256_CPX) ? l_vlen/2:l_vlen ;
        if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && (i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
          libxsmm_x86_instruction_mask_move_mem( io_generated_code, l_mask_ld_instr, i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_X86_GP_REG_UNDEF,  0, (im * l_vlen + in * i_micro_kernel_config->ldi_mask)/8,  cur_mask_reg );
        } else {
          if ( (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) &&
               (l_bf16_compute == 0)     ) {
            libxsmm_x86_instruction_unified_vec_move( io_generated_code,
              i_micro_kernel_config->vmove_instruction_in,
              i_gp_reg_mapping->gp_reg_relumask,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (im * l_vlen + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
              (io_generated_code->arch <=LIBXSMM_X86_AVX512_VL256_CPX) ? 'x' : 'y',
              i_micro_kernel_config->tmp_vreg, ( (i_mask_last_m_chunk == 1) && ( im == (i_m_blocking-1)) ) ? 1 : 0, ( (i_mask_last_m_chunk == 1) && ( im == (i_m_blocking-1)) ) ? i_mask_reg : 0, 0 );

            libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->tmp_vreg );
          } else if ( (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) &&
               (l_bf16_compute == 0)     ) {
            libxsmm_x86_instruction_unified_vec_move( io_generated_code,
              i_micro_kernel_config->vmove_instruction_in,
              i_gp_reg_mapping->gp_reg_relumask,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (im * l_vlen + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
              (io_generated_code->arch <=LIBXSMM_X86_AVX512_VL256_CPX) ? 'x' : 'y',
              i_micro_kernel_config->tmp_vreg, ( (i_mask_last_m_chunk == 1) && ( im == (i_m_blocking-1)) ) ? 1 : 0, ( (i_mask_last_m_chunk == 1) && ( im == (i_m_blocking-1)) ) ? i_mask_reg : 0, 0 );

            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->tmp_vreg );
          } else if ( ((LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0))) && (l_bf16_compute == 0)   ) {
            libxsmm_x86_instruction_unified_vec_move( io_generated_code,
              i_micro_kernel_config->vmove_instruction_in,
              i_gp_reg_mapping->gp_reg_relumask,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (im * l_vlen + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
              'x',
              i_micro_kernel_config->tmp_vreg, ( (i_mask_last_m_chunk == 1) && ( im == (i_m_blocking-1)) ) ? 1 : 0, ( (i_mask_last_m_chunk == 1) && ( im == (i_m_blocking-1)) ) ? i_mask_reg : 0, 0 );
            if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) {
              libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->tmp_vreg );
            } else {
              libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                  i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->tmp_vreg,
                  i_micro_kernel_config->dcvt_zmm_aux0, i_micro_kernel_config->dcvt_zmm_aux1, i_micro_kernel_config->dcvt_mask_aux0, i_micro_kernel_config->dcvt_mask_aux1 );
            }
          } else {
            libxsmm_x86_instruction_unified_vec_move( io_generated_code,
              i_micro_kernel_config->vmove_instruction_in,
              i_gp_reg_mapping->gp_reg_relumask,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (im * l_vlen + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
              i_micro_kernel_config->vector_name,
              i_micro_kernel_config->tmp_vreg, ( (i_mask_last_m_chunk == 1) && ( im == (i_m_blocking-1)) ) ? 1 : 0, ( (i_mask_last_m_chunk == 1) && ( im == (i_m_blocking-1)) ) ? i_mask_reg : 0, 0 );
          }

          /* Compare to generate mask */
          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
            l_vcmp_instr,
            i_micro_kernel_config->vector_name,
            i_micro_kernel_config->zero_vreg,
            i_micro_kernel_config->tmp_vreg,
            cur_mask_reg,
            6 );
        }

        if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV ) {
          libxsmm_x86_instruction_mask_compute_reg( io_generated_code, LIBXSMM_X86_INSTR_KNOTD, cur_mask_reg, LIBXSMM_X86_VEC_REG_UNDEF, cur_mask_reg, LIBXSMM_X86_IMM_UNDEF );
          /* we need to multiply with alpha */
          libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code, LIBXSMM_X86_INSTR_VMULPS, i_micro_kernel_config->vector_name, cur_vreg, i_micro_kernel_config->fam_lu_vreg_alpha, cur_vreg, cur_mask_reg, 0 );
        } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV ) {
          /* Blend output result with zero reg based on relu mask */
          libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code, l_vblend_instr, i_micro_kernel_config->vector_name, cur_vreg, i_micro_kernel_config->zero_vreg, cur_vreg, cur_mask_reg, 0 );
        } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                    LIBXSMM_X86_INSTR_VADDPS,
                                                    i_micro_kernel_config->vector_name,
                                                    i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->fam_lu_vreg_alpha, i_micro_kernel_config->tmp_vreg );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                    LIBXSMM_X86_INSTR_VMULPS,
                                                    i_micro_kernel_config->vector_name,
                                                    i_micro_kernel_config->tmp_vreg, cur_vreg, i_micro_kernel_config->tmp_vreg );

          /* Blend output result based on elu mask */
          libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                         l_vblend_instr,
                                                         i_micro_kernel_config->vector_name,
                                                         cur_vreg,
                                                         i_micro_kernel_config->tmp_vreg,
                                                         cur_vreg,
                                                         cur_mask_reg,
                                                         0 );
        } else {
          /* should not happen */
        }
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_compute_unary_2d_reg_block_dropout( libxsmm_generated_code*                 io_generated_code,
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
  LIBXSMM_UNUSED(i_mask_last_m_chunk);
  LIBXSMM_UNUSED(i_mask_reg);
  LIBXSMM_UNUSED(i_vlen);

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      unsigned int n_available_mask_regs = 8 - i_micro_kernel_config->reserved_mask_regs;
      unsigned int cur_mask_reg = i_micro_kernel_config->reserved_mask_regs + (in * i_m_blocking + im) % n_available_mask_regs;
      unsigned int l_vcmp_instr = LIBXSMM_X86_INSTR_VCMPPS;
      unsigned int l_vmul_instr = LIBXSMM_X86_INSTR_VMULPS;
      unsigned int l_mask_st_instr = LIBXSMM_X86_INSTR_KMOVW_ST;
      unsigned int l_vlen = 16;
      cur_vreg = i_start_vreg + in * i_m_blocking + im;

      if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
        l_vlen = l_vlen/2;
        cur_mask_reg = i_micro_kernel_config->dropout_vreg_avxmask;

        /* draw a random number */
        libxsmm_generator_xoshiro128p_f32_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                       i_micro_kernel_config->prng_state0_vreg, i_micro_kernel_config->prng_state1_vreg, i_micro_kernel_config->prng_state2_vreg, i_micro_kernel_config->prng_state3_vreg,
                                                       i_micro_kernel_config->dropout_vreg_tmp0, i_micro_kernel_config->dropout_vreg_tmp1, i_micro_kernel_config->dropout_vreg_one, i_micro_kernel_config->dropout_vreg_tmp2 );

        /* compare with p */
        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, l_vcmp_instr, 'y',
                                                       i_micro_kernel_config->dropout_vreg_tmp2, i_micro_kernel_config->dropout_prob_vreg, cur_mask_reg, 0x06  );

        /* weight */
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, l_vmul_instr, 'y',
                                                  cur_vreg, i_micro_kernel_config->dropout_invprob_vreg, cur_vreg );

        /* blend zero and multiplication result together */
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, 'y', i_micro_kernel_config->dropout_vreg_tmp0, i_micro_kernel_config->dropout_vreg_tmp0, i_micro_kernel_config->dropout_vreg_tmp0 );
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VBLENDVPS, 'y', cur_vreg, i_micro_kernel_config->dropout_vreg_tmp0, cur_vreg, 0, 0, 0, (cur_mask_reg) << 4);

        if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) {
          /* TODO: remove usage of R11.... */
          libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_scratch_0 );
          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VMOVMSKPS, i_micro_kernel_config->vector_name, cur_mask_reg, LIBXSMM_X86_VEC_REG_UNDEF, i_gp_reg_mapping->gp_reg_scratch_0, 0 );
          libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVB, i_gp_reg_mapping->gp_reg_dropoutmask, LIBXSMM_X86_GP_REG_UNDEF, 0, (im * l_vlen + in * i_micro_kernel_config->ldo_mask)/8, i_gp_reg_mapping->gp_reg_scratch_0, 1);
          libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_scratch_0 );
        }
      } else {
        l_vlen = (io_generated_code->arch <= LIBXSMM_X86_AVX512_VL256_CPX) ? l_vlen/2:l_vlen ;
        /* draw a random number */
        libxsmm_generator_xoshiro128p_f32_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name,
                                                       i_micro_kernel_config->prng_state0_vreg, i_micro_kernel_config->prng_state1_vreg, i_micro_kernel_config->prng_state2_vreg, i_micro_kernel_config->prng_state3_vreg,
                                                       i_micro_kernel_config->dropout_vreg_tmp0, i_micro_kernel_config->dropout_vreg_tmp1, i_micro_kernel_config->dropout_vreg_one, i_micro_kernel_config->dropout_vreg_tmp2 );

        /* compare with p */
        libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, l_vcmp_instr, i_micro_kernel_config->vector_name,
                                                       i_micro_kernel_config->dropout_vreg_tmp2, i_micro_kernel_config->dropout_prob_vreg, cur_mask_reg, 0x06  );

        /* weight and zero input */
        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code, l_vmul_instr, i_micro_kernel_config->vector_name,
                                                       cur_vreg, i_micro_kernel_config->dropout_invprob_vreg, cur_vreg, cur_mask_reg, 1 );

        /* Store dropout mask */
        if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) {
          libxsmm_x86_instruction_mask_move_mem( io_generated_code, l_mask_st_instr, i_gp_reg_mapping->gp_reg_dropoutmask, LIBXSMM_X86_GP_REG_UNDEF, 0,  (im * l_vlen + in * i_micro_kernel_config->ldo_mask)/8, cur_mask_reg );
        }
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_compute_unary_2d_reg_block_dropout_inv( libxsmm_generated_code*                 io_generated_code,
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
  LIBXSMM_UNUSED(i_mask_last_m_chunk);
  LIBXSMM_UNUSED(i_mask_reg);
  LIBXSMM_UNUSED(i_vlen);

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      unsigned int n_available_mask_regs = 8 - i_micro_kernel_config->reserved_mask_regs;
      unsigned int cur_mask_reg = i_micro_kernel_config->reserved_mask_regs + (in * i_m_blocking + im) % n_available_mask_regs;
      unsigned int l_vmul_instr = LIBXSMM_X86_INSTR_VMULPS;
      unsigned int l_mask_ld_instr = LIBXSMM_X86_INSTR_KMOVW_LD;
      unsigned int l_vlen = 16;
      cur_vreg = i_start_vreg + in * i_m_blocking + im;

      if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
        l_vlen = l_vlen/2;

        /* load mask */
        if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code, LIBXSMM_X86_INSTR_VPBROADCASTB,
                                                    i_gp_reg_mapping->gp_reg_dropoutmask, LIBXSMM_X86_GP_REG_UNDEF, 0, (im * l_vlen + in * i_micro_kernel_config->ldi_mask)/8,
                                                    i_micro_kernel_config->vector_name, i_micro_kernel_config->dropout_vreg_tmp0, 0, 0, 0 );

          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VPANDD, i_micro_kernel_config->vector_name,
                                       i_micro_kernel_config->dropout_vreg_tmp0, i_micro_kernel_config->dropout_vreg_avxmask, i_micro_kernel_config->dropout_vreg_tmp0);

          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPCMPEQD, i_micro_kernel_config->vector_name,
                                                         i_micro_kernel_config->dropout_vreg_tmp0, i_micro_kernel_config->dropout_vreg_avxmask, i_micro_kernel_config->dropout_vreg_tmp0, 0);
        } else {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BITMASK_REQUIRED );
          return;
        }

        /* weight */
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, l_vmul_instr, 'y', cur_vreg, i_micro_kernel_config->dropout_prob_vreg, cur_vreg );

        /* select which value is set to 0 */
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VBLENDVPS, 'y',
                                                                cur_vreg, i_micro_kernel_config->dropout_vreg_zero, cur_vreg, 0, 0, 0, (i_micro_kernel_config->dropout_vreg_tmp0) << 4);
      } else {
        l_vlen = (io_generated_code->arch <= LIBXSMM_X86_AVX512_VL256_CPX) ? l_vlen/2:l_vlen ;
        if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) {
          libxsmm_x86_instruction_mask_move_mem( io_generated_code, l_mask_ld_instr, i_gp_reg_mapping->gp_reg_dropoutmask, LIBXSMM_X86_GP_REG_UNDEF,  0, (im * l_vlen + in * i_micro_kernel_config->ldi_mask)/8,  cur_mask_reg );
        } else {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BITMASK_REQUIRED );
          return;
        }

        /* weight and zero input */
        libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code, l_vmul_instr, i_micro_kernel_config->vector_name,
                                                       cur_vreg, i_micro_kernel_config->dropout_prob_vreg, cur_vreg, cur_mask_reg, 1 );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_compute_ternary_2d_reg_block( libxsmm_generated_code*                 io_generated_code,
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
  unsigned int vbcast_instr = ( i_micro_kernel_config->datatype_size_in1 == 4 ) ? ((io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? LIBXSMM_X86_INSTR_VBROADCASTSS : LIBXSMM_X86_INSTR_VPBROADCASTD) : (( i_micro_kernel_config->datatype_size_in1 == 2 ) ? LIBXSMM_X86_INSTR_VPBROADCASTW : LIBXSMM_X86_INSTR_VPBROADCASTB);

  if ( i_micro_kernel_config->datatype_size_in1 == 8 ) {
    if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
      vbcast_instr = LIBXSMM_X86_INSTR_VPBROADCASTQ_VEX;
    } else {
      vbcast_instr = LIBXSMM_X86_INSTR_VPBROADCASTQ;
    }
  }

  if ( ( LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) ||
       ( LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) ||
       ( LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1))) {
    if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
      if (i_mask_last_m_chunk > 0) {
        i_mask_reg = i_mateltwise_desc->m % i_vlen;
      }
    }
  }

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      unsigned int l_bf16_compute = ( LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) ? 1 : 0;
      unsigned int l_vlen = (l_bf16_compute > 0) ? 32 : (( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) ? 16 : 8);
      unsigned int vmove_instr_in2 = (bcast_input == 0 || bcast_col == 1) ? i_micro_kernel_config->vmove_instruction_in1 : vbcast_instr;
      unsigned int _im = (bcast_row == 1) ? 0 : im;
      unsigned int _in = (bcast_col == 1) ? 0 : in;
      unsigned int _i_mask_reg = (bcast_row == 1) ? 0 : i_mask_reg;
      unsigned int in_offset;

      if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
        l_vlen = l_vlen/2;
      } else {
        if (io_generated_code->arch <= LIBXSMM_X86_AVX512_VL256_CPX) {
          l_vlen = l_vlen/2;
        }
      }

      if (bcast_scalar == 1) {
        _im = 0;
        _in = 0;
      }

      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      in_offset = _in * i_mateltwise_desc->ldi2;

      /* Optimize the input loading in case we have reuse */
      if ( (bcast_input == 0) ||
           (bcast_col == 1) ||
           ((bcast_row == 1) && (im == 0)) ||
           ((bcast_scalar == 1) && (im == 0) && (in == 0))) {
        if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            vmove_instr_in2,
            i_gp_reg_mapping->gp_reg_in2,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (_im * l_vlen + in_offset) * i_micro_kernel_config->datatype_size_in1,
            (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) ? 'y' : 'x',
            i_micro_kernel_config->tmp_vreg, ( (i_mask_last_m_chunk == 1) && ( _im == (i_m_blocking-1)) && (vmove_instr_in2 != vbcast_instr) ) ? 1 : 0,  ( (i_mask_last_m_chunk == 1) && ( _im == (i_m_blocking-1)) && (vmove_instr_in2 != vbcast_instr)) ? _i_mask_reg : 0, 0 );

          libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->tmp_vreg );
        } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            vmove_instr_in2,
            i_gp_reg_mapping->gp_reg_in2,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (_im * l_vlen + in_offset) * i_micro_kernel_config->datatype_size_in1,
            (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) ? 'y' : 'x',
            i_micro_kernel_config->tmp_vreg, ( (i_mask_last_m_chunk == 1) && ( _im == (i_m_blocking-1)) && (vmove_instr_in2 != vbcast_instr) ) ? 1 : 0,  ( (i_mask_last_m_chunk == 1) && ( _im == (i_m_blocking-1)) && (vmove_instr_in2 != vbcast_instr)) ? _i_mask_reg : 0, 0 );

          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->tmp_vreg );
        } else if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1) /*libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)*/) {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            ( (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && !((i_mask_last_m_chunk == 1) && ( _im == (i_m_blocking-1)) && (vmove_instr_in2 != vbcast_instr)) ) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instr_in2,
            i_gp_reg_mapping->gp_reg_in2,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (_im * l_vlen + in_offset) * i_micro_kernel_config->datatype_size_in1,
            'x',
            i_micro_kernel_config->tmp_vreg, ( (i_mask_last_m_chunk == 1) && ( _im == (i_m_blocking-1)) && (vmove_instr_in2 != vbcast_instr) ) ? 1 : 0,  ( (i_mask_last_m_chunk == 1) && ( _im == (i_m_blocking-1)) && (vmove_instr_in2 != vbcast_instr)) ? _i_mask_reg : 0, 0 );

          libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->tmp_vreg );
        } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1) /*libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)*/) {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            ( (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && !((i_mask_last_m_chunk == 1) && ( _im == (i_m_blocking-1)) && (vmove_instr_in2 != vbcast_instr)) ) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instr_in2,
            i_gp_reg_mapping->gp_reg_in2,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (_im * l_vlen + in_offset) * i_micro_kernel_config->datatype_size_in1,
            'x',
            i_micro_kernel_config->tmp_vreg, ( (i_mask_last_m_chunk == 1) && ( _im == (i_m_blocking-1)) && (vmove_instr_in2 != vbcast_instr) ) ? 1 : 0,  ( (i_mask_last_m_chunk == 1) && ( _im == (i_m_blocking-1)) && (vmove_instr_in2 != vbcast_instr)) ? _i_mask_reg : 0, 0 );
          libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->tmp_vreg,
              i_micro_kernel_config->dcvt_zmm_aux0, i_micro_kernel_config->dcvt_zmm_aux1, i_micro_kernel_config->dcvt_mask_aux0, i_micro_kernel_config->dcvt_mask_aux1 );
        } else {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            vmove_instr_in2,
            i_gp_reg_mapping->gp_reg_in2,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (_im * l_vlen + in_offset) * i_micro_kernel_config->datatype_size_in1,
            i_micro_kernel_config->vector_name,
            i_micro_kernel_config->tmp_vreg, ( (i_mask_last_m_chunk == 1) && ( _im == (i_m_blocking-1)) && (vmove_instr_in2 != vbcast_instr) ) ? 1 : 0, ( (i_mask_last_m_chunk == 1) && ( _im == (i_m_blocking-1)) && (vmove_instr_in2 != vbcast_instr) ) ? _i_mask_reg : 0, 0 );
        }
      }

      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_TERNARY_SELECT) {
        /* Load mask and perform blend */
        unsigned int l_vblend_instr = ( l_bf16_compute > 0 ) ? LIBXSMM_X86_INSTR_VPBLENDMW : LIBXSMM_X86_INSTR_VPBLENDMD;
        unsigned int l_mask_ld_instr = ( l_bf16_compute > 0  ) ? LIBXSMM_X86_INSTR_KMOVD_LD : LIBXSMM_X86_INSTR_KMOVW_LD;
        unsigned int cur_mask_reg = i_micro_kernel_config->blend_tmp_mask;
        unsigned int l_ld_mask = i_mateltwise_desc->ldi3;
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BITMASK_2BYTEMULT) > 0) {
          l_ld_mask = LIBXSMM_UPDIV(l_ld_mask, 16)*16;
        }
        if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
          if ( l_bf16_compute != 0 ) {
            LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
            return;
          }
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
              LIBXSMM_X86_INSTR_VPBROADCASTB,
              i_gp_reg_mapping->gp_reg_in3,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (im * l_vlen + in * l_ld_mask)/8,
              i_micro_kernel_config->vector_name,
              cur_mask_reg, 0, 0, 0 );
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VPANDD, i_micro_kernel_config->vector_name,
                                       cur_mask_reg, i_micro_kernel_config->vec_tmp0, cur_mask_reg);
          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPCMPEQD, i_micro_kernel_config->vector_name, cur_mask_reg, i_micro_kernel_config->vec_tmp0, cur_mask_reg, 0);
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8(io_generated_code,
              LIBXSMM_X86_INSTR_VBLENDVPS,
              'y',
              i_micro_kernel_config->tmp_vreg,
              cur_vreg,
              cur_vreg,
              0, 0, 0, cur_mask_reg << 4);
        } else {
          if (io_generated_code->arch <= LIBXSMM_X86_AVX512_VL256_CPX) {
            l_mask_ld_instr = ( l_bf16_compute > 0  ) ? LIBXSMM_X86_INSTR_KMOVW_LD : LIBXSMM_X86_INSTR_KMOVB_LD;
          }
          libxsmm_x86_instruction_mask_move_mem( io_generated_code, l_mask_ld_instr, i_gp_reg_mapping->gp_reg_in3, LIBXSMM_X86_GP_REG_UNDEF, 0, (im * l_vlen + in * l_ld_mask)/8, cur_mask_reg );
          libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code, l_vblend_instr, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_vreg, cur_vreg, cur_vreg, cur_mask_reg, 0 );
        }
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_compute_binary_2d_reg_block( libxsmm_generated_code*                 io_generated_code,
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
  unsigned int vbcast_instr = ( i_micro_kernel_config->datatype_size_in1 == 4 ) ? ((io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? LIBXSMM_X86_INSTR_VBROADCASTSS : LIBXSMM_X86_INSTR_VPBROADCASTD) : (( i_micro_kernel_config->datatype_size_in1 == 2 ) ? LIBXSMM_X86_INSTR_VPBROADCASTW : LIBXSMM_X86_INSTR_VPBROADCASTB);
  unsigned int i_mask_reg_out = i_mask_reg;
  unsigned int l_cmp_imm = 0;
  unsigned int cur_mask_reg = 0;
  unsigned int n_available_mask_regs = 8 - i_micro_kernel_config->reserved_mask_regs;
  unsigned int l_mask_st_instr = LIBXSMM_X86_INSTR_KMOVW_ST;

  if ( i_micro_kernel_config->datatype_size_in1 == 8 ) {
    if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
      vbcast_instr = LIBXSMM_X86_INSTR_VPBROADCASTQ_VEX;
    } else {
      vbcast_instr = LIBXSMM_X86_INSTR_VPBROADCASTQ;
    }
  }

  if ( ( LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ||
       ( LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ||
       ( LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT))) {
    if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
      if (i_mask_last_m_chunk > 0) {
        i_mask_reg_out = i_mateltwise_desc->m % i_vlen;
      }
    }
  }

  if ( ( LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) ||
       ( LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) ||
       ( LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1))) {
    if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
      if (i_mask_last_m_chunk > 0) {
        i_mask_reg = i_mateltwise_desc->m % i_vlen;
      }
    }
  }

  switch (i_mateltwise_desc->param) {
    case LIBXSMM_MELTW_TYPE_BINARY_ADD: {
      binary_op_instr = ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) ? LIBXSMM_X86_INSTR_VADDPS : LIBXSMM_X86_INSTR_VADDPD;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_MUL: {
      binary_op_instr = ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) ? LIBXSMM_X86_INSTR_VMULPS : LIBXSMM_X86_INSTR_VMULPD;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_SUB: {
      binary_op_instr = ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) ? LIBXSMM_X86_INSTR_VSUBPS : LIBXSMM_X86_INSTR_VSUBPD;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_DIV: {
      binary_op_instr = ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) ? LIBXSMM_X86_INSTR_VDIVPS : LIBXSMM_X86_INSTR_VDIVPD;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_MULADD: {
      binary_op_instr = ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) ? LIBXSMM_X86_INSTR_VFMADD213PS : LIBXSMM_X86_INSTR_VFMADD213PD;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_MAX: {
      binary_op_instr = ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) ? LIBXSMM_X86_INSTR_VMAXPS : LIBXSMM_X86_INSTR_VMAXPD;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_MIN: {
      binary_op_instr = ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) ? LIBXSMM_X86_INSTR_VMINPS : LIBXSMM_X86_INSTR_VMINPD;
    } break;
    default:;
  }

  if (libxsmm_generator_mateltwise_is_binary_cmp_op(i_mateltwise_desc) > 0) {
    binary_op_instr = LIBXSMM_X86_INSTR_VCMPPS;
    switch (i_mateltwise_desc->param) {
      case LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_GT: {
        l_cmp_imm = 6;
      } break;
      case LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_GE: {
        l_cmp_imm = 5;
      } break;
      case LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_LT: {
        l_cmp_imm = 1;
      } break;
      case LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_LE: {
        l_cmp_imm = 2;
      } break;
      case LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_EQ: {
        l_cmp_imm = 0;
      } break;
      case LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_NE: {
        l_cmp_imm = 4;
      } break;
      default:;
    }
  }

  if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_ZIP) {
    for (in = 0; in < i_n_blocking; in++) {
      for (im = 0; im < i_m_blocking; im++) {
        char pack_vname = (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) ? 'z' : 'y';
        unsigned int _im = (bcast_row == 1) ? 0 : im;
        unsigned int _in = (bcast_col == 1) ? 0 : in;
        unsigned int in_offset;
        unsigned int l_vlen = 16;
        if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
          l_vlen = l_vlen/2;
        }
        if (bcast_scalar == 1) {
          _im = 0;
          _in = 0;
        }
        in_offset = _in * i_mateltwise_desc->ldi2;
        cur_vreg = i_start_vreg + in * i_m_blocking + im;
        if ( (bcast_input == 0) ||
           (bcast_col == 1) ||
           ((bcast_row == 1) && (im == 0)) ||
           ((bcast_scalar == 1) && (im == 0) && (in == 0))) {
          unsigned int vmove_instr_in2 = (bcast_input == 0 || bcast_col == 1) ? LIBXSMM_X86_INSTR_VPMOVZXWD: vbcast_instr;
          libxsmm_x86_instruction_vec_move( io_generated_code,
              io_generated_code->arch,
              vmove_instr_in2,
              i_gp_reg_mapping->gp_reg_in2,
              LIBXSMM_X86_GP_REG_UNDEF,
              0,
              (_im * l_vlen + in_offset) * i_micro_kernel_config->datatype_size_in1,
              pack_vname,
              i_micro_kernel_config->tmp_vreg,
              ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0,
              0, 0);
          if (vmove_instr_in2 == vbcast_instr) {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VPMOVZXWD, pack_vname, i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->tmp_vreg );
          }
          libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, pack_vname, i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->tmp_vreg, 16 );
        }
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPORD, pack_vname, cur_vreg, i_micro_kernel_config->tmp_vreg, cur_vreg );
      }
    }
    return;
  }

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      unsigned int l_bf16_compute = ( LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) ? 1 : 0;
      unsigned int l_vlen = (l_bf16_compute > 0) ? 32 : (( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) ? 16 : 8);
      unsigned int vmove_instr_in2 = (bcast_input == 0 || bcast_col == 1) ? i_micro_kernel_config->vmove_instruction_in1 : vbcast_instr;
      unsigned int _im = (bcast_row == 1) ? 0 : im;
      unsigned int _in = (bcast_col == 1) ? 0 : in;
      unsigned int _i_mask_reg = (bcast_row == 1) ? 0 : i_mask_reg;
      unsigned int in_offset;

      if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
        l_vlen = l_vlen/2;
      }

      if (bcast_scalar == 1) {
        _im = 0;
        _in = 0;
      }

      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      in_offset = _in * i_mateltwise_desc->ldi2;

      /* Optimize the input loading in case we have reuse */
      if ( (bcast_input == 0) ||
           (bcast_col == 1) ||
           ((bcast_row == 1) && (im == 0)) ||
           ((bcast_scalar == 1) && (im == 0) && (in == 0))) {
        if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            vmove_instr_in2,
            i_gp_reg_mapping->gp_reg_in2,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (_im * l_vlen + in_offset) * i_micro_kernel_config->datatype_size_in1,
            (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) ? 'y' : 'x',
            i_micro_kernel_config->tmp_vreg, ( (i_mask_last_m_chunk == 1) && ( _im == (i_m_blocking-1)) && (vmove_instr_in2 != vbcast_instr) ) ? 1 : 0,  ( (i_mask_last_m_chunk == 1) && ( _im == (i_m_blocking-1)) && (vmove_instr_in2 != vbcast_instr)) ? _i_mask_reg : 0, 0 );

          libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->tmp_vreg );
        } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            vmove_instr_in2,
            i_gp_reg_mapping->gp_reg_in2,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (_im * l_vlen + in_offset) * i_micro_kernel_config->datatype_size_in1,
            (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) ? 'y' : 'x',
            i_micro_kernel_config->tmp_vreg, ( (i_mask_last_m_chunk == 1) && ( _im == (i_m_blocking-1)) && (vmove_instr_in2 != vbcast_instr) ) ? 1 : 0,  ( (i_mask_last_m_chunk == 1) && ( _im == (i_m_blocking-1)) && (vmove_instr_in2 != vbcast_instr)) ? _i_mask_reg : 0, 0 );

          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->tmp_vreg );
        } else if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1) /*libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)*/) {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            ( (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && !((i_mask_last_m_chunk == 1) && ( _im == (i_m_blocking-1)) && (vmove_instr_in2 != vbcast_instr)) ) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instr_in2,
            i_gp_reg_mapping->gp_reg_in2,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (_im * l_vlen + in_offset) * i_micro_kernel_config->datatype_size_in1,
            'x',
            i_micro_kernel_config->tmp_vreg, ( (i_mask_last_m_chunk == 1) && ( _im == (i_m_blocking-1)) && (vmove_instr_in2 != vbcast_instr) ) ? 1 : 0,  ( (i_mask_last_m_chunk == 1) && ( _im == (i_m_blocking-1)) && (vmove_instr_in2 != vbcast_instr)) ? _i_mask_reg : 0, 0 );

          libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->tmp_vreg );
        } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1) /*libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)*/) {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            ( (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && !((i_mask_last_m_chunk == 1) && ( _im == (i_m_blocking-1)) && (vmove_instr_in2 != vbcast_instr)) ) ? LIBXSMM_X86_INSTR_VMOVSD : vmove_instr_in2,
            i_gp_reg_mapping->gp_reg_in2,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (_im * l_vlen + in_offset) * i_micro_kernel_config->datatype_size_in1,
            'x',
            i_micro_kernel_config->tmp_vreg, ( (i_mask_last_m_chunk == 1) && ( _im == (i_m_blocking-1)) && (vmove_instr_in2 != vbcast_instr) ) ? 1 : 0,  ( (i_mask_last_m_chunk == 1) && ( _im == (i_m_blocking-1)) && (vmove_instr_in2 != vbcast_instr)) ? _i_mask_reg : 0, 0 );
          libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->tmp_vreg,
              i_micro_kernel_config->dcvt_zmm_aux0, i_micro_kernel_config->dcvt_zmm_aux1, i_micro_kernel_config->dcvt_mask_aux0, i_micro_kernel_config->dcvt_mask_aux1 );
        } else {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            vmove_instr_in2,
            i_gp_reg_mapping->gp_reg_in2,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (_im * l_vlen + in_offset) * i_micro_kernel_config->datatype_size_in1,
            i_micro_kernel_config->vector_name,
            i_micro_kernel_config->tmp_vreg, ( (i_mask_last_m_chunk == 1) && ( _im == (i_m_blocking-1)) && (vmove_instr_in2 != vbcast_instr) ) ? 1 : 0, ( (i_mask_last_m_chunk == 1) && ( _im == (i_m_blocking-1)) && (vmove_instr_in2 != vbcast_instr) ) ? _i_mask_reg : 0, 0 );
        }
      }

      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_MULADD) {
        if (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * l_vlen + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
            (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) ? 'y' : 'x',
            i_micro_kernel_config->tmp_vreg2, ( (i_mask_last_m_chunk == 1) && ( im == (i_m_blocking-1)) ) ? 1 : 0, ( (i_mask_last_m_chunk == 1) && ( im == (i_m_blocking-1)) ) ? i_mask_reg_out : 0, 0 );

          libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_vreg2, i_micro_kernel_config->tmp_vreg2 );
        } else if (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * l_vlen + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
            (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) ? 'y' : 'x',
            i_micro_kernel_config->tmp_vreg2, ( (i_mask_last_m_chunk == 1) && ( im == (i_m_blocking-1)) ) ? 1 : 0, ( (i_mask_last_m_chunk == 1) && ( im == (i_m_blocking-1)) ) ? i_mask_reg : 0, 0 );

          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTPH2PS, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_vreg2, i_micro_kernel_config->tmp_vreg2 );
        } else if (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            ( (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && !((i_mask_last_m_chunk == 1) && ( _im == (i_m_blocking-1))) ) ? LIBXSMM_X86_INSTR_VMOVSD : i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * l_vlen + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
            'x',
            i_micro_kernel_config->tmp_vreg2, ( (i_mask_last_m_chunk == 1) && ( im == (i_m_blocking-1)) ) ? 1 : 0, ( (i_mask_last_m_chunk == 1) && ( im == (i_m_blocking-1)) ) ? i_mask_reg : 0, 0 );

          libxsmm_generator_cvtbf8ps_avx512( io_generated_code, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_vreg2, i_micro_kernel_config->tmp_vreg2 );
        } else if (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            ( (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && !((i_mask_last_m_chunk == 1) && ( _im == (i_m_blocking-1))) ) ? LIBXSMM_X86_INSTR_VMOVSD : i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * l_vlen + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
            'x',
            i_micro_kernel_config->tmp_vreg2, ( (i_mask_last_m_chunk == 1) && ( im == (i_m_blocking-1)) ) ? 1 : 0, ( (i_mask_last_m_chunk == 1) && ( im == (i_m_blocking-1)) ) ? i_mask_reg : 0, 0 );

          libxsmm_generator_vcvthf8_to_f32_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_vreg2, i_micro_kernel_config->tmp_vreg2,
              i_micro_kernel_config->dcvt_zmm_aux0, i_micro_kernel_config->dcvt_zmm_aux1, i_micro_kernel_config->dcvt_mask_aux0, i_micro_kernel_config->dcvt_mask_aux1 );
        } else {
          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * l_vlen + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
            i_micro_kernel_config->vector_name,
            i_micro_kernel_config->tmp_vreg2, ( (i_mask_last_m_chunk == 1) && ( im == (i_m_blocking-1)) ) ? 1 : 0,  ( (i_mask_last_m_chunk == 1) && ( im == (i_m_blocking-1)) ) ? i_mask_reg : 0, 0 );
        }
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, binary_op_instr, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_vreg2, i_micro_kernel_config->tmp_vreg, cur_vreg );
      } else if (libxsmm_generator_mateltwise_is_binary_cmp_op(i_mateltwise_desc) > 0) {
        if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
          cur_mask_reg = i_micro_kernel_config->tmp_vreg2;
          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, binary_op_instr, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_vreg, cur_vreg, cur_mask_reg, l_cmp_imm );
          if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BITMASK_2BYTEMULT) > 0 ) {
            libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_scratch_0 );
            libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VMOVMSKPS, i_micro_kernel_config->vector_name, cur_mask_reg, LIBXSMM_X86_VEC_REG_UNDEF, i_gp_reg_mapping->gp_reg_scratch_0, 0 );
            libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVB, i_gp_reg_mapping->gp_reg_out,
                LIBXSMM_X86_GP_REG_UNDEF, 0, (im * l_vlen + in * i_micro_kernel_config->ldo_mask)/8, i_gp_reg_mapping->gp_reg_scratch_0, 1);
            libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_scratch_0 );
          }
        } else {
          if (io_generated_code->arch <= LIBXSMM_X86_AVX512_VL256_CPX) {
            l_mask_st_instr = LIBXSMM_X86_INSTR_KMOVB_ST;
          }
          cur_mask_reg = i_micro_kernel_config->reserved_mask_regs + (in * i_m_blocking + im) % n_available_mask_regs;
          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, binary_op_instr, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_vreg, cur_vreg, cur_mask_reg, l_cmp_imm );
          libxsmm_x86_instruction_mask_move_mem( io_generated_code, l_mask_st_instr, i_gp_reg_mapping->gp_reg_out, LIBXSMM_X86_GP_REG_UNDEF, 0, (im * l_vlen + in * i_micro_kernel_config->ldo_mask)/8, cur_mask_reg );
        }
      } else {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, binary_op_instr, i_micro_kernel_config->vector_name, i_micro_kernel_config->tmp_vreg, cur_vreg, cur_vreg );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_compute_unary_binary_2d_reg_block( libxsmm_generated_code*                 io_generated_code,
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
        libxsmm_compute_unary_2d_reg_block_op( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_vlen, i_start_vreg, i_m_blocking, i_n_blocking, i_mask_last_m_chunk, i_mask_reg);
      } break;
      case LIBXSMM_MELTW_TYPE_UNARY_RELU:
      case LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU:
      case LIBXSMM_MELTW_TYPE_UNARY_ELU: {
        libxsmm_compute_unary_2d_reg_block_relu( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_vlen, i_start_vreg, i_m_blocking, i_n_blocking, i_mask_last_m_chunk, i_mask_reg);
      } break;
      case LIBXSMM_MELTW_TYPE_UNARY_RELU_INV:
      case LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV:
      case LIBXSMM_MELTW_TYPE_UNARY_ELU_INV: {
        libxsmm_compute_unary_2d_reg_block_relu_inv( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_vlen, i_start_vreg, i_m_blocking, i_n_blocking, i_mask_last_m_chunk, i_mask_reg);
      } break;
      case LIBXSMM_MELTW_TYPE_UNARY_DROPOUT: {
        libxsmm_compute_unary_2d_reg_block_dropout( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_vlen, i_start_vreg, i_m_blocking, i_n_blocking, i_mask_last_m_chunk, i_mask_reg);
      } break;
      case LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV: {
        libxsmm_compute_unary_2d_reg_block_dropout_inv( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_vlen, i_start_vreg, i_m_blocking, i_n_blocking, i_mask_last_m_chunk, i_mask_reg);
      } break;
      default: /* Perform no compute */ ;
    }
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
    libxsmm_compute_binary_2d_reg_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_vlen, i_start_vreg, i_m_blocking, i_n_blocking, i_mask_last_m_chunk, i_mask_reg);
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
    libxsmm_compute_ternary_2d_reg_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_vlen, i_start_vreg, i_m_blocking, i_n_blocking, i_mask_last_m_chunk, i_mask_reg);
  }
}

LIBXSMM_API_INTERN
void libxsmm_setup_input_output_masks( libxsmm_generated_code*                 io_generated_code,
                                       libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                       const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                       unsigned int                            i_tmp_reg,
                                       unsigned int                            i_m,
                                       unsigned int*                           i_use_m_input_masking,
                                       unsigned int*                           i_mask_reg_in,
                                       unsigned int*                           i_use_m_output_masking,
                                       unsigned int*                           i_mask_reg_out ) {

  unsigned int mask_in_count, mask_out_count, mask_reg_in = 0, mask_reg_out = 0, use_m_input_masking, use_m_output_masking;
  unsigned int i_vlen_in = i_micro_kernel_config->vlen_in;
  unsigned int i_vlen_out = i_micro_kernel_config->vlen_out;
  unsigned int reserved_mask_regs = i_micro_kernel_config->reserved_mask_regs;
  LIBXSMM_UNUSED(i_mateltwise_desc);

  use_m_input_masking   = (i_m % i_vlen_in == 0 ) ? 0 : 1;
  use_m_output_masking  = (i_m % i_vlen_out == 0 ) ? 0 : 1;

  if (use_m_input_masking == 1) {
    if ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ) {
      libxsmm_datatype fake_dt = LIBXSMM_DATATYPE_UNSUPPORTED;
      if (i_vlen_in == 32) {
        fake_dt = LIBXSMM_DATATYPE_I8;
      } else if (i_vlen_in == 16) {
        fake_dt = LIBXSMM_DATATYPE_I16;
      } else if (i_vlen_in == 8) {
        fake_dt = LIBXSMM_DATATYPE_I32;
      } else if (i_vlen_in == 4) {
        fake_dt = LIBXSMM_DATATYPE_I64;
      } else {
        /* should not happen */
      }
      mask_in_count = i_vlen_in - i_m % i_vlen_in;
      mask_reg_in   = reserved_mask_regs;
      libxsmm_generator_initialize_avx512_mask(io_generated_code, i_tmp_reg, mask_reg_in, mask_in_count, fake_dt);
      reserved_mask_regs++;
    } else if ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) && (io_generated_code->arch <= LIBXSMM_X86_ALLFEAT) ) {
      libxsmm_datatype fake_dt = LIBXSMM_DATATYPE_UNSUPPORTED;
      if (i_vlen_in == 64) {
        fake_dt = LIBXSMM_DATATYPE_I8;
      } else if (i_vlen_in == 32) {
        fake_dt = LIBXSMM_DATATYPE_I16;
      } else if (i_vlen_in == 16) {
        fake_dt = LIBXSMM_DATATYPE_I32;
      } else if (i_vlen_in == 8) {
        fake_dt = LIBXSMM_DATATYPE_I64;
      } else {
        /* should not happen */
      }
      mask_in_count = i_vlen_in - i_m % i_vlen_in;
      mask_reg_in   = reserved_mask_regs;
      libxsmm_generator_initialize_avx512_mask(io_generated_code, i_tmp_reg, mask_reg_in, mask_in_count, fake_dt);
      reserved_mask_regs++;
    } else {
      mask_reg_in = i_micro_kernel_config->inout_vreg_mask;
      libxsmm_generator_initialize_avx_mask(io_generated_code, mask_reg_in, i_m % i_vlen_in, ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) ?  LIBXSMM_DATATYPE_F32 : LIBXSMM_DATATYPE_F64);
    }
  }

  if (use_m_output_masking == 1) {
    if (i_vlen_in == i_vlen_out) {
      mask_reg_out = mask_reg_in;
    } else {
      if ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ) {
        libxsmm_datatype fake_dt = LIBXSMM_DATATYPE_UNSUPPORTED;
        if (i_vlen_out == 64) {
          fake_dt = LIBXSMM_DATATYPE_I8;
        } else if (i_vlen_out == 32) {
          fake_dt = LIBXSMM_DATATYPE_I16;
        } else if (i_vlen_out == 16) {
          fake_dt = LIBXSMM_DATATYPE_I32;
        } else if (i_vlen_out == 8) {
          fake_dt = LIBXSMM_DATATYPE_I64;
        } else {
          /* should not happen */
        }
        mask_out_count = i_vlen_out - i_m % i_vlen_out;
        mask_reg_out   = reserved_mask_regs;
        libxsmm_generator_initialize_avx512_mask(io_generated_code, i_tmp_reg, mask_reg_out, mask_out_count, fake_dt);
        reserved_mask_regs++;
      } else if ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) && (io_generated_code->arch <= LIBXSMM_X86_ALLFEAT) ) {
        libxsmm_datatype fake_dt = LIBXSMM_DATATYPE_UNSUPPORTED;
        if (i_vlen_out == 64) {
          fake_dt = LIBXSMM_DATATYPE_I8;
        } else if (i_vlen_out == 32) {
          fake_dt = LIBXSMM_DATATYPE_I16;
        } else if (i_vlen_out == 16) {
          fake_dt = LIBXSMM_DATATYPE_I32;
        } else if (i_vlen_out == 8) {
          fake_dt = LIBXSMM_DATATYPE_I64;
        } else {
          /* should not happen */
        }
        mask_out_count = i_vlen_out - i_m % i_vlen_out;
        mask_reg_out   = reserved_mask_regs;
        libxsmm_generator_initialize_avx512_mask(io_generated_code, i_tmp_reg, mask_reg_out, mask_out_count, fake_dt);
        reserved_mask_regs++;
      } else {
        mask_reg_out = i_micro_kernel_config->reserved_zmms;
        i_micro_kernel_config->reserved_zmms =  i_micro_kernel_config->reserved_zmms + 1;
        libxsmm_generator_initialize_avx_mask(io_generated_code, mask_reg_out, i_m % i_vlen_out, ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) ?  LIBXSMM_DATATYPE_F32 : LIBXSMM_DATATYPE_F64 );
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
void libxsmm_configure_microkernel_loops( libxsmm_generated_code*                        io_generated_code,
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
  LIBXSMM_UNUSED(i_mateltwise_desc);

  if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
    max_nm_unrolling = 16;
  }

  m_trips               = (i_m + i_vlen_in - 1) / i_vlen_in;
  n_trips               = i_n;

  max_nm_unrolling  = max_nm_unrolling - reserved_zmms;

  if (i_use_m_input_masking == 1) {
    m_unroll_factor = m_trips;
  } else {
    m_unroll_factor = LIBXSMM_MIN(m_trips,16);
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
void libxsmm_configure_unary_kernel_vregs_masks( libxsmm_generated_code*                 io_generated_code,
                                                 libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                 libxsmm_datatype                        i_compute_dtype,
                                                 unsigned int                            op,
                                                 unsigned int                            flags,
                                                 unsigned int                            i_gp_reg_tmp,
                                                 const unsigned int                      i_gp_reg_aux0,
                                                 const unsigned int                      i_gp_reg_aux1 ) {
  const char vname = (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 'y' : 'z';
  unsigned int use_fp64_compute = (i_compute_dtype == LIBXSMM_DATATYPE_F64) ? 1 : 0;

  if ( op == LIBXSMM_MELTW_TYPE_UNARY_UNZIP ) {
    i_micro_kernel_config->tmp_vreg = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
  }

  if ((op == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (op == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV)       ||
      (op == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (op == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV)    ) {
    i_micro_kernel_config->zero_vreg = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
    /*tmp change */
    if ((io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) && ((flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0)) {
      i_micro_kernel_config->tmp_vreg = i_micro_kernel_config->reserved_zmms;
      i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
      if ((op == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) || (op == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV)) {
        unsigned int const_mask_array[8] = { 0x00000001, 0x00000002 , 0x00000004, 0x00000008, 0x00000010, 0x00000020, 0x00000040 , 0x00000080 };
        i_micro_kernel_config->vec_tmp0 = i_micro_kernel_config->reserved_zmms;
        i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
        libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) const_mask_array, "const_mask_array", vname, i_micro_kernel_config->vec_tmp0 );
      }
    } else if ( (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) && ((flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) == 0) ) {
      if ( op == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU ) {
        i_micro_kernel_config->tmp_vreg = i_micro_kernel_config->reserved_zmms;
        i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
      }
    }

    if ( ((op == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) || (op == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV)) && ((flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) == 0) ) {
      i_micro_kernel_config->tmp_vreg = i_micro_kernel_config->reserved_zmms;
      i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
    }

    if ( (op == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (op == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ) {
      i_micro_kernel_config->tmp_vreg2 = i_micro_kernel_config->reserved_zmms;
      i_micro_kernel_config->fam_lu_vreg_alpha = i_micro_kernel_config->reserved_zmms + 1;
      i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 2;

      /* load alpha */
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch,
                                        LIBXSMM_X86_INSTR_VBROADCASTSS,
                                        i_gp_reg_aux1, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                        vname, i_micro_kernel_config->fam_lu_vreg_alpha, 0, 1, 0 );
    }

    /* Set zero register needed for relu */
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, i_micro_kernel_config->vector_name, i_micro_kernel_config->zero_vreg, i_micro_kernel_config->zero_vreg, i_micro_kernel_config->zero_vreg );
  }

  if ((op == LIBXSMM_MELTW_TYPE_UNARY_ELU) || (op == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV)) {
    i_micro_kernel_config->zero_vreg = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->tmp_vreg = i_micro_kernel_config->reserved_zmms + 1;
    i_micro_kernel_config->fam_lu_vreg_alpha = i_micro_kernel_config->reserved_zmms + 2;
    i_micro_kernel_config->tmp_vreg2 = i_micro_kernel_config->reserved_zmms + 3;
    i_micro_kernel_config->tmp_vreg3 = i_micro_kernel_config->reserved_zmms + 4;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 5;

    if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
      if (i_micro_kernel_config->use_fp32bf16_cvt_replacement > 0) {
        i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms - 2;
        i_micro_kernel_config->tmp_vreg2     = i_micro_kernel_config->dcvt_zmm_aux0;
        i_micro_kernel_config->tmp_vreg3     = i_micro_kernel_config->dcvt_zmm_aux1;
      }
    }

    /* load alpha */
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch,
                                      LIBXSMM_X86_INSTR_VBROADCASTSS,
                                      i_gp_reg_aux1, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                      vname, i_micro_kernel_config->fam_lu_vreg_alpha, 0, 1, 0 );

    /* Set zero register needed for elu */
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, i_micro_kernel_config->vector_name, i_micro_kernel_config->zero_vreg, i_micro_kernel_config->zero_vreg, i_micro_kernel_config->zero_vreg );
  }

  if ((op == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (op == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
    if ((io_generated_code->arch >= LIBXSMM_X86_AVX) && (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) ) {
      i_micro_kernel_config->dropout_vreg_avxmask = i_micro_kernel_config->reserved_zmms;
      i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
      if ((op == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV) && ((flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0)) {
        unsigned int const_mask_array[8] = { 0x00000001, 0x00000002 , 0x00000004, 0x00000008, 0x00000010, 0x00000020, 0x00000040 , 0x00000080 };
        libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) const_mask_array, "const_mask_array", vname, i_micro_kernel_config->dropout_vreg_avxmask );
      }
    }

    if ((io_generated_code->arch >= LIBXSMM_X86_AVX) && (io_generated_code->arch < LIBXSMM_X86_ALLFEAT)) {
      unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;

      if (op == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) {
        reserved_zmms += 10;
        if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
          if (i_micro_kernel_config->use_fp32bf16_cvt_replacement > 0) {
            reserved_zmms -= 2;
          }
        }
        i_micro_kernel_config->prng_state0_vreg     = reserved_zmms - 1;
        i_micro_kernel_config->prng_state1_vreg     = reserved_zmms - 2;
        i_micro_kernel_config->prng_state2_vreg     = reserved_zmms - 3;
        i_micro_kernel_config->prng_state3_vreg     = reserved_zmms - 4;
        i_micro_kernel_config->dropout_vreg_tmp2    = reserved_zmms - 5;
        i_micro_kernel_config->dropout_vreg_one     = reserved_zmms - 6;
        i_micro_kernel_config->dropout_prob_vreg    = reserved_zmms - 7;
        i_micro_kernel_config->dropout_invprob_vreg = reserved_zmms - 8;
        i_micro_kernel_config->dropout_vreg_tmp0    = reserved_zmms - 9;
        i_micro_kernel_config->dropout_vreg_tmp1    = reserved_zmms - 10;
        if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
          if (i_micro_kernel_config->use_fp32bf16_cvt_replacement > 0) {
            i_micro_kernel_config->dropout_vreg_tmp0  = i_micro_kernel_config->dcvt_zmm_aux0;
            i_micro_kernel_config->dropout_vreg_tmp1  = i_micro_kernel_config->dcvt_zmm_aux1;
          }
        }

        libxsmm_generator_load_prng_state_avx_avx512( io_generated_code, vname, i_gp_reg_aux0,
                                                      i_micro_kernel_config->prng_state0_vreg, i_micro_kernel_config->prng_state1_vreg,
                                                      i_micro_kernel_config->prng_state2_vreg, i_micro_kernel_config->prng_state3_vreg );
        libxsmm_generator_prepare_dropout_avx_avx512( io_generated_code, vname, i_gp_reg_tmp, i_gp_reg_aux1,
                                                      i_micro_kernel_config->dropout_vreg_one,
                                                      i_micro_kernel_config->dropout_prob_vreg,
                                                      i_micro_kernel_config->dropout_invprob_vreg );
      }

      if (op == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV) {
        reserved_zmms += 6;

        i_micro_kernel_config->dropout_vreg_tmp0 = reserved_zmms - 1;
        i_micro_kernel_config->dropout_vreg_tmp1 = reserved_zmms - 2;
        i_micro_kernel_config->dropout_vreg_tmp2 = reserved_zmms - 3;
        i_micro_kernel_config->dropout_vreg_one  = reserved_zmms - 4;
        i_micro_kernel_config->dropout_vreg_zero = reserved_zmms - 5;
        i_micro_kernel_config->dropout_prob_vreg = reserved_zmms - 6;

        libxsmm_generator_prepare_dropout_inv_avx_avx512( io_generated_code, vname, i_gp_reg_tmp, i_gp_reg_aux1,
                                                          i_micro_kernel_config->dropout_vreg_one,
                                                          i_micro_kernel_config->dropout_vreg_zero,
                                                          i_micro_kernel_config->dropout_prob_vreg );
      }

      i_micro_kernel_config->reserved_zmms = reserved_zmms;
    }
  }

  if (op == LIBXSMM_MELTW_TYPE_UNARY_XOR) {
    i_micro_kernel_config->zero_vreg = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, i_micro_kernel_config->vector_name, i_micro_kernel_config->zero_vreg, i_micro_kernel_config->zero_vreg, i_micro_kernel_config->zero_vreg );
  }

  if (op == LIBXSMM_MELTW_TYPE_UNARY_NEGATE) {
    float f_neg_array[16] = { -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f, -0.0f };
    double d_neg_array[8] = { -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0 };
    unsigned char *neg_array = (use_fp64_compute > 0) ? (unsigned char*) d_neg_array : (unsigned char*) f_neg_array ;
    i_micro_kernel_config->neg_signs_vreg = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) neg_array, "neg_array", vname, i_micro_kernel_config->neg_signs_vreg );
  }

  if (op == LIBXSMM_MELTW_TYPE_UNARY_INC) {
    float f_ones_array[16] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
    double d_ones_array[16] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    unsigned char *ones_array = (use_fp64_compute > 0) ? (unsigned char*) d_ones_array : (unsigned char*) f_ones_array ;
    i_micro_kernel_config->vec_ones = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) ones_array, "ones_array", vname, i_micro_kernel_config->vec_ones );
  }

  if ((op == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL || op == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT) &&
      (use_fp64_compute > 0) &&
      (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) ) {
    double d_ones_array[16] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    i_micro_kernel_config->vec_ones = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) d_ones_array, "ones_array", vname, i_micro_kernel_config->vec_ones );
  }

  if ((op == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL || op == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT) && (libxsmm_cpuid_x86_use_high_prec_eltwise_approx() > 0)) {
    float f_ones_array[16] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
    double d_ones_array[8] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    unsigned char *ones_array = (use_fp64_compute > 0) ? (unsigned char*) d_ones_array : (unsigned char*) f_ones_array ;
    i_micro_kernel_config->vec_ones = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) ones_array, "ones_array", vname, i_micro_kernel_config->vec_ones );
  }

  if (op == LIBXSMM_MELTW_TYPE_UNARY_EXP) {
    i_micro_kernel_config->vec_y = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->vec_z = i_micro_kernel_config->reserved_zmms + 1;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 2;
  }

  if (op == LIBXSMM_MELTW_TYPE_UNARY_GELU || op == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV) {
    unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;

    if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
      reserved_zmms += 14;
      if (i_micro_kernel_config->use_fp32bf16_cvt_replacement > 0) {
        reserved_zmms -= 2;
      }
      i_micro_kernel_config->vec_c0_lo         = reserved_zmms - 1;
      i_micro_kernel_config->vec_c0_hi         = reserved_zmms - 2;
      i_micro_kernel_config->vec_c1_lo         = reserved_zmms - 3;
      i_micro_kernel_config->vec_c1_hi         = reserved_zmms - 4;
      i_micro_kernel_config->vec_c2_lo         = reserved_zmms - 5;
      i_micro_kernel_config->vec_c2_hi         = reserved_zmms - 6;
      i_micro_kernel_config->vec_tmp0          = reserved_zmms - 7;
      i_micro_kernel_config->vec_tmp1          = reserved_zmms - 8;
      i_micro_kernel_config->vec_tmp2          = reserved_zmms - 9;
      i_micro_kernel_config->vec_tmp3          = reserved_zmms - 10;
      i_micro_kernel_config->vec_tmp4          = reserved_zmms - 11;
      i_micro_kernel_config->vec_tmp5          = reserved_zmms - 12;
      i_micro_kernel_config->vec_tmp6          = reserved_zmms - 13;
      i_micro_kernel_config->vec_tmp7          = reserved_zmms - 14;
      if (i_micro_kernel_config->use_fp32bf16_cvt_replacement > 0) {
        i_micro_kernel_config->vec_tmp6        = i_micro_kernel_config->dcvt_zmm_aux0;
        i_micro_kernel_config->vec_tmp7        = i_micro_kernel_config->dcvt_zmm_aux1;
      }

      if (op == LIBXSMM_MELTW_TYPE_UNARY_GELU ) {
        libxsmm_generator_prepare_coeffs_gelu_ps_minimax3_avx( io_generated_code,
            i_gp_reg_tmp,
            i_micro_kernel_config->vec_c0_lo,
            i_micro_kernel_config->vec_c0_hi,
            i_micro_kernel_config->vec_c1_lo,
            i_micro_kernel_config->vec_c1_hi,
            i_micro_kernel_config->vec_c2_lo,
            i_micro_kernel_config->vec_c2_hi,
            i_micro_kernel_config->rbp_offs_thres,
            i_micro_kernel_config->rbp_offs_signmask,
            i_micro_kernel_config->rbp_offs_absmask,
            i_micro_kernel_config->rbp_offs_scale,
            i_micro_kernel_config->rbp_offs_shifter,
            i_micro_kernel_config->rbp_offs_half );
      }
      if (op == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV ) {
        libxsmm_generator_prepare_coeffs_gelu_inv_ps_minimax3_avx( io_generated_code,
            i_gp_reg_tmp,
            i_micro_kernel_config->vec_c0_lo,
            i_micro_kernel_config->vec_c0_hi,
            i_micro_kernel_config->vec_c1_lo,
            i_micro_kernel_config->vec_c1_hi,
            i_micro_kernel_config->vec_c2_lo,
            i_micro_kernel_config->vec_c2_hi,
            i_micro_kernel_config->rbp_offs_thres,
            i_micro_kernel_config->rbp_offs_signmask,
            i_micro_kernel_config->rbp_offs_absmask,
            i_micro_kernel_config->rbp_offs_scale,
            i_micro_kernel_config->rbp_offs_shifter,
            i_micro_kernel_config->rbp_offs_half );
      }
    } else if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
      reserved_zmms += 19;

      i_micro_kernel_config->vec_thres      = reserved_zmms - 1;
      i_micro_kernel_config->vec_absmask    = reserved_zmms - 2;
      i_micro_kernel_config->vec_scale      = reserved_zmms - 3;
      i_micro_kernel_config->vec_shifter    = reserved_zmms - 4;
      i_micro_kernel_config->vec_halves     = reserved_zmms - 5;
      i_micro_kernel_config->vec_c0_lo      = reserved_zmms - 6;
      i_micro_kernel_config->vec_c0_hi      = reserved_zmms - 7;
      i_micro_kernel_config->vec_c1_lo      = reserved_zmms - 8;
      i_micro_kernel_config->vec_c1_hi      = reserved_zmms - 9;
      i_micro_kernel_config->vec_c2_lo      = reserved_zmms - 10;
      i_micro_kernel_config->vec_c2_hi      = reserved_zmms - 11;
      i_micro_kernel_config->vec_tmp0       = reserved_zmms - 12;
      i_micro_kernel_config->vec_tmp1       = reserved_zmms - 13;
      i_micro_kernel_config->vec_xr         = reserved_zmms - 14;
      i_micro_kernel_config->vec_xa         = reserved_zmms - 15;
      i_micro_kernel_config->vec_index      = reserved_zmms - 16;
      i_micro_kernel_config->vec_C0         = reserved_zmms - 17;
      i_micro_kernel_config->vec_C1         = reserved_zmms - 18;
      i_micro_kernel_config->vec_C2         = reserved_zmms - 19;

      if (op == LIBXSMM_MELTW_TYPE_UNARY_GELU ) {
        libxsmm_generator_prepare_coeffs_gelu_ps_minimax3_avx512_vl256( io_generated_code,
            i_micro_kernel_config->vec_thres,
            i_micro_kernel_config->vec_absmask,
            i_micro_kernel_config->vec_scale,
            i_micro_kernel_config->vec_shifter,
            i_micro_kernel_config->vec_halves,
            i_micro_kernel_config->vec_c0_lo,
            i_micro_kernel_config->vec_c0_hi,
            i_micro_kernel_config->vec_c1_lo,
            i_micro_kernel_config->vec_c1_hi,
            i_micro_kernel_config->vec_c2_lo,
            i_micro_kernel_config->vec_c2_hi );
      }

      if (op == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV ) {
        libxsmm_generator_prepare_coeffs_gelu_inv_ps_minimax3_avx512_vl256( io_generated_code,
            i_micro_kernel_config->vec_thres,
            i_micro_kernel_config->vec_absmask,
            i_micro_kernel_config->vec_scale,
            i_micro_kernel_config->vec_shifter,
            i_micro_kernel_config->vec_halves,
            i_micro_kernel_config->vec_c0_lo,
            i_micro_kernel_config->vec_c0_hi,
            i_micro_kernel_config->vec_c1_lo,
            i_micro_kernel_config->vec_c1_hi,
            i_micro_kernel_config->vec_c2_lo,
            i_micro_kernel_config->vec_c2_hi );
      }
    } else {
      reserved_zmms += 14;

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

      if (op == LIBXSMM_MELTW_TYPE_UNARY_GELU ) {
        libxsmm_generator_prepare_coeffs_gelu_ps_minimax3_avx512( io_generated_code,
            i_micro_kernel_config->vec_thres,
            i_micro_kernel_config->vec_absmask,
            i_micro_kernel_config->vec_scale,
            i_micro_kernel_config->vec_shifter,
            i_micro_kernel_config->vec_halves,
            i_micro_kernel_config->vec_c0,
            i_micro_kernel_config->vec_c1,
            i_micro_kernel_config->vec_c2 );
      }

      if (op == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV ) {
        libxsmm_generator_prepare_coeffs_gelu_inv_ps_minimax3_avx512( io_generated_code,
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

    i_micro_kernel_config->reserved_zmms = reserved_zmms;
  }

  if ((op == LIBXSMM_MELTW_TYPE_UNARY_EXP) || (op == LIBXSMM_MELTW_TYPE_UNARY_ELU)) {
    unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;
    if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
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

      libxsmm_generator_prepare_coeffs_exp_ps_3dts_avx( io_generated_code,
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
      if (libxsmm_cpuid_x86_use_high_prec_eltwise_approx() == 0) {
        reserved_zmms += 6;
        i_micro_kernel_config->vec_halves     = reserved_zmms - 1;
        i_micro_kernel_config->vec_c0         = reserved_zmms - 2;
        i_micro_kernel_config->vec_c1         = reserved_zmms - 3;
        i_micro_kernel_config->vec_c2         = reserved_zmms - 4;
        i_micro_kernel_config->vec_c3         = reserved_zmms - 5;
        i_micro_kernel_config->vec_log2e      = reserved_zmms - 6;

        libxsmm_generator_prepare_coeffs_exp_ps_3dts_avx512( io_generated_code,
            i_micro_kernel_config->vec_c0,
            i_micro_kernel_config->vec_c1,
            i_micro_kernel_config->vec_c2,
            i_micro_kernel_config->vec_c3,
            i_micro_kernel_config->vec_halves,
            i_micro_kernel_config->vec_log2e,
            i_micro_kernel_config->vector_name);
      } else {
        unsigned int reserved_mask_regs = i_micro_kernel_config->reserved_mask_regs;
        reserved_zmms += 14;
        reserved_mask_regs += 1;
        i_micro_kernel_config->vec_halves     = reserved_zmms - 1;
        i_micro_kernel_config->vec_c0         = reserved_zmms - 2;
        i_micro_kernel_config->vec_c1         = reserved_zmms - 3;
        i_micro_kernel_config->vec_c2         = reserved_zmms - 4;
        i_micro_kernel_config->vec_c3         = reserved_zmms - 5;
        i_micro_kernel_config->vec_c4         = reserved_zmms - 6;
        i_micro_kernel_config->vec_c5         = reserved_zmms - 7;
        i_micro_kernel_config->vec_log2e      = reserved_zmms - 8;
        i_micro_kernel_config->vec_ln2        = reserved_zmms - 9;
        i_micro_kernel_config->vec_logfmax    = reserved_zmms - 10;
        i_micro_kernel_config->vec_logfmin    = reserved_zmms - 11;
        i_micro_kernel_config->vec_y          = reserved_zmms - 12;
        i_micro_kernel_config->vec_z          = reserved_zmms - 13;
        i_micro_kernel_config->vec_expmask    = reserved_zmms - 14;
        i_micro_kernel_config->aux_mask       = reserved_mask_regs - 1;

        libxsmm_generator_prepare_coeffs_exp_ps_5dts_avx512( io_generated_code,
            i_micro_kernel_config->vec_c0,
            i_micro_kernel_config->vec_c1,
            i_micro_kernel_config->vec_c2,
            i_micro_kernel_config->vec_c3,
            i_micro_kernel_config->vec_c4,
            i_micro_kernel_config->vec_c5,
            i_micro_kernel_config->vec_halves,
            i_micro_kernel_config->vec_log2e,
            i_micro_kernel_config->vec_ln2,
            i_micro_kernel_config->vec_expmask,
            i_micro_kernel_config->vec_logfmax,
            i_micro_kernel_config->vec_logfmin,
            i_micro_kernel_config->vector_name );

        i_micro_kernel_config->reserved_mask_regs = reserved_mask_regs;
      }
    }
    i_micro_kernel_config->reserved_zmms = reserved_zmms;
  }

  if (op == LIBXSMM_MELTW_TYPE_UNARY_TANH || op == LIBXSMM_MELTW_TYPE_UNARY_TANH_INV) {
    unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;
    unsigned int reserved_mask_regs = i_micro_kernel_config->reserved_mask_regs;
    reserved_zmms += 14;
    reserved_mask_regs += 2;
    if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
      if (i_micro_kernel_config->use_fp32bf16_cvt_replacement > 0) {
        reserved_zmms -= 2;
      }
    }

    i_micro_kernel_config->mask_hi       = reserved_mask_regs - 1;
    i_micro_kernel_config->mask_lo       = reserved_mask_regs - 2;
    i_micro_kernel_config->vec_x2        = reserved_zmms - 1;
    i_micro_kernel_config->vec_ones      = reserved_zmms - 2;
    i_micro_kernel_config->vec_neg_ones  = reserved_zmms - 3;
    i_micro_kernel_config->vec_c0        = reserved_zmms - 4;
    i_micro_kernel_config->vec_c1        = reserved_zmms - 5;
    i_micro_kernel_config->vec_c2        = reserved_zmms - 6;
    i_micro_kernel_config->vec_c3        = reserved_zmms - 7;
    i_micro_kernel_config->vec_c1_d      = reserved_zmms - 8;
    i_micro_kernel_config->vec_c2_d      = reserved_zmms - 9;
    i_micro_kernel_config->vec_c3_d      = reserved_zmms - 10;
    i_micro_kernel_config->vec_hi_bound  = reserved_zmms - 11;
    i_micro_kernel_config->vec_lo_bound  = reserved_zmms - 12;
    i_micro_kernel_config->vec_nom       = reserved_zmms - 13;
    i_micro_kernel_config->vec_denom     = reserved_zmms - 14;

    if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
      libxsmm_generator_prepare_coeffs_tanh_ps_rational_78_avx( io_generated_code,
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
          i_micro_kernel_config->vec_neg_ones );
      if (i_micro_kernel_config->use_fp32bf16_cvt_replacement > 0) {
        i_micro_kernel_config->vec_nom          = i_micro_kernel_config->dcvt_zmm_aux0;
        i_micro_kernel_config->vec_denom        = i_micro_kernel_config->dcvt_zmm_aux1;
      }
    } else {
      libxsmm_generator_prepare_coeffs_tanh_ps_rational_78_avx512( io_generated_code,
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
          i_micro_kernel_config->vector_name );
    }

    i_micro_kernel_config->reserved_mask_regs = reserved_mask_regs;
    i_micro_kernel_config->reserved_zmms = reserved_zmms;
  }

  if (op == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID || op == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV) {
    unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;
    unsigned int reserved_mask_regs = i_micro_kernel_config->reserved_mask_regs;
    reserved_zmms += 15;
    reserved_mask_regs += 2;

    if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
      reserved_zmms--;
      if (i_micro_kernel_config->use_fp32bf16_cvt_replacement > 0) {
        reserved_zmms -= 2;
      }
    }

    i_micro_kernel_config->mask_hi       = reserved_mask_regs - 1;
    i_micro_kernel_config->mask_lo       = reserved_mask_regs - 2;
    i_micro_kernel_config->vec_x2        = reserved_zmms - 1;
    i_micro_kernel_config->vec_neg_ones  = reserved_zmms - 2;
    i_micro_kernel_config->vec_c0        = reserved_zmms - 3;
    i_micro_kernel_config->vec_c1        = reserved_zmms - 4;
    i_micro_kernel_config->vec_c2        = reserved_zmms - 5;
    i_micro_kernel_config->vec_c3        = reserved_zmms - 6;
    i_micro_kernel_config->vec_c1_d      = reserved_zmms - 7;
    i_micro_kernel_config->vec_c2_d      = reserved_zmms - 8;
    i_micro_kernel_config->vec_c3_d      = reserved_zmms - 9;
    i_micro_kernel_config->vec_hi_bound  = reserved_zmms - 10;
    i_micro_kernel_config->vec_lo_bound  = reserved_zmms - 11;
    i_micro_kernel_config->vec_ones      = reserved_zmms - 12;
    i_micro_kernel_config->vec_nom       = reserved_zmms - 13;
    i_micro_kernel_config->vec_denom     = reserved_zmms - 14;
    i_micro_kernel_config->vec_halves    = reserved_zmms - 15;
    if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
      libxsmm_generator_prepare_coeffs_sigmoid_ps_rational_78_avx( io_generated_code,
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
         i_micro_kernel_config->vec_neg_ones );
      if (i_micro_kernel_config->use_fp32bf16_cvt_replacement > 0) {
        i_micro_kernel_config->vec_nom          = i_micro_kernel_config->dcvt_zmm_aux0;
        i_micro_kernel_config->vec_denom        = i_micro_kernel_config->dcvt_zmm_aux1;
      }
    } else {
      libxsmm_generator_prepare_coeffs_sigmoid_ps_rational_78_avx512( io_generated_code,
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
         i_micro_kernel_config->vector_name );
    }

    i_micro_kernel_config->reserved_mask_regs = reserved_mask_regs;
    i_micro_kernel_config->reserved_zmms = reserved_zmms;
  }

  if (op == LIBXSMM_MELTW_TYPE_UNARY_QUANT || op == LIBXSMM_MELTW_TYPE_UNARY_DEQUANT) {
    if ((flags & LIBXSMM_MELTW_FLAG_UNARY_NO_SCF_QUANT) == 0) {
      i_micro_kernel_config->quant_vreg_scf = i_micro_kernel_config->reserved_zmms;
      i_micro_kernel_config->reserved_zmms++;
    }

    if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX && op == LIBXSMM_MELTW_TYPE_UNARY_QUANT && i_micro_kernel_config->datatype_size_out != 4) {
      unsigned char perm_bytes[32] = { 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12};
      unsigned char perm_sbytes[32] = { 0, 1, 2, 3, 16, 17, 18, 19, 0, 1, 2, 3, 16, 17, 18, 19, 0, 1, 2, 3, 16, 17, 18, 19, 0, 1, 2, 3, 16, 17, 18, 19};
      unsigned int perm_words[8]  = { 0, 4, 0, 4, 0, 4, 0, 4};
      unsigned char perm_bytes_i16[32] = { 0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13};
      unsigned char perm_sbytes_i16[32] = { 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
      unsigned int perm_words_i16[8]  = { 0, 1, 4, 5, 0, 1, 4, 5};      i_micro_kernel_config->vec_tmp1       = i_micro_kernel_config->reserved_zmms;
      i_micro_kernel_config->vec_tmp2       = i_micro_kernel_config->reserved_zmms+1;
      i_micro_kernel_config->reserved_zmms += 2;
      if (i_micro_kernel_config->datatype_size_out == 1) {
        libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) (((flags & LIBXSMM_MELTW_FLAG_UNARY_SIGN_SAT_QUANT) > 0) ? perm_sbytes : perm_bytes), "perm_bytes_", 'y', i_micro_kernel_config->vec_tmp1 );
        libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) perm_words, "perm_words_", 'y', i_micro_kernel_config->vec_tmp2 );
      } else if (i_micro_kernel_config->datatype_size_out == 2) {
        libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) (((flags & LIBXSMM_MELTW_FLAG_UNARY_SIGN_SAT_QUANT) > 0) ? perm_sbytes_i16 : perm_bytes_i16), "perm_bytes_", 'y', i_micro_kernel_config->vec_tmp1 );
        libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) perm_words_i16, "perm_words_", 'y', i_micro_kernel_config->vec_tmp2 );
      } else {
        /* Should not happen */
      }
    }

    /* TODO: need fixing in case of different scaling (per channel etc) */
    if ((flags & LIBXSMM_MELTW_FLAG_UNARY_NO_SCF_QUANT) == 0) {
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch,
                                        LIBXSMM_X86_INSTR_VBROADCASTSS,
                                        i_gp_reg_aux0, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                        vname, i_micro_kernel_config->quant_vreg_scf, 0, 1, 0 );
    }

  }

  if (op == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X2 || op == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3) {
    unsigned int l_mask = 0xaaaaaaaa;
    i_micro_kernel_config->vec_tmp0           = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->vec_tmp1           = i_micro_kernel_config->reserved_zmms + 1;
    i_micro_kernel_config->vec_tmp2           = i_micro_kernel_config->reserved_zmms + 2;
    i_micro_kernel_config->reserved_zmms      = i_micro_kernel_config->reserved_zmms + 3;
    i_micro_kernel_config->mask_hi            = i_micro_kernel_config->reserved_mask_regs;
    i_micro_kernel_config->reserved_mask_regs = i_micro_kernel_config->reserved_mask_regs + 1;

    /* init mask for upper BF16 part */
    libxsmm_x86_instruction_alu_imm_i64( io_generated_code, LIBXSMM_X86_INSTR_MOVQ,
                                         i_gp_reg_tmp, l_mask );

    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
                                       i_gp_reg_tmp, i_micro_kernel_config->mask_hi );

    /* let's load the offsets */
    if ( op == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3 ) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
                                       i_micro_kernel_config->alu_mov_instruction,
                                       i_gp_reg_aux0,
                                       LIBXSMM_X86_GP_REG_UNDEF, 0,
                                       8,
                                       i_gp_reg_aux1,
                                       0 );
    }

    libxsmm_x86_instruction_alu_mem( io_generated_code,
                                     i_micro_kernel_config->alu_mov_instruction,
                                     i_gp_reg_aux0,
                                     LIBXSMM_X86_GP_REG_UNDEF, 0,
                                     0,
                                     i_gp_reg_aux0,
                                     0 );
  }

  /* load prng state into registers for stochastic rounding */
  if ( (flags & LIBXSMM_MELTW_FLAG_UNARY_STOCHASTIC_ROUND) > 0 ) {
    if ((io_generated_code->arch >= LIBXSMM_X86_AVX) && (io_generated_code->arch < LIBXSMM_X86_ALLFEAT)) {
      unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;
      reserved_zmms += 7;

      i_micro_kernel_config->prng_state0_vreg     = reserved_zmms - 1;
      i_micro_kernel_config->prng_state1_vreg     = reserved_zmms - 2;
      i_micro_kernel_config->prng_state2_vreg     = reserved_zmms - 3;
      i_micro_kernel_config->prng_state3_vreg     = reserved_zmms - 4;
      i_micro_kernel_config->prng_vreg_tmp0       = reserved_zmms - 5;
      i_micro_kernel_config->prng_vreg_tmp1       = reserved_zmms - 6;
      i_micro_kernel_config->prng_vreg_rand       = reserved_zmms - 7;

      libxsmm_generator_load_prng_state_avx_avx512( io_generated_code, vname, i_gp_reg_aux0,
                                                    i_micro_kernel_config->prng_state0_vreg, i_micro_kernel_config->prng_state1_vreg,
                                                    i_micro_kernel_config->prng_state2_vreg, i_micro_kernel_config->prng_state3_vreg );

      i_micro_kernel_config->reserved_zmms = reserved_zmms;
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_finalize_unary_kernel_vregs_masks( libxsmm_generated_code*                 io_generated_code,
                                                libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                unsigned int                            op,
                                                unsigned int                            flags,
                                                unsigned int                            i_gp_reg_tmp,
                                                const unsigned int                      i_gp_reg_aux0,
                                                const unsigned int                      i_gp_reg_aux1 ) {
  const char l_vname = (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 'y' : 'z';
  LIBXSMM_UNUSED(i_gp_reg_tmp);
  LIBXSMM_UNUSED(i_gp_reg_aux1);

  if ( op == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT || ((flags & LIBXSMM_MELTW_FLAG_UNARY_STOCHASTIC_ROUND) > 0) ) {
    libxsmm_generator_store_prng_state_avx_avx512( io_generated_code, l_vname, i_gp_reg_aux0,
                                                   i_micro_kernel_config->prng_state0_vreg, i_micro_kernel_config->prng_state1_vreg,
                                                   i_micro_kernel_config->prng_state2_vreg, i_micro_kernel_config->prng_state3_vreg );
  }
}

LIBXSMM_API_INTERN
void libxsmm_configure_kernel_vregs_masks( libxsmm_generated_code*                       io_generated_code,
                                                 libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_gp_reg_tmp,
                                                 const unsigned int                      i_gp_reg_aux0,
                                                 const unsigned int                      i_gp_reg_aux1) {
  /* initialize some values */
  i_micro_kernel_config->reserved_zmms = 0;
  i_micro_kernel_config->reserved_mask_regs = 1;
  i_micro_kernel_config->use_fp32bf16_cvt_replacement = 0;

  /* if we need FP32->BF16 downconverts and we do not have native instruction, then prepare stack */
  if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) || LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) &&
       LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) && ((io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX512_VL256_CPX) && (io_generated_code->arch != LIBXSMM_X86_AVX2_SRF)) ) {
    i_micro_kernel_config->use_fp32bf16_cvt_replacement = 1;
    libxsmm_generator_vcvtneps2bf16_avx512_prep_stack( io_generated_code, i_gp_reg_tmp );
    i_micro_kernel_config->dcvt_mask_aux0 = i_micro_kernel_config->reserved_mask_regs;
    i_micro_kernel_config->dcvt_mask_aux1 = i_micro_kernel_config->reserved_mask_regs + 1;
    i_micro_kernel_config->reserved_mask_regs = i_micro_kernel_config->reserved_mask_regs + 2;
    i_micro_kernel_config->dcvt_zmm_aux0 = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->dcvt_zmm_aux1 = i_micro_kernel_config->reserved_zmms + 1;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 2;
  } else if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) || LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) &&
       LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) && (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch <= LIBXSMM_X86_ALLFEAT)) {
    /* nothing to do */
  } else if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) || LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) &&
       LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) && (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch <= LIBXSMM_X86_ALLFEAT)) {
    libxsmm_generator_vcvtneps2bf8_avx512_prep_stack( io_generated_code, i_gp_reg_tmp );
    i_micro_kernel_config->dcvt_mask_aux0 = i_micro_kernel_config->reserved_mask_regs;
    i_micro_kernel_config->dcvt_mask_aux1 = i_micro_kernel_config->reserved_mask_regs + 1;
    i_micro_kernel_config->reserved_mask_regs = i_micro_kernel_config->reserved_mask_regs + 2;
    i_micro_kernel_config->dcvt_zmm_aux0 = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->dcvt_zmm_aux1 = i_micro_kernel_config->reserved_zmms + 1;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 2;
  } else if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) &&
        ((LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ||
         (LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) ) ) &&
              (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch <= LIBXSMM_X86_ALLFEAT)) {
    libxsmm_generator_vcvt_hf8_tofrom_f32_avx512_prep_stack( io_generated_code, i_gp_reg_tmp );
    i_micro_kernel_config->dcvt_mask_aux0 = i_micro_kernel_config->reserved_mask_regs;
    i_micro_kernel_config->dcvt_mask_aux1 = i_micro_kernel_config->reserved_mask_regs + 1;
    i_micro_kernel_config->reserved_mask_regs = i_micro_kernel_config->reserved_mask_regs + 2;
    i_micro_kernel_config->dcvt_zmm_aux0 = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->dcvt_zmm_aux1 = i_micro_kernel_config->reserved_zmms + 1;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 2;
    if ( LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) {
      i_micro_kernel_config->dcvt_zmm_aux2 = i_micro_kernel_config->reserved_zmms;
      i_micro_kernel_config->dcvt_zmm_aux3 = i_micro_kernel_config->reserved_zmms + 1;
      i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 2;
      i_micro_kernel_config->dcvt_mask_aux2 = i_micro_kernel_config->reserved_mask_regs;
      i_micro_kernel_config->reserved_mask_regs = i_micro_kernel_config->reserved_mask_regs + 1;
    }
  } else if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) || LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) &&
       LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) && (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch <= LIBXSMM_X86_ALLFEAT)) {
    libxsmm_generator_vcvt_hf8_tofrom_f32_avx512_prep_stack( io_generated_code, i_gp_reg_tmp );
    i_micro_kernel_config->dcvt_mask_aux0 = i_micro_kernel_config->reserved_mask_regs;
    i_micro_kernel_config->dcvt_mask_aux1 = i_micro_kernel_config->reserved_mask_regs + 1;
    i_micro_kernel_config->dcvt_mask_aux2 = i_micro_kernel_config->reserved_mask_regs + 2;
    i_micro_kernel_config->reserved_mask_regs = i_micro_kernel_config->reserved_mask_regs + 3;
    i_micro_kernel_config->dcvt_zmm_aux0 = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->dcvt_zmm_aux1 = i_micro_kernel_config->reserved_zmms + 1;
    i_micro_kernel_config->dcvt_zmm_aux2 = i_micro_kernel_config->reserved_zmms + 2;
    i_micro_kernel_config->dcvt_zmm_aux3 = i_micro_kernel_config->reserved_zmms + 3;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 4;
  }

  if ((io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) && (i_mateltwise_desc->m % i_micro_kernel_config->vlen_in != 0)) {
    i_micro_kernel_config->inout_vreg_mask = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
  }

  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
    const int datatype = libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP);
    libxsmm_configure_unary_kernel_vregs_masks( io_generated_code, i_micro_kernel_config, (libxsmm_datatype)datatype, i_mateltwise_desc->param, i_mateltwise_desc->flags, i_gp_reg_tmp, i_gp_reg_aux0, i_gp_reg_aux1);
  }

  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
    /* This is the temp register used to load the second input */
    i_micro_kernel_config->tmp_vreg = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;

    if (libxsmm_generator_mateltwise_is_binary_cmp_op(i_mateltwise_desc) > 0) {
      if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
        i_micro_kernel_config->tmp_vreg2 = i_micro_kernel_config->reserved_zmms;
        i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
      }
    }

    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_TERNARY_SELECT) {
      if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
        unsigned int const_mask_array[8] = { 0x00000001, 0x00000002 , 0x00000004, 0x00000008, 0x00000010, 0x00000020, 0x00000040 , 0x00000080 };
        i_micro_kernel_config->vec_tmp0 = i_micro_kernel_config->reserved_zmms;
        i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
        libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) const_mask_array, "const_mask_array", 'y', i_micro_kernel_config->vec_tmp0 );
        i_micro_kernel_config->blend_tmp_mask = i_micro_kernel_config->reserved_zmms;
        i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
      } else {
        i_micro_kernel_config->blend_tmp_mask = i_micro_kernel_config->reserved_mask_regs;
        i_micro_kernel_config->reserved_mask_regs = i_micro_kernel_config->reserved_mask_regs + 1;
      }
    }

    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_MULADD) {
      i_micro_kernel_config->tmp_vreg2 = i_micro_kernel_config->reserved_zmms;
      i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
    }

    /* incase we have stochastic round, we need to load the state */
    if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY && (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_STOCHASTIC_ROUND) > 0) || (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY && (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_STOCHASTIC_ROUND) > 0)) {
      if ((io_generated_code->arch >= LIBXSMM_X86_AVX) && (io_generated_code->arch < LIBXSMM_X86_ALLFEAT)) {
        const char l_vname = (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 'y' : 'z';
        unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;
        reserved_zmms += 7;

        i_micro_kernel_config->prng_state0_vreg     = reserved_zmms - 1;
        i_micro_kernel_config->prng_state1_vreg     = reserved_zmms - 2;
        i_micro_kernel_config->prng_state2_vreg     = reserved_zmms - 3;
        i_micro_kernel_config->prng_state3_vreg     = reserved_zmms - 4;
        i_micro_kernel_config->prng_vreg_tmp0       = reserved_zmms - 5;
        i_micro_kernel_config->prng_vreg_tmp1       = reserved_zmms - 6;
        i_micro_kernel_config->prng_vreg_rand       = reserved_zmms - 7;

        libxsmm_generator_load_prng_state_avx_avx512( io_generated_code, l_vname, i_gp_reg_aux0,
                                                      i_micro_kernel_config->prng_state0_vreg, i_micro_kernel_config->prng_state1_vreg,
                                                      i_micro_kernel_config->prng_state2_vreg, i_micro_kernel_config->prng_state3_vreg );

        i_micro_kernel_config->reserved_zmms = reserved_zmms;
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_finalize_kernel_vregs_masks( libxsmm_generated_code*                       io_generated_code,
                                                 libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                 unsigned int                            i_gp_reg_tmp,
                                                 const unsigned int                      i_gp_reg_aux0,
                                                 const unsigned int                      i_gp_reg_aux1) {
  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
    libxsmm_finalize_unary_kernel_vregs_masks( io_generated_code, i_micro_kernel_config, i_mateltwise_desc->param, i_mateltwise_desc->flags, i_gp_reg_tmp, i_gp_reg_aux0, i_gp_reg_aux1);
  }

  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
    /* incase we have stochastic round, we need to store the state */
    if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY && (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_STOCHASTIC_ROUND) > 0) || (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY && (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_STOCHASTIC_ROUND) > 0)) {
      const char l_vname = (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ? 'y' : 'z';
      libxsmm_generator_store_prng_state_avx_avx512( io_generated_code, l_vname, i_gp_reg_aux0,
                                                     i_micro_kernel_config->prng_state0_vreg, i_micro_kernel_config->prng_state1_vreg,
                                                     i_micro_kernel_config->prng_state2_vreg, i_micro_kernel_config->prng_state3_vreg );
    }
  }

  if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) || LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) &&
        LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) && (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch <= LIBXSMM_X86_ALLFEAT) ) {
    libxsmm_generator_vcvtneps2bf8_avx512_clean_stack( io_generated_code, i_gp_reg_tmp );
  } else if ( ((LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) &&
          ((LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ||(LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)))) &&
              (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch <= LIBXSMM_X86_ALLFEAT)) ||
      (( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) || LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) &&
       LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) && (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch <= LIBXSMM_X86_ALLFEAT)))) {
    libxsmm_generator_vcvt_hf8_tofrom_f32_avx512_clean_stack( io_generated_code, i_gp_reg_tmp );
  } else if (i_micro_kernel_config->use_fp32bf16_cvt_replacement == 1) {
    libxsmm_generator_vcvtneps2bf16_avx512_clean_stack( io_generated_code, i_gp_reg_tmp );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_unary_binary_2d_microkernel( libxsmm_generated_code*                     io_generated_code,
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

  /* Configure microkernel masks */
  libxsmm_setup_input_output_masks( io_generated_code, i_micro_kernel_config, i_mateltwise_desc,
      LIBXSMM_X86_GP_REG_R11, i_m, &use_m_input_masking, &mask_reg_in, &use_m_output_masking, &mask_reg_out);
  reserved_zmms = i_micro_kernel_config->reserved_zmms;

  /* Configure microkernel loops */
  libxsmm_configure_microkernel_loops( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc, i_m, i_n, use_m_input_masking,
    &m_trips, &n_trips, &m_unroll_factor, &n_unroll_factor, &m_assm_trips, &n_assm_trips,
    &out_loop_trips, &inner_loop_trips, &out_loop_bound, &inner_loop_bound, &out_loop_reg, &inner_loop_reg, &out_unroll_factor, &inner_unroll_factor );

  /* Headers of microkernel loops */
  if (out_loop_trips > 1) {
    libxsmm_generator_generic_loop_header(io_generated_code, io_loop_label_tracker, out_loop_reg, 0, out_unroll_factor);
  }

  if (inner_loop_trips > 1) {
    libxsmm_generator_generic_loop_header(io_generated_code, io_loop_label_tracker, inner_loop_reg, 0, inner_unroll_factor);
  }

  /* Load block of registers */
  libxsmm_load_2d_reg_block(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
      i_vlen_in, reserved_zmms, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in);

  /* Compute on registers */
  libxsmm_compute_unary_binary_2d_reg_block(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
      i_vlen_in, reserved_zmms, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in);

  /* Store block of registers */
  libxsmm_store_2d_reg_block(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
      i_vlen_out, reserved_zmms, m_unroll_factor, n_unroll_factor, use_m_output_masking, mask_reg_out);

  /* Footers of microkernel loops */
  if (inner_loop_trips > 1) {
    /* Advance input/output pointers */
    loop_type = (inner_loop_reg == i_gp_reg_mapping->gp_reg_m_loop) ? LOOP_TYPE_M : LOOP_TYPE_N;

    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_in, i_micro_kernel_config->alu_add_instruction, inner_unroll_factor, loop_type);

    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_out, i_micro_kernel_config->alu_add_instruction, inner_unroll_factor, loop_type);

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_out2, i_micro_kernel_config->alu_add_instruction, inner_unroll_factor, loop_type);
      }
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in2, i_micro_kernel_config->alu_add_instruction, inner_unroll_factor, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in3, i_micro_kernel_config->alu_add_instruction, inner_unroll_factor, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) ||
          (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
          (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU)        || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_relumask, i_micro_kernel_config->alu_add_instruction, inner_unroll_factor, loop_type);
      } else if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_dropoutmask, i_micro_kernel_config->alu_add_instruction, inner_unroll_factor, loop_type);
      }
    }

    libxsmm_generator_generic_loop_footer(io_generated_code, io_loop_label_tracker, inner_loop_reg, inner_loop_bound);

    /* Reset input/output pointers */
    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_in, i_micro_kernel_config->alu_sub_instruction, inner_unroll_factor * inner_loop_trips, loop_type);

    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_out, i_micro_kernel_config->alu_sub_instruction, inner_unroll_factor * inner_loop_trips, loop_type);

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_out2, i_micro_kernel_config->alu_sub_instruction, inner_unroll_factor * inner_loop_trips, loop_type);
      }
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in2, i_micro_kernel_config->alu_sub_instruction, inner_unroll_factor * inner_loop_trips, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in3, i_micro_kernel_config->alu_sub_instruction, inner_unroll_factor * inner_loop_trips, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) ||
          (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
          (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU)        || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_relumask, i_micro_kernel_config->alu_sub_instruction, inner_unroll_factor * inner_loop_trips, loop_type);
      } else if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_dropoutmask, i_micro_kernel_config->alu_sub_instruction, inner_unroll_factor * inner_loop_trips, loop_type);
      }
    }
  }

  if (out_loop_trips > 1) {
    /* Advance input/output pointers */
    loop_type = (out_loop_reg == i_gp_reg_mapping->gp_reg_m_loop) ? LOOP_TYPE_M : LOOP_TYPE_N;

    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_in, i_micro_kernel_config->alu_add_instruction, out_unroll_factor, loop_type);

    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_out, i_micro_kernel_config->alu_add_instruction, out_unroll_factor, loop_type);

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_out2, i_micro_kernel_config->alu_add_instruction, out_unroll_factor, loop_type);
      }
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in2, i_micro_kernel_config->alu_add_instruction, out_unroll_factor, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in3, i_micro_kernel_config->alu_add_instruction, out_unroll_factor, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) ||
          (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
          (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU)        || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_relumask, i_micro_kernel_config->alu_add_instruction, out_unroll_factor, loop_type);
      } else if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_dropoutmask, i_micro_kernel_config->alu_add_instruction, out_unroll_factor, loop_type);
      }
    }

    libxsmm_generator_generic_loop_footer(io_generated_code, io_loop_label_tracker, out_loop_reg, out_loop_bound);

    /* Reset input/output pointers */
    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_in, i_micro_kernel_config->alu_sub_instruction, out_unroll_factor * out_loop_trips, loop_type);

    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_out, i_micro_kernel_config->alu_sub_instruction, out_unroll_factor * out_loop_trips, loop_type);

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_out2, i_micro_kernel_config->alu_sub_instruction, out_unroll_factor * out_loop_trips, loop_type);
      }
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in2, i_micro_kernel_config->alu_sub_instruction, out_unroll_factor * out_loop_trips, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in3, i_micro_kernel_config->alu_sub_instruction, out_unroll_factor * out_loop_trips, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) ||
          (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
          (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU)        || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_relumask, i_micro_kernel_config->alu_sub_instruction, out_unroll_factor * out_loop_trips, loop_type);
      } else if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_dropoutmask, i_micro_kernel_config->alu_sub_instruction, out_unroll_factor * out_loop_trips, loop_type);
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_unary_binary_avx512_microkernel( libxsmm_generated_code*          io_generated_code,
                                                 libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                 libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                 libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                 const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  unsigned int loop_order, m_blocking, out_blocking, out_bound, out_block = 0, n_blocking, inner_blocking, inner_block, inner_bound, n_microkernel = 0, m_microkernel = 0;
  unsigned int out_ind, inner_ind, reset_regs, loop_type;
  unsigned int available_vregs = 32;
  unsigned int l_gp_reg_tmp = LIBXSMM_X86_GP_REG_R11;
  unsigned int l_gp_reg_aux0 = LIBXSMM_X86_GP_REG_UNDEF;
  unsigned int l_gp_reg_aux1 = LIBXSMM_X86_GP_REG_UNDEF;
  unsigned int bcast_row = (((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW) > 0)) ||
                            ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0) > 0)) ||
                            ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1) > 0)) ||
                            ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_0) > 0)) ||
                            ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_1) > 0)) ||
                            ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_2) > 0))) ? 1 : 0;
  unsigned int bcast_scalar = (((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR) > 0)) ||
                               ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0) > 0)) ||
                               ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1) > 0)) ||
                               ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_0) > 0)) ||
                               ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1) > 0)) ||
                               ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2) > 0))) ? 1 : 0;

  /* Some rudimentary checking of M, N and LDs*/
  if ( ((i_mateltwise_desc->m > i_mateltwise_desc->ldi) && !(bcast_row > 0 || bcast_scalar > 0)) ||
       (i_mateltwise_desc->m > i_mateltwise_desc->ldo)    ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
    return;
  }

  /* check datatype */
  if ( ((LIBXSMM_DATATYPE_F32  == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ||
         LIBXSMM_DATATYPE_U32  == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ||
         LIBXSMM_DATATYPE_I32  == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ||
         LIBXSMM_DATATYPE_F64  == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ||
         LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ||
         LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ||
         LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ||
         LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ||
         LIBXSMM_DATATYPE_U16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ||
         LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) )
       &&
       ( LIBXSMM_DATATYPE_F32  == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ||
         LIBXSMM_DATATYPE_U32  == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ||
         LIBXSMM_DATATYPE_I32  == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ||
         LIBXSMM_DATATYPE_F64  == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ||
         LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ||
         LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ||
         LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ||
         LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ||
         LIBXSMM_DATATYPE_U16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ||
         LIBXSMM_DATATYPE_IMPLICIT == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ||
         LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)))      ||
       (  LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) && LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ||
       ( LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) && LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ||
       ( LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) && LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ||
       ( LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&  LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ||
       ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&  LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ||
       ( LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) && LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ||
       ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) && LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ||
       ( LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) && LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ||
       ( LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) && LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) )    ) {
    /* fine */
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }

  /* Configure vlens */
  if ( (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_UNZIP) ) {
    if ((LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ||
         LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ||
         LIBXSMM_DATATYPE_U32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) &&
       (LIBXSMM_DATATYPE_U16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) &&
       (LIBXSMM_DATATYPE_IMPLICIT == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP))  ) {
      i_micro_kernel_config->vlen_in = 16;
      i_micro_kernel_config->vlen_comp = 16;
      i_micro_kernel_config->vlen_out = 16;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  } else if ( (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_ZIP) ) {
    if ((LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ||
         LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ||
         LIBXSMM_DATATYPE_U32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) &&
       (LIBXSMM_DATATYPE_U16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) && LIBXSMM_DATATYPE_U16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) &&
       (LIBXSMM_DATATYPE_IMPLICIT == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP))  ) {
      i_micro_kernel_config->vlen_in = 16;
      i_micro_kernel_config->vlen_comp = 16;
      i_micro_kernel_config->vlen_out = 16;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  } else {
    libxsmm_generator_configure_avx512_vlens(i_mateltwise_desc, i_micro_kernel_config);
  }
  if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) {
    i_micro_kernel_config->vlen_in = i_micro_kernel_config->vlen_in/2;
    i_micro_kernel_config->vlen_out = i_micro_kernel_config->vlen_out/2;
    available_vregs = 16;
  } else if (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) {
    i_micro_kernel_config->vlen_in = i_micro_kernel_config->vlen_in/2;
    i_micro_kernel_config->vlen_out = i_micro_kernel_config->vlen_out/2;
    available_vregs = 32;
  }

  /* set mask lds */
  if ( (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) ) {
    i_micro_kernel_config->ldo_mask = (i_mateltwise_desc->ldo+15) - ((i_mateltwise_desc->ldo+15)%16);
    i_micro_kernel_config->ldi_mask = (i_mateltwise_desc->ldi+15) - ((i_mateltwise_desc->ldi+15)%16);
  }

  if ((libxsmm_generator_mateltwise_is_binary_cmp_op(i_mateltwise_desc) > 0) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BITMASK_2BYTEMULT) > 0)) {
    i_micro_kernel_config->ldo_mask = (i_mateltwise_desc->ldo+15) - ((i_mateltwise_desc->ldo+15)%16);
  }

  /* let's check that we have bitmask set for dropput and relu backward */
  if ( ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
         (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV) ) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) == 0) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BITMASK_REQUIRED );
    return;
  }

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in     = LIBXSMM_X86_GP_REG_R8;
  i_gp_reg_mapping->gp_reg_out    = LIBXSMM_X86_GP_REG_R9;
  i_gp_reg_mapping->gp_reg_m_loop = LIBXSMM_X86_GP_REG_R10;
  i_gp_reg_mapping->gp_reg_n_loop = LIBXSMM_X86_GP_REG_R11;
  i_gp_reg_mapping->gp_reg_scratch_0 = LIBXSMM_X86_GP_REG_RCX;
  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
    i_gp_reg_mapping->gp_reg_in2  = LIBXSMM_X86_GP_REG_RAX;
    if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_STOCHASTIC_ROUND) > 0 ) {
      i_gp_reg_mapping->gp_reg_prngstate = LIBXSMM_X86_GP_REG_RDX;
    }
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
    i_gp_reg_mapping->gp_reg_in2  = LIBXSMM_X86_GP_REG_RAX;
    i_gp_reg_mapping->gp_reg_in3  = LIBXSMM_X86_GP_REG_R12;
    if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_STOCHASTIC_ROUND) > 0 ) {
      i_gp_reg_mapping->gp_reg_prngstate = LIBXSMM_X86_GP_REG_RDX;
    }
  } else {
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP) {
      i_gp_reg_mapping->gp_reg_out2 = LIBXSMM_X86_GP_REG_RAX;
    }
    if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV)       ||
         (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV)    ) {
      i_gp_reg_mapping->gp_reg_relumask = LIBXSMM_X86_GP_REG_RAX;
    }
    if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
      i_gp_reg_mapping->gp_reg_relumask = LIBXSMM_X86_GP_REG_RAX;
      i_gp_reg_mapping->gp_reg_fam_lualpha = LIBXSMM_X86_GP_REG_RCX;
    }
    if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ) {
      i_gp_reg_mapping->gp_reg_fam_lualpha = LIBXSMM_X86_GP_REG_RCX;
    }
    if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_UNZIP) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X2) ||
                                                                         (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3)    ) {
      i_gp_reg_mapping->gp_reg_offset = LIBXSMM_X86_GP_REG_RAX;
    } else {
      i_gp_reg_mapping->gp_reg_offset = LIBXSMM_X86_GP_REG_UNDEF;
    }
    if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3) ) {
      i_gp_reg_mapping->gp_reg_offset_2 = LIBXSMM_X86_GP_REG_RDX;
    } else {
      i_gp_reg_mapping->gp_reg_offset_2 = LIBXSMM_X86_GP_REG_UNDEF;
    }
    if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV) ) {
      i_gp_reg_mapping->gp_reg_dropoutmask = LIBXSMM_X86_GP_REG_RAX;
      i_gp_reg_mapping->gp_reg_dropoutprob = LIBXSMM_X86_GP_REG_RCX;
      if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT ) {
        i_gp_reg_mapping->gp_reg_prngstate = LIBXSMM_X86_GP_REG_RDX;
      } else {
        i_gp_reg_mapping->gp_reg_prngstate = LIBXSMM_X86_GP_REG_UNDEF;
      }
    } else {
      i_gp_reg_mapping->gp_reg_dropoutmask = LIBXSMM_X86_GP_REG_UNDEF;
      i_gp_reg_mapping->gp_reg_dropoutprob = LIBXSMM_X86_GP_REG_UNDEF;
      i_gp_reg_mapping->gp_reg_prngstate = LIBXSMM_X86_GP_REG_UNDEF;
    }
    if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_QUANT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DEQUANT) ) {
      i_gp_reg_mapping->gp_reg_quant_sf = LIBXSMM_X86_GP_REG_RAX;
    } else {
      i_gp_reg_mapping->gp_reg_quant_sf = LIBXSMM_X86_GP_REG_UNDEF;
    }
    if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_STOCHASTIC_ROUND) > 0 ) {
      i_gp_reg_mapping->gp_reg_prngstate = LIBXSMM_X86_GP_REG_RDX;
    }
  }

  /* load the input pointer and output pointer */
  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY ) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        32,
        i_gp_reg_mapping->gp_reg_in,
        0 );

     libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        64,
        i_gp_reg_mapping->gp_reg_out,
        0 );

    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          72,
          i_gp_reg_mapping->gp_reg_out2,
          0 );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          72,
          i_gp_reg_mapping->gp_reg_relumask,
          0 );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          40,
          i_gp_reg_mapping->gp_reg_relumask,
          0 );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU)  {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          0,
          i_gp_reg_mapping->gp_reg_fam_lualpha,
          0 );
      l_gp_reg_aux1 = i_gp_reg_mapping->gp_reg_fam_lualpha;
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          40,
          i_gp_reg_mapping->gp_reg_relumask,
          0 );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          0,
          i_gp_reg_mapping->gp_reg_fam_lualpha,
          0 );
      l_gp_reg_aux1 = i_gp_reg_mapping->gp_reg_fam_lualpha;
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          0,
          i_gp_reg_mapping->gp_reg_fam_lualpha,
          0 );
       libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          72,
          i_gp_reg_mapping->gp_reg_relumask,
          0 );
      l_gp_reg_aux1 = i_gp_reg_mapping->gp_reg_fam_lualpha;
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          40,
          i_gp_reg_mapping->gp_reg_relumask,
          0 );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          0,
          i_gp_reg_mapping->gp_reg_fam_lualpha,
          0 );
      l_gp_reg_aux1 = i_gp_reg_mapping->gp_reg_fam_lualpha;
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_UNZIP) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          72,
          i_gp_reg_mapping->gp_reg_offset,
          0 );
      libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_offset, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, i_gp_reg_mapping->gp_reg_offset, 0 );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X2) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          72,
          i_gp_reg_mapping->gp_reg_offset,
          0 );
      l_gp_reg_aux0 = i_gp_reg_mapping->gp_reg_offset;
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          72,
          i_gp_reg_mapping->gp_reg_offset,
          0 );
      l_gp_reg_aux0 = i_gp_reg_mapping->gp_reg_offset;
      l_gp_reg_aux1 = i_gp_reg_mapping->gp_reg_offset_2;
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          8,
          i_gp_reg_mapping->gp_reg_prngstate,
          0 );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          0,
          i_gp_reg_mapping->gp_reg_dropoutprob,
          0 );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          72,
          i_gp_reg_mapping->gp_reg_dropoutmask,
          0 );
      l_gp_reg_aux0 = i_gp_reg_mapping->gp_reg_prngstate;
      l_gp_reg_aux1 = i_gp_reg_mapping->gp_reg_dropoutprob;
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          40,
          i_gp_reg_mapping->gp_reg_dropoutmask,
          0 );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          0,
          i_gp_reg_mapping->gp_reg_dropoutprob,
          0 );
      l_gp_reg_aux1 = i_gp_reg_mapping->gp_reg_dropoutprob;
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_QUANT) {
      if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_NO_SCF_QUANT) == 0) {
        libxsmm_x86_instruction_alu_mem( io_generated_code,
            i_micro_kernel_config->alu_mov_instruction,
            i_gp_reg_mapping->gp_reg_param_struct,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            40,
            i_gp_reg_mapping->gp_reg_quant_sf,
            0 );
      }
      l_gp_reg_aux0 = i_gp_reg_mapping->gp_reg_quant_sf;
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DEQUANT) {
      if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_NO_SCF_QUANT) == 0) {
        libxsmm_x86_instruction_alu_mem( io_generated_code,
            i_micro_kernel_config->alu_mov_instruction,
            i_gp_reg_mapping->gp_reg_param_struct,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            40,
            i_gp_reg_mapping->gp_reg_quant_sf,
            0 );
      }
      l_gp_reg_aux0 = i_gp_reg_mapping->gp_reg_quant_sf;
    } else if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_STOCHASTIC_ROUND) > 0 ) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          8,
          i_gp_reg_mapping->gp_reg_prngstate,
          0 );
      l_gp_reg_aux0 = i_gp_reg_mapping->gp_reg_prngstate;
    }
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY ) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        32,
        i_gp_reg_mapping->gp_reg_in,
        0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        64,
        i_gp_reg_mapping->gp_reg_in2,
        0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        96,
        i_gp_reg_mapping->gp_reg_out,
        0 );
    if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_STOCHASTIC_ROUND) > 0 ) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          8,
          i_gp_reg_mapping->gp_reg_prngstate,
          0 );
      l_gp_reg_aux0 = i_gp_reg_mapping->gp_reg_prngstate;
    }
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY ) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        32,
        i_gp_reg_mapping->gp_reg_in,
        0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        64,
        i_gp_reg_mapping->gp_reg_in2,
        0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        96,
        i_gp_reg_mapping->gp_reg_in3,
        0 );

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        128,
        i_gp_reg_mapping->gp_reg_out,
        0 );
    if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_STOCHASTIC_ROUND) > 0 ) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          8,
          i_gp_reg_mapping->gp_reg_prngstate,
          0 );
      l_gp_reg_aux0 = i_gp_reg_mapping->gp_reg_prngstate;
    }
  } else {
    /* This hsould not happen */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }

  /* Based on kernel type reserve zmms and mask registers */
  libxsmm_configure_kernel_vregs_masks( io_generated_code, i_micro_kernel_config, i_mateltwise_desc, l_gp_reg_tmp, l_gp_reg_aux0, l_gp_reg_aux1 );

  available_vregs = available_vregs - i_micro_kernel_config->reserved_zmms;

  /* Configure M and N blocking factors */
  libxsmm_generator_configure_M_N_blocking(i_mateltwise_desc->m, i_mateltwise_desc->n, i_micro_kernel_config->vlen_in, &m_blocking, &n_blocking, available_vregs);
  libxsmm_generator_configure_loop_order(i_mateltwise_desc, &loop_order, &m_blocking, &n_blocking, &out_blocking, &inner_blocking, &out_bound, &inner_bound);
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

      libxsmm_generator_unary_binary_2d_microkernel(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc, m_microkernel, n_microkernel);

      inner_ind += inner_block;

      if (inner_ind != inner_bound) {
        reset_regs = 1;
        /* Advance input/output pointers */
        loop_type = (loop_order == NM_LOOP_ORDER) ? LOOP_TYPE_M : LOOP_TYPE_N;

        libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_in, i_micro_kernel_config->alu_add_instruction, m_microkernel, n_microkernel, loop_type );

        libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_out, i_micro_kernel_config->alu_add_instruction, m_microkernel, n_microkernel, loop_type );

        if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
          if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP) {
            libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
                i_gp_reg_mapping->gp_reg_out2, i_micro_kernel_config->alu_add_instruction, m_microkernel, n_microkernel, loop_type );
          }
        }

        if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
          libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
              i_gp_reg_mapping->gp_reg_in2, i_micro_kernel_config->alu_add_instruction, m_microkernel, n_microkernel, loop_type );
        }


        if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
          libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
              i_gp_reg_mapping->gp_reg_in3, i_micro_kernel_config->alu_add_instruction, m_microkernel, n_microkernel, loop_type );
        }

        if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
          if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) ||
              (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
              (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU)        || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
            libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
                i_gp_reg_mapping->gp_reg_relumask, i_micro_kernel_config->alu_add_instruction, m_microkernel, n_microkernel, loop_type );
          } else if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
            libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
                i_gp_reg_mapping->gp_reg_dropoutmask, i_micro_kernel_config->alu_add_instruction, m_microkernel, n_microkernel, loop_type );
          }
        }
      }
    }

    /* If needed, readjust the registers */
    if (reset_regs == 1) {
      loop_type = (loop_order == NM_LOOP_ORDER) ? LOOP_TYPE_M : LOOP_TYPE_N;

      libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in, i_micro_kernel_config->alu_sub_instruction, m_microkernel, n_microkernel, loop_type );

      libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_out, i_micro_kernel_config->alu_sub_instruction, m_microkernel, n_microkernel, loop_type );

      if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
        if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP) {
          libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_out2, i_micro_kernel_config->alu_sub_instruction, m_microkernel, n_microkernel, loop_type );
        }
      }

      if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
        libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_in2, i_micro_kernel_config->alu_sub_instruction, m_microkernel, n_microkernel, loop_type );
      }

      if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
        libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_in3, i_micro_kernel_config->alu_sub_instruction, m_microkernel, n_microkernel, loop_type );
      }

      if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
        if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) ||
            (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
            (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU)        || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
          libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
              i_gp_reg_mapping->gp_reg_relumask, i_micro_kernel_config->alu_sub_instruction, m_microkernel, n_microkernel, loop_type );
        } else if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
          libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
              i_gp_reg_mapping->gp_reg_dropoutmask, i_micro_kernel_config->alu_sub_instruction, m_microkernel, n_microkernel, loop_type );
        }
      }
    }

    out_ind += out_block;
    if (out_ind != out_bound) {
      /* Advance input/output pointers */
      loop_type = (loop_order == MN_LOOP_ORDER) ? LOOP_TYPE_M : LOOP_TYPE_N;

      libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in, i_micro_kernel_config->alu_add_instruction, m_microkernel, n_microkernel, loop_type );

      libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_out, i_micro_kernel_config->alu_add_instruction, m_microkernel, n_microkernel, loop_type );

      if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
        if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP) {
          libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
              i_gp_reg_mapping->gp_reg_out2, i_micro_kernel_config->alu_add_instruction, m_microkernel, n_microkernel, loop_type );
        }
      }

      if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
        libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_in2, i_micro_kernel_config->alu_add_instruction, m_microkernel, n_microkernel, loop_type );
      }

      if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
        libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_in3, i_micro_kernel_config->alu_add_instruction, m_microkernel, n_microkernel, loop_type );
      }

      if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
        if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) ||
            (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
            (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU)        || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
          libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
              i_gp_reg_mapping->gp_reg_relumask, i_micro_kernel_config->alu_add_instruction, m_microkernel, n_microkernel, loop_type );
        } else if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
          libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
              i_gp_reg_mapping->gp_reg_dropoutmask, i_micro_kernel_config->alu_add_instruction, m_microkernel, n_microkernel, loop_type );
        }
      }
    }
  }

  /* save some globale state if needed */
  libxsmm_finalize_kernel_vregs_masks( io_generated_code, i_micro_kernel_config, i_mateltwise_desc, l_gp_reg_tmp, l_gp_reg_aux0, l_gp_reg_aux1 );
}

#undef MN_LOOP_ORDER
#undef NM_LOOP_ORDER
#undef LOOP_TYPE_M
#undef LOOP_TYPE_N

