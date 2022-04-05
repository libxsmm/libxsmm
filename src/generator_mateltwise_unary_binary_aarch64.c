/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Deepti Aggarwal, Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include "generator_aarch64_instructions.h"
#include "generator_mateltwise_aarch64.h"
#include "generator_common_aarch64.h"
#include "generator_common.h"
#include "generator_mateltwise_unary_binary_aarch64.h"
#include "libxsmm_main.h"

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
  unsigned int is_inp_gp_reg = ((i_gp_reg == i_gp_reg_mapping->gp_reg_in) || ((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) )) ? 1 : 0;
  unsigned int is_out_gp_reg = (i_gp_reg == i_gp_reg_mapping->gp_reg_out) ? 1 : 0;
  unsigned int bcast_row = (((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1) > 0))) ? 1 : 0;
  unsigned int bcast_col = (((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1) > 0))) ? 1 : 0;
  unsigned int bcast_scalar = (((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR) > 0)) ||
                               ((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0) > 0)) ||
                               ((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1) > 0))) ? 1 : 0;
  unsigned int bcast_input = ( bcast_row == 1 || bcast_col == 1 || bcast_scalar == 1 ) ? 1 : 0;

  if ((is_inp_gp_reg > 0) || (is_out_gp_reg > 0)) {
    unsigned int tsize  = (is_inp_gp_reg > 0) ? i_micro_kernel_config->datatype_size_in : i_micro_kernel_config->datatype_size_out;
    unsigned int ld     = (is_inp_gp_reg > 0) ? (((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY)) ? i_mateltwise_desc->ldi2 : i_mateltwise_desc->ldi) : i_mateltwise_desc->ldo;

    if (bcast_input == 0) {
      if (i_loop_type == LOOP_TYPE_M) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, m_microkernel * tsize );
      } else {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, ld * n_microkernel * tsize);
      }
    } else {
      if (bcast_row > 0) {
        if (i_loop_type == LOOP_TYPE_N) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, ld * n_microkernel * tsize);
        }
      }
      if (bcast_col > 0) {
        if (i_loop_type == LOOP_TYPE_M) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, m_microkernel * tsize);
        }
      }
    }
  } else {
    /* Advance relumasks if need be */
    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY ) {
      if ( ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) )
           && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0)) {
        if (i_loop_type == LOOP_TYPE_M) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, m_microkernel/8);
        } else {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (i_micro_kernel_config->ldo_mask * n_microkernel)/8);
        }
      }

      if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV)       ||
           (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
           (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV)           ) {
        if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && (i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, m_microkernel/8);
          } else {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (i_micro_kernel_config->ldi_mask * n_microkernel)/8);
          }
        } else {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, m_microkernel * i_micro_kernel_config->datatype_size_in);
          } else {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, i_mateltwise_desc->ldi * n_microkernel * i_micro_kernel_config->datatype_size_in );
          }
        }
      }

      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0)) {
        if (i_loop_type == LOOP_TYPE_M) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, m_microkernel/8);
        } else {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (i_micro_kernel_config->ldo_mask * n_microkernel)/8);
        }
      }

      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, m_microkernel/8);
          } else {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (i_micro_kernel_config->ldi_mask * n_microkernel)/8);
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
  unsigned int is_inp_gp_reg = ((i_gp_reg == i_gp_reg_mapping->gp_reg_in) || ((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) )) ? 1 : 0;
  unsigned int is_out_gp_reg = (i_gp_reg == i_gp_reg_mapping->gp_reg_out) ? 1 : 0;
  unsigned int bcast_row = (((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1) > 0))) ? 1 : 0;
  unsigned int bcast_col = (((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0) > 0)) ||
                            ((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1) > 0))) ? 1 : 0;
  unsigned int bcast_scalar = (((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR) > 0)) ||
                               ((i_gp_reg == i_gp_reg_mapping->gp_reg_in) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0) > 0)) ||
                               ((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1) > 0))) ? 1 : 0;
  unsigned int bcast_input = ( bcast_row == 1 || bcast_col == 1 || bcast_scalar == 1 ) ? 1 : 0;

  if ((is_inp_gp_reg > 0) || (is_out_gp_reg > 0)) {
    unsigned int vlen   = (is_inp_gp_reg > 0) ? i_micro_kernel_config->vlen_in : i_micro_kernel_config->vlen_out;
    unsigned int tsize  = (is_inp_gp_reg > 0) ? i_micro_kernel_config->datatype_size_in : i_micro_kernel_config->datatype_size_out;
    unsigned int ld     = (is_inp_gp_reg > 0) ? (((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) ) ? i_mateltwise_desc->ldi2 : i_mateltwise_desc->ldi) : i_mateltwise_desc->ldo;

    if (bcast_input == 0) {
      if (i_loop_type == LOOP_TYPE_M) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, vlen * i_adjust_param * tsize  );
      } else {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, i_adjust_param * tsize * ld );
      }
    } else {
      if (bcast_row > 0) {
        if (i_loop_type == LOOP_TYPE_N) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, ld * i_adjust_param * tsize );
        }
      }
      if (bcast_col > 0) {
        if (i_loop_type == LOOP_TYPE_M) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, vlen * i_adjust_param * tsize );
        }
      }
    }
  } else {
    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY ) {
      if (((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU)) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0)) {
        /* @TODO Evangelos: why us here  i_gp_reg_mapping->gp_reg_relumask used? */
        if (i_loop_type == LOOP_TYPE_M) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg_mapping->gp_reg_relumask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_relumask, (i_micro_kernel_config->vlen_out * i_adjust_param)/8 );
        } else {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg_mapping->gp_reg_relumask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_relumask, (i_micro_kernel_config->ldo_mask * i_adjust_param)/8 );
        }
      }

      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV)) {
        if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && (i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (i_micro_kernel_config->vlen_in * i_adjust_param)/8);
          } else {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (i_micro_kernel_config->ldi_mask * i_adjust_param)/8);
          }
        } else {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, i_micro_kernel_config->vlen_in * i_adjust_param * i_micro_kernel_config->datatype_size_in);
          } else {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, i_mateltwise_desc->ldi * i_adjust_param * i_micro_kernel_config->datatype_size_in );
          }
        }
      }

      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0)) {
        /* @TODO Evangelos: copied from ReLU.... */
        if (i_loop_type == LOOP_TYPE_M) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg_mapping->gp_reg_dropoutmask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_dropoutmask, (i_micro_kernel_config->vlen_out * i_adjust_param)/8 );
        } else {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg_mapping->gp_reg_dropoutmask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_dropoutmask, (i_micro_kernel_config->ldo_mask * i_adjust_param)/8 );
        }
      }

      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (i_micro_kernel_config->vlen_in * i_adjust_param)/8);
          } else {
            libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (i_micro_kernel_config->ldi_mask * i_adjust_param)/8);
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
  /* First, determine the vlen compute based on the  add the architecture check in here D-, move it to the microkernel config */
  if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 )) {
    i_micro_kernel_config->vlen_comp = 16;
  } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ) ||
       LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ) ||
       LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 )) {
    i_micro_kernel_config->vlen_comp = 8;
  } else if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ) ||
       LIBXSMM_DATATYPE_I32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ))   {
    i_micro_kernel_config->vlen_comp = 4;
  } else if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ) ||
       LIBXSMM_DATATYPE_I64 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ))   {
    i_micro_kernel_config->vlen_comp = 2;
  }

  /* The vlen_in is aligned with the vlen compute */
  i_micro_kernel_config->vlen_in = i_micro_kernel_config->vlen_comp;

  /* The vlen_out depends on the output datatype */
  if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {
    i_micro_kernel_config->vlen_out = 16;
  } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ||
       LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ||
       LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {

    /* if the computation is done in F32 or the input is in F32, then set vlen_out to 16 */
    if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ) ||
        LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) )
    {
      i_micro_kernel_config->vlen_out= 4;
    } else {
      i_micro_kernel_config->vlen_out = 8;
    }
  } else if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ||
       LIBXSMM_DATATYPE_I32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ))   {
    i_micro_kernel_config->vlen_out = 4;
  } else if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ||
       LIBXSMM_DATATYPE_I64 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ))   {
    i_micro_kernel_config->vlen_out = 2;
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
  /* TODO: Make m chunk remainder depend on number of available zmm registers */
  unsigned int m_chunk_remainder = 4;
  unsigned int m_chunk_boundary = 32;
  unsigned int m_range, m_block_size, foo1, foo2;

  /* in order to work with bitmaks we need at least 8 entries, on ASIMD, that means 2 registers */
  if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && (vlen == 4) ) {
    vlen *= 2;
    if ( available_vregs < 3 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BITMASK_REQUIRED );
      return;
    }
    available_vregs /= 2;
    m_chunk_boundary /= 2;
  }

  m_chunks = (m+vlen-1)/vlen;
  m_chunks = (m_chunks/2)*2;

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
#if 0
  char vname = (i_vlen * i_micro_kernel_config->datatype_size_in == 64) ? 'z' : 'y';
#endif
  unsigned int bcast_row = (((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW) > 0)) ||
                            ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0) > 0))) ? 1 : 0;
  unsigned int bcast_col = (((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL) > 0)) ||
                            ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0) > 0))) ? 1 : 0;
  unsigned int bcast_scalar = (((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR) > 0)) ||
                               ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0) > 0))) ? 1 : 0;
  unsigned int bcast_input = ( bcast_row == 1 || bcast_col == 1 || bcast_scalar == 1 ) ? 1 : 0;
  unsigned int l_ld_bytes = i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in;
  unsigned int l_m_adjust = ( i_mask_last_m_chunk == 0 ) ? i_micro_kernel_config->datatype_size_in * i_vlen * i_m_blocking : i_micro_kernel_config->datatype_size_in * ( (i_vlen * (i_m_blocking-1)) + i_mask_last_m_chunk );
  unsigned int offset = 0;
  LIBXSMM_UNUSED(i_mask_reg);

  /* In this case we don't have to load any data  */
  if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_XOR)) return;
#if 0
  if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_PACK)) {
    for (in = 0; in < i_n_blocking; in++) {
      for (im = 0; im < i_m_blocking; im++) {
        cur_vreg = i_start_vreg + in * i_m_blocking + im;
        libxsmm_x86_instruction_vec_move( io_generated_code,
            io_generated_code->arch,
            LIBXSMM_X86_INSTR_VPMOVZXWD,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF,
            0,
            (im * i_vlen + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
            'z',
            cur_vreg,
            ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0,
            0, 0);
      }
    }
    return;
  }
#endif
  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      if ( bcast_input == 0) {
        offset = (l_ld_bytes*i_n_blocking);
        libxsmm_generator_vloadstore_masked_vreg_aarch64_asimd( io_generated_code, i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_1, cur_vreg,
                                                                i_micro_kernel_config->datatype_size_in, (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0, 1, 0 );
        /* If compute is in F32 and input is BF16 (or input is BF16 and output is F32), then upconvert BF16 -> FP32 */
#if 0
        if ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ) && LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype)) ||
             (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype  ) && LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype)) ) {
          libxsmm_generator_cvtbf16ps_avx512( io_generated_code, 'z', cur_vreg, cur_vreg );
        }
#endif
      } else {
        if ( (bcast_row == 1) || (bcast_scalar == 1) ) {
          offset = (bcast_scalar == 1) ?  i_micro_kernel_config->datatype_size_in:l_ld_bytes*i_n_blocking;
          if (im == 0) {
            if ((bcast_row == 1) || ((bcast_scalar == 1) && (in == 0))) {
              libxsmm_generator_bcastload_masked_vreg_aarch64_asimd( io_generated_code, i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_1, cur_vreg,
                                                                     i_micro_kernel_config->datatype_size_in, (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0, 1 );
            } else if ((bcast_scalar == 1) && (in > 0) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) ) {
              libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                                         i_start_vreg, i_start_vreg, 0, cur_vreg,
                                                         (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
            }
          }

           if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY)) {
            /* Copy the register to the rest of the "M-registers" in this case....  */
             if (im > 0) {
              libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                                         i_start_vreg + in * i_m_blocking, i_start_vreg + in * i_m_blocking, 0, cur_vreg,
                                                        (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );

            }
          }

        }

        if ( bcast_col == 1 ) {
          offset = ( i_mask_last_m_chunk == 0 ) ? i_micro_kernel_config->datatype_size_in * i_vlen * i_m_blocking : i_micro_kernel_config->datatype_size_in * ( (i_vlen * (i_m_blocking-1)) + i_mask_last_m_chunk );
          if (in == 0) {
            libxsmm_generator_vloadstore_masked_vreg_aarch64_asimd( io_generated_code, i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_1, cur_vreg,
                                                                    i_micro_kernel_config->datatype_size_in, (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0, 1, 0 );
          }

#if 0
            /* If compute is in F32 and input is BF16 (or input is BF16 and output is F32), then upconvert BF16 -> FP32 */
            if ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ) && LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype)) ||
                 (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype  ) && LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype)) ) {
              libxsmm_generator_cvtbf16ps_avx512( io_generated_code, 'z', cur_vreg, cur_vreg );
            }
          }
#endif
          if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY)) {
            /* Copy the register to the rest of the "N-REGISTERS" in this case....  */
            if (in > 0) {
              libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                                         i_start_vreg + im, i_start_vreg + im, 0, cur_vreg,
                                                        (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );

            }
          }
        }
      }
    }
    if ( bcast_input == 0 ) {
      if ( l_ld_bytes != l_m_adjust ) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                      i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in,
                                                      (unsigned long long)((unsigned long long)(l_ld_bytes-l_m_adjust)) );
      }
    } else {
      if (bcast_row == 1) {
        if ( l_ld_bytes != i_micro_kernel_config->datatype_size_in ) {
          libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                        i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in,
                                                        (unsigned long long)((unsigned long long)(l_ld_bytes-i_micro_kernel_config->datatype_size_in)) );
        }
      }
    }

  }
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                  i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in,
                                                  (unsigned long long)((unsigned long long)(offset)) );
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
  unsigned int im, in, cur_vreg;
#if 0
 cur_vreg_real;
#endif
#if 0
  char vname = (i_vlen * i_micro_kernel_config->datatype_size_out == 64) ? 'z' : 'y';
#endif
  unsigned int bcast_row = (((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW) > 0))) ? 1 : 0;
  unsigned int bcast_col = (((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL) > 0))) ? 1 : 0;
  unsigned int bcast_scalar = (((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR) > 0))) ? 1 : 0;
  unsigned int l_ld_bytes = i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out;
  unsigned int l_m_adjust = ( i_mask_last_m_chunk == 0 ) ? i_micro_kernel_config->datatype_size_in * i_vlen * i_m_blocking : i_micro_kernel_config->datatype_size_in * ( (i_vlen * (i_m_blocking-1)) + i_mask_last_m_chunk );
  LIBXSMM_UNUSED(i_mask_reg);
#if 0
  if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_UNPACK_TO_BLOCKS)) {
    for (in = 0; in < i_n_blocking; in++) {
      for (im = 0; im < i_m_blocking; im++) {
        cur_vreg = i_start_vreg + in * i_m_blocking + im;
        libxsmm_x86_instruction_vec_move( io_generated_code,
            io_generated_code->arch,
            LIBXSMM_X86_INSTR_VPMOVDW,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF,
            0,
            (im * i_vlen + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
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
            (im * i_vlen + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
            'z',
            cur_vreg,
            ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0,
            0, 1);
      }
    }
  } else if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_PACK)) {
    for (in = 0; in < i_n_blocking; in++) {
      for (im = 0; im < i_m_blocking; im++) {
        cur_vreg = i_start_vreg + in * i_m_blocking + im;
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
            i_micro_kernel_config->vmove_instruction_out,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * i_vlen + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
            'z',
            cur_vreg, ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? 1 : 0, ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0, 1 );
      }
    }
  }  else {
#endif
    for (in = 0; in < i_n_blocking; in++) {
      for (im = 0; im < i_m_blocking; im++) {
        cur_vreg = i_start_vreg + in * i_m_blocking + im;
#if 0
        cur_vreg_real = i_start_vreg + in * i_m_blocking + im;
#endif
        /* In the XOR case we have a constnt vreg  */
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
#if 0
        /* If compute is in F32 and output is BF16 (or input is F32 and output is BF16), then downconvert BF16 -> FP32 */
        if ( (cur_vreg == cur_vreg_real) &&
           ((LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) &&
             LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) ) {
          if ( io_generated_code->arch >= LIBXSMM_X86_AVX512_CPX ) {
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VCVTNEPS2BF16,
                                                      i_micro_kernel_config->vector_name,
                                                      cur_vreg, cur_vreg );
          } else {
            libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_micro_kernel_config->vector_name,
                                                                 cur_vreg, cur_vreg,
                                                                 i_micro_kernel_config->dcvt_zmm_aux0, i_micro_kernel_config->dcvt_zmm_aux1, i_micro_kernel_config->dcvt_mask_aux0, i_micro_kernel_config->dcvt_mask_aux1);
          }
        }
#endif
        libxsmm_generator_vloadstore_masked_vreg_aarch64_asimd( io_generated_code, i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_1, cur_vreg,
                                                                i_micro_kernel_config->datatype_size_out, (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0, 1, 1 );
      }
      if ( l_ld_bytes != l_m_adjust ) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out,
                                                       (unsigned long long)((unsigned long long)(l_ld_bytes-l_m_adjust)) );
      }
    }
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out,
                                                   (unsigned long long)((unsigned long long)l_ld_bytes*i_n_blocking) );
#if 0
  }
#endif
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
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                                   cur_vreg, cur_vreg, 0, cur_vreg,
                                                   (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_NEGATE) {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FNEG_V,
                                                   cur_vreg, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, cur_vreg,
                                                   (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_INC) {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V,
                                                   cur_vreg, i_micro_kernel_config->vec_ones, 0, cur_vreg,
                                                   (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL) {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FRECPE_V,
                                                    cur_vreg, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, i_micro_kernel_config->vec_tmp0,
                                                    (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FRECPS_V,
                                                    cur_vreg, i_micro_kernel_config->vec_tmp0, 0, cur_vreg,
                                                    (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );

          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                                    cur_vreg, i_micro_kernel_config->vec_tmp0, 0, cur_vreg,
                                                    (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT) {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FRSQRTE_V,
                                                    cur_vreg, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, cur_vreg,
                                                    (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_SQRT) {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FSQRT_V,
                                                    cur_vreg, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, cur_vreg,
                                                    (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_EXP) {
        libxsmm_generator_exp_ps_3dts_aarch64( io_generated_code,
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
            (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D);

      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TANH || i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TANH_INV ) {
        libxsmm_generator_tanh_ps_rational_78_aarch64( io_generated_code,
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
            (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D);

        if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TANH_INV) {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                                     cur_vreg, cur_vreg, 0, i_micro_kernel_config->vec_tmp0,
                                                     (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D);
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FNEG_V,
                                                     i_micro_kernel_config->vec_tmp0, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0,  i_micro_kernel_config->vec_tmp0,
                                                     (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D);
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FSUB_V,
                                                     i_micro_kernel_config->vec_tmp0, i_micro_kernel_config->vec_neg_ones, 0, cur_vreg,
                                                    (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D);
        }
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID || i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV) {

        libxsmm_generator_sigmoid_ps_rational_78_aarch64( io_generated_code,
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
          (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D);

        if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV) {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FSUB_V,
                                                     i_micro_kernel_config->vec_ones, cur_vreg, 0, i_micro_kernel_config->vec_x2,
                                                    (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D);
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V,
                                                     i_micro_kernel_config->vec_x2, cur_vreg, 0, cur_vreg,
                                                    (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D);
        }
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GELU) {

          libxsmm_generator_gelu_ps_minimax3_aarch64( io_generated_code,
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
              (i_micro_kernel_config->datatype_size_in == 4) ?  LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );

      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV) {
          libxsmm_generator_gelu_inv_ps_minimax3_aarch64( io_generated_code,
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
              (i_micro_kernel_config->datatype_size_in == 4) ?  LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
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
  LIBXSMM_UNUSED(i_mask_last_m_chunk);
  LIBXSMM_UNUSED(i_mask_reg);
  LIBXSMM_UNUSED(i_vlen);

  for (in = 0; in < i_n_blocking; in++) {
    unsigned int l_mask_adv = 0;
    for (im = 0; im < i_m_blocking; im++) {
      /*unsigned int l_vlen = ( l_bf16_compute > 0 ) ? 32 : 16;*/
      unsigned int cur_vreg = i_start_vreg + in * i_m_blocking + im;

      if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) ) {
        if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) ) {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FCMGT_Z_V, cur_vreg, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, i_micro_kernel_config->tmp_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
        }

        if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) {
          if ( im % 2 == 0 ) {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_AND_V, i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->mask_helper0_vreg, 0, i_micro_kernel_config->tmp_vreg2, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
          } else {
            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_AND_V, i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->mask_helper1_vreg, 0, i_micro_kernel_config->tmp_vreg2, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
          }
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ADDV_V, i_micro_kernel_config->tmp_vreg2, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, i_micro_kernel_config->tmp_vreg2, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
          if ( im % 2 == 0 ) {
            if ( im == i_m_blocking - 1 ) {
              libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                                    i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_AARCH64_GP_REG_UNDEF, 1, i_micro_kernel_config->tmp_vreg2, LIBXSMM_AARCH64_ASIMD_WIDTH_B );
              l_mask_adv++;
            } else {
              libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF,
                                                    i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_micro_kernel_config->tmp_vreg2, LIBXSMM_AARCH64_ASIMD_WIDTH_B );
            }
          } else {
            libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                                    i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_micro_kernel_config->tmp_vreg3, LIBXSMM_AARCH64_ASIMD_WIDTH_B );

            libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V, i_micro_kernel_config->tmp_vreg2, i_micro_kernel_config->tmp_vreg3, 0, i_micro_kernel_config->tmp_vreg2, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

            libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                                    i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_AARCH64_GP_REG_UNDEF, 1, i_micro_kernel_config->tmp_vreg2, LIBXSMM_AARCH64_ASIMD_WIDTH_B );
            l_mask_adv++;
          }
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
        /* Compute exp */
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V, cur_vreg, cur_vreg, 0, i_micro_kernel_config->tmp_vreg2, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
        libxsmm_generator_exp_ps_3dts_aarch64( io_generated_code,
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
            LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

        /* compute mask */
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FCMGT_Z_V, cur_vreg, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, i_micro_kernel_config->tmp_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

        /* compute ELU for < 0 */
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V, i_micro_kernel_config->tmp_vreg2, i_micro_kernel_config->fam_lu_vreg_alpha, 0, i_micro_kernel_config->tmp_vreg2, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FSUB_V, i_micro_kernel_config->tmp_vreg2, i_micro_kernel_config->fam_lu_vreg_alpha, 0, i_micro_kernel_config->tmp_vreg2, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );

        /* Blend exp-fma result with input reg based on elu mask */
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_BIF_V, i_micro_kernel_config->tmp_vreg2, i_micro_kernel_config->tmp_vreg, 0, cur_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
      } else {
        /* shouldn't happen */
      }
    }
    if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && (i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_ELU) ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, i_gp_reg_mapping->gp_reg_relumask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_relumask, (i_micro_kernel_config->ldo_mask - (l_mask_adv*8))/8 );
    }
  }
  if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && (i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_ELU) ) {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB, i_gp_reg_mapping->gp_reg_relumask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_relumask, i_n_blocking * (i_micro_kernel_config->ldo_mask/8) );
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
  LIBXSMM_UNUSED(i_mask_reg);

  for (in = 0; in < i_n_blocking; in++) {
    unsigned int l_mask_adv = 0;
    for (im = 0; im < i_m_blocking; im++) {
      unsigned int cur_vreg = i_start_vreg + in * i_m_blocking + im;

      if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && (i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
        if ( im % 2 == 0 ) {
          if ( im == i_m_blocking - 1 ) {
            libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R_R_POST,
                                                             i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_AARCH64_GP_REG_XZR, i_micro_kernel_config->tmp_vreg,
                                                             LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
            l_mask_adv++;
          } else {
            libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R,
                                                             i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->tmp_vreg,
                                                             LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
          }

          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_AND_V, i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->mask_helper0_vreg, 0, i_micro_kernel_config->tmp_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );

          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_CMEQ_R_V, i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->mask_helper0_vreg, 0, i_micro_kernel_config->tmp_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
        } else {
          libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R_R_POST,
                                                           i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_AARCH64_GP_REG_XZR, i_micro_kernel_config->tmp_vreg,
                                                           LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
          l_mask_adv++;

          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_AND_V, i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->mask_helper1_vreg, 0, i_micro_kernel_config->tmp_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );

          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_CMEQ_R_V, i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->mask_helper1_vreg, 0, i_micro_kernel_config->tmp_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
        }
      } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV ) {
        libxsmm_generator_vloadstore_masked_vreg_aarch64_asimd( io_generated_code, i_gp_reg_mapping->gp_reg_relumask, i_gp_reg_mapping->gp_reg_scratch_1, i_micro_kernel_config->tmp_vreg,
                                                                i_micro_kernel_config->datatype_size_in, (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0, 1, 0 );

        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FCMGT_Z_V, i_micro_kernel_config->tmp_vreg, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, i_micro_kernel_config->tmp_vreg2, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
      } else {
        /* shouldn't happen */
      }

      if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV ) {
        /* we need to multiply with alpha */
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V, cur_vreg, i_micro_kernel_config->fam_lu_vreg_alpha, 0, i_micro_kernel_config->tmp_vreg2, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

        /* now we blend both together */
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_BIF_V, i_micro_kernel_config->tmp_vreg2, i_micro_kernel_config->tmp_vreg, 0, cur_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
      } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV ) {
        /* now we blend both together */
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_BIF_V, i_micro_kernel_config->zero_vreg, i_micro_kernel_config->tmp_vreg, 0, cur_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
      } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV ) {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V, i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->fam_lu_vreg_alpha, 0, i_micro_kernel_config->tmp_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V, i_micro_kernel_config->tmp_vreg, cur_vreg, 0, i_micro_kernel_config->tmp_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_BIF_V, i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->tmp_vreg2, 0, cur_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
      } else {
        /* shouldn't happen */
      }
    }
    if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && ( i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_ELU_INV ) ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, i_gp_reg_mapping->gp_reg_relumask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_relumask, (i_micro_kernel_config->ldi_mask - (l_mask_adv*8))/8 );
    } else {
      if ( l_ld_bytes != l_m_adjust ) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_mapping->gp_reg_relumask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_relumask,
                                                       (unsigned long long)((unsigned long long)(l_ld_bytes-l_m_adjust)) );
      }
    }
  }
  if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && ( i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_ELU_INV ) ) {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB, i_gp_reg_mapping->gp_reg_relumask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_relumask, i_n_blocking * (i_micro_kernel_config->ldi_mask/8) );
  } else {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_relumask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_relumask,
                                                   (unsigned long long)((unsigned long long)l_ld_bytes*i_n_blocking) );
  }
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
  LIBXSMM_UNUSED(i_mask_last_m_chunk);
  LIBXSMM_UNUSED(i_mask_reg);
  LIBXSMM_UNUSED(i_vlen);

  for (in = 0; in < i_n_blocking; in++) {
    unsigned int l_mask_adv = 0;
    for (im = 0; im < i_m_blocking; im++) {
      /*unsigned int l_vlen = ( l_bf16_compute > 0 ) ? 32 : 16;*/
      unsigned int cur_vreg = i_start_vreg + in * i_m_blocking + im;

      /* draw a random number */
      libxsmm_generator_xoshiro128p_f32_aarch64_asimd( io_generated_code, i_micro_kernel_config->prng_state0_vreg, i_micro_kernel_config->prng_state1_vreg, i_micro_kernel_config->prng_state2_vreg, i_micro_kernel_config->prng_state3_vreg,
                                                       i_micro_kernel_config->dropout_vreg_tmp0, i_micro_kernel_config->dropout_vreg_tmp1, i_micro_kernel_config->dropout_vreg_one, i_micro_kernel_config->dropout_vreg_tmp2 );

      /* compare with p */
      libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FCMGT_R_V, i_micro_kernel_config->dropout_prob_vreg, i_micro_kernel_config->dropout_vreg_tmp2, 0, i_micro_kernel_config->dropout_vreg_tmp0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
      if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) {
        if ( im % 2 == 0 ) {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_AND_V, i_micro_kernel_config->dropout_vreg_tmp0, i_micro_kernel_config->mask_helper0_vreg, 0, i_micro_kernel_config->dropout_vreg_tmp1, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_AND_V, i_micro_kernel_config->dropout_vreg_tmp0, i_micro_kernel_config->mask_helper1_vreg, 0, i_micro_kernel_config->dropout_vreg_tmp1, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
        }
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ADDV_V, i_micro_kernel_config->dropout_vreg_tmp1, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, i_micro_kernel_config->dropout_vreg_tmp1, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
        if ( im % 2 == 0 ) {
          if ( im == i_m_blocking - 1 ) {
            libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                                    i_gp_reg_mapping->gp_reg_dropoutmask, LIBXSMM_AARCH64_GP_REG_UNDEF, 1, i_micro_kernel_config->dropout_vreg_tmp1, LIBXSMM_AARCH64_ASIMD_WIDTH_B );
            l_mask_adv++;
          } else {
            libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF,
                                                    i_gp_reg_mapping->gp_reg_dropoutmask, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_micro_kernel_config->dropout_vreg_tmp1, LIBXSMM_AARCH64_ASIMD_WIDTH_B );
          }
        } else {
          libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                                  i_gp_reg_mapping->gp_reg_dropoutmask, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_micro_kernel_config->dropout_vreg_tmp2, LIBXSMM_AARCH64_ASIMD_WIDTH_B );

          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V, i_micro_kernel_config->dropout_vreg_tmp1, i_micro_kernel_config->dropout_vreg_tmp2, 0, i_micro_kernel_config->dropout_vreg_tmp1, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

          libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST,
                                                    i_gp_reg_mapping->gp_reg_dropoutmask, LIBXSMM_AARCH64_GP_REG_UNDEF, 1, i_micro_kernel_config->dropout_vreg_tmp1, LIBXSMM_AARCH64_ASIMD_WIDTH_B );
          l_mask_adv++;
        }
      }

      /* weight */
      libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V, cur_vreg, i_micro_kernel_config->dropout_invprob_vreg, 0, cur_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

      /* blend zero and multiplication result together */
      libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V, i_micro_kernel_config->dropout_vreg_tmp1, i_micro_kernel_config->dropout_vreg_tmp1, 0, i_micro_kernel_config->dropout_vreg_tmp1, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
      libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_BIF_V, i_micro_kernel_config->dropout_vreg_tmp1, i_micro_kernel_config->dropout_vreg_tmp0, 0, cur_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
    }
    if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, i_gp_reg_mapping->gp_reg_dropoutmask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_dropoutmask, (i_micro_kernel_config->ldo_mask - (l_mask_adv*8))/8 );
    }
  }
  if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB, i_gp_reg_mapping->gp_reg_dropoutmask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_dropoutmask, i_n_blocking * (i_micro_kernel_config->ldo_mask/8) );
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
                                                             unsigned int                            i_mask_reg) {
  unsigned int im, in;
  LIBXSMM_UNUSED(i_mask_last_m_chunk);
  LIBXSMM_UNUSED(i_mask_reg);
  LIBXSMM_UNUSED(i_vlen);

  for (in = 0; in < i_n_blocking; in++) {
    unsigned int l_mask_adv = 0;
    for (im = 0; im < i_m_blocking; im++) {
      unsigned int cur_vreg = i_start_vreg + in * i_m_blocking + im;

      if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) ) {
        if ( im % 2 == 0 ) {
          if ( im == i_m_blocking - 1 ) {
            libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R_R_POST,
                                                             i_gp_reg_mapping->gp_reg_dropoutmask, LIBXSMM_AARCH64_GP_REG_XZR, i_micro_kernel_config->dropout_vreg_tmp0,
                                                             LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
            l_mask_adv++;
          } else {
            libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R,
                                                             i_gp_reg_mapping->gp_reg_dropoutmask, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->dropout_vreg_tmp0,
                                                             LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
          }

          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_AND_V, i_micro_kernel_config->dropout_vreg_tmp0, i_micro_kernel_config->mask_helper0_vreg, 0, i_micro_kernel_config->dropout_vreg_tmp0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );

          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_CMEQ_R_V, i_micro_kernel_config->dropout_vreg_tmp0, i_micro_kernel_config->mask_helper0_vreg, 0, i_micro_kernel_config->dropout_vreg_tmp0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
        } else {
          libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R_R_POST,
                                                           i_gp_reg_mapping->gp_reg_dropoutmask, LIBXSMM_AARCH64_GP_REG_XZR, i_micro_kernel_config->dropout_vreg_tmp0,
                                                           LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
          l_mask_adv++;

          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_AND_V, i_micro_kernel_config->dropout_vreg_tmp0, i_micro_kernel_config->mask_helper1_vreg, 0, i_micro_kernel_config->dropout_vreg_tmp0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );

          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_CMEQ_R_V, i_micro_kernel_config->dropout_vreg_tmp0, i_micro_kernel_config->mask_helper1_vreg, 0, i_micro_kernel_config->dropout_vreg_tmp0, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
        }
      } else {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BITMASK_REQUIRED );
        return;
      }

      /* weight */
      libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V, cur_vreg, i_micro_kernel_config->dropout_prob_vreg, 0, cur_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );

      /* select which value is set to 0 */
      libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_BIF_V, i_micro_kernel_config->dropout_vreg_zero, i_micro_kernel_config->dropout_vreg_tmp0, 0, cur_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
    }
    if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, i_gp_reg_mapping->gp_reg_dropoutmask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_dropoutmask, (i_micro_kernel_config->ldi_mask - (l_mask_adv*8))/8 );
    }
  }
  if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ) {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB, i_gp_reg_mapping->gp_reg_dropoutmask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_dropoutmask, i_n_blocking * (i_micro_kernel_config->ldi_mask/8) );
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
  unsigned int l_ld_bytes_in2 = i_mateltwise_desc->ldi2 * i_micro_kernel_config->datatype_size_in;
  unsigned int l_m_adjust = ( i_mask_last_m_chunk == 0 ) ? i_micro_kernel_config->datatype_size_out * i_vlen * i_m_blocking : i_micro_kernel_config->datatype_size_out * ( (i_vlen * (i_m_blocking-1)) + i_mask_last_m_chunk );
  unsigned int l_m_adjust_in2 = ( i_mask_last_m_chunk == 0 ) ? i_micro_kernel_config->datatype_size_in * i_vlen * i_m_blocking : i_micro_kernel_config->datatype_size_in * ( (i_vlen * (i_m_blocking-1)) + i_mask_last_m_chunk );
  unsigned int offset = 0, offset2 = 0;
  unsigned int _in_blocking = (bcast_col == 1) ? 1 : i_n_blocking;
  LIBXSMM_UNUSED(i_mask_reg);

  switch (i_mateltwise_desc->param) {
    case LIBXSMM_MELTW_TYPE_BINARY_ADD: {
      binary_op_instr = LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_MUL: {
      binary_op_instr = LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_SUB: {
      binary_op_instr = LIBXSMM_AARCH64_INSTR_ASIMD_FSUB_V;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_DIV: {
      binary_op_instr = LIBXSMM_AARCH64_INSTR_ASIMD_FDIV_V;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_MULADD: {
      binary_op_instr = LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V;
    } break;
    default:;
  }
#if 0
  if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_PACK) {
    for (in = 0; in < i_n_blocking; in++) {
      for (im = 0; im < i_m_blocking; im++) {
        cur_vreg = i_start_vreg + in * i_m_blocking + im;

        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST,
                                                i_gp_reg_mapping->gp_reg_in2, LIBXSMM_AARCH64_GP_REG_UNDEF, 16,
                                                cur_vreg,
                                                LIBXSMM_AARCH64_ASIMD_WIDTH_Q );


        libxsmm_x86_instruction_vec_move( io_generated_code,
            io_generated_code->arch,
            LIBXSMM_X86_INSTR_VPMOVZXWD,
            i_gp_reg_mapping->gp_reg_in2,
            LIBXSMM_X86_GP_REG_UNDEF,
            0,
            (im * i_vlen + in * i_mateltwise_desc->ldi2) * i_micro_kernel_config->datatype_size_in,
            'z',
            i_micro_kernel_config->tmp_vreg,
            ((i_mask_last_m_chunk == 1) && (im == i_m_blocking - 1)) ? i_mask_reg : 0,
            0, 0);

        libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, 'z', i_micro_kernel_config->tmp_vreg, i_micro_kernel_config->tmp_vreg, 16 );
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPORD, 'z', cur_vreg, i_micro_kernel_config->tmp_vreg, cur_vreg );
      }
    }
    return;
  }
#endif
  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      cur_vreg = i_start_vreg + in * i_m_blocking + im;

      if ( (bcast_row == 1) || (bcast_scalar == 1) ) {
        offset2 = (bcast_scalar == 1) ?  0:i_micro_kernel_config->datatype_size_in*i_mateltwise_desc->ldi2*i_n_blocking;
        offset = (l_ld_bytes*i_n_blocking);
        libxsmm_generator_bcastload_masked_vreg_aarch64_asimd( io_generated_code, i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_1, i_micro_kernel_config->tmp_vreg,
                                                               i_micro_kernel_config->datatype_size_in, (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0, 0 );

      }
      if ( bcast_input == 0 || bcast_col == 1) {
        offset = (l_ld_bytes*i_n_blocking);
        offset2 = (bcast_col == 1) ? l_m_adjust:(l_ld_bytes_in2*_in_blocking);
        libxsmm_generator_vloadstore_masked_vreg_aarch64_asimd( io_generated_code, i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_1, i_micro_kernel_config->tmp_vreg,
                                                                i_micro_kernel_config->datatype_size_in, (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0, 1, 0 );
      }

      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_MULADD) {
        libxsmm_generator_vloadstore_masked_vreg_aarch64_asimd( io_generated_code, i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_1, i_micro_kernel_config->tmp_vreg2,
                                                                i_micro_kernel_config->datatype_size_in, (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0, 1, 0 );

        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, binary_op_instr,
                                                  cur_vreg,i_micro_kernel_config->tmp_vreg, 0, i_micro_kernel_config->tmp_vreg2,
                                                  (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V,
                                                  i_micro_kernel_config->tmp_vreg2, i_micro_kernel_config->tmp_vreg2, 0, cur_vreg,
                                                  (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
      } else {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, binary_op_instr,
                                                  cur_vreg, i_micro_kernel_config->tmp_vreg, 0, cur_vreg,
                                                  (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
      }
    }
    if ( (l_ld_bytes != l_m_adjust) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_MULADD) ) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                              i_gp_reg_mapping->gp_reg_out,i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out,
                                              (unsigned long long)((unsigned long long)(l_ld_bytes-l_m_adjust)) );
    }
    if ( bcast_input == 0 ) {
      if ( l_ld_bytes_in2 != l_m_adjust ) {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                      i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in2,
                                                      (unsigned long long)((unsigned long long)(l_ld_bytes_in2-l_m_adjust_in2)) );
      }
    }
    if (bcast_col == 1 && in < (i_n_blocking-1)){
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                     i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in2,
                                                     (unsigned long long)((unsigned long long)(offset2)) );
    }

    if (bcast_row == 1) {
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                    i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in2,
                                                    (unsigned long long)((unsigned long long)i_mateltwise_desc->ldi2 * i_micro_kernel_config->datatype_size_in) );
    }
  }
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in2,
                                                 (unsigned long long)((unsigned long long)(offset2)) );
  if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_MULADD) {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out,
                                                   (unsigned long long)((unsigned long long)(offset)) );
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
  LIBXSMM_UNUSED(io_generated_code);
  LIBXSMM_UNUSED(i_micro_kernel_config);
  LIBXSMM_UNUSED(i_mateltwise_desc);
  LIBXSMM_UNUSED(i_tmp_reg);

  *i_mask_reg_in = 0;
  *i_use_m_input_masking = i_m % i_vlen_in;
  *i_mask_reg_out = 0;
  *i_use_m_output_masking = i_m % i_vlen_out;
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
  LIBXSMM_UNUSED(i_mateltwise_desc);
  LIBXSMM_UNUSED(io_generated_code);

  m_trips               = (i_m + i_vlen_in - 1) / i_vlen_in;
  n_trips               = i_n;

  max_nm_unrolling  = max_nm_unrolling - reserved_zmms;

  if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && (i_vlen_in == 4) ) {
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

  LIBXSMM_UNUSED(flags);
  LIBXSMM_UNUSED(i_gp_reg_aux0);
  LIBXSMM_UNUSED(i_gp_reg_aux1);

  if ((op == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (op == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV)       ||
      (op == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (op == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV)    ) {
    i_micro_kernel_config->zero_vreg = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->tmp_vreg  = i_micro_kernel_config->reserved_zmms + 1;
    i_micro_kernel_config->tmp_vreg2 = i_micro_kernel_config->reserved_zmms + 2;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 3;

    if ( (flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0 ){
      i_micro_kernel_config->tmp_vreg3 = i_micro_kernel_config->reserved_zmms;
      i_micro_kernel_config->mask_helper0_vreg =  i_micro_kernel_config->reserved_zmms + 1;
      i_micro_kernel_config->mask_helper1_vreg =  i_micro_kernel_config->reserved_zmms + 2;
      i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 3;

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

    if ( (op == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (op == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ) {
      i_micro_kernel_config->fam_lu_vreg_alpha = i_micro_kernel_config->reserved_zmms;
      i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;

      /* load alpha */
      libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R,
                                                       i_gp_reg_aux1, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->fam_lu_vreg_alpha,
                                                       LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
    }

    /* Set zero register needed for relu  */
    libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                               i_micro_kernel_config->zero_vreg, i_micro_kernel_config->zero_vreg, 0, i_micro_kernel_config->zero_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
  }

  if ((op == LIBXSMM_MELTW_TYPE_UNARY_ELU) || (op == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV)) {
    i_micro_kernel_config->tmp_vreg = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->tmp_vreg2 = i_micro_kernel_config->reserved_zmms + 1;
    i_micro_kernel_config->tmp_vreg3 = i_micro_kernel_config->reserved_zmms + 2;
    i_micro_kernel_config->fam_lu_vreg_alpha = i_micro_kernel_config->reserved_zmms + 3;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 4;

    /* load alpha */
    libxsmm_aarch64_instruction_asimd_struct_r_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R,
                                                     i_gp_reg_aux1, LIBXSMM_AARCH64_GP_REG_UNDEF, i_micro_kernel_config->fam_lu_vreg_alpha,
                                                     LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S );
  }

  if ((op == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (op == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
    i_micro_kernel_config->mask_helper0_vreg =  i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->mask_helper1_vreg =  i_micro_kernel_config->reserved_zmms + 1;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 2;

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
    i_micro_kernel_config->zero_vreg = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
    libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V, i_micro_kernel_config->zero_vreg, i_micro_kernel_config->zero_vreg, 0, i_micro_kernel_config->zero_vreg, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
  }

  if ((op == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL) || (op == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT)) {
    i_micro_kernel_config->vec_tmp0 = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
  }

  if (op == LIBXSMM_MELTW_TYPE_UNARY_INC) {
    i_micro_kernel_config->vec_ones = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;
    libxsmm_aarch64_instruction_broadcast_scalar_to_vec ( io_generated_code, i_micro_kernel_config->vec_ones, i_gp_reg_tmp0,
                                                          (i_micro_kernel_config->datatype_size_in == 4) ?  LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D,
                                                          0x3f800000 );
  }

  if (op == LIBXSMM_MELTW_TYPE_UNARY_EXP) {
    i_micro_kernel_config->vec_y = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->vec_z = i_micro_kernel_config->reserved_zmms + 1;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 2;
  }

  if (op == LIBXSMM_MELTW_TYPE_UNARY_GELU || op == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV) {
    unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;


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

    if (op == LIBXSMM_MELTW_TYPE_UNARY_GELU ) {
      libxsmm_generator_prepare_coeffs_gelu_ps_minimax3_aarch64( io_generated_code,
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
          (i_micro_kernel_config->datatype_size_in == 4) ?  LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
    }

    if (op == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV ) {
      libxsmm_generator_prepare_coeffs_gelu_inv_ps_minimax3_aarch64( io_generated_code,
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
          (i_micro_kernel_config->datatype_size_in == 4) ?  LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
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

    libxsmm_generator_prepare_coeffs_exp_ps_3dts_aarch64( io_generated_code,
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
        (i_micro_kernel_config->datatype_size_in == 4) ?  LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
    i_micro_kernel_config->reserved_zmms = reserved_zmms;
  }
  if (op == LIBXSMM_MELTW_TYPE_UNARY_TANH || op == LIBXSMM_MELTW_TYPE_UNARY_TANH_INV) {
    unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;
    unsigned int reserved_mask_regs = i_micro_kernel_config->reserved_mask_regs;
    reserved_zmms += 17;

    i_micro_kernel_config->vec_x2        = reserved_zmms - 1;
    i_micro_kernel_config->mask_hi       = reserved_zmms - 2;
    i_micro_kernel_config->mask_lo       = reserved_zmms - 3;
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

    libxsmm_generator_prepare_coeffs_tanh_ps_rational_78_aarch64( io_generated_code,
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
        (i_micro_kernel_config->datatype_size_in == 4) ?  LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );

    i_micro_kernel_config->reserved_mask_regs = reserved_mask_regs;
    i_micro_kernel_config->reserved_zmms = reserved_zmms;
  }

  if (op == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID || op == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV) {
    unsigned int reserved_zmms = i_micro_kernel_config->reserved_zmms;
    unsigned int reserved_mask_regs = i_micro_kernel_config->reserved_mask_regs;
    reserved_zmms += 18;

    i_micro_kernel_config->vec_x2        = reserved_zmms - 1;
    i_micro_kernel_config->mask_hi       = reserved_zmms - 2;
    i_micro_kernel_config->mask_lo       = reserved_zmms - 3;
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



    libxsmm_generator_prepare_coeffs_sigmoid_ps_rational_78_aarch64( io_generated_code,
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
      (i_micro_kernel_config->datatype_size_in == 4) ?  LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );

    i_micro_kernel_config->reserved_mask_regs = reserved_mask_regs;
    i_micro_kernel_config->reserved_zmms = reserved_zmms;
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
  i_micro_kernel_config->reserved_mask_regs = 1;
  i_micro_kernel_config->use_fp32bf16_cvt_replacement = 0;

  /* if we need FP32->BF16 downconverts and we don't have native instruction, then prepare stack */
#if 0
  if ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype2 ) || LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) &&
       LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) && (io_generated_code->arch < LIBXSMM_X86_AVX512_CPX)) {
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

  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
    /* This is the temp register used to load the second input */
    i_micro_kernel_config->tmp_vreg = i_micro_kernel_config->reserved_zmms;
    i_micro_kernel_config->reserved_zmms = i_micro_kernel_config->reserved_zmms + 1;

    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_MULADD) {
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


  /* Footers of microkernel loops  */
  if (inner_loop_trips > 1) {
    /* Advance input/output pointers */
    loop_type = (inner_loop_reg == i_gp_reg_mapping->gp_reg_m_loop) ? LOOP_TYPE_M : LOOP_TYPE_N;

    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_in, LIBXSMM_AARCH64_INSTR_GP_META_ADD, inner_unroll_factor, loop_type);

    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_out, LIBXSMM_AARCH64_INSTR_GP_META_ADD, inner_unroll_factor, loop_type);

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in2, LIBXSMM_AARCH64_INSTR_GP_META_ADD, inner_unroll_factor, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
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

    /* Reset input/output pointers  */
    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_in, LIBXSMM_AARCH64_INSTR_GP_META_SUB, inner_unroll_factor * inner_loop_trips, loop_type);

    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_out, LIBXSMM_AARCH64_INSTR_GP_META_SUB, inner_unroll_factor * inner_loop_trips, loop_type);

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in2, LIBXSMM_AARCH64_INSTR_GP_META_SUB, inner_unroll_factor * inner_loop_trips, loop_type);
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

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in2, LIBXSMM_AARCH64_INSTR_GP_META_ADD, out_unroll_factor, loop_type);
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

    /* Reset input/output pointers  */
    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_in, LIBXSMM_AARCH64_INSTR_GP_META_SUB, out_unroll_factor * out_loop_trips, loop_type);

    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_out, LIBXSMM_AARCH64_INSTR_GP_META_SUB, out_unroll_factor * out_loop_trips, loop_type);

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in2, LIBXSMM_AARCH64_INSTR_GP_META_SUB, out_unroll_factor * out_loop_trips, loop_type);
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
  if ( ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) )
       &&
       ( LIBXSMM_DATATYPE_F32  == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) ) {
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

  /* let's check that we have bitmask set for dropput and relu backward */
  if ( ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
         (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV) ) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) == 0) ) {
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
  } else {
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
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_UNPACK_TO_BLOCKS) {
      i_gp_reg_mapping->gp_reg_offset = LIBXSMM_AARCH64_GP_REG_X12;
    } else {
      i_gp_reg_mapping->gp_reg_offset = LIBXSMM_AARCH64_GP_REG_UNDEF;
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
  }

  /* load the input pointer and output pointer */
  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY ) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 32, i_gp_reg_mapping->gp_reg_in );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 64, i_gp_reg_mapping->gp_reg_out );
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) {
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
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_UNPACK_TO_BLOCKS) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 72, i_gp_reg_mapping->gp_reg_offset );
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
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 40, i_gp_reg_mapping->gp_reg_quant_sf );
      l_gp_reg_aux0 = i_gp_reg_mapping->gp_reg_quant_sf;
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DEQUANT) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                            LIBXSMM_AARCH64_GP_REG_UNDEF, 40, i_gp_reg_mapping->gp_reg_quant_sf );
      l_gp_reg_aux0 = i_gp_reg_mapping->gp_reg_quant_sf;
    }
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY ) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 32, i_gp_reg_mapping->gp_reg_in );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 64, i_gp_reg_mapping->gp_reg_in2 );
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct,
                                          LIBXSMM_AARCH64_GP_REG_UNDEF, 96, i_gp_reg_mapping->gp_reg_out );
  } else {
    /* This should not happen */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }

  /* Based on kernel type reserve zmms and mask registers  */
  libxsmm_configure_aarch64_kernel_vregs_masks( io_generated_code, i_micro_kernel_config, i_mateltwise_desc, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_scratch_1, l_gp_reg_aux0, l_gp_reg_aux1 );

  available_vregs = available_vregs - i_micro_kernel_config->reserved_zmms;

  /* Configure M and N blocking factors */
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

        if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
          libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
              i_gp_reg_mapping->gp_reg_in2, LIBXSMM_AARCH64_INSTR_GP_META_ADD, m_microkernel, n_microkernel, loop_type );
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

      if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
        libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_in2, LIBXSMM_AARCH64_INSTR_GP_META_SUB, m_microkernel, n_microkernel, loop_type );
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

      if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
        libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_aarch64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_in2, LIBXSMM_AARCH64_INSTR_GP_META_ADD, m_microkernel, n_microkernel, loop_type );
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
