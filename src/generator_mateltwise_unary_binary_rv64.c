/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Siddharth Rai, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_rv64_instructions.h"
#include "generator_mateltwise_rv64.h"
#include "generator_common_rv64.h"
#include "generator_common.h"
#include "generator_mateltwise_unary_binary_rv64.h"

#define MN_LOOP_ORDER 0
#define NM_LOOP_ORDER 1
#define LOOP_TYPE_M   0
#define LOOP_TYPE_N   1

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_rv64_gp_reg( libxsmm_generated_code*                 io_generated_code,
                                                   libxsmm_mateltwise_gp_reg_mapping*   i_gp_reg_mapping,
                                                   libxsmm_mateltwise_kernel_config*    i_micro_kernel_config,
                                                   const libxsmm_meltw_descriptor*      i_mateltwise_desc,
                                                   unsigned int                         i_gp_reg,
                                                   unsigned int                         i_adjust_instr,
                                                   unsigned int                         m_microkernel,
                                                   unsigned int                         n_microkernel,
                                                   unsigned int                         i_loop_type ) {
  unsigned int is_inp_gp_reg = ((i_gp_reg == i_gp_reg_mapping->gp_reg_in) || ((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) )) ? 1 : 0;
  unsigned int is_out_gp_reg = (i_gp_reg == i_gp_reg_mapping->gp_reg_out || ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP) && (i_gp_reg == i_gp_reg_mapping->gp_reg_out2))) ? 1 : 0;
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
    unsigned int tsize  = (is_inp_gp_reg > 0) ? (((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY)) ? i_micro_kernel_config->datatype_size_in1 : i_micro_kernel_config->datatype_size_in) : i_micro_kernel_config->datatype_size_out;
    unsigned int ld     = (is_inp_gp_reg > 0) ? (((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY)) ? i_mateltwise_desc->ldi2 : i_mateltwise_desc->ldi) : i_mateltwise_desc->ldo;

    if (bcast_input == 0) {
      if (i_loop_type == LOOP_TYPE_M) {
        libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)m_microkernel * tsize );
      } else {
        libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)ld * n_microkernel * tsize);
      }
    }
  } else {
    /* Advance relumasks if need be */
    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY ) {
      if ( ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) )
           && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0)) {
        if (i_loop_type == LOOP_TYPE_M) {
          libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)m_microkernel/8);
        } else {
          libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, ((long long)i_micro_kernel_config->ldo_mask * n_microkernel)/8);
        }
      }

      if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV)       ||
           (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
           (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV)           ) {
        if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && (i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)m_microkernel/8);
          } else {
            libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, ((long long)i_micro_kernel_config->ldi_mask * n_microkernel)/8);
          }
        } else {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)m_microkernel * i_micro_kernel_config->datatype_size_in);
          } else {
            libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)i_mateltwise_desc->ldi * n_microkernel * i_micro_kernel_config->datatype_size_in );
          }
        }
      }

      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0)) {
        if (i_loop_type == LOOP_TYPE_M) {
          libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)m_microkernel/8);
        } else {
          libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, ((long long)i_micro_kernel_config->ldo_mask * n_microkernel)/8);
        }
      }

      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)m_microkernel/8);
          } else {
            libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, ((long long)i_micro_kernel_config->ldi_mask * n_microkernel)/8);
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
void libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_rv64_gp_reg( libxsmm_generated_code*                 io_generated_code,
                                                libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                unsigned int                            i_gp_reg,
                                                unsigned int                            i_adjust_instr,
                                                unsigned int                            i_adjust_param,
                                                unsigned int                            i_loop_type ) {
  unsigned int is_inp_gp_reg = ((i_gp_reg == i_gp_reg_mapping->gp_reg_in) || ((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) )) ? 1 : 0;
  unsigned int is_out_gp_reg = (i_gp_reg == i_gp_reg_mapping->gp_reg_out || ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP) && (i_gp_reg == i_gp_reg_mapping->gp_reg_out2))) ? 1 : 0;
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
    unsigned int tsize  = (is_inp_gp_reg > 0) ? (((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) ) ? i_micro_kernel_config->datatype_size_in1 : i_micro_kernel_config->datatype_size_in) : i_micro_kernel_config->datatype_size_out;
    unsigned int ld     = (is_inp_gp_reg > 0) ? (((i_gp_reg == i_gp_reg_mapping->gp_reg_in2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) ) ? i_mateltwise_desc->ldi2 : i_mateltwise_desc->ldi) : i_mateltwise_desc->ldo;

    if (bcast_input == 0) {
      if (i_loop_type == LOOP_TYPE_M) {
        libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)vlen * i_adjust_param * tsize );
      } else {
        libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)i_adjust_param * tsize * ld );
      }
    } else {
      if (bcast_row > 0) {
        if (i_loop_type == LOOP_TYPE_N) {
        libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)ld * i_adjust_param * tsize );
        }
      }
      if (bcast_col > 0) {
        if (i_loop_type == LOOP_TYPE_M) {
        libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)vlen * i_adjust_param * tsize );
        }
      }
    }
  } else {
    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY ) {
      if (((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU)) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0)) {
        /* TODO: Evangelos: why is here i_gp_reg_mapping->gp_reg_relumask used? */
        if (i_loop_type == LOOP_TYPE_M) {
          libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg_mapping->gp_reg_relumask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_relumask, ((long long)i_micro_kernel_config->vlen_out * i_adjust_param)/8 );
        } else {
          libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg_mapping->gp_reg_relumask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_relumask, ((long long)i_micro_kernel_config->ldo_mask * i_adjust_param)/8 );
        }
      }

      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV)) {
        if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && (i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, ((long long)i_micro_kernel_config->vlen_in * i_adjust_param)/8);
          } else {
            libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, ((long long)i_micro_kernel_config->ldi_mask * i_adjust_param)/8);
          }
        } else {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)i_micro_kernel_config->vlen_in * i_adjust_param * i_micro_kernel_config->datatype_size_in);
          } else {
            libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, (long long)i_mateltwise_desc->ldi * i_adjust_param * i_micro_kernel_config->datatype_size_in );
          }
        }
      }

      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0)) {
        /* TODO: Evangelos: copied from ReLU.... */
        if (i_loop_type == LOOP_TYPE_M) {
          libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg_mapping->gp_reg_dropoutmask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_dropoutmask, ((long long)i_micro_kernel_config->vlen_out * i_adjust_param)/8 );
        } else {
          libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg_mapping->gp_reg_dropoutmask, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_dropoutmask, ((long long)i_micro_kernel_config->ldo_mask * i_adjust_param)/8 );
        }
      }

      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) {
          if (i_loop_type == LOOP_TYPE_M) {
            libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, ((long long)i_micro_kernel_config->vlen_in * i_adjust_param)/8);
          } else {
            libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, i_adjust_instr, i_gp_reg, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg, ((long long)i_micro_kernel_config->ldi_mask * i_adjust_param)/8);
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
void libxsmm_generator_configure_rv64_vlens(const libxsmm_meltw_descriptor* i_mateltwise_desc, libxsmm_mateltwise_kernel_config* i_micro_kernel_config) {
  /* First, determine the vlen compute based on the architecture; there may be architectures with different widths for different types */
  /* At the moment, all types are assumed to be of the same length */
  unsigned int l_rvv_bytes_per_register = libxsmm_cpuid_vlen(i_micro_kernel_config->instruction_set);

  unsigned char l_inp_type = (LIBXSMM_DATATYPE_IMPLICIT == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)) ? LIBXSMM_CAST_UCHAR(libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) : LIBXSMM_CAST_UCHAR(libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP));
  unsigned int  l_inp_type_size = LIBXSMM_TYPESIZE(l_inp_type); /* like libxsmm_typesize; returns 0 if type is unknown */
  /* The vlen_out depends on the output datatype */
  unsigned char l_out_type = LIBXSMM_CAST_UCHAR(libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT));
  unsigned int  l_out_type_size = LIBXSMM_TYPESIZE(l_out_type);
  unsigned int  l_is_comp_zip = ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_ZIP)) ? 1 : 0;

  if (l_inp_type_size > 0) i_micro_kernel_config->vlen_comp = l_rvv_bytes_per_register / l_inp_type_size;
  if (l_out_type_size > 0) i_micro_kernel_config->vlen_out = l_rvv_bytes_per_register / l_out_type_size;

  if ( l_is_comp_zip ) {
    i_micro_kernel_config->vlen_comp = l_rvv_bytes_per_register / l_out_type_size;
  }

  /* The vlen_in is the same as vlen compute */
  i_micro_kernel_config->vlen_in = i_micro_kernel_config->vlen_comp;

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
void libxsmm_generator_configure_rv64_M_N_blocking( libxsmm_generated_code*         io_generated_code,
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

  /* in order to work with bitmasks we need at least 8 entries, on ASIMD, that means 2 registers */
  /* TODO: for SVE, we maybe could use the predicate registers, so we do not need to half the count; however, we only have 15 predicate registers, so this still may be correct*/
  if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && (vlen == 4) ) {
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
      if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && (vlen == 4) ) {
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
        if ( ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) && (vlen == 4) ) {
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
void libxsmm_generator_configure_rv64_loop_order(const libxsmm_meltw_descriptor* i_mateltwise_desc, unsigned int *loop_order, unsigned int *m_blocking, unsigned int *n_blocking, unsigned int *out_blocking, unsigned int *inner_blocking, unsigned int *out_bound, unsigned int *inner_bound) {
  unsigned int _loop_order = NM_LOOP_ORDER;

  /* TODO: Potentially reorder loops given the kernel type */
  *loop_order = _loop_order;

  if (_loop_order == NM_LOOP_ORDER) {
    *out_blocking = *n_blocking;
    *out_bound = i_mateltwise_desc->n;
    *inner_blocking = *m_blocking;
    *inner_bound = i_mateltwise_desc->m;
  } else {
    *out_blocking = *m_blocking;
    *out_bound = i_mateltwise_desc->m;
    *inner_blocking = *n_blocking;
    *inner_bound = i_mateltwise_desc->n;
  }
}

LIBXSMM_API_INTERN
void libxsmm_load_rv64_2d_reg_block( libxsmm_generated_code*                 io_generated_code,
                                     libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                     const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                     const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                     unsigned int                            i_vlen,
                                     unsigned int                            i_avlen,
                                     unsigned int                            i_start_vreg,
                                     unsigned int                            i_m_blocking,
                                     unsigned int                            i_n_blocking,
                                     unsigned int                            i_mask_last_m_chunk,
                                     unsigned int                            i_mask_reg) {
  unsigned int im, in, cur_vreg;
  unsigned int l_load_instr = (i_micro_kernel_config->datatype_size_in == 8) ? LIBXSMM_RV64_INSTR_RVV_VLE64_V : LIBXSMM_RV64_INSTR_RVV_VLE32_V;
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
  unsigned char l_is_sve = 0;
  unsigned int l_is_comp_unzip = ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_UNZIP)) ? 1 : 0;
  unsigned int l_is_comp_zip = ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_ZIP)) ? 1 : 0;
  unsigned int l_sew = (i_micro_kernel_config->datatype_size_in == 8) ? LIBXSMM_RV64_SEW_Q : (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_RV64_SEW_D : (i_micro_kernel_config->datatype_size_in == 2) ? LIBXSMM_RV64_SEW_W : LIBXSMM_RV64_SEW_B;

 LIBXSMM_UNUSED(l_m_adjust);
 LIBXSMM_UNUSED(l_is_sve);
 LIBXSMM_UNUSED(l_is_comp_unzip);
 LIBXSMM_UNUSED(l_is_comp_zip);

  /* In this case we do not have to load any data */
  if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_XOR)) return;

  if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_ZIP)) {
      offset = (l_ld_bytes*i_n_blocking);

      for (in = 0; in < i_n_blocking; in++) {
        for (im = 0; im < i_m_blocking; im++) {
          unsigned int l_is_f32_or_f64 = (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) || LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ? 1 : 0;
          unsigned int l_masked_elements = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0 : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_mask_last_m_chunk : i_vlen : i_vlen;
 LIBXSMM_UNUSED(l_masked_elements);
          cur_vreg = i_start_vreg + in * i_m_blocking + im;
      }

      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI,
                                                  i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in, offset );
    }
    return;
  }

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      cur_vreg = i_start_vreg + in * i_m_blocking + im;

      if (bcast_input == 0){
        offset   = (l_ld_bytes*i_n_blocking);
        if ((im == (i_m_blocking - 1)) && i_mask_last_m_chunk != 0)
          libxsmm_rv64_instruction_rvv_setvli( io_generated_code, i_avlen, i_gp_reg_mapping->gp_reg_scratch_0, l_sew, LIBXSMM_RV64_LMUL_M1);

        libxsmm_rv64_instruction_rvv_move( io_generated_code, l_load_instr, i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, cur_vreg, 1);

        /*  Increament in address */
        if ((im == (i_m_blocking - 1)) && i_mask_last_m_chunk != 0){
          libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
              i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in,
              (unsigned long)i_mask_last_m_chunk * i_micro_kernel_config->datatype_size_in);
        } else {
          libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
              i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in,
              (unsigned long)i_vlen * i_micro_kernel_config->datatype_size_in);
        }
      } else {
        if ( (bcast_row == 1) || (bcast_scalar == 1) ) {
          offset = (bcast_scalar == 1) ?  i_micro_kernel_config->datatype_size_in:l_ld_bytes*i_n_blocking;
          if (im == 0) {
            if ((bcast_row == 1) || ((bcast_scalar == 1) && (in == 0))) {
              /* Masked load and broadcast */
              libxsmm_generator_bcastload_masked_vreg_rv64( io_generated_code, i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_1, cur_vreg,
                  i_micro_kernel_config->datatype_size_in, (im == i_m_blocking - 1) ? 1 : 0, i_vlen, i_avlen, (bcast_scalar == 1) ? 1 : 0 );

            } else if ((bcast_scalar == 1) && (in > 0) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) ) {
              libxsmm_rv64_instruction_rvv_compute( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VOR_VV, i_start_vreg, i_start_vreg, cur_vreg, 1);
            }
          }

          if (bcast_row == 1){
            /*  Increament in address */
            if ((im == (i_m_blocking - 1)) && i_mask_last_m_chunk != 0){
              libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                  i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in,
                  (unsigned long )i_mask_last_m_chunk * i_micro_kernel_config->datatype_size_in);
            } else {
              libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                  i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in,
                  (unsigned long)i_vlen * i_micro_kernel_config->datatype_size_in);
            }
          }

          if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY)) {
            /* Copy the register to the rest of the "M-registers" in this case.... */
            if (im > 0) {
                libxsmm_rv64_instruction_rvv_compute( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VOR_VV,
                    i_start_vreg + in * i_m_blocking, i_start_vreg + in * i_m_blocking, cur_vreg, 1);
            }
          }
        }

        if ( bcast_col == 1 ) {
          offset = ( i_mask_last_m_chunk == 0 ) ? i_micro_kernel_config->datatype_size_in * i_vlen * i_m_blocking : i_micro_kernel_config->datatype_size_in * ( (i_vlen * (i_m_blocking-1)) + i_mask_last_m_chunk );
          if (in == 0) {
            if ((im == (i_m_blocking - 1)) && i_mask_last_m_chunk != 0)
              libxsmm_rv64_instruction_rvv_setvli( io_generated_code, i_avlen, i_gp_reg_mapping->gp_reg_scratch_0, l_sew, LIBXSMM_RV64_LMUL_M1);

            libxsmm_rv64_instruction_rvv_move( io_generated_code, l_load_instr, i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, cur_vreg, 1);

            /*  Increament in address */
            if ((im == (i_m_blocking - 1)) && i_mask_last_m_chunk != 0){
              libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                  i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in,
                  (unsigned long)i_mask_last_m_chunk * i_micro_kernel_config->datatype_size_in);
            } else {
              libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                  i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in,
                  (unsigned long)i_vlen * i_micro_kernel_config->datatype_size_in);
            }
          }

          if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY)) {
            /* Copy the register to the rest of the "N-REGISTERS" in this case.... */
            if (in > 0) {
                libxsmm_rv64_instruction_rvv_compute( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VOR_VV,
                    i_start_vreg + im, i_start_vreg + im, cur_vreg, 1 );
            }
          }
        }
      }
    }

    if ( bcast_input == 0 ) {
      if (l_m_adjust != l_ld_bytes)
        libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
            i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0,
            i_gp_reg_mapping->gp_reg_in, ((long long)l_ld_bytes - l_m_adjust));
    }
    else if (bcast_row == 1){
        if (l_m_adjust != l_ld_bytes){
          libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
              i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0,
              i_gp_reg_mapping->gp_reg_in, ((long long)l_ld_bytes - l_m_adjust));
        }
    }

    libxsmm_rv64_instruction_rvv_setivli( io_generated_code, i_vlen, i_gp_reg_mapping->gp_reg_scratch_0, l_sew, LIBXSMM_RV64_LMUL_M1);
  }

  /* Reset the base address */
  libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_SUB,
                                              i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in,
                                              offset);
}

LIBXSMM_API_INTERN
void libxsmm_store_rv64_2d_reg_block( libxsmm_generated_code*                 io_generated_code,
                                         libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                         const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                         const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                         unsigned int                            i_vlen,
                                         unsigned int                            i_avlen,
                                         unsigned int                            i_start_vreg,
                                         unsigned int                            i_m_blocking,
                                         unsigned int                            i_n_blocking,
                                         unsigned int                            i_mask_last_m_chunk,
                                         unsigned int                            i_mask_reg) {
  unsigned int im, in, cur_vreg;
  unsigned int bcast_row = ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW) > 0))
                             ? 1 : 0;
  unsigned int bcast_col = ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL) > 0))
                             ? 1 : 0;
  unsigned int bcast_scalar = ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR) > 0)) ? 1 : 0;
  unsigned int bcast_input = ( bcast_row == 1 || bcast_col == 1 || bcast_scalar == 1 ) ? 1 : 0;
  unsigned int l_ld_bytes = i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out;
  unsigned int l_m_adjust = ( i_mask_last_m_chunk == 0 ) ? i_micro_kernel_config->datatype_size_out * i_vlen * i_m_blocking : i_micro_kernel_config->datatype_size_out * ( (i_vlen * (i_m_blocking-1)) + i_mask_last_m_chunk );
  unsigned int offset = 0;
  unsigned char l_is_sve = 0;
  unsigned int l_is_comp_unzip = ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_UNZIP)) ? 1 : 0;
  unsigned int l_is_comp_zip = ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_ZIP)) ? 1 : 0;
  unsigned int l_sew = (i_micro_kernel_config->datatype_size_in == 8) ? LIBXSMM_RV64_SEW_Q : (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_RV64_SEW_D : (i_micro_kernel_config->datatype_size_in == 2) ? LIBXSMM_RV64_SEW_W : LIBXSMM_RV64_SEW_B;

 LIBXSMM_UNUSED(cur_vreg);
 LIBXSMM_UNUSED(bcast_input);
 LIBXSMM_UNUSED(l_m_adjust);
 LIBXSMM_UNUSED(l_is_sve);
 LIBXSMM_UNUSED(l_is_comp_unzip);
 LIBXSMM_UNUSED(l_is_comp_zip);

 /* In this case we do not have to load any data */
 if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_XOR)) return;

 if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_ZIP)) {
     offset = (l_ld_bytes*i_n_blocking*i_m_blocking);
     /*libxsmm_generator_set_p_register_rv64_sve( io_generated_code, 2, i_vlen * i_micro_kernel_config->datatype_size_in, i_gp_reg_mapping->gp_reg_scratch_0 );*/
     for (in = 0; in < i_n_blocking; in++) {
       for (im = 0; im < i_m_blocking; im++) {
         unsigned int l_is_f32_or_f64 = (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) || LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ? 1 : 0;
         unsigned int l_masked_elements = (l_is_f32_or_f64) ? (im == i_m_blocking - 1) ? i_mask_last_m_chunk : 0 : (im == i_m_blocking - 1) ? (i_mask_last_m_chunk > 0) ? i_mask_last_m_chunk : i_vlen : i_vlen;
         LIBXSMM_UNUSED(l_masked_elements);
      }
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                                                  i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_scratch_1, i_gp_reg_mapping->gp_reg_in,
                                                  offset );
    }
    return;
  }

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      offset = (l_ld_bytes*i_n_blocking);
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

      if ((im == (i_m_blocking - 1)) && i_mask_last_m_chunk != 0)
        libxsmm_rv64_instruction_rvv_setvli( io_generated_code, i_avlen, i_gp_reg_mapping->gp_reg_scratch_0, l_sew, LIBXSMM_RV64_LMUL_M1);

      libxsmm_rv64_instruction_rvv_move( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VSE32_V,
            i_gp_reg_mapping->gp_reg_out, LIBXSMM_RV64_GP_REG_UNDEF, cur_vreg, 1);

      /*  Increament in address */
      if ((im == (i_m_blocking - 1)) && i_mask_last_m_chunk != 0){
        libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                                                  i_gp_reg_mapping->gp_reg_out,  i_gp_reg_mapping->gp_reg_scratch_1, i_gp_reg_mapping->gp_reg_out,
                                                  (unsigned long)i_mask_last_m_chunk * i_micro_kernel_config->datatype_size_out);
      } else {
        libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                                                  i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_1, i_gp_reg_mapping->gp_reg_out,
                                                  (unsigned long)i_vlen * i_micro_kernel_config->datatype_size_out);
      }
    }

    libxsmm_rv64_instruction_rvv_setivli( io_generated_code, i_vlen, i_gp_reg_mapping->gp_reg_scratch_0, l_sew, LIBXSMM_RV64_LMUL_M1);

    if (l_m_adjust != l_ld_bytes)
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                                                i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_1, i_gp_reg_mapping->gp_reg_out,
                                                ((long long)l_ld_bytes - l_m_adjust));
  }

  /* Reset the base address */
  libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_SUB,
                                              i_gp_reg_mapping->gp_reg_out, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out,
                                              offset);

}

LIBXSMM_API_INTERN
void libxsmm_compute_unary_rv64_2d_reg_block_op( libxsmm_generated_code*                 io_generated_code,
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
  unsigned char l_pred_reg = LIBXSMM_CAST_UCHAR(i_mask_reg);
  unsigned char l_op_needs_predicates =
    i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_X2 &&
    i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL &&
    i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT;

  LIBXSMM_UNUSED(i_gp_reg_mapping);
  LIBXSMM_UNUSED(i_vlen);
  LIBXSMM_UNUSED(i_mask_last_m_chunk);
  LIBXSMM_UNUSED(l_op_needs_predicates);
  LIBXSMM_UNUSED(l_pred_reg);

  /* TODO: Setup vector length */
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
          libxsmm_rv64_instruction_rvv_compute( io_generated_code, LIBXSMM_RV64_INSTR_RVV_VFMUL_VV,
                                                   cur_vreg, cur_vreg, cur_vreg, 1);
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_compute_binary_rv64_2d_reg_block( libxsmm_generated_code*                 io_generated_code,
                                               libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                               const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                               const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                               unsigned int                            i_vlen,
                                               unsigned int                            i_avlen,
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
  unsigned int offset2 = 0;
  unsigned int _in_blocking = (bcast_col == 1) ? 1 : i_n_blocking;
  unsigned int l_load_instr = (i_micro_kernel_config->datatype_size_in == 8) ? LIBXSMM_RV64_INSTR_RVV_VLE64_V : LIBXSMM_RV64_INSTR_RVV_VLE32_V;
  unsigned int l_sew = (i_micro_kernel_config->datatype_size_in == 8) ? LIBXSMM_RV64_SEW_Q : (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_RV64_SEW_D : (i_micro_kernel_config->datatype_size_in == 2) ? LIBXSMM_RV64_SEW_W : LIBXSMM_RV64_SEW_B;

  switch (i_mateltwise_desc->param) {
    case LIBXSMM_MELTW_TYPE_BINARY_ADD: {
      binary_op_instr = LIBXSMM_RV64_INSTR_RVV_VFADD_VV;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_MUL: {
      binary_op_instr = LIBXSMM_RV64_INSTR_RVV_VFMUL_VV;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_SUB: {
      binary_op_instr = LIBXSMM_RV64_INSTR_RVV_VFSUB_VV;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_DIV: {
      binary_op_instr = LIBXSMM_RV64_INSTR_RVV_VFDIV_VV;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_MULADD: {
      binary_op_instr = LIBXSMM_RV64_INSTR_RVV_VFMACC_VV;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_MAX: {
      binary_op_instr = LIBXSMM_RV64_INSTR_RVV_VFMAX_VV;
    } break;
    case LIBXSMM_MELTW_TYPE_BINARY_MIN: {
      binary_op_instr = LIBXSMM_RV64_INSTR_RVV_VFMIN_VV;
    } break;
    default:;
  }

  for (in = 0; in < i_n_blocking; in++) {
    for (im = 0; im < i_m_blocking; im++) {
      cur_vreg = i_start_vreg + in * i_m_blocking + im;
      if ( (bcast_row == 1) || (bcast_scalar == 1) ) {
        offset2 = (bcast_scalar == 1) ? 0 : i_micro_kernel_config->datatype_size_in1*i_mateltwise_desc->ldi2*i_n_blocking;

        libxsmm_generator_bcastload_masked_vreg_rv64( io_generated_code, i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_1, i_micro_kernel_config->tmp_vreg,
          i_micro_kernel_config->datatype_size_in1, (im == i_m_blocking - 1) ? 1 : 0, i_vlen, i_avlen, (bcast_scalar == 1) ? 1 : 0 );
      }

      if ( bcast_input == 0 || bcast_col == 1) {
        offset2 = (bcast_col == 1) ? l_m_adjust_in2:(l_ld_bytes_in2*_in_blocking);

        if ((im == (i_m_blocking - 1)) && i_mask_last_m_chunk != 0)
          libxsmm_rv64_instruction_rvv_setvli( io_generated_code, i_avlen, i_gp_reg_mapping->gp_reg_scratch_0, l_sew, LIBXSMM_RV64_LMUL_M1);

        libxsmm_rv64_instruction_rvv_move( io_generated_code, l_load_instr, i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_1, i_micro_kernel_config->tmp_vreg, 1);

        /*  Increament in address */
        if ((im == (i_m_blocking - 1)) && i_mask_last_m_chunk != 0){
          libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
              i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_1, i_gp_reg_mapping->gp_reg_in2,
              (unsigned long)i_mask_last_m_chunk * i_micro_kernel_config->datatype_size_in);
        } else {
          libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
              i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_1, i_gp_reg_mapping->gp_reg_in2,
              (unsigned long)i_vlen * i_micro_kernel_config->datatype_size_in);
        }
      }

      libxsmm_rv64_instruction_rvv_compute( io_generated_code, binary_op_instr, cur_vreg, i_micro_kernel_config->tmp_vreg, cur_vreg, 1);
    }

    libxsmm_rv64_instruction_rvv_setivli( io_generated_code, i_vlen, i_gp_reg_mapping->gp_reg_scratch_0, l_sew, LIBXSMM_RV64_LMUL_M1);

    if ( (l_ld_bytes != l_m_adjust) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_MULADD) ) {
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                                                     i_gp_reg_mapping->gp_reg_out,i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_out,
                                                     ((long long)l_ld_bytes - l_m_adjust) );
    }

    if ( bcast_input == 0 ) {
      if ( l_ld_bytes_in2 != l_m_adjust_in2 ) {
        libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                                                       i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in2,
                                                       (long long)l_ld_bytes_in2 - l_m_adjust_in2 );
      }
    }

    if (bcast_col == 1 && in < (i_n_blocking-1)) {
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_SUB,
                                                     i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in2,
                                                     offset2 );
    }

    if (bcast_row == 1) {
      libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                                                  i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in2,
                                                  (long long)i_mateltwise_desc->ldi2 * i_micro_kernel_config->datatype_size_in1 );
    }
  }

  libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_SUB,
                                              i_gp_reg_mapping->gp_reg_in2, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_in2,
                                              offset2 );
}

LIBXSMM_API_INTERN
void libxsmm_compute_unary_binary_rv64_2d_reg_block( libxsmm_generated_code*                 io_generated_code,
                                                     libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                     const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                     const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                     unsigned int                            i_vlen,
                                                     unsigned int                            i_avlen,
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
        libxsmm_compute_unary_rv64_2d_reg_block_op( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_vlen, i_start_vreg, i_m_blocking, i_n_blocking, i_mask_last_m_chunk, i_mask_reg);
      } break;
    }
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
    libxsmm_compute_binary_rv64_2d_reg_block( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_vlen, i_avlen, i_start_vreg, i_m_blocking, i_n_blocking, i_mask_last_m_chunk, i_mask_reg);
  }
}

LIBXSMM_API_INTERN
void libxsmm_setup_input_output_rv64_masks( libxsmm_generated_code*                 io_generated_code,
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

  /* reserving predicate 0 and 1 for ptrue and remainder (when loading/storing) */
  i_micro_kernel_config->reserved_mask_regs += 2;
}

LIBXSMM_API_INTERN
void libxsmm_configure_microkernel_rv64_loops( libxsmm_generated_code*                 io_generated_code,
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

  m_trips = LIBXSMM_UPDIV(i_m, i_vlen_in);
  n_trips = i_n;

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
void libxsmm_finalize_unary_rv64_kernel_vregs_masks( libxsmm_generated_code*                 io_generated_code,
                                                     libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                     unsigned int                            op,
                                                     unsigned int                            flags,
                                                     unsigned int                            i_gp_reg_tmp,
                                                     const unsigned int                      i_gp_reg_aux0,
                                                     const unsigned int                      i_gp_reg_aux1 ) {

  LIBXSMM_UNUSED(flags);
  LIBXSMM_UNUSED(i_gp_reg_tmp);
  LIBXSMM_UNUSED(i_gp_reg_aux1);
}

LIBXSMM_API_INTERN
void libxsmm_configure_rv64_kernel_vregs_masks( libxsmm_generated_code*                  io_generated_code,
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
void libxsmm_finalize_kernel_vregs_rv64_masks( libxsmm_generated_code*                 io_generated_code,
                                               libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                               const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                               unsigned int                            i_gp_reg_tmp,
                                               const unsigned int                      i_gp_reg_aux0,
                                               const unsigned int                      i_gp_reg_aux1) {
  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
    libxsmm_finalize_unary_rv64_kernel_vregs_masks( io_generated_code, i_micro_kernel_config, i_mateltwise_desc->param, i_mateltwise_desc->flags, i_gp_reg_tmp, i_gp_reg_aux0, i_gp_reg_aux1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_unary_rv64_binary_2d_microkernel( libxsmm_generated_code*                 io_generated_code,
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

  unsigned int l_sew = (i_micro_kernel_config->datatype_size_in == 8) ? LIBXSMM_RV64_SEW_Q : (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_RV64_SEW_D : (i_micro_kernel_config->datatype_size_in == 2) ? LIBXSMM_RV64_SEW_W : LIBXSMM_RV64_SEW_B;

  use_m_input_masking = 0;
  use_m_output_masking = 0;
  mask_reg_in = LIBXSMM_RV64_GP_REG_UNDEF;
  mask_reg_out = LIBXSMM_RV64_GP_REG_UNDEF;

  LIBXSMM_UNUSED(reserved_zmms);
  LIBXSMM_UNUSED(i_vlen_in);
  LIBXSMM_UNUSED(i_vlen_out);
  LIBXSMM_UNUSED(loop_type);
  LIBXSMM_UNUSED(mask_reg_out);
  LIBXSMM_UNUSED(use_m_output_masking);

  /* Configure microkernel masks */
  libxsmm_setup_input_output_rv64_masks( io_generated_code, i_micro_kernel_config, i_mateltwise_desc,
    i_gp_reg_mapping->gp_reg_scratch_0, i_m, &use_m_input_masking, &mask_reg_in, &use_m_output_masking, &mask_reg_out);

  /* Configure microkernel loops */
  libxsmm_configure_microkernel_rv64_loops( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc, i_m, i_n, use_m_input_masking,
    &m_trips, &n_trips, &m_unroll_factor, &n_unroll_factor, &m_assm_trips, &n_assm_trips,
    &out_loop_trips, &inner_loop_trips, &out_loop_bound, &inner_loop_bound, &out_loop_reg, &inner_loop_reg, &out_unroll_factor, &inner_unroll_factor );

  libxsmm_rv64_instruction_rvv_setivli( io_generated_code, i_vlen_in, i_gp_reg_mapping->gp_reg_scratch_0, l_sew, LIBXSMM_RV64_LMUL_M1);

  /* Headers of microkernel loops */
  if (out_loop_trips > 1) {
    libxsmm_generator_loop_header_rv64(io_generated_code, io_loop_label_tracker, out_loop_reg, out_loop_bound);
  }

  if (inner_loop_trips > 1) {
    libxsmm_generator_loop_header_rv64(io_generated_code, io_loop_label_tracker, inner_loop_reg, inner_loop_bound);
  }

  if (use_m_input_masking)
    libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, LIBXSMM_RV64_GP_REG_X20, use_m_input_masking);
  else
    libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, LIBXSMM_RV64_GP_REG_X20, i_vlen_in);

  libxsmm_rv64_instruction_rvv_setivli( io_generated_code, i_vlen_in, i_gp_reg_mapping->gp_reg_scratch_0, l_sew, LIBXSMM_RV64_LMUL_M1);

  /* Load block of registers */
  libxsmm_load_rv64_2d_reg_block(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc, i_vlen_in, LIBXSMM_RV64_GP_REG_X20, reserved_zmms, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in);

  /* Compute on registers */
  libxsmm_compute_unary_binary_rv64_2d_reg_block(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
      i_vlen_in, LIBXSMM_RV64_GP_REG_X20, reserved_zmms, m_unroll_factor, n_unroll_factor, use_m_input_masking, mask_reg_in);

  libxsmm_rv64_instruction_rvv_setivli( io_generated_code, i_vlen_out, i_gp_reg_mapping->gp_reg_scratch_0, l_sew, LIBXSMM_RV64_LMUL_M1);

  /* Store block of registers */
  libxsmm_store_rv64_2d_reg_block(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc, i_vlen_out, LIBXSMM_RV64_GP_REG_X20, reserved_zmms, m_unroll_factor, n_unroll_factor, use_m_output_masking, mask_reg_out);

  /* Footers of microkernel loops */
  if (inner_loop_trips > 1) {
    /* Advance input/output pointers */
    loop_type = (inner_loop_reg == i_gp_reg_mapping->gp_reg_m_loop) ? LOOP_TYPE_M : LOOP_TYPE_N;
    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_in, LIBXSMM_RV64_INSTR_GP_ADD, inner_unroll_factor, loop_type);

    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_out, LIBXSMM_RV64_INSTR_GP_ADD, inner_unroll_factor, loop_type);

    if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP)) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_out2, LIBXSMM_RV64_INSTR_GP_ADD, inner_unroll_factor, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in2, LIBXSMM_RV64_INSTR_GP_ADD, inner_unroll_factor, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {/* adjust relu/dropout mask pointers */
      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) ||
          (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
          (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU)        || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_RV64_INSTR_GP_ADD, inner_unroll_factor, loop_type);
      } else if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_dropoutmask, LIBXSMM_RV64_INSTR_GP_ADD, inner_unroll_factor, loop_type);
      }
    }
    libxsmm_generator_loop_footer_rv64(io_generated_code, io_loop_label_tracker, inner_loop_reg, inner_unroll_factor);
    /* Reset input/output pointers */
    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_in, LIBXSMM_RV64_INSTR_GP_SUB, inner_unroll_factor * inner_loop_trips, loop_type);

    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_out, LIBXSMM_RV64_INSTR_GP_SUB, inner_unroll_factor * inner_loop_trips, loop_type);

    if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP)) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_out2, LIBXSMM_RV64_INSTR_GP_SUB, inner_unroll_factor * inner_loop_trips, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in2, LIBXSMM_RV64_INSTR_GP_SUB, inner_unroll_factor * inner_loop_trips, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) ||
          (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
          (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU)        || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_RV64_INSTR_GP_SUB, inner_unroll_factor * inner_loop_trips, loop_type);
      } else if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_dropoutmask, LIBXSMM_RV64_INSTR_GP_SUB, inner_unroll_factor * inner_loop_trips, loop_type);
      }
    }
  }

  if (out_loop_trips > 1) {
    /* Advance input/output pointers */
    loop_type = (out_loop_reg == i_gp_reg_mapping->gp_reg_m_loop) ? LOOP_TYPE_M : LOOP_TYPE_N;

    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_in, LIBXSMM_RV64_INSTR_GP_ADD, out_unroll_factor, loop_type);

    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_out, LIBXSMM_RV64_INSTR_GP_ADD, out_unroll_factor, loop_type);

    if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP)) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_out2, LIBXSMM_RV64_INSTR_GP_ADD, out_unroll_factor, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in2, LIBXSMM_RV64_INSTR_GP_ADD, out_unroll_factor, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) ||
          (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
          (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU)        || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_RV64_INSTR_GP_ADD, out_unroll_factor, loop_type);
      } else if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_dropoutmask, LIBXSMM_RV64_INSTR_GP_ADD, out_unroll_factor, loop_type);
      }
    }
    libxsmm_generator_loop_footer_rv64(io_generated_code, io_loop_label_tracker, out_loop_reg, out_unroll_factor);
    /* Reset input/output pointers */
    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_in, LIBXSMM_RV64_INSTR_GP_SUB, out_unroll_factor * out_loop_trips, loop_type);

    libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
        i_gp_reg_mapping->gp_reg_out, LIBXSMM_RV64_INSTR_GP_SUB, out_unroll_factor * out_loop_trips, loop_type);

    if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP)) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_out2, LIBXSMM_RV64_INSTR_GP_SUB, out_unroll_factor * out_loop_trips, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
      libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in2, LIBXSMM_RV64_INSTR_GP_SUB, out_unroll_factor * out_loop_trips, loop_type);
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
      if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) ||
          (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
          (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU)        || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_RV64_INSTR_GP_SUB, out_unroll_factor * out_loop_trips, loop_type);
      } else if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
        libxsmm_generator_mateltwise_unary_binary_adjust_in_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_dropoutmask, LIBXSMM_RV64_INSTR_GP_SUB, out_unroll_factor * out_loop_trips, loop_type);
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_unary_binary_rv64_microkernel( libxsmm_generated_code*             io_generated_code,
                                                      libxsmm_loop_label_tracker*         io_loop_label_tracker,
                                                      libxsmm_mateltwise_gp_reg_mapping*  i_gp_reg_mapping,
                                                      libxsmm_mateltwise_kernel_config*   i_micro_kernel_config,
                                                      const libxsmm_meltw_descriptor*     i_mateltwise_desc ) {
  unsigned int loop_order, m_blocking = 0, out_blocking = 0, out_bound = 0, out_block = 0, n_blocking = 0, inner_blocking = 0, inner_block = 0, inner_bound = 0, n_microkernel = 0, m_microkernel = 0;
  /*unsigned int out_ind, inner_ind, reset_regs, loop_type;*/
  unsigned int reset_regs, loop_type;
  unsigned int out_ind, inner_ind;
  unsigned int available_vregs = 32;

  unsigned int l_gp_reg_tmp = LIBXSMM_RV64_GP_REG_X17;
  unsigned int l_gp_reg_aux0 = LIBXSMM_RV64_GP_REG_UNDEF;
  unsigned int l_gp_reg_aux1 = LIBXSMM_RV64_GP_REG_UNDEF;

  int l_offset_ptr_a = (int)sizeof(libxsmm_matrix_op_arg);
  int l_offset_ptr_b = (int)(sizeof(libxsmm_matrix_op_arg) + sizeof(libxsmm_matrix_arg));
  int l_offset_ptr_c = (int)(sizeof(libxsmm_matrix_op_arg) + 2*sizeof(libxsmm_matrix_arg));
  int l_offset_ptr_d = (int)(sizeof(libxsmm_matrix_op_arg) + 3*sizeof(libxsmm_matrix_arg));

  LIBXSMM_UNUSED(l_gp_reg_tmp);
  LIBXSMM_UNUSED(l_gp_reg_aux0);
  LIBXSMM_UNUSED(l_gp_reg_aux1);

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
       ( LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) && LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) ) ) {
    /* fine */
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }

  /* Configure vlens */
  libxsmm_generator_configure_rv64_vlens(i_mateltwise_desc, i_micro_kernel_config);

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in        = LIBXSMM_RV64_GP_REG_X18;
  i_gp_reg_mapping->gp_reg_out       = LIBXSMM_RV64_GP_REG_X19;

  i_gp_reg_mapping->gp_reg_m_loop    = LIBXSMM_RV64_GP_REG_X13;
  i_gp_reg_mapping->gp_reg_n_loop    = LIBXSMM_RV64_GP_REG_X14;
  i_gp_reg_mapping->gp_reg_scratch_0 = LIBXSMM_RV64_GP_REG_X15;
  i_gp_reg_mapping->gp_reg_scratch_1 = LIBXSMM_RV64_GP_REG_X16;

  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY ) {
    i_gp_reg_mapping->gp_reg_in2  = LIBXSMM_RV64_GP_REG_X17;
  } else {
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP) {
      i_gp_reg_mapping->gp_reg_out2 = LIBXSMM_RV64_GP_REG_X12;
    }
    if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV)       ||
         (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV)    ) {
      i_gp_reg_mapping->gp_reg_relumask = LIBXSMM_RV64_GP_REG_X12;
    }
    if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
      i_gp_reg_mapping->gp_reg_relumask = LIBXSMM_RV64_GP_REG_X12;
      i_gp_reg_mapping->gp_reg_fam_lualpha = LIBXSMM_RV64_GP_REG_X13;
    }
    if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ) {
      i_gp_reg_mapping->gp_reg_fam_lualpha = LIBXSMM_RV64_GP_REG_X13;
    }
    if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_UNZIP ) {
      i_gp_reg_mapping->gp_reg_offset = LIBXSMM_RV64_GP_REG_X12;
    }
    if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X2 ) {
      i_gp_reg_mapping->gp_reg_offset = LIBXSMM_RV64_GP_REG_X12;
      i_gp_reg_mapping->gp_reg_shift_vals = LIBXSMM_RV64_GP_REG_X14;
    }
    if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3 ) {
      i_gp_reg_mapping->gp_reg_offset = LIBXSMM_RV64_GP_REG_X12;
      i_gp_reg_mapping->gp_reg_offset_2 = LIBXSMM_RV64_GP_REG_X13;
      i_gp_reg_mapping->gp_reg_shift_vals = LIBXSMM_RV64_GP_REG_X14;
      i_gp_reg_mapping->gp_reg_shift_vals2 = LIBXSMM_RV64_GP_REG_X15;
    }
    if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV) ) {
      i_gp_reg_mapping->gp_reg_dropoutmask = LIBXSMM_RV64_GP_REG_X12;
      i_gp_reg_mapping->gp_reg_dropoutprob = LIBXSMM_RV64_GP_REG_X13;
      if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT ) {
        i_gp_reg_mapping->gp_reg_prngstate = LIBXSMM_RV64_GP_REG_X14;
      } else {
        i_gp_reg_mapping->gp_reg_prngstate = LIBXSMM_RV64_GP_REG_UNDEF;
      }
    } else {
      i_gp_reg_mapping->gp_reg_dropoutmask = LIBXSMM_RV64_GP_REG_UNDEF;
      i_gp_reg_mapping->gp_reg_dropoutprob = LIBXSMM_RV64_GP_REG_UNDEF;
      i_gp_reg_mapping->gp_reg_prngstate = LIBXSMM_RV64_GP_REG_UNDEF;
    }
    if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_QUANT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DEQUANT) ) {
      i_gp_reg_mapping->gp_reg_quant_sf = LIBXSMM_RV64_GP_REG_X12;
    } else {
      i_gp_reg_mapping->gp_reg_quant_sf = LIBXSMM_X86_GP_REG_UNDEF;
    }
  }

  /* load the input pointer and output pointer */
  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY ) {
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_in, l_offset_ptr_a );
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_out, l_offset_ptr_b );
    if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP) {
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_out2, l_offset_ptr_b + 8 );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU) {
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_relumask, l_offset_ptr_b + 8 );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) {
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_relumask, l_offset_ptr_a + 8 );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU)  {
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_fam_lualpha, 0 );
      /*l_gp_reg_aux1 = i_gp_reg_mapping->gp_reg_fam_lualpha;*/
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) {
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_relumask, l_offset_ptr_a + 8 );
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_fam_lualpha, 0 );
      /*l_gp_reg_aux1 = i_gp_reg_mapping->gp_reg_fam_lualpha;*/
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) {
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_fam_lualpha, 0 );
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_relumask, l_offset_ptr_b + 8 );
      /*l_gp_reg_aux1 = i_gp_reg_mapping->gp_reg_fam_lualpha;*/
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) {
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_relumask, l_offset_ptr_a + 8 );
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_fam_lualpha, 0 );
      /*l_gp_reg_aux1 = i_gp_reg_mapping->gp_reg_fam_lualpha;*/
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_UNZIP) {
      libxsmm_rv64_instruction_alu_move( io_generated_code,LIBXSMM_RV64_INSTR_GP_LW,  i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_offset, l_offset_ptr_b + 8 );
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_offset, i_gp_reg_mapping->gp_reg_offset, 0 );
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X2) {
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_offset, l_offset_ptr_b + 8 );
      /*l_gp_reg_aux0 = i_gp_reg_mapping->gp_reg_offset;*/
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3) {
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_offset, l_offset_ptr_b + 8 );
      /*l_gp_reg_aux0 = i_gp_reg_mapping->gp_reg_offset;*/
      /*l_gp_reg_aux1 = i_gp_reg_mapping->gp_reg_offset_2;*/
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) {
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_prngstate, 8 );
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_dropoutprob, 0 );
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_dropoutmask, l_offset_ptr_b + 8 );
      /*l_gp_reg_aux0 = i_gp_reg_mapping->gp_reg_prngstate;*/
      /*l_gp_reg_aux1 = i_gp_reg_mapping->gp_reg_dropoutprob;*/
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV) {
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_dropoutmask, l_offset_ptr_a + 8 );
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_dropoutprob, 0 );
      /*l_gp_reg_aux1 = i_gp_reg_mapping->gp_reg_dropoutprob;*/
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_QUANT) {
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_quant_sf, l_offset_ptr_a + 8 );
      /*l_gp_reg_aux0 = i_gp_reg_mapping->gp_reg_quant_sf;*/
    } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DEQUANT) {
      libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_quant_sf, l_offset_ptr_a + 8 );
      /*l_gp_reg_aux0 = i_gp_reg_mapping->gp_reg_quant_sf;*/
    }
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY ) {
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_in, l_offset_ptr_a );
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_in2, l_offset_ptr_b );
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_out, l_offset_ptr_c );
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY ) {
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_in, l_offset_ptr_a );
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_in2, l_offset_ptr_b );
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_in3, l_offset_ptr_c );
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LW, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_out, l_offset_ptr_d );
  } else {
    /* This should not happen */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }

  /* Based on kernel type reserve zmms and mask registers */
  libxsmm_configure_rv64_kernel_vregs_masks( io_generated_code, i_micro_kernel_config, i_mateltwise_desc, i_gp_reg_mapping->gp_reg_scratch_0, i_gp_reg_mapping->gp_reg_scratch_1, l_gp_reg_aux0, l_gp_reg_aux1 );

  available_vregs = available_vregs - i_micro_kernel_config->reserved_zmms;

  /* Configure M and N blocking factors */
  /* todo sve: here we might intercept the m, n blocking size */
  libxsmm_generator_configure_rv64_M_N_blocking( io_generated_code, i_mateltwise_desc, i_mateltwise_desc->m, i_mateltwise_desc->n, i_micro_kernel_config->vlen_in, &m_blocking, &n_blocking, available_vregs);
  libxsmm_generator_configure_rv64_loop_order(i_mateltwise_desc, &loop_order, &m_blocking, &n_blocking, &out_blocking, &inner_blocking, &out_bound, &inner_bound);
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

      libxsmm_generator_unary_rv64_binary_2d_microkernel(io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc, m_microkernel, n_microkernel);

      inner_ind += inner_block;

      if (inner_ind != inner_bound) {
        reset_regs = 1;
        /* Advance input/output pointers */
        loop_type = (loop_order == NM_LOOP_ORDER) ? LOOP_TYPE_M : LOOP_TYPE_N;

        libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_in, LIBXSMM_RV64_INSTR_GP_ADD, m_microkernel, n_microkernel, loop_type );

        libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_out, LIBXSMM_RV64_INSTR_GP_ADD, m_microkernel, n_microkernel, loop_type );

        if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP)) {
          libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
              i_gp_reg_mapping->gp_reg_out2, LIBXSMM_RV64_INSTR_GP_ADD, m_microkernel, n_microkernel, loop_type );
        }

        if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
          libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
              i_gp_reg_mapping->gp_reg_in2, LIBXSMM_RV64_INSTR_GP_ADD, m_microkernel, n_microkernel, loop_type );
        }

        if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
          if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) ||
              (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
              (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU)        || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
            libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
                i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_RV64_INSTR_GP_ADD, m_microkernel, n_microkernel, loop_type );
          } else if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
            libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
                i_gp_reg_mapping->gp_reg_dropoutmask, LIBXSMM_RV64_INSTR_GP_ADD, m_microkernel, n_microkernel, loop_type );
          }
        }
      }
    }

    /* If needed, readjust the registers */
    if (reset_regs == 1) {
      loop_type = (loop_order == NM_LOOP_ORDER) ? LOOP_TYPE_M : LOOP_TYPE_N;

      libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in, LIBXSMM_RV64_INSTR_GP_SUB, m_microkernel, n_microkernel, loop_type );

      libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_out, LIBXSMM_RV64_INSTR_GP_SUB, m_microkernel, n_microkernel, loop_type );

      if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP)) {
        libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_out2, LIBXSMM_RV64_INSTR_GP_SUB, m_microkernel, n_microkernel, loop_type );
      }

      if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
        libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_in2, LIBXSMM_RV64_INSTR_GP_SUB, m_microkernel, n_microkernel, loop_type );
      }

      if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
        if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) ||
            (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
            (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU)        || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
          libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
              i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_RV64_INSTR_GP_SUB, m_microkernel, n_microkernel, loop_type );
        } else if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
          libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
              i_gp_reg_mapping->gp_reg_dropoutmask, LIBXSMM_RV64_INSTR_GP_SUB, m_microkernel, n_microkernel, loop_type );
        }
      }
    }

    out_ind += out_block;

    if (out_ind != out_bound) {
      /* Advance input/output pointers */
      loop_type = (loop_order == MN_LOOP_ORDER) ? LOOP_TYPE_M : LOOP_TYPE_N;

      libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_in, LIBXSMM_RV64_INSTR_GP_ADD, m_microkernel, n_microkernel, loop_type );

      libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
          i_gp_reg_mapping->gp_reg_out, LIBXSMM_RV64_INSTR_GP_ADD, m_microkernel, n_microkernel, loop_type );

      if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP)) {
        libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_out2, LIBXSMM_RV64_INSTR_GP_ADD, m_microkernel, n_microkernel, loop_type );
      }

      if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
        libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
            i_gp_reg_mapping->gp_reg_in2, LIBXSMM_RV64_INSTR_GP_ADD, m_microkernel, n_microkernel, loop_type );
      }

      if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
        if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU)       || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV) ||
            (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV) ||
            (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU)        || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_ELU_INV) ) {
          libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
              i_gp_reg_mapping->gp_reg_relumask, LIBXSMM_RV64_INSTR_GP_ADD, m_microkernel, n_microkernel, loop_type );
        } else if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV)) {
          libxsmm_generator_mateltwise_unary_binary_adjust_after_microkernel_addr_rv64_gp_reg( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc,
              i_gp_reg_mapping->gp_reg_dropoutmask, LIBXSMM_RV64_INSTR_GP_ADD, m_microkernel, n_microkernel, loop_type );
        }
      }
    }
  }

  /* save some globale state if needed */
  libxsmm_finalize_kernel_vregs_rv64_masks( io_generated_code, i_micro_kernel_config, i_mateltwise_desc, l_gp_reg_tmp, l_gp_reg_aux0, l_gp_reg_aux1 );
}
