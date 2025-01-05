/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include "generator_mateltwise_transform_rv64.h"
#include "generator_common_rv64.h"
#include "generator_rv64_instructions.h"
#include "generator_common.h"
#include "generator_mateltwise_rv64.h"

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_mbit_scalar_rv64_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                              libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                              const unsigned int                      i_gp_reg_in,
                                                                              const unsigned int                      i_gp_reg_out,
                                                                              const unsigned int                      i_gp_reg_m_loop,
                                                                              const unsigned int                      i_gp_reg_n_loop,
                                                                              const unsigned int                      i_gp_reg_scratch,
                                                                              const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                              const libxsmm_meltw_descriptor*         i_mateltwise_desc ) {
  /* m loop header */
  libxsmm_generator_loop_header_rv64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, i_mateltwise_desc->m );

  /* n loop header */
  libxsmm_generator_loop_header_rv64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, i_mateltwise_desc->n );

  /* actual transpose */
  libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_FLW,
                                          i_gp_reg_in, LIBXSMM_RV64_GP_REG_X5, 0 );

  libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_FSW,
                                          i_gp_reg_out, LIBXSMM_RV64_GP_REG_X5, 0 );

  /* advance input pointer */
  libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                                                 i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in,
                                                 (long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in );

  /* advance output pointer */
  libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                                                 i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out,
                                                 i_micro_kernel_config->datatype_size_out );

  /* close n loop */
  libxsmm_generator_loop_footer_rv64( io_generated_code, io_loop_label_tracker, i_gp_reg_n_loop, (i_mateltwise_desc->n) ? 1 : 0 );

  /* advance output pointer */
  libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADD,
                                                 i_gp_reg_out, i_gp_reg_scratch, i_gp_reg_out,
                                                 ((long long)i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out) - ((long long)i_micro_kernel_config->datatype_size_out * i_mateltwise_desc->n) );

  /* advance input pointer */
  libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_SUB,
                                                 i_gp_reg_in, i_gp_reg_scratch, i_gp_reg_in,
                                                 ((long long)i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in * i_mateltwise_desc->n) - ((long long)i_micro_kernel_config->datatype_size_in) );

  /* close m loop */
  libxsmm_generator_loop_footer_rv64( io_generated_code, io_loop_label_tracker, i_gp_reg_m_loop, (i_mateltwise_desc->n) ? 1 : 0 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_rv64_microkernel( libxsmm_generated_code*                        io_generated_code,
                                                   libxsmm_loop_label_tracker*                    io_loop_label_tracker,
                                                   libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
                                                   const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
                                                   const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {
  i_gp_reg_mapping->gp_reg_in        = LIBXSMM_RV64_GP_REG_X18;
  i_gp_reg_mapping->gp_reg_out       = LIBXSMM_RV64_GP_REG_X19;
  i_gp_reg_mapping->gp_reg_m_loop    = LIBXSMM_RV64_GP_REG_X20;
  i_gp_reg_mapping->gp_reg_n_loop    = LIBXSMM_RV64_GP_REG_X21;
  i_gp_reg_mapping->gp_reg_scratch_0 = LIBXSMM_RV64_GP_REG_X22;
  i_gp_reg_mapping->gp_reg_scratch_1 = LIBXSMM_RV64_GP_REG_X23;

  /* load pointers from struct */
  libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_in, 32 );
  libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_LD, i_gp_reg_mapping->gp_reg_param_struct, i_gp_reg_mapping->gp_reg_out, 64 );

  /* check leading dimnesions and sizes */
  if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) ||
       (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2_TO_VNNI2T)    ) {
    /* coverity[copy_paste_error] */
    if ( (i_mateltwise_desc->m > i_mateltwise_desc->ldi) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
      return;
    }
    if ( (i_mateltwise_desc->n > i_mateltwise_desc->ldo) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB );
      return;
    }
  } else if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2)     ||
              (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD) ||
              (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD2)         ||
              (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD2)         ||
              (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2)          ) {
    if ( (i_mateltwise_desc->m > i_mateltwise_desc->ldi) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
      return;
    }
    if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD2)  ||
         (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2)    ) {
      if ( (i_mateltwise_desc->m + i_mateltwise_desc->m%2) > i_mateltwise_desc->ldo ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB );
        return;
      }
    } else {
      if ( (i_mateltwise_desc->m > i_mateltwise_desc->ldo) ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB );
        return;
      }
    }
  } else {
    /* should not happen */
  }

 libxsmm_generator_transform_norm_to_normt_mbit_scalar_rv64_microkernel( io_generated_code, io_loop_label_tracker,
                                                                                 i_gp_reg_mapping->gp_reg_in, i_gp_reg_mapping->gp_reg_out,
                                                                                 i_gp_reg_mapping->gp_reg_m_loop, i_gp_reg_mapping->gp_reg_n_loop,
                                                                                 i_gp_reg_mapping->gp_reg_scratch_0, i_micro_kernel_config, i_mateltwise_desc );
}
