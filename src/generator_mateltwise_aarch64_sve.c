/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Antonio Noack (Friedrich Schiller University Jena)
******************************************************************************/

#include "generator_mateltwise_aarch64.h"
#include "generator_mateltwise_aarch64_sve.h"
#include "generator_mateltwise_transform_common.h"
#include "generator_mateltwise_unary_binary_aarch64.h"
#include "generator_mateltwise_reduce_aarch64.h"
#include "generator_mateltwise_misc_aarch64.h"
#include "libxsmm_matrixeqn.h"
#include "generator_aarch64_instructions.h"
#include "generator_common.h"
#include "libxsmm_main.h"


LIBXSMM_API_INTERN
libxsmm_aarch64_sve_type libxsmm_generator_aarch64_get_sve_type(unsigned char i_size){
   if(i_size == 1) return LIBXSMM_AARCH64_SVE_TYPE_B;
   if(i_size == 2) return LIBXSMM_AARCH64_SVE_TYPE_H;
   if(i_size == 4) return LIBXSMM_AARCH64_SVE_TYPE_S;
   return LIBXSMM_AARCH64_SVE_TYPE_D;
 }

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_aarch64_sve_init_micro_kernel_config_fullvector( libxsmm_generated_code*           io_generated_code,
                                                                               libxsmm_mateltwise_kernel_config* io_micro_kernel_config,
                                                                               const libxsmm_meltw_descriptor*   i_mateltwise_desc) {
  memset(io_micro_kernel_config, 0, sizeof(*io_micro_kernel_config)); /* avoid warning "maybe used uninitialized" */
  libxsmm_generator_mateltwise_aarch64_update_micro_kernel_config_vectorlength( io_generated_code, io_micro_kernel_config, i_mateltwise_desc );
}

LIBXSMM_API_INTERN
void libxsmm_generator_meltw_setup_stack_frame_aarch64_sve( libxsmm_generated_code*            io_generated_code,
                                              const libxsmm_meltw_descriptor*      i_mateltwise_desc,
                                              libxsmm_mateltwise_gp_reg_mapping*   i_gp_reg_mapping,
                                              libxsmm_mateltwise_kernel_config*    i_micro_kernel_config) {
      /* for x86 saves registers to the stack; on aarch64, it is done in libxsmm_aarch64_instruction_open_stream() already */
  LIBXSMM_UNUSED(io_generated_code);
  LIBXSMM_UNUSED(i_mateltwise_desc);
  LIBXSMM_UNUSED(i_gp_reg_mapping);
  LIBXSMM_UNUSED(i_micro_kernel_config);
}

LIBXSMM_API_INTERN
void libxsmm_generator_meltw_destroy_stack_frame_aarch64_sve( libxsmm_generated_code*            io_generated_code,
    const libxsmm_meltw_descriptor*     i_mateltwise_desc,
    const libxsmm_mateltwise_kernel_config*  i_micro_kernel_config ) {
      /* for x86 restores registers from the stack; on aarch64, it is done in libxsmm_aarch64_instruction_close_stream() already */
  LIBXSMM_UNUSED(i_mateltwise_desc);
  LIBXSMM_UNUSED(io_generated_code);
  LIBXSMM_UNUSED(i_micro_kernel_config);
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_aarch64_sve_kernel( libxsmm_generated_code*         io_generated_code,
                                                  const libxsmm_meltw_descriptor* i_mateltwise_desc ) {
  libxsmm_mateltwise_kernel_config  l_kernel_config;
  libxsmm_mateltwise_gp_reg_mapping l_gp_reg_mapping;
  libxsmm_loop_label_tracker        l_loop_label_tracker;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define gp register mapping */
  memset(&l_gp_reg_mapping, 0, sizeof(l_gp_reg_mapping));
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_AARCH64_GP_REG_X0;

  /* define mateltwise kernel config */
  libxsmm_generator_mateltwise_aarch64_init_micro_kernel_config_fullvector( io_generated_code, &l_kernel_config, i_mateltwise_desc );

  /* open asm */
  libxsmm_aarch64_instruction_open_stream( io_generated_code, 0xe0f );

  /* being BLAS aligned, for empty kernels, do nothing */
  if ( (i_mateltwise_desc->m > 0) && ((i_mateltwise_desc->n > 0) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX)) ) {
    /* Stack management for melt kernel */
    libxsmm_generator_meltw_setup_stack_frame_aarch64_sve( io_generated_code, i_mateltwise_desc, &l_gp_reg_mapping, &l_kernel_config);

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_OPREDUCE_VECS_IDX) {
      libxsmm_generator_opreduce_vecs_index_aarch64_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
    } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY ) {
      if (libxsmm_matrix_eqn_is_unary_opcode_reduce_kernel(i_mateltwise_desc->param) > 0) {
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS) > 0) {
          libxsmm_generator_reduce_rows_aarch64_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
        } else if (((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS) > 0) && (i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD_NCNC_FORMAT)) {
          libxsmm_generator_reduce_cols_aarch64_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
        } else if (((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS) > 0) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD_NCNC_FORMAT)) {
          libxsmm_generator_reduce_cols_ncnc_aarch64_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
        } else {
          /* This should not happen  */
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_MISSING_REDUCE_FLAGS );
          return;
        }
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX) {
        libxsmm_generator_reduce_cols_index_aarch64_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) {
        libxsmm_generator_replicate_col_var_aarch64_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      } else if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI)     ||
           (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT)    ||
           (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI_TO_VNNIT)    ||
           (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNIT)    ||
           (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI_PAD) ||
           (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD2)        ||
           (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD2)        ||
           (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2)          ) {
        libxsmm_generator_transform_aarch64_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      } else {
        libxsmm_generator_unary_binary_aarch64_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      }
    } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY ) {
      libxsmm_generator_unary_binary_aarch64_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
    } else  {
      /* This should not happen  */
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
      return;
    }

    /* Stack management formelt kernel */
    libxsmm_generator_meltw_destroy_stack_frame_aarch64_sve(  io_generated_code, i_mateltwise_desc, &l_kernel_config );
  }

  /* close asm */
  libxsmm_aarch64_instruction_close_stream( io_generated_code, 0xe0f );
}
