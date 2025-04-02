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
#include "generator_mateltwise_rv64.h"
#include "generator_mateltwise_transform_common.h"
#include "generator_mateltwise_transform_rv64.h"
#include "generator_mateltwise_unary_binary_rv64.h"
#include "libxsmm_matrixeqn.h"
#include "generator_common.h"
#include "generator_mateltwise_common.h"

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_rv64_update_micro_kernel_config_vectorlength( libxsmm_generated_code*   io_generated_code,
                                                                                libxsmm_mateltwise_kernel_config* io_micro_kernel_config,
                                                                                const libxsmm_meltw_descriptor*   i_mateltwise_desc) {
  if ( io_generated_code->arch == LIBXSMM_AARCH64_V81 ||
       io_generated_code->arch == LIBXSMM_AARCH64_V82 ||
       io_generated_code->arch == LIBXSMM_AARCH64_APPL_M1 ||
       io_generated_code->arch == LIBXSMM_AARCH64_SVE256 ||
       io_generated_code->arch == LIBXSMM_AARCH64_NEOV1 ||
       io_generated_code->arch == LIBXSMM_AARCH64_SVE512 ||
       io_generated_code->arch == LIBXSMM_RV64_MVL128 ||
       io_generated_code->arch == LIBXSMM_RV64_MVL256 ||
       io_generated_code->arch == LIBXSMM_AARCH64_A64FX ) {
    io_micro_kernel_config->instruction_set = io_generated_code->arch;
    io_micro_kernel_config->vector_reg_count = 32;
    /* Configure input specific microkernel options */
    if ( (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_I64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ) {
      io_micro_kernel_config->datatype_size_in = 8;
    } else if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_U32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ) {
      io_micro_kernel_config->datatype_size_in = 4;
    } else if ( (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_U16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ) {
      io_micro_kernel_config->datatype_size_in = 2;
    } else if ( (LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) ) {
      io_micro_kernel_config->datatype_size_in = 1;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY ) {
      if ( (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) || (LIBXSMM_DATATYPE_I64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) ) {
        io_micro_kernel_config->datatype_size_in1 = 8;
      } else if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) || (LIBXSMM_DATATYPE_U32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) ) {
        io_micro_kernel_config->datatype_size_in1 = 4;
      } else if ( (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) || (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) || (LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) || (LIBXSMM_DATATYPE_U16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) ) {
        io_micro_kernel_config->datatype_size_in1 = 2;
      } else if ( (LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) || (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1)) ) {
        io_micro_kernel_config->datatype_size_in1 = 1;
      } else {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY ) {
      if ( (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_I64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ) {
        io_micro_kernel_config->datatype_size_in2 = 8;
      } else if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_U32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0)) || (LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ) {
        io_micro_kernel_config->datatype_size_in2 = 4;
      } else if ( (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_U16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ) {
        io_micro_kernel_config->datatype_size_in2 = 2;
      } else if ( (LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) || (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) ) {
        io_micro_kernel_config->datatype_size_in2 = 1;
      } else {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }
    }

    /* Configure output specific microkernel options */
    if ( (LIBXSMM_DATATYPE_F64 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) || (LIBXSMM_DATATYPE_I64 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ) {
      io_micro_kernel_config->datatype_size_out = 8;
    } else if ( (LIBXSMM_DATATYPE_F32 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) || (LIBXSMM_DATATYPE_U32 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ))  || (LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ) {
      io_micro_kernel_config->datatype_size_out = 4;
    } else if ( (LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) || (LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) || (LIBXSMM_DATATYPE_U16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) || (LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ) {
      io_micro_kernel_config->datatype_size_out = 2;
    } else if ( (LIBXSMM_DATATYPE_I8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ||  (LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT )) ) {
      io_micro_kernel_config->datatype_size_out = 1;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  } else {
    /* That should not happen */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_rv64_init_micro_kernel_config_fullvector( libxsmm_generated_code*           io_generated_code,
                                                                            libxsmm_mateltwise_kernel_config* io_micro_kernel_config,
                                                                            const libxsmm_meltw_descriptor*   i_mateltwise_desc) {
  memset(io_micro_kernel_config, 0, sizeof(*io_micro_kernel_config)); /* avoid warning "maybe used uninitialized" */
  libxsmm_generator_mateltwise_rv64_update_micro_kernel_config_vectorlength( io_generated_code, io_micro_kernel_config, i_mateltwise_desc );
}

LIBXSMM_API_INTERN
void libxsmm_generator_meltw_setup_stack_frame_rv64( libxsmm_generated_code*            io_generated_code,
                                                     const libxsmm_meltw_descriptor*    i_mateltwise_desc,
                                                     libxsmm_mateltwise_gp_reg_mapping* i_gp_reg_mapping,
                                                     libxsmm_mateltwise_kernel_config*  i_micro_kernel_config) {
  LIBXSMM_UNUSED(io_generated_code);
  LIBXSMM_UNUSED(i_mateltwise_desc);
  LIBXSMM_UNUSED(i_gp_reg_mapping);
  LIBXSMM_UNUSED(i_micro_kernel_config);
}

LIBXSMM_API_INTERN
void libxsmm_generator_meltw_destroy_stack_frame_rv64( libxsmm_generated_code*                 io_generated_code,
                                                       const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                       const libxsmm_mateltwise_kernel_config* i_micro_kernel_config ) {
  LIBXSMM_UNUSED(i_mateltwise_desc);
  LIBXSMM_UNUSED(io_generated_code);
  LIBXSMM_UNUSED(i_micro_kernel_config);
}

LIBXSMM_API_INTERN
libxsmm_blasint libxsmm_generator_mateltwise_rv64_valid_arch_precision( libxsmm_generated_code*  io_generated_code,
                                                                        const libxsmm_meltw_descriptor*   i_mateltwise_desc) {
  libxsmm_blasint is_valid_arch_prec = 0;

  unsigned int is_transform_tpp = i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT;

  unsigned int is_unary_simple_rv64_tpp = ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY )  &&
      ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_IDENTITY)     ||
       (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_XOR)     ||
       (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_X2))) ? 1 : 0;

  unsigned int  is_binary_simple_rv64_tpp = ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY  &&
        i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_ADD) ) ? 1 : 0;

  unsigned int dtype_in0 = (libxsmm_datatype)libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0);
  unsigned int dtype_out = (libxsmm_datatype)libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT);
  unsigned int dtype_in1 = LIBXSMM_DATATYPE_F32;

  unsigned int dtype_comp = libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP);

  unsigned int is_fp32_inp_out = 0;

  if (is_binary_simple_rv64_tpp)
    dtype_in1 = (libxsmm_datatype)libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1);

  is_fp32_inp_out = (LIBXSMM_DATATYPE_F32 == dtype_in0 && LIBXSMM_DATATYPE_F32 == dtype_in1 &&
                              LIBXSMM_DATATYPE_F32 == dtype_out && LIBXSMM_DATATYPE_F32 == dtype_comp) ? 1 : 0;

  is_valid_arch_prec = (is_unary_simple_rv64_tpp || is_binary_simple_rv64_tpp || is_transform_tpp) && is_fp32_inp_out;
  return is_valid_arch_prec;
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_rv64_kernel( libxsmm_generated_code*         io_generated_code,
                                               const libxsmm_meltw_descriptor* i_mateltwise_desc ) {
  libxsmm_mateltwise_kernel_config  l_kernel_config;
  libxsmm_mateltwise_gp_reg_mapping l_gp_reg_mapping;
  libxsmm_loop_label_tracker        l_loop_label_tracker;

  /* Check if TPP is supported on current arch */
  if ( libxsmm_generator_mateltwise_rv64_valid_arch_precision(io_generated_code, i_mateltwise_desc) == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
    return;
  }

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define gp register mapping */
  memset(&l_gp_reg_mapping, 0, sizeof(l_gp_reg_mapping));

  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_RV64_GP_REG_X10; /*LIBXSMM_RV64_GP_REG_R0*/;

  /* define mateltwise kernel config */
  libxsmm_generator_mateltwise_rv64_init_micro_kernel_config_fullvector( io_generated_code, &l_kernel_config, i_mateltwise_desc );

  /* open asm */
  libxsmm_rv64_instruction_open_stream( io_generated_code, 0xfff );

  libxsmm_generator_meltw_setup_stack_frame_rv64( io_generated_code, i_mateltwise_desc, &l_gp_reg_mapping, &l_kernel_config);

  if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT) {
    libxsmm_generator_transform_rv64_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
  } else if ( (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) ||
      (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) ) {
    libxsmm_generator_unary_binary_rv64_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
  } else {
    /* This should not happen */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
    return;
  }

  libxsmm_generator_meltw_destroy_stack_frame_rv64(  io_generated_code, i_mateltwise_desc, &l_kernel_config );

  /* close asm */
  libxsmm_rv64_instruction_close_stream( io_generated_code, 0xfff );
}
