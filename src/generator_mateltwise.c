/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
*               Friedrich Schiller University Jena - All rights reserved.     *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.), Antonio Noack (FSU Jena)
******************************************************************************/
#include <libxsmm_generator.h>
#include "generator_common.h"
#include "generator_mateltwise_sse_avx_avx512.h"
#include "generator_mateltwise_aarch64.h"
#include "generator_mateltwise_aarch64_sve.h"

LIBXSMM_API_INTERN
libxsmm_blasint libxsmm_generator_mateltwise_all_inp_comp_out_prec(const libxsmm_meltw_descriptor*   i_mateltwise_desc, libxsmm_datatype i_dtype ) {
  libxsmm_blasint result = 0;
  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY ) {
    if (i_dtype == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
        i_dtype == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) &&
        i_dtype == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
      result = 1;
    }
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY ) {
    if (i_dtype == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
        i_dtype == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1) &&
        i_dtype == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) &&
        i_dtype == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
      result = 1;
    }
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY )  {
    if (i_dtype == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
        i_dtype == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1) &&
        i_dtype == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN2) &&
        i_dtype == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) &&
        i_dtype == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
      result = 1;
    }
  } else {
    /* Should not happen */
  }
  return result;
}

LIBXSMM_API_INTERN
libxsmm_blasint libxsmm_generator_mateltwise_involves_prec(const libxsmm_meltw_descriptor*   i_mateltwise_desc, libxsmm_datatype i_dtype ) {
  libxsmm_blasint result = 0;
  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY ) {
    if (i_dtype == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ||
        i_dtype == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ||
        i_dtype == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
      result = 1;
    }
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY ) {
    if (i_dtype == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ||
        i_dtype == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1) ||
        i_dtype == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ||
        i_dtype == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
      result = 1;
    }
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY )  {
    if (i_dtype == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ||
        i_dtype == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1) ||
        i_dtype == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN2) ||
        i_dtype == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ||
        i_dtype == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
      result = 1;
    }
  } else {
    /* Should not happen */
  }
  return result;
}




LIBXSMM_API
void libxsmm_generator_mateltwise_kernel( libxsmm_generated_code*          io_generated_code,
                                          const libxsmm_meltw_descriptor*  i_mateltw_desc ) {
  /* generate kernel */
  if ( (io_generated_code->arch >= LIBXSMM_X86_GENERIC) && (io_generated_code->arch <= LIBXSMM_X86_ALLFEAT) ) {
    libxsmm_generator_mateltwise_sse_avx_avx512_kernel( io_generated_code, i_mateltw_desc );
  } else if ( (io_generated_code->arch >= LIBXSMM_AARCH64_V81) && (io_generated_code->arch <= LIBXSMM_AARCH64_V82) ) {
    libxsmm_generator_mateltwise_aarch64_kernel( io_generated_code, i_mateltw_desc );
  } else if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE256) && (io_generated_code->arch <= LIBXSMM_AARCH64_A64FX) ) {
    libxsmm_generator_mateltwise_aarch64_sve_kernel( io_generated_code, i_mateltw_desc );
  } else if ( (io_generated_code->arch >= LIBXSMM_AARCH64_APPL_M1 && io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT) ) {
    libxsmm_generator_mateltwise_aarch64_kernel(io_generated_code, i_mateltw_desc);
  } else {
    /* TODO fix this error and support for more architectures */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }
}

