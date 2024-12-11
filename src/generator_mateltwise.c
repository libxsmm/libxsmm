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

LIBXSMM_API
void libxsmm_generator_mateltwise_kernel( libxsmm_generated_code*          io_generated_code,
                                          const libxsmm_meltw_descriptor*  i_mateltw_desc ) {
  /* generate kernel */
  if ( (io_generated_code->arch >= LIBXSMM_X86_GENERIC) && (io_generated_code->arch <= LIBXSMM_X86_ALLFEAT) ) {
    libxsmm_generator_mateltwise_sse_avx_avx512_kernel( io_generated_code, i_mateltw_desc );
  } else if ( (io_generated_code->arch >= LIBXSMM_AARCH64_V81) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT) ) {
    libxsmm_generator_mateltwise_aarch64_kernel( io_generated_code, i_mateltw_desc );
  } else {
    /* TODO fix this error and support for more architectures */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }
}

