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
#include <libxsmm_generator.h>
#include "generator_common.h"
#include "generator_matequation_avx_avx512.h"

LIBXSMM_API
void libxsmm_generator_matequation_kernel( libxsmm_generated_code*         io_generated_code,
                                           const libxsmm_meqn_descriptor*  i_mateqn_desc ) {
  /* generate kernel */
  if ( io_generated_code->arch >= LIBXSMM_X86_AVX2  ) {
    libxsmm_generator_matequation_avx_avx512_kernel( io_generated_code, i_mateqn_desc );
   } else {
    /* TODO fix this error and support for more architectures */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }
}

