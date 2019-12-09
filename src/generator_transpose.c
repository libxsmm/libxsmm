/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Greg Henry (Intel Corp.)
******************************************************************************/
#include <libxsmm_generator.h>
#include "generator_common.h"
#include "generator_transpose_avx_avx512.h"


/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_transpose_kernel( libxsmm_generated_code*          io_generated_code,
                                         const libxsmm_trans_descriptor*  i_trans_desc,
                                         int                              i_arch ) {
  /* generate kernel */
  if ( LIBXSMM_X86_AVX <= i_arch ) {
    libxsmm_generator_transpose_avx_avx512_kernel( io_generated_code, i_trans_desc, i_arch );
  } else {
    /* TODO fix this error */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }
}

