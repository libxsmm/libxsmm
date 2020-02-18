/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm_generator.h>
#include "generator_common.h"
#include "generator_matcopy_avx_avx512.h"

/* @TODO change int based architecture value */
LIBXSMM_API
void libxsmm_generator_matcopy_kernel( libxsmm_generated_code*          io_generated_code,
                                       const libxsmm_mcopy_descriptor*  i_matcopy_desc,
                                       const char*                      i_arch ) {
  /* generate kernel */
  if ( (strcmp(i_arch, "skx") == 0) ||
       (strcmp(i_arch, "knm") == 0) ||
       (strcmp(i_arch, "knl") == 0) ||
       (strcmp(i_arch, "hsw") == 0) ||
       (strcmp(i_arch, "snb") == 0) ||
       (strcmp(i_arch, "clx") == 0) ||
       (strcmp(i_arch, "cpx") == 0)    ) {
    libxsmm_generator_matcopy_avx_avx512_kernel( io_generated_code, i_matcopy_desc, i_arch );
  } else {
    /* TODO fix this error */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }
}

