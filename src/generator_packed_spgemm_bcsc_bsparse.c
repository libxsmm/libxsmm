/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/

#include "generator_packed_spgemm_bcsc_bsparse.h"
#include "generator_packed_spgemm_bcsc_bsparse_aarch64.h"

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_bcsc_bsparse( libxsmm_generated_code*         io_generated_code,
                                                   const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                   const unsigned int*             i_row_idx,
                                                   const unsigned int*             i_column_idx,
                                                   const unsigned int              i_packed_width,
                                                   const unsigned int              i_bk,
                                                   const unsigned int              i_bn ) {
  if ( (io_generated_code->arch >= LIBXSMM_X86_AVX) &&
       (io_generated_code->arch <= LIBXSMM_X86_ALLFEAT) ) {
    fprintf( stderr, "PACKED BCSC is only available for AARCH64 at this point\n" );
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  } else if ( (io_generated_code->arch >= LIBXSMM_AARCH64_V81) &&
              (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT) ) {
    libxsmm_generator_packed_spgemm_bcsc_bsparse_aarch64( io_generated_code,
                                                          i_xgemm_desc,
                                                          i_row_idx,
                                                          i_column_idx,
                                                          i_packed_width,
                                                          i_bk,
                                                          i_bn );
  } else {
    fprintf( stderr, "PACKED BCSC is only available for AARCH64 at this point\n" );
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}
