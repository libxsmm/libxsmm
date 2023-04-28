/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_packed_spgemm_csr_bsparse.h"
#include "generator_packed_spgemm_csr_bsparse_avx_avx2_avx512.h"
#include "generator_packed_spgemm_csr_bsparse_aarch64.h"

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_csr_bsparse( libxsmm_generated_code*         io_generated_code,
                                                  const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                  const unsigned int*             i_row_idx,
                                                  const unsigned int*             i_column_idx,
                                                  const void*                     i_values,
                                                  const unsigned int              i_packed_width ) {
  if ( (io_generated_code->arch >= LIBXSMM_X86_AVX) &&
       (io_generated_code->arch <= LIBXSMM_X86_ALLFEAT) ) {
    libxsmm_generator_packed_spgemm_csr_bsparse_avx_avx2_avx512( io_generated_code,
                                                                 i_xgemm_desc,
                                                                 i_row_idx,
                                                                 i_column_idx,
                                                                 i_values,
                                                                 i_packed_width );
  } else if ( (io_generated_code->arch >= LIBXSMM_AARCH64_V81) &&
              (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT) ) {
    libxsmm_generator_packed_spgemm_csr_bsparse_aarch64( io_generated_code,
                                                         i_xgemm_desc,
                                                         i_row_idx,
                                                         i_column_idx,
                                                         i_values,
                                                         i_packed_width );
  } else {
    fprintf( stderr, "packed CSR is only available for AVX/AVX2/AVX512 or AARCH64 at this point\n" );
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}
