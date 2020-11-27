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
#include "generator_packed_gemm_ac_rm_avx_avx2_avx512.h"
#include "generator_gemm_common.h"
#include "libxsmm_main.h"

LIBXSMM_API void libxsmm_generator_packed_gemm_ac_rm( libxsmm_generated_code*         io_generated_code,
                                                      const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                      const unsigned int              i_packed_width,
                                                      const char*                     i_arch ) {
  if ( strcmp(i_arch, "knl") == 0 ||
       strcmp(i_arch, "knm") == 0 ||
       strcmp(i_arch, "skx") == 0 ||
       strcmp(i_arch, "clx") == 0 ||
       strcmp(i_arch, "cpx") == 0 ||
       strcmp(i_arch, "hsw") == 0 ||
       strcmp(i_arch, "snb") == 0 ) {
    if ( strcmp(i_arch, "snb") == 0 ) {
      io_generated_code->arch = LIBXSMM_X86_AVX;
    } else if ( strcmp(i_arch, "hsw") == 0 ) {
      io_generated_code->arch = LIBXSMM_X86_AVX2;
    } else if ( strcmp(i_arch, "knl") == 0 ) {
      io_generated_code->arch = LIBXSMM_X86_AVX512_MIC;
    } else if ( strcmp(i_arch, "knm") == 0 ) {
      io_generated_code->arch = LIBXSMM_X86_AVX512_KNM;
    } else if ( strcmp(i_arch, "skx") == 0 ) {
      io_generated_code->arch = LIBXSMM_X86_AVX512_CORE;
    } else if ( strcmp(i_arch, "clx") == 0 ) {
      io_generated_code->arch = LIBXSMM_X86_AVX512_CLX;
    } else if ( strcmp(i_arch, "cpx") == 0 ) {
      io_generated_code->arch = LIBXSMM_X86_AVX512_CPX;
    } else {
      /* cannot happen */
    }

    libxsmm_generator_packed_gemm_ac_rm_avx_avx2_avx512( io_generated_code,
                                                         i_xgemm_desc, i_packed_width );
  } else {
    fprintf( stderr, "RM AC SOA is only available for AVX/AVX2/AVX512 at this point\n" );
    exit(-1);
  }
}

